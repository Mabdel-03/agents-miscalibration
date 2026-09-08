"""Phase B: generated booktabs tables. Every number comes from atlas_data.json; none is hand-typed."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _common as C

D = json.load(open(f"{C.DATA}/atlas_data.json"))
def V(k): return D[k]["value"]
def has(k): return k in D
os.makedirs(C.TABS, exist_ok=True)
def w(name, body):
    with open(os.path.join(C.TABS, f"{name}.tex"), "w") as f: f.write(body)
    print(f"  table -> {name}.tex")
def f(x, n=3):
    return "--" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
PN = {m: C.POLICY_LABEL[m] for m in C.POLICIES}

# T4 -- the five policies
pol, dom, pc = V("policy.flagship"), V("policy.by_domain"), V("policy.paired_contrasts")
va = V("validity.by_policy")
rows = []
for m in sorted(C.POLICIES, key=lambda k: -pol[k]["acc"]):
    p = pol[m]
    stop = max(p["stop"], key=p["stop"].get)
    rows.append(f"{PN[m]} & {f(p['acc'])} & [{f(p['lo'])}, {f(p['hi'])}] & {f(dom[m]['hle'])} & {f(dom[m]['bcb'])} & "
                f"{p['calls']:.0f} & {f(p['spent_frac'], 3)} & {stop.replace('_', chr(92) + '_')} & {f(p['conf'])} \\\\")
w("t_policies", r"""\begin{tabular}{lccccrrlc}
\toprule
& \multicolumn{3}{c}{accuracy} & & \multicolumn{3}{c}{compute} & \\
\cmidrule(lr){2-4}\cmidrule(lr){6-8}
Routing topology & pooled & 95\% CI & exam & code & calls & spent & stop reason & conf. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# T5 -- difficulty moderation
pb, sd, sp, nb = V("difficulty.policy_by_band"), V("difficulty.single_draw_by_band"), V("difficulty.policy_spread_by_band"), V("difficulty.items_per_band")
BANDS = ["never", "hard", "contested", "easy"]
rows = [f"{PN[m]} & " + " & ".join(f(pb[b][m]) for b in BANDS) + r" \\" for m in C.POLICIES]
w("t_difficulty", r"""\begin{tabular}{lcccc}
\toprule
& never solved & hard & contested & easy \\
items & """ + " & ".join(str(nb[b]) for b in BANDS) + r""" \\
single draw & """ + " & ".join(f(sd[b]) for b in BANDS) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
best $-$ worst & """ + " & ".join(f"\\textbf{{{f(sp[b])}}}" if b == "contested" else f(sp[b]) for b in BANDS) + r""" \\
\bottomrule
\end{tabular}""")

# T7 -- calibration
cp = V("calib.by_policy_flagship")
rows = [f"{PN[m]} & {f(cp[m]['acc'])} & {f(cp[m]['conf'])} & {f(cp[m]['bias'])} & {f(cp[m]['brier'])} & "
        f"{f(cp[m]['share_ge95'])} & {f(cp[m]['resolution'])} \\\\" for m in sorted(cp, key=lambda k: cp[k]["brier"])]
w("t_calibration", r"""\begin{tabular}{lcccccc}
\toprule
Routing topology & accuracy & stated conf. & bias & Brier & share $\ge$0.95 & resolution \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# T8 -- model scale
cb, nz, vc = V("calib.by_checkpoint"), V("calib.by_checkpoint_excl_placeholder"), V("validity.by_checkpoint")
sdm = V("scale.by_domain")
rows = []
for c in ["4B", "8B", "14B", "32B"]:
    hle = np.mean([sdm[k] for k in sdm if k.endswith(f"|{c}|hle")])
    bcb = np.mean([sdm[k] for k in sdm if k.endswith(f"|{c}|bcb")])
    rows.append(f"{c} & {f(cb[c]['acc'])} & {f(hle)} & {f(bcb)} & {f(cb[c]['conf'])} & {f(nz[c]['conf'])} & "
                f"{f(cb[c]['share_eq50'])} & {f(vc[c]['valid_final'])} & {f(cb[c]['brier'])} \\\\")
w("t_scale", r"""\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{3}{c}{accuracy} & \multicolumn{3}{c}{stated confidence} & & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Checkpoint & pooled & exam & code & all & excl.\ 0.50 & share $=$0.50 & valid final & Brier \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# T9 -- geometry at the answer anchor
tr = V("geom.trajectory")["31_FINAL_OBJECT_CLOSE"]
LBL = [("pr", "participation ratio", "scale-free"), ("er", "entropy effective rank", "scale-free"),
       ("lam1_share", "share of variance on $\\lambda_1$", "scale-free"), ("total_var", "total variance", "scale-carrying"),
       ("pair_mean", "mean pairwise distance", "scale-carrying"), ("pair_min", "nearest pair", "scale-carrying"),
       ("pair_max", "farthest pair", "scale-carrying"), ("cos_mean", "mean pairwise cosine distance", "angular"),
       ("norm_mean", "mean state norm", "scale-carrying")]
rows = []
for k, name, kind in LBL:
    r = tr[k]
    n = 3 if abs(r["IND"]) < 100 else 1
    p = r["p"]
    ps = "$<$1e-16" if p < 1e-16 else f"{p:.2g}"
    rows.append(f"{name} & {kind} & {f(r['IND'], n)} & {f(r['DEC'], n)} & {f(r['ratio'])} & {ps} \\\\")
w("t_geometry", r"""\begin{tabular}{llcccc}
\toprule
Observable & family & independent & peer-exposed & ratio & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# T10 -- decodability
if has("probe.trajectory"):
    tj, cc, cf = V("probe.trajectory"), V("probe.candidate_correctness"), V("probe.confidence_representation")
    mj, ra, rc = V("probe.majority_membership"), V("probe.report_architecture"), V("probe.report_correctness")
    hd = V("probe.hub_delegation")
    rows = [
        ("which orchestration produced this report", "report state, b31", f(ra["b31|report_state"]["accuracy"]), "3-way acc., chance 0.333"),
        ("\\quad from the task text alone", "task anchor, b31", f(ra["b31|task_only"]["accuracy"]), "3-way acc."),
        ("item difficulty", "prompt, b31", f(tj["NATIVE_PREFILL"]["item_difficulty_rho"]), "Spearman $\\rho$, item level"),
        ("whether the delivered answer is right", "report state, b31", f(rc["b31|report_state"]["auc"]), "AUC"),
        ("\\quad from the task text alone", "task anchor, b31", f(rc["b31|task_only"]["auc"]), "AUC"),
        ("is \\emph{this} candidate right (exam)", "answer token, b31", f(cc["b31|hle"]["within_item_centred"]), "within-item AUC"),
        ("is \\emph{this} candidate right (code)", "answer token, b31", f(cc["b31|bcb"]["within_item_centred"]), "within-item AUC"),
        ("will it join the majority (exam)", "answer token, b31", f(mj["hle"]["within_item_auc"]), "within-item AUC"),
        ("will it join the majority (code)", "answer token, b31", f(mj["bcb"]["within_item_auc"]), "within-item AUC"),
        ("the stated confidence value", "128 tokens in", f(cf["GENERATED_128"]["r2"]), "grouped-CV $R^2$"),
        ("the stated confidence value", "512 tokens in", f(cf["GENERATED_512"]["r2"]), "grouped-CV $R^2$"),
        ("the stated confidence value", "answer token", f(cf["FINAL_OBJECT_CLOSE"]["r2"]), "grouped-CV $R^2$"),
    ]
    if hd.get("hle_multi_call"): rows.append(("hub will make extra calls (exam)", "hub prompt, b31", f(hd["hle_multi_call"]["auc"]), "AUC"))
    if hd.get("hle_never_band_multi_call"): rows.append(("\\quad within the never-solved band", "hub prompt, b31", f(hd["hle_never_band_multi_call"]["auc"]), "AUC, difficulty held"))
    w("t_probes", r"""\begin{tabular}{llcl}
\toprule
Decoded target & site & value & metric \\
\midrule
""" + "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows) + r"""
\bottomrule
\end{tabular}""")

# T6 -- compute and the caps that bind
bl = V("budget.by_level"); bp = V("budget.paired_contrasts")
rows = []
for m in C.POLICIES:
    cells = [f"{m}|B{b}" for b in (1, 2, 4) if f"{m}|B{b}" in bl]
    if not cells: continue
    accs = " & ".join(f(bl[c]["acc"]) for c in cells)
    spent = " & ".join(f(bl[c]["spent_frac"], 2) for c in cells)
    k = f"{m}|B4-B1"
    d = f"{bp[k]['diff']:+.3f} [{bp[k]['lo']:+.3f}, {bp[k]['hi']:+.3f}]" if k in bp else "--"
    rows.append(f"{PN[m]} & {accs} & {spent} & {d} \\\\")
w("t_budget", r"""\begin{tabular}{lccccccl}
\toprule
& \multicolumn{3}{c}{accuracy} & \multicolumn{3}{c}{fraction of allowance spent} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Routing topology & B1 & B2 & B4 & B1 & B2 & B4 & B4$-$B1 paired \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")

# T-selection
q = V("selection.archive_quality"); bd = V("selection.by_domain")
rows = []
for m in ["IND_VOTE", "S_FRESH", "S_HISTORY", "DEC"]:
    kv, kj = f"{m}|VOTE", f"{m}|JUDGE_BEST"
    if kv not in q: continue
    r = q[kv]; j = q.get(kj, {})
    rows.append(f"{PN[m]} & {f(r['cand_mean'])} & {f(r['selected'])} & {f(j.get('selected'))} & {f(r['oracle'])} & "
                f"{f(r['oracle'] - r['selected'])} & {f(r['all_singleton'])} \\\\")
w("t_selection", r"""\begin{tabular}{lcccccc}
\toprule
Routing topology & mean candidate & vote & LLM judge & oracle & gap & all-singleton \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
print("tables done")
