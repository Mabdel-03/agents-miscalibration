"""Phase B: every atlas figure, generated only from data/atlas_data.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import _common as C
import style as S

D = json.load(open(f"{C.DATA}/atlas_data.json"))
def V(k): return D[k]["value"]
def has(k): return k in D
BANDS = ["never", "hard", "contested", "easy"]
BAND_LABEL = {"never": "never solved\n(55%)", "hard": "hard\n(14%)", "contested": "contested\n(11%)", "easy": "easy\n(19%)"}

# ---------------- F2: difficulty gates every orchestration effect ----------------
def f2():
    pb = V("difficulty.policy_by_band"); sd = V("difficulty.single_draw_by_band")
    sp = V("difficulty.policy_spread_by_band"); n = V("difficulty.items_per_band")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.55, 1]))
    x = np.arange(len(BANDS)); w = 0.15
    for i, m in enumerate(C.POLICIES):
        vals = [pb[b][m] for b in BANDS]
        ax[0].bar(x + (i - 2) * w, vals, w, color=S.POLICY_COLOR[m], label=C.POLICY_LABEL[m], zorder=3)
    ax[0].plot(x, [sd[b] for b in BANDS], "o--", color=S.INK, lw=1.1, ms=4, zorder=4, label="single draw")
    ax[0].set_xticks(x); ax[0].set_xticklabels([BAND_LABEL[b] for b in BANDS])
    for i, b in enumerate(BANDS): ax[0].text(i, 1.04, f"n={n[b]}", ha="center", fontsize=6.5, color=S.MUTED)
    ax[0].set_ylim(0, 1.12); ax[0].legend(ncol=2, loc="upper left", fontsize=6.8)
    S.finish(ax[0], "accuracy", title="a  Policy accuracy within difficulty bands")
    pooled = V("difficulty.policy_spread_pooled")
    ax[1].bar(x, [sp[b] for b in BANDS], 0.55, color=[S.WARM if b == "contested" else S.SLATE for b in BANDS], zorder=3)
    ax[1].axhline(pooled, color=S.INK, ls="--", lw=1)
    ax[1].text(3.35, pooled + 0.004, f"pooled {pooled:.3f}", ha="right", fontsize=6.8, color=S.INK)
    ax[1].set_xticks(x); ax[1].set_xticklabels([b for b in BANDS], fontsize=7)
    S.finish(ax[1], "best − worst policy", title="b  Spread between policies")
    S.save(fig, "f2_difficulty")

# ---------------- F3: the five policies, pooled and per domain ----------------
def f3():
    pol = V("policy.flagship"); dom = V("policy.by_domain"); pc = V("policy.paired_contrasts")
    order = sorted(C.POLICIES, key=lambda m: -pol[m]["acc"])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.25, 0.95, 1.05]))
    fig.subplots_adjust(wspace=0.62)
    y = np.arange(len(order))[::-1]
    ax[0].barh(y, [pol[m]["acc"] for m in order], 0.55, color=[S.POLICY_COLOR[m] for m in order], zorder=3)
    ax[0].errorbar([pol[m]["acc"] for m in order], y,
                   xerr=[[pol[m]["acc"] - pol[m]["lo"] for m in order], [pol[m]["hi"] - pol[m]["acc"] for m in order]],
                   fmt="none", ecolor=S.INK, elinewidth=0.9, capsize=2)
    ax[0].set_yticks(y); ax[0].set_yticklabels([C.POLICY_LABEL[m] for m in order], fontsize=7.5)
    ax[0].set_xlim(0, 0.45); S.finish(ax[0], xlab="accuracy", title="a  Pooled (400 items)", grid="x")
    for i, dd in enumerate(("bcb", "hle")):
        ax[1].barh(y + (0.19 if dd == "bcb" else -0.19), [dom[m][dd] for m in order], 0.34,
                   color=S.DOMAIN_COLOR[dd], label=C.DOMAIN_LABEL[dd], zorder=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlim(0, 0.62)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=6.6)
    S.finish(ax[1], xlab="accuracy", title="b  By domain", grid="x")
    SHORT = {"IND_VOTE": "ind", "CEN_FLAT": "hub", "DEC": "peer", "S_FRESH": "fresh", "S_HISTORY": "hist"}
    keys = [k for k in ["IND_VOTE-CEN_FLAT", "IND_VOTE-DEC", "IND_VOTE-S_FRESH", "S_HISTORY-CEN_FLAT", "DEC-CEN_FLAT"] if k in pc]
    yy = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        r = pc[k]
        ax[2].plot([r["lo"], r["hi"]], [yy[i], yy[i]], color=S.MUTED, lw=1.2, solid_capstyle="round")
        ax[2].plot(r["diff"], yy[i], "o", color=S.ACCENT, ms=4.5, zorder=4)
    ax[2].axvline(0, color=S.INK, lw=0.8)
    a, b = zip(*[k.split("-") for k in keys])
    ax[2].set_yticks(yy); ax[2].set_yticklabels([f"{SHORT[x]} $-$ {SHORT[z]}" for x, z in zip(a, b)], fontsize=7)
    S.finish(ax[2], xlab="paired difference", title="c  Contrasts (95%)", grid="x")
    S.save(fig, "f3_policies")


# ---------------- F4: realised compute, caps, and the hub's loop ----------------
def f4():
    pol = V("policy.flagship"); hb = V("hub.by_call_bucket"); tk = V("hub.tokens_by_call_index")
    nb = V("hub.within_never_band") if has("hub.within_never_band") else None
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    order = [m for m in C.POLICIES]
    x = np.arange(len(order))
    ax[0].bar(x, [pol[m]["spent_frac"] for m in order], 0.55, color=[S.POLICY_COLOR[m] for m in order], zorder=3)
    for i, m in enumerate(order):
        ax[0].text(i, pol[m]["spent_frac"] + 0.012, f"{pol[m]['calls']:.0f}", ha="center", fontsize=6.5, color=S.MUTED)
    ax[0].set_xticks(x); ax[0].set_xticklabels([C.POLICY_LABEL[m].replace(" ", "\n") for m in order], fontsize=6.3)
    ax[0].set_ylim(0, 0.55)
    S.finish(ax[0], "fraction of FLOP allowance used", title="a  Nobody spends the budget")
    ks = sorted(tk, key=int); xs = np.arange(len(ks))
    ax[1].plot(xs, [tk[k]["mean_tokens"] for k in ks], "o-", color=S.WARM, label="all calls")
    ax[1].plot(xs, [tk[k]["mean_tokens_uncapped"] for k in ks], "s--", color=S.SLATE, label="excluding capped")
    ax[1].axhline(8192, color=S.INK, ls=":", lw=1); ax[1].text(len(ks) - 1, 8300, "output cap", ha="right", fontsize=6.5, color=S.INK)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([k if int(k) < 5 else "6+" for k in ks])
    ax[1].set_ylim(0, 9200); ax[1].legend(loc="lower right", fontsize=6.8)
    S.finish(ax[1], "hub reasoning tokens", "hub call index", "b  Each retry reasons longer")
    bs = ["1", "2", "3-4", ">=5"]; xb = np.arange(len(bs))
    ax[2].bar(xb - 0.2, [hb[b]["acc"] for b in bs], 0.38, color=S.WARM, label="all items", zorder=3)
    ax[2].plot(xb - 0.2, [hb[b]["p1"] for b in bs], "^", color=S.INK, ms=5, zorder=5, label="item difficulty (pass@1)")
    if nb: ax[2].bar(xb + 0.2, [nb.get(b, {}).get("acc", 0) for b in bs], 0.38, color=S.SLATE,
                     label="never-solved band only", zorder=3)
    ax[2].set_xticks(xb); ax[2].set_xticklabels(bs); ax[2].legend(fontsize=6.3, loc="upper right")
    S.finish(ax[2], "accuracy", "hub calls", "c  The gradient is composition")
    S.save(fig, "f4_compute")

# ---------------- F5: test-time compute widens the selection gap ----------------
def f5():
    pk = V("ttc.pass_at_k"); sk = V("ttc.selector_at_k")
    ks = [1, 2, 3, 5, 10]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for i, dd in enumerate(("hle", "bcb")):
        a = ax[i]
        a.plot(ks, [pk[dd][str(k)] for k in ks], "o-", color=S.ACCENT, label="pass@k (any draw correct)")
        vk = sk[f"VOTE|{dd}"]; jk = sk[f"JUDGE_BEST|{dd}"]
        a.plot(ks, [vk.get(str(float(k)), vk.get(str(k))) for k in ks], "s-", color=S.WARM, label="plurality vote@k")
        a.plot(ks, [jk.get(str(float(k)), jk.get(str(k))) for k in ks], "^--", color=S.SAND, label="LLM judge@k")
        a.fill_between(ks, [pk[dd][str(k)] for k in ks],
                       [vk.get(str(float(k)), vk.get(str(k))) for k in ks], color=S.ACCENT, alpha=0.09, zorder=1)
        a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels(ks)
        S.finish(a, "accuracy" if i == 0 else None, "samples k", f"{'ab'[i]}  {C.DOMAIN_LABEL[dd]}")
        if i == 0: a.legend(loc="upper left", fontsize=7)
        a.annotate("", xy=(10, pk[dd]["10"]), xytext=(10, vk.get("10.0", vk.get("10"))),
                   arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.9))
        a.text(9.2, (pk[dd]["10"] + vk.get("10.0", vk.get("10"))) / 2, "selection\ngap", ha="right", fontsize=6.5, color=S.MUTED)
    S.save(fig, "f5_ttc")

# ---------------- F6: where the pool's correct answers are lost ----------------
def f6():
    q = V("selection.archive_quality")
    meths = ["IND_VOTE", "S_FRESH", "S_HISTORY", "DEC"]
    rows = [(m, q.get(f"{m}|VOTE")) for m in meths if f"{m}|VOTE" in q]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(len(rows))[::-1]
    for i, (m, r) in enumerate(rows):
        ax[0].plot([r["cand_mean"], r["oracle"]], [y[i], y[i]], color=S.RULE, lw=4, solid_capstyle="round", zorder=2)
        ax[0].plot(r["cand_mean"], y[i], "o", color=S.SLATE, ms=5, zorder=4)
        ax[0].plot(r["selected"], y[i], "D", color=S.ACCENT, ms=5.5, zorder=5)
        ax[0].plot(r["oracle"], y[i], "o", color=S.WARM, ms=5, zorder=4)
        ax[0].text(r["oracle"] + 0.012, y[i], f"gap {r['oracle'] - r['selected']:.3f}", va="center", fontsize=6.5, color=S.MUTED)
    ax[0].set_yticks(y); ax[0].set_yticklabels([C.POLICY_LABEL[m] for m, _ in rows], fontsize=7.5)
    ax[0].set_xlim(0, 0.72)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=S.SLATE, label="mean candidate"),
                          Line2D([], [], marker="D", ls="", color=S.ACCENT, label="selected (vote)"),
                          Line2D([], [], marker="o", ls="", color=S.WARM, label="oracle over pool")],
                 loc="lower right", fontsize=6.8)
    S.finish(ax[0], xlab="accuracy", title="a  Pool quality vs delivered answer", grid="x")
    bd = V("selection.by_domain")
    doms = ["hle", "bcb"]; x = np.arange(2)
    sing = [bd[f"IND_VOTE|VOTE|{d}"]["all_singleton"] for d in doms]
    ax[1].bar(x, sing, 0.5, color=[S.DOMAIN_COLOR[d] for d in doms], zorder=3)
    for i, s_ in enumerate(sing): ax[1].text(i, s_ + 0.02, f"{s_:.2f}", ha="center", fontsize=7, color=S.INK)
    ax[1].set_xticks(x); ax[1].set_xticklabels([C.DOMAIN_LABEL[d] for d in doms], fontsize=7)
    ax[1].set_ylim(0, 1.0)
    S.finish(ax[1], "share of pools where every\ncandidate is its own class", title="b  Voting has nothing to vote on")
    S.save(fig, "f6_selection")

# ---------------- F7: calibration ----------------
def f7():
    rel = V("calib.reliability"); cp = V("calib.by_policy_flagship")
    SHORT = {"IND_VOTE": "ind", "S_FRESH": "fresh", "CEN_FLAT": "hub", "DEC": "peer", "S_HISTORY": "hist"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(wspace=0.42)
    order = [b for b in ["<=0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"] if b in rel]
    conf = [rel[b]["conf"] for b in order]; acc = [rel[b]["acc"] for b in order]
    ax[0].plot([0, 1], [0, 1], ls=":", color=S.INK, lw=1)
    ax[0].plot(conf, acc, "o-", color=S.ACCENT)
    for b, cx, ay in zip(order, conf, acc):
        ax[0].annotate(f"{rel[b]['n']}", (cx, ay), textcoords="offset points", xytext=(4, -7), fontsize=6, color=S.MUTED)
    ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02)
    S.finish(ax[0], "observed accuracy", "stated confidence", "a  Reliability (flagship)")
    ms = sorted(cp, key=lambda m: cp[m]["brier"])
    x = np.arange(len(ms))
    for i, (metric, lab, ttl) in enumerate(((("brier"), "Brier (lower better)", "b  Sharing costs calibration"),
                                            ("share_ge95", "share of answers at conf $\\geq$ 0.95", "c  Certainty, not accuracy"))):
        a_ = ax[i + 1]
        a_.bar(x, [cp[m][metric] for m in ms], 0.55, color=[S.POLICY_COLOR[m] for m in ms], zorder=3)
        a_.set_xticks(x); a_.set_xticklabels([SHORT[m] for m in ms], fontsize=7)
        S.finish(a_, lab, None, ttl)
    S.save(fig, "f7_calibration")


# ---------------- F8/F9: model scale ----------------
def f8():
    cb = V("calib.by_checkpoint"); nz = V("calib.by_checkpoint_excl_placeholder")
    sd = V("scale.by_domain"); vc = V("validity.by_checkpoint")
    cks = ["4B", "8B", "14B", "32B"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    x = np.arange(len(cks))
    ax[0].plot(x, [cb[c]["conf"] for c in cks], "o-", color=S.WARM, label="stated confidence")
    ax[0].plot(x, [nz[c]["conf"] for c in cks], "s--", color=S.SAND, label="excl. the 0.50 default")
    ax[0].plot(x, [cb[c]["acc"] for c in cks], "^-", color=S.ACCENT, label="accuracy")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cks); ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=6.6, loc="center left")
    S.finish(ax[0], None, "checkpoint", "a  Confidence scales, accuracy does not")
    ax[1].bar(x - 0.2, [cb[c]["share_eq50"] for c in cks], 0.38, color=S.SAND, label="stated exactly 0.50", zorder=3)
    ax[1].bar(x + 0.2, [1 - vc[c]["valid_final"] for c in cks], 0.38, color=S.SLATE, label="final answer unparseable", zorder=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(cks); ax[1].legend(fontsize=6.6)
    S.finish(ax[1], "share of episodes", "checkpoint", "b  Two mechanical channels")
    for i, dd in enumerate(("bcb", "hle")):
        vals = []
        for c in cks:
            got = [sd[k] for k in sd if k.endswith(f"|{c}|{dd}")]
            vals.append(np.mean(got) if got else np.nan)
        ax[2].plot(x, vals, "o-", color=S.DOMAIN_COLOR[dd], label=C.DOMAIN_LABEL[dd])
    ax[2].set_xticks(x); ax[2].set_xticklabels(cks); ax[2].legend(fontsize=6.8)
    S.finish(ax[2], "accuracy", "checkpoint", "c  Scale ordering flips by domain")
    S.save(fig, "f8_scale")

# ---------------- F10: peer revision round by round ----------------
def f10():
    br = V("dec.by_round"); rounds = sorted(br, key=int)
    x = [int(r) for r in rounds]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].plot(x, [br[r]["conf"] for r in rounds], "o-", color=S.WARM, label="stated confidence")
    ax[0].plot(x, [br[r]["agreement"] for r in rounds], "s-", color=S.SLATE, label="answer agreement")
    ax[0].set_ylim(0, 1.0); ax[0].legend(fontsize=6.8, loc="center right")
    S.finish(ax[0], None, "revision round", "a  Certainty and consensus rise")
    ax[1].plot(x, [br[r]["cand_correct"] for r in rounds], "o-", color=S.ACCENT, label="mean member correct")
    ax[1].plot(x, [br[r]["oracle"] for r in rounds], "^-", color=S.WARM, label="any member correct")
    ax[1].legend(fontsize=6.8); ax[1].set_ylim(0, 0.5)
    S.finish(ax[1], "accuracy", "revision round", "b  Coverage falls as consensus rises")
    fl = V("dec.flip_accounting"); sw = V("dec.switching")
    labels = ["wrong→correct", "correct→wrong"]
    vals = [fl["wrong_to_correct"], fl["correct_to_wrong"]]
    ax[2].bar([0, 1], vals, 0.5, color=[S.ACCENT, S.WARM], zorder=3)
    vals = [int(v) for v in vals]
    for i, v in enumerate(vals): ax[2].text(i, v + 3, str(v), ha="center", fontsize=7.5, color=S.INK)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(labels, fontsize=7)
    ax[2].text(0.5, max(vals) * 0.72, f"{int(fl['labelled_transitions']):,} labelled\nmember-transitions\nnet gain {vals[0]-vals[1]:+d}",
               ha="center", fontsize=6.6, color=S.MUTED)
    S.finish(ax[2], "transitions", title="c  Revision is nearly a wash")
    S.save(fig, "f10_rounds")

# ---------------- F11/F12: geometry ----------------
def f11():
    tr = V("geom.trajectory")
    anchors = ["31_NATIVE_PREFILL", "31_GENERATED_32", "31_GENERATED_128", "31_GENERATED_512", "31_FINAL_OBJECT_CLOSE"]
    lab = ["prompt", "32 tok", "128 tok", "512 tok", "answer"]
    x = np.arange(len(anchors))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for arm, col, mk in (("IND", S.ACCENT, "o"), ("DEC", S.WARM, "s")):
        ax[0].plot(x, [tr[a]["pr"][arm] for a in anchors], mk + "-", color=col,
                   label="independent solvers" if arm == "IND" else "peer-exposed members")
    ax[0].axhline(4, color=S.INK, ls=":", lw=0.9); ax[0].text(0, 4.06, "rank ceiling (s−1)", fontsize=6.3, color=S.MUTED)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylim(0, 4.4)
    ax[0].legend(fontsize=6.8, loc="lower left")
    for i, a in enumerate(anchors):
        p = tr[a]["pr"]["p"]
        if p < 0.01: ax[0].text(i, tr[a]["pr"]["IND"] + 0.18, "∗", ha="center", fontsize=10, color=S.INK)
    S.finish(ax[0], "participation ratio", None, "a  Effective dimensionality along the call")
    r = tr["31_FINAL_OBJECT_CLOSE"]
    mets = ["pair_mean", "pair_min", "pair_max", "total_var", "lam1_share", "pr"]
    names = ["mean pairwise\ndistance", "nearest pair", "farthest pair", "total variance",
             "share on top\naxis (λ₁)", "participation\nratio"]
    ratios = [r[m]["ratio"] for m in mets]
    cols = [S.SLATE if abs(v - 1) < 0.02 else (S.WARM if v > 1 else S.ACCENT) for v in ratios]
    yy = np.arange(len(mets))[::-1]
    ax[1].barh(yy, [v - 1 for v in ratios], 0.55, color=cols, zorder=3)
    ax[1].axvline(0, color=S.INK, lw=0.9)
    for i, (m, v) in enumerate(zip(mets, ratios)):
        ax[1].text((v - 1) + (0.008 if v >= 1 else -0.008), yy[i], f"{v:.2f}×",
                   va="center", ha="left" if v >= 1 else "right", fontsize=6.6, color=S.INK)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(names, fontsize=6.8); ax[1].set_xlim(-0.36, 0.32)
    S.finish(ax[1], xlab="peer-exposed ÷ independent (answer token)", title="b  Anisotropy, not contraction", grid="x")
    S.save(fig, "f11_geometry")

# ---------------- F13: what is decodable ----------------
def f13():
    if not has("probe.trajectory"): return
    tj = V("probe.trajectory"); cc = V("probe.candidate_correctness")
    mj = V("probe.majority_membership")
    ra = V("probe.report_architecture"); rc = V("probe.report_correctness")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.15, 0.85, 1.2]))
    fig.subplots_adjust(wspace=0.5)
    anchors = ["NATIVE_PREFILL", "GENERATED_32", "GENERATED_128", "GENERATED_512", "FINAL_OBJECT_CLOSE"]
    lab = ["prompt", "32", "128", "512", "answer"]
    x = np.arange(len(anchors))
    ax[0].plot(x, [tj[a]["pooled_auc"] for a in anchors], "o-", color=S.SLATE, label="pooled AUC")
    ax[0].plot(x, [tj[a]["within_item_centred"] for a in anchors], "s-", color=S.ACCENT, label="within-item AUC")
    ax[0].plot(x, [tj[a]["item_difficulty_rho"] for a in anchors], "^--", color=S.WARM, label="item difficulty $\\rho$")
    ax[0].axhline(0.5, color=S.INK, ls=":", lw=0.9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylim(-0.08, 0.92)
    ax[0].legend(fontsize=6.2, loc="upper left", ncol=1)
    S.finish(ax[0], "AUC / $\\rho$", "anchor", "a  Correctness along the call")
    keys = ["b31|hle", "b31|bcb"]
    x2 = np.arange(2); w = 0.26
    ax[1].bar(x2 - w, [cc[k]["within_item_centred"] for k in keys], w, color=S.ACCENT, label="correct", zorder=3)
    ax[1].bar(x2, [mj[d]["within_item_auc"] for d in ("hle", "bcb")], w, color=S.WARM, label="in majority", zorder=3)
    ax[1].bar(x2 + w, [cc[k]["conf_within_item"] for k in keys], w, color=S.SLATE, label="stated conf.", zorder=3)
    ax[1].axhline(0.5, color=S.INK, ls=":", lw=0.9); ax[1].set_ylim(0.40, 0.66)
    ax[1].set_xticks(x2); ax[1].set_xticklabels(["Exam", "Code"], fontsize=7)
    ax[1].legend(fontsize=6.2, loc="upper left")
    S.finish(ax[1], "within-item AUC", None, "b  Mainstream $>$ correct")
    items = [("architecture\n(report state)", ra["b31|report_state"]["accuracy"], S.WARM),
             ("architecture\n(task only)", ra["b31|task_only"]["accuracy"], S.SLATE),
             ("answer correct\n(report state)", rc["b31|report_state"]["auc"], S.ACCENT),
             ("answer correct\n(task only)", rc["b31|task_only"]["auc"], S.SLATE)]
    y = np.arange(len(items))[::-1]
    ax[2].barh(y, [v for _, v, _ in items], 0.58, color=[c for _, _, c in items], zorder=3)
    ax[2].axvline(1 / 3, color=S.MUTED, ls=":", lw=0.9); ax[2].axvline(0.5, color=S.MUTED, ls=":", lw=0.9)
    for i, (_, v, _) in enumerate(items):
        ax[2].text(v + 0.025, y[i], f"{v:.3f}", va="center", fontsize=6.4, color=S.INK)
    ax[2].set_yticks(y); ax[2].set_yticklabels([n for n, _, _ in items], fontsize=6.0)
    ax[2].tick_params(axis="y", pad=1); ax[2].set_xlim(0, 1.2)
    S.finish(ax[2], xlab="accuracy / AUC", title="c  The reader knows the harness", grid="x")
    S.save(fig, "f13_probes")


for fn in (f2, f3, f4, f5, f6, f7, f8, f10, f11, f13):
    try: fn()
    except Exception as e:
        import traceback; print(f"  !! {fn.__name__}: {type(e).__name__}: {e}"); traceback.print_exc(limit=2)
