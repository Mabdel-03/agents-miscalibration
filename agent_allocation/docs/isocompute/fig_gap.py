"""F11-F14: where the oracle gap sits and which lever closes it."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st
import isogap

BENCH_COLOR = {"math500": "#0072B2", "mmlu_pro": "#009E73",
               "mbpp_plus": "#D55E00", "bbh_logic4": "#999999", "macro3": "#000000"}


# ---------------------------------------------------------------- F11
def anatomy(resp, buck, outdir, name="f11_gap_anatomy"):
    """Why coverage is not enough, and where the item mass sits."""
    st.style()
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))

    ax = axes[0]
    for bench in ["math500", "mmlu_pro", "mbpp_plus"]:
        g = resp[resp["benchmark"] == bench]
        g = g.set_index("bin").reindex(isogap.CONC_LABEL).reset_index()
        ax.plot(range(len(g)), g["deliver"], marker="o", ms=3,
                color=BENCH_COLOR[bench], label=st.PANEL_TITLE[bench])
    ax.axvline(2.5, color="#000000", lw=0.8, ls=":")
    ax.text(2.6, 0.06, "correct answer\nbecomes modal", fontsize=5.4, color="#444444")
    st.grid(ax)
    ax.set_xticks(range(len(isogap.CONC_LABEL)))
    ax.set_xticklabels(isogap.CONC_LABEL, rotation=90, fontsize=5.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(delivered correct)")
    ax.set_xlabel("Correct share of the pool")
    ax.set_title("(a) The selector is a step", fontsize=7.6)
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.2)

    for bi, bench in enumerate(["math500", "mmlu_pro", "mbpp_plus"]):
        ax = axes[bi + 1]
        g = buck[buck["benchmark"] == bench].sort_values("k")
        ax.stackplot(g["k"], g["majority"], g["minority"], g["no_correct"],
                     colors=["#BBD7E8", "#D55E00", "#5A5A5A"],
                     labels=["Correct is modal\n(a vote finds it)",
                             "Correct present but\nnot modal (the gap)",
                             "No correct answer\n(generation-limited)"])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 25])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "25"])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel("Samples $k$")
        ax.set_title(f"({'bcd'[bi]}) {st.PANEL_TITLE[bench]}", fontsize=7.6)
        if bi == 0:
            ax.set_ylabel("Share of items")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.30))
    fig.subplots_adjust(wspace=0.34)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F12
def split(dec, outdir, level="macro3", rungs=(8, 25), name="f12_gap_split"):
    """Each allocation's budget-matched gap, split into generation and selection losses."""
    st.style()
    fig, axes = plt.subplots(1, len(rungs), figsize=(7.4, 2.9))
    for ai, rung in enumerate(rungs):
        ax = axes[ai]
        d = dec[(dec["level"] == level) & (dec["rung"] == rung)
                & dec["family"].isin(ic.LADDER_FAMILIES)].copy()
        d = d.sort_values("y")
        ypos = np.arange(len(d))
        ax.barh(ypos, d["y"], color="#BBD7E8", label="Delivered", zorder=2)
        ax.barh(ypos, d["sel_loss"], left=d["y"], color="#D55E00", alpha=0.85,
                label="Selection loss (pool held it, selector missed)", zorder=2)
        ax.barh(ypos, d["cov_deficit"], left=d["cov_own"], color="#5A5A5A", alpha=0.75,
                label="Coverage deficit (pool never held it)", zorder=2)
        ax.plot(d["y_arc"], ypos, ls="none", marker="|", ms=7, mew=1.4, color="#000000",
                label="Delivered, archive readout (free)", zorder=4)
        ax.axvline(float(d["cov_sample"].iloc[0]), color="#000000", ls="--", lw=0.9, zorder=3)
        ax.text(float(d["cov_sample"].iloc[0]), len(d) - 0.3, " sampling\n coverage",
                fontsize=5.4, va="top")
        ax.set_yticks(ypos)
        ax.set_yticklabels(d["name"], fontsize=5.8)
        ax.set_xlim(0.70, 0.96)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"$C/A_0 = {rung}$", fontsize=8)
        st.grid(ax, axis="x")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.4, bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(wspace=0.42)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F13
def verifier(sweep, best_alloc, outdir, name="f13_verifier_sweep"):
    """What a reranker of a given discriminative power would buy from the same candidates."""
    st.style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))

    ax = axes[0]
    for bench in ["math500", "mmlu_pro", "mbpp_plus", "macro3"]:
        g = sweep[sweep["benchmark"] == bench].sort_values("auc")
        ax.plot(g["auc"], g["delivered"], marker="o", ms=2.6, color=BENCH_COLOR[bench],
                lw=1.8 if bench == "macro3" else 1.1,
                label=st.PANEL_TITLE.get(bench, bench))
    st.grid(ax)
    ax.set_xlabel("Reranker pairwise AUC")
    ax.set_ylabel("Delivered accuracy at $k=25$")
    ax.set_title("(a) A verifier on the same 25 candidates", fontsize=7.8)
    ax.legend(fontsize=6, loc="lower right", handlelength=1.4)

    ax = axes[1]
    g = sweep[sweep["benchmark"] == "macro3"].sort_values("auc")
    ax.plot(g["auc"], g["delivered"], color="#000000", lw=1.8, marker="o", ms=3,
            label="Reranker on 25 samples", zorder=4)
    plur = float(g["plurality"].iloc[0])
    orc = float(g["oracle"].iloc[0])
    ax.axhline(plur, color=ic.FAMILY_COLOR["sample"], lw=1.2, ls="-",
               label=f"Plurality vote today ({plur:.3f})")
    ax.axhline(best_alloc, color=ic.FAMILY_COLOR["private_team"], lw=1.2, ls="-.",
               label=f"Best of all 95 allocations ({best_alloc:.3f})")
    ax.axhline(orc, color="#000000", lw=1.0, ls="--", label=f"Oracle pass@25 ({orc:.3f})")
    eq = isogap.equivalent_auc(sweep, "macro3", plur)
    ax.plot([eq], [plur], marker="*", ms=9, color=ic.FAMILY_COLOR["sample"], zorder=5)
    ax.annotate(f"today's selector\n$\\approx$ AUC {eq:.2f}", xy=(eq, plur),
                xytext=(eq + 0.03, plur - 0.045), fontsize=6,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    st.grid(ax)
    ax.set_xlabel("Reranker pairwise AUC")
    ax.set_ylabel("Delivered accuracy (macro, 3)")
    ax.set_title("(b) Verification against allocation", fontsize=7.8)
    ax.legend(fontsize=6, loc="lower right", handlelength=1.6, framealpha=0.9, frameon=True,
              edgecolor="none")
    fig.subplots_adjust(wspace=0.30)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F14
def allowance(aa, buck, outdir, name="f14_allowance_and_selector"):
    """The one generation-side lever that works, and the one selector that backfires."""
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    benches = ["math500", "mmlu_pro", "mbpp_plus"]
    w = 0.36

    ax = axes[0]
    for pi, (panel, lab) in enumerate([("main", "$A_0$=1024"), ("high", "$A_0$=4096")]):
        g = aa[aa["panel"] == panel].set_index("benchmark").reindex(benches)
        ax.bar(np.arange(len(benches)) + (pi - 0.5) * w, g["draw_acc"], width=w,
               color=["#9ECAE1", "#08519C"][pi], linewidth=0, label=lab)
    st.grid(ax)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([st.PANEL_TITLE[b] for b in benches], fontsize=6, rotation=12)
    ax.set_ylabel("Per-draw accuracy")
    ax.set_title("(a) A raises per-draw accuracy", fontsize=7.6)
    ax.legend(fontsize=6)

    ax = axes[1]
    for pi, (panel, lab) in enumerate([("main", "$A_0$=1024"), ("high", "$A_0$=4096")]):
        g = aa[aa["panel"] == panel].set_index("benchmark").reindex(benches)
        ax.bar(np.arange(len(benches)) + (pi - 0.5) * w, g["gap"], width=w,
               color=["#FDBE85", "#A63603"][pi], linewidth=0, label=lab)
    st.grid(ax)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([st.PANEL_TITLE[b] for b in benches], fontsize=6, rotation=12)
    ax.set_ylabel("Oracle gap at $k$=25")
    ax.set_title("(b) and closes the gap where it does", fontsize=7.6)
    ax.legend(fontsize=6)

    ax = axes[2]
    g = buck[buck["benchmark"] == "mbpp_plus"].sort_values("k")
    ax.plot(g["k"], g["readout"], color="#D55E00", marker="o", ms=3,
            label="Public-test selection")
    ax.plot(g["k"], g["random_draw"], color="#666666", ls="--", label="Take one at random")
    ax.plot(g["k"], g["oracle"], color="#000000", ls=":", label="Oracle pass@$k$")
    st.grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 25]); ax.set_xticklabels(["1", "2", "4", "8", "16", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Candidates $k$")
    ax.set_ylabel("Accuracy")
    ax.set_title("(c) MBPP+: a selector that backfires", fontsize=7.6)
    ax.legend(fontsize=6, loc="center left")
    fig.subplots_adjust(wspace=0.38)
    return st.save(fig, name, outdir)
