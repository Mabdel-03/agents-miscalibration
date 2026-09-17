"""F15-F16: the coverage-selectability trade-off, and what a cheap adjudicator would buy."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st
import isoselect

BENCH_COLOR = {"math500": "#0072B2", "mmlu_pro": "#009E73",
               "mbpp_plus": "#D55E00", "bbh_logic4": "#999999"}


# ---------------------------------------------------------------- F15
def tradeoff(ss, conc, matched, outdir, name="f15_coverage_selectability"):
    """Delivered = coverage x selection success, so allocations trade one against the other."""
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))

    # (a) the plane, with iso-delivered contours
    ax = axes[0]
    xs = np.linspace(0.76, 0.95, 200)
    for lvl in [0.78, 0.80, 0.82, 0.84, 0.86]:
        ys = lvl / xs
        m = ys <= 1.0
        ax.plot(xs[m], ys[m], color="#CCCCCC", lw=0.6, zorder=1)
        if m.any():
            ax.annotate(f"{lvl:.2f}", xy=(xs[m][0], ys[m][0]), fontsize=4.8,
                        color="#999999", xytext=(2, -5), textcoords="offset points")
    for fam in ["sample", "private_team", "peer"]:
        g = ss[(ss["family"] == fam) & (ss["rung"] >= 4)]
        ax.scatter(g["coverage"], g["S"], s=9, color=ic.FAMILY_COLOR[fam],
                   linewidths=0, alpha=0.85, label=ic.FAMILY_LABEL[fam], zorder=3)
    st.grid(ax, axis="both")
    ax.set_xlabel("Pool coverage")
    ax.set_ylabel("Selection success $S$")
    ax.set_title("(a) The trade-off", fontsize=7.6)
    ax.legend(fontsize=5.6, loc="lower left", handletextpad=0.3)

    # (b) selection success against budget
    ax = axes[1]
    for fam in ["sample", "private_team", "peer"]:
        g = ss[ss["family"] == fam].groupby("rung")["S"].mean().reset_index()
        ax.plot(g["rung"], g["S"], marker="o", ms=3, color=ic.FAMILY_COLOR[fam],
                label=ic.FAMILY_LABEL[fam])
    st.grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 25]); ax.set_xticklabels(["2", "4", "8", "16", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Total compute $C/A_0$")
    ax.set_ylabel("Selection success $S$")
    ax.set_title("(b) Width makes selection harder", fontsize=7.6)
    ax.legend(fontsize=6, loc="lower left")

    # (c) pools of the same size, produced differently
    ax = axes[2]
    lab = {"sample/25": "iid\n$25\\times1$", "private_team/5/4": "team\n$5\\times5$",
           "private_single/25": "solo\n$1\\times25$", "peer/5/4": "peer\n$5\\times5$"}
    m = matched.set_index("name").reindex(list(lab))
    x = np.arange(len(m))
    ax.bar(x - 0.21, m["coverage"], width=0.2, color="#5A5A5A", linewidth=0, label="Coverage")
    ax.bar(x, m["S"], width=0.2, color="#D55E00", linewidth=0, label="Selection success $S$")
    ax.bar(x + 0.21, m["y"], width=0.2, color="#BBD7E8", linewidth=0, label="Delivered")
    st.grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[i] for i in m.index], fontsize=6.2)
    ax.set_ylim(0.70, 1.0)
    ax.set_title("(c) 25 candidates, four ways", fontsize=7.6)
    ax.legend(fontsize=5.6, loc="lower right")
    fig.subplots_adjust(wspace=0.34)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F16
def adjudication(struct, topm, selv, outdir, name="f16_selection_repair"):
    """The answer space decides which repair works."""
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    benches = ["math500", "mmlu_pro", "mbpp_plus"]

    ax = axes[0]
    g = struct.groupby("benchmark")[["n_distinct", "top_share"]].mean().reindex(benches)
    x = np.arange(len(benches))
    ax.bar(x, g["n_distinct"], width=0.5, color="#5A5A5A", linewidth=0)
    for i, v in enumerate(g["n_distinct"]):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=6)
    st.grid(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([st.PANEL_TITLE[b] for b in benches], fontsize=6, rotation=12)
    ax.set_ylabel("Distinct answers among 25")
    ax.set_title("(a) How wide is the answer space", fontsize=7.6)

    ax = axes[1]
    for bench in benches:
        g = topm[topm["benchmark"] == bench].sort_values("m")
        ax.plot(g["m"], g["acc"], marker="o", ms=3, color=BENCH_COLOR[bench],
                label=st.PANEL_TITLE[bench])
        ax.axhline(float(g["oracle"].iloc[0]), color=BENCH_COLOR[bench], lw=0.6, ls=":")
    st.grid(ax)
    ax.set_xscale("log", base=2)
    ax.set_xticks(isoselect.TOP_M)
    ax.set_xticklabels([str(m) for m in isoselect.TOP_M])
    ax.minorticks_off()
    ax.set_xlabel("Adjudicate the top $m$ answers")
    ax.set_ylabel("Accuracy")
    ax.set_title("(b) Perfect choice among the top $m$", fontsize=7.6)
    ax.legend(fontsize=5.8, loc="lower right")

    ax = axes[2]
    for bench in benches:
        g = selv[selv["benchmark"] == bench].sort_values("verified")
        ax.plot(g["verified"] * 100, g["recovered"] * 100, marker="o", ms=3,
                color=BENCH_COLOR[bench], label=st.PANEL_TITLE[bench])
    st.grid(ax)
    ax.set_xlabel("Items sent to the adjudicator (\\%)")
    ax.set_ylabel("Gap recovered (\\%)")
    ax.set_title("(c) Adjudicate only close calls", fontsize=7.6)
    ax.legend(fontsize=5.8, loc="upper left")
    fig.subplots_adjust(wspace=0.36)
    return st.save(fig, name, outdir)
