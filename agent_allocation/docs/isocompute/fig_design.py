"""F1: the design map. Where the evaluated configurations sit in the width-depth plane, and what
one fixed compute budget buys under each allocation."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import isocore as ic
import isostyle as st


def draw(idx, outdir, name="f1_design_map", a0=1024):
    st.style()
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

    # ---- (a) the width-depth plane with iso-compute contours
    ax = fig.add_subplot(gs[0, 0])
    for G in [2, 4, 8, 16, 25]:
        b = np.linspace(1, 25, 400)
        ax.plot(b, G / b, color="#BBBBBB", lw=0.7, zorder=1)
        # Label each contour at its top-left end, clear of the marker cloud.
        ax.text(1.12, min(G, 25), f"$C/A_0$={G}", fontsize=5.5, color="#888888",
                va="center", ha="left", zorder=1)
    # Private team and peer occupy identical (B, R) cells, so nudge them apart to keep both visible.
    JIT = {"sample": 1.0, "private_single": 1.0, "private_team": 0.94, "peer": 1.06}
    for fam in ["sample", "private_single", "private_team", "peer"]:
        g = idx[idx["family"] == fam]
        ax.scatter(g["B"] * JIT[fam], g["R"], s=14, color=ic.FAMILY_COLOR[fam], alpha=0.85,
                   linewidths=0, label=ic.FAMILY_LABEL[fam], zorder=3)
    g = idx[idx["family"] == "deep"]
    ax.scatter(g["B"], g["R"], s=26, marker="D", color=ic.FAMILY_COLOR["deep"],
               label="Deep single call ($A$ scaled)", zorder=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 25]); ax.set_xticklabels(["1", "2", "4", "8", "16", "25"])
    ax.set_yticks([1, 2, 4, 8, 16, 25]); ax.set_yticklabels(["1", "2", "4", "8", "16", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Width $B$ (agents)")
    ax.set_ylabel("Depth $R$ (rounds)")
    ax.set_title("(a) Evaluated configurations", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", handletextpad=0.3, borderpad=0.2)
    st.grid(ax, axis="both")

    # ---- (b) what one budget buys: C/A0 = 8 under five allocations
    ax = fig.add_subplot(gs[0, 1])
    rows = [("sample/8", "sample", 8, 1, 1, "8 independent samples, vote"),
            ("private_team/4/1", "private_team", 4, 2, 1, "4 agents, 2 rounds"),
            ("private_team/2/3", "private_team", 2, 4, 1, "2 agents, 4 rounds"),
            ("private_single/8", "private_single", 1, 8, 1, "1 agent, 8 rounds"),
            ("deep/8", "deep", 1, 1, 8, "1 call at $8A_0$")]
    u = 0.44
    for i, (nm, fam, B, R, mult, desc) in enumerate(rows):
        y = len(rows) - 1 - i
        col = ic.FAMILY_COLOR[fam]
        if mult == 1:
            for bi in range(B):
                for ri in range(R):
                    ax.add_patch(Rectangle((bi * u * 1.12, y - 0.34 + ri * (0.68 / max(R, 1))),
                                           u * 0.92, (0.68 / max(R, 1)) * 0.82,
                                           facecolor=col, edgecolor="white", lw=0.4))
        else:
            ax.add_patch(Rectangle((0, y - 0.34), u * 2.04, 0.68,
                                   facecolor=col, edgecolor="white", lw=0.4))
            ax.text(u * 1.02, y, "$8A_0$", ha="center", va="center", fontsize=6.4, color="white")
        ax.text(8 * u * 1.12 + 0.12, y, desc, va="center", fontsize=6.6)
    ax.set_xlim(-0.1, 9.6); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axis("off")
    ax.set_title("(b) One budget, $C = 8\\,A_0$, five ways", fontsize=8)
    ax.text(0, -0.62, "Each tile is one generation at allowance $A_0$; every row spends the same $C$.",
            fontsize=6, va="top")
    return st.save(fig, name, outdir)
