"""F10: does the plateau survive a four-times larger allowance and a stronger model?"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st

SETTINGS = [
    ("main", "(a) Main: Flash-Lite, $A_0$=1024, 400 items"),
    ("high", "(b) High allowance: $A_0$=4096, 96 items"),
    ("replication", "(c) Replication: Gemini 2.5 Flash, 48 items"),
]


def draw(frames, curves, outdir, level="macro3", name="f10_robustness"):
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ai, (part, title) in enumerate(SETTINGS):
        ax = axes[ai]
        f = frames[part]; f = f[f["level"] == level]
        c = curves[part]; c = c[c["level"] == level].sort_values("k")
        best = f.groupby("rung")["acc"].max().reset_index().sort_values("rung")
        samp = f[f["family"] == "sample"].sort_values("rung")
        p1 = float(f["pass1"].iloc[0])
        ax.fill_between(c["k"], c["oracle"], np.interp(c["k"], best["rung"], best["acc"]),
                        color="#D55E00", alpha=0.16, lw=0, label="Unrealised headroom")
        ax.plot(c["k"], c["oracle"], color="#000000", ls="--", lw=1.1, label="Oracle pass@$k$")
        ax.plot(best["rung"], best["acc"], color="#0072B2", lw=1.4,
                label="Best configuration at budget")
        ax.plot(samp["rung"], samp["acc"], color=ic.FAMILY_COLOR["sample"], lw=1.4,
                label="Sample + vote")
        ax.axhline(p1, color="#999999", lw=0.9, ls=":", label="Single sample")
        st.grid(ax)
        ax.set_xscale("log", base=2)
        ticks = [t for t in [1, 2, 4, 8, 16, 25] if t <= f["rung"].max()]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_title(title, fontsize=7.4)
        ax.set_xlabel("Total compute $C/A_0$")
        if ai == 0:
            ax.set_ylabel("Accuracy (macro, 3 benchmarks)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.6, bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.30)
    return st.save(fig, name, outdir)
