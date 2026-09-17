"""F2: the iso-compute ladder. Delivered accuracy against total compute, per benchmark,
with the oracle pass@k ceiling and the sample-and-vote baseline."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st


def draw(frame, curve, outdir, partition="main", name="f2_isocompute_ladder"):
    st.style()
    rungs = sorted(frame["rung"].unique())
    ticks = [r for r in [1, 2, 4, 8, 16, 25] if r in rungs]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.5), sharex=True)
    axes = axes.ravel()

    for pi, panel in enumerate(st.PANELS):
        ax = axes[pi]
        f = frame[frame["level"] == panel]
        c = curve[curve["level"] == panel]

        # Oracle pass@k ceiling: what the same budget makes available to a perfect selector.
        ax.plot(c["k"], c["oracle"], color=ic.ORACLE_COLOR, ls="--", lw=1.1,
                label="Oracle pass@k (ceiling)", zorder=5)

        # Baseline: independent sampling read out by the benchmark's own selector.
        s = f[f["family"] == "sample"].sort_values("rung")
        ax.plot(s["rung"], s["acc"], color=ic.FAMILY_COLOR["sample"], lw=2.0,
                label="Sample + vote (baseline)", zorder=4)

        ps = f[f["family"] == "private_single"].sort_values("rung")
        ax.plot(ps["rung"], ps["acc"], color=ic.FAMILY_COLOR["private_single"],
                label="Private single (B=1)", zorder=3)

        # Teams: individual (n, r) cells as faint markers, the mean over n as the line.
        for fam in ["private_team", "peer"]:
            g = f[f["family"] == fam]
            ax.scatter(g["rung"], g["acc"], s=5, color=ic.FAMILY_COLOR[fam],
                       alpha=0.35, linewidths=0, zorder=2)
            mg = g.groupby("rung")["acc"].mean().reset_index().sort_values("rung")
            ax.plot(mg["rung"], mg["acc"], color=ic.FAMILY_COLOR[fam],
                    label=ic.FAMILY_LABEL[fam] + " (mean over B)", zorder=3)

        dp = f[f["family"] == "deep"].sort_values("rung")
        ax.plot(dp["rung"], dp["acc"], color=ic.FAMILY_COLOR["deep"], marker="D",
                ms=3, label="Deep single call (one call, C tokens)", zorder=4)

        st.grid(ax)
        st.rung_axis(ax, ticks)
        ax.set_title(st.PANEL_TITLE[panel])
        if panel == "bbh_logic4":
            # Saturated: hold a fixed window so estimator noise does not read as structure.
            ax.set_ylim(0.95, 1.005)
        if pi % 3 == 0:
            ax.set_ylabel("Delivered accuracy")
        if pi >= 2:
            ax.set_xlabel("Total compute $C/A_0$")
            ax.tick_params(labelbottom=True)

    # Legend in the sixth cell.
    ax = axes[5]
    ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    ax.legend(h, l, loc="center", fontsize=7, handlelength=1.8, borderpad=0)
    return st.save(fig, name, outdir)
