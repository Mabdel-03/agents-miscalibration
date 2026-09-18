"""Shared figure style. Serif, colour-blind-safe, no chartjunk, legible at print size."""
from __future__ import annotations
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import isocore as ic


def style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, outdir):
    p = f"{outdir}/figures/{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def rung_axis(ax, rungs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(rungs)
    ax.set_xticklabels([str(r) for r in rungs])
    ax.minorticks_off()


PANELS = ["math500", "mmlu_pro", "bbh_logic4", "mbpp_plus", "macro3"]
PANEL_TITLE = {
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "bbh_logic4": "BBH-logic4 (saturated)",
    "mbpp_plus": "MBPP+",
    "macro3": "Macro, 3 informative benchmarks",
    "macro4": "Macro, all 4 benchmarks",
}
