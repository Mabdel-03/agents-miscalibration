"""One matplotlib style for every atlas figure: print-legible, colour-blind-safe, no chartjunk."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"; MUTED = "#6b6b6b"; RULE = "#d4d0c8"
ACCENT = "#1b5e8f"      # independent / primary
WARM = "#b5502a"        # coordinated / contrast
SAND = "#c9a227"        # third series
SLATE = "#5c6b73"       # fourth
SERIES = [ACCENT, WARM, SAND, SLATE, "#7a5c8f"]
POLICY_COLOR = {"IND_VOTE": ACCENT, "CEN_FLAT": WARM, "DEC": SAND, "S_FRESH": SLATE, "S_HISTORY": "#7a5c8f"}
DOMAIN_COLOR = {"hle": "#7a5c8f", "bcb": "#2e7d5b"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.edgecolor": RULE, "axes.linewidth": 0.7, "axes.labelcolor": INK, "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 1.4,
    "grid.color": RULE, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
})


def finish(ax, ylab=None, xlab=None, title=None, grid="y"):
    if title: ax.set_title(title, color=INK)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)
    if grid: ax.grid(axis=grid, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    return ax


def save(fig, name):
    import os
    from _common import FIGS
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"))
    plt.close(fig)
    print(f"  figure -> {name}.pdf")
