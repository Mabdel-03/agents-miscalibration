"""Plotting conventions for the analysis notebooks.

Matches the repo style established in analysis/examples/plot_reliability.py:
#2563eb primary blue, #94a3b8 slate gray, dashed 0.6-gray reference lines,
grid alpha 0.25, dpi=180 PNGs. Saved figures always go through savefig() so they land
in analysis/figures/ on the data filesystem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as _data

PRIMARY = "#2563eb"
GRAY = "#94a3b8"
REF_GRAY = "0.6"

TOPO_COLORS = {
    "single_agent": "#64748b",
    "independent": "#2563eb",
    "decentralized": "#d97706",
    "centralized": "#dc2626",
}
BENCH_COLORS = {
    "gpqa": "#7c3aed",
    "mmlu_pro": "#2563eb",
    "truthfulqa": "#059669",
    "math": "#d97706",
}
REAS_COLORS = {  # light -> dark with reasoning depth
    "off": "#cbd5e1",
    "b512": "#93c5fd",
    "b2048": "#3b82f6",
    "b8192": "#1d4ed8",
    "unlimited": "#1e3a8a",
}
SIZE_MARKERS = {"0.6B": "o", "1.7B": "s", "4B": "D", "8B": "^", "14B": "v", "32B": "*"}


def style() -> None:
    """Session-wide matplotlib defaults matching repo conventions."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
    })


def savefig(fig: plt.Figure, name: str) -> str:
    """Save under analysis/figures/<name>.png at dpi=180 (repo convention)."""
    _data.FIGURES.mkdir(parents=True, exist_ok=True)
    out = _data.FIGURES / f"{name}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return str(out)


def annotate_n(ax: plt.Axes, n: int, loc: str = "lower right") -> None:
    """Per-panel sample-size annotation — every faceted figure carries its n."""
    xy = {"lower right": (0.98, 0.02), "upper left": (0.02, 0.95),
          "upper right": (0.98, 0.95), "lower left": (0.02, 0.02)}[loc]
    ax.text(*xy, f"n={n}", transform=ax.transAxes, fontsize=8, color="0.4",
            ha="right" if "right" in loc else "left")


def flag_capped(ax: plt.Axes, note: str = "32B ctx-capped (16k)") -> None:
    """Standard caption flag for panels containing 32B x {b8192, unlimited} points."""
    ax.text(0.02, 1.02, note, transform=ax.transAxes, fontsize=7.5, color="#dc2626")


def reliability_diagram(conf: np.ndarray, correct: np.ndarray, ax: plt.Axes | None = None,
                        n_bins: int = 10, equal_mass: bool = True, label: str | None = None,
                        color: str = PRIMARY, show_hist: bool = True) -> plt.Axes:
    """Reliability diagram in repo style; equal-mass bins by default (see calib.py)."""
    from . import calib

    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(conf) & np.isfinite(correct)
    conf, correct = conf[ok], correct[ok]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=REF_GRAY, linestyle="--", linewidth=1)
    if len(conf) == 0:
        return ax
    if equal_mass:
        edges = calib.equal_mass_edges(conf, n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    mids, accs, cnts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        mids.append(conf[m].mean())
        accs.append(correct[m].mean())
        cnts.append(int(m.sum()))
    ax.plot(mids, accs, marker="o", color=color, label=label)
    if show_hist:
        ax2 = ax.twinx()
        ax2.bar(mids, cnts, width=0.9 / n_bins, color=GRAY, alpha=0.35)
        ax2.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    annotate_n(ax, len(conf))
    return ax


def facet_grid(n_rows: int, n_cols: int, width: float = 3.2, height: float = 2.8,
               sharex: bool = True, sharey: bool = True):
    """Small-multiple grid with repo sizing; returns (fig, 2-D axes array)."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows),
                             sharex=sharex, sharey=sharey, squeeze=False)
    return fig, axes


def forest_plot(coefs: pd.DataFrame, ax: plt.Axes | None = None,
                est_col: str = "coef", lo_col: str = "lo", hi_col: str = "hi",
                color: str = PRIMARY) -> plt.Axes:
    """Horizontal coefficient forest plot. Index = term names."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.34 * len(coefs) + 1))
    y = np.arange(len(coefs))[::-1]
    ax.axvline(0, color=REF_GRAY, linestyle="--", linewidth=1)
    ax.hlines(y, coefs[lo_col], coefs[hi_col], color=color, linewidth=1.6)
    ax.plot(coefs[est_col], y, "o", color=color, markersize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(coefs.index, fontsize=8)
    return ax
