"""F3: the amplification matrix. Rows are iso-compute budgets, columns are allocations of that
budget, cells are performance relative to independent sampling at the same budget."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import isocore as ic
import isostyle as st

# Column order: width grows left to right within each family; deep is the pure depth-in-one-call arm.
COLS = [("private_single", 1), ("private_team", 2), ("private_team", 3), ("private_team", 4),
        ("private_team", 5), ("peer", 2), ("peer", 3), ("peer", 4), ("peer", 5), ("deep", 1)]
COL_LABEL = ["Private single", "PT $B$=2", "PT $B$=3", "PT $B$=4", "PT $B$=5",
             "Peer $B$=2", "Peer $B$=3", "Peer $B$=4", "Peer $B$=5", "Deep (1 call)"]
# Vertical rules separating the four allocation regimes.
SEPS = [0.5, 4.5, 8.5]
ROWS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25]


def _grids(frame, panel, value):
    f = frame[frame["level"] == panel]
    V = np.full((len(ROWS), len(COLS)), np.nan)
    S = np.zeros((len(ROWS), len(COLS)), dtype=bool)
    for i, rung in enumerate(ROWS):
        g = f[f["rung"] == rung]
        for j, (fam, B) in enumerate(COLS):
            sel = g[(g["family"] == fam) & ((g["B"] == B) if fam != "deep" else True)]
            if fam == "private_single":
                sel = g[g["family"] == "private_single"]
            if len(sel) != 1:
                continue
            r = sel.iloc[0]
            V[i, j] = r[value]
            # Nominal (uncorrected) paired interval. Nothing in this study survives the
            # Benjamini-Hochberg correction across a panel, which is stated in the caption.
            S[i, j] = bool(np.isfinite(r["delta_lo"]) and
                           (r["delta_lo"] > 0 or r["delta_hi"] < 0)) if value == "delta" else False
    return V, S


def draw(frame, outdir, value="delta", name="f3a_amplification_delta", partition="main"):
    st.style()
    scale = 100.0 if value == "delta" else 1.0
    allv = np.concatenate([_grids(frame, p, value)[0].ravel() for p in st.PANELS])
    allv = allv[np.isfinite(allv)] * scale
    if value == "delta":
        vmax = float(np.ceil(np.nanmax(np.abs(allv))))
        vmin, cmap, center = -vmax, "RdBu_r", True
        cbar_label = "Accuracy minus sample-and-vote\nat the same budget (points)"
    else:
        vmin, vmax, cmap, center = 0.0, 1.0, "viridis", False
        cbar_label = "Ceiling capture $\\eta$: share of the\noracle pass@$k$ headroom delivered"

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 6.2))
    axes = axes.ravel()
    im = None
    for pi, panel in enumerate(st.PANELS):
        ax = axes[pi]
        V, S = _grids(frame, panel, value)
        Vp = V * scale
        im = ax.imshow(Vp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        # Mask absent cells in a neutral tone so blank never reads as zero.
        for i in range(len(ROWS)):
            for j in range(len(COLS)):
                if not np.isfinite(Vp[i, j]):
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#F2F2F2",
                                           edgecolor="white", lw=0.4, zorder=2))
                    continue
                v = Vp[i, j]
                rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                dark = rel < 0.28 or rel > 0.80 if not center else abs(v) > 0.66 * vmax
                txt = f"{v:+.0f}" if value == "delta" else f"{v:.2f}".lstrip("0")
                ax.text(j, i, txt, ha="center", va="center", fontsize=4.6,
                        color="white" if dark else "#222222", zorder=3)
                if S[i, j]:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#000000", lw=1.0, zorder=4))
        for x in SEPS:
            ax.axvline(x, color="#FFFFFF", lw=1.6, zorder=5)
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels(COL_LABEL, fontsize=5.6, rotation=90)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([str(r) for r in ROWS], fontsize=5.4)
        ax.set_title(st.PANEL_TITLE[panel], fontsize=8)
        if pi % 3 == 0:
            ax.set_ylabel("Budget $C/A_0$  (= $k$ of pass@$k$)", fontsize=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    ax = axes[5]
    ax.axis("off")
    cax = ax.inset_axes([0.05, 0.86, 0.78, 0.030])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=6.2)
    cb.ax.tick_params(labelsize=6)
    note = ("PT = private team (no peer exposure).\n"
            "Rows are iso-compute: every cell in a row\n"
            "spends the same $C = B \\cdot R \\cdot A$.\n"
            "Grey = configuration does not exist\n"
            "at that budget.")
    if value == "delta":
        note += ("\nOutline = nominal paired 95% CI excludes 0.\n"
                 "No cell survives BH correction across\nthe 70 tests in a panel.")
    ax.text(0.05, 0.60, note, fontsize=6, va="top", ha="left", transform=ax.transAxes)
    fig.subplots_adjust(hspace=0.42, wspace=0.28)
    return st.save(fig, name, outdir)
