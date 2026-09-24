"""dynorch_v1, WP-C: the nine figures d1..d9 of the experiment design section 8 (contract section 7).

Same style as ``docs/forced_v2_report/figures.py`` (serif, Okabe-Ito colour-blind-safe hues, one axis per panel):
ladder families keep their sealed colours, the three arms get fixed hues plus a fixed marker shape as a second
encoding (``core.ARM_COLOR`` / ``core.ARM_MARKER``), the oracle is the black dashed line, and every multi-series
panel carries a legend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import core as c  # noqa: E402

RUNG_TICKS = [1, 2, 4, 8, 16, 25]
PANELS = [*c.BENCH_ORDER, "macro"]
SHORT = {"aime": "AIME 2025+2026", "hmmt": "HMMT Feb 2025+2026", "omni_math_hard": "Omni-MATH ($\\geq$7)",
         "gpqa_diamond": "GPQA-Diamond", "macro": "Macro (4 benchmarks)"}
WRAPPED = {"aime": "AIME\n2025+2026", "hmmt": "HMMT Feb\n2025+2026", "omni_math_hard": "Omni-MATH\n($\\geq$7)",
           "gpqa_diamond": "GPQA-Diamond", "macro": "Macro\n(4 benchmarks)"}


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "lines.linewidth": 1.3, "lines.markersize": 3.5, "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def rung_axis(ax, ticks=RUNG_TICKS):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(ticks) / 1.25, max(ticks) * 1.25)


def save(fig, out, name):
    fig.savefig(f"{out}/figures/{name}.pdf")
    plt.close(fig)


def panel_grid(h=4.3):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, h))
    axes = axes.ravel()
    axes[5].axis("off")
    return fig, axes[:5], axes[5]


def arm_style(arm):
    hollow = arm == "H0"
    return dict(color=c.ARM_COLOR[arm], marker=c.ARM_MARKER[arm], mfc="white" if hollow else c.ARM_COLOR[arm],
                mec=c.ARM_COLOR[arm], ms=4.5)


# ------------------------------------------------------------------ d1 design
def d1_design(N, cfg, arms, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.axis("off")
    rungs = [int(r) for r in cfg["dynorch"]["rungs"]]
    # row labels sit right-aligned in their own column left of the first box (boxes span j - 0.45 .. j + 0.45)
    ax.set_xlim(-3.1, len(rungs) + 3.6)
    ax.set_ylim(-0.6, len(arms) + 0.6)
    for j, k in enumerate(rungs):
        ax.text(j, len(arms) - 0.45, f"$C={k}A$\n{k * c.A0:,}", ha="center", va="bottom", fontsize=7)
    for i, arm in enumerate(reversed(arms)):
        label = c.ARM_LABEL.get(arm, arm).replace(" harness-free control", "\nharness-free control")
        ax.text(-0.62, i, label, ha="right", va="center", fontsize=7.5,
                color="#000000" if arm == "H0" else c.ARM_COLOR.get(arm, "k"), linespacing=1.1)
        for j, k in enumerate(rungs):
            n = N["n_runs_by_arm_rung"].get(f"{arm}/{k}", 0)
            comp = N["completeness"].get(f"{arm}/{k}")
            face = "#EEEEEE" if comp is None else plt.cm.Greys(0.08 + 0.25 * comp)
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=face, edgecolor="#888888", lw=0.5))
            ax.text(j, i, f"{n}", ha="center", va="center", fontsize=7)
    text = (f"one run per cell; $A$ = {c.A0:,} completion tokens\n"
            f"cap $C = kA$ on all completion tokens of every call\n"
            f"one call in flight per run; grant = $C -$ spent\n"
            f"grace $G$ = {cfg['dynorch']['grace_tokens']:,} outside $C$, root only, once\n"
            f"primary family at $k$ = 16, Holm over 4; seed {c.SEED}")
    ax.text(len(rungs) - 0.3, (len(arms) - 1) / 2, text, fontsize=6.8, va="center", ha="left")
    ax.set_title("Cells and budget semantics (numbers: runs with a terminal record)", fontsize=8)
    save(fig, out, "d1_design")


# ------------------------------------------------------------------ d2 ladder with harness points (F1)
def d2_ladder_f1(N, F, bank, acc, hnames, out):
    fig, axes, lax = panel_grid()
    arms = sorted({c.arm_of(n) for n in hnames if c.repeat_of(n) == 1}, key=lambda a: (a not in c.CORE_ARMS, a))
    for ax, lv in zip(axes, PANELS):
        grid(ax)
        b = bank[bank["level"] == lv]
        ax.plot(b["k"], b["oracle_pass_at_k"], color=c.ORACLE_COLOR, ls="--", lw=1.0, label="Oracle pass@$k$")
        for fam, names in (("sample", [f"sample/{k}" for k in range(1, 26)]),
                           ("private_single", [f"private_single/{k}" for k in range(1, 26)]),
                           ("deep", ["sample/1", *[f"deep/{m}" for m in (2, 4, 8, 16, 25)]])):
            xs, ys = [], []
            for n in names:
                if n in acc:
                    xs.append(1 if n == "sample/1" else int(n.split("/")[-1]))
                    ys.append(acc[n][lv]["acc"])
            if xs:
                ax.plot(xs, ys, color=c.FAMILY_COLOR[fam], marker="o" if fam == "deep" else None, ms=2.5,
                        label=c.FAMILY_LABEL[fam] if fam != "sample" else "Sample + vote")
        for j, arm in enumerate(arms):
            pts_ = [(c.rung_of(n), acc[n][lv]) for n in hnames if c.arm_of(n) == arm and c.repeat_of(n) == 1]
            if not pts_:
                continue
            x = np.array([p[0] for p in pts_]) * np.exp((j - 1) * 0.05)
            y = [p[1]["acc"] for p in pts_]
            ax.errorbar(x, y, yerr=[[p[1]["acc"] - p[1]["lo"] for p in pts_], [p[1]["hi"] - p[1]["acc"] for p in pts_]],
                        ls="", capsize=1.5, lw=0.8, label=c.ARM_LABEL.get(arm, arm), zorder=4, **arm_style(arm))
        rung_axis(ax)
        ax.set_title(SHORT[lv], fontsize=7.5)
        ax.set_xlabel("$C/A$ (nominal)")
        ax.set_ylim(0, 1.02)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Delivered accuracy")
    h, l = axes[4].get_legend_handles_labels()
    lax.legend(h, l, loc="center", fontsize=7, title="F1: nominal cap; 95% item bootstrap", title_fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "d2_ladder_f1")


# ------------------------------------------------------------------ d3 / d4 frontiers
def _frontier(front, measure, xlabel, name, out, log=True):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    grid(ax, "both")
    lad = front[front["side"] == "ladder"]
    for fam, g in lad.groupby("family"):
        ax.scatter(g[measure], g["acc"], s=12, color=c.FAMILY_COLOR.get(fam, "#999999"),
                   label=c.FAMILY_LABEL.get(fam, fam), zorder=3)
    har = front[front["side"] == "harness"]
    for arm, g in har.groupby(har["name"].map(c.arm_of)):
        st = arm_style(arm)
        ax.scatter(g[measure], g["acc"], s=26, marker=st["marker"], facecolor=st["mfc"], edgecolor=st["mec"],
                   linewidths=0.9, label=c.ARM_LABEL.get(arm, arm), zorder=4)
        for _, r in g.iterrows():
            ax.annotate(r["name"].split("/")[1], (r[measure], r["acc"]), fontsize=5, xytext=(3, 2),
                        textcoords="offset points", color=c.ARM_COLOR[arm])
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if measure == "x_tokens":
            ticks = [1000, 2000, 5000, 10000, 20000, 50000]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
        else:
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="upper left")
    save(fig, out, name)


def d3_realized_f2(N, front, out):
    _frontier(front, "x_tokens", "Realized completion tokens per item (F2)", "d3_realized_f2", out)


def d4_dollars(N, front, out):
    _frontier(front, "usd", "Charged USD per item", "d4_dollars", out)


# ------------------------------------------------------------------ d5 per benchmark at k = 16
def d5_per_benchmark(N, acc, arms, best_non_deep, out):
    cols = [("sample/16", c.FAMILY_COLOR["sample"]), ("private_single/16", c.FAMILY_COLOR["private_single"])]
    if best_non_deep.get(16):
        cols.append((best_non_deep[16], c.FAMILY_COLOR.get(best_non_deep[16].split("/")[0], "#999999")))
    cols.append(("deep/16", c.FAMILY_COLOR["deep"]))
    cols += [(f"{a}/16", c.ARM_COLOR[a]) for a in arms if f"{a}/16" in acc]
    cols = [(n, col) for n, col in dict(cols).items() if n in acc]  # best non-deep may equal private_single/16
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    grid(ax)
    width = 0.8 / len(cols)
    for j, (n, col) in enumerate(cols):
        xs = np.arange(len(PANELS)) + (j - len(cols) / 2 + 0.5) * width
        ys = [acc[n][lv]["acc"] for lv in PANELS]
        err = [[acc[n][lv]["acc"] - acc[n][lv]["lo"] for lv in PANELS], [acc[n][lv]["hi"] - acc[n][lv]["acc"] for lv in PANELS]]
        hollow = c.arm_of(n) == "H0"
        ax.bar(xs, ys, width * 0.92, color="white" if hollow else col, edgecolor=col, linewidth=0.8, label=n,
               yerr=err, error_kw=dict(lw=0.6, capsize=1.2, ecolor="#444444"), zorder=3)
    ax.set_xticks(np.arange(len(PANELS)))
    ax.set_xticklabels([WRAPPED[lv] for lv in PANELS], fontsize=7)
    ax.set_ylabel("Delivered accuracy at $C = 16A$")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(cols), fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.0)
    save(fig, out, "d5_per_benchmark")


# ------------------------------------------------------------------ d6 diagnostics vs k
def d6_diagnostics(N, arms, base_names, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    specs = [("calls", "Calls per run"), ("spawn_share", "Share of runs with $\\geq$ 1 spawn"),
             ("orch_share", "Orchestrator token share")]
    for ax, (key, lab) in zip(axes, specs):
        grid(ax)
        for arm in arms:
            pts_ = [(c.rung_of(n), N["diagnostics"][n][key]) for n in base_names
                    if c.arm_of(n) == arm and N["diagnostics"].get(n) and N["diagnostics"][n][key] is not None]
            if pts_:
                ax.plot([p[0] for p in pts_], [p[1] for p in pts_], label=c.ARM_LABEL.get(arm, arm), **arm_style(arm))
        rung_axis(ax)
        ax.set_xlabel("$C/A$")
        ax.set_title(lab, fontsize=7.5)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, out, "d6_diagnostics")


# ------------------------------------------------------------------ d7 cap binding
def d7_cap_binding(current, arms, out):
    rungs = sorted(current["rung"].unique())
    fig, axes = plt.subplots(1, max(1, len(rungs)), figsize=(7.0, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1.05, 22)
    for ax, k in zip(axes, rungs):
        grid(ax)
        for arm in arms:
            g = current[(current["rung"] == k) & (current["arm"] == arm)]
            if len(g):
                ax.hist(np.clip(g["realized_share"], 0, 1.05), bins=bins, histtype="step", color=c.ARM_COLOR[arm],
                        lw=1.0, label=c.ARM_LABEL.get(arm, arm), ls="--" if arm == "H0" else "-")
        ax.axvline(1.0, color="#999999", lw=0.6)
        ax.set_title(f"$C={k}A$", fontsize=7.5)
        ax.set_xlabel("realized / $C$")
    axes[0].set_ylabel("runs")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out, "d7_cap_binding")


# ------------------------------------------------------------------ d8 provider sensitivity
def d8_provider_sensitivity(N, out):
    contrasts = list(N["provider_sensitivity"])
    subsets = [("mancer_items", "Mancer items", "#0072B2"), ("parasail_items", "Parasail items", "#D55E00"),
               ("same_provider_only", "same provider only", "#009E73")]
    fig, ax = plt.subplots(figsize=(5.2, 0.5 + 0.55 * max(1, len(contrasts))))
    grid(ax, "x")
    ax.axvline(0, color="#999999", lw=0.6)
    for i, name in enumerate(contrasts):
        for j, (tag, lab, col) in enumerate(subsets):
            e = N["provider_sensitivity"][name][tag]
            if e["pts"] is None:
                continue
            y = i + (j - 1) * 0.22
            ax.errorbar([e["pts"]], [y], xerr=[[e["pts"] - e["lo"]], [e["hi"] - e["pts"]]], marker="o", ms=3,
                        color=col, capsize=1.5, lw=0.8, label=lab if i == 0 else None)
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels(contrasts, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("difference (points, macro over the items of the subset)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(subsets))
    save(fig, out, "d8_provider_sensitivity")


# ------------------------------------------------------------------ d9 repeat agreement
def d9_repeat_agreement(N, F, out):
    cells = N["repeat_agreement"]["cells"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6) if cells else (4.2, 0.4))
    if not cells:
        ax.axis("off")
        ax.text(0.5, 0.5, "No second repeat (E1) in this run: repeat agreement is not available.",
                ha="center", va="center", fontsize=8)
        save(fig, out, "d9_repeat_agreement")
        return
    grid(ax, "both")
    for name, e in cells.items():
        arm = c.arm_of(name)
        ax.errorbar([c.rung_of(name)], [e["delta_pts"]], yerr=[[e["delta_pts"] - e["lo"]], [e["hi"] - e["delta_pts"]]],
                    capsize=1.5, lw=0.8, ls="", label=f"{c.ARM_LABEL.get(arm, arm)} (kappa {e['kappa']:.2f})",
                    **arm_style(arm))
    ax.axhline(0, color="#999999", lw=0.6)
    rung_axis(ax)
    ax.set_xlabel("$C/A$")
    ax.set_ylabel("repeat 2 $-$ repeat 1 (points)")
    ax.legend(fontsize=6)
    save(fig, out, "d9_repeat_agreement")
