"""Figures for the forced_budget_v2 report. Serif, colour-blind-safe, one module."""
from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core as c

RUNG_TICKS = [1, 2, 4, 8, 16, 25]
PANELS = [*c.BENCH_ORDER, "macro"]


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


SHORT = {"aime": "AIME 2025+2026", "hmmt": "HMMT Feb 2025+2026", "omni_math_hard": "Omni-MATH ($\\geq$7)",
         "gpqa_diamond": "GPQA-Diamond", "macro": "Macro (4 benchmarks)"}


def title(lv):
    return SHORT[lv]


def panel_grid(h=4.3):
    """2 x 3 grid: five benchmark/macro panels plus one cell for the legend."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, h))
    axes = axes.ravel()
    axes[5].axis("off")
    return fig, axes[:5], axes[5]


# ------------------------------------------------------------------ F1 design
def f1_design(idx, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    grid(ax, "both")
    off = {"private_team": -0.06, "peer": 0.06}
    for fam in ["sample", "private_single", "private_team", "peer"]:
        g = idx[idx["family"] == fam]
        ax.scatter(g["B"] * np.exp(off.get(fam, 0)), g["R"], s=12, color=c.FAMILY_COLOR[fam],
                   label=c.FAMILY_LABEL[fam], zorder=3)
    d = idx[idx["family"] == "deep"]
    ax.scatter(np.ones(len(d)) * 0.7, d["rung"], s=22, marker="D", color=c.FAMILY_COLOR["deep"],
               label="Deep single call ($m\\cdot A$ in one call)", zorder=3)
    for C in (2, 4, 8, 16, 25):
        b = np.linspace(1, 25, 50)
        ax.plot(b, C / b, color="#BBBBBB", lw=0.6, zorder=1)
        ax.text(1.05, C * 1.03, f"$C/A$={C}", fontsize=5.5, color="#888888")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 25]); ax.set_xticklabels(["1", "2", "4", "8", "16", "25"])
    ax.set_yticks([1, 2, 4, 8, 16, 25]); ax.set_yticklabels(["1", "2", "4", "8", "16", "25"])
    ax.minorticks_off()
    ax.set_xlim(0.6, 30); ax.set_ylim(0.8, 30)
    ax.set_xlabel("Width $B$ (agents; deep plotted at left)"); ax.set_ylabel("Depth $R$ (rounds; deep: $m$)")
    ax.legend(loc="upper right", fontsize=6)
    save(fig, out, "f1_design")


# ------------------------------------------------------------------ F2 ladder
def f2_ladder(N, F, bank, out):
    fig, axes, lax = panel_grid()
    for ax, lv in zip(axes, PANELS):
        grid(ax)
        b = bank[bank["level"] == lv]
        ax.plot(b["k"], b["oracle_pass_at_k"], color=c.ORACLE_COLOR, ls="--", lw=1.0, label="Oracle pass@$k$")
        for fam, names in (("sample", [f"sample/{k}" for k in range(1, 26)]),
                           ("private_single", [f"private_single/{k}" for k in range(1, 26)]),
                           ("deep", ["sample/1", *[f"deep/{m}" for m in (2, 4, 8, 16, 25)]])):
            xs, ys, lo, hi = [], [], [], []
            for n in names:
                if n not in F.col:
                    continue
                a_, l_, h_ = F.acc(n, lv)
                xs.append(1 if n == "sample/1" else int(n.split("/")[-1])); ys.append(a_); lo.append(l_); hi.append(h_)
            ax.plot(xs, ys, color=c.FAMILY_COLOR[fam], marker="o" if fam == "deep" else None,
                    label=c.FAMILY_LABEL[fam] if fam != "sample" else "Sample + vote")
            if fam == "deep":
                ax.fill_between(xs, lo, hi, color=c.FAMILY_COLOR[fam], alpha=0.15, lw=0)
        teams = F.idx[F.idx["family"].isin(["private_team", "peer"])]
        for _, r in teams.iterrows():
            ax.scatter(r["rung"], F.acc(r["name"], lv)[0], s=5, color=c.FAMILY_COLOR[r["family"]], alpha=0.6, zorder=2)
        rung_axis(ax)
        ax.set_title(title(lv), fontsize=7.5)
        ax.set_xlabel("$C/A$")
        ax.set_ylim(0, 1.02)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Delivered accuracy")
    h, l = axes[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="o", color=c.FAMILY_COLOR["private_team"], ms=3),
          plt.Line2D([], [], ls="", marker="o", color=c.FAMILY_COLOR["peer"], ms=3)]
    l += ["Private team (any $n$, $r$)", "Peer (any $n$, $r$)"]
    lax.legend(h, l, loc="center", fontsize=7)
    fig.tight_layout()
    save(fig, out, "f2_ladder")


# ------------------------------------------------------------------ F3 depth vs width
def f3_depth_vs_width(N, out):
    fig, axes, lax = panel_grid()
    ms = [2, 4, 8, 16, 25]
    spec = (("deep_minus_sample", "Deep $-$ sample", c.FAMILY_COLOR["deep"]),
            ("private_minus_sample", "Private single $-$ sample", c.FAMILY_COLOR["private_single"]),
            ("deep_minus_private", "Deep $-$ private single", "#444444"))
    for ax, lv in zip(axes, PANELS):
        grid(ax)
        ax.axhline(0, color="#999999", lw=0.6)
        for j, (tag, lab, col) in enumerate(spec):
            e = [N["depth_vs_width"][lv][m][tag] for m in ms]
            x = np.array(ms) * np.exp((j - 1) * 0.07)
            y = [v["pts"] for v in e]
            ax.errorbar(x, y, yerr=[[v["pts"] - v["lo"] for v in e], [v["hi"] - v["pts"] for v in e]],
                        color=col, marker="o", ms=3, lw=1, capsize=1.5, label=lab)
        rung_axis(ax, ms)
        ax.set_title(title(lv), fontsize=7.5)
        ax.set_xlabel("$m$ (= $C/A$)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Accuracy difference (points)")
    h, l = axes[0].get_legend_handles_labels()
    lax.legend(h, l, loc="center", fontsize=7, title="paired, 95% item bootstrap", title_fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "f3_depth_vs_width")


# ------------------------------------------------------------------ F4 requested vs realized (v1 vs v2)
def f4_requested_vs_realized(N, out):
    if "v1_deep" not in N:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax in (a1, a2):
        grid(ax, "both")
    v1 = N["v1_deep"]
    m1 = sorted(v1)
    a1.plot([v1[m]["requested"] for m in m1], [v1[m]["realized"] for m in m1], marker="s", color="#E69F00",
            label="v1: Gemini 2.5 Flash-Lite, requested allowance")
    m2 = sorted(N["v2_deep_realized"])
    a1.plot([m * c.A0 for m in m2], [N["v2_deep_realized"][m] for m in m2], marker="o", color=c.FAMILY_COLOR["deep"],
            label="v2: Qwen3.8-27B, forced")
    lim = [800, 30000]
    a1.plot(lim, lim, color="#999999", lw=0.6, ls=":")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Requested reasoning tokens (deep/$m$)"); a1.set_ylabel("Realized reasoning tokens")
    a1.set_title("(a) Tokens one deep call actually spent")
    a1.legend(fontsize=6, loc="upper left")
    # (b) deep minus sample at matched nominal m, v1 vs v2 (macro)
    d1 = [(m, 100 * (v1[m]["acc"] - v1[m]["sample_acc"])) for m in m1 if m > 1 and v1[m]["sample_acc"] is not None]
    d2 = [(m, N["depth_vs_width"]["macro"][m]["deep_minus_sample"]["pts"]) for m in (2, 4, 8, 16, 25)]
    a2.axhline(0, color="#999999", lw=0.6)
    a2.plot([m for m, _ in d1], [d for _, d in d1], marker="s", color="#E69F00", label="v1 (allowance)")
    a2.plot([m for m, _ in d2], [d for _, d in d2], marker="o", color=c.FAMILY_COLOR["deep"], label="v2 (forced)")
    rung_axis(a2, [2, 4, 8, 16, 25])
    a2.set_xlabel("$m$"); a2.set_ylabel("deep/$m$ $-$ sample/$m$ (points, macro)")
    a2.set_title("(b) deep/$m$ vs sample/$m$, same nominal $C$")
    a2.legend(fontsize=6)
    save(fig, out, "f4_requested_vs_realized")


# ------------------------------------------------------------------ F5 amplification matrix
def f5_matrix(iso, out):
    fams = ["private_single", "private_team", "peer", "deep"]
    g = iso[iso["family"].isin(fams)].copy()
    order = (g[g["level"] == "macro"].sort_values(["family", "rung", "B", "name"], key=lambda s: s.map(
        {f: i for i, f in enumerate(fams)}) if s.name == "family" else s)["name"].tolist())
    lvls = PANELS
    M = np.full((len(lvls), len(order)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, lv in enumerate(lvls):
        sub = g[g["level"] == lv].set_index("name")
        for j, n in enumerate(order):
            if n in sub.index:
                M[i, j] = 100 * sub.loc[n, "delta"]
                S[i, j] = bool(sub.loc[n, "sig"])
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-v, vmax=v, interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if S[i, j]:
                ax.plot(j, i, marker=".", color="black", ms=2.2)
    ax.set_yticks(range(len(lvls))); ax.set_yticklabels([c.BENCH_PLAIN[l] for l in lvls], fontsize=6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=4.6)
    fams_of = [n.split("/")[0] for n in order]
    for j in range(1, len(order)):
        if fams_of[j] != fams_of[j - 1]:
            ax.axvline(j - 0.5, color="black", lw=0.6)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("vs sample/$C$ (pts)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, out, "f5_amplification_matrix")


# ------------------------------------------------------------------ F6 selection
def f6_selection(N, bank, out):
    fig, axes, lax = panel_grid()
    for ax, lv in zip(axes, PANELS):
        grid(ax)
        b = bank[bank["level"] == lv]
        ax.plot(b["k"], b["oracle_pass_at_k"], color="black", ls="--", lw=1, label="Oracle pass@$k$")
        ax.plot(b["k"], b["readout"], color=c.FAMILY_COLOR["sample"], label="Vote readout")
        ax.plot(b["k"], b["mean_draw_correct"], color="#AAAAAA", lw=0.8, label="Mean single draw")
        ms = [2, 4, 8, 16, 25]
        ax.scatter(ms, [N["acc"][lv][f"deep/{m}"]["acc"] for m in ms], color=c.FAMILY_COLOR["deep"], s=12,
                   zorder=3, label="deep/$m$ (one call)")
        rung_axis(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(title(lv), fontsize=7.5); ax.set_xlabel("$k$ or $m$")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Accuracy")
    h, l = axes[0].get_legend_handles_labels()
    lax.legend(h, l, loc="center", fontsize=7)
    fig.tight_layout()
    save(fig, out, "f6_selection")


# ------------------------------------------------------------------ F7 communication
def f7_communication(comm, out):
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    for ax, lv in zip(axes, PANELS):
        g = comm[comm["level"] == lv]
        M = np.full((4, 4), np.nan)
        for _, r in g.iterrows():
            M[int(r["n"]) - 2, int(r["r"]) - 1] = 100 * r["delta"]
        v = 25
        im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, origin="lower")
        for i in range(4):
            for j in range(4):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=5.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([1, 2, 3, 4]); ax.set_yticks(range(4)); ax.set_yticklabels([2, 3, 4, 5])
        ax.set_xlabel("peer rounds $r$")
        ax.set_title({"aime": "AIME", "hmmt": "HMMT", "omni_math_hard": "Omni-MATH ($\\geq$7)",
                      "gpqa_diamond": "GPQA-Diamond", "macro": "Macro"}[lv], fontsize=7.5)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("team size $n$")
    save(fig, out, "f7_communication")


# ------------------------------------------------------------------ F8 effective depth
def f8_effective_depth(N, deep_nodes, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax in (a1, a2):
        grid(ax)
    ms = sorted(N["effective_depth"], key=int)
    x = np.arange(len(ms))
    a1.bar(x, [N["effective_depth"][m]["looped_share"] for m in ms], color=c.FAMILY_COLOR["deep"], width=0.6)
    for xi, m in zip(x, ms):
        e = N["effective_depth"][m]
        a1.text(xi, e["looped_share"] + 0.008, f"{e['eff_fraction_mean']:.2f}", ha="center", fontsize=6)
    a1.set_xticks(x); a1.set_xticklabels([f"deep/{m}" for m in ms])
    a1.set_ylabel("Share of deep calls that loop")
    a1.set_title("(a) Repetition by forced length\n(label: mean effective share of tokens)", fontsize=7.5)
    acc = N["deep_acc_by_loop"]
    mm = [m for m in ms if m in acc]
    xx = np.arange(len(mm))
    a2.bar(xx - 0.18, [acc[m]["clean"] for m in mm], width=0.34, color="#BBBBBB", label="call did not loop")
    a2.bar(xx + 0.18, [acc[m]["looped"] or 0 for m in mm], width=0.34, color=c.FAMILY_COLOR["deep"],
           label="call looped")
    a2.set_xticks(xx); a2.set_xticklabels([f"deep/{m}" for m in mm])
    a2.set_ylim(0, 1.05); a2.set_ylabel("Accuracy of the call")
    a2.set_title("(b) Looping calls are the ones that had already solved it", fontsize=7.5)
    a2.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, out, "f8_effective_depth")


# ------------------------------------------------------------------ F9 dollars
def f9_dollars(csm, N, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    grid(ax, "both")
    for fam, g in csm.groupby("family"):
        ax.scatter(g["mean_total_charged_usd"] * 1000, g["acc"], s=10, color=c.FAMILY_COLOR[fam],
                   label=c.FAMILY_LABEL[fam], zorder=3)
    fr = csm[csm["name"].isin(N["frontier"])].sort_values("mean_total_charged_usd")
    ax.step(fr["mean_total_charged_usd"] * 1000, fr["acc"], where="post", color="black", lw=0.8, label="Frontier")
    for _, r in fr.iterrows():
        ax.annotate(r["name"], (r["mean_total_charged_usd"] * 1000, r["acc"]), fontsize=5, xytext=(3, -6),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per item (USD $\\times 10^{-3}$, log)"); ax.set_ylabel("Macro accuracy")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "f9_dollars")


# ------------------------------------------------------------------ F10 robustness by provider
def f10_robustness(N, out):
    rob = N.get("robustness") or {}
    if not rob:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    grid(ax)
    ms = [8, 16, 25]
    for j, (sid, d) in enumerate(sorted(rob.items())):
        x = np.arange(len(ms)) + (j - 0.5) * 0.35
        ax.bar(x, [d["deep_vs_sample"][m]["delta_pts"] for m in ms], width=0.33,
               label=f"{sid} ({d['items']} items)", color=["#0072B2", "#E69F00"][j % 2])
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"$m$={m}" for m in ms])
    ax.set_ylabel("deep/$m$ $-$ sample/$m$ (points)")
    ax.legend(fontsize=6)
    save(fig, out, "f10_robustness")
