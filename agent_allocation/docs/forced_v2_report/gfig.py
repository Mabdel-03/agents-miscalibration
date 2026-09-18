"""Figures for the generation/selection analysis (gap.py)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core as c
from figures import PANELS, grid, panel_grid, rung_axis, save, title

FAMS = ["sample", "private_single", "private_team", "peer", "deep"]


def g1_plane(rung_tab, out):
    """Generation-selection plane: archive coverage (x) against delivered accuracy (y), one point per configuration."""
    fig, axes, lax = panel_grid(h=4.6)
    for ax, lv in zip(axes, PANELS):
        grid(ax, "both")
        t = rung_tab[(rung_tab.level == lv) & (rung_tab.family.isin(FAMS))]
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls=":")
        for fam in FAMS:
            g = t[t.family == fam]
            ax.scatter(g["cov_archive"], g["delivered"], s=4 + 1.2 * g["rung"], color=c.FAMILY_COLOR[fam],
                       alpha=0.75, lw=0, label=c.FAMILY_LABEL[fam], zorder=3)
        s = t[t.family == "sample"].sort_values("rung")
        ax.plot(s["pass_at_k"], s["delivered"], color="black", lw=0.8, ls="--", label="pass@$k$ vs vote@$k$", zorder=2)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.set_title(title(lv), fontsize=7.5)
        ax.set_xlabel("Coverage of all candidates generated")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Delivered accuracy")
    h, l = axes[0].get_legend_handles_labels()
    lax.legend(h, l, loc="center", fontsize=7, title="marker size ~ $C/A$", title_fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "g1_generation_selection_plane")


def g2_decomposition(fam_tab, out):
    """Per family against C: best delivered (solid) and best archive coverage (dashed), with pass@k."""
    fig, axes, lax = panel_grid(h=4.4)
    for ax, lv in zip(axes, PANELS):
        grid(ax)
        t = fam_tab[fam_tab.level == lv]
        pk = t[t.family == "sample"].sort_values("rung")
        ax.plot(pk["rung"], pk["pass_at_k"], color="black", ls="--", lw=1, label="pass@$k$ at $A$")
        for fam in FAMS:
            g = t[t.family == fam].sort_values("rung")
            if fam in ("private_team", "peer"):
                g = g[g.rung.isin([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25])]
            ax.plot(g["rung"], g["delivered"], color=c.FAMILY_COLOR[fam], lw=1.2, label=f"{c.FAMILY_LABEL[fam]}: delivered")
            if fam not in ("deep", "sample"):
                ax.plot(g["rung"], g["cov_archive"], color=c.FAMILY_COLOR[fam], lw=0.9, ls=":",
                        label=f"{c.FAMILY_LABEL[fam]}: coverage")
        rung_axis(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(title(lv), fontsize=7.5)
        ax.set_xlabel("$C/A$")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Accuracy / coverage")
    h, l = axes[0].get_legend_handles_labels()
    lax.legend(h, l, loc="center", fontsize=6.3)
    fig.tight_layout()
    save(fig, out, "g2_family_decomposition")


def g3_verifier(vs, N, out):
    rungs = [4, 8, 16, 25]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
    for ax, r in zip(axes, rungs):
        grid(ax)
        for fam, col, lab in (("sample", c.FAMILY_COLOR["sample"], "$m$ independent samples"),
                              ("private_single", c.FAMILY_COLOR["private_single"], "private chain of $m$ calls, all answers"),
                              ("team", c.FAMILY_COLOR["private_team"], "best-coverage team, all answers")):
            g = vs[(vs.rung == r) & (vs.family == fam)].sort_values("auc")
            if len(g):
                ax.plot(g["auc"], g["macro"], marker="o", ms=2.5, color=col, label=lab)
        ax.axhline(N["acc"]["macro"][f"deep/{r}"]["acc"], color=c.FAMILY_COLOR["deep"], lw=1.2, label=f"deep/$m$ (one call)")
        ax.set_title(f"$C = {r}A$", fontsize=7.5)
        ax.set_xlabel("Verifier pairwise AUC")
        ax.set_xticks([0.5, 0.7, 0.9, 1.0])
    axes[0].set_ylabel("Macro accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "g3_verifier_sweep")


def g4_strata(G, out):
    st = G["strata"]
    cfgs = [("sample/25", "sample"), ("private_single/25", "private_single"), ("peer/5/4", "peer"),
            ("private_team/5/4", "private_team"), ("deep/4", "deep"), ("deep/16", "deep")]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    grid(ax)
    x = np.arange(len(st))
    w = 0.13
    for i, (cfg, fam) in enumerate(cfgs):
        vals = [s[cfg] if s[cfg] is not None else 0 for s in st]
        alpha = 0.55 if cfg == "deep/4" else 1.0
        ax.bar(x + (i - 2.5) * w, vals, width=w, color=c.FAMILY_COLOR[fam], alpha=alpha, label=cfg)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['stratum']} correct\n($n$={s['n']})" for s in st], fontsize=6.5)
    ax.set_xlabel("Items grouped by how many of their 25 bank samples were correct")
    ax.set_ylabel("Delivered accuracy"); ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, out, "g4_difficulty_strata")


def g5_vote(G, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax in (a1, a2):
        grid(ax)
    cal = [r for r in G["calibration"] if r["n"]]
    x = np.arange(len(cal))
    a1.bar(x - 0.18, [r["plurality_correct"] for r in cal], width=0.34, color=c.FAMILY_COLOR["sample"],
           label="plurality of 25 samples correct")
    a1.bar(x + 0.18, [r["deep16"] for r in cal], width=0.34, color=c.FAMILY_COLOR["deep"], label="deep/16 correct")
    a1.set_xticks(x); a1.set_xticklabels([f"{r['bin']}\n$n$={r['n']}" for r in cal], fontsize=6.3)
    a1.set_xlabel("Share of the 25 votes held by the plurality answer"); a1.set_ylabel("Accuracy")
    a1.set_title("(a) Vote margin as a confidence signal", fontsize=7.5)
    a1.set_ylim(0, 1.3); a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a1.legend(fontsize=5.8, loc="upper left", ncol=2)
    dv = [r for r in G["diversity"] if r["n"]]
    x = np.arange(len(dv))
    a2.bar(x - 0.26, [r["coverage25"] for r in dv], width=0.26, color="#BBBBBB", label="coverage of 25 samples")
    a2.bar(x, [r["vote25"] for r in dv], width=0.26, color=c.FAMILY_COLOR["sample"], label="vote of 25")
    a2.bar(x + 0.26, [r["deep16"] for r in dv], width=0.26, color=c.FAMILY_COLOR["deep"], label="deep/16")
    a2.set_xticks(x); a2.set_xticklabels([f"{r['classes']}\n$n$={r['n']}" for r in dv], fontsize=6.3)
    a2.set_xlabel("Distinct answers among the 25 samples"); a2.set_title("(b) Answer diversity", fontsize=7.5)
    a2.set_ylim(0, 1.3); a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.legend(fontsize=5.8, loc="upper right", ncol=3)
    fig.tight_layout()
    save(fig, out, "g5_vote_calibration")


def g6_portfolios(prt, N, out):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    grid(ax, "both")
    p = prt[prt["C"] <= 30]
    ax.scatter(p["C"], p["cov"], marker="^", color="#999999", s=16, label="portfolio coverage (any call correct)")
    ax.scatter(p["C"], p["vote"], marker="o", color=c.FAMILY_COLOR["deep"], s=14, label="portfolio: vote, tie to deepest")
    for _, r in p.iterrows():
        if r["portfolio"] in ("16", "25", "16+8", "4+2+1", "8+4+2+1", "16+8+4"):
            ax.annotate(r["portfolio"], (r["C"], r["vote"]), fontsize=5.2, xytext=(3, -7), textcoords="offset points")
    ks = list(range(1, 26))
    ax.plot(ks, [N["oracle"]["macro"][k] for k in ks], color="black", ls="--", lw=0.9, label="pass@$k$ at $A$")
    ax.plot(ks, [N["readout"]["macro"][k] for k in ks], color=c.FAMILY_COLOR["sample"], lw=1, label="vote@$k$ at $A$")
    ax.set_xlabel("$C/A$"); ax.set_ylabel("Macro accuracy")
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "g6_deep_portfolios")


def g7_transitions(G, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax in (a1, a2):
        grid(ax)
        ax.axhline(0, color="#999999", lw=0.6)
    tr = G["transitions"]
    ps = tr.get("private_single", [])
    st = pd.DataFrame(ps)
    sm = st[["W_to_C", "C_to_W"]].rolling(3, min_periods=1, center=True).mean()
    a1.plot(st["step"], sm["W_to_C"], color=c.FAMILY_COLOR["private_single"], label="wrong $\\to$ right (3-step mean)")
    a1.plot(st["step"], sm["C_to_W"], color="#D55E00", label="right $\\to$ wrong (3-step mean)")
    a1.set_xlabel("Revision step of the private chain"); a1.set_ylabel("Share of items")
    a1.set_title("(a) One agent revising its own answer", fontsize=7.5); a1.legend(fontsize=6)
    for n, ls in ((2, ":"), (3, "-."), (4, "--"), (5, "-")):
        for fam, col in (("peer", c.FAMILY_COLOR["peer"]), ("private_team", c.FAMILY_COLOR["private_team"])):
            t = tr.get(f"{fam}/{n}", [])
            if t:
                a2.plot([r["step"] + 1 for r in t], [r["net"] / n for r in t], color=col, ls=ls, lw=1,
                        label=f"{'peer' if fam == 'peer' else 'private'} $n$={n}")
    a2.set_xticks([1, 2, 3, 4]); a2.set_xlabel("Peer / revision round"); a2.set_ylabel("Net corrections per agent (W$\\to$C $-$ C$\\to$W)")
    a2.set_title("(b) Teams: net corrections per agent by round", fontsize=7.5)
    a2.legend(fontsize=5.3, ncol=2)
    fig.tight_layout()
    save(fig, out, "g7_round_dynamics")


def g8_slopes(G, out):
    fams = [("sample", "Sample + vote"), ("private_single", "Private single"), ("deep", "Deep single call")]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    grid(ax, "x")
    levels = PANELS
    y0 = np.arange(len(levels))
    for i, (fam, lab) in enumerate(fams):
        y = y0 + (i - 1) * 0.22
        e = [G["slopes"][lv][fam] for lv in levels]
        ax.errorbar([v["pts_per_doubling"] for v in e], y,
                    xerr=[[v["pts_per_doubling"] - v["lo"] for v in e], [v["hi"] - v["pts_per_doubling"] for v in e]],
                    fmt="o", ms=3, color=c.FAMILY_COLOR[fam], capsize=1.5, label=lab)
    ax.scatter([G["slopes"][lv]["pass_at_k"]["pts_per_doubling"] for lv in levels], y0 + 0.4, marker="|", s=60,
               color="black", label="pass@$k$ (ceiling of sampling)")
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_yticks(y0); ax.set_yticklabels([title(lv) for lv in levels], fontsize=6.5)
    ax.set_xlabel("Accuracy points per doubling of compute ($C$ = 1 to 16 $A$)")
    ax.invert_yaxis()
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, out, "g8_scaling_slopes")


def g9_team_shape(G, out):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    grid(ax)
    Cs = sorted(G["team_shape"], key=int)
    for i, C in enumerate(Cs):
        for e in G["team_shape"][C]:
            fam, n, r = e["name"].split("/")
            ax.scatter(i + (0.12 if fam == "peer" else -0.12), e["acc"], s=10 + 6 * int(n),
                       color=c.FAMILY_COLOR[fam], alpha=0.8, zorder=3)
            ax.annotate(f"{n}x{int(r) + 1}", (i + (0.12 if fam == "peer" else -0.12), e["acc"]), fontsize=4.8,
                        xytext=(3, 0), textcoords="offset points")
    ax.set_xticks(range(len(Cs))); ax.set_xticklabels([f"$C$={C}$A$" for C in Cs], fontsize=6.5)
    ax.set_ylabel("Macro accuracy")
    ax.scatter([], [], color=c.FAMILY_COLOR["private_team"], label="private team"); ax.scatter([], [], color=c.FAMILY_COLOR["peer"], label="peer")
    ax.legend(fontsize=6); ax.set_title("Team shapes at equal compute (label: agents x total rounds)", fontsize=7.5)
    save(fig, out, "g9_team_shape")
