"""F4-F9: width versus depth, the selection gap, communication cost, round dynamics,
depth in one call, and the dollar frontier."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st

RUNGS_WD = [4, 6, 8, 10, 12, 15, 20, 25]


# ---------------------------------------------------------------- F4 width vs depth
def width_depth(frame, outdir, name="f4_width_vs_depth"):
    """Allocation shape at fixed compute, measured against sampling at the same budget.

    B=1 is pure depth (one agent revising). B=C/A0 is pure width (independent samples), which is
    the baseline itself and therefore sits at zero by construction. Everything between is a mix.
    """
    st.style()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    axes = axes.ravel()
    WID = [1, 2, 3, 4, 5]
    for pi, panel in enumerate(st.PANELS):
        ax = axes[pi]
        f = frame[(frame["level"] == panel) & (frame["rung"].isin(RUNGS_WD))]
        for fam, off in [("private_team", 0.06), ("peer", -0.06)]:
            sub = f[f["family"].isin([fam] if fam == "peer" else ["private_single", fam])]
            sub = sub[sub["B"].isin(WID)]
            ax.scatter(sub["B"] + off, sub["delta"] * 100, s=5, alpha=0.30,
                       color=ic.FAMILY_COLOR[fam], linewidths=0, zorder=2)
            m = sub.groupby("B")["delta"].mean().reset_index().sort_values("B")
            ax.plot(m["B"], m["delta"] * 100, color=ic.FAMILY_COLOR[fam], marker="o", ms=3.5,
                    label=ic.FAMILY_LABEL[fam], zorder=3)
        ax.axhline(0, color="#000000", lw=0.9, zorder=1)
        ax.text(0.97, 0.04, "pure width\n(sampling baseline)", transform=ax.transAxes,
                fontsize=5.4, ha="right", va="bottom", color="#555555")
        st.grid(ax)
        ax.set_xticks(WID)
        ax.set_title(st.PANEL_TITLE[panel])
        if pi % 3 == 0:
            ax.set_ylabel("Accuracy minus sampling\nat same budget (points)")
        if pi >= 2:
            ax.set_xlabel("Width $B$  (depth $R = C/(A_0 B)$)")
    ax = axes[5]; ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", fontsize=7, handlelength=1.6,
              title="Allocation family", title_fontsize=7)
    ax.text(0.5, 0.42, "Averaged over the budgets\n$C/A_0 \\in \\{4,6,8,10,12,15,20,25\\}$.\n"
            "Faint points are individual\nbudget cells. $B=1$ is one agent\nrevising $R$ times; the zero\nline is $B=C/A_0$ independent\nsamples with a vote.",
            transform=ax.transAxes, fontsize=6.2, ha="center", va="top")
    fig.subplots_adjust(hspace=0.45, wspace=0.34)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F5 selection gap
def selection_gap(frame, curve, outdir, name="f5_selection_gap"):
    st.style()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    axes = axes.ravel()
    for pi, panel in enumerate(st.PANELS):
        ax = axes[pi]
        c = curve[curve["level"] == panel].sort_values("k")
        f = frame[frame["level"] == panel]
        best = f.groupby("rung")["acc"].max().reset_index().sort_values("rung")
        p1 = float(f["pass1"].iloc[0])
        ax.fill_between(c["k"], c["oracle"], np.interp(c["k"], best["rung"], best["acc"]),
                        color="#D55E00", alpha=0.16, lw=0, label="Unrealised headroom")
        ax.plot(c["k"], c["oracle"], color="#000000", ls="--", lw=1.1, label="Oracle pass@$k$")
        ax.plot(best["rung"], best["acc"], color="#0072B2", lw=1.5,
                label="Best configuration at that budget")
        ax.axhline(p1, color="#999999", lw=0.9, ls=":", label="Single sample (pass@1 readout)")
        st.grid(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 25]); ax.set_xticklabels(["1", "2", "4", "8", "16", "25"])
        ax.minorticks_off()
        ax.set_title(st.PANEL_TITLE[panel])
        if panel == "bbh_logic4":
            ax.set_ylim(0.95, 1.005)
        if pi % 3 == 0:
            ax.set_ylabel("Accuracy")
        if pi >= 2:
            ax.set_xlabel("Total compute $C/A_0$")
    ax = axes[5]; ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    ax.legend(h, l, loc="center", fontsize=7, handlelength=1.8)
    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F6 communication cost
def communication(contrasts, outdir, name="f6_communication_cost"):
    st.style()
    levels = ["math500", "mmlu_pro", "bbh_logic4", "mbpp_plus", "macro3"]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.3), sharey=True)
    for pi, panel in enumerate(levels):
        ax = axes[pi]
        d = contrasts[contrasts["level"] == panel].copy()
        d["n"] = d["a"].str.split("/").str[1].astype(int)
        d["r"] = d["a"].str.split("/").str[2].astype(int)
        for n, g in d.groupby("n"):
            g = g.sort_values("r")
            ax.plot(g["r"], g["estimate"] * 100, marker="o", ms=3, lw=1.0, label=f"$B$={n}")
        ax.axhline(0, color="#000000", lw=0.7)
        st.grid(ax)
        short = {"bbh_logic4": "BBH-logic4 (sat.)", "macro3": "Macro (3 bench.)"}
        ax.set_title(short.get(panel, st.PANEL_TITLE[panel]), fontsize=7.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Peer rounds $r$")
        if pi == 0:
            ax.set_ylabel("Peer minus private team\n(accuracy points)")
            ax.legend(fontsize=6, ncol=2, handlelength=1.2)
    fig.subplots_adjust(wspace=0.18)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F7 round dynamics
def _slots(traj: str) -> int:
    return 1 if traj == "private_single" else int(traj.split("/")[1])


def dynamics(trans, outdir, partition="main", name="f7_round_dynamics"):
    """Churn per revision round: answers flipping into and out of correctness.

    Bars above the axis are wrong-to-correct flips, bars below are correct-to-wrong. The
    diamond is the net. Substantial churn with a net near zero is why extra rounds buy nothing.
    """
    st.style()
    t = trans[(trans["partition_id"] == partition) & (trans["level"] == "benchmark")].copy()
    t = t[t["trajectory"].str.startswith(("peer/", "private_team/"))]
    t["slots"] = t["trajectory"].map(_slots)
    t["kind"] = np.where(t["trajectory"].str.startswith("peer"), "peer", "private_team")
    t["up"] = t["W_to_C"] / t["slots"]
    t["down"] = t["C_to_W"] / t["slots"]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    w = 0.36
    for bi, bench in enumerate(ic.BENCH_ORDER):
        ax = axes[bi]
        g = t[t["benchmark"] == bench]
        for ki, kind in enumerate(["private_team", "peer"]):
            k = g[g["kind"] == kind].groupby("to_step")[["up", "down"]].mean().reset_index()
            k = k[k["to_step"] <= 4].sort_values("to_step")
            x = k["to_step"] + (ki - 0.5) * w
            col = ic.FAMILY_COLOR[kind]
            ax.bar(x, k["up"] * 100, width=w, color=col, alpha=0.85, linewidth=0,
                   label=ic.FAMILY_LABEL[kind] if bi == 0 else None)
            ax.bar(x, -k["down"] * 100, width=w, color=col, alpha=0.38, linewidth=0)
            ax.plot(x, (k["up"] - k["down"]) * 100, ls="none", marker="D", ms=2.6,
                    color="#000000", zorder=4,
                    label="Net" if (bi == 0 and ki == 0) else None)
        ax.axhline(0, color="#000000", lw=0.7)
        st.grid(ax)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Round")
        ax.set_title(st.PANEL_TITLE[bench], fontsize=7.5)
        if bi == 0:
            ax.set_ylabel("Flips per agent slot\n(percentage points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(wspace=0.34)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F8 depth in one call
def deep_vs_split(frame, outdir, name="f8_depth_one_call"):
    st.style()
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.2), sharex=True)
    ms = [1, 2, 4, 8, 16]
    for pi, panel in enumerate(st.PANELS):
        ax = axes[pi]
        f = frame[(frame["level"] == panel) & (frame["rung"].isin(ms))]
        for fam, lab in [("deep", "Deep: 1 call, $A=C$"),
                         ("private_single", "Private single: $R=C/A_0$ calls"),
                         ("sample", "Sample+vote: $B=C/A_0$ calls")]:
            g = f[f["family"] == fam].sort_values("rung")
            ax.plot(g["rung"], g["acc"], color=ic.FAMILY_COLOR[fam], marker="o", ms=3,
                    label=lab)
        st.grid(ax)
        ax.set_xscale("log", base=2); ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms]); ax.minorticks_off()
        short = {"bbh_logic4": "BBH-logic4 (sat.)", "macro3": "Macro (3 bench.)"}
        ax.set_title(short.get(panel, st.PANEL_TITLE[panel]), fontsize=7.5)
        ax.set_xlabel("Total compute $C/A_0$")
        if panel == "bbh_logic4":
            ax.set_ylim(0.95, 1.005)
        if pi == 0:
            ax.set_ylabel("Delivered accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.22))
    fig.subplots_adjust(wspace=0.32)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F9 dollar frontier
def dollar_frontier(costs, outdir, partition="main", name="f9_dollar_frontier"):
    st.style()
    c = costs[(costs["partition_id"] == partition) & (costs["level"] == "benchmark")].copy()
    c = c[c["family"].isin(ic.LADDER_FAMILIES)]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
    for bi, bench in enumerate(ic.BENCH_ORDER):
        ax = axes[bi]
        g = c[c["benchmark"] == bench]
        for fam, gg in g.groupby("family"):
            ax.scatter(gg["mean_total_charged_usd"], gg["acc"], s=9,
                       color=ic.FAMILY_COLOR[fam], label=ic.FAMILY_LABEL[fam],
                       linewidths=0, alpha=0.85)
        # Descriptive Pareto front over the evaluated configurations.
        gs = g.sort_values("mean_total_charged_usd")
        bestacc, fx, fy = -1, [], []
        for _, r in gs.iterrows():
            if r["acc"] > bestacc:
                bestacc = r["acc"]; fx.append(r["mean_total_charged_usd"]); fy.append(r["acc"])
        ax.step(fx, fy, where="post", color="#000000", lw=0.9, alpha=0.75, zorder=1)
        st.grid(ax)
        ax.set_xscale("log")
        ax.set_title(st.PANEL_TITLE[bench], fontsize=7.5)
        ax.set_xlabel("USD per item (log)")
        if bench == "bbh_logic4":
            ax.set_ylim(0.95, 1.005)
        if bi == 0:
            ax.set_ylabel("Delivered accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(wspace=0.34)
    return st.save(fig, name, outdir)
