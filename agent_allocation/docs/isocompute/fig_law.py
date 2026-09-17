"""F17-F18: the item-state law, and the axis it prescribes."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import isocore as ic
import isostyle as st
import isolaw

BENCH_MARK = {"math500": "o", "mmlu_pro": "s", "mbpp_plus": "^"}


# ---------------------------------------------------------------- F17
def law(state, kstar, fit, outdir, name="f17_state_law"):
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # (a) the state plane
    ax = axes[0]
    s = state[state["benchmark"].isin(ic.INFORMATIVE)]
    ax.plot([0, 1], [0, 1], color="#000000", lw=0.9, ls="--", zorder=2)
    ax.text(0.52, 0.42, "$m=0$", fontsize=6, rotation=45, color="#333333")
    for reg in isolaw.REGIMES:
        g = s[s["regime"] == reg]
        ax.scatter(g["p"], g["q1"], s=7, color=isolaw.REGIME_COLOR[reg], linewidths=0,
                   alpha=0.8, label=reg.capitalize(), zorder=3)
    ax.set_xlabel("$p$ (per-draw correct)")
    ax.set_ylabel("$q_1$ (largest wrong share)")
    ax.set_title("(a) Item state", fontsize=7.6)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    st.grid(ax, axis="both")
    ax.legend(fontsize=5.6, loc="upper right", handletextpad=0.2)

    # (b) regime mix per benchmark
    ax = axes[1]
    mix = (s.groupby("benchmark")["regime"].value_counts(normalize=True)
           .unstack(fill_value=0).reindex(columns=isolaw.REGIMES)
           .reindex(["math500", "mmlu_pro", "mbpp_plus"]))
    bot = np.zeros(len(mix))
    for reg in isolaw.REGIMES:
        ax.bar(range(len(mix)), mix[reg], bottom=bot, width=0.55,
               color=isolaw.REGIME_COLOR[reg], linewidth=0, label=reg.capitalize())
        for i, v in enumerate(mix[reg]):
            if v > 0.04:
                ax.text(i, bot[i] + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if reg == "dead" else "#222222")
        bot = bot + mix[reg].to_numpy()
    ax.set_xticks(range(len(mix)))
    ax.set_xticklabels([st.PANEL_TITLE[b] for b in mix.index], fontsize=6, rotation=12)
    ax.set_ylabel("Share of items")
    ax.set_title("(b) Regime mix is a benchmark property", fontsize=7.6)
    ax.legend(fontsize=5.6, loc="lower left")

    # (c) the width law
    ax = axes[2]
    ax.scatter(kstar["m"], kstar["kstar"], s=6, color="#0072B2", alpha=0.45, linewidths=0)
    xs = np.linspace(max(kstar["m"].min(), 0.06), 1.0, 100)
    ax.plot(xs, fit["coef"] * xs ** fit["exponent"], color="#000000", lw=1.2,
            label=f"$k^* = {fit['coef']:.2f}\\,m^{{{fit['exponent']:.2f}}}$")
    ax.plot(xs, xs ** -2.0, color="#999999", lw=0.9, ls=":", label="$m^{-2}$ (binomial)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.set_yticks([1, 2, 5, 10, 25])
    ax.set_yticklabels(["1", "2", "5", "10", "25"])
    ax.minorticks_off()
    ax.set_xlabel("Margin $m$")
    ax.set_ylabel("Samples to resolve, $k^*$")
    ax.set_title("(c) What width costs", fontsize=7.6)
    st.grid(ax, axis="both")
    ax.legend(fontsize=6, loc="upper right")
    fig.subplots_adjust(wspace=0.36)
    return st.save(fig, name, outdir)


# ---------------------------------------------------------------- F18
def prescribe(axis, obs, pred_auc, outdir, name="f18_axis_prescription"):
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # (a) width against depth at matched compute, by regime
    ax = axes[0]
    a = axis[axis["C"] == 8].set_index("regime").reindex(isolaw.REGIMES)
    x = np.arange(len(a))
    ax.bar(x - 0.19, a["width"], width=0.36, color="#5A5A5A", linewidth=0, label="Width (sample/8)")
    ax.bar(x + 0.19, a["depth"], width=0.36, color="#CC79A7", linewidth=0, label="Depth (deep/8)")
    for i, r in enumerate(a.itertuples()):
        ax.annotate(f"{r.diff:+.2f}", xy=(i, max(r.width, r.depth) + 0.04),
                    ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in a.index], fontsize=6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy at $C=8A_0$")
    ax.set_title("(a) The axis depends on the regime", fontsize=7.6)
    st.grid(ax)
    ax.legend(fontsize=6, loc="upper right")

    # (b) predicted against observed
    ax = axes[1]
    BC = {"math500": "#0072B2", "mmlu_pro": "#009E73", "mbpp_plus": "#D55E00"}
    for _, r in obs.iterrows():
        ax.errorbar(r["predicted_gain"], r["observed"],
                    yerr=[[r["observed"] - r["lo"]], [r["hi"] - r["observed"]]],
                    fmt=BENCH_MARK.get(r["benchmark"], "o"), ms=4.5, lw=0.9, capsize=2,
                    color=BC.get(r["benchmark"], "#0072B2"),
                    label=st.PANEL_TITLE[r["benchmark"]])
    lim = [-0.05, 0.09]
    ax.plot(lim, lim, color="#000000", lw=0.8, ls="--")
    ax.axhline(0, color="#CCCCCC", lw=0.6); ax.axvline(0, color="#CCCCCC", lw=0.6)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Predicted gain from depth")
    ax.set_ylabel("Observed gain")
    ax.set_title("(b) The mix predicts the payoff", fontsize=7.6)
    st.grid(ax, axis="both")
    ax.legend(fontsize=5.8, loc="upper left")

    # (c) can a cheap pilot tell you the regime
    ax = axes[2]
    p = pred_auc[pred_auc["scope"] != "pooled"].set_index("scope")
    cols = ["margin", "n_distinct", "entropy", "mean_chars"]
    lbl = ["vote margin", "distinct answers", "entropy", "response length"]
    x = np.arange(len(cols))
    w = 0.26
    for i, b in enumerate(["math500", "mmlu_pro", "mbpp_plus"]):
        if b not in p.index:
            continue
        ax.bar(x + (i - 1) * w, [p.loc[b, c] for c in cols], width=w, linewidth=0,
               color=["#0072B2", "#009E73", "#D55E00"][i], label=st.PANEL_TITLE[b])
    ax.axhline(0.5, color="#000000", lw=0.8, ls="--")
    ax.text(len(cols) - 0.6, 0.515, "chance", fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=5.6, rotation=20)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC, predicting not-decided")
    ax.set_title("(c) But the regime is not observable", fontsize=7.6)
    st.grid(ax)
    ax.legend(fontsize=5.6, loc="upper left")
    fig.subplots_adjust(wspace=0.38)
    return st.save(fig, name, outdir)
