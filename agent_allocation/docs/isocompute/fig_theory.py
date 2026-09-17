"""F19: the probability model behind the width law, against simulation."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import isostyle as st
import isolaw


def draw(tc, fit, outdir, name="f19_theory_check"):
    st.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

    # (a) predicted against simulated, every item and k
    ax = axes[0]
    d = tc[tc["k"] >= 2]
    ax.scatter(d["theory"], d["obs"], s=3, alpha=0.20, linewidths=0, color="#0072B2")
    ax.plot([0, 1], [0, 1], color="#000000", lw=0.9, ls="--")
    ax.set_xlabel("$\\Phi(m\\sqrt{k}/\\sigma)$, predicted")
    ax.set_ylabel("Simulated plurality accuracy")
    ax.set_title("(a) No free parameters", fontsize=7.6)
    ax.text(0.04, 0.93, f"$r$ = {fit['corr']:.3f}\nMAE = {fit['mae']:.3f}\n"
            f"{fit['n_points']} points", transform=ax.transAxes, fontsize=6, va="top")
    st.grid(ax, axis="both")

    # (b) the family of curves the model predicts
    ax = axes[1]
    ks = np.arange(1, 26)
    for m, c in [(0.1, "#DDDDDD"), (0.2, "#BBD7E8"), (0.3, "#7FB3D5"),
                 (0.5, "#0072B2"), (0.7, "#00517F"), (-0.2, "#D55E00")]:
        s = np.sqrt(max(1.0 - m ** 2, 1e-9))
        ax.plot(ks, norm.cdf(m * np.sqrt(ks) / s), color=c, lw=1.3, label=f"$m$={m:+.1f}")
    ax.axhline(0.95, color="#000000", lw=0.7, ls=":")
    ax.text(15.5, 0.965, "95% resolved", fontsize=5.6)
    ax.set_xlabel("Samples $k$")
    ax.set_ylabel("P(vote correct)")
    ax.set_title("(b) Width only pays in a band", fontsize=7.6)
    ax.set_ylim(0, 1.02)
    st.grid(ax)
    ax.legend(fontsize=5.6, loc="center right", handlelength=1.2)

    # (c) the closed form for k*
    ax = axes[2]
    ms = np.linspace(0.08, 0.98, 200)
    ax.plot(ms, [isolaw.kstar_closed_form(m) for m in ms], color="#000000", lw=1.4,
            label="$k^* = z^2(1-m^2)/m^2$")
    ax.plot(ms, 2.706 / ms ** 2, color="#999999", lw=0.9, ls=":", label="$z^2/m^2$")
    ax.axhline(25, color="#D55E00", lw=0.9, ls="--")
    ax.text(0.45, 28, "this study's bank", fontsize=5.6, color="#D55E00")
    ax.set_yscale("log")
    ax.set_xlabel("Margin $m$")
    ax.set_ylabel("Samples to resolve, $k^*$")
    ax.set_title("(c) The cost of a narrow margin", fontsize=7.6)
    ax.set_ylim(0.8, 400)
    st.grid(ax, axis="both")
    ax.legend(fontsize=6, loc="upper right")
    fig.subplots_adjust(wspace=0.36)
    return st.save(fig, name, outdir)
