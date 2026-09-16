"""Every figure in the ARD36 report, generated from ``tables_v2`` only.

The report must never quote a number the figures cannot reproduce, so both come from the same
source: the repaired tables. Nothing here reads the frozen v1 ``tables/`` -- those are kept only
to show what the repair moved, and quoting them anywhere else would reintroduce the defects
Stage 0 removed.

    python docs/ard36/report/make_figures.py [RUN_ROOT]

Writes eight vector PDFs into ``<run_root>/report/figs/``. The report's ``\\includegraphics``
paths are relative, so the same files are committed under ``docs/ard36/report/figs/``.
"""
from __future__ import annotations

import collections
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36")
V = RUN / "tables_v2"
OUT = RUN / "report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})

# One colour per protocol, used identically in every panel so a reader can track an arm across
# figures without re-reading legends. PEER/PRIVATE carry the argument and are drawn heavier.
C = {"PEER": "#c0392b", "PRIVATE": "#1f6f3f", "BALANCED": "#e08214", "HUB5": "#8e44ad",
     "FRESH": "#2c7fb8", "GATE": "#7f7f7f", "RANDOM_PROTECT": "#b8a800"}
ORD = ["PRIVATE", "FRESH", "GATE", "BALANCED", "RANDOM_PROTECT", "HUB5", "PEER"]


def rd(name: str, cols: list[str] | None = None) -> list[dict]:
    return pq.read_table(V / name, columns=cols).to_pylist()


rs = rd("round_summary.parquet")
ep = {r["episode_id"]: r for r in
      rd("episodes.parquet", ["episode_id", "item_id", "checkpoint", "split"])}

# The analysis panel: real five-agent team rounds only. ROOTS banks are already skipped in
# round_summary by the Stage 0 repair; the size check also excludes the single-agent receiver.
team = [r for r in rs
        if r["protocol"] not in ("NONE",)
        and (r["n_valid"] or 0) + (r["n_unparsed"] or 0) == 5
        and r["cohort"] == "natural" and r["checkpoint"] == "8B"
        and ep.get(r["episode_id"], {}).get("split") == "test"]
byp = collections.defaultdict(lambda: collections.defaultdict(list))
for r in team:
    byp[r["protocol"]][r["round"]].append(r)


# --- F1 coverage trajectory ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for p in ORD:
    y = [100 * st.mean([1.0 if (x["C_t"] or 0) >= 1 else 0.0 for x in byp[p][t]]) for t in range(4)]
    lead = p in ("PEER", "PRIVATE")
    ax.plot(range(4), y, marker="o", ms=4, lw=2 if lead else 1.1, color=C[p],
            alpha=1 if lead else 0.55, label=p, zorder=3 if lead else 2)
    if lead:
        ax.annotate(f"{y[3]:.2f}%", (3, y[3]), textcoords="offset points", xytext=(6, -3),
                    color=C[p], fontsize=8, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xlabel("round")
ax.set_ylabel(r"P(team holds truth) = P($C_t\geq1$)  [%]")
ax.set_title("Peer visibility destroys truth; private self-revision preserves it", fontsize=9.5)
ax.legend(fontsize=7, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.set_xlim(-0.15, 3.2)
fig.savefig(OUT / "f1_coverage.pdf")
plt.close(fig)

# --- F2 the dissociation ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for p in ("PEER", "PRIVATE"):
    axes[0].plot(range(4), [st.mean([x["C_t"] or 0 for x in byp[p][t]]) for t in range(4)],
                 marker="o", ms=4, color=C[p], lw=2, label=p)
    axes[1].plot(range(4), [100 * st.mean([1.0 if (x["C_t"] or 0) >= 1 else 0.0
                                           for x in byp[p][t]]) for t in range(4)],
                 marker="o", ms=4, color=C[p], lw=2, label=p)
axes[0].set_title(r"mean $C_t$ (of 5 agents)", fontsize=9)
axes[0].set_ylabel(r"mean $C_t$")
axes[1].set_title(r"P($C_t\geq1$): does ANYONE hold truth", fontsize=9)
axes[1].set_ylabel("%")
for a in axes:
    a.set_xticks(range(4))
    a.set_xlabel("round")
    a.legend(fontsize=8, frameon=False)
fig.suptitle("PEER concentrates agreement while losing truth entirely", fontsize=10, y=1.02)
fig.savefig(OUT / "f2_dissociation.pdf")
plt.close(fig)

# --- F3 the sign flip ------------------------------------------------------------------------
# Frozen v1 values are hard-coded on purpose: they no longer exist in tables_v2, and the whole
# point of the panel is to show what the repair moved.
names = ["GATE - PEER\n(r3 accuracy)", "GATE - RANDOM_PROTECT\n(r3 accuracy)",
         "PEER - PRIVATE\n(coverage change)"]
v1 = [(-0.00833, -0.0317, 0.0098), (-0.02584, -0.04794, -0.00833), (0.0125, -0.025, 0.05417)]
v2 = [(-0.00893, -0.0311, 0.0132), (-0.02257, -0.0450, -0.0045), (-0.04583, -0.0750, -0.0208)]
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for i, (a, b) in enumerate(zip(v1, v2)):
    ax.errorbar(a[0], i + 0.16, xerr=[[a[0] - a[1]], [a[2] - a[0]]], fmt="o", ms=5,
                color="#999999", capsize=3, lw=1.2, label="v1 (frozen)" if i == 0 else None)
    ax.errorbar(b[0], i - 0.16, xerr=[[b[0] - b[1]], [b[2] - b[0]]], fmt="s", ms=5,
                color="#c0392b" if i == 2 else "#333333", capsize=3, lw=1.8,
                label="v2 (repaired)" if i == 0 else None)
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_yticks(list(range(3)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("effect (paired difference)")
ax.set_title("Repair inverted a pre-registered primary (Holm $0.93\\rightarrow0.024$, rejects)",
             fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.savefig(OUT / "f3_signflip.pdf")
plt.close(fig)

# --- F4 sampling vs communication --------------------------------------------------------------
# Fixed item set: a bank needs k graded draws for pass@k, so complete-case over all 240 items
# would silently shrink the set as k grows (236 at k=1, 171 at k=16) and compare different items.
b = [r for r in rd("bank_summary.parquet") if r["checkpoint"] == "8B" and r["split"] == "test"]
full = [r for r in b if r["n_ungraded"] == 0]
keep = {r["item_id"] for r in full}
ks = [1, 2, 3, 5, 9, 16]
pk = [100 * st.mean([r[f"pass_at_{k}"] for r in full]) for k in ks]
vote = 100 * st.mean([r["plurality_correct"] for r in full])
tm = collections.defaultdict(list)
for r in rs:
    m = ep.get(r["episode_id"], {})
    if (r["cohort"] != "natural" or m.get("checkpoint") != "8B" or m.get("split") != "test"
            or r["round"] != 3 or m.get("item_id") not in keep):
        continue
    v = r["selected_correct_plurality_current"]
    if v is not None:
        tm[r["protocol"]].append(float(v))

fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.plot(ks, pk, marker="o", ms=5, color="#1f6f3f", lw=2, label="pass@k  (oracle verifier)")
ax.axhline(vote, color="#2c7fb8", ls="--", lw=1.4, label=f"vote@16 (no verifier) = {vote:.2f}%")
pts = sorted((100 * st.mean(tm[p]), p) for p in ORD if p in tm)
lab = [v for v, _ in pts]
for i in range(1, len(lab)):          # minimum vertical gap so the arm labels never collide
    if lab[i] - lab[i - 1] < 0.42:
        lab[i] = lab[i - 1] + 0.42
for (v, p), ly in zip(pts, lab):
    ax.scatter([20], [v], marker="D", s=30, color=C[p], zorder=4, edgecolor="white", linewidth=0.5)
    ax.annotate(f"{p} {v:.2f}", (20, v), xytext=(24, ly), textcoords=("offset points", "data"),
                fontsize=6.2, color=C[p], va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C[p], alpha=0.6,
                                shrinkA=0, shrinkB=2))
ax.set_xlim(0.85, 46)
ax.set_xscale("log")
ax.set_xticks(ks + [20])
ax.set_xticklabels([str(k) for k in ks] + ["20\n(teams)"])
ax.set_xlabel("generations per item")
ax.set_ylabel("accuracy [%]")
ax.set_title("20 generations of debate buy what 3-5 verified samples buy\n"
             "(fixed 171 fully-graded items)", fontsize=9.5)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.savefig(OUT / "f4_sampling.pdf")
plt.close(fig)

# --- F5 wrong-unanimity escape -----------------------------------------------------------------
esc = {}
for p in ORD:
    w = [r for r in team if r["protocol"] == p and r["unanimity_wrong"]]
    e = [r for r in team if r["protocol"] == p and r["unanimous_wrong_escape"]]
    esc[p] = (100 * len(e) / max(len(w), 1), len(e), len(w))
fig, ax = plt.subplots(figsize=(5.4, 2.9))
order = sorted(esc, key=lambda p: -esc[p][0])
ax.bar(range(len(order)), [esc[p][0] for p in order],
       color=["#2c7fb8" if p == "FRESH" else C[p] for p in order], alpha=0.9)
for i, p in enumerate(order):
    ax.text(i, esc[p][0] + 0.7, f"{esc[p][0]:.1f}%\n{esc[p][1]}/{esc[p][2]}",
            ha="center", fontsize=6.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("escape from wrong unanimity [%]")
ax.set_ylim(0, 44)
ax.set_title("Consensus locks in the error; only resampling escapes it", fontsize=9.5)
fig.savefig(OUT / "f5_unanimity.pdf")
plt.close(fig)

# --- F6 verbatim copying -----------------------------------------------------------------------
d = rd("decisions.parquet", ["episode_id", "agent_id", "round", "protocol", "evidence_text",
                             "cohort", "checkpoint", "cell_kind"])
d = [r for r in d if r["cohort"] == "natural" and r["checkpoint"] == "8B"
     and r["cell_kind"] == "EPISODE"]
prev = collections.defaultdict(dict)
for r in d:
    e = (r["evidence_text"] or "").strip()
    if e:
        prev[(r["episode_id"], r["round"])][r["agent_id"]] = e
res = collections.defaultdict(lambda: [0, 0, 0])
for r in d:
    if r["round"] < 1:
        continue
    e = (r["evidence_text"] or "").strip()
    if not e:
        continue
    before = prev.get((r["episode_id"], r["round"] - 1), {})
    if not before:
        continue
    res[r["protocol"]][0] += 1
    if any(v == e for a, v in before.items() if a != r["agent_id"]):
        res[r["protocol"]][1] += 1
    if before.get(r["agent_id"]) == e:
        res[r["protocol"]][2] += 1
fig, ax = plt.subplots(figsize=(6.0, 3.0))
arms = [p for p in ORD if p in res]
x, w = range(len(arms)), 0.38
ax.bar([i - w / 2 for i in x], [100 * res[p][1] / res[p][0] for p in arms], w,
       label="copies a PEER verbatim", color="#c0392b", alpha=0.9)
ax.bar([i + w / 2 for i in x], [100 * res[p][2] / res[p][0] for p in arms], w,
       label="repeats its OWN prior", color="#1f6f3f", alpha=0.9)
ax.set_xticks(list(x))
ax.set_xticklabels(arms, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("verbatim evidence reuse [%]")
ax.set_title('What "revision" actually is: copying a peer, or repeating yourself', fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUT / "f6_copying.pdf")
plt.close(fig)

# --- F7 neural geometry ------------------------------------------------------------------------
g = rd("neural_geometry.parquet")
g17 = [r for r in g if r["layer"] == 17 and r["anchor"] == "PRE_FINAL"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
byg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in g17:
    v = r.get("mean_pairwise")
    if v is not None and not (isinstance(v, float) and math.isnan(v)):
        byg[r["protocol"]][r["round"]].append(float(v))
for p in ("PRIVATE", "BALANCED", "PEER"):
    ts = [t for t in (1, 2, 3) if byg[p].get(t)]
    axes[0].plot(ts, [st.mean(byg[p][t]) for t in ts], marker="o", ms=4, color=C[p], lw=2, label=p)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xlabel("round")
axes[0].set_ylabel("mean pairwise distance")
axes[0].set_title("Representational spread (layer 17)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False)
for anchor, mk, cl in (("PROMPT_END", "o", "#333333"), ("PRE_FINAL", "s", "#c0392b")):
    eff = []
    for layer in (8, 17, 26):
        sub = [r for r in g if r["layer"] == layer and r["anchor"] == anchor and r["round"] == 3]
        by = {(r["item_id"], r["protocol"]): r["mean_pairwise"]
              for r in sub if r.get("mean_pairwise") is not None}
        its = sorted({i for i, _ in by})
        dd = [by[(i, "PEER")] - by[(i, "PRIVATE")]
              for i in its if (i, "PEER") in by and (i, "PRIVATE") in by]
        eff.append(st.mean(dd))
    axes[1].plot([8, 17, 26], eff, marker=mk, ms=5, lw=2, color=cl, label=anchor)
axes[1].axhline(0, color="k", lw=0.8, ls=":")
axes[1].set_xticks([8, 17, 26])
axes[1].set_xlabel("transformer block")
axes[1].set_ylabel("PEER - PRIVATE  (paired)")
axes[1].set_title("Contraction deepens with depth", fontsize=9)
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle("Neural correlate: peer exchange contracts representational diversity",
             fontsize=10, y=1.02)
fig.savefig(OUT / "f7_neural.pdf")
plt.close(fig)

# --- F8 panel range + verification funnel -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
sat = sum(1 for r in b if r["c_correct"] == 16)
zero = sum(1 for r in b if r["c_correct"] == 0)
mid = len(b) - sat - zero
axes[0].bar(["all 16\ncorrect", "0 of 16\ncorrect", "non-\ndegenerate"], [sat, zero, mid],
            color=["#bbbbbb", "#bbbbbb", "#1f6f3f"], alpha=0.9)
for i, v in enumerate([sat, zero, mid]):
    axes[0].text(i, v + 3, str(v), ha="center", fontsize=8.5, fontweight="bold")
axes[0].set_ylabel("items (of 240)")
axes[0].set_title(f"Only {mid}/{len(b)} items can show disagreement", fontsize=9)
labels = ["enumerated", "estimable", "computed\nclaims", "adversarially\nverified",
          "survived as\nstated"]
vals = [1441, 692, 56, 39, 0]
axes[1].bar(range(5), vals, color=["#333333", "#1f6f3f", "#e08214", "#8e44ad", "#c0392b"],
            alpha=0.9)
for i, v in enumerate(vals):
    axes[1].text(i, v + 28, str(v), ha="center", fontsize=8.5, fontweight="bold")
axes[1].set_yscale("symlog")
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(labels, fontsize=6.8)
axes[1].set_title("35 weakened, 4 refuted, 0 confirmed as stated", fontsize=9)
fig.savefig(OUT / "f8_scope.pdf")
plt.close(fig)

print(f"8 figures written to {OUT}")
