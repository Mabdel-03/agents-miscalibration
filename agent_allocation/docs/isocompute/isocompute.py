#!/usr/bin/env python
"""Regenerate every figure, table and quoted number for the iso-compute allocation report.

Reads only the sealed tables of the finished run; writes figures/, tables/ and numbers.json.
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import isocore as ic
import isoframe
import fig_design, fig_ladder, fig_matrix, fig_rest, fig_robust

DEFAULT_RUN = ("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/"
               "openrouter_agent_allocation_v1/artifacts/openrouter_agent_allocation_v1")
PARTS = ["main", "high", "replication"]
PEER_PAIRS = [(f"peer/{n}/{r}", f"private_team/{n}/{r}") for n in [2, 3, 4, 5] for r in [1, 2, 3, 4]]


def fmt(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("RUN_DIR", DEFAULT_RUN))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--self-check", action="store_true", help="run the accounting checks only")
    args = ap.parse_args()
    RUN, OUT = args.run_dir, args.out
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/cache", exist_ok=True)

    # ---------------------------------------------------------------- checks
    fails = [f"{p}: {m}" for p in PARTS for m in ic.self_check(RUN, p)]
    if fails:
        print("SELF-CHECK FAILED"); [print("  ", f) for f in fails]; raise SystemExit(1)
    print("self-check: C = B*R*A matches the sealed requested_reasoning_tokens on every panel")
    if args.self_check:
        return

    frames, curves = {}, {}
    for p in PARTS:
        frames[p], _ = isoframe.build(RUN, p)
        curves[p] = isoframe.readout_curve(RUN, p)
        frames[p].to_parquet(f"{OUT}/cache/frame_{p}.parquet")
        curves[p].to_parquet(f"{OUT}/cache/curve_{p}.parquet")
    fm, cm = frames["main"], curves["main"]

    # Reproduce published numbers before drawing anything.
    checks = {
        "sample/1": 0.835, "private_team/5/4": 0.8875, "peer/4/4": 0.855, "deep/8": 0.885,
    }
    macro4 = fm[fm["level"] == "macro4"].set_index("name")["acc"]
    for n, want in checks.items():
        got = float(macro4.loc[n])
        assert abs(got - want) < 5e-4, f"{n}: recomputed {got:.4f}, published {want}"
    o25 = float(cm[(cm["level"] == "macro4") & (cm["k"] == 25)]["oracle"].iloc[0])
    assert abs(o25 - 0.9375) < 5e-4, f"oracle pass@25 macro {o25}"
    print("published-number check: sample/1, private_team/5/4, peer/4/4, deep/8, pass@25 all match")

    # Design identity: n agents with zero exposure is n independent samples.
    oc = ic.outcomes(RUN, "main")
    piv = oc.pivot_table(index="item_id", columns="name", values="y", aggfunc="first")
    ident = {n: int((piv[f"private_team/{n}/0"] == piv[f"sample/{n}"]).sum()) for n in [2, 3, 4, 5]}
    ident_peer = {n: int((piv[f"peer/{n}/0"] == piv[f"sample/{n}"]).sum()) for n in [2, 3, 4, 5]}
    n_items = len(piv)
    assert all(v == n_items for v in {**ident, **ident_peer}.values()), "zero-round identity broken"
    print(f"identity check: private_team/n/0 = peer/n/0 = sample/n on {n_items}/{n_items} items")

    # ---------------------------------------------------------------- figures
    idx = ic.checkpoint_index(oc)
    contrasts = isoframe.paired_contrasts(RUN, PEER_PAIRS, "main")
    contrasts.to_parquet(f"{OUT}/cache/peer_vs_pt_main.parquet")
    trans = ic.load(RUN, "transitions_summary")
    costs = ic.load(RUN, "cost_summary")
    prim = ic.load(RUN, "primary_contrasts")

    fig_design.draw(idx, OUT)
    fig_ladder.draw(fm, cm, OUT)
    fig_matrix.draw(fm, OUT, "delta", "f3a_amplification_delta")
    fig_matrix.draw(fm, OUT, "eta", "f3b_ceiling_capture")
    fig_rest.width_depth(fm, OUT)
    fig_rest.selection_gap(fm, cm, OUT)
    fig_rest.communication(contrasts, OUT)
    fig_rest.dynamics(trans, OUT)
    fig_rest.deep_vs_split(fm, OUT)
    fig_rest.dollar_frontier(costs, OUT)
    fig_robust.draw(frames, curves, OUT)
    print("figures: 11 written")

    # ---------------------------------------------------------------- numbers
    N: dict = {}
    N["run"] = dict(run_id="openrouter_agent_allocation_v1", items_main=n_items,
                    generations=67514, checkpoints=101088, spend_usd=47.64,
                    ceiling_usd=135, audit_pass=29, audit_fail=0,
                    model="google/gemini-2.5-flash-lite",
                    replication_model="google/gemini-2.5-flash",
                    a0_main=1024, a0_high=4096, bank=25, replicates=ic.REPLICATES)
    N["identity_items"] = n_items

    for lv in ["macro4", "macro3", "math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"]:
        f = fm[fm["level"] == lv]
        c = cm[cm["level"] == lv]
        acc = f.set_index("name")["acc"]
        best = f.loc[f["acc"].idxmax()]
        N[lv] = dict(
            pass1=fmt(f["pass1"].iloc[0]),
            vote2=fmt(acc.get("sample/2")), vote25=fmt(acc.get("sample/25")),
            oracle2=fmt(float(c[c["k"] == 2]["oracle"].iloc[0])),
            oracle25=fmt(float(c[c["k"] == 25]["oracle"].iloc[0])),
            gap25=fmt(float(c[c["k"] == 25]["oracle"].iloc[0]) - acc.get("sample/25")),
            best_name=str(best["name"]), best_acc=fmt(best["acc"]), best_rung=int(best["rung"]),
            best_at_25=fmt(float(f[f["rung"] == 25]["acc"].max())),
            headroom_best25=fmt(float(c[c["k"] == 25]["oracle"].iloc[0])
                                - float(f[f["rung"] == 25]["acc"].max())),
            deep8=fmt(acc.get("deep/8")), deep16=fmt(acc.get("deep/16")),
            ps25=fmt(acc.get("private_single/25")), pt54=fmt(acc.get("private_team/5/4")),
            peer54=fmt(acc.get("peer/5/4")),
            n_tests=int(f["delta_p"].notna().sum()),
            n_bh_sig=int(f["delta_sig"].sum()),
            n_nominal_sig=int(((f["delta_lo"] > 0) | (f["delta_hi"] < 0)).sum()),
            max_delta=fmt(f["delta"].max()), min_delta=fmt(f["delta"].min()),
            min_p=fmt(f["delta_p"].min(), 4),
        )

    # Ceiling capture at the low and high ends of the ladder.
    for lv in ["macro3", "math500", "mmlu_pro", "mbpp_plus"]:
        f = fm[fm["level"] == lv]
        N[lv]["eta_lo_rung"] = fmt(f[f["rung"].isin([2, 3])]["eta"].median())
        N[lv]["eta_hi_rung"] = fmt(f[f["rung"] >= 16]["eta"].median())

    # Mean amplification by allocation width, which the width-versus-depth section quotes.
    N["width_depth"] = {}
    for lv in ["macro3", "math500", "mmlu_pro", "mbpp_plus"]:
        f = fm[(fm["level"] == lv) & (fm["rung"].isin([4, 6, 8, 10, 12, 15, 20, 25]))]
        d = {}
        for fam in ["private_team", "peer"]:
            sub = f[f["family"].isin(["private_single", fam] if fam == "private_team" else [fam])]
            sub = sub[sub["B"] <= 5]
            d[fam] = {int(b): fmt(v * 100, 1) for b, v in
                      sub.groupby("B")["delta"].mean().items()}
        N["width_depth"][lv] = d

    # Prespecified contrasts, quoted from the sealed table.
    pr = prim[(prim["role"] == "primary")]
    N["primary"] = {}
    for _, r in pr.iterrows():
        key = ("i" if "primary_i_" in r["contrast"] else "ii") + "_" + str(r["benchmark"])
        N["primary"][key] = dict(estimate=fmt(r["estimate"], 4), lo=fmt(r["ci95_lo"], 4),
                                 hi=fmt(r["ci95_hi"], 4),
                                 p=fmt(r["p_value"], 4), p_holm=fmt(r["p_holm"], 4),
                                 n=int(r["n_items"]))
        # The frozen contrasts are quoted, never recomputed: assert we reproduce the sealed file.
        assert abs(N["primary"][key]["estimate"] - float(r["estimate"])) < 5e-5, key

    # Peer cost at the largest team and most rounds.
    N["peer_cost"] = {}
    for lv in ["macro3", "math500", "mmlu_pro", "mbpp_plus"]:
        g = contrasts[(contrasts["level"] == lv) & (contrasts["a"] == "peer/5/4")].iloc[0]
        N["peer_cost"][lv] = dict(estimate=fmt(g["estimate"]), lo=fmt(g["ci_lo"]),
                                  hi=fmt(g["ci_hi"]), p=fmt(g["p"], 4))

    # Dollars: where the frontier stops paying.
    cmn = costs[(costs["partition_id"] == "main") & (costs["level"] == "macro")]
    N["cost"] = dict(
        sample1=fmt(float(cmn[cmn["name"] == "sample/1"]["mean_total_charged_usd"].iloc[0]), 6),
        deep8=fmt(float(cmn[cmn["name"] == "deep/8"]["mean_total_charged_usd"].iloc[0]), 6),
        peer54=fmt(float(cmn[cmn["name"] == "peer/5/4"]["mean_total_charged_usd"].iloc[0]), 6),
        sample25=fmt(float(cmn[cmn["name"] == "sample/25"]["mean_total_charged_usd"].iloc[0]), 6),
    )
    N["cost"]["ratio_peer54_sample1"] = fmt(N["cost"]["peer54"] / N["cost"]["sample1"], 1)

    # Robustness panels.
    for p in ["high", "replication"]:
        f = frames[p]; f = f[f["level"] == "macro3"]
        c = curves[p]; c = c[c["level"] == "macro3"]
        kmax = int(c["k"].max())
        N[p] = dict(pass1=fmt(f["pass1"].iloc[0]),
                    oracle_max=fmt(float(c[c["k"] == kmax]["oracle"].iloc[0])),
                    best_at_max=fmt(float(f[f["rung"] == kmax]["acc"].max())),
                    kmax=kmax, n_items=int(f["n_items"].iloc[0]))
        N[p]["headroom"] = fmt(N[p]["oracle_max"] - N[p]["best_at_max"])

    with open(f"{OUT}/numbers.json", "w") as fh:
        json.dump(N, fh, indent=1, sort_keys=True)
    print(f"numbers.json: {len(N)} top-level keys")

    emit_tables(OUT, fm, cm, N)
    print("tables written")


def emit_tables(OUT, fm, cm, N):
    """LaTeX booktabs fragments. Every number here also lives in numbers.json."""
    # T1: the iso-compute rung inventory.
    rows = []
    for rung in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25]:
        g = fm[(fm["level"] == "macro3") & (fm["rung"] == rung)]
        names = sorted(set(g["name"]))
        rows.append((rung, len(names), ", ".join(n.replace("_", " ") for n in names)))
    with open(f"{OUT}/tables/rungs.tex", "w") as fh:
        fh.write("\\begin{tabular}{rrl}\n\\toprule\n")
        fh.write("$C/A_0$ & configs & evaluated allocations \\\\\n\\midrule\n")
        for r, k, s in rows:
            s = s.replace("private single", "\\texttt{ps}").replace("private team", "\\texttt{pt}")
            s = s.replace("sample", "\\texttt{s}").replace("peer", "\\texttt{peer}")
            s = s.replace("deep", "\\texttt{deep}")
            fh.write(f"{r} & {k} & \\footnotesize {s} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # T2: headline per-benchmark summary.
    order = ["math500", "mmlu_pro", "bbh_logic4", "mbpp_plus", "macro3"]
    lab = {"math500": "MATH-500", "mmlu_pro": "MMLU-Pro", "bbh_logic4": "BBH-logic4",
           "mbpp_plus": "MBPP+", "macro3": "Macro (3)"}
    with open(f"{OUT}/tables/headline.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        fh.write("Benchmark & pass@1 & vote@2 & vote@25 & best any & oracle@25 & headroom \\\\\n")
        fh.write("\\midrule\n")
        for k in order:
            n = N[k]
            fh.write(f"{lab[k]} & {n['pass1']:.3f} & {n['vote2']:.3f} & {n['vote25']:.3f} & "
                     f"{n['best_acc']:.3f} & {n['oracle25']:.3f} & {n['headroom_best25']:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # T3: the two prespecified contrasts, quoted from the sealed table.
    with open(f"{OUT}/tables/primary.tex", "w") as fh:
        fh.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        fh.write("Contrast & Level & $n$ & Estimate & 95\\% CI & Holm $p$ \\\\\n\\midrule\n")
        names = {"i": "Peer $-$ private team, $B$=4, $r$=4",
                 "ii": "CP $-$ private-team gain, $h$=3"}
        for tag in ["i", "ii"]:
            for bench in ["macro", "math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"]:
                key = f"{tag}_{bench}"
                if key not in N["primary"]:
                    continue
                d = N["primary"][key]
                blab = "Macro (4)" if bench == "macro" else lab.get(bench, bench)
                nm = names[tag] if bench == "macro" else "\\quad"
                ph = f"{d['p_holm']:.3f}" if d["p_holm"] is not None else "n/a"
                fh.write(f"{nm} & {blab} & {d['n']} & {d['estimate']:+.4f} & "
                         f"[{d['lo']:+.3f}, {d['hi']:+.3f}] & {ph} \\\\\n")
            fh.write("\\midrule\n" if tag == "i" else "")
        fh.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
