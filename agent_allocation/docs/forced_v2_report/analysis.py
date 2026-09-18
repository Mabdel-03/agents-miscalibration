"""Driver: every number, table and figure of the forced_budget_v2 report, from the merged sealed tables.

    analysis.py --run <merged run dir> [--v1 <v1 artifacts dir>] [--calibration <calibration.json>] [--out DIR]

Writes ``numbers.json`` (every number quoted in report.tex), ``tables/*.tex`` and ``figures/*.pdf``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import core as c
import figures as fg

RUNGS = [1, 2, 4, 8, 16, 25]


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def pts(x):                      # accuracy points, 1 decimal
    return None if x is None or np.isnan(x) else round(100 * float(x), 1)


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--v1", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--replicates", type=int, default=c.REPLICATES)
    a = ap.parse_args()
    c.REPLICATES = a.replicates
    out = a.out
    os.makedirs(f"{out}/figures", exist_ok=True)
    os.makedirs(f"{out}/tables", exist_ok=True)
    N: dict = {}

    # ------------------------------------------------------------ run facts
    F = c.Frame(a.run, c.LADDER_FAMILIES, label="ladder")
    oc = F.oc
    N["n_items"] = F.n_items
    N["n_items_by_bench"] = {b: int(oc[oc["benchmark"] == b]["item_id"].nunique()) for b in c.BENCH_ORDER}
    N["n_checkpoints_ladder"] = len(F.names)
    cand = c.T(a.run, "candidates")
    main = cand[cand["partition_id"] == "main"]
    N["n_generations_main"] = int(len(main))
    N["n_generations_all"] = int(len(cand))
    fbc = c.T(a.run, "forced_budget_check") if os.path.exists(f"{a.run}/tables/forced_budget_check.parquet") else None
    if fbc is not None:
        N["forced_check_rows"] = int(len(fbc))
        flag = [col for col in fbc.columns if "mismatch" in col or "flag" in col or col == "exact"]
        N["forced_check_columns"] = list(fbc.columns)
    rt = main["reasoning_tokens"].astype(float)
    bud = main["reasoning_budget"].astype(float)
    N["forced_exact_share"] = r3(float((rt == bud).mean()))
    N["forced_not_exact"] = int((rt != bud).sum())
    N["grade_unknown_main"] = int((main["grade_state"] == "unknown_evaluator_error").sum())
    N["grade_unknown_share"] = r3(N["grade_unknown_main"] / len(main))
    N["invalid_output_main"] = int((main["grade_status"] == "invalid_output").sum())
    N["invalid_output_share"] = r3(N["invalid_output_main"] / len(main))
    N["parse_status_main"] = {k: int(v) for k, v in main["parse_status"].value_counts().items()}
    N["checkpoint_rows"] = int(len(oc))
    N["checkpoint_rows_unknown"] = int(oc["y"].isna().sum())
    N["unknown_items"] = sorted(oc[oc["y"].isna()]["item_id"].unique().tolist())
    N["shard_items"] = {s: int(g["item_id"].nunique()) for s, g in oc.groupby("shard_id")} if "shard_id" in oc else {}
    cs = c.T(a.run, "cost_summary")
    cr = c.T(a.run, "cost_reconciliation") if os.path.exists(f"{a.run}/tables/cost_reconciliation.parquet") else None
    N["answer_tokens_median"] = int(np.median((main["completion_tokens"] - main["reasoning_tokens"]).dropna()))
    N["spend_main_usd"] = round(float(pd.to_numeric(main["actual_cost_usd"], errors="coerce").sum()), 2)
    N["spend_all_usd"] = round(float(pd.to_numeric(cand["actual_cost_usd"], errors="coerce").sum()), 2)

    if cr is not None:
        q = {(r["section"], r["quantity"]): r["value"] for _, r in cr.iterrows()}
        def qf(sec, name):
            v = q.get((sec, name))
            return None if v is None else round(float(v), 2)
        N["ledger"] = {"settled_usd": qf("ledger", "settled_usd"), "charged_max_usd": qf("ledger", "charged_max_usd"),
                       "charged_total_usd": qf("ledger", "charged_total_usd"),
                       "invariant_violations": int(float(q.get(("ledger", "invariant_violations"), 0))),
                       "main_settled_usd": qf("ledger:main", "settled"),
                       "pilot_settled_usd": qf("ledger:pilot", "settled"),
                       "canary_settled_usd": qf("ledger:canary", "settled"),
                       "calibration_settled_usd": round((qf("ledger:calibration", "settled") or 0)
                                                        + (qf("ledger:calibration_natural", "settled") or 0), 2)}
        key = [(sec, k, v) for (sec, k), v in q.items() if "provider_usage_delta" in k]
        if key:
            N["ledger"]["key_delta_usd"] = round(float(key[0][2]), 2)
    br = c.T(a.run, "billed_retries")
    brm = br[br["partition_id"] == "main"]
    N["retries"] = {"nodes": int(len(brm)), "attempts": int(brm["retried_attempts"].sum()),
                    "usd": round(float(pd.to_numeric(brm["retry_charges_usd"], errors="coerce").sum()), 2)}

    # ------------------------------------------------------------ headline table: accuracies
    acc = {}
    for level in ["macro", *c.BENCH_ORDER]:
        acc[level] = {n: dict(zip(("acc", "lo", "hi"), map(r3, F.acc(n, level)))) for n in F.names}
    N["acc"] = acc
    bank = c.bank_curve(a.run)
    N["oracle"] = {lv: {int(k): r3(v) for k, v in g.set_index("k")["oracle_pass_at_k"].items()}
                   for lv, g in bank.groupby("level")}
    N["readout"] = {lv: {int(k): r3(v) for k, v in g.set_index("k")["readout"].items()}
                    for lv, g in bank.groupby("level")}
    N["coverage"] = {lv: {int(k): r3(v) for k, v in g.set_index("k")["prefix_coverage"].items()}
                     for lv, g in bank.groupby("level")}

    # best configuration per level and family
    best = {}
    for level in ["macro", *c.BENCH_ORDER]:
        rows = [(acc[level][n]["acc"], n) for n in F.names]
        rows.sort(reverse=True)
        best[level] = {"name": rows[0][1], "acc": rows[0][0],
                       "best_non_deep": max((x for x in rows if not x[1].startswith("deep/")))}
    N["best"] = {lv: {"name": v["name"], "acc": v["acc"], "non_deep_name": v["best_non_deep"][1],
                      "non_deep_acc": v["best_non_deep"][0]} for lv, v in best.items()}

    # ------------------------------------------------------------ iso-compute frame (delta vs sample at same rung)
    rows = []
    for level in ["macro", *c.BENCH_ORDER]:
        for _, r in F.idx.iterrows():
            name, rung = r["name"], int(r["rung"])
            base = f"sample/{rung}"
            if name == base or base not in F.col:
                continue
            est, lo, hi, p = F.diff(name, base, level)
            rows.append(dict(level=level, name=name, family=r["family"], rung=rung, B=int(r["B"]), R=int(r["R"]),
                             delta=est, lo=lo, hi=hi, p=p))
    iso = pd.DataFrame(rows)
    iso["sig"] = False
    for level, g in iso.groupby("level"):
        iso.loc[g.index, "sig"] = c.bh(g["p"].to_numpy())
    iso.to_csv(f"{out}/tables/iso_frame.csv", index=False)
    N["iso"] = {}
    for level, g in iso.groupby("level"):
        fam = {}
        for f_, gg in g.groupby("family"):
            fam[f_] = {"n": int(len(gg)), "n_sig_pos": int(((gg["sig"]) & (gg["delta"] > 0)).sum()),
                       "n_sig_neg": int(((gg["sig"]) & (gg["delta"] < 0)).sum()),
                       "mean_delta_pts": pts(gg["delta"].mean()), "max_delta_pts": pts(gg["delta"].max()),
                       "min_delta_pts": pts(gg["delta"].min())}
        N["iso"][level] = {"n_tests": int(len(g)), "n_sig": int(g["sig"].sum()),
                           "n_sig_pos": int(((g["sig"]) & (g["delta"] > 0)).sum()),
                           "n_sig_neg": int(((g["sig"]) & (g["delta"] < 0)).sum()), "by_family": fam}

    # ------------------------------------------------------------ depth vs width at matched realized compute
    dvw = {}
    for level in ["macro", *c.BENCH_ORDER]:
        dvw[level] = {}
        for m in RUNGS[1:]:
            e = {}
            for tag, x, y in (("deep_minus_sample", f"deep/{m}", f"sample/{m}"),
                              ("deep_minus_private", f"deep/{m}", f"private_single/{m}"),
                              ("private_minus_sample", f"private_single/{m}", f"sample/{m}")):
                if x in F.col and y in F.col:
                    est, lo, hi, p = F.diff(x, y, level)
                    e[tag] = {"pts": pts(est), "lo": pts(lo), "hi": pts(hi), "p": r3(p)}
            dvw[level][m] = e
    N["depth_vs_width"] = dvw
    # deep against the sampling ceiling (oracle pass@k at the same k)
    N["deep_vs_oracle"] = {lv: {m: pts(acc[lv][f"deep/{m}"]["acc"] - N["oracle"][lv][m])
                                for m in RUNGS[1:] if f"deep/{m}" in F.col}
                           for lv in ["macro", *c.BENCH_ORDER]}
    # cross-budget contrasts: less depth against more width
    N["cross"] = {}
    for tag, x, y in (("deep25_minus_deep16", "deep/25", "deep/16"), ("deep4_minus_sample25", "deep/4", "sample/25"),
                      ("deep8_minus_private25", "deep/8", "private_single/25"),
                      ("deep16_minus_peer54", "deep/16", "peer/5/4"),
                      ("deep16_minus_pteam54", "deep/16", "private_team/5/4")):
        N["cross"][tag] = {}
        for level in ["macro", *c.BENCH_ORDER]:
            est, lo, hi, p = F.diff(x, y, level)
            N["cross"][tag][level] = {"pts": pts(est), "lo": pts(lo), "hi": pts(hi), "p": r3(p)}
    # best allocation at each rung
    N["best_at_rung"] = {}
    for level in ["macro", *c.BENCH_ORDER]:
        d = {}
        for rung, g in F.idx.groupby("rung"):
            names = list(g["name"])
            top = max(names, key=lambda n: acc[level][n]["acc"])
            d[int(rung)] = {"name": top, "acc": acc[level][top]["acc"], "n_configs": len(names)}
        N["best_at_rung"][level] = d

    # ------------------------------------------------------------ communication (peer vs private team)
    comm = []
    for n in (2, 3, 4, 5):
        for r in (1, 2, 3, 4):
            p_, q_ = f"peer/{n}/{r}", f"private_team/{n}/{r}"
            if p_ in F.col and q_ in F.col:
                for level in ["macro", *c.BENCH_ORDER]:
                    est, lo, hi, p = F.diff(p_, q_, level)
                    comm.append(dict(level=level, n=n, r=r, delta=est, lo=lo, hi=hi, p=p))
    comm = pd.DataFrame(comm)
    comm.to_csv(f"{out}/tables/communication.csv", index=False)
    cm = comm[comm["level"] == "macro"]
    N["comm"] = {"mean_pts": pts(cm["delta"].mean()), "n_pos": int((cm["delta"] > 0).sum()),
                 "n_cells": int(len(cm)), "max_pts": pts(cm["delta"].max()), "min_pts": pts(cm["delta"].min()),
                 "cell_4_4": {k: pts(v) if k != "p" else r3(v)
                              for k, v in cm[(cm.n == 4) & (cm.r == 4)][["delta", "lo", "hi", "p"]].iloc[0].items()},
                 "cell_5_4": {k: pts(v) if k != "p" else r3(v)
                              for k, v in cm[(cm.n == 5) & (cm.r == 4)][["delta", "lo", "hi", "p"]].iloc[0].items()}}
    pc = c.T(a.run, "primary_contrasts")
    pc = pc[pc["level"] == "macro"]
    N["primary"] = {}
    for _, r in pc.iterrows():
        N["primary"][r["contrast"]] = {"estimate_pts": pts(r["estimate"]), "lo_pts": pts(r["ci95_lo"]),
                                       "hi_pts": pts(r["ci95_hi"]), "p_holm": r3(r["p_holm"]),
                                       "bound_lo_pts": pts(r["bound_lower"]), "bound_hi_pts": pts(r["bound_upper"]),
                                       "n": int(r["n_items"]), "n_complete": int(r["n_complete"]),
                                       "status": r["status"], "role": r["role"], "description": r["description"]}
    # known-items-only paired interval for the frozen peer contrast (the sealed table gives bounds only when partial)
    e, lo, hi, p = F.diff("peer/4/4", "private_team/4/4", "macro")
    N["primary_peer_known"] = {"pts": pts(e), "lo": pts(lo), "hi": pts(hi), "p": r3(p)}

    # ------------------------------------------------------------ selection gap
    N["selection_gap_pts"] = {lv: {k: pts(N["oracle"][lv][k] - N["readout"][lv][k]) for k in (1, 2, 4, 8, 16, 25)}
                              for lv in N["oracle"]}

    # ------------------------------------------------------------ effective depth
    eds = c.T(a.run, "effective_depth_summary")
    eds = eds[eds["partition_id"] == "main"]
    edn = c.T(a.run, "effective_depth_nodes")
    edn = edn[edn["partition_id"] == "main"]
    deep_nodes = edn[edn["kind"] == "deep"]
    eff = {}
    for m in RUNGS[1:]:
        g = deep_nodes[deep_nodes["reasoning_budget"] == m * c.A0]
        if len(g):
            eff[m] = {"n": int(len(g)), "looped_share": r3(g["looped"].mean()),
                      "eff_tokens_mean": int(g["effective_tokens"].mean()),
                      "eff_fraction_mean": r3((g["effective_tokens"] / g["reasoning_budget"]).mean()),
                      "onset_tokens_median": (int(np.nanmedian(g["loop_onset_fraction"] * g["reasoning_budget"]))
                                              if g["looped"].any() else None)}
    N["effective_depth"] = eff
    nondeep = edn[edn["kind"] != "deep"]
    N["looped_share_1k_calls"] = r3(nondeep["looped"].mean())
    # accuracy of deep/m split by whether that deep call looped
    dl = deep_nodes.merge(oc[oc["family"] == "deep"][["item_id", "name", "y"]]
                          .assign(reasoning_budget=lambda d: d["name"].str.split("/").str[1].astype(int) * c.A0),
                          on=["item_id", "reasoning_budget"], how="left")
    N["deep_acc_by_loop"] = {int(m // c.A0): {"looped": r3(g[g["looped"]]["y"].mean()) if g["looped"].any() else None,
                                              "clean": r3(g[~g["looped"]]["y"].mean())}
                             for m, g in dl.groupby("reasoning_budget") if m > 2 * c.A0}

    # ------------------------------------------------------------ dollars
    csm = cs[(cs["partition_id"] == "main") & (cs["benchmark"] == "macro")][["name", "family", "mean_total_charged_usd",
                                                                            "mean_reasoning_tokens"]]
    csm = csm[csm["family"].isin(c.LADDER_FAMILIES)].copy()
    csm["acc"] = csm["name"].map(lambda n: acc["macro"][n]["acc"] if n in acc["macro"] else np.nan)
    csm = csm.dropna(subset=["acc"]).sort_values("mean_total_charged_usd")
    front, best_acc = [], -1
    for _, r in csm.iterrows():
        if r["acc"] > best_acc:
            front.append(r["name"]); best_acc = r["acc"]
    N["frontier"] = front
    N["cost_usd"] = {r["name"]: round(float(r["mean_total_charged_usd"]), 4) for _, r in csm.iterrows()}
    csm.to_csv(f"{out}/tables/dollars.csv", index=False)

    # ------------------------------------------------------------ provider robustness (shard covariate)
    rob = {}
    if "shard_id" in oc:
        for sid, g in oc.groupby("shard_id"):
            d = {}
            for m in (8, 16, 25):
                x = g[g["name"] == f"deep/{m}"].set_index("item_id")["y"]
                y = g[g["name"] == f"sample/{m}"].set_index("item_id")["y"]
                j = pd.concat([x, y], axis=1, keys=["d", "s"]).dropna()
                d[m] = {"n": int(len(j)), "deep": r3(j["d"].mean()), "sample": r3(j["s"].mean()),
                        "delta_pts": pts(j["d"].mean() - j["s"].mean())}
            s1 = g[g["name"] == "sample/1"]["y"].mean()
            rob[sid] = {"items": int(g["item_id"].nunique()), "pass1": r3(s1), "deep_vs_sample": d}
    N["robustness"] = rob

    # ------------------------------------------------------------ calibration (natural reference; dev items)
    if a.calibration and os.path.exists(a.calibration):
        cal = json.load(open(a.calibration))
        N["calibration"] = {b: {k: (r3(v) if isinstance(v, float) else v)
                                for k, v in d.items() if k in ("forced_pass_at_1", "natural_pass_at_1",
                                                               "natural_reasoning_tokens_median",
                                                               "natural_reasoning_tokens_mean", "items")}
                            for b, d in cal["benchmarks"].items()}
        N["calibration_share_at_or_above_deep"] = {b: d.get("natural_share_at_or_above_deep_budget")
                                                   for b, d in cal["benchmarks"].items()}

    # ------------------------------------------------------------ v1 comparison (deep realized vs requested)
    if a.v1:
        v1cs = pd.read_parquet(f"{a.v1}/tables/cost_summary.parquet")
        v1cs = v1cs[(v1cs["partition_id"] == "main") & (v1cs["benchmark"] == "macro")]
        v1 = {}
        for m in (1, 2, 4, 8, 16):
            dn = "deep/1" if m == 1 else f"deep/{m}"
            r = v1cs[v1cs["name"] == dn]
            s = v1cs[v1cs["name"] == f"sample/{m}"]
            if len(r):
                v1[m] = {"requested": int(r["mean_requested_reasoning_tokens"].iloc[0]),
                         "realized": int(r["mean_reasoning_tokens"].iloc[0]), "acc": r3(r["acc"].iloc[0]),
                         "sample_acc": r3(s["acc"].iloc[0]) if len(s) else None}
        N["v1_deep"] = v1
        v2cs = cs[(cs["partition_id"] == "main") & (cs["benchmark"] == "macro")]
        N["v2_deep_realized"] = {m: int(v2cs[v2cs["name"] == f"deep/{m}"]["mean_reasoning_tokens"].iloc[0])
                                 for m in RUNGS if len(v2cs[v2cs["name"] == f"deep/{m}"])}

    # ------------------------------------------------------------ generation vs selection (gap.py)
    import gap
    import gfig
    G, rung_tab, fam_tab, vs, prt, cal, strat = gap.run(os.path.dirname(a.run.rstrip("/")), N, out, F)
    N["gap"] = G
    # proposed follow-up: k independent calls at depth m with k*m = C, for C = 16A and 32A, reusing the
    # one deep call per depth already collected; cost from this run's measured per-item cost of each depth
    cu = N["cost_usd"]
    per = {m: cu[f"deep/{m}"] for m in (2, 4, 8, 16)}
    new_calls = {16: {2: 7, 4: 3, 8: 1}, 32: {2: 15, 4: 7, 8: 3, 16: 1}}
    item_cost = sum(k * per[m] for C in new_calls for m, k in new_calls[C].items())
    N["followup"] = {"per_item_usd": round(item_cost, 3), "total_usd_96": round(96 * item_cost, 2),
                     "new_calls_per_item": sum(sum(d.values()) for d in new_calls.values()),
                     "tokens_per_item": sum(k * m * c.A0 for C in new_calls for m, k in new_calls[C].items())}
    write_gap_tables(N, G, rung_tab, vs, prt, out)

    c.dump(N, f"{out}/numbers.json")

    # ------------------------------------------------------------ tables
    write_tables(N, F, iso, comm, out)
    # ------------------------------------------------------------ figures
    fg.style()
    fg.f1_design(F.idx, out)
    fg.f2_ladder(N, F, bank, out)
    fg.f3_depth_vs_width(N, out)
    fg.f4_requested_vs_realized(N, out)
    fg.f5_matrix(iso, out)
    fg.f6_selection(N, bank, out)
    fg.f7_communication(comm, out)
    fg.f8_effective_depth(N, deep_nodes, out)
    fg.f9_dollars(csm, N, out)
    fg.f10_robustness(N, out)
    gfig.g1_plane(rung_tab, out)
    gfig.g2_decomposition(fam_tab, out)
    gfig.g3_verifier(vs, N, out)
    gfig.g4_strata(G, out)
    gfig.g5_vote(G, out)
    gfig.g6_portfolios(prt, N, out)
    gfig.g7_transitions(G, out)
    gfig.g8_slopes(G, out)
    gfig.g9_team_shape(G, out)
    print(json.dumps({k: N[k] for k in ("n_items", "forced_exact_share", "spend_all_usd")}, indent=1))


def write_tables(N, F, iso, comm, out):
    L = N["acc"]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Benchmark & pass@1 & vote@25 & oracle@25 & priv.\ 25 & best team & deep/16 & deep/25 \\",
             r"\midrule"]
    for lv in [*c.BENCH_ORDER, "macro"]:
        teams = [n for n in F.names if n.startswith(("peer/", "private_team/"))]
        bt = max(teams, key=lambda n: L[lv][n]["acc"])
        lines.append(f"{c.BENCH_LABEL[lv]} & {fmt(L[lv]['sample/1']['acc'])} & {fmt(L[lv]['sample/25']['acc'])} & "
                     f"{fmt(N['oracle'][lv][25])} & {fmt(L[lv]['private_single/25']['acc'])} & "
                     f"{fmt(L[lv][bt]['acc'])} & {fmt(L[lv]['deep/16']['acc'])} & {fmt(L[lv]['deep/25']['acc'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/headline.tex", "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"$m$ & contrast & AIME & HMMT & Omni & GPQA \\ \midrule"]
    for m in RUNGS[1:]:
        for tag, lab in (("deep_minus_sample", "deep $-$ sample"), ("deep_minus_private", "deep $-$ private"),
                         ("private_minus_sample", "private $-$ sample")):
            cells = []
            for b in c.BENCH_ORDER:
                e = N["depth_vs_width"][b][m].get(tag)
                cells.append("--" if e is None else f"{e['pts']:+.1f}")
            lines.append(f"{m} & {lab} & " + " & ".join(cells) + r" \\")
        if m != RUNGS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/depth_by_bench.tex", "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$m$ & macro deep $-$ sample & 95\% CI & deep $-$ private & 95\% CI & private $-$ sample \\ \midrule"]
    for m in RUNGS[1:]:
        e = N["depth_vs_width"]["macro"][m]
        ds, dp, ps = e["deep_minus_sample"], e["deep_minus_private"], e["private_minus_sample"]
        lines.append(f"{m} & {ds['pts']:+.1f} & [{ds['lo']:+.1f}, {ds['hi']:+.1f}] & {dp['pts']:+.1f} & "
                     f"[{dp['lo']:+.1f}, {dp['hi']:+.1f}] & {ps['pts']:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/depth_macro.tex", "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Contrast (sealed, frozen before collection) & estimate (pts) & bounds (pts) & complete & status \\ \midrule"]
    labels = {"primary_i_peer_vs_private_team_n4_r4": r"(i) peer/4/4 $-$ private\_team/4/4",
              "primary_ii_cp_minus_private_team_gain_h3": r"(ii) CP gain $-$ private-team gain, $h$=3"}
    for k, v in N["primary"].items():
        if k not in labels:
            continue
        lines.append(f"{labels[k]} & {v['estimate_pts']:+.1f} & [{v['bound_lo_pts']:+.1f}, {v['bound_hi_pts']:+.1f}] & "
                     f"{v['n_complete']}/{v['n']} & {v['status']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/primary.tex", "w").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"deep/$m$ & reasoning tokens & calls looping & mean effective tokens & effective share & "
             r"loop onset (median) \\ \midrule"]
    for m, e in N["effective_depth"].items():
        on = "--" if e["onset_tokens_median"] is None else f"{e['onset_tokens_median']:,}"
        lines.append(f"{m} & {m * c.A0:,} & {e['looped_share']:.3f} & {e['eff_tokens_mean']:,} & "
                     f"{e['eff_fraction_mean']:.3f} & {on} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/effective_depth.tex", "w").write("\n".join(lines) + "\n")
    # typographic minus for every negative number in the generated tables
    import re
    for name in ("depth_by_bench", "depth_macro", "primary"):
        path = f"{out}/tables/{name}.tex"
        text = re.sub(r"(?<![\w$])-(\d)", r"$-$\1", open(path).read())
        open(path, "w").write(text)


def _f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _s(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{d}f}".replace("-", "$-$")


def write_gap_tables(N, G, rung_tab, vs, prt, out):
    fam_order = {"sample": 0, "private_single": 1, "private_team": 2, "peer": 3, "deep": 4, "cp": 5, "pooled": 6}
    # compact: per rung, macro
    L = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$C/A$ & pass@$k$ & vote@$k$ & \multicolumn{2}{c}{private chain} & \multicolumn{2}{c}{best team} & deep & best non-deep \\",
         r" & & & delivered & coverage & delivered & coverage & & \\ \midrule"]
    for r in (2, 4, 8, 16, 25):
        t = rung_tab[(rung_tab.level == "macro") & (rung_tab.rung == r)]
        ps = t[t.family == "private_single"]
        tm = t[t.family.isin(["peer", "private_team"])]
        bt = tm.sort_values("delivered").iloc[-1] if len(tm) else None
        dp = t[t.family == "deep"]
        L.append(f"{r} & {_f(t['pass_at_k'].iloc[0])} & {_f(t[t.family == 'sample']['delivered'].max())} & "
                 f"{_f(ps['delivered'].max())} & {_f(ps['cov_archive'].max())} & "
                 f"{_f(None if bt is None else bt['delivered'])} & {_f(None if bt is None else bt['cov_archive'])} & "
                 f"{_f(dp['delivered'].max())} & {_f(t[t.family != 'deep']['delivered'].max())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/rung_compact.tex", "w").write("\n".join(L) + "\n")

    # full, macro, every configuration at the five depth rungs (appendix)
    L = [r"\begin{longtable}{rlrrrrrr}", r"\toprule",
         r"$C/A$ & configuration & $|$pool$|$ & delivered & cov.\ (current) & cov.\ (all) & gap (all) & $-$ pass@$k$ \\",
         r"\midrule", r"\endhead"]
    for r in (2, 4, 8, 16, 25):
        t = rung_tab[(rung_tab.level == "macro") & (rung_tab.rung == r)].copy()
        t["o"] = t["family"].map(fam_order)
        t = t.sort_values(["o", "B", "name"])
        L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{$C={r}A$: pass@{r} = {_f(t['pass_at_k'].iloc[0])}, "
                 f"prefix coverage = {_f(t['prefix_cov'].iloc[0])}}}}} \\\\")
        for _, x in t.iterrows():
            L.append(f"{r} & \\texttt{{{x['name'].replace('_', chr(92) + '_')}}} & {x['n_pool_archive']:.0f} & "
                     f"{_f(x['delivered'])} & {_f(x['cov_current'])} & {_f(x['cov_archive'])} & "
                     f"{_s(100 * x['gap_archive'])} & {_s(100 * x['vs_pass_at_k'])} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{longtable}"]
    open(f"{out}/tables/rung_full_macro.tex", "w").write("\n".join(L) + "\n")

    # per benchmark compact rung table (pass@k, vote, chain, best team, deep)
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Benchmark & $C/A$ & pass@$k$ & vote@$k$ & private chain & best team & deep \\ \midrule"]
    for b in c.BENCH_ORDER:
        for r in (4, 8, 16, 25):
            e = G["rung"][b][r]
            L.append(f"{c.BENCH_LABEL[b] if r == 4 else ''} & {r} & {_f(e['pass_at_k'])} & {_f(e['vote'])} & "
                     f"{_f(e['private_single'])} & {_f(e['teams_best'])} & {_f(e['deep'])} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/rung_by_bench.tex", "w").write("\n".join(L) + "\n")

    # difficulty strata
    cfgs = ["sample/25", "private_single/25", "peer/5/4", "private_team/5/4", "deep/4", "deep/8", "deep/16", "deep/25"]
    L = [r"\begin{tabular}{lr" + "r" * len(cfgs) + "}", r"\toprule",
         "Samples correct & items & " + " & ".join(
             {"sample/25": "vote 25", "private_single/25": "chain 25", "peer/5/4": "peer 5/4",
              "private_team/5/4": "team 5/4"}.get(x, x) for x in cfgs) + r" \\ \midrule"]
    for st in G["strata"]:
        L.append(f"{st['stratum']} & {st['n']} & " + " & ".join(_f(st[x]) for x in cfgs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/strata.tex", "w").write("\n".join(L) + "\n")

    # portfolios
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Deep calls (multiples of $A$) & $C/A$ & coverage & deepest & vote & pass@$C$ & vote@$C$ \\ \midrule"]
    for _, x in prt.sort_values("C").iterrows():
        L.append(f"{x['portfolio']} & {x['C']} & {_f(x['cov'])} & {_f(x['deepest'])} & {_f(x['vote'])} & "
                 f"{_f(x['pass_at_C'])} & {_f(x['vote_at_C'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/portfolios.tex", "w").write("\n".join(L) + "\n")

    # verifier: macro accuracy by AUC at C=16 and C=25
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"$C/A$ & pool & AUC 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 0.95 & 1.0 \\ \midrule"]
    for r in (4, 8, 16, 25):
        for fam, lab in (("sample", f"{r} samples"), ("private_single", f"chain of {r}, all answers"),
                         ("team", "best team, all answers")):
            g = vs[(vs.rung == r) & (vs.family == fam)].sort_values("auc")
            if len(g):
                L.append(f"{r} & {lab} & " + " & ".join(_f(v) for v in g["macro"]) + r" \\")
        L.append(f"{r} & \\emph{{deep/{r}, one call}} & \\multicolumn{{7}}{{c}}{{{_f(N['acc']['macro'][f'deep/{r}']['acc'])}}} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{out}/tables/verifier.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
