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
import fig_design, fig_ladder, fig_matrix, fig_rest, fig_robust, fig_gap, fig_select, fig_law, fig_theory
import isogap, isoselect, isolaw

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

    # ---- where the gap sits, and what would collapse it
    dec = isogap.decomposition(RUN)
    resp = isogap.selector_response(RUN)
    buck = isogap.buckets(RUN)
    sweep = isogap.verifier_sweep(RUN)
    aa = isogap.allowance_axis(RUN)
    arc = isogap.archive_gain(RUN)
    conc = isogap.concentration(RUN)
    for nm, df in [("decomposition", dec), ("selector_response", resp), ("buckets", buck),
                   ("verifier_sweep", sweep), ("allowance_axis", aa), ("archive_gain", arc),
                   ("concentration", conc)]:
        df.to_parquet(f"{OUT}/cache/{nm}.parquet")
    best_alloc = float(fm[fm["level"] == "macro3"]["acc"].max())
    fig_gap.anatomy(resp, buck, OUT)
    fig_gap.split(dec, OUT)
    fig_gap.verifier(sweep, best_alloc, OUT)
    fig_gap.allowance(aa, buck, OUT)

    # ---- does allocation change selection, and what would read a pool out better
    ss = isoselect.selection_success(RUN)
    sconc = isoselect.success_by_concentration(RUN)
    matched, matched_ci = isoselect.matched_pool(RUN)
    struct = isoselect.answer_structure(RUN, cache=f"{OUT}/cache/bank_answers.pkl")
    topm = isoselect.topm_curve(struct)
    selv = isoselect.selective_verification(struct)
    tci = isoselect.topm_ci(struct, 2)
    for nm, df in [("selection_success", ss), ("success_by_conc", sconc),
                   ("matched_pool", matched), ("matched_pool_ci", matched_ci),
                   ("answer_structure", struct), ("topm", topm), ("selective", selv),
                   ("topm_ci", tci)]:
        df.to_parquet(f"{OUT}/cache/{nm}.parquet")
    fig_select.tradeoff(ss, sconc, matched, OUT)
    fig_select.adjudication(struct, topm, selv, OUT)

    # ---- the conditional scaling law
    BANK = f"{OUT}/cache/bank_answers.pkl"
    lstate = isolaw.item_state(RUN, "main", BANK)
    vcurve = isolaw.vote_curve(RUN, "main", BANK)
    kstar, wfit = isolaw.width_law(vcurve)
    axis = isolaw.axis_by_regime(RUN, lstate)
    presc = isolaw.prescription(lstate, axis, 8)
    hstate = isolaw.item_state(RUN, "high", f"{OUT}/cache/bank_answers_high.pkl")
    trans_j, trans_x = isolaw.allowance_transition(lstate, hstate)
    predf, pauc = isolaw.predictability(RUN, lstate, cache=BANK)
    yv = po_y = ic.load(RUN, "policy_outcomes")
    po_y = po_y[(po_y["partition_id"] == "main") & (po_y["view"] == "current")]
    yv = po_y.pivot_table(index="item_id", columns="name", values="y", aggfunc="first")
    dj = lstate.set_index("item_id").join(yv)
    rng2 = np.random.default_rng(ic.STUDY_SEED)
    orows = []
    for b, g in dj[dj["benchmark"].isin(ic.INFORMATIVE)].groupby("benchmark"):
        df = (g["deep/8"] - g["sample/8"]).to_numpy(float)
        bs = df[rng2.integers(0, len(df), (10000, len(df)))].mean(axis=1)
        orows.append(dict(benchmark=b, observed=float(df.mean()),
                          lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975))))
    obs = pd.DataFrame(orows).merge(presc[["benchmark", "predicted_gain"]], on="benchmark")
    for nm, df in [("item_state", lstate), ("vote_curve", vcurve), ("kstar", kstar),
                   ("axis_by_regime", axis), ("prescription", presc),
                   ("predictability", pauc), ("pred_obs", obs)]:
        df.to_parquet(f"{OUT}/cache/law_{nm}.parquet")
    tcheck = isolaw.theory_check(RUN, "main", BANK)
    tfit = isolaw.theory_fit(tcheck)
    tcheck.to_parquet(f"{OUT}/cache/law_theory_check.parquet")
    fig_law.law(lstate, kstar, wfit, OUT)
    fig_law.prescribe(axis, obs, pauc, OUT)
    fig_theory.draw(tcheck, tfit, OUT)
    print("figures: 20 written")

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

    # ---------------------------------------------------------------- prescription numbers
    P: dict = {}
    d25 = dec[(dec["level"] == "macro3") & (dec["rung"] == 25)].set_index("name")
    P["decomp25"] = {n: dict(cov_sample=fmt(r["cov_sample"]), cov_own=fmt(r["cov_own"]),
                             cov_arc=fmt(r["cov_arc"]), y=fmt(r["y"]), y_arc=fmt(r["y_arc"]),
                             cov_deficit=fmt(r["cov_deficit"]), sel_loss=fmt(r["sel_loss"]))
                     for n, r in d25.iterrows()}
    for bench in ["math500", "mmlu_pro", "mbpp_plus"]:
        b1 = buck[(buck["benchmark"] == bench) & (buck["k"] == 1)].iloc[0]
        b25 = buck[(buck["benchmark"] == bench) & (buck["k"] == 25)].iloc[0]
        P.setdefault("buckets", {})[bench] = dict(
            none1=fmt(b1["no_correct"]), none25=fmt(b25["no_correct"]),
            minority1=fmt(b1["minority"]), minority25=fmt(b25["minority"]),
            majority1=fmt(b1["majority"]), majority25=fmt(b25["majority"]),
            gained=fmt(b1["no_correct"] - b25["no_correct"]),
            to_minority=fmt(b25["minority"] - b1["minority"]),
            to_majority=fmt(b25["majority"] - b1["majority"]))
        r = resp[resp["benchmark"] == bench].set_index("bin")["deliver"]
        P.setdefault("selector", {})[bench] = {str(k): fmt(v) for k, v in r.items()}
    mac = sweep[sweep["benchmark"] == "macro3"].set_index("auc")["delivered"]
    P["verifier"] = {f"auc{int(a*100)}": fmt(v) for a, v in mac.items()}
    P["verifier"]["plurality"] = fmt(float(sweep[sweep["benchmark"] == "macro3"]["plurality"].iloc[0]))
    P["verifier"]["oracle"] = fmt(float(sweep[sweep["benchmark"] == "macro3"]["oracle"].iloc[0]))
    P["verifier"]["best_alloc"] = fmt(best_alloc)
    P["verifier"]["equiv_auc_plurality"] = fmt(isogap.equivalent_auc(sweep, "macro3",
                                                                    P["verifier"]["plurality"]), 2)
    P["verifier"]["auc_to_beat_best"] = fmt(isogap.equivalent_auc(sweep, "macro3", best_alloc), 2)
    # Derived gains the prescription quotes directly.
    P["verifier"]["gain_auc80_over_plurality"] = fmt(P["verifier"]["auc80"] - P["verifier"]["plurality"])
    P["verifier"]["gain_auc80_over_best_alloc"] = fmt(P["verifier"]["auc80"] - P["verifier"]["best_alloc"])
    P["verifier"]["gain_auc70_over_plurality"] = fmt(P["verifier"]["auc70"] - P["verifier"]["plurality"])
    P["allowance"] = {f"{r['panel']}_{r['benchmark']}":
                      dict(draw_acc=fmt(r["draw_acc"]), gap=fmt(r["gap"]),
                           minority=fmt(r["minority"]), oracle=fmt(r["oracle"]),
                           readout=fmt(r["readout"]))
                      for _, r in aa.iterrows()}
    a3 = arc[arc["level"] == "macro3"].set_index("name")
    P["archive"] = {n: dict(estimate=fmt(r["estimate"]), lo=fmt(r["ci_lo"]), hi=fmt(r["ci_hi"]))
                    for n, r in a3.iterrows()
                    if n in ("peer/5/4", "peer/4/4", "private_team/5/4", "private_single/25",
                             "sample/25")}
    c3 = conc[conc["rung"] == 25].set_index("family")["conc"]
    P["concentration25"] = {k: fmt(v) for k, v in c3.items()}
    for lv in ["math500", "mmlu_pro", "bbh_logic4", "mbpp_plus", "macro3"]:
        g = dec[dec["level"] == lv]
        N[lv]["best_arc"] = fmt(g["y_arc"].max())
        N[lv]["best_arc_name"] = str(g.loc[g["y_arc"].idxmax(), "name"])
    mb = buck[buck["benchmark"] == "mbpp_plus"].set_index("k")
    P["mbpp_selector"] = {f"k{k}": dict(readout=fmt(mb.loc[k, "readout"]),
                                        random=fmt(mb.loc[k, "random_draw"]))
                          for k in [1, 4, 8, 16, 25] if k in mb.index}
    # Sizing for the proposed validation run (Appendix A).
    gm = fm[(fm["level"] == "macro3") & fm["delta_lo"].notna()]
    se400 = float(((gm["delta_hi"] - gm["delta_lo"]) / (2 * 1.96)).median())
    pcm = ic.load(RUN, "policy_costs")
    pcm = pcm[(pcm["partition_id"] == "main") & (pcm["view"] == "current")
              & (pcm["name"] == "sample/25")]
    per_cand = int(pcm["completion_tokens"].mean() / 25)
    n_calls = 400 * 25
    tin = n_calls * (600 + per_cand) / 1e6
    tout = n_calls * 128 / 1e6
    P["followup"] = dict(
        se400=fmt(se400, 4), mde400=fmt(2.8 * se400, 4),
        mde192=fmt(2.8 * se400 * np.sqrt(400 / 192), 4),
        predicted_effect=P["verifier"]["gain_auc80_over_plurality"],
        judge_calls=n_calls, tokens_in_m=fmt(tin, 1), tokens_out_m=fmt(tout, 2),
        cost_lite=fmt(tin * 0.10 + tout * 0.40, 2),
        cost_flash=fmt(tin * 0.30 + tout * 2.50, 2),
        per_candidate_tokens=per_cand)
    # ---- selection-side numbers
    Q: dict = {}
    mp = matched.set_index("name")
    Q["matched_pool"] = {n: dict(n_pool=fmt(r["n_pool"], 1), coverage=fmt(r["coverage"]),
                                 conc=fmt(r["conc"]), y=fmt(r["y"]), S=fmt(r["S"]))
                         for n, r in mp.iterrows()}
    mc = matched_ci[matched_ci["level"] == "macro3"].set_index("name")
    Q["matched_ci"] = {n: dict(estimate=fmt(r["estimate"]), lo=fmt(r["ci_lo"]), hi=fmt(r["ci_hi"]))
                       for n, r in mc.iterrows()}
    for fam in ["sample", "private_team", "peer"]:
        g = ss[ss["family"] == fam]
        lo = g[g["rung"] <= 4]["S"].mean()
        hi = g[g["rung"] >= 20]["S"].mean()
        Q.setdefault("S_by_family", {})[fam] = dict(low_budget=fmt(lo), high_budget=fmt(hi))
    sstr = struct.groupby("benchmark")[["n_distinct", "top_share", "margin"]].mean()
    Q["answer_space"] = {b: dict(n_distinct=fmt(r["n_distinct"], 1),
                                 top_share=fmt(r["top_share"]), margin=fmt(r["margin"]))
                         for b, r in sstr.iterrows()}
    Q["topm"] = {}
    for b, g in topm.groupby("benchmark"):
        gg = g.set_index("m")["acc"]
        Q["topm"][b] = {f"top{m}": fmt(gg.loc[m]) for m in isoselect.TOP_M}
        Q["topm"][b]["plurality"] = fmt(float(g["plurality"].iloc[0]))
        Q["topm"][b]["oracle"] = fmt(float(g["oracle"].iloc[0]))
    Q["topm_ci"] = {r["benchmark"]: dict(estimate=fmt(r["estimate"]), lo=fmt(r["ci_lo"]),
                                         hi=fmt(r["ci_hi"])) for _, r in tci.iterrows()}
    sv = selv[(selv["benchmark"] == "mmlu_pro")].set_index("tau")
    Q["selective_mmlu"] = {f"tau{int(t*100)}": dict(verified=fmt(sv.loc[t, "verified"]),
                                                    recovered=fmt(sv.loc[t, "recovered"]))
                           for t in [0.10, 0.50, 1.01] if t in sv.index}
    # Rank of the correct answer on the items where plurality fails but a correct answer exists.
    fail = struct[(struct["has_correct"]) & (~struct["plurality_ok"])]
    Q["failing_items"] = {b: dict(n=int(len(g)), mean_rank=fmt(g["correct_rank"].mean(), 2),
                                  rank2_share=fmt((g["correct_rank"] == 2).mean()))
                          for b, g in fail.groupby("benchmark") if len(g)}
    # ---- conditional scaling law numbers
    L: dict = {}
    L["width_law"] = dict(coef=fmt(wfit["coef"], 2), exponent=fmt(wfit["exponent"], 2),
                          n=wfit["n"], corr=fmt(wfit["corr"], 3))
    L["kstar"] = {f"m{int(m*10)}": fmt(wfit["coef"] * m ** wfit["exponent"], 1)
                  for m in [0.2, 0.3, 0.5, 0.7, 0.9]}
    L["theory"] = dict(corr=fmt(tfit["corr"], 3), mae=fmt(tfit["mae"], 3),
                       bias=fmt(abs(tfit["bias"]), 3), n_points=tfit["n_points"],
                       n_items=tfit["n_items"], step_vote=fmt(tfit["step_agreement"], 3))
    L["kstar_closed"] = {f"m{int(m*10)}": fmt(isolaw.kstar_closed_form(m), 1)
                         for m in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}
    L["regime_mix"] = {r["benchmark"]: dict(decided=fmt(r["decided"]), outvoted=fmt(r["outvoted"]),
                                            dead=fmt(r["dead"]),
                                            predicted_gain=fmt(r["predicted_gain"]))
                       for _, r in presc.iterrows()}
    L["axis"] = {f"C{int(r['C'])}_{r['regime']}": dict(width=fmt(r["width"]), depth=fmt(r["depth"]),
                                                       diff=fmt(r["diff"]), lo=fmt(r["ci_lo"]),
                                                       hi=fmt(r["ci_hi"]), n=int(r["n"]))
                 for _, r in axis.iterrows()}
    L["pred_obs"] = {r["benchmark"]: dict(predicted=fmt(r["predicted_gain"]),
                                          observed=fmt(r["observed"]), lo=fmt(r["lo"]),
                                          hi=fmt(r["hi"]),
                                          error=fmt(abs(r["observed"] - r["predicted_gain"])))
                     for _, r in obs.iterrows()}
    L["pred_obs"]["macro3_predicted"] = fmt(float(obs["predicted_gain"].mean()))
    L["pred_obs"]["macro3_observed"] = fmt(float(obs["observed"].mean()))
    L["predictability"] = {r["scope"]: {c: fmt(r[c], 2) for c in
                                        ["margin", "n_distinct", "entropy", "mean_chars",
                                         "mean_reason"]}
                           for _, r in pauc.iterrows()}
    for _, r in pauc.iterrows():
        L["predictability"][r["scope"]]["base"] = fmt(r["base"])
        L["predictability"][r["scope"]]["n"] = int(r["n"])
    tx = trans_x.reindex(index=isolaw.REGIMES, columns=isolaw.REGIMES).fillna(0).astype(int)
    L["allowance_transition"] = {f"{a}_to_{b}": int(tx.loc[a, b])
                                 for a in isolaw.REGIMES for b in isolaw.REGIMES}
    ov = trans_j[trans_j["regime_lo"] == "outvoted"]
    L["allowance_transition"]["outvoted_n"] = int(len(ov))
    L["allowance_transition"]["outvoted_rescued"] = fmt((ov["regime_hi"] == "decided").mean())
    # Does the step law hold: is vote@25 the sign of the margin?
    sbm = ic.load(RUN, "sample_bank_items")
    sbm = sbm[(sbm["partition_id"] == "main") & (sbm["k"] == 25)].set_index("item_id")
    ls = lstate.set_index("item_id")
    L["step_law_agreement"] = fmt(float(((ls["m"] > 0).astype(float)
                                         == sbm["readout"].reindex(ls.index)).mean()), 3)
    N["law"] = L

    N["selection"] = Q

    N["prescription"] = P

    with open(f"{OUT}/numbers.json", "w") as fh:
        json.dump(N, fh, indent=1, sort_keys=True)
    print(f"numbers.json: {len(N)} top-level keys")

    emit_tables(OUT, fm, cm, N, dec)
    print("tables written")


def emit_tables(OUT, fm, cm, N, dec=None):
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
        fh.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        fh.write("Benchmark & pass@1 & vote@2 & vote@25 & best any & best archive & "
                 "oracle@25 & headroom \\\\\n")
        fh.write("\\midrule\n")
        for k in order:
            n = N[k]
            fh.write(f"{lab[k]} & {n['pass1']:.3f} & {n['vote2']:.3f} & {n['vote25']:.3f} & "
                     f"{n['best_acc']:.3f} & {n['best_arc']:.3f} & "
                     f"{n['oracle25']:.3f} & {n['headroom_best25']:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # T4: the budget-matched gap decomposition at the top of the ladder.
    if dec is not None:
        d = dec[(dec["level"] == "macro3") & (dec["rung"] == 25)
                & dec["family"].isin(["sample", "private_single", "private_team", "peer"])]
        d = d.sort_values("y", ascending=False)
        with open(f"{OUT}/tables/gap_split.tex", "w") as fh:
            fh.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
            fh.write("Allocation & Pool coverage & Delivered & Coverage deficit & "
                     "Selection loss & Archive \\\\\n\\midrule\n")
            for _, r in d.iterrows():
                nm = str(r["name"]).replace("_", "\\_")
                fh.write(f"\\texttt{{{nm}}} & {r['cov_own']:.3f} & {r['y']:.3f} & "
                         f"{r['cov_deficit']:.3f} & {r['sel_loss']:.3f} & {r['y_arc']:.3f} \\\\\n")
            fh.write("\\midrule\n")
            fh.write(f"\\emph{{Sampling ceiling}} & {d['cov_sample'].iloc[0]:.3f} & & & & \\\\\n")
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
