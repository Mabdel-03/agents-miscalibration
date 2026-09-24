"""dynorch_v1, WP-C: every number, table and figure of the dynorch report (contract section 7; plan section 7).

    analysis.py --run <dynorch run root> --ladder <forced_budget_v2 merged dir> --config configs/dynorch_v1.json
                [--ladder-numbers docs/forced_v2_report/numbers.json] [--out DIR] [--replicates N]

Reads the dynorch run's ``tables/`` (``dynorch/tables.py`` output) and the sealed ladder tables (read-only), joins
them in memory through ``core.Frame`` and writes ``numbers.json`` (every number quoted in the prose), ``tables/*``
and ``figures/*.pdf``. Sections, in the order of the experiment design section 8: run facts, accuracies, the
anchor and the Holm-4 primary family (``primary.dynorch_primary_analysis``), the descriptive set (vs deep/k,
sample/k, private_single/k, the ladder's best non-deep config at that rung fixed from the sealed numbers, H0/k,
and the per-item oracle pass@k), the F2 realized frontier and the $-frontier, realized-budget and harness
diagnostics, the 0-spawn vs >= 1-spawn exploratory split, provider sensitivity, repeat agreement, and the
Gate 2 re-fit. ``run_analysis`` is the callable entry point used by ``tests/unit/dynorch/test_report_pipeline.py``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import core as c  # noqa: E402
import figures as fg  # noqa: E402
import primary as P  # noqa: E402
from agent_allocation.dynorch import forecast as fcst  # noqa: E402
from agent_allocation.dynorch import tables as T  # noqa: E402

RUNGS = [1, 2, 4, 8, 16, 25]
PRIMARY_RUNG = 16
FORCED_CLOSE_KINDS = ["none", "mid_call", "visible_continuation", "terminal"]
COMPLETENESS_MIN = 0.95


def r3(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)


def pts(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * float(x), 1)


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def signed(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.{d}f}".replace("-", "$-$")


def tex_name(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def _mean(series) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    return None if s.notna().sum() == 0 else float(s.mean())


def _share(mask) -> float | None:
    return None if len(mask) == 0 else float(np.mean(np.asarray(mask, dtype=float)))


# ---------------------------------------------------------------------------------------------------- driver

def run_analysis(run: str | Path, ladder: str | Path, config: str | Path, out: str | Path, *,
                 ladder_numbers: str | Path | None = None, replicates: int = c.REPLICATES,
                 panels: str | Path | None = None, partition: str = "main", figures: bool = True) -> dict:
    run, ladder, out = Path(run), Path(ladder), Path(out)
    cfg = json.loads(Path(config).read_text())
    ladder_numbers = Path(ladder_numbers) if ladder_numbers else HERE.parent / "forced_v2_report" / "numbers.json"
    LN = json.loads(ladder_numbers.read_text())
    panels = Path(panels) if panels else ladder.parent / "manifests" / "panels.json"
    os.makedirs(out / "figures", exist_ok=True)
    os.makedirs(out / "tables", exist_ok=True)
    N: dict[str, Any] = {"study": {"id": cfg["study_id"], "seed": c.SEED, "replicates": int(replicates),
                                   "ladder_study": cfg["base_study"]}}

    # ------------------------------------------------------------ inputs
    runs = pd.read_parquet(run / "tables" / "dynorch_runs.parquet")
    calls = pd.read_parquet(run / "tables" / "dynorch_calls.parquet")
    runs_all = runs.copy()  # every stage (canary, pilot, main): the run facts and the non-delegation counts
    po_h = pd.read_parquet(run / "tables" / "policy_outcomes_dynorch.parquet")
    pc_h = pd.read_parquet(run / "tables" / "policy_costs_dynorch.parquet")
    pc_l = c.fc.T(str(ladder), "policy_costs")
    pc_l = pc_l[(pc_l["partition_id"] == "main") & (pc_l["view"] == "current")]
    items = c.main_items(panels)
    F = c.Frame(ladder, po_h, items=items, replicates=replicates, partition=partition)
    po_h = po_h[(po_h["partition_id"] == partition) & (po_h["view"] == "current")]
    pc_h = pc_h[(pc_h["partition_id"] == partition) & (pc_h["view"] == "current")]
    runs = runs[runs["partition"] == partition]
    calls = calls[calls["partition"] == partition]
    hnames = sorted({n for n in F.names if c.arm_of(n) in c.ARM_LABEL}, key=lambda n: (c.arm_of(n), c.rung_of(n),
                                                                                       c.repeat_of(n)))
    arms = sorted({c.arm_of(n) for n in hnames}, key=lambda a: (a not in c.CORE_ARMS, a))
    base_names = [n for n in hnames if c.repeat_of(n) == 1]
    if not hnames:
        raise ValueError("no harness checkpoints in the run's policy_outcomes_dynorch")
    current = runs.sort_values("attempt").groupby(["arm", "item_id", "rung", "repeat"], as_index=False).tail(1)
    current = current[current["item_id"].isin(items)]
    calls_cur = calls[calls["run_key"].isin(current["run_key"])]

    # ------------------------------------------------------------ run facts
    N["n_items"] = F.n_items
    N["n_items_by_bench"] = {b: int(po_h[po_h["benchmark"] == b]["item_id"].nunique()) for b in c.BENCH_ORDER}
    N["n_cells_planned"] = int(len(po_h))
    N["n_runs_core"] = int(len(current))
    N["n_runs_all_attempts"] = int(len(runs))
    N["n_reruns"] = int(len(runs) - len(current))
    N["n_runs_by_arm_rung"] = {n: int((current["name"] == n).sum()) for n in hnames}
    N["n_cells_by_state"] = {k: int(v) for k, v in po_h["delivered_state"].value_counts().items()}
    comp = {}
    for n in hnames:
        st = F.state_counts(n)
        done = st.get("delivered", 0) + st.get("op_no_output", 0)
        comp[n] = r3(done / F.n_items)
    N["completeness"] = comp
    N["partial_cells"] = [n for n in hnames if comp[n] < COMPLETENESS_MIN]
    N["n_partial_cells"] = len(N["partial_cells"])
    N["stages_present"] = sorted(p.name.replace("dynorch_plan_", "").replace(".jsonl", "")
                                 for p in (run / "manifests").glob("dynorch_plan_*.jsonl")) if (run / "manifests").is_dir() else []
    money = pd.to_numeric(current["cost_usd"], errors="coerce")
    N["spend"] = {"core_usd": round(float(money.sum()), 2),
                  "all_attempts_usd": round(float(pd.to_numeric(runs["cost_usd"], errors="coerce").sum()), 2),
                  "retry_usd": round(float(pd.to_numeric(pc_h["retry_cost_usd"], errors="coerce").sum()), 2),
                  "standardized_usd": round(float(pd.to_numeric(pc_h["standardized_uncached_cost_usd"],
                                                                errors="coerce").sum()), 2)}
    N["ledger"] = {"settled_usd": N["spend"]["all_attempts_usd"], "n_records": int(len(runs))}
    N["harness_versions"] = {a: ", ".join(sorted(set(current[current["arm"] == a]["harness_version"].astype(str))))
                             for a in arms}
    N["proxy_versions"] = ", ".join(sorted(set(current["proxy_version"].astype(str))))
    # provider covariate
    ladder_shard = (F.oc[F.oc["family"] != c.HARNESS_FAMILY].drop_duplicates("item_id")
                    .set_index("item_id")["shard_id"].astype(str).to_dict())
    N["ladder_provider_assignment"] = {s: int(sum(1 for i in items if ladder_shard.get(i) == s))
                                       for s in sorted(set(ladder_shard.values()))}
    hs = po_h[po_h["delivered_state"] != "pending"]
    N["provider_assignment"] = {str(s): int(g["item_id"].nunique()) for s, g in hs.groupby("shard_id")}
    same = hs["shard_id"].astype(str).to_numpy() == hs["item_id"].map(ladder_shard).astype(str).to_numpy()
    N["same_provider_share"] = r3(_share(same)) if len(hs) else None
    N["same_provider_items"] = int(hs[same]["item_id"].nunique()) if len(hs) else 0
    # scaffolding S_h = prompt_tokens(first root call) - prompt_tokens(H0 call, same item, same k)
    first = current[["arm", "item_id", "rung", "prompt_tokens_first_root"]].dropna()
    h0 = first[first["arm"] == "H0"].set_index(["item_id", "rung"])["prompt_tokens_first_root"]
    N["harness_prompt_tokens"] = {}
    for a in arms:
        if a == "H0":
            continue
        g = first[first["arm"] == a].set_index(["item_id", "rung"])["prompt_tokens_first_root"]
        j = pd.concat([g, h0], axis=1, keys=["h", "h0"]).dropna()
        d = j["h"] - j["h0"]
        N["harness_prompt_tokens"][a] = {"mean": None if d.empty else round(float(d.mean()), 1),
                                         "sd": None if len(d) < 2 else round(float(d.std(ddof=1)), 1),
                                         "n": int(len(d))}
    N["h0_prompt_tokens_mean"] = None if h0.empty else round(float(h0.mean()), 1)

    # ------------------------------------------------------------ accuracies (harness + comparators)
    best_non_deep, best_team = {}, {}
    ladder_idx = F.idx[F.idx["family"] != c.HARNESS_FAMILY]
    for k in RUNGS:
        names_k = list(ladder_idx[ladder_idx["rung"] == k]["name"])
        nd = [n for n in names_k if not n.startswith("deep/") and n in LN["acc"]["macro"]]
        tm = [n for n in nd if n.startswith(("peer/", "private_team/"))]
        best_non_deep[k] = max(nd, key=lambda n: LN["acc"]["macro"][n]["acc"]) if nd else None
        best_team[k] = max(tm, key=lambda n: LN["acc"]["macro"][n]["acc"]) if tm else None
    N["best_non_deep_at_rung"] = {str(k): v for k, v in best_non_deep.items()}
    N["best_team_at_rung"] = {str(k): v for k, v in best_team.items()}
    comparators = ["sample/1"] + [f"{f}/{k}" for k in RUNGS for f in ("deep", "sample", "private_single")]
    comparators += [v for v in best_non_deep.values() if v] + [v for v in best_team.values() if v]
    comparators = [n for n in dict.fromkeys(comparators) if F.has(n)]
    acc: dict[str, dict[str, dict]] = {}
    for n in hnames + comparators:
        acc[n] = {}
        for level in c.LEVELS:
            a_, lo, hi = F.acc(n, level)
            acc[n][level] = {"acc": r3(a_), "lo": r3(lo), "hi": r3(hi)}
        acc[n]["unknown_share"] = r3(F.unknown_share(n))
    N["acc"] = acc
    # sealed point estimates reproduced (Gate 4 criterion)
    N["ladder_check"] = {n: {"sealed": LN["acc"]["macro"][n]["acc"], "here": acc[n]["macro"]["acc"],
                             "equal_3dp": abs(LN["acc"]["macro"][n]["acc"] - acc[n]["macro"]["acc"]) < 5e-4}
                         for n in ("deep/16", "sample/16", "deep/25", "sample/1") if n in acc}
    bank = c.bank_curve(ladder)
    N["oracle"] = {lv: {str(int(k)): r3(v) for k, v in g.set_index("k")["oracle_pass_at_k"].items()}
                   for lv, g in bank.groupby("level")}

    # ------------------------------------------------------------ primary family + anchor
    prim, _, pnotes = P.dynorch_primary_analysis(F.oc, cfg, replicates=replicates, study_seed=c.SEED, items=items,
                                                 n_primary=int(cfg["statistics"]["n_primary"]))
    prim.to_csv(out / "tables" / "primary_contrasts.csv", index=False)
    N["primary_notes"] = pnotes
    N["primary"] = {}
    for _, r in prim[prim["level"] == "macro"].iterrows():
        e = {"est_pts": pts(r["estimate"]), "lo_pts": pts(r["ci95_lo"]), "hi_pts": pts(r["ci95_hi"]),
             "p": r3(r["p_value"]) if r["p_value"] is not None and not pd.isna(r["p_value"]) else None,
             "p_holm": r3(r["p_holm"]) if r["p_holm"] is not None and not pd.isna(r["p_holm"]) else None,
             "reject": None if r["holm_reject"] is None or pd.isna(r["holm_reject"]) else bool(r["holm_reject"]),
             "n": int(r["n_items"]), "n_complete": int(r["n_complete"]), "status": r["status"], "role": r["role"],
             "bound_lo_pts": pts(r["bound_lower"]), "bound_hi_pts": pts(r["bound_upper"]),
             "reading": _reading(r)}
        e["reading_words"] = READING_WORDS.get(e["reading"]) if e["reading"] else None
        e["p_holm_tex"] = _p_tex(e["p_holm"], r["p_holm"])
        N["primary"][r["contrast"]] = e
    for tag, other in (("deep", f"deep/{PRIMARY_RUNG}"), ("sample", f"sample/{PRIMARY_RUNG}")):
        words = {N["primary"].get(f"{a}/{PRIMARY_RUNG}-{other}", {}).get("reading_words") for a in ("H1", "H3")}
        # one phrase when both readings agree; otherwise each harness's own reading
        N.setdefault("primary_reading_both", {})[tag] = (words.pop() if len(words) == 1 else "; ".join(
            f"{a}: {N['primary'].get(f'{a}/{PRIMARY_RUNG}-{other}', {}).get('reading_words')}" for a in ("H1", "H3")))
    N["primary_family"] = list(cfg["statistics"]["primary_family"])
    N["anchor"] = str(cfg["statistics"]["anchor_reported_first"])
    N["primary_p_reported"] = all(N["primary"][k]["p_holm"] is not None for k in N["primary_family"])
    N["mde_note"] = LN["depth_vs_width"]["macro"]["16"]["deep_minus_sample"]

    # ------------------------------------------------------------ descriptive set
    bank_correct = c.bank_correct_by_item(ladder)
    desc_rows = []
    N["descriptive"] = {}
    for n in hnames:
        k = c.rung_of(n)
        comps = {"vs_deep": f"deep/{k}", "vs_sample": f"sample/{k}", "vs_private": f"private_single/{k}",
                 "vs_best_non_deep": best_non_deep.get(k), "vs_best_team": best_team.get(k), "vs_H0": f"H0/{k}"}
        entry = {}
        for tag, other in comps.items():
            if other is None or not F.has(other) or other == n:
                entry[tag] = None
                continue
            for level in c.LEVELS:
                est, lo, hi, p = F.diff(n, other, level)
                desc_rows.append(dict(name=n, arm=c.arm_of(n), rung=k, contrast=tag, other=other, level=level,
                                      delta=est, lo=lo, hi=hi, p_null_centred=p))
                if level == "macro":
                    entry[tag] = {"other": other, "pts": pts(est), "lo": pts(lo), "hi": pts(hi)}
        oracle = c.oracle_by_item(bank_correct, k)
        for level in c.LEVELS:
            est, lo, hi, n_known = F.diff_vs_reference(n, oracle, level)
            desc_rows.append(dict(name=n, arm=c.arm_of(n), rung=k, contrast="vs_oracle", other=f"pass@{k}",
                                  level=level, delta=est, lo=lo, hi=hi, p_null_centred=None))
            if level == "macro":
                entry["vs_oracle"] = {"other": f"pass@{k}", "pts": pts(est), "lo": pts(lo), "hi": pts(hi),
                                      "n_known": n_known}
        N["descriptive"][n] = entry
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(out / "tables" / "descriptive.csv", index=False)
    # slope per doubling: OLS of macro accuracy points on log2(rung) over the core rungs, per arm
    N["slopes"] = {}
    for a in arms:
        xs = [math.log2(c.rung_of(n)) for n in base_names if c.arm_of(n) == a]
        ys = [100 * acc[n]["macro"]["acc"] for n in base_names if c.arm_of(n) == a]
        N["slopes"][a] = None if len(xs) < 2 else round(float(np.polyfit(xs, ys, 1)[0]), 1)
    for fam in ("deep", "sample", "private_single"):
        xs = [math.log2(k) for k in RUNGS if F.has(f"{fam}/{k}")]
        ys = [100 * acc[f"{fam}/{k}"]["macro"]["acc"] for k in RUNGS if F.has(f"{fam}/{k}")]
        N["slopes"][fam] = None if len(xs) < 2 else round(float(np.polyfit(xs, ys, 1)[0]), 1)

    # ------------------------------------------------------------ realized budget (B10) and F2 / $ frontiers
    N["realized"] = {}
    for n in hnames:
        g = current[current["name"] == n]
        if g.empty:
            N["realized"][n] = None
            continue
        N["realized"][n] = {
            "n": int(len(g)), "cap": int(g["cap_tokens"].iloc[0]),
            "mean_completion": round(float(g["completion_total"].mean()), 1),
            "median_completion": round(float(g["completion_total"].median()), 1),
            "mean_grace": round(float(g["grace_tokens"].mean()), 1),
            "share_grace": r3(_share(g["grace_tokens"] > 0)),
            "share_cap_bound": r3(_share(g["cap_bound_calls"] > 0)),
            "share_forced_close": r3(_share(g["forced_close_kind"] != "none")),
            "share_terminal": r3(_share(g["forced_close_kind"] == "terminal")),
            "share_mid_call": r3(_share(g["forced_close_kind"] == "mid_call")),
            "realized_share": r3(float(g["realized_share"].mean())),
            "mean_overdraft": round(float(g["overdraft_tokens"].mean()), 1),
            "max_completion": int(g["completion_total"].max()),
            "cap_overrun_runs": int((g["completion_total"] > g["cap_tokens"]).sum()),
            "grace_overrun_runs": int((g["grace_tokens"] > int(cfg["dynorch"]["grace_tokens"])).sum())}
    N["cap_overrun_runs_total"] = int((current["completion_total"] > current["cap_tokens"]).sum())
    ladder_cost = pc_l[pc_l["family"].isin(c.LADDER_FAMILIES)].copy()
    for col in ("total_charged_usd", "standardized_uncached_cost_usd"):
        ladder_cost[col] = pd.to_numeric(ladder_cost[col], errors="coerce")
    lc = ladder_cost.groupby("name").agg(completion_tokens=("completion_tokens", "mean"),
                                         usd=("total_charged_usd", "mean"),
                                         std_usd=("standardized_uncached_cost_usd", "mean"),
                                         prompt_tokens=("prompt_tokens", "mean"))
    hc = pc_h[pc_h["delivered_state"].isin(["delivered", "op_no_output"])].copy()
    for col in ("total_charged_usd", "standardized_uncached_cost_usd"):
        hc[col] = pd.to_numeric(hc[col], errors="coerce")
    hcg = hc.groupby("name").agg(completion_tokens=("completion_tokens", "mean"), usd=("total_charged_usd", "mean"),
                                 std_usd=("standardized_uncached_cost_usd", "mean"),
                                 prompt_tokens=("prompt_tokens", "mean"))
    front_rows = []
    for n in comparators:
        if n in lc.index:
            front_rows.append(dict(name=n, family=n.split("/")[0], side="ladder", x_tokens=float(lc.loc[n, "completion_tokens"]),
                                   nominal=c.rung_of(n) * c.A0, usd=float(lc.loc[n, "usd"]), std_usd=float(lc.loc[n, "std_usd"]),
                                   prompt_tokens=float(lc.loc[n, "prompt_tokens"]), acc=acc[n]["macro"]["acc"]))
    for n in hnames:
        if n in hcg.index:
            front_rows.append(dict(name=n, family=c.HARNESS_FAMILY, side="harness", x_tokens=float(hcg.loc[n, "completion_tokens"]),
                                   nominal=c.rung_of(n) * c.A0, usd=float(hcg.loc[n, "usd"]), std_usd=float(hcg.loc[n, "std_usd"]),
                                   prompt_tokens=float(hcg.loc[n, "prompt_tokens"]), acc=acc[n]["macro"]["acc"]))
    front = pd.DataFrame(front_rows)
    front.to_csv(out / "tables" / "frontier.csv", index=False)
    N["frontier_f2"] = {r["name"]: {"x_tokens": round(r["x_tokens"], 1), "acc": r3(r["acc"]), "side": r["side"]}
                        for _, r in front.iterrows()}
    N["frontier_usd"] = {r["name"]: {"usd": round(r["usd"], 4), "std_usd": round(r["std_usd"], 4), "acc": r3(r["acc"])}
                         for _, r in front.iterrows()}
    N["frontier"] = {"f2": _pareto(front, "x_tokens"), "usd": _pareto(front, "usd"), "std_usd": _pareto(front, "std_usd")}
    N["frontier_harness_points"] = {m: [n for n in N["frontier"][m] if c.arm_of(n) in c.ARM_LABEL]
                                    for m in ("f2", "usd", "std_usd")}
    # F1 bias: the ladder's C is reasoning-only; its uncounted ANSWER tokens per configuration
    N["f1_bias_answer_tokens"] = {n: round(float(pc_l[pc_l["name"] == n]["answer_tokens"].mean()), 1)
                                  for n in ("deep/16", "sample/16", "deep/4", "sample/4") if (pc_l["name"] == n).any()}

    # ------------------------------------------------------------ diagnostics
    N["diagnostics"] = {}
    for n in hnames:
        g = current[current["name"] == n]
        gc = calls_cur[calls_cur["name"] == n]
        if g.empty:
            N["diagnostics"][n] = None
            continue
        root_tokens = gc[gc["conversation_class"] == "root"]["completion_tokens"].fillna(0).sum()
        all_tokens = gc["completion_tokens"].fillna(0).sum()
        N["diagnostics"][n] = {
            "n": int(len(g)), "calls": round(float(g["calls"].mean()), 2),
            "calls_root": round(float(g["calls_root"].mean()), 2), "calls_solver": round(float(g["calls_solver"].mean()), 2),
            "calls_other": round(float(g["calls_other"].mean()), 2),
            "other_share": r3(float(g["calls_other"].sum() / max(1, g["calls"].sum()))),
            "spawns": round(float(g["spawns"].mean()), 2), "spawn_share": r3(_share(g["spawns"] > 0)),
            "conversations": round(float(g["conversations"].mean()), 2),
            "orch_share": r3(float(root_tokens / all_tokens)) if all_tokens else None,
            "forced_close": {k: r3(_share(g["forced_close_kind"] == k)) for k in FORCED_CLOSE_KINDS},
            "exhaustion": r3(_share(g["exhausted"].astype(bool))),
            "abstention": r3(_share(~g["valid_final"].astype(bool))),
            "parse_fail": r3(_share(g["tool_parse_failures"] > 0)),
            "reinjection_hit_rate": _rate(g["reinjection_hits"].sum(), g["reinjection_misses"].sum()),
            "duplicates": round(float(g["duplicates_served"].mean()), 3),
            "guard": r3(_share(g["guard_triggered"].astype(bool))),
            "refused_calls": round(float(g["refused_calls"].mean()), 2),
            "wall_s": round(float(g["wall_seconds"].mean()), 1),
            "wall_s_p90": round(float(g["wall_seconds"].quantile(0.9)), 1),
            "timeout_share": r3(_share(g["status"] == "timeout")),
            "crash_share": r3(_share((g["status"] == "failed"))),
            "transport_failed_share": r3(_share(g["status"] == "transport_failed")),
            "cli_mismatch": r3(_share(g["cli_result_mismatch"].fillna(False).astype(bool))),
            "prompt_tokens": round(float(g["prompt_total"].mean()), 1),
            "usd": round(float(pd.to_numeric(g["cost_usd"], errors="coerce").mean()), 4),
            "queue_wait_s": round(float(gc["queue_wait_seconds"].mean()), 2) if len(gc) else None}
    # exploratory 0-spawn vs >= 1-spawn accuracy (outcome-conditioned, labelled as such)
    N["spawn_split"] = {}
    for n in hnames:
        g = current[(current["name"] == n) & current["y"].notna()]
        z, s = g[g["spawns"] == 0], g[g["spawns"] > 0]
        N["spawn_split"][n] = {"zero": {"n": int(len(z)), "acc": r3(z["y"].mean()) if len(z) else None},
                               "some": {"n": int(len(s)), "acc": r3(s["y"].mean()) if len(s) else None}}

    # ------------------------------------------------------------ provider sensitivity
    mancer = {i for i in items if ladder_shard.get(i) == "mancer"}
    parasail = {i for i in items if ladder_shard.get(i) == "parasail"}
    hshard = hs.drop_duplicates(["item_id", "name"]).set_index(["item_id", "name"])["shard_id"].astype(str).to_dict()
    N["provider_sensitivity"] = {}
    prov_rows = []
    for text in [N["anchor"], *N["primary_family"]]:
        left, right = text.split("-")
        if not (F.has(left) and F.has(right)):
            continue
        same_items = {i for i in items if hshard.get((i, left)) == ladder_shard.get(i)}
        entry = {}
        for tag, subset in (("mancer_items", mancer), ("parasail_items", parasail), ("same_provider_only", same_items)):
            if not subset:
                entry[tag] = {"n": 0, "pts": None, "lo": None, "hi": None}
                continue
            est, lo, hi, _ = F.diff(left, right, "macro", items=subset)
            entry[tag] = {"n": len(subset), "pts": pts(est), "lo": pts(lo), "hi": pts(hi)}
            prov_rows.append(dict(contrast=text, subset=tag, n=len(subset), delta=est, lo=lo, hi=hi))
        N["provider_sensitivity"][text] = entry
    pd.DataFrame(prov_rows).to_csv(out / "tables" / "provider_sensitivity.csv", index=False)

    # ------------------------------------------------------------ repeat agreement (E1)
    N["repeat_agreement"] = {"available": False, "cells": {}}
    rep_rows = []
    for n in hnames:
        if c.repeat_of(n) != 2:
            continue
        base = f"{c.arm_of(n)}/{c.rung_of(n)}"
        if not F.has(base):
            continue
        a1 = F.oc[F.oc["name"] == base].set_index("item_id")["y"]
        a2 = F.oc[F.oc["name"] == n].set_index("item_id")["y"]
        j = pd.concat([a1, a2], axis=1, keys=["r1", "r2"]).dropna()
        est, lo, hi, _ = F.diff(n, base, "macro")
        e = {"n": int(len(j)), "agreement": r3((j["r1"] == j["r2"]).mean()) if len(j) else None,
             "kappa": r3(_kappa(j["r1"], j["r2"])) if len(j) else None, "delta_pts": pts(est), "lo": pts(lo),
             "hi": pts(hi), "pass2": r3(((j["r1"] > 0) | (j["r2"] > 0)).mean()) if len(j) else None}
        N["repeat_agreement"]["cells"][base] = e
        rep_rows.append({"cell": base, **e})
    N["repeat_agreement"]["available"] = bool(rep_rows)
    N["repeat_agreement"]["n_cells"] = len(rep_rows)
    pd.DataFrame(rep_rows).to_csv(out / "tables" / "repeat_agreement.csv", index=False)

    # ------------------------------------------------------------ re-fit (Gate 2 style, from the measured runs)
    refit = fcst.refit(runs, cfg, planned_runs=fcst.planned_runs_core(cfg))
    refit.to_csv(out / "tables" / "forecast_refit.csv", index=False)
    N["refit"] = {f"{r['arm']}/{r['rung']}": {"cost_usd_per_run": round(r["cost_usd_per_run_refit"], 4),
                                             "measured_usd_per_run": round(r["mean_cost_usd_measured"], 4),
                                             "mean_calls": round(r["mean_calls"], 2)}
                  for _, r in refit.iterrows()}
    N["refit_total_core_usd"] = round(float(sum(r["cost_usd_per_run_refit"] * (r["planned_runs"] or 0)
                                                for _, r in refit.iterrows())), 2)

    # ------------------------------------------------------------ interpretation (behaviour, decomposition, F2 vs deep)
    N["harness_display"] = {a: _display_name(a, N["harness_versions"].get(a, "")) for a in arms if a != "H0"}
    dyn = cfg["dynorch"]
    N["design"] = {"A": int(dyn["a_tokens"]), "grace_tokens": int(dyn["grace_tokens"]),
                   "cap16": int(PRIMARY_RUNG * int(dyn["a_tokens"])), "n_primary": int(cfg["statistics"]["n_primary"]),
                   "items_per_bench": int(min(N["n_items_by_bench"].values())),
                   "n_benchmarks": len(N["n_items_by_bench"]), "rung_max": max(int(k) for k in dyn["rungs"]),
                   "e0_root_ceiling": str(dyn["extension_arms"]["H1c"]["root_call_ceiling"]),
                   "tool_surface": str(dyn["tool_surface"]), "model": _model_label(cfg["primary_model"]),
                   "completeness_min": COMPLETENESS_MIN,
                   "contingency_share": float(cfg["budget"]["forecast_contingency_share"]),
                   "allowed_tools": {a: list(dyn["arms"][a]["allowed_tools"]) for a in dyn["arms"]}}
    N["behaviour"] = behaviour_numbers(runs_all, current, calls_cur, run, partition)
    N["decomposition"] = decomposition_numbers(F.oc, N, arms, replicates=replicates, items=items)
    rvd = realized_vs_deep(F.oc, N, base_names, replicates=replicates, items=items)
    N["realized_vs_deep"] = rvd
    N["run_facts"] = run_facts(run, runs_all, current, calls, partition)
    N["e0"] = e0_numbers(ladder, run, items, runs_all, pd.read_parquet(run / "tables" / "dynorch_calls.parquet"),
                         replicates=replicates, partition=partition)

    # ------------------------------------------------------------ headline shortcuts
    N["headline"] = {}
    for a in arms:
        n16 = f"{a}/{PRIMARY_RUNG}"
        if F.has(n16):
            N["headline"][a] = {"acc16": acc[n16]["macro"]["acc"], "lo16": acc[n16]["macro"]["lo"],
                                "hi16": acc[n16]["macro"]["hi"], "spawn_share16": N["diagnostics"][n16]["spawn_share"]
                                if N["diagnostics"].get(n16) else None,
                                "realized_share16": N["realized"][n16]["realized_share"] if N["realized"].get(n16) else None}
    N["headline"]["deep16"] = acc["deep/16"]["macro"]["acc"]
    N["headline"]["sample16"] = acc["sample/16"]["macro"]["acc"]
    N["headline"]["oracle16"] = N["oracle"]["macro"]["16"]
    N["headline"]["private16"] = acc["private_single/16"]["macro"]["acc"]
    N["headline"]["best_non_deep16"] = best_non_deep[16]
    N["headline"]["best_non_deep16_acc"] = acc[best_non_deep[16]]["macro"]["acc"] if best_non_deep[16] else None

    c.dump(N, out / "numbers.json")
    write_tables(N, F, acc, arms, hnames, base_names, best_non_deep, front, out)
    if figures:
        fg.style()
        fg.d1_design(N, cfg, arms, out)
        fg.d2_ladder_f1(N, F, bank, acc, hnames, out)
        fg.d3_realized_f2(N, front, out)
        fg.d4_dollars(N, front, out)
        fg.d5_per_benchmark(N, acc, arms, best_non_deep, out)
        fg.d6_diagnostics(N, arms, base_names, out)
        fg.d7_cap_binding(current, arms, out)
        fg.d8_provider_sensitivity(N, out)
        fg.d9_repeat_agreement(N, F, out)
    return N


# ---------------------------------------------------------------------------------------------------- interpretation

HARNESS_NAME = {"H1": "OpenClaude", "H3": "Codex CLI"}
READING_WORDS = {"loses_to_fixed_rule": "dynamic orchestration loses to the fixed rule",
                 "rule_incomplete": "the fixed rule is incomplete",
                 "inconclusive": "inconclusive", "above_baseline": "above the sampling baseline",
                 "below_baseline": "below the sampling baseline"}


def _model_label(slug: str) -> str:
    """``"qwen/qwen3.8-27b"`` -> ``"Qwen3.8-27B"``."""
    name = str(slug).split("/")[-1]
    head, _, size = name.partition("-")
    return head[:1].upper() + head[1:] + ("-" + size.upper() if size else "")


def _p_tex(p_rounded, p_raw) -> str | None:
    """Holm p for the prose, in math mode: ``= 0.004``, or ``< 0.001`` when it rounds to zero."""
    if p_raw is None or pd.isna(p_raw):
        return None
    return "< 0.001" if float(p_raw) < 0.0005 else f"= {float(p_rounded):.3f}"


def _display_name(arm: str, version: str) -> str:
    """``"0.13.0 (OpenClaude)"`` -> ``"OpenClaude 0.13.0"``; ``"codex-cli 0.156.1"`` -> ``"Codex CLI 0.156.1"``."""
    m = re.search(r"\d+(?:\.\d+)+", str(version))
    return f"{HARNESS_NAME.get(arm, arm)} {m.group(0)}" if m else f"{HARNESS_NAME.get(arm, arm)} {version}"


def _contrast_row(oc: pd.DataFrame, name: str, terms, *, replicates: int, items) -> dict:
    """One descriptive linear contrast through ``analysis/contrasts.evaluate_contrast`` (the machinery of the
    anchor and the primary family: one population, studentized item-cluster bootstrap; no p-value, role
    ``descriptive``). Returns the macro estimate and 95% CI in points."""
    from agent_allocation.analysis.contrasts import LinearContrast, evaluate_contrast
    lc = LinearContrast(name, tuple(("main", n, float(w)) for n, w in terms), "main", "descriptive",
                        f"{name}, delivered accuracy (macro, paired item bootstrap)")
    rows, _ = evaluate_contrast(oc, lc, replicates=replicates, study_seed=c.SEED, items=set(items), n_primary=4)
    r = next(r for r in rows if r["level"] == "macro")
    lo = r.get("ci95_lo") if r.get("status") == "complete" else r.get("ci95_percentile_lo")
    hi = r.get("ci95_hi") if r.get("status") == "complete" else r.get("ci95_percentile_hi")
    return {"pts": pts(r.get("estimate")), "lo": pts(lo), "hi": pts(hi), "status": r.get("status"),
            "n": int(r.get("n_items") or 0), "n_complete": int(r.get("n_complete") or 0)}


def behaviour_numbers(runs_all: pd.DataFrame, current: pd.DataFrame, calls_cur: pd.DataFrame, run: Path,
                      partition: str) -> dict:
    """What the harnesses did: calls per run, spawns, tool calls, per arm and per stage; the tool behind every
    second call of a Codex run (read from the raw call journal of the scored attempt)."""
    out: dict[str, Any] = {"by_arm": {}}
    tool_runs = set(calls_cur[calls_cur["tool_calls"].fillna(0) > 0]["run_key"])
    for a, g in current.groupby("arm"):
        out["by_arm"][a] = {"runs": int(len(g)), "one_call": int((g["calls"] == 1).sum()),
                            "multi_call": int((g["calls"] > 1).sum()), "max_calls": int(g["calls"].max()),
                            "spawn_runs": int((g["spawns"] > 0).sum()), "spawns": int(g["spawns"].sum()),
                            "tool_call_runs": int(g["run_key"].isin(tool_runs).sum()),
                            "multi_call_share": r3((g["calls"] > 1).mean())}
    har = current[current["arm"] != "H0"]
    out["core_runs"] = int(len(current))
    out["core_spawn_runs"] = int((current["spawns"] > 0).sum())
    out["core_harness_runs"] = int(len(har))
    out["core_harness_spawn_runs"] = int((har["spawns"] > 0).sum())
    main_all = runs_all[runs_all["partition"] == partition]
    out["main_all_attempts"] = int(len(main_all))
    out["main_all_attempts_spawn_runs"] = int((main_all["spawns"] > 0).sum())
    for stage in ("canary", "pilot"):
        g = runs_all[(runs_all["partition"] == stage) & (runs_all["arm"] != "H0")]
        out[f"{stage}_harness_runs"] = int(len(g))
        out[f"{stage}_harness_spawn_runs"] = int((g["spawns"] > 0).sum())
        out[f"{stage}_harness_multi_call"] = int((g["calls"] > 1).sum())
    # the core study's stages only: extension stages (ext_*) share the tables but are reported separately
    allh = runs_all[runs_all["partition"].isin(["canary", "pilot", partition]) & (runs_all["arm"] != "H0")]
    out["all_stages_harness_runs"] = int(len(allh))
    out["all_stages_harness_spawn_runs"] = int((allh["spawns"] > 0).sum())
    # Codex second calls: which tool the first call invoked before the second (raw journal of call 2)
    multi = current[(current["arm"] == "H3") & (current["calls"] > 1)]
    names_per_run, unreadable = [], 0
    for _, r in multi.iterrows():
        d = run / "raw" / partition / r["arm"] / str(int(r["rung"])) / str(int(r["repeat"])) / str(r["item_id"]).replace("/", "__")
        files = sorted(d.glob(f"a{int(r['attempt'])}*/call_0002.json.gz"))
        if not files:
            unreadable += 1
            continue
        with gzip.open(files[-1], "rt") as fh:
            doc = json.load(fh)
        names = []
        for m in (doc.get("harness_request") or {}).get("messages") or []:
            for tc in m.get("tool_calls") or []:
                names.append((tc.get("function") or tc).get("name"))
        names_per_run.append(names)
    planning = sum(1 for n in names_per_run if n and set(n) == {"update_plan"})
    out["H3_second_call"] = {"runs": int(len(multi)), "after_update_plan": int(planning),
                             "after_other_tool": int(sum(1 for n in names_per_run if n and set(n) != {"update_plan"})),
                             "unreadable": int(unreadable),
                             "tool_names": sorted({x for n in names_per_run for x in n if x}),
                             "rungs": sorted(int(k) for k in multi["rung"].unique()),
                             "share": r3(len(multi) / max(1, (current["arm"] == "H3").sum()))}
    out["H3_second_call"]["all_planning"] = bool(multi.shape[0] and planning == len(multi))
    out["H3_second_call"]["first_tool"] = out["H3_second_call"]["tool_names"][0] if out["H3_second_call"]["tool_names"] else "none"
    return out


E0_STAGE = "ext_e0"


def e0_numbers(ladder: Path, run: Path, items, runs_all: pd.DataFrame, calls_all: pd.DataFrame, *, replicates: int,
               partition: str) -> dict:
    """Extension E0 (capped root calls, plan §1): accuracy of H1c/H3c at rungs 4 and 16 and descriptive paired
    contrasts against the uncapped cell, deep/k, sample/k and the deep call at the root ceiling max(A, C/4); what the
    capped harnesses did. E0 rows join the main harness rows in one frame (their names are distinct), so every
    contrast is one population on the same items. Empty when the stage has not been sealed."""
    from agent_allocation.dynorch.gates import seal_path
    if not seal_path(run / "tables", E0_STAGE).exists():
        return {"present": False}
    po = pd.read_parquet(run / "tables" / "policy_outcomes_dynorch.parquet")
    po = po[po["partition_id"].isin([partition, E0_STAGE]) & (po["view"] == "current")].copy()
    po["partition_id"] = partition
    F0 = c.Frame(ladder, po, items=items, replicates=replicates, partition=partition)
    out: dict[str, Any] = {"present": True, "acc": {}, "vs": {}, "by_arm": {}}
    runs = runs_all[runs_all["partition"] == E0_STAGE]
    cur = runs.sort_values("attempt").groupby(["arm", "item_id", "rung", "repeat"], as_index=False).tail(1)
    cur = cur[cur["item_id"].isin(items)]
    calls = calls_all[calls_all["run_key"].isin(cur["run_key"])]
    for arm in ("H1c", "H3c"):
        base = arm[:-1]
        for k in (4, 16):
            n = f"{arm}/{k}"
            if not F0.has(n):
                continue
            a_, lo, hi = F0.acc(n)
            out["acc"][n] = {"acc": r3(a_), "lo": r3(lo), "hi": r3(hi)}
            ceil_rung = max(1, k // 4)
            for other in (f"{base}/{k}", f"deep/{k}", f"sample/{k}", f"deep/{ceil_rung}"):
                if F0.has(other):
                    out["vs"].setdefault(n, {})[other.replace("/", "_")] = _contrast_row(
                        F0.oc, f"{n}-{other}", [(n, 1.0), (other, -1.0)], replicates=replicates, items=items)
            out["vs"][n]["ceiling_rung"] = ceil_rung
            g = cur[cur["name"] == n]
            gc = calls[calls["run_key"].isin(g["run_key"])]
            closed = set(gc[gc["forced_close_kind"] == "ceiling_close"]["run_key"])
            bound = g[g["run_key"].isin(closed)]
            unbound = g[~g["run_key"].isin(closed)]
            ended = bound[bound["calls"] == 1]
            out["by_arm"][n] = {
                "runs": int(len(g)), "spawn_runs": int((g["spawns"] > 0).sum()),
                "spawn_share": r3((g["spawns"] > 0).mean()), "multi_call_share": r3((g["calls"] > 1).mean()),
                "realized_share": r3(g["realized_share"].mean()),
                "ceiling_close_runs": int(len(bound)), "ceiling_close_share": r3(len(bound) / max(1, len(g))),
                "ceiling_close_then_end": int(len(ended)),
                "ceiling_close_then_end_share": r3(len(ended) / max(1, len(bound))),
                "acc_ceiling_closed": r3(bound["y"].mean()) if len(bound) else None,
                "acc_not_closed": r3(unbound["y"].mean()) if len(unbound) else None,
                "terminal_share": r3((g["forced_close_kind"] == "terminal").mean()),
                "no_output_share": r3((g["delivered_state"] == "op_no_output").mean()),
                "acc_spawned": r3(g[g["spawns"] > 0]["y"].mean()) if (g["spawns"] > 0).any() else None,
                "acc_not_spawned": r3(g[g["spawns"] == 0]["y"].mean()) if (g["spawns"] == 0).any() else None}
    money = pd.to_numeric(runs["cost_usd"], errors="coerce")
    out["runs_all_attempts"] = int(len(runs))
    out["runs"] = int(len(cur))
    out["spend_usd"] = round(float(money.sum()), 2)
    out["spawn_runs"] = int((cur["spawns"] > 0).sum())
    out["spawn_share"] = r3((cur["spawns"] > 0).mean())
    out["ceil16"] = int(runs[runs["rung"] == 16]["root_call_ceiling"].dropna().astype(int).max()) \
        if (runs["rung"] == 16).any() else None
    return out


def decomposition_numbers(oc: pd.DataFrame, N: dict, arms, *, replicates: int, items) -> dict:
    """H/16 - deep/16 = (H0/16 - deep/16) + (H/16 - H0/16): the natural-stop part (the anchor) and the
    scaffolding part, the latter per harness through the same contrast machinery; descriptive, no p-values."""
    out = {}
    anchor = N["primary"].get(f"H0/{PRIMARY_RUNG}-deep/{PRIMARY_RUNG}")
    for a in arms:
        if a == "H0" or f"{a}/{PRIMARY_RUNG}-deep/{PRIMARY_RUNG}" not in N["primary"]:
            continue
        total = N["primary"][f"{a}/{PRIMARY_RUNG}-deep/{PRIMARY_RUNG}"]
        sc = _contrast_row(oc, f"{a}/{PRIMARY_RUNG}-H0/{PRIMARY_RUNG}",
                           [(f"{a}/{PRIMARY_RUNG}", 1.0), (f"H0/{PRIMARY_RUNG}", -1.0)], replicates=replicates, items=items)
        out[a] = {"total_pts": total["est_pts"], "total_lo": total["lo_pts"], "total_hi": total["hi_pts"],
                  "anchor_pts": anchor["est_pts"], "anchor_lo": anchor["lo_pts"], "anchor_hi": anchor["hi_pts"],
                  "scaffold_pts": sc["pts"], "scaffold_lo": sc["lo"], "scaffold_hi": sc["hi"],
                  "scaffold_status": sc["status"], "scaffold_ci_covers_zero": bool(sc["lo"] <= 0 <= sc["hi"]),
                  "prompt_tokens": N["harness_prompt_tokens"].get(a, {}).get("mean"),
                  "prompt_tokens_k": round(N["harness_prompt_tokens"][a]["mean"] / 1000, 1)
                  if N["harness_prompt_tokens"].get(a, {}).get("mean") is not None else None,
                  "anchor_plus_scaffold_pts": round(anchor["est_pts"] + sc["pts"], 1),
                  "scaffold_share_of_gap": r3(sc["pts"] / total["est_pts"]) if total["est_pts"] else None}
    return out


def realized_vs_deep(oc: pd.DataFrame, N: dict, base_names, *, replicates: int, items) -> dict:
    """Each harness cell against the deep-call curve at the SAME realized completion tokens per item (F2): the deep
    accuracy log-linearly interpolated between the two adjacent deep rungs (by realized tokens), macro. The paired
    CI is the linear contrast ``cell - (1-w) deep_lo - w deep_hi`` through the same machinery (descriptive)."""
    f2 = N["frontier_f2"]
    deep = sorted([(f2[n]["x_tokens"], n) for n in f2 if n.startswith("deep/")])
    out: dict[str, Any] = {"cells": {}, "deep_points": [n for _, n in deep],
                           "deep_min_x": round(deep[0][0], 1) if deep else None}
    for n in base_names:
        if n not in f2:
            continue
        x, acc_h = f2[n]["x_tokens"], f2[n]["acc"]
        pair = next(((a, b) for a, b in zip(deep, deep[1:]) if a[0] <= x <= b[0]), None)
        if pair is None:
            out["cells"][n] = {"x_tokens": round(x, 1), "acc": acc_h, "in_range": False}
            continue
        (xa, na), (xb, nb) = pair
        w = math.log(x / xa) / math.log(xb / xa)
        interp = (1 - w) * f2[na]["acc"] + w * f2[nb]["acc"]
        d = _contrast_row(oc, f"{n}-deep@x", [(n, 1.0), (na, -(1 - w)), (nb, -w)], replicates=replicates, items=items)
        out["cells"][n] = {"x_tokens": round(x, 1), "acc": acc_h, "in_range": True, "deep_lo": na, "deep_hi": nb,
                           "w": round(w, 3), "deep_interp": r3(interp), "diff_pts": d["pts"], "lo": d["lo"],
                           "hi": d["hi"], "status": d["status"]}
    inr = {n: e for n, e in out["cells"].items() if e["in_range"]}
    out["n_in_range"] = len(inr)
    out["below_range"] = [n for n, e in out["cells"].items() if not e["in_range"]]
    out["n_below_range"] = len(out["below_range"])
    out["n_ci_below"] = sum(1 for e in inr.values() if e["hi"] < 0)
    out["n_ci_above"] = sum(1 for e in inr.values() if e["lo"] > 0)
    out["n_ci_covers"] = out["n_in_range"] - out["n_ci_below"] - out["n_ci_above"]
    out["ci_above_cells"] = [n for n, e in inr.items() if e["lo"] > 0]
    out["ci_above_first"] = out["ci_above_cells"][0] if out["ci_above_cells"] else "none"
    out["ci_below_cells"] = [n for n, e in inr.items() if e["hi"] < 0]
    har = {n: e for n, e in inr.items() if c.arm_of(n) != "H0"}
    out["harness_n"] = len(har)
    out["harness_ci_excludes_zero"] = sum(1 for e in har.values() if e["hi"] < 0 or e["lo"] > 0)
    out["harness_min_pts"] = min((e["diff_pts"] for e in har.values()), default=None)
    out["harness_max_pts"] = max((e["diff_pts"] for e in har.values()), default=None)
    out["by_arm"] = {}
    for a in sorted({c.arm_of(n) for n in inr}):
        v = [e["diff_pts"] for n, e in inr.items() if c.arm_of(n) == a]
        out["by_arm"][a] = {"mean_pts": round(float(np.mean(v)), 1), "min_pts": min(v), "max_pts": max(v),
                            "n": len(v)}
    return out


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC")


RUN_FACT_SOURCES = ("state/ledger_journal.jsonl", "state/run_decisions.jsonl", "superseded_executions/canary",
                    "gates/gate1_canary.json", "gates/gate2_pilot.json", "gates/gate4_main.json")


def run_facts(run: Path, runs_all: pd.DataFrame, current: pd.DataFrame, calls: pd.DataFrame, partition: str) -> dict:
    """Spend by stage from the ledger journal, the gate history and owner decisions, and the in-run defects, all
    read from the run root (read-only). A source that is absent (a synthetic test root) contributes what an empty
    source says (no ledger events: $0; no decisions: empty quotes and zero counts) and is listed in
    ``sources_missing``; the report build (``main``) refuses to proceed when any source is missing."""
    from decimal import Decimal
    out: dict[str, Any] = {"sources_missing": [src for src in RUN_FACT_SOURCES if not (run / src).exists()]}
    stages = ("canary", "pilot", "main")
    settled = {s: Decimal(0) for s in stages}
    cmax = {s: Decimal(0) for s in stages}
    ledger_path = run / "state" / "ledger_journal.jsonl"
    if ledger_path.exists():
        with open(ledger_path) as fh:
            for line in fh:
                e = json.loads(line)
                p = e.get("partition_id")
                if p not in settled or e.get("amount_usd") is None:
                    continue
                if e["kind"] == "settle":
                    settled[p] += Decimal(e["amount_usd"])
                elif e["kind"] == "charge_max":
                    cmax[p] += Decimal(e["amount_usd"])
    out["ledger"] = {s: {"settled_usd": round(float(settled[s]), 2), "charged_max_usd": round(float(cmax[s]), 2)}
                     for s in stages}
    out["study_settled_usd"] = round(float(sum(settled.values())), 2)
    out["study_upper_usd"] = round(float(sum(settled.values()) + sum(cmax.values())), 2)
    out["canary_pilot_settled_usd"] = round(float(settled["canary"] + settled["pilot"]), 2)
    out["canary_g1"] = {"records": 0, "settled_usd": 0.0, "spawns": 0}
    sup = sorted((run / "superseded_executions" / "canary").glob("g*/SUPERSEDED.json"))
    if sup:
        sd = json.loads(sup[0].read_text())
        out["canary_g1"] = {"records": int(sd["records"]), "settled_usd": round(float(Decimal(sd["ledger_settled_usd"])), 2),
                            "spawns": sum(int(json.loads(f.read_text()).get("spawns") or 0)
                                          for f in (sup[0].parent / "runs").rglob("*.json"))}
    # owner decisions and lead records
    dpath = run / "state" / "run_decisions.jsonl"
    dec = [json.loads(line) for line in open(dpath) if line.strip()] if dpath.exists() else []
    waiver = next((d for d in dec if d["kind"] == "owner_waiver" and d.get("gate") == "gate1"), {"text": ""})
    quoted = re.findall(r"'([^']+)'", waiver["text"])
    out["owner_waiver_quote"] = quoted[0] if quoted else waiver["text"]
    rest = waiver["text"].split("' - ", 1)
    out["owner_waiver_rationale"] = rest[1] if len(rest) == 2 else ""
    out["waived_criteria"] = {d["gate"]: d["criteria"] for d in dec if d["kind"] == "owner_waiver"}
    go = next((d for d in dec if d["kind"] == "owner_go"), {"text": ""})
    out["owner_go_quote"] = (re.findall(r"'([^']+)'", go["text"]) or [go["text"]])[0]
    starts = [d for d in dec if d["kind"] == "dynorch_run_start" and d["stage"] == partition]
    restart = next((d for d in dec if d["kind"] == "lead_restart" and d["stage"] == partition), None)
    out["first_launch"] = {"drained_after_min": 0.0, "max_usd": 0.0, "active_ceiling_usd": 0.0, "cells_kept": 0,
                           "restarted": restart is not None}
    if starts:
        out["first_launch"]["max_usd"] = float(starts[0]["max_usd"])
    if restart is not None and len(starts) >= 2:
        out["first_launch"]["drained_after_min"] = round(
            (_utc(restart["recorded_at"]) - _utc(starts[0]["recorded_at"])).total_seconds() / 60, 1)
        m = re.search(r"active ceiling ([\d.]+) USD", restart["reason"])
        out["first_launch"]["active_ceiling_usd"] = float(m.group(1)) if m else 0.0
        fin = pd.to_datetime(runs_all[runs_all["partition"] == partition]["finished_at"], utc=True)
        out["first_launch"]["cells_kept"] = int((fin < _utc(starts[1]["recorded_at"])).sum())
    inv = next((d for d in dec if d["kind"] == "infra_invalidation"), {"count": 0, "cells": []})
    out["infra_invalidation"] = {"cells": int(inv["count"]),
                                 "by_arm": {a: sum(1 for x in inv["cells"] if x["arm"] == a) for a in ("H1", "H3")}}
    # reruns and provider-cut calls (tables)
    main_all = runs_all[runs_all["partition"] == partition]
    old = main_all[~main_all["run_key"].isin(current["run_key"])]
    out["reruns_by_reason"] = {str(k): int(v) for k, v in old["rerun_reason"].value_counts().items()}
    out["reruns_transport"] = int((old["rerun_reason"] == "last_reply_transport_or_ledger_error").sum())
    out["reruns_transport_arms"] = sorted(old[old["rerun_reason"] == "last_reply_transport_or_ledger_error"]["arm"].unique().tolist())
    mc = calls[calls["partition"] == partition]
    out["main_calls"] = int(len(mc))
    out["main_failed_calls"] = int((mc["transport_status"] != "ok").sum())
    out["main_failed_envelopes"] = int((mc["transport_status"] == "error_envelope").sum())
    out["main_failed_timeouts"] = int((mc["transport_status"] == "ambiguous_timeout").sum())
    # the first Gate 4's grader re-run mismatches (contract 10.40): a valid <final> the grader cannot parse
    out["gate4_invalid_valid_final"] = int(((current["grade_status"] == "invalid_output")
                                            & current["valid_final"].astype(bool)).sum())
    # gates
    crit_of = {}
    for gname, fname in (("gate1", "gate1_canary.json"), ("gate2", "gate2_pilot.json"), ("gate4", "gate4_main.json")):
        p = run / "gates" / fname
        gd = json.loads(p.read_text()) if p.exists() else {}
        out[f"{gname}_decision"] = gd.get("decision", "")
        out[f"{gname}_passed"] = bool(gd.get("passed", False))
        crit_of[gname] = {x["name"]: x for x in gd.get("criteria", [])}
    ob = crit_of["gate1"].get("ceiling_probe_spawns", {}).get("observed")
    ob = json.loads(ob) if ob else {"of": 0, "spawning_runs": 0}
    out["gate1_probe"] = {"of": int(ob["of"]), "spawning": int(ob["spawning_runs"])}
    out["gate2_k16"] = {}
    for a in ("H1", "H3"):
        ob = crit_of["gate2"].get(f"{a}_spawn_share_k16", {}).get("observed")
        ob = json.loads(ob) if ob else {"runs_k16": 0, "spawning": 0}
        out["gate2_k16"][a] = {"runs": int(ob["runs_k16"]), "spawning": int(ob["spawning"])}
    gs = crit_of["gate4"].get("generation_sample_matches", {}).get("observed")
    out["gate4_generation_sampled"] = int(json.loads(gs)["sampled"]) if gs else 0
    out["core_generation2_cells"] = int((current["generation"] > 1).sum()) if "generation" in current else 0
    return out


def _reading(row) -> str | None:
    """Pre-registered reading rule (plan section 7) on the studentized CI of a deep/16 contrast."""
    if row["status"] != "complete" or row["ci95_lo"] is None or pd.isna(row["ci95_lo"]):
        return None
    if not str(row["contrast"]).endswith("deep/16"):
        return "above_baseline" if row["ci95_lo"] > 0 else ("below_baseline" if row["ci95_hi"] < 0 else "inconclusive")
    if row["ci95_hi"] < 0:
        return "loses_to_fixed_rule"
    if row["ci95_lo"] > 0:
        return "rule_incomplete"
    return "inconclusive"


def _rate(hits, misses) -> float | None:
    total = float(hits) + float(misses)
    return None if total == 0 else round(float(hits) / total, 3)


def _kappa(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(int).to_numpy(), b.astype(int).to_numpy()
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _pareto(front: pd.DataFrame, measure: str) -> list[str]:
    """Upper-left step frontier (accuracy up, cost down) over ladder + harness points; descriptive only."""
    known = front.dropna(subset=[measure, "acc"]).sort_values([measure, "acc"], ascending=[True, False])
    out, best = [], -1.0
    for _, r in known.iterrows():
        if r["acc"] > best:
            out.append(r["name"])
            best = r["acc"]
    return out


# ---------------------------------------------------------------------------------------------------- tables

def write_tables(N, F, acc, arms, hnames, base_names, best_non_deep, front, out):
    out = Path(out)
    core16 = [a for a in arms if F.has(f"{a}/16")]
    # headline: per benchmark + macro
    head = ["pass@1", "vote@16", "oracle@16", "private/16", "best non-deep", "deep/16"] + [f"{a}/16" for a in core16]
    lines = [r"\begin{tabular}{l" + "r" * len(head) + "}", r"\toprule",
             "Benchmark & " + " & ".join(h.replace("_", r"\_") for h in head) + r" \\", r"\midrule"]
    for lv in [*c.BENCH_ORDER, "macro"]:
        cells = [fmt(acc["sample/1"][lv]["acc"]), fmt(acc["sample/16"][lv]["acc"]), fmt(N["oracle"][lv]["16"]),
                 fmt(acc["private_single/16"][lv]["acc"]),
                 fmt(acc[best_non_deep[16]][lv]["acc"]) if best_non_deep[16] else "--", fmt(acc["deep/16"][lv]["acc"])]
        cells += [fmt(acc[f"{a}/16"][lv]["acc"]) for a in core16]
        lines.append(f"{c.BENCH_LABEL[lv]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "headline.tex").write_text("\n".join(lines) + "\n")

    # primary family (anchor first)
    lines = [r"\begin{tabular}{lrrrrl}", r"\toprule",
             r"Contrast (macro, 96 items) & estimate (pts) & 95\% CI (pts) & $p$ & Holm $p$ & reading \\ \midrule"]
    for name in [N["anchor"], *N["primary_family"]]:
        e = N["primary"].get(name)
        if e is None:
            continue
        ci = f"[{signed(e['lo_pts'])}, {signed(e['hi_pts'])}]" if e["lo_pts"] is not None else "--"
        lines.append(f"{tex_name(name)} & {signed(e['est_pts'])} & {ci} & {fmt(e['p'])} & {fmt(e['p_holm'])} & "
                     f"{(e['reading'] or e['status']).replace('_', ' ')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "primary.tex").write_text("\n".join(lines) + "\n")

    # every arm x rung, macro + per benchmark, with the descriptive deltas (csv)
    rows = []
    for n in hnames:
        row = {"name": n, "arm": c.arm_of(n), "rung": c.rung_of(n), "repeat": c.repeat_of(n),
               "completeness": N["completeness"][n]}
        for lv in c.LEVELS:
            row[f"acc_{lv}"] = acc[n][lv]["acc"]
            row[f"lo_{lv}"] = acc[n][lv]["lo"]
            row[f"hi_{lv}"] = acc[n][lv]["hi"]
        for tag, e in N["descriptive"][n].items():
            row[f"{tag}_pts"] = None if e is None else e["pts"]
            row[f"{tag}_other"] = None if e is None else e["other"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "tables" / "rung_table.csv", index=False)

    # realized budget
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"cell & $C$ & mean & median & realized share & grace & cap-bound & terminal \\ \midrule"]
    for n in base_names:
        e = N["realized"][n]
        if e is None:
            continue
        lines.append(f"{tex_name(n)} & {e['cap']:,} & {e['mean_completion']:,.0f} & {e['median_completion']:,.0f} & "
                     f"{fmt(e['realized_share'])} & {e['mean_grace']:.0f} & {fmt(e['share_cap_bound'])} & "
                     f"{fmt(e['share_terminal'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "realized.tex").write_text("\n".join(lines) + "\n")

    # diagnostics
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"cell & calls & root & solver & spawns & spawn share & orch.\ share & abstain & parse fail & wall s \\ \midrule"]
    for n in base_names:
        e = N["diagnostics"][n]
        if e is None:
            continue
        lines.append(f"{tex_name(n)} & {e['calls']:.1f} & {e['calls_root']:.1f} & {e['calls_solver']:.1f} & "
                     f"{e['spawns']:.2f} & {fmt(e['spawn_share'])} & {fmt(e['orch_share'])} & {fmt(e['abstention'])} & "
                     f"{fmt(e['parse_fail'])} & {e['wall_s']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "diagnostics.tex").write_text("\n".join(lines) + "\n")

    # provider sensitivity
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"contrast & Mancer items & Parasail items & same provider only \\ \midrule"]
    for name, e in N["provider_sensitivity"].items():
        cells = []
        for tag in ("mancer_items", "parasail_items", "same_provider_only"):
            s = e[tag]
            cells.append("--" if s["pts"] is None else f"{signed(s['pts'])} [{signed(s['lo'])}, {signed(s['hi'])}] ($n$={s['n']})")
        lines.append(f"{tex_name(name)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "provider_sensitivity.tex").write_text("\n".join(lines) + "\n")

    # cost / frontier
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"configuration & nominal $C$ & realized completion & prompt tokens & USD (charged) & macro accuracy \\ \midrule"]
    for _, r in front.sort_values(["side", "nominal", "name"]).iterrows():
        lines.append(f"{tex_name(r['name'])} & {int(r['nominal']):,} & {r['x_tokens']:,.0f} & {r['prompt_tokens']:,.0f} & "
                     f"{r['usd']:.4f} & {fmt(r['acc'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "cost.tex").write_text("\n".join(lines) + "\n")

    # each harness cell against the deep curve at the same realized tokens (F2, descriptive)
    rvd = N["realized_vs_deep"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"cell & realized tokens & accuracy & deep at same tokens & difference (pts) [95\% CI] \\ \midrule"]
    for n in base_names:
        e = rvd["cells"].get(n)
        if e is None or not e["in_range"]:
            continue
        lines.append(f"{tex_name(n)} & {e['x_tokens']:,.0f} & {fmt(e['acc'])} & {fmt(e['deep_interp'])} & "
                     f"{signed(e['diff_pts'])} [{signed(e['lo'])}, {signed(e['hi'])}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "realized_vs_deep.tex").write_text("\n".join(lines) + "\n")

    for name in ("headline", "realized", "diagnostics", "cost", "realized_vs_deep"):
        path = out / "tables" / f"{name}.tex"
        path.write_text(re.sub(r"(?<![\w$])-(\d)", r"$-$\1", path.read_text()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ladder-numbers", default=None)
    ap.add_argument("--panels", default=None)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--replicates", type=int, default=c.REPLICATES)
    a = ap.parse_args(argv)
    N = run_analysis(a.run, a.ladder, a.config, a.out, ladder_numbers=a.ladder_numbers, replicates=a.replicates,
                     panels=a.panels)
    missing = N["run_facts"]["sources_missing"] + (["raw/ (Codex second-call journals)"]
                                                    if N["behaviour"]["H3_second_call"]["unreadable"] else [])
    if missing:  # the report's run facts must come from a real run root
        raise SystemExit(f"run facts incomplete, missing under {a.run}: {missing}")
    print(json.dumps({k: N[k] for k in ("n_items", "n_runs_core", "n_partial_cells", "spend")}, indent=1))


if __name__ == "__main__":
    main()
