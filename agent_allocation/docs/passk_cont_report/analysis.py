"""Build numbers.json, tables/*.tex and figures/*.pdf for the passk_cont_v1 report from the run root.

Slots (spec §18): F1 allocation frontier, F2 predictive depth response (WP-M ``sealed/v48`` + ``scored/``),
F3 bottleneck decomposition (E0 plots + scored pools), F6 prospective prediction (WP-M), T1 audit/cohorts,
T2 costs, T3 primary tests. A missing input leaves its keys absent, so render.py fails loudly on any placeholder
that depends on it (the report never ships a hand-typed or stale number).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
B16 = ["16x1", "8x2", "4x4", "2x8", "1x16"]
B8 = ["8x1", "4x2", "2x4", "1x8"]


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def numbers(run_root: Path) -> dict:
    N: dict = {"study": {"id": "openrouter_passk_cont_v1", "model": "qwen/qwen3.8-27b", "A": 1024}}
    audit = _json(run_root / "e0" / "audit.json")
    if audit:
        N["e0"] = {"cohorts": audit["cohorts"], "replay": audit["replay"], "n_candidates": audit["n_candidates"],
                   "headlines": {k.split(" ")[0]: v for k, v in audit["headlines"].items()},
                   "model_pinned": str(audit["observed_model_all_pinned"])}
    e1 = _json(run_root / "sealed" / "e1_selection.json")
    if e1:
        N["e1"] = {"locked": e1["selected"], "table": e1["table"]}
    primary = _json(run_root / "scored" / "primary.json")
    if primary:
        N["primary"] = primary
    draws_path = run_root / "analysis_inputs" / "draws.parquet"
    if draws_path.exists():                                                   # T2 physical ledger
        from agent_allocation.passk.costs import physical_ledger, physical_summary
        ledger = physical_ledger(pd.read_parquet(draws_path))
        by = physical_summary(ledger, by=("source", "partition"))
        N["costs"] = {"total_usd": float(ledger["cost_usd"].sum()), "requests": int(len(ledger)),
                      "by_partition": {f"{r.source}/{r.partition}": {"usd": float(r.cost_usd or 0),
                                                                     "requests": int(r.requests)}
                                       for r in by.itertuples()}}
    menus = run_root / "scored" / "menus.csv"
    if menus.exists():
        m = pd.read_csv(menus)
        N["menus"] = {panel: {row.config_id: {"S_macro": row.S_known_macro, "H_macro": row.H_known_macro,
                                              "S_mixture": row.S_known_mixture, "n": int(row.n_items)}
                              for row in sub.itertuples()} for panel, sub in m.groupby("panel")}
    return N


def figures(run_root: Path) -> None:
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    for name in ("decomposition_sample_k.png", "decomposition_deep_m.png", "decomposition_b16_by_benchmark.png"):
        src = run_root / "e0" / name
        if src.exists():
            shutil.copy(src, out / name)                                   # F3 (legacy)
    menus = run_root / "scored" / "menus.csv"
    if not menus.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = pd.read_csv(menus)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)                 # F1 allocation frontier
    for ax, menu, title in ((axes[0], B16, "B = 16A"), (axes[1], B8, "B = 8A")):
        for panel, marker in (("L96", "o"), ("V48", "s"), ("T200", "^")):
            sub = m[(m.panel == panel) & m.config_id.isin(menu)].set_index("config_id").reindex(menu)
            ax.plot(menu, sub["S_known_macro"], marker=marker, label=f"{panel} S")
            ax.plot(menu, sub["H_known_macro"], marker=marker, linestyle=":", alpha=0.6, label=f"{panel} H")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "f1_frontier.pdf")
    plt.close(fig)


def tables(N: dict) -> None:
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    if "e0" in N:                                                            # T1 audit / cohorts
        c = N["e0"]["cohorts"]
        (out / "t1_cohorts.tex").write_text(
            "\\begin{tabular}{lrrr}\\toprule cohort & 96 & 94 & 93\\\\\\midrule items & "
            f"{c['96']} & {c['94']} & {c['93']}\\\\\\bottomrule\\end{{tabular}}\n")
    if "costs" in N:                                                         # T2 costs (physical)
        esc = "\\_"
        rows = [k.replace("_", esc) + f" & {v['requests']:,} & {v['usd']:.2f}\\\\"
                for k, v in sorted(N["costs"]["by_partition"].items())]
        (out / "t2_costs.tex").write_text(
            "\\begin{tabular}{lrr}\\toprule source/partition & requests & USD\\\\\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\\end{tabular}\n")
    if "primary" in N and N["primary"].get("holm"):                          # T3 primary tests
        p = N["primary"]
        rows = []
        for key, label in (("mixed_vs_deep", f"{p['locked']} vs 1x16 (T200, mixture)"),
                           ("p2_vs_p1", "P2 vs P1 Brier (T200, mixture)")):
            r = p[key]
            lo, hi = r["ci95_studentized"]
            rows.append(f"{label} & {r['estimate']:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {r['p']:.4f} & "
                        f"{p['holm'][key]['p_holm']:.4f}\\\\")
        (out / "t3_primary.tex").write_text(
            "\\begin{tabular}{lrrrr}\\toprule contrast & effect & 95\\% CI & p & Holm p\\\\\\midrule\n"
            + "\n".join(rows).replace("-", "$-$") + "\n\\bottomrule\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    N = numbers(run_root)
    (HERE / "numbers.json").write_text(json.dumps(N, indent=2, sort_keys=True, default=float) + "\n")
    figures(run_root)
    tables(N)
    print(f"numbers.json: {sorted(N)}")


if __name__ == "__main__":
    main()
