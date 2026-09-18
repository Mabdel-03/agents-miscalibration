#!/usr/bin/env python
"""Iso-compute ladder of a forced-budget run (openrouter_forced_budget_v2): requested C, realized compute and effective
depth per checkpoint, from the sealed (or merged-shard) tables only.

Writes ``tables/forced_ladder.{csv,tex}``, ``tables/effective_depth.{csv,tex}`` and ``numbers_forced.json``. The v1
report (``isocompute.py`` / ``report.tex``) quotes v1 numbers and is not rebuilt for a forced run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import isocore as ic


def fmt(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def tex_table(frame, cols, headers, path):
    with open(path, "w") as fh:
        fh.write("\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(headers) + " \\\\\n\\midrule\n")
        for _, row in frame.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, str):
                    cells.append("\\texttt{" + v.replace("_", "\\_") + "}")
                elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                    cells.append("n/a")
                elif isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                else:
                    cells.append(f"{float(v):.3f}")
            fh.write(" & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("RUN_DIR"), required=os.environ.get("RUN_DIR") is None,
                    help="a forced run dir (or <run_root>/merged) holding tables/")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--partition", default="main")
    args = ap.parse_args()
    run, out, part = args.run_dir, args.out, args.partition
    os.makedirs(f"{out}/tables", exist_ok=True)

    fails = ic.self_check(run, part)
    if fails:
        print("SELF-CHECK FAILED"); [print("  ", f) for f in fails]; raise SystemExit(1)
    print("self-check: C = B*R*A matches the sealed requested_reasoning_tokens")

    ladder = ic.realized_compute(run, part)
    if "realized_over_requested" in ladder:
        off = ladder[(ladder["realized_over_requested"] - 1).abs() > 1e-9]
        if len(off):
            print(f"WARNING: realized reasoning != requested C for {len(off)} checkpoint(s): "
                  + ", ".join(off["name"].head(8)))
    ladder.to_csv(f"{out}/tables/forced_ladder.csv", index=False)
    cols = [c for c in ("name", "rung", "C", "mean_realized_tokens", "acc") if c in ladder]
    tex_table(ladder[cols], cols, ["checkpoint", "$C/A_0$", "$C$", "realized tokens", "macro acc"][:len(cols)],
              f"{out}/tables/forced_ladder.tex")

    numbers = {"deep_rungs_present": sorted(int(r) for r in ladder.loc[ladder["family"] == "deep", "rung"].unique())}
    if ic.has_table(run, "effective_depth_summary"):
        eff = ic.effective_depth(run, part)
        eff.to_csv(f"{out}/tables/effective_depth.csv", index=False)
        tex_table(eff[["name", "rung", "C", "mean_effective_tokens", "mean_effective_fraction"]],
                  ["name", "rung", "C", "mean_effective_tokens", "mean_effective_fraction"],
                  ["checkpoint", "$C/A_0$", "$C$", "effective tokens", "effective / $C$"],
                  f"{out}/tables/effective_depth.tex")
        deep = eff[eff["family"] == "deep"].set_index("name")
        numbers["deep_effective_fraction"] = {n: fmt(deep.loc[n, "mean_effective_fraction"]) for n in deep.index}
        numbers["deep_effective_rung"] = {n: fmt(deep.loc[n, "effective_rung"]) for n in deep.index}
    else:
        print("no effective_depth_summary table (run `allocation analyze` / `merge-shards` on a forced run)")
    with open(f"{out}/numbers_forced.json", "w") as fh:
        json.dump(numbers, fh, indent=1, sort_keys=True)
    print(f"forced ladder: {len(ladder)} checkpoints; tables in {out}/tables")


if __name__ == "__main__":
    main()
