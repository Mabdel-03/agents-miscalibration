#!/usr/bin/env python
"""Aggregate a run's JSONL into a tidy table; flag missing SAS baselines loudly.

Usage:
  python scripts/aggregate_results.py --run-id run_1234 [--out results.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys

from agents_scaling.experiment.analyze import aggregate_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", help="write tidy parquet here (requires pandas/pyarrow)")
    args = ap.parse_args()

    records = aggregate_run(args.run_id)
    if not records:
        sys.exit(f"no cell results found for run {args.run_id!r}")

    missing = [r["cell_id"] for r in records if r["topology"] != "single_agent" and r.get("efficiency") is None]
    if missing:
        sys.exit(
            "FATAL: missing single-agent baseline for these cells (plan Risk 6):\n  "
            + "\n  ".join(missing)
        )

    print(f"[aggregate] {len(records)} cells")
    for r in records:
        cal = r["calibration"].get("per_agent")
        ece = f"{cal['ece']:.3f}" if cal else "n/a"
        print(f"  {r['cell_id']:55s} acc={r['accuracy']:.3f}  per-agent ECE={ece}")

    if args.out:
        import pandas as pd

        # Flatten nested dicts to JSON strings for a tidy single table.
        flat = []
        for r in records:
            row = {k: v for k, v in r.items() if k not in ("calibration", "efficiency", "prompt_quality")}
            row["calibration_json"] = json.dumps(r["calibration"])
            row["efficiency_json"] = json.dumps(r.get("efficiency"))
            row["prompt_quality_json"] = json.dumps(r["prompt_quality"])
            flat.append(row)
        pd.DataFrame(flat).to_parquet(args.out)
        print(f"[aggregate] wrote {args.out}")


if __name__ == "__main__":
    main()
