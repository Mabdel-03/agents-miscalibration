#!/usr/bin/env python
"""Summarize an aggregated agents_scaling parquet table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json_column(df: pd.DataFrame, name: str) -> list[dict]:
    if name not in df.columns:
        raise SystemExit(f"missing column {name!r}; run scripts/aggregate_results.py first")
    out: list[dict] = []
    for raw in df[name].fillna("{}"):
        if isinstance(raw, dict):
            out.append(raw)
        else:
            out.append(json.loads(raw) if raw else {})
    return out


def _per_agent_ece(cal: dict) -> float | None:
    report = cal.get("per_agent") if isinstance(cal, dict) else None
    return None if not report else float(report.get("ece", 0.0))


def _system_ece(cal: dict, key: str) -> float | None:
    system = cal.get("system") if isinstance(cal, dict) else None
    if not system or key not in system or system[key] is None:
        return None
    return float(system[key].get("ece", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="path produced by aggregate_results.py")
    parser.add_argument("--system-conf", default="final_producer_logprob")
    args = parser.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(f"missing parquet file: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise SystemExit(f"parquet file has no rows: {path}")

    cals = _load_json_column(df, "calibration_json")
    df = df.copy()
    df["per_agent_ece"] = [_per_agent_ece(c) for c in cals]
    df["system_ece"] = [_system_ece(c, args.system_conf) for c in cals]
    df["delta_ece"] = df["system_ece"] - df["per_agent_ece"]

    print(f"file: {path}")
    print(f"cells: {len(df)}")
    print(f"questions: {int(df['n_questions'].sum()) if 'n_questions' in df else 'n/a'}")
    print(f"mean accuracy: {df['accuracy'].mean():.3f}")
    print(f"mean per-agent ECE: {df['per_agent_ece'].mean():.3f}")
    if df["system_ece"].notna().any():
        print(f"mean system ECE ({args.system_conf}): {df['system_ece'].mean():.3f}")
        print(f"mean Delta ECE: {df['delta_ece'].mean():+.3f}")
    else:
        print(f"system ECE ({args.system_conf}) unavailable in this table")

    if "topology" in df.columns:
        print("\nsummary by topology")
        cols = ["accuracy", "per_agent_ece", "system_ece", "delta_ece"]
        print(df.groupby("topology")[cols].mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
