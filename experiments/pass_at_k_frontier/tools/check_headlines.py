#!/usr/bin/env python3
"""Recompute this investigation's headline numbers from the tables in this folder.

This is also the record of which table and filter produces each number: read the entry to see how to derive it.
A failure means the data and the documented value disagree. Needs pandas and pyarrow.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1, S3, S4, S5 = "01_allocation_v1", "03_forced_budget_v2", "04_dynorch_v1", "05_passk_cont_v1"
PACKAGE = f"{S5}/results_package/csv"
MIXTURE = {"aime": 0.10, "hmmt": 0.10, "omni_math_hard": 0.50, "gpqa_diamond": 0.30}
_cache: dict = {}


def load(path):
    if path not in _cache:
        full = os.path.join(ROOT, path)
        _cache[path] = pd.read_parquet(full) if path.endswith(".parquet") else pd.read_csv(full)
    return _cache[path]


def macro(study, name, table="policy_outcomes", partition="main"):
    """Mean of y within each benchmark, then an equal-weight mean across benchmarks."""
    df = load(f"{study}/tables/{table}.parquet")
    rows = df[(df["view"] == "current") & (df["partition_id"] == partition) & (df["name"] == name)]
    return rows.groupby("benchmark")["y"].mean().mean()


def oracle(study, k):
    df = load(f"{study}/tables/pass_at_k_vs_readout.csv")
    return df[(df["partition_id"] == "main") & (df["level"] == "macro") & (df["k"] == k)]["oracle_pass_at_k"].iloc[0]


def reasoning(stat, study=S1, name="deep/16"):
    df = load(f"{study}/tables/policy_costs.parquet")
    rows = df[(df["view"] == "current") & (df["partition_id"] == "main") & (df["name"] == name)]
    return getattr(rows["reasoning_tokens"], stat)()


def mean_cost(study, name):
    df = load(f"{study}/tables/policy_costs.parquet")
    rows = df[(df["view"] == "current") & (df["partition_id"] == "main") & (df["name"] == name)]
    return rows["total_charged_usd"].astype(float).mean()  # *_usd columns are decimal strings


def primary(study, contrast, column):
    df = load(f"{study}/tables/primary_contrasts.csv")
    return df[(df["contrast"] == contrast) & (df["level"] == "macro")][column].iloc[0]


def forced_exact():
    df = load(f"{S3}/tables/forced_budget_check.csv")
    main = df[df["partition_id"] == "main"]
    return main["exact"].sum() if main["exact"].sum() == main["generations"].sum() else -1


def unknown_current():
    df = load(f"{S3}/tables/policy_outcomes.parquet")
    return int(df[(df["view"] == "current") & df["y"].isna()].shape[0])


def spawned(partition, arms):
    """status == "done" leaves exactly one row per cell; retried attempts are transport_failed rows."""
    df = load(f"{S4}/tables/dynorch_runs.parquet")
    rows = df[(df["partition"] == partition) & (df["status"] == "done") & df["arm"].isin(arms)]
    return int((rows["spawns"] > 0).sum()), len(rows)


def t200(config):
    """T200 delivered accuracy under the pre-registered mixture weights."""
    df = load(f"{PACKAGE}/03_pools/pool_outcomes.csv")
    rows = df[(df["panel"] == "T200") & (df["config_id"] == config)]
    means = rows["delivered_correct"].astype(float).groupby(rows["benchmark"]).mean()
    return sum(MIXTURE[b] * means[b] for b in MIXTURE)


def test(name, column):
    df = load(f"{PACKAGE}/04_e1/primary_tests.csv")
    return df[df["test"] == name][column].iloc[0]


CHECKS = [
    # study, headline, documented value, tolerance, computed from
    (1, "sample/25 macro", 0.875, 5e-4, lambda: macro(S1, "sample/25")),
    (1, "oracle pass@25 macro, main panel", 0.9375, 5e-4, lambda: oracle(S1, 25)),
    (1, "deep/16 median reasoning tokens (allowance 16,384)", 1187, 0.5, lambda: reasoning("median")),
    (1, "deep/16 mean reasoning tokens", 2774, 0.5, lambda: reasoning("mean")),
    (1, "peer/4/4 - private_team/4/4", -0.025, 5e-4, lambda: primary(S1, "primary_i_peer_vs_private_team_n4_r4", "estimate")),
    (1, "  its Holm p", 0.027, 5e-4, lambda: primary(S1, "primary_i_peer_vs_private_team_n4_r4", "p_holm")),
    (3, "deep/16 macro", 0.792, 5e-4, lambda: macro(S3, "deep/16")),
    (3, "sample/16 macro", 0.417, 5e-4, lambda: macro(S3, "sample/16")),
    (3, "deep/4 macro", 0.610, 5e-4, lambda: macro(S3, "deep/4")),
    (3, "sample/25 macro", 0.427, 5e-4, lambda: macro(S3, "sample/25")),
    (3, "oracle pass@16 macro", 0.580, 5e-4, lambda: oracle(S3, 16)),
    (3, "main generations at exactly their budget", 13344, 0.5, forced_exact),
    (3, "current rows with unknown outcome", 49, 0.5, unknown_current),
    (3, "deep/16 mean cost per item, USD", 0.0443, 5e-5, lambda: mean_cost(S3, "deep/16")),
    (4, "H1/16 macro (OpenClaude)", 0.667, 5e-4, lambda: macro(S4, "H1/16", "policy_outcomes_dynorch")),
    (4, "H3/16 macro (Codex)", 0.646, 5e-4, lambda: macro(S4, "H3/16", "policy_outcomes_dynorch")),
    (4, "H0/16 macro (plain capped call)", 0.719, 5e-4, lambda: macro(S4, "H0/16", "policy_outcomes_dynorch")),
    (4, "core harness runs", 1152, 0.5, lambda: spawned("main", ["H1", "H3"])[1]),
    (4, "  of which spawned a sub-agent", 0, 0.5, lambda: spawned("main", ["H1", "H3"])[0]),
    (4, "E0 capped-root runs that spawned", 41, 0.5, lambda: spawned("ext_e0", ["H1c", "H3c"])[0]),
    (4, "E1 second-repeat runs that spawned", 1, 0.5, lambda: spawned("ext_e1", ["H1", "H3"])[0]),
    (5, "T200 1x16, mixture", 0.705, 5e-4, lambda: t200("1x16")),
    (5, "T200 4x4, mixture", 0.580, 5e-4, lambda: t200("4x4")),
    (5, "4x4 - 1x16 (primary)", -0.125, 5e-4, lambda: test("mixed_vs_deep", "estimate")),
    (5, "  its Holm p", 0.0008, 5e-5, lambda: test("mixed_vs_deep", "p_holm")),
    (5, "P2 - P1 item Brier (primary)", 0.009, 5e-4, lambda: test("p2_vs_p1", "estimate")),
    (5, "  its Holm p", 0.034, 5e-4, lambda: test("p2_vs_p1", "p_holm")),
]


def main():
    failures = 0
    for study, label, expected, tolerance, compute in CHECKS:
        value = compute()
        ok = abs(float(value) - expected) <= tolerance
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} study {study}  {label:52} {float(value):>12.4f}  documented {expected}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} headlines reproduce from the tables")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
