"""One flat CSV of the whole study: a row per item x configuration x view.

    export_flat.py [--run <artifacts dir>] [--out <path.csv>] [--generations <path.csv>]

Joins the sealed merged tables on ``policy_id``: the delivered outcome and pool (policy_outcomes), the cost
and token ledger (policy_costs), effective depth (effective_depth), and the item's own reference numbers from
the 25-sample bank (difficulty, and pass@k / prefix coverage / vote readout at that configuration's own
k = C/A). Every row carries the design coordinates (family, B, R, A, C), the benchmark and the provider.

``--generations`` additionally writes a per-generation CSV (one row per API node: budget, realized reasoning
tokens, loop onset, grade, cost, provider).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

A0 = 1024
OUT_COLS = [
    # keys and design
    "policy_id", "partition_id", "view", "item_id", "benchmark", "subtask", "stratum", "wave",
    "name", "family", "B_agents", "R_cycles", "peer_rounds", "A_reasoning_budget", "C_compute", "C_over_A",
    "shard_id", "endpoint_tag",
    # what it delivered
    "delivered_correct", "delivered_state", "selection_status", "selector_name", "selected_candidate_id",
    # what it generated
    "n_pool", "n_pool_valid", "n_pool_correct", "n_pool_unknown", "coverage", "selection_gap",
    # tokens and money
    "nominal_generations", "requested_reasoning_tokens", "reasoning_tokens", "answer_tokens", "prompt_tokens",
    "completion_tokens", "realized_tokens", "total_charged_usd", "standardized_uncached_cost_usd",
    "total_latency_seconds", "critical_path_latency_seconds",
    # effective depth (forced runs)
    "effective_tokens", "effective_fraction", "looped_generations", "generations_with_text",
    # the item's own reference points, from the 25-sample bank at A
    "bank_size", "bank_correct", "pass_at_k_same_C", "prefix_coverage_same_C", "vote_readout_same_C",
    "mean_draw_correct", "pass_at_25", "bank_zero",
]


def build(run: str) -> pd.DataFrame:
    T = lambda n: pd.read_parquet(os.path.join(run, "merged", "tables", n + ".parquet"))  # noqa: E731
    po, pc, ed, sb = T("policy_outcomes"), T("policy_costs"), T("effective_depth"), T("sample_bank_items")

    cost_cols = ["policy_id", "nominal_generations", "requested_reasoning_tokens", "prompt_tokens",
                 "completion_tokens", "reasoning_tokens", "answer_tokens", "realized_tokens", "total_charged_usd",
                 "standardized_uncached_cost_usd", "total_latency_seconds", "critical_path_latency_seconds"]
    df = po.merge(pc[cost_cols], on="policy_id", how="left")
    df = df.merge(ed[["policy_id", "effective_tokens", "effective_fraction", "looped_generations", "generations"]]
                  .rename(columns={"generations": "generations_with_text"}), on="policy_id", how="left")

    df["B_agents"] = df["team_size"].astype(int)
    df["R_cycles"] = df["total_cycles"].astype(int)
    df["A_reasoning_budget"] = df["reasoning_budget"].astype(int)
    df["C_compute"] = df["B_agents"] * df["R_cycles"] * df["A_reasoning_budget"]
    df["C_over_A"] = df["C_compute"] // A0
    df["delivered_correct"] = df["y"]
    df["selection_gap"] = df["coverage"] - df["y"]

    # the item's reference numbers at its own k = C/A, and the full-bank difficulty
    bank = sb[["partition_id", "item_id", "k", "bank_size", "bank_correct", "oracle_pass_at_k", "prefix_coverage",
               "readout", "mean_draw_correct"]].rename(columns={
                   "oracle_pass_at_k": "pass_at_k_same_C", "prefix_coverage": "prefix_coverage_same_C",
                   "readout": "vote_readout_same_C"})
    df = df.merge(bank, left_on=["partition_id", "item_id", "C_over_A"], right_on=["partition_id", "item_id", "k"],
                  how="left").drop(columns=["k"])
    b25 = (sb[sb["k"] == 25][["partition_id", "item_id", "oracle_pass_at_k", "bank_correct"]]
           .rename(columns={"oracle_pass_at_k": "pass_at_25", "bank_correct": "_bc25"}))
    df = df.merge(b25, on=["partition_id", "item_id"], how="left")
    df["bank_zero"] = (df["_bc25"] == 0)
    df["bank_correct"] = df["bank_correct"].fillna(df["_bc25"])
    df = df.drop(columns=["_bc25"])
    return df[[c for c in OUT_COLS if c in df.columns]].sort_values(
        ["partition_id", "benchmark", "item_id", "C_over_A", "family", "name", "view"]).reset_index(drop=True)


def generations(run: str) -> pd.DataFrame:
    T = lambda n: pd.read_parquet(os.path.join(run, "merged", "tables", n + ".parquet"))  # noqa: E731
    cand, ed = T("candidates"), T("effective_depth_nodes")
    keep = ["partition_id", "node_id", "item_id", "benchmark", "stratum", "kind", "root_slot", "team_size",
            "update_round", "reasoning_budget", "completion_headroom", "transport_status", "completion_status",
            "finish_reason", "parse_status", "final_answer_present", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "actual_cost_usd", "grade_status", "grade_state", "correct", "grade_detail_code",
            "latency_seconds", "observed_provider", "shard_id", "endpoint_tag"]
    df = cand[[c for c in keep if c in cand.columns]].merge(
        ed[["node_id", "reasoning_chars", "looped", "loop_onset_fraction", "effective_tokens", "text_available"]],
        on="node_id", how="left")
    df["answer_tokens"] = df["completion_tokens"] - df["reasoning_tokens"]
    df["realized_equals_budget"] = df["reasoning_tokens"] == df["reasoning_budget"]
    return df.sort_values(["partition_id", "benchmark", "item_id", "kind", "reasoning_budget", "root_slot",
                           "update_round"]).reset_index(drop=True)


DICT = """column,meaning
policy_id,"unique key: partition/item/configuration/view"
partition_id,"main (the 96-item study panel) or pilot (32 development items)"
view,"current = the candidates the selector actually chose among; archive = every candidate the configuration ever generated"
item_id,"benchmark item"
name,"configuration, e.g. sample/16, private_single/16, peer/4/3, deep/16"
family,"sample | private_single | private_team | peer | deep | cp | pooled"
B_agents,"B: agents (deep and private_single = 1; sample/k = k)"
R_cycles,"R: total generations per agent including the first (1 for sample and deep)"
A_reasoning_budget,"A: forced reasoning tokens per call (1024; for deep/m it is m*1024 in one call)"
C_compute,"B * R * A: total forced reasoning tokens, both requested and realized in this study"
C_over_A,"C / 1024: the budget in units of one short call"
delivered_correct,"1/0: the configuration's delivered answer was correct (blank = grade unknown)"
coverage,"1/0: this view's pool contained a correct answer"
selection_gap,"coverage - delivered_correct: what the pool held but the selector did not deliver"
n_pool,"candidates in this view's pool"
n_pool_correct,"correct candidates in the pool"
nominal_generations,"API calls charged to this configuration (deep = 1)"
requested_reasoning_tokens,"sum of the reasoning budgets of the configuration's own calls (= C_compute)"
reasoning_tokens,"realized reasoning tokens (equals requested: every call was forced)"
answer_tokens,"visible answer tokens (the second, unforced call of each node)"
total_charged_usd,"cost of this configuration for this item"
effective_tokens,"reasoning tokens before the onset of degenerate repetition (all of them if none)"
effective_fraction,"effective_tokens / requested_reasoning_tokens"
looped_generations,"how many of the configuration's calls fell into repetition"
bank_correct,"how many of the item's 25 independent 1024-token samples were correct (difficulty)"
bank_zero,"True when none of the 25 samples was correct: unreachable by any selector over samples"
pass_at_k_same_C,"pass@k of the 25-sample bank at k = C_over_A: the generation ceiling of spending the same compute on independent short samples"
prefix_coverage_same_C,"realized coverage of the first k samples"
vote_readout_same_C,"what a plurality vote over those k samples delivered"
pass_at_25,"pass@25 of the full bank"
shard_id,"provider that served this item (mancer or parasail); items are disjoint between providers"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--generations", default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "data", "forced_budget_v2_item_config.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = build(a.run)
    df.to_csv(out, index=False)
    open(os.path.join(os.path.dirname(out), "DATA_DICTIONARY.csv"), "w").write(DICT)
    print(f"{out}: {len(df):,} rows x {len(df.columns)} columns")
    gen = a.generations or os.path.join(os.path.dirname(out), "forced_budget_v2_generations.csv")
    g = generations(a.run)
    g.to_csv(gen, index=False)
    print(f"{gen}: {len(g):,} rows x {len(g.columns)} columns")


if __name__ == "__main__":
    main()
