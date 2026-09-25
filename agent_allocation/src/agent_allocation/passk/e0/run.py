"""E0 audit outputs under ``<run_root>/e0/`` (plan "Rest of E0"; spec §5 E0 outputs).

Writes ``audit.json``, ``provenance.parquet`` (one row per HTTP call: THINK and ANSWER of every legacy bank node),
``candidate_catalog.parquet`` (candidates + logical memberships from ``manifests/policies/*.jsonl``),
``exclusions.csv`` (96/94/93 cohorts with reasons), ``legacy_metrics.csv`` (every headline under 96-macro, 94-macro
and 96-micro from one library), ``legacy_replay.parquet`` + ``legacy_replay_drift.csv``, ``reuse_manifest.json``
and three decomposition plots. Read-only on the legacy run root.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent_allocation.passk.e0.replay import replay
from agent_allocation.passk.extract import LEGACY_ROOT, SHARDS, legacy_draws, load_draws
from agent_allocation.passk.metrics import (BENCH_ORDER, MACRO_WEIGHTS, MIXTURE_WEIGHTS, config_summary,
                                            pool_metrics, weighted_mean)
from agent_allocation.passk.pairs import PairCache
from agent_allocation.passk.pools import build_pools
from agent_allocation.passk.readout import readout_pools

__all__ = ["cohorts", "run_e0"]

MODEL = "qwen/qwen3.8-27b"
PROVIDER_PIN = {"mancer": "mancer/fp8", "parasail": "parasail/fp8"}
LEGACY_CONFIGS = ("1x1", "2x1", "4x1", "8x1", "16x1", "25x1", "1x2", "1x4", "1x8", "1x16", "1x25")
HEADLINES = {"deep/16 (79.2 / 78.5)": "1x16", "vote@25 (42.7 / 43.3)": "25x1", "sample/16 vote (41.7)": "16x1"}


def cohorts(legacy_root: Path, draws: pd.DataFrame) -> pd.DataFrame:
    """96 = all L96; 94 = drop items with an unknown grade in the root/deep bank; 93 = drop any unknown grade in
    the legacy main partition (all families, from ``merged/tables/candidates``)."""
    main = draws[draws["partition"] == "legacy_main"]
    items = sorted(main["item_id"].unique())
    bench = main.drop_duplicates("item_id").set_index("item_id")["benchmark"]
    bank_unknown = main[main["grade_state"] == "unknown"].groupby("item_id").size()
    cands = pd.read_parquet(Path(legacy_root) / "merged" / "tables" / "candidates.parquet",
                            columns=["partition_id", "item_id", "correct", "grade_state", "grade_detail_code"])
    cands = cands[(cands["partition_id"] == "main") & cands["correct"].isna()
                  & ~cands["grade_state"].astype(str).str.startswith("op_")]
    any_unknown = cands.groupby("item_id").size()
    codes = cands.groupby("item_id")["grade_detail_code"].agg(lambda s: ",".join(sorted(set(map(str, s)))))
    rows = []
    for item in items:
        nb, na = int(bank_unknown.get(item, 0)), int(any_unknown.get(item, 0))
        rows.append({"item_id": item, "benchmark": bench[item], "in_96": True, "in_94": nb == 0, "in_93": na == 0,
                     "bank_unknown_grades": nb, "main_unknown_grades": na,
                     "unknown_codes": codes.get(item, ""),
                     "reason": "" if na == 0 else (
                         f"{nb} unknown bank grade(s); {na} unknown main-partition grade(s) ({codes.get(item, '')})")})
    return pd.DataFrame(rows)


def _provenance(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in draws.itertuples(index=False):
        for purpose in ("think", "answer"):
            rows.append({"candidate_id": r.candidate_id, "purpose": purpose, "item_id": r.item_id,
                         "partition": r.partition, "shard": r.shard, "kind": r.kind, "allowance_a": r.allowance_a,
                         "replicate": r.replicate,
                         "request_sha": r.think_body_sha if purpose == "think" else r.answer_body_sha,
                         "generation_id": r.generation_id_think if purpose == "think" else r.generation_id_answer,
                         "provider": r.observed_provider, "provider_only": r.provider_only,
                         "served_model": r.observed_model, "status": "absent" if r.absent else "completed",
                         "tokens_in": r.think_in if purpose == "think" else r.answer_in,
                         "tokens_out": r.think_out if purpose == "think" else r.answer_out,
                         "cost_usd_node": r.cost_usd if purpose == "think" else None,
                         "latency_s_node": r.latency_s, "raw_path": r.raw_response_path,
                         "raw_sha256": r.raw_response_sha256})
    return pd.DataFrame(rows)


def _memberships(legacy_root: Path) -> dict[str, list[str]]:
    member: dict[str, list[str]] = {}
    for shard in SHARDS:
        pdir = Path(legacy_root) / "shards" / shard / "manifests" / "policies"
        for path in sorted(pdir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        cp = json.loads(line)
                        for cid in cp["candidate_ids"]:
                            member.setdefault(cid, []).append(cp["policy_id"])
    return member


def _reuse_manifest(draws: pd.DataFrame) -> dict:
    keys = ["observed_model", "provider_only", "item_id", "think_body_sha", "prompt_version", "decoding_config_hash",
            "allowance_a", "completion_headroom", "grader_version"]
    entries = []
    for r in draws[~draws["absent"].astype(bool)].itertuples(index=False):
        entries.append({"candidate_id": r.candidate_id, "model": r.observed_model,
                        "permaslug": "qwen/qwen3.8-27b-20260814 (verified on sampled /generation records)",
                        "provider_tag": r.provider_only, "item_id": r.item_id, "think_body_sha": r.think_body_sha,
                        "prompt_version": r.prompt_version, "decoding_config_hash": r.decoding_config_hash,
                        "allowance_a": int(r.allowance_a), "headroom": r.completion_headroom,
                        "stopping": "forced_think_exact", "parser_grader": r.grader_version,
                        "split_role": ",".join(list(r.panel_roles or [])), "partition": r.partition,
                        "reusable_as": ("L96 A slot r%d" % r.replicate if r.kind == "root" and r.partition
                                        == "legacy_main" else f"{r.partition} {r.kind} {r.allowance_a}A r{r.replicate}")})
    return {"version": "passk-reuse-v1", "key_fields": keys, "not_keyed_on": ["endpoint_lock_hash"],
            "n": len(entries), "entries": entries}


def _plots(metrics: pd.DataFrame, out_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    known = metrics.dropna(subset=["H_known", "S_known"])

    def stacked(ax, frame, order, title):
        g = frame.groupby(["config_id", "benchmark"])[["gen_fail", "sel_fail", "S_known"]].mean()
        vals = {c: {k: weighted_mean(g.loc[c][k].to_dict(), MACRO_WEIGHTS) for k in g.columns}
                for c in order if c in g.index.get_level_values(0)}
        xs = list(vals)
        gen = [vals[c]["gen_fail"] for c in xs]
        sel = [vals[c]["sel_fail"] for c in xs]
        acc = [vals[c]["S_known"] for c in xs]
        ax.bar(xs, acc, color="#2a7ab0", label="delivered correct S")
        ax.bar(xs, sel, bottom=acc, color="#e0a030", label="selection failure H−S")
        ax.bar(xs, gen, bottom=[a + s for a, s in zip(acc, sel)], color="#b0b0b0", label="generation failure 1−H")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    stacked(ax, known, ["1x1", "2x1", "4x1", "8x1", "16x1", "25x1"], "L96 sample/k: generation vs selection (macro)")
    path = out_dir / "decomposition_sample_k.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    stacked(ax, known, ["1x1", "1x2", "1x4", "1x8", "1x16", "1x25"], "L96 deep/m: single-call depth (macro)")
    path = out_dir / "decomposition_deep_m.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); written.append(path.name)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, bench in zip(axes, BENCH_ORDER):
        sub = known[(known["benchmark"] == bench) & known["config_id"].isin(["16x1", "1x16"])]
        g = sub.groupby("config_id")[["gen_fail", "sel_fail", "S_known"]].mean().reindex(["16x1", "1x16"])
        ax.bar(g.index, g["S_known"], color="#2a7ab0")
        ax.bar(g.index, g["sel_fail"], bottom=g["S_known"], color="#e0a030")
        ax.bar(g.index, g["gen_fail"], bottom=g["S_known"] + g["sel_fail"], color="#b0b0b0")
        ax.set_title(bench, fontsize=9)
        ax.set_ylim(0, 1)
    fig.suptitle("B = 16A: 16×A vote vs 1×16A deep, by benchmark")
    path = out_dir / "decomposition_b16_by_benchmark.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); written.append(path.name)
    return written


def run_e0(run_root: Path, *, legacy_root: Path = LEGACY_ROOT, draws: pd.DataFrame | None = None,
           cache: PairCache | None = None) -> dict:
    run_root, legacy_root = Path(run_root), Path(legacy_root)
    out = run_root / "e0"
    out.mkdir(parents=True, exist_ok=True)
    if draws is None:
        inputs = run_root / "analysis_inputs" / "draws.parquet"
        draws = load_draws(run_root) if inputs.exists() else legacy_draws(legacy_root)
    draws = draws[draws["source"] == "legacy"].copy()
    cache = cache or PairCache(run_root / "catalog" / "pair_verdicts.parquet")

    cohort = cohorts(legacy_root, draws)
    cohort.to_csv(out / "exclusions.csv", index=False)
    items94 = set(cohort.loc[cohort["in_94"], "item_id"])
    items93 = set(cohort.loc[cohort["in_93"], "item_id"])

    pools = build_pools(draws, panels=("L96",), configs=LEGACY_CONFIGS + ("diag5",))
    metrics = pool_metrics(readout_pools(pools, draws, cache=cache, mode="legacy"), draws)
    tables = []
    for label, items in (("96", None), ("94", items94), ("93", items93)):
        s = config_summary(metrics, items=items)
        s.insert(0, "cohort", label)
        tables.append(s)
    legacy_metrics = pd.concat(tables, ignore_index=True)
    legacy_metrics.to_csv(out / "legacy_metrics.csv", index=False)

    rep = replay(legacy_root, draws, cache)
    rep.drop(columns=["member_ids"]).to_parquet(out / "legacy_replay.parquet", index=False)
    rep.loc[rep["drift_eligible"], ["policy_id", "item_id", "name", "equivalence_unknown", "n_tied", "match"]] \
        .to_csv(out / "legacy_replay_drift.csv", index=False)
    clean = rep[~rep["drift_eligible"]]

    _provenance(draws).to_parquet(out / "provenance.parquet", index=False)
    member = _memberships(legacy_root)
    catalog = draws.drop(columns=["correct", "grade_state"]).copy()
    catalog["policy_memberships"] = catalog["candidate_id"].map(lambda c: member.get(c, []))
    catalog["n_policy_memberships"] = catalog["policy_memberships"].map(len)
    catalog.to_parquet(out / "candidate_catalog.parquet", index=False)
    (out / "reuse_manifest.json").write_text(json.dumps(_reuse_manifest(draws), indent=1, sort_keys=True) + "\n")
    plots = _plots(metrics, out)
    from agent_allocation.passk.costs import logical_ledger, physical_ledger, physical_summary
    ledger = physical_ledger(draws)
    physical_summary(ledger).to_csv(out / "legacy_physical_ledger.csv", index=False)
    logical = logical_ledger(pools[pools["config_id"] != "diag5"], draws)
    logical.groupby("config_id")[["nominal_C_tokens", "cost_usd", "think_out", "answer_out"]].mean() \
        .reset_index().to_csv(out / "legacy_logical_ledger_by_config.csv", index=False)

    present = draws[~draws["absent"].astype(bool)]
    gen_ids = pd.concat([present["generation_id_think"], present["generation_id_answer"]]).dropna()
    provider_ok = all(
        (present.loc[present["shard"] == shard, "provider_only"].dropna() == PROVIDER_PIN[shard]).all()
        for shard in SHARDS)
    s96 = legacy_metrics[legacy_metrics["cohort"] == "96"].set_index("config_id")
    s94 = legacy_metrics[legacy_metrics["cohort"] == "94"].set_index("config_id")
    audit = {
        "version": "passk-e0-v1", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "legacy_root": str(legacy_root), "n_candidates": int(len(draws)),
        "n_by_partition_kind": {f"{p}/{k}/{a}A": int(n) for (p, k, a), n
                                in draws.groupby(["partition", "kind", "allowance_a"]).size().items()},
        "observed_model_all_pinned": bool((present["observed_model"] == MODEL).all()),
        "observed_models": sorted(present["observed_model"].dropna().unique().tolist()),
        "observed_providers": sorted(present["observed_provider"].dropna().unique().tolist()),
        "provider_only_matches_shard_pin": bool(provider_ok),
        "generation_ids_unique": bool(gen_ids.is_unique), "n_generation_ids": int(len(gen_ids)),
        "think_body_sha_distinct_nodes": int(present["think_body_sha"].nunique()),
        "think_tokens_exact": bool((present["think_out"] == present["allowance_a"] * 1024).all()),
        "cohorts": {"96": int(cohort["in_96"].sum()), "94": int(cohort["in_94"].sum()),
                    "93": int(cohort["in_93"].sum())},
        "headlines": {name: {"macro_96": float(s96.loc[c, "S_known_macro"]),
                             "macro_94": float(s94.loc[c, "S_known_macro"]),
                             "micro_96": float(s96.loc[c, "S_known_micro"]),
                             "mixture_96": float(s96.loc[c, "S_known_mixture"])}
                      for name, c in HEADLINES.items()},
        "replay": {"checkpoints": int(len(rep)), "match_rate": float(rep["match"].mean()),
                   "clean_checkpoints": int(len(clean)), "clean_match_rate": float(clean["match"].mean())
                   if len(clean) else None, "drift_eligible": int(rep["drift_eligible"].sum()),
                   "drift_mismatches": int((~rep["match"] & rep["drift_eligible"]).sum()),
                   "gate_100pct_clean": bool(len(clean) and clean["match"].all())},
        "pair_cache_rows": len(cache), "plots": plots,
        "legacy_bank_cost_usd": float(ledger["cost_usd"].sum()), "legacy_bank_requests": int(len(ledger)), "mixture_weights": MIXTURE_WEIGHTS,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit
