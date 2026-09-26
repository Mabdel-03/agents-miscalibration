"""LEX front analysis (``docs/passk_cont/PREREG_LEX_front.md``): per-problem accuracy -> consistency -> latency.

    python -m agent_allocation.passk.lexfront --study-root R [--workers 16]

Per C48 item, the C48 u V48 (+ ``c48_25a``) draws are pooled per allowance. For every menu option n x a, subsets of n
draws at allowance a are enumerated (or sampled) without replacement: accuracy = mean tie-averaged plurality
correctness, latency = mean of the subset's slowest call (parallel wall-clock), consistency = p^2 + (1 - p)^2.
Answer classes are built once per item over its whole bank (exact vote key, then the gold-free Math-Verify pair
cache, mutual complete-link; GPQA option identity) and reused by every subset.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MENU = ("1x1", "1x2", "1x4", "1x8", "1x16", "1x25", "2x1", "2x2", "2x4", "2x8", "4x1", "4x2", "4x4", "8x1", "8x2",
        "16x1", "25x1")
DELTA = 0.02
SEED = 20260925
MAX_SUBSETS = 5000
SPLIT_SALT = f"passk_cont_v1/lex/split|seed={SEED}"
BANK_PARTITIONS = ("legacy_main", "e1_fill", "c48_extra", "v48_bank")


def parse(config: str) -> tuple[int, int]:
    n, a = config.split("x")
    return int(n), int(a)


# ------------------------------------------------------------------------------------------------------------------
# banks
# ------------------------------------------------------------------------------------------------------------------

def load_banks(study_root: Path) -> pd.DataFrame:
    """One row per usable draw of a C48 item: item, benchmark, allowance, candidate, class-vote fields, correct, lat."""
    draws = pd.read_parquet(study_root / "analysis_inputs" / "draws.parquet")
    splits = json.loads((study_root / "artifacts" / "manifests" / "splits.json").read_text())
    c48 = set(splits["panels"]["C48"])
    bank = draws[draws.item_id.isin(c48) & draws.partition.isin(BANK_PARTITIONS)].copy()
    bank = bank[~bank.get("absent", False).fillna(False).astype(bool)]
    bank = bank[["candidate_id", "item_id", "benchmark", "partition", "allowance_a", "replicate", "parse_status",
                 "vote_exact_key", "vote_text", "option", "correct", "latency_s"]]
    extra_path = study_root / "lex" / "c48_25a_draws.parquet"
    if extra_path.exists():
        extra = pd.read_parquet(extra_path)
        extra = extra.assign(partition="c48_25a", vote_exact_key=None, vote_text=None, option=None)
        bank = pd.concat([bank, extra[bank.columns.intersection(extra.columns)]], ignore_index=True)
    bank["correct"] = bank["correct"].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                                          else float(bool(v)))
    return bank


def item_classes(bank: pd.DataFrame, cache_path: Path, workers: int) -> pd.Series:
    """candidate_id -> item-level class label (or None = no vote). Fills any missing Math-Verify pair first."""
    from agent_allocation.passk.pairs import PairCache, enumerate_pairs, math_votes_from_frame
    from agent_allocation.selection.keys import VoteKey
    from agent_allocation.selection.plurality import build_classes
    from agent_allocation.contracts import OutputKind

    cache = PairCache(cache_path)
    missing = cache.missing(enumerate_pairs(math_votes_from_frame(bank)))
    if missing:
        print(f"[lexfront] filling {len(missing)} missing pairs with {workers} workers", flush=True)
        cache.fill(missing, workers=workers)
        cache.reload()
    labels: dict[str, str | None] = {}
    for item_id, g in bank.groupby("item_id"):
        gpqa = g.benchmark.iloc[0] == "gpqa_diamond"
        votes = []
        for row in g.itertuples(index=False):
            if row.parse_status != "valid":
                labels[row.candidate_id] = None
                continue
            if gpqa:
                labels[row.candidate_id] = f"opt:{row.option}" if isinstance(row.option, str) else None
            elif isinstance(row.vote_exact_key, str) and isinstance(row.vote_text, str):
                votes.append((row.candidate_id, VoteKey(row.vote_exact_key, row.vote_text)))
            else:
                labels[row.candidate_id] = None
        if votes:
            classes, _calls, _unknown = build_classes(votes, kind=OutputKind.MATH,
                                                      math_equivalent=cache.equivalence(item_id))
            for c in classes:
                for m in c.member_ids:
                    labels[m] = c.class_id
    return bank.candidate_id.map(labels)


# ------------------------------------------------------------------------------------------------------------------
# per-item estimates
# ------------------------------------------------------------------------------------------------------------------

def _subsets(m: int, n: int, key: str):
    if math.comb(m, n) <= MAX_SUBSETS:
        return list(itertools.combinations(range(m), n))
    rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:16], 16))
    return [tuple(rng.choice(m, size=n, replace=False)) for _ in range(MAX_SUBSETS)]


def _pool_value(cls: np.ndarray, cor: np.ndarray, idx: tuple) -> float:
    """Tie-averaged plurality correctness of the draws ``idx`` (no valid vote -> 0)."""
    labels = [cls[i] for i in idx if cls[i] is not None]
    if not labels:
        return 0.0
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    top = max(counts.values())
    tied = [c for c, k in counts.items() if k == top]
    class_ok = {}
    for i in idx:
        if cls[i] is not None:
            class_ok.setdefault(cls[i], []).append(cor[i])
    return float(np.mean([np.mean(class_ok[c]) > 0.5 for c in tied]))


def estimate(bank: pd.DataFrame, *, subset_filter=None) -> pd.DataFrame:
    """One row per (item, config): p, consistency, latency, bank size m. ``subset_filter`` restricts the bank."""
    rows = []
    frame = bank if subset_filter is None else bank[subset_filter]
    for (item_id, benchmark), g in frame.groupby(["item_id", "benchmark"]):
        for config in MENU:
            n, a = parse(config)
            b = g[(g.allowance_a == a) & g.correct.notna()].sort_values(["partition", "replicate"])
            m = len(b)
            if m < n:
                rows.append({"item_id": item_id, "benchmark": benchmark, "config": config, "n": n, "a": a, "m": m,
                             "p": np.nan, "C": np.nan, "latency": np.nan})
                continue
            cls = list(b["cls"])
            cor = b["correct"].to_numpy(float)
            lat = b["latency_s"].to_numpy(float)
            subs = _subsets(m, n, f"{item_id}|{config}|{m}")
            if n == 1:
                p = float(cor.mean())
                latency = float(np.nanmean(lat))
            else:
                p = float(np.mean([_pool_value(cls, cor, s) for s in subs]))
                latency = float(np.mean([np.nanmax(lat[list(s)]) for s in subs]))
            rows.append({"item_id": item_id, "benchmark": benchmark, "config": config, "n": n, "a": a, "m": m,
                         "p": p, "C": p * p + (1 - p) * (1 - p), "latency": latency})
    return pd.DataFrame(rows)


def lex_choose(est: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    """Per item: accuracy within delta of the best -> consistency within delta of the best -> fastest -> fewest n*a."""
    out = []
    for item_id, g in est.dropna(subset=["p"]).groupby("item_id"):
        s = g[g.p >= g.p.max() - delta]
        s = s[s.C >= s.C.max() - delta]
        s = s.assign(cost=s.n * s.a).sort_values(["latency", "cost"])
        out.append(s.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------------

def _half(candidate_id: str) -> int:
    return int(hashlib.sha256(f"{SPLIT_SALT}|{candidate_id}".encode()).hexdigest(), 16) % 2


def analyze(study_root: Path, workers: int) -> dict[str, Any]:
    from agent_allocation.passk.stats import MACRO, contrast, level
    bank = load_banks(study_root)
    bank["cls"] = item_classes(bank, study_root / "catalog" / "pair_verdicts.parquet", workers)
    out_dir = study_root / "lex"
    out_dir.mkdir(exist_ok=True)
    census = bank.groupby(["allowance_a"]).agg(draws=("candidate_id", "size"),
                                                unknown=("correct", lambda s: int(s.isna().sum()))).to_dict("index")
    est = estimate(bank)
    est.to_parquet(out_dir / "front_estimates.parquet")
    win = lex_choose(est)
    win.to_parquet(out_dir / "front_winners.parquet")

    # population ordering of the fixed options (macro over benchmarks)
    fixed = est.groupby(["config"]).apply(
        lambda g: pd.Series({"p": g.groupby("benchmark").p.mean().mean(), "C": g.groupby("benchmark").C.mean().mean(),
                             "latency": g.groupby("benchmark").latency.mean().mean(), "n_items": g.p.notna().sum()}),
        include_groups=False).reset_index()
    fx = fixed.dropna(subset=["p"])
    s = fx[fx.p >= fx.p.max() - DELTA]
    s = s[s.C >= s.C.max() - DELTA].sort_values("latency")
    population_choice = s.iloc[0].config

    # in-sample policy vs fixed deep options, per item
    per_item = est.pivot_table(index=["item_id", "benchmark"], columns="config", values=["p", "C", "latency"])
    items = per_item.index.to_frame(index=False)
    policy = items.merge(win[["item_id", "config", "p", "C", "latency"]].rename(columns={
        "config": "lex_config", "p": "lex_p", "C": "lex_C", "latency": "lex_latency"}), on="item_id")
    for ref in ("1x16", "1x25", population_choice):
        for q in ("p", "C", "latency"):
            policy[f"{ref}_{q}"] = policy.item_id.map(per_item[(q, ref)].droplevel(1))

    # split-half: choose on one half, score on the other
    bank["half"] = bank.candidate_id.map(_half)
    halves = {h: estimate(bank, subset_filter=bank.half == h) for h in (0, 1)}
    feasible = set.intersection(*[set(h.dropna(subset=["p"]).config) for h in halves.values()])
    split_rows = []
    for choose_h, score_h in ((0, 1), (1, 0)):
        ch = lex_choose(halves[choose_h][halves[choose_h].config.isin(feasible)])
        sc = halves[score_h].set_index(["item_id", "config"])
        for r in ch.itertuples(index=False):
            got = sc.loc[(r.item_id, r.config)]
            deep = sc.loc[(r.item_id, "1x16")]
            deep25 = sc.loc[(r.item_id, "1x25")]
            split_rows.append({"item_id": r.item_id, "benchmark": r.benchmark, "chosen_on": choose_h,
                               "config": r.config, "p": got.p, "C": got.C, "latency": got.latency,
                               "deep_p": deep.p, "deep_C": deep.C, "deep_latency": deep.latency,
                               "deep25_p": deep25.p, "deep25_C": deep25.C, "deep25_latency": deep25.latency})
    split = pd.DataFrame(split_rows).groupby(["item_id", "benchmark"]).agg(
        p=("p", "mean"), C=("C", "mean"), latency=("latency", "mean"), deep_p=("deep_p", "mean"),
        deep_C=("deep_C", "mean"), deep_latency=("deep_latency", "mean"), deep25_p=("deep25_p", "mean"),
        deep25_C=("deep25_C", "mean"), deep25_latency=("deep25_latency", "mean")).reset_index()
    split.to_parquet(out_dir / "front_splithalf.parquet")

    def lvl(frame, col):
        r = level(frame, col, weights=MACRO)
        return {"mean": r["mean"], "ci95": r.get("ci95_percentile") or r.get("ci95")}

    def diff(frame, a, b):
        r = contrast(frame, a, b, weights=MACRO, n_family=1)
        return {"estimate": r["estimate"], "ci95": r["ci95_percentile"], "p": r["p"]}

    report = {
        "delta": DELTA, "menu": MENU, "bank_census_by_allowance": census,
        "winner_distribution": win.config.value_counts().to_dict(),
        "winner_by_benchmark": win.groupby("benchmark").config.value_counts().unstack(fill_value=0)
                                  .to_dict(orient="index"),
        "fixed_options": fixed.round(4).to_dict(orient="records"),
        "population_lex_choice": population_choice,
        "in_sample": {
            "lex": {q: lvl(policy, f"lex_{q}") for q in ("p", "C", "latency")},
            **{ref: {q: lvl(policy, f"{ref}_{q}") for q in ("p", "C", "latency")}
               for ref in ("1x16", "1x25", population_choice)},
            "lex_minus_1x16": {q: diff(policy, f"lex_{q}", f"1x16_{q}") for q in ("p", "C", "latency")},
            "lex_minus_1x25": {q: diff(policy, f"lex_{q}", f"1x25_{q}") for q in ("p", "C", "latency")},
        },
        "split_half": {
            "feasible_menu": sorted(feasible),
            "lex": {q: lvl(split, q) for q in ("p", "C", "latency")},
            "deep_1x16": {q: lvl(split, f"deep_{q}") for q in ("p", "C", "latency")},
            "deep_1x25": {q: lvl(split, f"deep25_{q}") for q in ("p", "C", "latency")},
            "lex_minus_1x16": {q: diff(split, q, f"deep_{q}") for q in ("p", "C", "latency")},
            "lex_minus_1x25": {q: diff(split, q, f"deep25_{q}") for q in ("p", "C", "latency")},
        },
        "t200_policy": t200_policy(study_root, est, lvl, diff),
    }
    (out_dir / "front_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    return report


def t200_policy(study_root: Path, est: pd.DataFrame, lvl, diff) -> dict[str, Any]:
    """Exploratory (T200 labels already released): the lexicographic rule on each sealed predictor's per-item
    predicted delivered accuracy (its 9-option menu), consistency p^2+(1-p)^2 from the prediction, latency predicted
    by the C48 per-(shard, option) mean; scored on T200's realized delivered correctness and realized wall-clock
    (the slowest member of the chosen pool)."""
    run_dir = study_root / "artifacts"
    assignment = json.loads((run_dir / "shards" / "assignment.json").read_text())["assignment"]
    pred = pd.read_parquet(study_root / "sealed" / "t200" / "predictions.parquet")
    ro = pd.read_parquet(study_root / "analysis_inputs" / "readouts.parquet")
    ro = ro[(ro["mode"] == "passk") & (ro.panel == "T200")]
    pools = pd.read_parquet(study_root / "analysis_inputs" / "pools.parquet")
    pools = pools[pools.panel == "T200"]
    draws = pd.read_parquet(study_root / "analysis_inputs" / "draws.parquet", columns=["candidate_id", "latency_s"])
    lat_of = dict(zip(draws.candidate_id, draws.latency_s))
    realized_lat = {(r.item_id, r.config_id): float(np.nanmax([lat_of.get(m, np.nan) for m in r.member_ids]))
                    for r in pools.itertuples(index=False)}
    realized_acc = {(r.item_id, r.config_id): r.S_known for r in ro.itertuples(index=False)}
    est = est.assign(shard=est.item_id.map(assignment))
    lat_table = est.groupby(["shard", "config"]).latency.mean().to_dict()
    out: dict[str, Any] = {}
    for predictor, g in pred.groupby("predictor"):
        g = g.assign(shard=g.item_id.map(assignment))
        g = g.assign(p=g.p_delivered, C=g.p_delivered ** 2 + (1 - g.p_delivered) ** 2,
                     latency=[lat_table.get((s, c), np.nan) for s, c in zip(g.shard, g.config_id)],
                     n=g.config_id.map(lambda c: parse(c)[0]), a=g.config_id.map(lambda c: parse(c)[1]),
                     config=g.config_id)
        chosen = lex_choose(g)
        frame = chosen[["item_id", "benchmark", "config"]].copy()
        frame["acc"] = [realized_acc.get((i, c), np.nan) for i, c in zip(frame.item_id, frame.config)]
        frame["lat"] = [realized_lat.get((i, c), np.nan) for i, c in zip(frame.item_id, frame.config)]
        frame["deep_acc"] = [realized_acc.get((i, "1x16"), np.nan) for i in frame.item_id]
        frame["deep_lat"] = [realized_lat.get((i, "1x16"), np.nan) for i in frame.item_id]
        frame["acc"] = frame.acc.astype(float)
        frame["deep_acc"] = frame.deep_acc.astype(float)
        out[predictor] = {"choices": frame.config.value_counts().to_dict(),
                          "acc": lvl(frame, "acc"), "latency": lvl(frame, "lat"),
                          "deep_acc": lvl(frame, "deep_acc"), "deep_latency": lvl(frame, "deep_lat"),
                          "acc_minus_deep": diff(frame, "acc", "deep_acc"),
                          "latency_minus_deep": diff(frame, "lat", "deep_lat")}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.lexfront")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    report = analyze(Path(args.study_root), args.workers)
    print(json.dumps({k: report[k] for k in ("winner_distribution", "population_lex_choice")}, indent=1))
    print(json.dumps(report["in_sample"], indent=1, default=float)[:3000])
    print(json.dumps(report["split_half"], indent=1, default=float)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
