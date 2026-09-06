"""Join sealed selections with evaluator verdicts → ``<run_root>/tables/*.parquet`` (WP5).

Spec §5.1–§5.3 (selected accuracy, candidate mean with invalid = 0, oracle coverage,
selection gap), §5.6 (pass@K from the ten-draw banks via the unbiased estimator; never the
plug-in), §6.5 (slack and stop reason per episode), §3.4 (calls by role), §9.1 (analysis on
the smallest common completed hash prefix; equal HLE/code weights), §3.6/§10.3 (labels join
only after seals).  Architecture §1.13; corrections P0-3 (refuse a join unless every
evaluation of an item started after its selections sealed), §5 item 7 (bootstrap/Holm are
post-deadline: per-item tables + means only).

Tables
* ``selections.parquet`` — one row per (item × cell × pool × selector) with correctness.
* ``banks.parquet`` — one row per (item × F cell): ``c`` correct of 10, ``pass_at_{1,2,3,5,10}``.
* ``episodes.parquet`` — one row per (item × generate cell): native final correctness,
  stop reason, slack, spent FLOPs, calls by role, counters.
* ``summary.json`` — per-config means per domain and pooled 0.5/0.5, on the raw completed
  set and on the smallest common completed prefix per module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study import metrics_reference as MR
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.evaluation import bcb as bcb_eval
from agents_scaling.study.evaluation import hle_judge
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.types import (
    SELECTOR_VOTE,
    CandidateRecord,
    Domain,
    ProtocolError,
    PublicTask,
)

PASS_KS: tuple[int, ...] = (1, 2, 3, 5, 10)
BLOCK_PER_DOMAIN = 25
TABLES_DIR = "tables"


class JoinRefused(ProtocolError):
    """A correctness join that would violate the seal order (P0-3)."""


# --------------------------------------------------------------------------- correctness


class Correctness:
    """Lazy per-item correctness maps from the evaluator outputs (evaluator identity)."""

    def __init__(self, run_root: Path, tasks: Mapping[str, PublicTask]) -> None:
        self.run_root = run_root
        self.tasks = tasks
        self._hle: dict[str, dict[str, Any] | None] = {}
        self._bcb: dict[tuple[str, str], dict[str, Any]] = {}
        for row in bcb_eval.iter_bcb_eval_rows(run_root):
            key = (str(row["candidate_id"]), str(row["seal"]))
            prev = self._bcb.get(key)
            if prev is not None and prev["status"] != row["status"]:
                raise ProtocolError(f"{key[0][:12]}: conflicting BCB verdicts under seal {key[1][:12]}")
            self._bcb[key] = row

    def hle_record(self, sid: str) -> dict[str, Any] | None:
        if sid not in self._hle:
            self._hle[sid] = hle_judge.load_hle_eval(self.run_root, sid)
        return self._hle[sid]

    def of(self, sid: str, candidates: Sequence[CandidateRecord], *, seal: str, sealed_at: float) -> dict[str, bool] | None:
        """``{candidate_id: correct}`` or ``None`` when the item is not (fully) evaluated.

        Raises :class:`JoinRefused` when an evaluation used here started before the
        selection was sealed (P0-3).
        """
        task = self.tasks[sid]
        if Domain(task.domain) is Domain.HLE:
            record = self.hle_record(sid)
            if record is None:
                return None
            out: dict[str, bool] = {}
            for cand in candidates:
                if not cand.valid or cand.candidate is None:
                    out[cand.candidate_id] = False
                    continue
                entry = record["judgements"].get(hle_judge.answer_sha256(cand.candidate.final_answer))
                if entry is None:
                    return None
                started = entry.get("started_at_by_seal", {}).get(seal)
                if started is None:
                    return None
                if started <= sealed_at:
                    raise JoinRefused(f"{sid}: HLE judgement for seal {seal[:12]} started at {started} <= sealed_at {sealed_at}")
                out[cand.candidate_id] = bool(entry["correct"])
            return out
        out = {}
        for cand in candidates:
            if not cand.valid or cand.candidate is None:
                out[cand.candidate_id] = False
                continue
            row = self._bcb.get((cand.candidate_id, seal))
            if row is None:
                return None
            if float(row["started_at"]) <= sealed_at:
                raise JoinRefused(f"{sid}: BCB evaluation of {cand.candidate_id[:12]} started at {row['started_at']} <= sealed_at {sealed_at}")
            out[cand.candidate_id] = row["status"] == "pass"
        return out


# --------------------------------------------------------------------------- rows


def _episode_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    episode = item.get("episode") or {}
    ledger = episode.get("ledger") or {}
    counters = episode.get("counters") or {}
    return {
        "stop_reason": episode.get("stop_reason"),
        "slack": ledger.get("slack"),
        "spent_flops": ledger.get("spent"),
        "B_flops": ledger.get("B_flops"),  # EpisodeLedger.summary() key (resources/broker.py)
        "calls_admitted": ledger.get("calls_admitted"),
        "calls_by_role": json.dumps(counters.get("calls_by_role", {}), sort_keys=True),
        "complete_candidates": counters.get("complete_candidates"),
        "aliased_calls": counters.get("aliased_calls"),
        "generated_calls": counters.get("generated_calls"),
        "item_status": item.get("status"),
    }


def _public_pool(records: Sequence[CandidateRecord], selector_id: str) -> list[MR.PublicCandidate]:
    """Public view of a pool under a selector's eligibility rule: VOTE counts keyed valid
    candidates (``selection.vote``); JUDGE_BEST counts every valid one (``judge_best_select``)."""
    out: list[MR.PublicCandidate] = []
    for r in records:
        valid = bool(r.valid and r.candidate is not None)
        if selector_id == SELECTOR_VOTE:
            valid = valid and bool(r.vote_key)
        key = (r.vote_key or f"unkeyed:{r.candidate_id}") if valid else None
        out.append(MR.PublicCandidate(r.candidate_id, key, valid))
    return out


def selection_row(sel: Mapping[str, Any], records: Sequence[CandidateRecord], correct: Mapping[str, bool], task: PublicTask,
                  item: Mapping[str, Any], native_correct: bool | None) -> dict[str, Any]:
    rec = sel["record"]
    public = _public_pool(records, sel["selector_id"])
    valid_ids = {p.candidate_id for p in public if p.valid}
    labels = {cid: correct[cid] for cid in valid_ids}
    selection = MR.Selection(rec["selected_candidate_id"], int(rec["planned_count"]), int(rec["valid_count"]), rec.get("winning_count"), rec.get("tied_classes"))
    evaluated = MR.evaluate_sealed_selection(public, selection, labels)
    n_correct = sum(1 for r in records if correct[r.candidate_id])
    return {
        "source_id": task.source_id, "domain": task.domain.value, "split": task.split, "rank": task.rank, "stratum": task.stratum,
        "answer_format": task.answer_format, "seal": sel.get("seal"), "selection_id": sel["selection_id"], "pool_id": sel["pool_id"],
        "cell_id": sel["cell_id"], "module": sel["module"], "method": sel["method"], "checkpoint": sel["checkpoint"], "N": sel["N"],
        "B": sel["B"], "framing": sel["framing"], "episode_rep": sel["episode_rep"], "degree": sel["degree"], "pool_kind": sel["pool_kind"],
        "prefix_k": sel["prefix_k"], "selector_id": sel["selector_id"], "selected_candidate_id": rec["selected_candidate_id"],
        "planned_count": evaluated.planned_count, "valid_count": evaluated.valid_count, "selected_correct": evaluated.selected_correctness,
        "candidate_mean": evaluated.candidate_mean_correctness, "oracle_coverage": evaluated.oracle_coverage, "selection_gap": evaluated.selection_gap,
        "n_correct": n_correct, "no_valid_candidate": bool(rec["no_valid_candidate"]), "all_singleton": bool(rec["all_singleton"]),
        "tied_classes": rec.get("tied_classes"), "winning_count": rec.get("winning_count"), "grouping_mode": rec.get("grouping_mode"),
        "native_final_correct": native_correct, "sealed_at": sel["sealed_at"], **_episode_fields(item),
    }


def bank_row(sel: Mapping[str, Any], records: Sequence[CandidateRecord], correct: Mapping[str, bool], task: PublicTask) -> dict[str, Any]:
    outcomes = [bool(correct[r.candidate_id]) for r in records]
    n, c = len(outcomes), sum(outcomes)
    row = {"source_id": task.source_id, "domain": task.domain.value, "rank": task.rank, "cell_id": sel["cell_id"], "framing": sel["framing"],
           "checkpoint": sel["checkpoint"], "seal": sel.get("seal"), "n": n, "c": c}
    for k in PASS_KS:
        row[f"pass_at_{k}"] = MR.pass_at_k(n, c, k) if k <= n else None
    return row


# --------------------------------------------------------------------------- aggregate


def aggregate(run_root: str | os.PathLike, *, strict: bool = True) -> dict[str, Any]:
    """Build the three tables + summary; returns the summary (also written to disk)."""
    run_root = Path(run_root)
    tasks = {t.source_id: t for t in load_public_tasks(run_root)}
    corr = Correctness(run_root, tasks)
    sel_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    ep_rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    seals_root = run_root / seals.SEALS_DIR
    seal_dirs = sorted(p.parent.name for p in seals_root.glob(f"*/{seals.SELECTIONS_FILE}")) if seals_root.exists() else []
    seen_episodes: set[tuple[str, str]] = set()
    for seal in seal_dirs:
        pools = seals.load_pools(run_root, seal)
        selections = seals.load_selections(run_root, seal)
        pool_index = pools["pools"]
        item_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
        for sel in sorted(selections["selections"].values(), key=lambda s: s["selection_id"]):
            sel = {**sel, "seal": seal}
            pool = pool_index.get(sel["pool_id"])
            if pool is None:
                raise ProtocolError(f"seal {seal[:12]}: selection {sel['selection_id'][:12]} references an unknown pool")
            key = (pool["cell_id"], pool["source_id"])
            if key not in item_cache:
                item_cache[key] = seals.load_item_file(run_root, *key)
            item = item_cache[key]
            if item is None:
                raise ProtocolError(f"seal {seal[:12]}: item file {key} vanished after sealing")
            records_by_id = {r.candidate_id: r for r in seals.candidate_records_of(item)}
            # the same fail-closed keying the seal applied (WP4 records keys, not always modes)
            records = seals.rekey([records_by_id[cid] for cid in pool["candidate_ids"]], pool["answer_format"])
            task = tasks[pool["source_id"]]
            try:
                correct = corr.of(task.source_id, list(records_by_id.values()), seal=seal, sealed_at=float(sel["sealed_at"]))
            except JoinRefused:
                if strict:
                    raise
                skipped["join_refused"] += 1
                continue
            if correct is None:
                skipped["unevaluated"] += 1
                continue
            native = (item.get("episode") or {}).get("native_final") or {}
            native_correct = correct.get(native.get("candidate_id")) if native.get("candidate_id") else None
            sel_rows.append(selection_row(sel, records, correct, task, item, native_correct))
            if pool["pool_kind"] == seals.POOL_BANK_PREFIX and pool["prefix_k"] == max(seals.PREFIXES) and sel["selector_id"] == SELECTOR_VOTE:
                bank_rows.append(bank_row(sel, records, correct, task))
            if key not in seen_episodes and item.get("episode") is not None:
                seen_episodes.add(key)
                ep_rows.append({
                    "source_id": task.source_id, "domain": task.domain.value, "rank": task.rank, "cell_id": pool["cell_id"], "module": pool["module"],
                    "method": pool["method"], "checkpoint": pool["checkpoint"], "N": pool["N"], "B": pool["B"], "framing": pool["framing"],
                    "episode_rep": pool["episode_rep"], "degree": pool["degree"], "seal": seal, "native_final_correct": native_correct,
                    "native_final_valid": native.get("valid"), "native_confidence": native.get("confidence"), **_episode_fields(item),
                })
    tables = run_root / TABLES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    _write_parquet(tables / "selections.parquet", sel_rows)
    _write_parquet(tables / "banks.parquet", bank_rows)
    _write_parquet(tables / "episodes.parquet", ep_rows)
    summary = summarize(sel_rows, bank_rows, tasks)
    summary["skipped"] = dict(skipped)
    summary["n_rows"] = {"selections": len(sel_rows), "banks": len(bank_rows), "episodes": len(ep_rows)}
    summary["seals"] = seal_dirs
    io.write_json(tables / "summary.json", summary)
    return summary


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        io.atomic_write_text(path.with_suffix(".empty"), "no rows\n")
        return
    columns = sorted({k for row in rows for k in row})
    table = pa.Table.from_pylist([{k: row.get(k) for k in columns} for row in rows])
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- summary


def _config_key(row: Mapping[str, Any]) -> str:
    return "|".join(str(row[k]) for k in ("module", "method", "checkpoint", "N", "B", "framing", "episode_rep", "degree", "pool_kind", "prefix_k", "selector_id"))


def common_prefix(items_by_config: Mapping[str, Mapping[str, set[int]]], *, block: int = BLOCK_PER_DOMAIN) -> int:
    """Largest ``n`` (a multiple of ``block``) such that every config has ranks ``0..n-1`` of
    BOTH domains (§9.1 smallest-common-completed-prefix; 0 when any config lacks a block)."""
    best: int | None = None
    for ranks_by_domain in items_by_config.values():
        for domain in Domain:
            ranks = ranks_by_domain.get(domain.value, set())
            n = 0
            while n in ranks:
                n += 1
            best = n if best is None else min(best, n)
    if best is None:
        return 0
    return (best // block) * block


def _mean(values: Sequence[float]) -> float | None:
    return (math.fsum(values) / len(values)) if values else None


def summarize(sel_rows: Sequence[Mapping[str, Any]], bank_rows: Sequence[Mapping[str, Any]], tasks: Mapping[str, PublicTask]) -> dict[str, Any]:
    """Per-config means (per domain + pooled 0.5/0.5) on the raw rows and on the common prefix."""
    by_config: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sel_rows:
        by_config[_config_key(row)].append(row)
    per_module_ranks: dict[str, dict[str, dict[str, set[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for key, rows in by_config.items():
        module = rows[0]["module"]
        for row in rows:
            per_module_ranks[module][key][row["domain"]].add(int(row["rank"]))
    prefix_by_module = {m: common_prefix(cfgs) for m, cfgs in per_module_ranks.items()}

    def stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in ("selected_correct", "candidate_mean", "oracle_coverage", "selection_gap", "native_final_correct"):
            per_domain = {}
            for domain in Domain:
                vals = [float(r[metric]) for r in rows if r["domain"] == domain.value and r.get(metric) is not None]
                per_domain[domain.value] = {"mean": _mean(vals), "n": len(vals)}
            means = [v["mean"] for v in per_domain.values() if v["mean"] is not None]
            pooled = (math.fsum(means) / len(means)) if len(means) == len(Domain) else None
            out[metric] = {**per_domain, "pooled_equal_weight": pooled}
        return out

    configs: dict[str, Any] = {}
    for key, rows in sorted(by_config.items()):
        module = rows[0]["module"]
        n_prefix = prefix_by_module[module]
        prefix_rows = [r for r in rows if int(r["rank"]) < n_prefix]
        configs[key] = {"raw": stats(rows), "common_prefix": {"n_per_domain": n_prefix, **stats(prefix_rows)}, "n_rows": len(rows)}
    banks: dict[str, Any] = {}
    by_bank: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in bank_rows:
        by_bank[f"{row['checkpoint']}|{row['framing']}"].append(row)
    for key, rows in sorted(by_bank.items()):
        entry: dict[str, Any] = {"n_items": len(rows)}
        for k in PASS_KS:
            per_domain = {d.value: _mean([float(r[f"pass_at_{k}"]) for r in rows if r["domain"] == d.value and r.get(f"pass_at_{k}") is not None]) for d in Domain}
            means = [v for v in per_domain.values() if v is not None]
            entry[f"pass_at_{k}"] = {**per_domain, "pooled_equal_weight": (math.fsum(means) / len(means)) if len(means) == len(Domain) else None}
        banks[key] = entry
    return {"configs": configs, "banks": banks, "common_prefix_by_module": prefix_by_module, "prefix_block_per_domain": BLOCK_PER_DOMAIN,
            "weights": {"hle": 0.5, "bcb": 0.5}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.aggregate", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--lenient", action="store_true", help="skip (count) seal-order violations instead of refusing the whole join")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = aggregate(Path(args.results_root) / args.run_id, strict=not args.lenient)
    print(json.dumps({"n_rows": summary["n_rows"], "skipped": summary["skipped"], "common_prefix_by_module": summary["common_prefix_by_module"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["BLOCK_PER_DOMAIN", "Correctness", "JoinRefused", "PASS_KS", "TABLES_DIR", "aggregate", "bank_row", "common_prefix", "selection_row", "summarize"]
