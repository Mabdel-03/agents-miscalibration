"""``allocation calibrate``: the pre-registered benchmark gate on the development panel (forced_budget_v2).

For every development item (``calibration.dev_items_per_benchmark`` per benchmark, the manifest's DEV panel) the
command collects ``--roots`` FORCED root generations at ``A = reasoning.main_a`` (partition ``calibration``) and one
NATURAL, unforced root generation (partition ``calibration_natural``, requested allowance
``reasoning.requested_reasoning_max`` so the model stops where it wants), grades them offline, and writes
``calibration.json``:

* per benchmark: forced pass@1 (mean over items of the per-item mean over roots), the natural generation's realized
  reasoning length (mean / median / p90 / max) and natural pass@1, and the share of natural lengths at or above each
  deep budget ``m x A`` (where deep/m sits relative to what the model wants);
* the gate verdict: ``pass`` when ``forced_pass_at_1_min <= pass@1 <= forced_pass_at_1_max``, else ``too_hard`` with
  the pre-named ``calibration.fallback_easier[benchmark]`` or ``too_easy`` with ``calibration.fallback_harder`` (the
  swap itself is an owner step: new manifests, new panels). No per-item selection ever happens here.

Sharded configs: each selected shard collects its own development items on its own endpoint (concurrently, one
dispatcher per shard, each with its slice of ``calibration.max_usd``), writes ``reports/calibration_rows.jsonl`` in
its run dir, and the gate is evaluated on the union at ``<run_root>/calibration.json``.

Unforced calls: the dispatcher calls ``client.generate(spec, messages, attempt_id=, lock=, sampling=)``. For nodes of
the natural partition :class:`NaturalRoutingClient` forwards ``forced=False`` when the underlying client accepts that
keyword (WP-1's ``ForcedClient``); a client without a forced mode (the chat-completions ``OpenRouterClient``) is
unforced already and is called unchanged. INTEGRATION NOTE: if the forced client exposes the unforced mode some other
way, only :meth:`NaturalRoutingClient.generate` changes.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from agent_allocation.config import load_config, validate_config

FORCED_PARTITION = "calibration"
NATURAL_PARTITION = "calibration_natural"
DEFAULT_ROOTS = 4
DEFAULT_MAX_USD = Decimal("3")
ROWS_NAME = "calibration_rows.jsonl"
RESULT_NAME = "calibration.json"


class NaturalRoutingClient:
    """Wraps the runtime's client: natural-partition nodes are generated unforced."""

    def __init__(self, inner: Any, natural_partitions: Sequence[str] = (NATURAL_PARTITION,)):
        self.inner = inner
        self.natural_partitions = frozenset(natural_partitions)
        try:
            params = inspect.signature(inner.generate).parameters
            self._accepts_forced = "forced" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            self._accepts_forced = False

    async def generate(self, spec, messages, *, attempt_id, lock, sampling):
        if spec.partition_id in self.natural_partitions and self._accepts_forced:
            return await self.inner.generate(spec, messages, attempt_id=attempt_id, lock=lock, sampling=sampling,
                                             forced=False)
        return await self.inner.generate(spec, messages, attempt_id=attempt_id, lock=lock, sampling=sampling)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# ------------------------------------------------------------------------------------------------------------------
# Graphs
# ------------------------------------------------------------------------------------------------------------------

def calibration_items(cfg: Mapping[str, Any], paths) -> dict[str, list[str]]:
    """Development items per benchmark (at most ``dev_items_per_benchmark``), restricted to a shard's items."""
    from agent_allocation.live import dev_ids, shard_items
    per = int(cfg.get("calibration", {}).get("dev_items_per_benchmark", 8))
    allowed = shard_items(cfg, paths)
    out = {}
    for benchmark, ids in dev_ids(paths).items():
        chosen = list(ids[:per]) if allowed is None else [i for i in ids if i in allowed][:per]
        out[benchmark] = chosen
    return out


def calibration_graphs(cfg, paths, tasks, lock, amendments, *, roots: int) -> tuple[list, list]:
    from agent_allocation.contracts import GenerationKind
    from agent_allocation.live import nodes_only_graph, standalone_spec
    main_a = int(cfg["reasoning"]["main_a"])
    natural_budget = int(cfg["reasoning"]["requested_reasoning_max"])
    forced, natural = [], []
    for benchmark, ids in sorted(calibration_items(cfg, paths).items()):
        for item in ids:
            task = tasks[item]
            nodes = [standalone_spec(cfg=cfg, partition=FORCED_PARTITION, task=task, lock=lock,
                                     kind=GenerationKind.ROOT, budget=main_a, root_slot=j, amendments=amendments)
                     for j in range(roots)]
            forced.append(nodes_only_graph(task, FORCED_PARTITION, nodes, "calibration_forced"))
            node = standalone_spec(cfg=cfg, partition=NATURAL_PARTITION, task=task, lock=lock,
                                   kind=GenerationKind.ROOT, budget=natural_budget, amendments=amendments)
            natural.append(nodes_only_graph(task, NATURAL_PARTITION, [node], "calibration_natural"))
    return forced, natural


def _waves(graphs: Sequence, size: int) -> list[list]:
    return [list(graphs[k:k + size]) for k in range(0, len(graphs), size)]


# ------------------------------------------------------------------------------------------------------------------
# Grading and rows
# ------------------------------------------------------------------------------------------------------------------

def grade_calibration(paths, graphs: Sequence) -> int:
    """Offline text grading of every calibration candidate (idempotent: an existing grade is kept)."""
    from agent_allocation.config import CODE_BENCHMARKS
    from agent_allocation.contracts import Candidate, PrivateEvalSpec, dump_record, load_record
    from agent_allocation.grading.text.grade import grade_text
    privates: dict[str, PrivateEvalSpec] = {}
    graded = 0
    for graph in graphs:
        for node in graph.nodes:
            if node.benchmark in CODE_BENCHMARKS:
                continue
            target = paths.grade(node.node_id)
            parsed = paths.parsed(node.node_id)
            if target.exists() or not parsed.exists():
                continue
            payload = json.loads(parsed.read_text())
            if not payload.get("candidate"):
                continue
            if node.benchmark not in {p.benchmark for p in privates.values()}:
                source = paths.private_manifest(node.benchmark)
                if source.exists():
                    for line in source.read_text().splitlines():
                        if line.strip():
                            spec = load_record(PrivateEvalSpec, json.loads(line))
                            privates[spec.item_id] = spec
            private = privates.get(node.item_id)
            if private is None:
                continue
            grade = grade_text(load_record(Candidate, payload["candidate"]), private)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(dump_record(grade), sort_keys=True))
            graded += 1
    return graded


def calibration_rows(paths, shard_id: str | None = None) -> list[dict[str, Any]]:
    """One row per calibration node from the stored job manifests, parsed results and grades."""
    from agent_allocation.contracts import GenerationResult, GenerationSpec, GradeResult, load_record
    rows = []
    for partition, mode in ((FORCED_PARTITION, "forced"), (NATURAL_PARTITION, "natural")):
        jobs = paths.jobs(partition)
        if not jobs.exists():
            continue
        for line in jobs.read_text().splitlines():
            if not line.strip():
                continue
            spec = load_record(GenerationSpec, json.loads(line)["record"])
            row: dict[str, Any] = {"mode": mode, "node_id": spec.node_id, "item_id": spec.item_id,
                                   "benchmark": spec.benchmark, "root_slot": spec.root_slot,
                                   "reasoning_budget": spec.reasoning_budget, "shard_id": shard_id,
                                   "terminal": False, "correct": None, "reasoning_tokens": None,
                                   "completion_tokens": None, "finish_reason": None}
            parsed = paths.parsed(spec.node_id)
            if parsed.exists():
                payload = json.loads(parsed.read_text())
                result = load_record(GenerationResult, payload["result"])
                usage = result.usage
                row.update(terminal=True, finish_reason=result.finish_reason,
                           reasoning_tokens=usage.reasoning_tokens if usage else None,
                           completion_tokens=usage.completion_tokens if usage else None,
                           has_candidate=bool(payload.get("candidate")))
                grade_path = paths.grade(spec.node_id)
                if grade_path.exists():
                    grade = load_record(GradeResult, json.loads(grade_path.read_text()))
                    row["correct"] = bool(grade.semantic_correct)
                elif not payload.get("candidate"):
                    row["correct"] = False          # terminal without a candidate scores 0 (never redrawn)
            rows.append(row)
    return rows


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def calibration_summary(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-benchmark forced pass@1, natural reasoning length and the pre-registered gate verdict."""
    cal = cfg.get("calibration", {})
    lo, hi = float(cal.get("forced_pass_at_1_min", 0.15)), float(cal.get("forced_pass_at_1_max", 0.85))
    easier, harder = dict(cal.get("fallback_easier") or {}), dict(cal.get("fallback_harder") or {})
    main_a = int(cfg["reasoning"]["main_a"])
    multipliers = list(cfg["protocols"].get("deep_main_multipliers", []))
    benchmarks: dict[str, Any] = {}
    for benchmark in sorted({r["benchmark"] for r in rows}):
        forced = [r for r in rows if r["benchmark"] == benchmark and r["mode"] == "forced"]
        natural = [r for r in rows if r["benchmark"] == benchmark and r["mode"] == "natural"]
        per_item: dict[str, list[bool]] = {}
        for r in forced:
            if r.get("correct") is not None:
                per_item.setdefault(r["item_id"], []).append(bool(r["correct"]))
        items = sorted({r["item_id"] for r in forced})
        item_means = [mean(v) for v in per_item.values() if v]
        pass1 = mean(item_means) if item_means else None
        realized = [int(r["reasoning_tokens"]) for r in forced if r.get("reasoning_tokens") is not None]
        exact = sum(1 for r in forced if r.get("reasoning_tokens") == r["reasoning_budget"])
        lengths = [float(r["reasoning_tokens"]) for r in natural if r.get("reasoning_tokens") is not None]
        nat_correct = [bool(r["correct"]) for r in natural if r.get("correct") is not None]
        if pass1 is None or len(per_item) < len(items):
            verdict, fallback = "incomplete", None
        elif pass1 < lo:
            verdict, fallback = "too_hard", easier.get(benchmark)
        elif pass1 > hi:
            verdict, fallback = "too_easy", harder.get(benchmark)
        else:
            verdict, fallback = "pass", None
        benchmarks[benchmark] = {
            "items": len(items), "items_graded": len(per_item),
            "forced_generations": len(forced), "forced_graded": sum(len(v) for v in per_item.values()),
            "forced_pass_at_1": pass1, "forced_realized_equals_budget": exact,
            "forced_realized_reasoning_mean": mean(realized) if realized else None,
            "natural_generations": len(natural), "natural_with_usage": len(lengths),
            "natural_reasoning_tokens_mean": mean(lengths) if lengths else None,
            "natural_reasoning_tokens_median": median(lengths) if lengths else None,
            "natural_reasoning_tokens_p90": _quantile(lengths, 0.9) if lengths else None,
            "natural_reasoning_tokens_max": max(lengths) if lengths else None,
            "natural_pass_at_1": mean(nat_correct) if nat_correct else None,
            "natural_share_at_or_above_deep_budget": {
                f"deep/{m}": (sum(1 for x in lengths if x >= m * main_a) / len(lengths)) if lengths else None
                for m in [1, *multipliers]},
            "gate": {"verdict": verdict, "min": lo, "max": hi, "fallback": fallback},
        }
    verdicts = {b: v["gate"]["verdict"] for b, v in benchmarks.items()}
    overall = ("advance" if verdicts and all(v == "pass" for v in verdicts.values())
               else "incomplete" if any(v == "incomplete" for v in verdicts.values()) or not verdicts
               else "swap_benchmarks")
    return {"study_id": cfg.get("study_id"), "main_a": main_a, "gate_range": [lo, hi], "benchmarks": benchmarks,
            "decision": overall, "note": cal.get("note", "pre-registered benchmark-level gate; no per-item selection")}


def write_rows(paths, rows: Sequence[Mapping[str, Any]]) -> Path:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    target = paths.reports_dir / ROWS_NAME
    target.write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    return target


# ------------------------------------------------------------------------------------------------------------------
# Collection
# ------------------------------------------------------------------------------------------------------------------

async def collect(cfg: Mapping[str, Any], paths, *, max_usd: Decimal, roots: int, transport=None,
                  api_key: str | None = None, allow_paid: bool = True) -> dict[str, Any]:
    """Dispatch (resumable) and grade one run dir's calibration nodes; returns the dispatcher summary."""
    from agent_allocation.live import (
        Runtime, active_ceiling, ledger_settled, load_amendments, load_tasks, provider_snapshot, read_endpoint_locks,
        rehearsal, write_jobs,
    )
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    if cfg["primary_model"] not in locks:
        raise RuntimeError(f"{paths.run_dir}: no endpoint lock (run the canary for this shard first)")
    lock = locks[cfg["primary_model"]]
    forced, natural = calibration_graphs(cfg, paths, tasks, lock, amendments, roots=roots)
    write_jobs(paths, FORCED_PARTITION, forced)
    write_jobs(paths, NATURAL_PARTITION, natural)
    settled = ledger_settled(paths)
    cap = Decimal(str(cfg.get("calibration", {}).get("max_usd", DEFAULT_MAX_USD)))
    fraction = Decimal(1)
    if cfg.get("shard"):
        from agent_allocation.config import shard_fraction
        fraction = shard_fraction(cfg["shard"]["share"], cfg["shard"]["total_share"])
    snap = None if (transport is not None or rehearsal()) else provider_snapshot(paths, "calibration")
    ceiling = min(active_ceiling(max_usd, settled, snap), settled + (cap * fraction).quantize(Decimal("0.01")))
    runtime = Runtime(cfg, paths, lock=lock, tasks=tasks, ceiling=ceiling, transport=transport, api_key=api_key)
    wrapped = NaturalRoutingClient(runtime.client)
    runtime.dispatcher.client = wrapped
    size = int(cfg["scheduler"]["wave_items"])
    try:
        summary = await runtime.run(_waves([*forced, *natural], size), allow_paid=allow_paid, max_usd=max_usd,
                                    ceiling=ceiling)
    finally:
        await runtime.aclose()
    graded = grade_calibration(paths, [*forced, *natural])
    return {"summary": str(summary), "graded": graded, "ceiling": str(ceiling),
            "stop_reason": getattr(summary, "stop_reason", None)}


def evaluate(cfg: Mapping[str, Any], shard_paths: Sequence[tuple[str | None, Any]], out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shard_id, paths in shard_paths:
        part = calibration_rows(paths, shard_id)
        write_rows(paths, part)
        rows.extend(part)
    result = calibration_summary(cfg, rows)
    result["shards"] = [s for s, _ in shard_paths if s is not None]
    result["rows"] = len(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    return result


def cmd_calibrate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation calibrate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="run dir (run root for a sharded config)")
    parser.add_argument("--shard", action="append", default=None,
                        help="shard id (repeatable; default: every shard of a sharded config)")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-usd", required=True, help="cumulative study spend cap per run dir (shard)")
    parser.add_argument("--roots", type=int, default=DEFAULT_ROOTS, help="forced roots per development item")
    parser.add_argument("--evaluate-only", action="store_true", help="grade + gate from stored outputs; no dispatch")
    args = parser.parse_args(argv)
    from agent_allocation.live import RunRefused, checked_max_usd, rehearsal
    from agent_allocation.shards import base_config, resolve_shard, run_root, shard_ids
    raw = load_config(args.config)
    base = base_config(raw)
    if base.get("shards") and raw.get("shard") is None:
        selected = args.shard or shard_ids(base)
        targets = [(sid, *resolve_shard(base, args.run_dir, sid)) for sid in selected]
        # The gate is evaluated on the union at the run root; a subset (one shard job) writes its own partial file,
        # and `calibrate --evaluate-only` without --shard then gates the union of every shard's stored rows.
        name = RESULT_NAME if set(selected) == set(shard_ids(base)) else f"calibration_{'_'.join(selected)}.json"
        out = run_root(args.run_dir) / name
    else:
        cfg, paths = resolve_shard(raw, args.run_dir, None)
        targets = [((raw.get("shard") or {}).get("id"), cfg, paths)]
        out = paths.reports_dir / RESULT_NAME
    for _, cfg, _paths in targets:
        problems = validate_config(cfg)
        if problems:
            raise RunRefused(f"config invalid: {problems}")
    if not args.evaluate_only:
        if not args.allow_paid:
            raise RunRefused("calibrate without --allow-paid: nothing is sent (use --evaluate-only to re-grade)")
        caps = [checked_max_usd(cfg, args.max_usd) for _, cfg, _ in targets]
        api_key = None
        if not rehearsal():
            from agent_allocation.apikey import load_api_key
            api_key = load_api_key()

        async def run_all():
            return await asyncio.gather(*(collect(cfg, paths, max_usd=cap, roots=args.roots, api_key=api_key,
                                                  allow_paid=not rehearsal())
                                          for (_, cfg, paths), cap in zip(targets, caps)))
        for (sid, _, _), summary in zip(targets, asyncio.run(run_all())):
            print(json.dumps({"shard": sid, **summary}, default=str))
    else:
        for _, _, paths in targets:
            grade_calibration(paths, _stored_graphs(paths))
    result = evaluate(base, [(sid, paths) for sid, _, paths in targets], out)
    print(json.dumps({"calibration": str(out), "decision": result["decision"],
                      "verdicts": {b: v["gate"]["verdict"] for b, v in result["benchmarks"].items()}}, indent=1))
    return 0 if result["decision"] == "advance" else 3


def _stored_graphs(paths) -> list:
    """Minimal graph stand-ins (``.nodes``) from the stored calibration job manifests, for re-grading."""
    from types import SimpleNamespace

    from agent_allocation.contracts import GenerationSpec, load_record
    graphs = []
    for partition in (FORCED_PARTITION, NATURAL_PARTITION):
        jobs = paths.jobs(partition)
        if jobs.exists():
            nodes = [load_record(GenerationSpec, json.loads(l)["record"]) for l in jobs.read_text().splitlines()
                     if l.strip()]
            graphs.append(SimpleNamespace(nodes=nodes))
    return graphs
