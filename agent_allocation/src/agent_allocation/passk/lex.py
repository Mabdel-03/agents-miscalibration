"""LEX: the lexicographic accuracy -> consistency -> latency front (``docs/passk_cont/PREREG_LEX_front.md``).

``gen25``  8 forced 25A draws (DEEP, r1-8) per C48 item, partition ``c48_25a``, via the unchanged
           ``ForcedClient.generate`` (exact THINK or a retried transport failure), provider pinned, $ cap; then graded
           with the unchanged forced_v2 extractor + grader.

    python -m agent_allocation.passk.lex gen25 --study-root R [--cap-usd 32]
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

PARTITION = "c48_25a"
MULTIPLIER = 25
REPLICATES = tuple(range(1, 9))
MAX_TRANSPORT_ATTEMPTS = 3
CAP_USD = Decimal("32")
CONCURRENCY = {"mancer": 64, "parasail": 36}
READ_TIMEOUT_S = 1800.0


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


async def gen25(raw_cfg: Mapping[str, Any], study_root: Path, *, cap_usd: Decimal = CAP_USD,
                limit: int | None = None) -> dict[str, Any]:
    from agent_allocation.api.forced import make_generation_client
    from agent_allocation.apikey import load_api_key
    from agent_allocation.contracts import GenerationKind, TransportStatus, dump_record
    from agent_allocation.live import load_amendments, load_tasks, read_endpoint_locks, standalone_spec
    from agent_allocation.passk.run import resolve
    from agent_allocation.prompts.render import render_messages   # the harness's own ROOT/DEEP rendering

    run_dir = study_root / "artifacts"
    out = study_root / "lex"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    calls_path = out / f"{PARTITION}_calls.jsonl"
    splits = json.loads((run_dir / "manifests" / "splits.json").read_text())
    items = list(splits["panels"]["C48"])[: limit or None]
    assignment = json.loads((run_dir / "shards" / "assignment.json").read_text())["assignment"]
    prior = _rows(calls_path)
    done = {(r["item_id"], r["replicate"]) for r in prior if r["transport_status"] == "ok"}
    state = {"spent": sum(Decimal(str(r.get("cost_usd") or 0)) for r in prior), "stopped": False}
    key = load_api_key()
    shards: dict[str, dict[str, Any]] = {}
    for shard in sorted({assignment[i] for i in items}):
        cfg, paths = resolve(raw_cfg, str(run_dir), shard)
        shards[shard] = {"cfg": cfg, "lock": read_endpoint_locks(paths)[cfg["primary_model"]],
                         "client": make_generation_client(cfg, api_key=key, read_timeout=READ_TIMEOUT_S),
                         "tasks": load_tasks(paths), "amend": load_amendments(paths),
                         "sem": asyncio.Semaphore(CONCURRENCY.get(shard, 16))}
    io_lock = asyncio.Lock()
    main_a = int(raw_cfg["reasoning"]["main_a"])

    async def one(item_id: str, r: int) -> None:
        if (item_id, r) in done:
            return
        sh = shards[assignment[item_id]]
        task = sh["tasks"][item_id]
        spec = standalone_spec(cfg=sh["cfg"], partition=PARTITION, task=task, lock=sh["lock"],
                               kind=GenerationKind.DEEP, budget=MULTIPLIER * main_a, root_slot=r, draw_id=r,
                               amendments=sh["amend"])
        async with sh["sem"]:
            for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
                if state["spent"] >= cap_usd:
                    state["stopped"] = True
                    return
                outcome = await sh["client"].generate(spec, render_messages(spec, task, [], seed=int(sh["cfg"]["seed"])),
                                                      attempt_id=f"{spec.node_id}:{attempt}", lock=sh["lock"],
                                                      sampling=sh["cfg"]["sampling"])
                res = outcome.result
                cost = Decimal(str(res.usage.actual_cost_usd)) if res.usage and res.usage.actual_cost_usd else \
                    Decimal(0)
                raw_path = out / "raw" / f"{spec.node_id}_{attempt}.json.gz"
                raw_path.write_bytes(gzip.compress(json.dumps(
                    {"request_body": outcome.request_body,
                     "response_body_text": outcome.response_body.decode("utf-8", "replace"),
                     "response_headers": dict(outcome.response_headers), "result": dump_record(res)},
                    default=str).encode()))
                row = {"item_id": item_id, "benchmark": task.benchmark, "shard": assignment[item_id],
                       "replicate": r, "node_id": spec.node_id, "attempt": attempt,
                       "transport_status": res.transport_status.value, "error": res.error_detail,
                       "reasoning_tokens": res.usage.reasoning_tokens if res.usage else None,
                       "latency_s": res.latency_seconds, "cost_usd": str(cost),
                       "observed_model": res.observed_model, "observed_provider": res.observed_provider,
                       "raw": str(raw_path.relative_to(study_root)),
                       "spec": dump_record(spec), "result": dump_record(res)}
                async with io_lock:
                    state["spent"] += cost
                    with calls_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, default=str) + "\n")
                if res.transport_status is TransportStatus.OK:
                    return

    try:
        await asyncio.gather(*(one(i, r) for i in items for r in REPLICATES))
    finally:
        for sh in shards.values():
            close = getattr(sh["client"], "aclose", None)
            if close:
                await close()
    ok = {(r["item_id"], r["replicate"]) for r in _rows(calls_path) if r["transport_status"] == "ok"}
    return {"planned": len(items) * len(REPLICATES), "ok": len(ok), "spent_usd": str(state["spent"]),
            "stopped_at_cap": state["stopped"], "calls_file": str(calls_path)}


def grade25(study_root: Path) -> dict[str, Any]:
    """Extract + grade every ok 25A draw with the unchanged forced_v2 extractor and grader -> lex/c48_25a_draws.parquet."""
    import pandas as pd
    from agent_allocation.contracts import GenerationResult, GenerationSpec, PrivateEvalSpec, PublicTask, load_record
    from agent_allocation.extraction.final import extract_candidate
    from agent_allocation.grading.text.grade import grade_text
    run_dir = study_root / "artifacts"
    tasks: dict[str, PublicTask] = {}
    private: dict[str, PrivateEvalSpec] = {}
    for f in (run_dir / "manifests" / "public").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                t = load_record(PublicTask, json.loads(line))
                tasks[t.item_id] = t
    for f in (run_dir / "private").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                p = load_record(PrivateEvalSpec, json.loads(line))
                private[p.item_id] = p
    rows = []
    for r in _rows(study_root / "lex" / f"{PARTITION}_calls.jsonl"):
        if r["transport_status"] != "ok":
            continue
        spec = load_record(GenerationSpec, r["spec"])
        res = load_record(GenerationResult, r["result"])
        cand = extract_candidate(res, spec, tasks[r["item_id"]])
        grade = grade_text(cand, private[r["item_id"]])
        rows.append({"item_id": r["item_id"], "benchmark": r["benchmark"], "shard": r["shard"],
                     "replicate": r["replicate"], "candidate_id": spec.node_id, "allowance_a": MULTIPLIER,
                     "parse_status": cand.parse_status.value, "final_answer": cand.final_answer,
                     "correct": grade.semantic_correct if grade.semantic_correct is not None else None,
                     "operational": grade.operational_success, "grade_status": grade.status.value,
                     "latency_s": r["latency_s"], "reasoning_tokens": r["reasoning_tokens"],
                     "cost_usd": float(r["cost_usd"]), "observed_provider": r["observed_provider"]})
    frame = pd.DataFrame(rows).drop_duplicates(["item_id", "replicate"], keep="last")
    frame.to_parquet(study_root / "lex" / f"{PARTITION}_draws.parquet")
    return {"graded": int(len(frame)), "correct": int((frame.correct == True).sum()),   # noqa: E712
            "unknown": int(frame.correct.isna().sum()), "exact_budget": int((frame.reasoning_tokens == 25600).sum())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passk.lex")
    parser.add_argument("verb", choices=("gen25", "grade25"))
    parser.add_argument("--config", default="configs/passk_cont_v1.json")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--cap-usd", default=str(CAP_USD))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    root = Path(args.study_root)
    if args.verb == "gen25":
        from agent_allocation.config import load_config
        print(json.dumps(asyncio.run(gen25(load_config(args.config), root, cap_usd=Decimal(args.cap_usd),
                                           limit=args.limit)), indent=1))
        return 0
    print(json.dumps(grade25(root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
