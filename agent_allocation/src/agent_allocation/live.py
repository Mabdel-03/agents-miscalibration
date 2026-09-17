"""Canary / pilot / run orchestration (lead-owned). The only module that constructs a paid dispatcher.

Every paid path requires ``--allow-paid``; ``--max-usd`` is cumulative study spend from the one ledger and may not
exceed the preset ceiling. The active ceiling handed to the ledger is additionally bounded by the key's remaining
limit and the live account balance (other projects bill the same account), minus a safety margin.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.config import (
    credit_ceiling, decoding_config_hash, grids_from_config, headroom_for, load_config, validate_config,
)
from agent_allocation.contracts import (
    BBH_SUBTASKS, BENCHMARKS, STUDY_ID, EndpointLock, GenerationKind, GenerationSpec, GridSpec, ItemGraph,
    PublicTask, dump_record, load_record, public_task_hash,
)
from agent_allocation.paths import RunPaths

SAFETY_MARGIN_USD = Decimal("1.00")
CANARY_BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)
CANARY_CODE_BUDGETS = (1024, 4096)
PILOT_CEILING_USD = Decimal("5")
BUDGET_STOP_TEST_CEILING_USD = Decimal("0.25")


class RunRefused(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------------------------

def load_tasks(paths: RunPaths) -> dict[str, PublicTask]:
    tasks: dict[str, PublicTask] = {}
    for benchmark in BENCHMARKS:
        with open(paths.public_manifest(benchmark), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task = load_record(PublicTask, json.loads(line))
                    tasks[task.item_id] = task
    return tasks


def load_amendments(paths: RunPaths) -> dict[str, int]:
    """Recorded headroom amendments: {benchmark: headroom}."""
    result: dict[str, int] = {}
    if paths.amendments.exists():
        for line in paths.amendments.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("rule") == "headroom_once":
                    for benchmark in row["scope"]:
                        result[benchmark] = int(row["after"])
    return result


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def read_endpoint_locks(paths: RunPaths) -> dict[str, EndpointLock]:
    if not paths.endpoint_lock.exists():
        return {}
    data = json.loads(paths.endpoint_lock.read_text())
    return {model: load_record(EndpointLock, row) for model, row in data["locks"].items()}


async def ensure_endpoint_locks(paths: RunPaths, client, models: Sequence[str], *, allow_fetch: bool) -> dict[str, EndpointLock]:
    """Write-once. Missing models are fetched and appended only before the lock is frozen."""
    from agent_allocation.api.endpoints import select_standard_endpoint
    locks = read_endpoint_locks(paths)
    missing = [m for m in models if m not in locks]
    if missing and not allow_fetch:
        raise RunRefused(f"endpoint lock missing for {missing}; run the canary first")
    frozen = paths.config_lock.exists()
    if missing and frozen:
        raise RunRefused("endpoint lock is frozen; a new model needs a new approved configuration")
    for model in missing:
        raw, parsed = await client.list_endpoints(model)
        paths.lock_snapshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot = paths.lock_snapshots_dir / f"endpoints_{model_slug(model)}_{stamp}.json"
        snapshot.write_bytes(raw)
        locks[model] = select_standard_endpoint(parsed, model, snapshot_path=str(snapshot),
                                                snapshot_sha256=sha256(raw).hexdigest(), fetched_at=stamp)
    if missing:
        payload = {"locks": {m: dump_record(l) for m, l in locks.items()},
                   "lock_hashes": {m: l.lock_hash for m, l in locks.items()}, "updated_at": utcnow()}
        paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
        paths.endpoint_lock.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return locks


async def verify_endpoint_drift(client, locks: Mapping[str, EndpointLock]) -> list[str]:
    from agent_allocation.api.endpoints import EndpointBlocked, select_standard_endpoint
    drift = []
    for model, lock in locks.items():
        _raw, parsed = await client.list_endpoints(model)
        try:
            live = select_standard_endpoint(parsed, model, preferred_tag=lock.provider_tag)
        except EndpointBlocked as exc:
            drift.append(f"{model}: {exc}")
            continue
        if live.lock_hash != lock.lock_hash:
            drift.append(f"{model}: live endpoint semantics differ from the lock")
    return drift


def provider_snapshot(paths: RunPaths, source: str) -> dict[str, Any]:
    from agent_allocation.apikey import load_api_key
    from agent_allocation.doctor import provider_snapshot as fetch
    from agent_allocation.doctor import write_provider_usage
    snap = fetch(load_api_key())
    write_provider_usage(paths, source, snap)
    return snap


def active_ceiling(max_usd: Decimal, ledger_settled: Decimal, snapshot: Mapping[str, Any] | None) -> Decimal:
    ceiling = max_usd
    if snapshot is not None:
        remaining = snapshot["key"].get("limit_remaining")
        if remaining is not None:
            ceiling = min(ceiling, ledger_settled + Decimal(str(remaining)) - SAFETY_MARGIN_USD)
        ceiling = min(ceiling, ledger_settled + Decimal(snapshot["account_balance_usd"]) - SAFETY_MARGIN_USD)
    return max(ceiling, Decimal("0"))


# ------------------------------------------------------------------------------------------------------------------
# Graphs
# ------------------------------------------------------------------------------------------------------------------

def graph_for(grid: GridSpec, *, cfg: Mapping[str, Any], partition: str, task: PublicTask, lock: EndpointLock,
              base_budget: int, amendments: Mapping[str, int]) -> ItemGraph:
    from agent_allocation.prompts.render import prompt_version
    from agent_allocation.protocols.grid import build_item_graph
    return build_item_graph(grid, run_id=STUDY_ID, partition_id=partition, task=task, model=lock.model,
                            base_budget=base_budget, headroom=headroom_for(cfg, task.benchmark, amendments),
                            endpoint_lock_hash=lock.lock_hash, decoding_config_hash=decoding_config_hash(cfg),
                            prompt_version=prompt_version())


def standalone_spec(*, cfg: Mapping[str, Any], partition: str, task: PublicTask, lock: EndpointLock,
                    kind: GenerationKind, budget: int, root_slot: int = 0, draw_id: int = 0,
                    amendments: Mapping[str, int]) -> GenerationSpec:
    from agent_allocation.prompts.render import prompt_version
    return GenerationSpec(run_id=STUDY_ID, partition_id=partition, item_id=task.item_id, benchmark=task.benchmark,
                          kind=kind, model=lock.model, endpoint_lock_hash=lock.lock_hash,
                          input_manifest_hash=public_task_hash(task), decoding_config_hash=decoding_config_hash(cfg),
                          prompt_version=prompt_version(), reasoning_budget=budget,
                          completion_headroom=headroom_for(cfg, task.benchmark, amendments), root_slot=root_slot,
                          team_size=0, update_round=0, draw_id=draw_id)


def nodes_only_graph(task: PublicTask, partition: str, nodes: Sequence[GenerationSpec], name: str) -> ItemGraph:
    grid = GridSpec(name=name, bank_size=len(nodes), private_single_cycles=1, private_team_rounds=0, team_sizes=(),
                    peer_rounds=0, deep_multipliers=())
    return ItemGraph(item_id=task.item_id, partition_id=partition, grid=grid, nodes=tuple(nodes), checkpoints=())


def merge_graphs(base: ItemGraph, extra: Sequence[GenerationSpec]) -> ItemGraph:
    known = {n.node_id for n in base.nodes}
    added = tuple(n for n in extra if n.node_id not in known)
    return replace(base, nodes=base.nodes + added)


def write_jobs(paths: RunPaths, partition: str, graphs: Sequence[ItemGraph]) -> str:
    target = paths.jobs(partition)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for graph in graphs:
        for node in graph.nodes:
            lines.append(json.dumps({"node_id": node.node_id, "record": dump_record(node)}, sort_keys=True))
    text = "\n".join(lines) + "\n"
    target.write_text(text)
    policies = paths.policies(partition)
    policies.parent.mkdir(parents=True, exist_ok=True)
    policies.write_text("".join(json.dumps(dump_record(c), sort_keys=True) + "\n"
                                for g in graphs for c in g.checkpoints))
    return sha256(text.encode()).hexdigest()


def dev_ids(paths: RunPaths) -> dict[str, tuple[str, ...]]:
    from agent_allocation.manifest import load_panels
    return {b: d.dev_item_ids for b, d in load_panels(paths).items()}


def first_per_subtask(ids: Sequence[str], tasks: Mapping[str, PublicTask], occurrence: int) -> list[str]:
    picked = []
    for subtask in BBH_SUBTASKS:
        matches = [i for i in ids if tasks[i].subtask == subtask]
        picked.append(matches[occurrence])
    return picked


# ------------------------------------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------------------------------------

async def dispatch(cfg: Mapping[str, Any], paths: RunPaths, waves: Sequence[Sequence[ItemGraph]], *,
                   lock: EndpointLock, tasks: Mapping[str, PublicTask], allow_paid: bool, max_usd: Decimal,
                   ceiling: Decimal, transport=None, api_key: str | None = None):
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.extraction.final import extract_candidate
    from agent_allocation.prompts.render import render_messages_with_report
    from agent_allocation.runtime.ledger import CollectionLedger
    from agent_allocation.runtime.scheduler import AdmissionPlan, Dispatcher
    from agent_allocation.storage.journal import LedgerJournal
    from agent_allocation.storage.state import StateStore

    scheduler = cfg["scheduler"]
    client = OpenRouterClient(api_key=api_key, transport=transport,
                              connect_timeout=float(scheduler["connect_timeout_seconds"]),
                              read_timeout=float(scheduler["read_timeout_seconds"]))
    journal = LedgerJournal(paths.ledger_journal)
    ledger = CollectionLedger(journal, ceiling_usd=min(ceiling, max_usd),
                              max_unresolved_usd=Decimal(str(cfg["budget"]["max_unresolved_spend_usd"])))
    store = StateStore.recover(paths) if paths.state_db_backup.exists() or paths.ledger_journal.exists() else StateStore(paths)
    seed = int(cfg["seed"])

    render_log = paths.render_reports
    paths.state_dir.mkdir(parents=True, exist_ok=True)

    def render(spec, parents):
        try:
            messages, report = render_messages_with_report(spec, tasks[spec.item_id], parents, seed=seed)
        except Exception as exc:
            if "context" in type(exc).__name__.lower():
                record = getattr(exc, "record", None) or {}
                with open(paths.context_limit_events, "a") as handle:
                    handle.write(json.dumps({"node_id": spec.node_id, "recorded_at": utcnow(), **record},
                                            sort_keys=True, default=str) + "\n")
                with open(render_log, "a") as handle:
                    handle.write(json.dumps({"node_id": spec.node_id, "context_limit": True,
                                             "error": type(exc).__name__, "recorded_at": utcnow()}) + "\n")
            raise
        with open(render_log, "a") as handle:
            handle.write(json.dumps({"node_id": spec.node_id, "context_limit": False,
                                     "trimmed_packets": report.get("trimmed_packets"),
                                     "bytes_removed_total": report.get("bytes_removed_total"),
                                     "rendered_chars": report.get("rendered_chars"), "recorded_at": utcnow()}) + "\n")
        return messages

    def extract(result, spec):
        return extract_candidate(result, spec, tasks[spec.item_id])

    from agent_allocation.protocols.grid import check_budgets
    for wave in waves:
        for graph in wave:
            check_budgets(graph, max_completion_tokens=lock.max_completion_tokens,
                          requested_reasoning_max=int(cfg["reasoning"]["requested_reasoning_max"]))
    dispatcher = Dispatcher(config=cfg, paths=paths, store=store, ledger=ledger, client=client, lock=lock,
                            render=render, extract=extract)
    try:
        return await dispatcher.run(AdmissionPlan(waves=tuple(tuple(w) for w in waves)), allow_paid=allow_paid,
                                    max_usd=max_usd)
    finally:
        await client.aclose()
        store.backup()


def checked_max_usd(cfg: Mapping[str, Any], requested: str) -> Decimal:
    value = Decimal(requested)
    if value > credit_ceiling(cfg):
        raise RunRefused(f"--max-usd {value} exceeds the preset ceiling {credit_ceiling(cfg)}; "
                         "a new owner-approved configuration is required")
    return value


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-usd", required=True)


def preflight(cfg: Mapping[str, Any], args) -> tuple[RunPaths, Decimal]:
    problems = validate_config(cfg)
    if problems:
        raise RunRefused(f"config invalid: {problems}")
    if not args.allow_paid:
        raise RunRefused("live command without --allow-paid: nothing is sent (use `allocation plan --dry-run`)")
    return RunPaths.for_run(args.run_dir), checked_max_usd(cfg, args.max_usd)


# ------------------------------------------------------------------------------------------------------------------
# Canary (Gate 1)
# ------------------------------------------------------------------------------------------------------------------

def canary_graphs(cfg, paths, tasks, locks, amendments) -> tuple[list[ItemGraph], list[ItemGraph]]:
    primary = locks[cfg["primary_model"]]
    replication = locks[cfg["replication_model"]]
    dev = dev_ids(paths)
    bbh_tracking = [i for i in dev["bbh_logic4"] if tasks[i].subtask == "tracking_shuffled_objects_seven_objects"][0]
    text_items = [dev["math500"][0], dev["mmlu_pro"][0], bbh_tracking]
    lite: list[ItemGraph] = []
    for item in text_items:
        nodes = [standalone_spec(cfg=cfg, partition="canary", task=tasks[item], lock=primary, kind=GenerationKind.ROOT,
                                 budget=a, amendments=amendments) for a in CANARY_BUDGETS]
        lite.append(nodes_only_graph(tasks[item], "canary", nodes, "canary_budget_sweep"))
    code_item = dev["mbpp_plus"][0]
    nodes = [standalone_spec(cfg=cfg, partition="canary", task=tasks[code_item], lock=primary, kind=GenerationKind.ROOT,
                             budget=a, amendments=amendments) for a in CANARY_CODE_BUDGETS]
    lite.append(nodes_only_graph(tasks[code_item], "canary", nodes, "canary_code"))
    flash = []
    for item in (dev["math500"][0], code_item):
        node = standalone_spec(cfg=cfg, partition="canary", task=tasks[item], lock=replication,
                               kind=GenerationKind.ROOT, budget=1024, amendments=amendments)
        flash.append(nodes_only_graph(tasks[item], "canary", [node], "canary_flash"))
    return lite, flash


async def fetch_metadata(paths: RunPaths, generation_ids: Sequence[str], *, attempts: int = 12, delay_s: float = 10.0):
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    client = OpenRouterClient(api_key=load_api_key())
    found: dict[str, Any] = {}
    try:
        for _ in range(attempts):
            pending = [g for g in generation_ids if g and g not in found]
            if not pending:
                break
            for gid in pending:
                data = await client.generation_metadata(gid)
                if data is not None:
                    found[gid] = data
            if len(found) < len([g for g in generation_ids if g]):
                await asyncio.sleep(delay_s)
    finally:
        await client.aclose()
    with open(paths.generation_metadata, "a") as handle:
        for gid, data in found.items():
            handle.write(json.dumps({"generation_id": gid, "data": data, "fetched_at": utcnow()}, sort_keys=True) + "\n")
    return found


def load_call_rows(paths: RunPaths, partition: str, metadata: Mapping[str, Any]):
    from agent_allocation.contracts import Candidate, GenerationResult
    from agent_allocation.gates import CallRow
    rows = []
    with open(paths.jobs(partition)) as handle:
        specs = [load_record(GenerationSpec, json.loads(l)["record"]) for l in handle if l.strip()]
    for spec in specs:
        parsed = paths.parsed(spec.node_id)
        if not parsed.exists():
            continue
        payload = json.loads(parsed.read_text())
        result = load_record(GenerationResult, payload["result"])
        candidate = load_record(Candidate, payload["candidate"]) if payload.get("candidate") else None
        rows.append(CallRow(spec, result, candidate, metadata.get(result.generation_id)))
    return rows


def ledger_settled(paths: RunPaths) -> Decimal:
    from agent_allocation.storage.journal import LedgerJournal
    from agent_allocation.contracts import LedgerEventKind
    total = Decimal("0")
    if paths.ledger_journal.exists():
        for event in LedgerJournal(paths.ledger_journal).replay():
            if event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX):
                total += event.amount_usd
    return total


async def run_canary(cfg, paths: RunPaths, max_usd: Decimal) -> int:
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    from agent_allocation.gates import evaluate_gate1, write_gate
    api_key = load_api_key()
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    client = OpenRouterClient(api_key=api_key)
    try:
        locks = await ensure_endpoint_locks(paths, client, [cfg["primary_model"], cfg["replication_model"]],
                                            allow_fetch=True)
    finally:
        await client.aclose()
    before = provider_snapshot(paths, "canary_before")
    settled_before = ledger_settled(paths)
    lite, flash = canary_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "canary", [*lite, *flash])
    ceiling = min(active_ceiling(max_usd, settled_before, before), settled_before + Decimal("0.50"))
    summaries = []
    for graphs, model in ((lite, cfg["primary_model"]), (flash, cfg["replication_model"])):
        summaries.append(await dispatch(cfg, paths, [graphs], lock=locks[model], tasks=tasks, allow_paid=True,
                                        max_usd=max_usd, ceiling=ceiling, api_key=api_key))
    rows = load_call_rows(paths, "canary", {})
    metadata = await fetch_metadata(paths, [r.result.generation_id for r in rows])
    rows = load_call_rows(paths, "canary", metadata)
    await asyncio.sleep(30)
    after = provider_snapshot(paths, "canary_after")
    key_delta = Decimal(str(after["key"].get("usage", 0))) - Decimal(str(before["key"].get("usage", 0)))
    settled_delta = ledger_settled(paths) - settled_before
    result = evaluate_gate1(rows, locks, key_usage_delta_usd=key_delta, ledger_settled_usd=settled_delta)
    write_gate(paths, result)
    from agent_allocation.api.canary import canary_report
    report = canary_report([(r.spec, r.result, r.metadata) for r in rows])
    (paths.reports_dir / "CANARY_REPORT.json").parent.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "CANARY_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(json.dumps({"gate": dump_record(result), "summaries": [str(s) for s in summaries]}, indent=1, default=str))
    return 0 if result.decision == "advance" else 3


def cmd_canary(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation canary")
    common_args(parser)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths, max_usd = preflight(cfg, args)
    paths.ensure()
    return asyncio.run(run_canary(cfg, paths, max_usd))


# ------------------------------------------------------------------------------------------------------------------
# Independent audit helpers (do not trust the ledger object's own view)
# ------------------------------------------------------------------------------------------------------------------

def journal_replay_overshoot(paths: RunPaths, ceiling: Decimal, since_seq: int = 0) -> Decimal:
    """Max over the event sequence of (settled + outstanding reservations) - ceiling, replayed from the journal."""
    from agent_allocation.contracts import LedgerEventKind
    from agent_allocation.storage.journal import LedgerJournal
    outstanding: dict[str, Decimal] = {}
    settled = Decimal("0")
    worst = Decimal("-Infinity")
    for event in LedgerJournal(paths.ledger_journal).replay():
        if event.kind == LedgerEventKind.RESERVE:
            outstanding[event.attempt_id] = event.amount_usd
        elif event.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX):
            outstanding.pop(event.attempt_id, None)
            settled += event.amount_usd
        elif event.kind == LedgerEventKind.RELEASE_UNSENT:
            outstanding.pop(event.attempt_id, None)
        if event.seq >= since_seq:
            worst = max(worst, settled + sum(outstanding.values()) - ceiling)
    return worst if worst != Decimal("-Infinity") else Decimal("0")


def last_journal_seq(paths: RunPaths) -> int:
    from agent_allocation.storage.journal import LedgerJournal
    seq = 0
    if paths.ledger_journal.exists():
        for event in LedgerJournal(paths.ledger_journal).replay():
            seq = max(seq, event.seq)
    return seq


def duplicate_successful_dispatches(paths: RunPaths, node_ids: set[str]) -> int:
    """Nodes with two or more raw HTTP-200 completion bodies: a known-successful node was sent again."""
    import gzip
    counts: dict[str, int] = {}
    raw_root = paths.run_dir / "raw"
    if not raw_root.exists():
        return 0
    for path in raw_root.rglob("*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("node_id") in node_ids and payload.get("response_status") == 200:
            body = payload.get("response_body_text") or ""
            if '"choices"' in body:
                counts[payload["node_id"]] = counts.get(payload["node_id"], 0) + 1
    return sum(1 for c in counts.values() if c > 1)


# ------------------------------------------------------------------------------------------------------------------
# Pilot (Gate 2)
# ------------------------------------------------------------------------------------------------------------------

def pilot_graphs(cfg, paths, tasks, locks, amendments) -> tuple[list[list[ItemGraph]], list[ItemGraph]]:
    grids = grids_from_config(cfg)
    primary = locks[cfg["primary_model"]]
    dev = dev_ids(paths)
    r2: set[str] = set()
    deep_items: list[str] = []
    high_items: list[str] = []
    for benchmark in BENCHMARKS:
        ids = list(dev[benchmark])
        if benchmark == "bbh_logic4":
            first = first_per_subtask(ids, tasks, 0)
            second = first_per_subtask(ids, tasks, 1)
            r2 |= set(first)
            deep_items.append(first[0])
            high_items += second[:2]
        else:
            r2 |= set(ids[:4])
            deep_items.append(ids[0])
            high_items += ids[4:6]
    wave_r2, wave_r1 = [], []
    all_graphs = []
    for benchmark in BENCHMARKS:
        for item in dev[benchmark]:
            task = tasks[item]
            grid = grids["pilot_r2"] if item in r2 else grids["pilot_r1"]
            graph = graph_for(grid, cfg=cfg, partition="pilot", task=task, lock=primary,
                              base_budget=int(cfg["reasoning"]["main_a"]), amendments=amendments)
            extra = []
            if item in deep_items:
                extra.append(standalone_spec(cfg=cfg, partition="pilot", task=task, lock=primary,
                                             kind=GenerationKind.DEEP, budget=int(cfg["reasoning"]["main_a"]) * 16,
                                             amendments=amendments))
            if item in high_items:
                extra += [standalone_spec(cfg=cfg, partition="pilot", task=task, lock=primary, kind=GenerationKind.ROOT,
                                          budget=int(cfg["reasoning"]["high_a"]), root_slot=j, amendments=amendments)
                          for j in range(2)]
            graph = merge_graphs(graph, extra)
            (wave_r2 if item in r2 else wave_r1).append(graph)
            all_graphs.append(graph)
    return [wave_r2, wave_r1], all_graphs


def render_report_violations(paths: RunPaths) -> int:
    target = paths.context_limit_events
    if not target.exists():
        return 0
    return len({json.loads(line)["node_id"] for line in target.read_text().splitlines() if line.strip()})


def select_and_seal(cfg, paths: RunPaths, partition: str, graphs: Sequence[ItemGraph], tasks) -> dict[str, int]:
    from agent_allocation.contracts import Candidate, PublicCheckResult
    from agent_allocation.grading.text.math_equiv import math_equivalent
    from agent_allocation.selection.readout import select_checkpoint
    from agent_allocation.selection.seal import seal_selections
    results = []
    missing = 0
    for graph in graphs:
        candidates: dict[str, Candidate] = {}
        checks: dict[str, PublicCheckResult] = {}
        for node in graph.nodes:
            parsed = paths.parsed(node.node_id)
            if parsed.exists():
                payload = json.loads(parsed.read_text())
                if payload.get("candidate"):
                    candidates[node.node_id] = load_record(Candidate, payload["candidate"])
            check = paths.public_check(node.node_id)
            if check.exists():
                checks[node.node_id] = load_record(PublicCheckResult, json.loads(check.read_text()))
        for checkpoint in graph.checkpoints:
            if not all(c in candidates for c in checkpoint.candidate_ids):
                missing += 1
                continue
            task = tasks[graph.item_id]
            equivalence = math_equivalent if task.output_kind.value == "math" else None
            results.append(select_checkpoint(checkpoint, task, candidates, checks, seed=int(cfg["seed"]),
                                             math_equivalent=equivalence))
    seal = seal_selections(paths, partition, results)
    return {"selections": len(results), "checkpoints_missing_candidates": missing, "seal": seal}


def run_subprocess_kill_test(args_list: list[str], paths: RunPaths, kill_after_parsed: int, timeout_s: float = 900) -> dict:
    import signal
    import subprocess
    import time
    before = sum(1 for _ in (paths.run_dir / "parsed").rglob("*.json")) if (paths.run_dir / "parsed").exists() else 0
    proc = subprocess.Popen(args_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.monotonic()
    killed = False
    while proc.poll() is None and time.monotonic() - started < timeout_s:
        now = sum(1 for _ in (paths.run_dir / "parsed").rglob("*.json")) if (paths.run_dir / "parsed").exists() else 0
        if now - before >= kill_after_parsed:
            proc.send_signal(signal.SIGKILL)
            killed = True
            break
        time.sleep(0.5)
    proc.wait(timeout=60)
    return {"killed": killed, "returncode": proc.returncode}


async def run_pilot(cfg, paths: RunPaths, max_usd: Decimal, config_path: str, phase: str) -> int:
    from agent_allocation.apikey import load_api_key
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    if cfg["primary_model"] not in locks:
        raise RunRefused("no endpoint lock: the canary must pass first")
    gate1 = paths.gate_result("gate1", "canary")
    if not gate1.exists() or json.loads(gate1.read_text()).get("decision") != "advance":
        raise RunRefused("Gate 1 has not advanced")
    pilot_cap = min(max_usd, PILOT_CEILING_USD)
    waves, graphs = pilot_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "pilot", graphs)
    api_key = load_api_key()
    lock = locks[cfg["primary_model"]]

    if phase == "collect":          # subprocess entry used by the kill/resume test
        snap = provider_snapshot(paths, "pilot_collect")
        ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
        await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=True, max_usd=pilot_cap,
                       ceiling=ceiling, api_key=api_key)
        return 0

    report: dict[str, Any] = {"started_at": utcnow()}
    # 1) budget-stop test: a ceiling just above current spend must stop admissions without over-reserving
    settled = ledger_settled(paths)
    stop_ceiling = settled + BUDGET_STOP_TEST_CEILING_USD
    seq0 = last_journal_seq(paths)
    summary = await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=True, max_usd=pilot_cap,
                             ceiling=stop_ceiling, api_key=api_key)
    report["budget_stop"] = {"ceiling": str(stop_ceiling), "summary": str(summary),
                             "max_overshoot_usd": str(journal_replay_overshoot(paths, stop_ceiling, seq0))}
    # 2) crash/resume test: a subprocess continues the pilot and is SIGKILLed mid-flight
    cmd = [str(Path(sys.executable).parent / "allocation"), "pilot", "--config", config_path, "--run-dir",
           str(paths.run_dir), "--allow-paid", "--max-usd", str(pilot_cap), "--phase", "collect"]
    report["kill_test"] = run_subprocess_kill_test(cmd, paths, kill_after_parsed=40)
    # 3) resume to completion under the full pilot ceiling
    snap = provider_snapshot(paths, "pilot_resume")
    ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
    report["resume_summary"] = str(await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=True,
                                                  max_usd=pilot_cap, ceiling=ceiling, api_key=api_key))
    report["collected_at"] = utcnow()
    (paths.state_dir / "pilot_collection.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)


def run_grader(paths: RunPaths, workers: int = 4) -> int:
    cmd = [str(Path(sys.executable).parent / "allocation"), "grade", "--run-dir", str(paths.run_dir), "--resume",
           "--workers", str(workers)]
    return subprocess.run(cmd, check=False).returncode


def pilot_gate_inputs(cfg, paths: RunPaths, graphs: Sequence[ItemGraph], rows, amendments_used) -> dict[str, Any]:
    from agent_allocation.contracts import GradeResult, GradeStatus
    node_ids = {n.node_id for g in graphs for n in g.nodes}
    code_nodes = {n.node_id for g in graphs for n in g.nodes if n.benchmark == "mbpp_plus"}
    code_candidates = [nid for nid in code_nodes if paths.parsed(nid).exists()
                       and json.loads(paths.parsed(nid).read_text()).get("candidate")]
    public_rows = sum(1 for nid in code_candidates if paths.public_check(nid).exists())
    private_rows = sum(1 for nid in code_candidates if paths.grade(nid).exists())
    graded = [load_record(GradeResult, json.loads(paths.grade(nid).read_text()))
              for nid in node_ids if paths.grade(nid).exists()]
    collection = json.loads((paths.state_dir / "pilot_collection.json").read_text())
    return {
        "duplicate_terminal_nodes": duplicate_successful_dispatches(paths, node_ids),
        "budget_stop_max_overshoot_usd": Decimal(collection["budget_stop"]["max_overshoot_usd"]),
        "code_candidates": len(code_candidates), "code_public_rows": public_rows,
        "code_private_grades": private_rows,
        "evaluator_errors": sum(1 for g in graded if g.status == GradeStatus.EVALUATOR_ERROR),
        "graded_total": len(graded),
        "packet_envelope_violations": render_report_violations(paths),
        "amendments_used": amendments_used,
    }


def forecast_from_rows(cfg, paths: RunPaths, rows, locks, amendments) -> dict[str, Any]:
    from agent_allocation.forecast import core_forecast, fit_models, observations_from_rows, per_item_costs, wave_admission
    primary = locks[cfg["primary_model"]]
    models = fit_models(observations_from_rows([r for r in rows if r.spec.model == primary.model]))
    headroom = {b: headroom_for(cfg, b, amendments) for b in BENCHMARKS}
    costs = per_item_costs(models, grids_from_config(cfg), lock=primary, headroom_by_benchmark=headroom,
                           main_a=int(cfg["reasoning"]["main_a"]), high_a=int(cfg["reasoning"]["high_a"]))
    per_benchmark = {name: count // len(BENCHMARKS) for name, count in cfg["panels"].items()}
    core = core_forecast(costs, per_benchmark)
    spent = ledger_settled(paths)
    reserve = Decimal(str(cfg["budget"]["contingency_usd"]))
    snap = provider_snapshot(paths, "pilot_forecast")
    ceiling = active_ceiling(credit_ceiling(cfg), spent, snap)
    available = min(ceiling, Decimal(str(cfg["budget"]["soft_admission_ceiling_usd"]))) - spent
    admission = wave_admission(costs, n_waves=cfg["panels"]["main_items"] // 16,
                               extension_waves=cfg["panels"]["extension_items"] // 16,
                               high_waves=cfg["panels"]["high_items"] // 16, available_usd=available)
    return {"per_item_costs": {b: {k: str(v) for k, v in c.items()} for b, c in costs.items()},
            "core_forecast_usd": str(core), "spent_usd": str(spent), "reserve_usd": str(reserve),
            "core_plus_spent_plus_reserve_usd": str(core + spent + reserve),
            "active_ceiling_usd": str(ceiling), "available_for_core_usd": str(available),
            "account_balance_usd": snap["account_balance_usd"], "admission": admission}


def write_pilot_report(paths: RunPaths, gate, forecast: Mapping[str, Any], inputs: Mapping[str, Any],
                       selection: Mapping[str, Any]) -> None:
    lines = ["# PILOT_REPORT — openrouter_agent_allocation_v1", "",
             f"Evaluated {gate.evaluated_at}. **Gate 2 decision: `{gate.decision}`.**", "",
             "## Gate 2 criteria (frozen before the canary)", "", "| Criterion | Observed | Threshold | Pass |",
             "|---|---|---|---|"]
    for c in gate.criteria:
        lines.append(f"| {c.name} | `{c.observed}` | {c.threshold} | {'yes' if c.passed else '**no**'} |")
    lines += ["", "## Measured forecast and admission plan", "", "```json",
              json.dumps(forecast, indent=1, sort_keys=True), "```", "",
              "## Integrity checks", "", "```json", json.dumps({k: str(v) for k, v in inputs.items()}, indent=1,
                                                                   sort_keys=True), "```", "",
              "## Selection", "", "```json", json.dumps(selection, indent=1, sort_keys=True), "```", "",
              "Floor/ceiling effects and per-benchmark pilot accuracy are informational and never block.", ""]
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "PILOT_REPORT.md").write_text("\n".join(lines))


async def finish_pilot(cfg, paths: RunPaths, graphs, tasks, locks, amendments) -> int:
    from agent_allocation.gates import evaluate_gate2, write_gate
    grade_rc = run_grader(paths)
    selection = select_and_seal(cfg, paths, "pilot", graphs, tasks)
    rows = load_call_rows(paths, "pilot", {})
    metadata = await fetch_metadata(paths, [r.result.generation_id for r in rows], attempts=6)
    rows = load_call_rows(paths, "pilot", metadata)
    used = [json.loads(l).get("id") for l in paths.amendments.read_text().splitlines() if l.strip()] \
        if paths.amendments.exists() else []
    inputs = pilot_gate_inputs(cfg, paths, graphs, rows, used)
    forecast = forecast_from_rows(cfg, paths, rows, locks, amendments)
    gate = evaluate_gate2(rows, selector_rows_missing=selection["checkpoints_missing_candidates"],
                          core_forecast_usd=Decimal(forecast["core_plus_spent_plus_reserve_usd"]), **inputs)
    write_gate(paths, gate)
    (paths.state_dir / "pilot_forecast.json").write_text(json.dumps(forecast, indent=1, sort_keys=True))
    write_pilot_report(paths, gate, forecast, {**inputs, "grader_returncode": grade_rc}, selection)
    print(json.dumps({"gate2": gate.decision, "report": str(paths.reports_dir / "PILOT_REPORT.md")}))
    return {"advance": 0, "amend_headroom_once": 4}.get(gate.decision, 3)


def cmd_pilot(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation pilot")
    common_args(parser)
    parser.add_argument("--phase", choices=["all", "collect"], default="all")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths, max_usd = preflight(cfg, args)
    paths.ensure()
    return asyncio.run(run_pilot(cfg, paths, max_usd, args.config, args.phase))


# ------------------------------------------------------------------------------------------------------------------
# Main collection (Gates 3-4)
# ------------------------------------------------------------------------------------------------------------------

FORECAST_CHECK_WAVES = 4


def record_decision(paths: RunPaths, **fields: Any) -> None:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.run_decisions, "a") as handle:
        handle.write(json.dumps({"recorded_at": utcnow(), **fields}, sort_keys=True, default=str) + "\n")


def graphs_by_wave(cfg, paths: RunPaths, tasks, locks, amendments) -> dict[str, list[list[ItemGraph]]]:
    from agent_allocation.manifest import load_waves
    grids = grids_from_config(cfg)
    primary, replication = locks[cfg["primary_model"]], locks[cfg["replication_model"]]
    main_a, high_a = int(cfg["reasoning"]["main_a"]), int(cfg["reasoning"]["high_a"])
    out: dict[str, list[list[ItemGraph]]] = {"main": [], "high": [], "replication": []}
    for wave in load_waves(paths):
        main, high, rep = [], [], []
        for item in wave.item_ids:
            task = tasks[item]
            grid = grids["main_extension"] if "extension" in wave.panels else grids["main"]
            main.append(graph_for(grid, cfg=cfg, partition="main", task=task, lock=primary, base_budget=main_a,
                                  amendments=amendments))
            if "high" in wave.panels:
                high.append(graph_for(grids["high"], cfg=cfg, partition="high", task=task, lock=primary,
                                      base_budget=high_a, amendments=amendments))
            if "replication" in wave.panels:
                rep.append(graph_for(grids["replication"], cfg=cfg, partition="replication", task=task,
                                     lock=replication, base_budget=main_a, amendments=amendments))
        out["main"].append(main)
        if high:
            out["high"].append(high)
        if rep:
            out["replication"].append(rep)
    return out


def verify_authorization(paths: RunPaths) -> dict[str, Any]:
    if not paths.authorization.exists():
        raise RunRefused("no state/authorization.json: main collection requires a passed pilot and `allocation freeze`")
    auth = json.loads(paths.authorization.read_text())
    for lock_path, key in ((paths.config_lock, "config_lock_sha256"), (paths.prompt_lock, "prompt_lock_sha256"),
                           (paths.environment_lock, "environment_lock_sha256")):
        if json.loads(lock_path.read_text()).get("lock_sha256") != auth[key]:
            raise RunRefused(f"{lock_path.name} no longer matches the authorization")
    from agent_allocation.prompts.render import verify_prompt_lock
    record = {k: v for k, v in json.loads(paths.prompt_lock.read_text()).items() if k not in ("lock_sha256", "written_at")}
    try:
        verify_prompt_lock(record)
    except Exception as exc:  # noqa: BLE001
        raise RunRefused(f"prompt lock verification failed after freeze: {exc}") from exc
    return auth


def all_terminal(paths: RunPaths, waves: Sequence[Sequence[ItemGraph]]) -> bool:
    return all(paths.parsed(n.node_id).exists() for wave in waves for g in wave for n in g.nodes)


def stopped_early(summary: Any) -> str | None:
    reason = getattr(summary, "stop_reason", None)
    return None if reason in (None, "completed", "complete", "done") else str(reason)


async def run_main(cfg, paths: RunPaths, max_usd: Decimal) -> int:
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    from agent_allocation.forecast import (
        fit_models, item_cost, observations_from_rows, per_item_costs, wave_admission, SAFETY_MULTIPLIER,
    )
    auth = verify_authorization(paths)
    api_key = load_api_key()
    locks = read_endpoint_locks(paths)
    client = OpenRouterClient(api_key=api_key)
    try:
        drift = await verify_endpoint_drift(client, locks)
    finally:
        await client.aclose()
    if drift:
        record_decision(paths, phase="preflight", decision="stop_notify", reason="endpoint drift", detail=drift)
        raise RunRefused(f"endpoint drift: {drift}")
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    by_wave = graphs_by_wave(cfg, paths, tasks, locks, amendments)
    primary = locks[cfg["primary_model"]]
    soft = Decimal(str(cfg["budget"]["soft_admission_ceiling_usd"]))
    reserve = Decimal(str(cfg["budget"]["contingency_usd"]))

    # Phase A: the first waves establish a trustworthy forecast; extension CP/POOLED ride with their items.
    snap = provider_snapshot(paths, "run_phase_a")
    ceiling = min(active_ceiling(max_usd, ledger_settled(paths), snap), soft)
    summary = await dispatch(cfg, paths, by_wave["main"][:FORECAST_CHECK_WAVES], lock=primary, tasks=tasks,
                             allow_paid=True, max_usd=max_usd, ceiling=ceiling, api_key=api_key)
    record_decision(paths, phase="A", summary=str(summary), ceiling=str(ceiling))
    if stopped_early(summary):
        record_decision(paths, phase="A", decision="stop_notify", reason=stopped_early(summary))
        return 3

    # Phase B: re-forecast from main rows; admit remaining main waves and high-A waves under the cut order.
    rows = load_call_rows(paths, "main", {})
    models = fit_models(observations_from_rows([r for r in rows if r.spec.model == primary.model]))
    headroom = {b: headroom_for(cfg, b, amendments) for b in BENCHMARKS}
    costs = per_item_costs(models, grids_from_config(cfg), lock=primary, headroom_by_benchmark=headroom,
                           main_a=int(cfg["reasoning"]["main_a"]), high_a=int(cfg["reasoning"]["high_a"]))
    spent = ledger_settled(paths)
    snap = provider_snapshot(paths, "run_phase_b")
    ceiling = min(active_ceiling(max_usd, spent, snap), soft)
    n_waves = len(by_wave["main"])
    ext_waves = cfg["panels"]["extension_items"] // 16
    admission = wave_admission(costs, n_waves=n_waves - FORECAST_CHECK_WAVES,
                               extension_waves=max(ext_waves - FORECAST_CHECK_WAVES, 0),
                               high_waves=len(by_wave["high"]), available_usd=ceiling - spent)
    record_decision(paths, phase="B", admission=admission, spent=str(spent), ceiling=str(ceiling),
                    per_item_costs={b: {k: str(v) for k, v in c.items()} for b, c in costs.items()})
    main_rest = by_wave["main"][FORECAST_CHECK_WAVES:FORECAST_CHECK_WAVES + admission["main_waves"]]
    high = by_wave["high"][:admission["high_waves"]]
    interleaved: list[list[ItemGraph]] = []
    for index in range(max(len(main_rest), len(high))):
        if index < len(high):
            interleaved.append(high[index])
        if index < len(main_rest):
            interleaved.append(main_rest[index])
    summary = await dispatch(cfg, paths, interleaved, lock=primary, tasks=tasks, allow_paid=True, max_usd=max_usd,
                             ceiling=ceiling, api_key=api_key)
    record_decision(paths, phase="B", summary=str(summary))
    if stopped_early(summary):
        record_decision(paths, phase="B", decision="stop_notify", reason=stopped_early(summary))
        return 3

    # Phase C: optional Flash check only after core completion, if core + forecast + reserve fit.
    core_waves = by_wave["main"][:FORECAST_CHECK_WAVES] + main_rest + high
    if not all_terminal(paths, core_waves):
        record_decision(paths, phase="C", decision="skip_replication", reason="core incomplete")
        return 0
    replication_lock = locks[cfg["replication_model"]]
    flash_rows = [r for r in load_call_rows(paths, "canary", {}) if r.spec.model == replication_lock.model]
    flash_models = fit_models(observations_from_rows(flash_rows))
    grids = grids_from_config(cfg)
    flash_items = sum(len(w) for w in by_wave["replication"])
    flash_forecast = Decimal("0")
    for wave in by_wave["replication"]:
        for graph in wave:
            model = flash_models.get(graph.nodes[0].benchmark) or next(iter(flash_models.values()))
            flash_forecast += item_cost(model, grids["replication"], int(cfg["reasoning"]["main_a"]),
                                        headroom_for(cfg, graph.nodes[0].benchmark, amendments), replication_lock)
    flash_forecast *= SAFETY_MULTIPLIER
    spent = ledger_settled(paths)
    snap = provider_snapshot(paths, "run_phase_c")
    absolute = active_ceiling(max_usd, spent, snap)
    fits = spent + flash_forecast + reserve <= absolute
    record_decision(paths, phase="C", flash_items=flash_items, flash_forecast=str(flash_forecast), spent=str(spent),
                    absolute_ceiling=str(absolute), reserve=str(reserve),
                    decision="run_replication" if fits else "skip_replication")
    if not fits:
        return 0
    summary = await dispatch(cfg, paths, by_wave["replication"], lock=replication_lock, tasks=tasks, allow_paid=True,
                             max_usd=max_usd, ceiling=absolute - reserve, api_key=api_key)
    record_decision(paths, phase="C", summary=str(summary))
    return 3 if stopped_early(summary) else 0


def cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation run")
    common_args(parser)
    parser.add_argument("--resume", action="store_true", help="accepted for the spec's command contract; runs always resume")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths, max_usd = preflight(cfg, args)
    paths.ensure()
    return asyncio.run(run_main(cfg, paths, max_usd))


def cmd_seal(argv: list[str]) -> int:
    """Select and seal every checkpoint of the frozen partitions (after grading's public checks exist)."""
    parser = argparse.ArgumentParser(prog="allocation seal")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--partition", action="append", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    tasks = load_tasks(paths)
    locks = read_endpoint_locks(paths)
    by_wave = graphs_by_wave(cfg, paths, tasks, locks, load_amendments(paths))
    out = {}
    for partition in args.partition or ["main", "high", "replication"]:
        graphs = [g for wave in by_wave[partition] for g in wave]
        out[partition] = select_and_seal(cfg, paths, partition, graphs, tasks)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0
