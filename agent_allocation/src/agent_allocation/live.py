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
    snap = fetch(load_api_key())
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.state_dir / "provider_snapshots.jsonl", "a") as handle:
        handle.write(json.dumps({"source": source, **snap}, sort_keys=True) + "\n")
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
    from agent_allocation.prompts.render import render_messages
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

    def render(spec, parents):
        return render_messages(spec, tasks[spec.item_id], parents, seed=seed)

    def extract(result, spec):
        return extract_candidate(result, spec, tasks[spec.item_id])

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
    with open(paths.state_dir / "generation_metadata.jsonl", "a") as handle:
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
# Pilot and main run are wired after the M1 merge (next lead commit).
# ------------------------------------------------------------------------------------------------------------------

def cmd_pilot(argv: list[str]) -> int:
    raise RunRefused("pilot wiring lands after the M1 merge")


def cmd_run(argv: list[str]) -> int:
    raise RunRefused("run wiring lands after the pilot passes")
