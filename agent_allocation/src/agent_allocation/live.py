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
REHEARSAL_ENV = "ALLOCATION_REHEARSAL"   # set by `--rehearsal`: fake transport end to end, never sends a request


def rehearsal() -> bool:
    return os.environ.get(REHEARSAL_ENV) == "1"


def rehearsal_transport():
    from agent_allocation.api.fake import fake_openrouter_transport
    return fake_openrouter_transport()
CANARY_BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)
CANARY_CODE_BUDGETS = (1024, 4096)
PILOT_CEILING_USD = Decimal("5")
BUDGET_STOP_TEST_CEILING_USD = Decimal("0.10")     # ~150 real pilot calls, leaving ~300 for the kill/resume test
REHEARSAL_BUDGET_STOP_CEILING_USD = Decimal("0.03")


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
                if row.get("rule") in ("headroom_once", "headroom_owner_approved"):
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


async def ensure_endpoint_locks(paths: RunPaths, client, models: Sequence[str], *, allow_fetch: bool,
                                cfg: Mapping[str, Any] | None = None) -> dict[str, EndpointLock]:
    """Write-once. Missing models are fetched and appended only before the lock is frozen. ``cfg`` selects the
    endpoint (``provider.endpoint_tag``; forced mode requires it and locks under the forced parameter contract)."""
    from agent_allocation.api.endpoints import select_standard_endpoint
    from agent_allocation.api.forced import endpoint_selection
    selection = endpoint_selection(cfg)
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
        locks[model] = select_standard_endpoint(parsed, model, **selection, snapshot_path=str(snapshot),
                                                snapshot_sha256=sha256(raw).hexdigest(), fetched_at=stamp)
    if missing:
        payload = {"locks": {m: dump_record(l) for m, l in locks.items()},
                   "lock_hashes": {m: l.lock_hash for m, l in locks.items()}, "updated_at": utcnow()}
        paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
        paths.endpoint_lock.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return locks


async def verify_endpoint_drift(client, locks: Mapping[str, EndpointLock],
                                cfg: Mapping[str, Any] | None = None) -> list[str]:
    from agent_allocation.api.endpoints import EndpointBlocked, select_standard_endpoint
    from agent_allocation.api.forced import forced_enabled
    forced = cfg is not None and forced_enabled(cfg)
    drift = []
    for model, lock in locks.items():
        _raw, parsed = await client.list_endpoints(model)
        try:
            live = select_standard_endpoint(parsed, model, preferred_tag=lock.provider_tag, forced=forced)
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
    if rehearsal():
        spent = ledger_settled(paths)
        snap = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                "key": {"usage": float(spent), "limit": 200, "limit_remaining": float(200 - spent), "is_free_tier": False},
                "credits": {"total_credits": 100, "total_usage": float(spent)},
                "account_balance_usd": str(Decimal("100") - spent)}
        write_provider_usage(paths, source, snap)
        return snap
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
    return build_item_graph(grid, run_id=cfg.get("study_id", STUDY_ID), partition_id=partition, task=task, model=lock.model,
                            base_budget=base_budget, headroom=headroom_for(cfg, task.benchmark, amendments),
                            endpoint_lock_hash=lock.lock_hash, decoding_config_hash=decoding_config_hash(cfg),
                            prompt_version=prompt_version())


def standalone_spec(*, cfg: Mapping[str, Any], partition: str, task: PublicTask, lock: EndpointLock,
                    kind: GenerationKind, budget: int, root_slot: int = 0, draw_id: int = 0,
                    amendments: Mapping[str, int]) -> GenerationSpec:
    from agent_allocation.prompts.render import prompt_version
    return GenerationSpec(run_id=cfg.get("study_id", STUDY_ID), partition_id=partition, item_id=task.item_id, benchmark=task.benchmark,
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

class Runtime:
    """One client, ledger, state store and dispatcher reused across successive plans, so the admission-rate ramp and
    any 429 back-off carry over between collection phases on the same endpoint lock."""

    def __init__(self, cfg: Mapping[str, Any], paths: RunPaths, *, lock: EndpointLock,
                 tasks: Mapping[str, PublicTask], ceiling: Decimal, transport=None, api_key: str | None = None):
        from agent_allocation.api.forced import forced_enabled, make_generation_client
        from agent_allocation.extraction.final import extract_candidate
        from agent_allocation.prompts.render import render_messages_with_report
        from agent_allocation.runtime.ledger import CollectionLedger
        from agent_allocation.runtime.scheduler import Dispatcher
        from agent_allocation.storage.journal import LedgerJournal
        from agent_allocation.storage.state import StateStore

        self.cfg, self.paths, self.lock = cfg, paths, lock
        if rehearsal():
            transport, api_key = rehearsal_transport(), None
        scheduler = cfg["scheduler"]
        # forced.enabled -> ForcedClient (THINK + ANSWER over /completions); otherwise the chat client, unchanged.
        self.forced = forced_enabled(cfg)
        self.gate0: Any = None
        self.client = make_generation_client(cfg, api_key=api_key, transport=transport,
                                             connect_timeout=float(scheduler["connect_timeout_seconds"]),
                                             read_timeout=float(scheduler["read_timeout_seconds"]))
        self.ledger = CollectionLedger(LedgerJournal(paths.ledger_journal), ceiling_usd=ceiling,
                                       max_unresolved_usd=Decimal(str(cfg["budget"]["max_unresolved_spend_usd"])))
        self.store = (StateStore.recover(paths) if paths.state_db_backup.exists() or paths.ledger_journal.exists()
                      else StateStore(paths))
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
                                         "rendered_chars": report.get("rendered_chars"),
                                         "recorded_at": utcnow()}) + "\n")
            return messages

        def extract(result, spec):
            return extract_candidate(result, spec, tasks[spec.item_id])

        self.dispatcher = Dispatcher(config=cfg, paths=paths, store=self.store, ledger=self.ledger,
                                     client=self.client, lock=lock, render=render, extract=extract)

    async def run(self, waves: Sequence[Sequence[ItemGraph]], *, allow_paid: bool, max_usd: Decimal,
                  ceiling: Decimal | None = None):
        from agent_allocation.protocols.grid import check_budgets
        from agent_allocation.runtime.scheduler import AdmissionPlan
        if ceiling is not None:
            self.ledger.set_active_ceiling(min(ceiling, max_usd))
        if self.forced and allow_paid:
            await self.forced_gate0()
        for wave in waves:
            for graph in wave:
                check_budgets(graph, max_completion_tokens=self.lock.max_completion_tokens,
                              requested_reasoning_max=int(self.cfg["reasoning"]["requested_reasoning_max"]))
        summary = await self.dispatcher.run(AdmissionPlan(waves=tuple(tuple(w) for w in waves)),
                                            allow_paid=allow_paid, max_usd=max_usd)
        self.store.backup()
        return summary

    async def forced_gate0(self):
        """Gate 0 once per Runtime before any paid forced dispatch: banana, ``</think>``-id, forced-2,048 and ANSWER
        probes against the locked endpoint, charged through the ledger. A failure refuses to dispatch."""
        if self.gate0 is not None and self.gate0.passed:
            return self.gate0
        from agent_allocation.api.forced import forced_endpoint_gate
        from agent_allocation.gates import write_gate
        result = await forced_endpoint_gate(self.client, self.lock, self.cfg, ledger=self.ledger,
                                            evidence_path=self.paths.reports_dir / "GATE0_FORCED_ENDPOINT.json")
        write_gate(self.paths, result)
        self.gate0 = result
        if not result.passed:
            failing = [c.name for c in result.criteria if not c.passed]
            raise RunRefused(f"Gate 0 (forced endpoint {self.lock.provider_tag}) failed: {failing}; nothing dispatched")
        return result

    async def aclose(self) -> None:
        try:
            await self.client.aclose()
        finally:
            self.store.backup()


async def dispatch(cfg: Mapping[str, Any], paths: RunPaths, waves: Sequence[Sequence[ItemGraph]], *,
                   lock: EndpointLock, tasks: Mapping[str, PublicTask], allow_paid: bool, max_usd: Decimal,
                   ceiling: Decimal, transport=None, api_key: str | None = None):
    runtime = Runtime(cfg, paths, lock=lock, tasks=tasks, ceiling=min(ceiling, max_usd), transport=transport,
                      api_key=api_key)
    try:
        return await runtime.run(waves, allow_paid=allow_paid, max_usd=max_usd)
    finally:
        await runtime.aclose()


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
    shard_arg(parser)


def shard_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard", default=None,
                        help="provider shard id (config shards[]); --run-dir then names the run root and the shard "
                             "runs in <run_root>/shards/<id> under its derived config")


def resolve_run(cfg: Mapping[str, Any], args) -> tuple[dict[str, Any], RunPaths]:
    """(config, paths) for a verb: the shard's derived config and run dir with --shard, else the config as given."""
    from agent_allocation.shards import ShardError, resolve_shard
    try:
        return resolve_shard(cfg, args.run_dir, getattr(args, "shard", None))
    except ShardError as exc:
        raise RunRefused(str(exc)) from exc


def preflight(cfg: Mapping[str, Any], args) -> tuple[dict[str, Any], RunPaths, Decimal]:
    cfg, paths = resolve_run(cfg, args)
    problems = validate_config(cfg)
    if problems:
        raise RunRefused(f"config invalid: {problems}")
    if not args.allow_paid:
        raise RunRefused("live command without --allow-paid: nothing is sent (use `allocation plan --dry-run`)")
    return cfg, paths, checked_max_usd(cfg, args.max_usd)


def shard_items(cfg: Mapping[str, Any], paths: RunPaths) -> set[str] | None:
    """The items a shard config may build graphs for (``None``: unsharded)."""
    from agent_allocation.shards import shard_filter
    return shard_filter(cfg, paths)



# ------------------------------------------------------------------------------------------------------------------
# Canary (Gate 1)
# ------------------------------------------------------------------------------------------------------------------

def canary_graphs(cfg, paths, tasks, locks, amendments) -> tuple[list[ItemGraph], list[ItemGraph]]:
    primary = locks[cfg["primary_model"]]
    replication = locks.get(cfg["replication_model"]) if cfg.get("replication_model") else None
    dev = dev_ids(paths)
    text_items = [dev[benchmark][0] for benchmark in BENCHMARKS]
    lite: list[ItemGraph] = []
    for item in text_items:
        nodes = [standalone_spec(cfg=cfg, partition="canary", task=tasks[item], lock=primary, kind=GenerationKind.ROOT,
                                 budget=a, amendments=amendments) for a in CANARY_BUDGETS]
        lite.append(nodes_only_graph(tasks[item], "canary", nodes, "canary_budget_sweep"))
    flash = []
    for item in ((dev["aime"][0], dev["gpqa_diamond"][0]) if replication is not None else ()):
        node = standalone_spec(cfg=cfg, partition="canary", task=tasks[item], lock=replication,
                               kind=GenerationKind.ROOT, budget=1024, amendments=amendments)
        flash.append(nodes_only_graph(tasks[item], "canary", [node], "canary_flash"))
    return lite, flash


async def fetch_metadata(paths: RunPaths, generation_ids: Sequence[str], *, attempts: int = 12, delay_s: float = 10.0):
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    if rehearsal():
        return {}
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
            handle.write(json.dumps({"generation_id": gid, "data": data, "fetched_at": utcnow()}, sort_keys=True,
                                    default=str) + "\n")
    return found


def forced_row_evidence(paths: RunPaths, result, metadata: Mapping[str, Any]) -> tuple[Any, Any]:
    """(raw payload, /generation metadata) for a call row. A forced result's metadata is THINK + ANSWER merged
    (``None`` until both are known); a chat result keeps its single-id metadata and no raw payload."""
    from agent_allocation.api.forced import forced_generation_ids, is_forced_usage
    if not is_forced_usage(result.usage) or not result.raw_response_path:
        return None, metadata.get(result.generation_id)
    from agent_allocation.api.reconcile import merge_forced_metadata
    from agent_allocation.storage.raw import read_raw_response
    raw = read_raw_response(paths, result.raw_response_path)
    think_id, answer_id = forced_generation_ids(raw)
    think, answer = metadata.get(think_id), metadata.get(answer_id)
    return raw, (merge_forced_metadata(think, answer) if think is not None and answer is not None else None)


def forced_think_ids(rows) -> list[str]:
    """THINK generation ids of forced rows (their /generation metadata is fetched alongside the ANSWER ids)."""
    from agent_allocation.api.forced import forced_generation_ids
    return [gid for r in rows if r.raw_payload is not None
            for gid in [forced_generation_ids(r.raw_payload)[0]] if gid]


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
        raw, meta = forced_row_evidence(paths, result, metadata)
        rows.append(CallRow(spec, result, candidate, meta, raw))
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


def _before_snapshot(paths: RunPaths, stage: str) -> tuple[Decimal, Decimal]:
    """(key usage, account balance) from the latest provider_usage lines of ``stage``."""
    key_usage, balance = None, None
    for line in paths.provider_usage.read_text().splitlines():
        row = json.loads(line)
        if row.get("stage") != stage:
            continue
        if row["endpoint"] == "key":
            key_usage = Decimal(str(row["data"].get("usage", 0)))
        elif row["endpoint"] == "credits":
            balance = Decimal(str(row["data"]["total_credits"])) - Decimal(str(row["data"]["total_usage"]))
    if key_usage is None:
        raise RunRefused(f"no provider snapshot for stage {stage}")
    return key_usage, balance


async def evaluate_canary(cfg, paths: RunPaths) -> int:
    from agent_allocation.api.canary import canary_report
    from agent_allocation.gates import evaluate_gate1, write_gate
    locks = read_endpoint_locks(paths)
    before = json.loads((paths.state_dir / "canary_before.json").read_text()) \
        if (paths.state_dir / "canary_before.json").exists() else None
    key_before = Decimal(before["key_usage_before"]) if before else _before_snapshot(paths, "canary_before")[0]
    settled_before = Decimal(before["settled_before"]) if before else Decimal("0")
    rows = load_call_rows(paths, "canary", {})
    known = {}
    if paths.generation_metadata.exists():
        for line in paths.generation_metadata.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                known[row["generation_id"]] = row["data"]
    missing = [g for g in [r.result.generation_id for r in rows] + forced_think_ids(rows) if g and g not in known]
    if missing:
        known.update(await fetch_metadata(paths, missing))
    rows = load_call_rows(paths, "canary", known)
    await asyncio.sleep(20)
    after = provider_snapshot(paths, "canary_after")
    key_delta = Decimal(str(after["key"].get("usage", 0))) - key_before
    settled_delta = ledger_settled(paths) - settled_before
    result = evaluate_gate1(rows, locks, key_usage_delta_usd=key_delta, ledger_settled_usd=settled_delta)
    write_gate(paths, result)
    report = canary_report([(r.spec, r.result, r.metadata) for r in rows])
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "CANARY_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(json.dumps({"gate": dump_record(result), "calls": len(rows), "key_delta": str(key_delta),
                      "ledger_settled_delta": str(settled_delta)}, indent=1, default=str))
    return 0 if result.decision == "advance" else 3


async def run_canary(cfg, paths: RunPaths, max_usd: Decimal) -> int:
    from agent_allocation.api.client import OpenRouterClient
    from agent_allocation.apikey import load_api_key
    api_key = load_api_key()
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    from agent_allocation.api.forced import make_generation_client
    client = make_generation_client(cfg, api_key=api_key)       # ForcedClient in forced mode: Gate 0 needs it
    try:
        models = [m for m in (cfg["primary_model"], cfg.get("replication_model")) if m]
        locks = await ensure_endpoint_locks(paths, client, models, allow_fetch=True, cfg=cfg)
        gate0 = await run_gate0(cfg, paths, client, locks[cfg["primary_model"]])
    finally:
        await client.aclose()
    if gate0 is not None and not gate0["passed"]:
        record_decision(paths, phase="gate0", decision="stop_notify", detail=gate0)
        raise RunRefused(f"Gate 0 (forced endpoint screen) failed for {cfg['provider'].get('endpoint_tag')}: "
                         f"{gate0.get('detail')}")
    before = provider_snapshot(paths, "canary_before")
    settled_before = ledger_settled(paths)
    (paths.state_dir / "canary_before.json").write_text(json.dumps(
        {"key_usage_before": str(before["key"].get("usage", 0)), "settled_before": str(settled_before),
         "recorded_at": utcnow()}, sort_keys=True))
    lite, flash = canary_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "canary", [*lite, *flash])
    ceiling = min(active_ceiling(max_usd, settled_before, before), settled_before + Decimal("0.50"))
    for graphs, model in ((lite, cfg["primary_model"]), (flash, cfg.get("replication_model"))):
        if not graphs:
            continue
        summary = await dispatch(cfg, paths, [graphs], lock=locks[model], tasks=tasks, allow_paid=True,
                                 max_usd=max_usd, ceiling=ceiling, api_key=api_key)
        print(json.dumps({"model": model, "summary": str(summary)}, default=str))
    return await evaluate_canary(cfg, paths)


async def run_gate0(cfg: Mapping[str, Any], paths: RunPaths, client, lock: EndpointLock) -> dict[str, Any] | None:
    """Gate 0 (forced mode only): the per-endpoint screen -- logit_bias honoured (banana test), the ``</think>`` id
    banned, forced length exact with finish_reason=length, and an ANSWER call that returns a ``<final>``. WP-1 owns
    the check itself (``forced_endpoint_gate(client, lock, cfg)``); this records its verdict per shard run dir.
    Returns ``None`` outside forced mode. Missing implementation fails closed (except in a rehearsal)."""
    from agent_allocation.config import forced_mode
    if not forced_mode(cfg):
        return None
    gate = None
    for module in ("agent_allocation.api.forced", "agent_allocation.api.canary", "agent_allocation.gates"):
        try:
            gate = getattr(__import__(module, fromlist=["forced_endpoint_gate"]), "forced_endpoint_gate", None)
        except ImportError:
            gate = None
        if gate is not None:
            break
    if gate is None:
        verdict = {"passed": rehearsal(), "detail": "forced_endpoint_gate is not available in this build"}
    else:
        raw = gate(client, lock, cfg)
        if asyncio.iscoroutine(raw):
            raw = await raw
        verdict = {"passed": gate0_passed(raw), "detail": raw if isinstance(raw, (dict, list, str)) else str(raw)}
    verdict.update({"endpoint_tag": cfg["provider"].get("endpoint_tag"), "shard": (cfg.get("shard") or {}).get("id"),
                    "model": lock.model, "evaluated_at": utcnow()})
    target = paths.gate_result("gate0", "endpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(verdict, indent=1, sort_keys=True, default=str))
    return verdict


def gate0_passed(result: Any) -> bool:
    """Accept the shapes a gate verdict may take: bool, {passed|decision}, or an object with those attributes."""
    if isinstance(result, bool):
        return result
    for key in ("passed", "ok"):
        value = result.get(key) if isinstance(result, Mapping) else getattr(result, key, None)
        if isinstance(value, bool):
            return value
    decision = result.get("decision") if isinstance(result, Mapping) else getattr(result, "decision", None)
    return decision in ("advance", "pass", "passed")


def cmd_canary(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation canary")
    common_args(parser)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="re-evaluate Gate 1 from stored canary artifacts without dispatching")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg, paths, max_usd = preflight(cfg, args)
    paths.ensure()
    if args.evaluate_only:
        return asyncio.run(evaluate_canary(cfg, paths))
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
    high_a = cfg["reasoning"].get("high_a")
    allowed = shard_items(cfg, paths)
    for benchmark in BENCHMARKS:
        for item in dev[benchmark]:
            if allowed is not None and item not in allowed:
                continue
            task = tasks[item]
            grid = grids["pilot_r2"] if item in r2 else grids["pilot_r1"]
            graph = graph_for(grid, cfg=cfg, partition="pilot", task=task, lock=primary,
                              base_budget=int(cfg["reasoning"]["main_a"]), amendments=amendments)
            extra = []
            if item in deep_items:
                extra.append(standalone_spec(cfg=cfg, partition="pilot", task=task, lock=primary,
                                             kind=GenerationKind.DEEP, budget=int(cfg["reasoning"]["main_a"]) * 16,
                                             amendments=amendments))
            if item in high_items and high_a is not None:
                extra += [standalone_spec(cfg=cfg, partition="pilot", task=task, lock=primary, kind=GenerationKind.ROOT,
                                          budget=int(cfg["reasoning"]["high_a"]), root_slot=j, amendments=amendments)
                          for j in range(2)]
            graph = merge_graphs(graph, extra)
            (wave_r2 if item in r2 else wave_r1).append(graph)
            all_graphs.append(graph)
    return [w for w in (wave_r2, wave_r1) if w], all_graphs


def render_report_violations(paths: RunPaths) -> int:
    target = paths.context_limit_events
    if not target.exists():
        return 0
    return len({json.loads(line)["node_id"] for line in target.read_text().splitlines() if line.strip()})


def select_and_seal(cfg, paths: RunPaths, partition: str, graphs: Sequence[ItemGraph], tasks) -> dict[str, Any]:
    """Delegates to agent 06's ``select_partition``: reads the partition's policy manifest, treats confirmed
    terminal-without-output members as absent (07-R12), waits for code public checks, memoizes Math-Verify
    equivalence for MATH checkpoints only, and seals an append-only batch. ``graphs``/``tasks`` are kept for callers."""
    from agent_allocation.grading.text.math_equiv import math_equivalent
    from agent_allocation.selection.partition import select_partition
    outcome = select_partition(paths, partition, seed=int(cfg["seed"]), math_equivalent=math_equivalent, seal=True)
    return {"selections": len(outcome.results), "checkpoints_missing_candidates": len(outcome.pending_policy_ids),
            "pending_policy_ids_sample": list(outcome.pending_policy_ids[:5]), "seal": outcome.sealed_sha256}


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
    if rehearsal() and not paths.endpoint_lock.exists():
        from agent_allocation.api.fake import fake_endpoint_lock
        fake = {m: fake_endpoint_lock(m) for m in (cfg["primary_model"], cfg.get("replication_model")) if m}
        paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
        paths.endpoint_lock.write_text(json.dumps({"locks": {m: dump_record(l) for m, l in fake.items()},
                                                   "rehearsal": True}, indent=1, sort_keys=True))
    locks = read_endpoint_locks(paths)
    if cfg["primary_model"] not in locks:
        raise RunRefused("no endpoint lock: the canary must pass first")
    gate1 = paths.gate_result("gate1", "canary")
    if not rehearsal() and (not gate1.exists() or json.loads(gate1.read_text()).get("decision") != "advance"):
        raise RunRefused("Gate 1 has not advanced")
    pilot_cap = min(max_usd, PILOT_CEILING_USD)
    waves, graphs = pilot_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "pilot", graphs)
    api_key = None if rehearsal() else load_api_key()
    paid = not rehearsal()
    lock = locks[cfg["primary_model"]]

    if phase == "finish":           # re-evaluate Gate 2 from stored pilot artifacts without dispatching
        return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)
    if phase == "amend":            # the single pre-authorized headroom amendment, then re-collect and re-evaluate
        return await amend_and_rerun_pilot(cfg, paths, max_usd, config_path)
    if phase == "collect":          # subprocess entry used by the kill/resume test
        snap = provider_snapshot(paths, "pilot_collect")
        ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
        await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=paid, max_usd=pilot_cap,
                       ceiling=ceiling, api_key=api_key)
        return 0

    report: dict[str, Any] = {"started_at": utcnow()}
    # 1) budget-stop test: a ceiling just above current spend must stop admissions without over-reserving
    settled = ledger_settled(paths)
    stop_ceiling = settled + (REHEARSAL_BUDGET_STOP_CEILING_USD if rehearsal() else BUDGET_STOP_TEST_CEILING_USD)
    seq0 = last_journal_seq(paths)
    summary = await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=paid, max_usd=pilot_cap,
                             ceiling=stop_ceiling, api_key=api_key)
    report["budget_stop"] = {"ceiling": str(stop_ceiling), "summary": str(summary),
                             "max_overshoot_usd": str(journal_replay_overshoot(paths, stop_ceiling, seq0))}
    # 2) crash/resume test: a subprocess continues the pilot and is SIGKILLed mid-flight
    cmd = [str(Path(sys.executable).parent / "allocation"), "pilot", "--config", subprocess_config(cfg, paths, config_path),
           "--run-dir", str(paths.run_dir), "--allow-paid", "--max-usd", str(pilot_cap), "--phase", "collect"]
    if rehearsal():
        cmd.append("--rehearsal")
    report["kill_test"] = run_subprocess_kill_test(cmd, paths, kill_after_parsed=40)
    # 3) resume to completion under the full pilot ceiling
    snap = provider_snapshot(paths, "pilot_resume")
    ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
    report["resume_summary"] = str(await dispatch(cfg, paths, waves, lock=lock, tasks=tasks, allow_paid=paid,
                                                  max_usd=pilot_cap, ceiling=ceiling, api_key=api_key))
    report["collected_at"] = utcnow()
    (paths.state_dir / "pilot_collection.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)


def subprocess_config(cfg: Mapping[str, Any], paths: RunPaths, config_path: str) -> str:
    """Config path a child ``allocation`` process should load: a shard's derived config (written to its state dir,
    so the child runs in the shard dir without --shard), else the caller's config file."""
    if not cfg.get("shard"):
        return config_path
    from agent_allocation.shards import SHARD_CONFIG_NAME
    target = paths.state_dir / SHARD_CONFIG_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    return str(target)


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
                           main_a=int(cfg["reasoning"]["main_a"]), high_a=cfg["reasoning"].get("high_a"))
    per_benchmark = forecast_panels(cfg, paths)
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


def forecast_panels(cfg: Mapping[str, Any], paths: RunPaths) -> dict[str, Any]:
    """Items per benchmark the core forecast prices. A shard prices only its own items (its waves, which hold just
    them): the mean per benchmark of its main / extension items, so its forecast is comparable to its budget slice."""
    per = {name: count // len(BENCHMARKS) for name, count in cfg["panels"].items()}
    if not cfg.get("shard") or not paths.waves.exists():
        return per
    from agent_allocation.manifest import load_waves
    waves = load_waves(paths)
    main = sum(len(w.item_ids) for w in waves if "main" in w.panels)
    ext = sum(len(w.item_ids) for w in waves if "extension" in w.panels)
    n = Decimal(len(BENCHMARKS))
    return {**per, "main_items": Decimal(main) / n, "extension_items": Decimal(ext) / n, "high_items": 0,
            "replication_items": 0}


def write_pilot_report(paths: RunPaths, gate, forecast: Mapping[str, Any], inputs: Mapping[str, Any],
                       selection: Mapping[str, Any]) -> None:
    lines = [f"# PILOT_REPORT — {_study_label(paths)}", "",
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


def _study_label(paths: RunPaths) -> str:
    """Study id (and shard) of a run dir for report titles, from the shard config when present."""
    from agent_allocation.shards import SHARD_CONFIG_NAME
    target = paths.state_dir / SHARD_CONFIG_NAME
    if target.is_file():
        cfg = json.loads(target.read_text())
        return f"{cfg.get('study_id')} (shard {cfg.get('shard', {}).get('id')})"
    return STUDY_ID


async def finish_pilot(cfg, paths: RunPaths, graphs, tasks, locks, amendments) -> int:
    from agent_allocation.gates import evaluate_gate2, write_gate
    grade_rc = run_grader(paths)
    selection = select_and_seal(cfg, paths, "pilot", graphs, tasks)
    rows = load_call_rows(paths, "pilot", {})
    known: dict[str, Any] = {}
    if paths.generation_metadata.exists():
        for line in paths.generation_metadata.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                known[row["generation_id"]] = row["data"]
    missing = [g for g in [r.result.generation_id for r in rows] + forced_think_ids(rows) if g and g not in known]
    if missing:
        known.update(await fetch_metadata(paths, missing, attempts=6))
    rows = load_call_rows(paths, "pilot", known)
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


async def amend_and_rerun_pilot(cfg, paths: RunPaths, max_usd: Decimal, config_path: str) -> int:
    from agent_allocation.config import CODE_BENCHMARKS
    from agent_allocation.contracts import GateResult
    from agent_allocation.gates import HEADROOM_AMENDMENT, family_of
    gate2 = load_record(GateResult, json.loads(paths.gate_result("gate2", "pilot").read_text()))
    if gate2.decision != "amend_headroom_once":
        raise RunRefused(f"Gate 2 decision is {gate2.decision!r}, not amend_headroom_once")
    failing = {c.name.rsplit("_truncated_without_final", 1)[0] for c in gate2.criteria
               if not c.passed and c.name.endswith("_truncated_without_final")}
    families = {family_of(b) for b in failing}
    used = set()
    if paths.amendments.exists():
        used = {json.loads(l).get("id") for l in paths.amendments.read_text().splitlines() if l.strip()}
    with open(paths.amendments, "a") as handle:
        for family in sorted(families):
            amendment_id = f"headroom_{family}"
            if amendment_id in used:
                raise RunRefused(f"{amendment_id} was already used; a second amendment needs the owner")
            before, after = HEADROOM_AMENDMENT[family]
            scope = sorted(b for b in BENCHMARKS if (b in CODE_BENCHMARKS) == (family == "code"))
            handle.write(json.dumps({
                "id": amendment_id, "recorded_at": utcnow(), "rule": "headroom_once", "scope": scope,
                "before": before, "after": after, "trigger_benchmarks": sorted(failing),
                "reason": "Gate 2 truncation-without-final above 3%; pre-authorized single headroom doubling "
                          "(approved plan, frozen before the canary)",
                "gate2_evaluated_at": gate2.evaluated_at}, sort_keys=True) + "\n")
    record_decision(paths, phase="pilot_amend", families=sorted(families), trigger=sorted(failing))
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    waves, graphs = pilot_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "pilot", graphs)
    pilot_cap = min(max_usd, PILOT_CEILING_USD)
    snap = provider_snapshot(paths, "pilot_amend")
    ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
    summary = await dispatch(cfg, paths, waves, lock=locks[cfg["primary_model"]], tasks=tasks,
                             allow_paid=not rehearsal(), max_usd=pilot_cap, ceiling=ceiling,
                             api_key=None if rehearsal() else __import__("agent_allocation.apikey", fromlist=["x"]).load_api_key())
    record_decision(paths, phase="pilot_amend_collect", summary=str(summary))
    supersede_pilot_readouts(paths, reason="headroom amendment re-collected text pilot cells; checkpoint ids are "
                                           "headroom-independent, so the earlier sealed pilot readouts are superseded")
    return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)


async def owner_headroom_and_rerun_pilot(cfg, paths: RunPaths, max_usd: Decimal, *, family: str, headroom: int,
                                        approval: str) -> int:
    """Owner-approved headroom change beyond the single pre-authorized amendment (never automatic)."""
    from agent_allocation.config import CODE_BENCHMARKS
    if not approval.strip():
        raise RunRefused("an owner approval reference is required")
    current = load_amendments(paths)
    scope = sorted(b for b in BENCHMARKS if (b in CODE_BENCHMARKS) == (family == "code"))
    before = sorted({headroom_for(cfg, b, current) for b in scope})
    with open(paths.amendments, "a") as handle:
        handle.write(json.dumps({"id": f"headroom_{family}_owner_{headroom}", "recorded_at": utcnow(),
                                 "rule": "headroom_owner_approved", "scope": scope, "before": before[-1],
                                 "after": int(headroom), "approval": approval,
                                 "reason": "Gate 2 truncation still above 3% after the pre-authorized amendment; "
                                           "owner approved a further headroom change"}, sort_keys=True) + "\n")
    record_decision(paths, phase="pilot_owner_headroom", family=family, headroom=headroom, approval=approval)
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    waves, graphs = pilot_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "pilot", graphs)
    pilot_cap = min(max_usd, PILOT_CEILING_USD)
    from agent_allocation.apikey import load_api_key
    snap = provider_snapshot(paths, "pilot_owner_headroom")
    ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
    summary = await dispatch(cfg, paths, waves, lock=locks[cfg["primary_model"]], tasks=tasks, allow_paid=True,
                             max_usd=pilot_cap, ceiling=ceiling, api_key=load_api_key())
    record_decision(paths, phase="pilot_owner_headroom_collect", summary=str(summary))
    supersede_pilot_readouts(paths, reason=f"owner-approved {family} headroom {headroom}: text pilot cells re-collected")
    return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)


async def recollect_pilot(cfg, paths: RunPaths, max_usd: Decimal, *, approval: str, reason: str) -> int:
    """Re-collect every pilot cell under the current prompts and amendments (node ids change with the prompt version),
    supersede the earlier pilot readouts, and re-evaluate Gate 2. Requires an owner approval reference."""
    if not approval.strip():
        raise RunRefused("an owner approval reference is required")
    record_decision(paths, phase="pilot_recollect", approval=approval, reason=reason)
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    waves, graphs = pilot_graphs(cfg, paths, tasks, locks, amendments)
    write_jobs(paths, "pilot", graphs)
    pilot_cap = min(max_usd, PILOT_CEILING_USD)
    from agent_allocation.apikey import load_api_key
    snap = provider_snapshot(paths, "pilot_recollect")
    ceiling = active_ceiling(pilot_cap, ledger_settled(paths), snap)
    summary = await dispatch(cfg, paths, waves, lock=locks[cfg["primary_model"]], tasks=tasks, allow_paid=True,
                             max_usd=pilot_cap, ceiling=ceiling, api_key=load_api_key())
    record_decision(paths, phase="pilot_recollect_collect", summary=str(summary))
    supersede_pilot_readouts(paths, reason=reason)
    return await finish_pilot(cfg, paths, graphs, tasks, locks, amendments)


def owner_accept_gate2(paths: RunPaths, approval: str) -> int:
    """Owner accepts the remaining Gate 2 failures as documented limitations; freeze then treats Gate 2 as advanced."""
    from agent_allocation.contracts import GateResult
    from agent_allocation.gates import write_gate
    if not approval.strip():
        raise RunRefused("an owner approval reference is required")
    gate = load_record(GateResult, json.loads(paths.gate_result("gate2", "pilot").read_text()))
    failing = [c.name for c in gate.criteria if not c.passed]
    supersede_pilot_readouts(paths, reason="owner acceptance of remaining Gate 2 failures")
    accepted = replace(gate, decision="advance_owner_override", evaluated_at=utcnow())
    write_gate(paths, accepted)
    record_decision(paths, phase="gate2_owner_override", failing=failing, approval=approval)
    return 0


def supersede_pilot_readouts(paths: RunPaths, *, reason: str) -> None:
    """Pilot selections and the Gate 2 record are gate evidence, not study results: move them aside (never rewrite a
    sealed batch in place) before re-evaluating from the same or amended stored outputs."""
    import shutil
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []
    selections = paths.selections_dir("pilot")
    if selections.exists() and "owner acceptance" not in reason:
        target = paths.run_dir / "selections_superseded" / f"pilot_{stamp}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(selections), str(target))
        moved.append(f"{paths.relative(selections)} -> {paths.relative(target)}")
    gate = paths.gate_result("gate2", "pilot")
    if gate.exists():
        target = gate.with_name(f"gate2_pilot.superseded_{stamp}.json")
        shutil.copy2(gate, target)
        moved.append(f"{paths.relative(gate)} copied to {paths.relative(target)}")
    record_decision(paths, phase="pilot_supersede", moved=moved, reason=reason)


def cmd_pilot(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation pilot")
    common_args(parser)
    parser.add_argument("--phase", choices=["all", "collect", "finish", "amend", "owner-headroom", "owner-accept",
                                            "recollect"],
                        default="all")
    parser.add_argument("--family", choices=["text", "code"], default="text")
    parser.add_argument("--headroom", type=int, default=None)
    parser.add_argument("--approval", default="", help="owner approval reference (message timestamp/quote)")
    parser.add_argument("--rehearsal", action="store_true",
                        help="run the whole pilot flow on the fake transport in a scratch run dir; sends nothing")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.rehearsal:
        if RunPaths.for_run(args.run_dir).run_dir.resolve() == RunPaths.for_run(None).run_dir.resolve():
            raise RunRefused("--rehearsal must use a scratch --run-dir, never the study run dir")
        os.environ[REHEARSAL_ENV] = "1"
    cfg, paths, max_usd = preflight(cfg, args)
    paths.ensure()
    if args.phase == "owner-headroom":
        if not args.headroom:
            raise RunRefused("--headroom is required with --phase owner-headroom")
        return asyncio.run(owner_headroom_and_rerun_pilot(cfg, paths, max_usd, family=args.family,
                                                          headroom=args.headroom, approval=args.approval))
    if args.phase == "owner-accept":
        return owner_accept_gate2(paths, args.approval)
    if args.phase == "recollect":
        return asyncio.run(recollect_pilot(cfg, paths, max_usd, approval=args.approval,
                                           reason="owner-approved prompt amendment prompt_system_v2: all pilot cells "
                                                  "re-collected under the new prompt version"))
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
    """Item graphs per partition and wave. Disabled partitions (no grid: high_a / replication_model null) stay empty;
    a shard config builds only its own items (waves left empty by the filter are dropped)."""
    from agent_allocation.manifest import load_waves
    grids = grids_from_config(cfg)
    primary = locks[cfg["primary_model"]]
    replication = locks.get(cfg["replication_model"]) if cfg.get("replication_model") else None
    main_a = int(cfg["reasoning"]["main_a"])
    high_a = int(cfg["reasoning"]["high_a"]) if cfg["reasoning"].get("high_a") is not None else None
    allowed = shard_items(cfg, paths)
    out: dict[str, list[list[ItemGraph]]] = {"main": [], "high": [], "replication": []}
    for wave in load_waves(paths):
        main, high, rep = [], [], []
        for item in wave.item_ids:
            if allowed is not None and item not in allowed:
                continue
            task = tasks[item]
            grid = grids["main_extension"] if "extension" in wave.panels else grids["main"]
            main.append(graph_for(grid, cfg=cfg, partition="main", task=task, lock=primary, base_budget=main_a,
                                  amendments=amendments))
            if "high" in wave.panels and "high" in grids:
                high.append(graph_for(grids["high"], cfg=cfg, partition="high", task=task, lock=primary,
                                      base_budget=high_a, amendments=amendments))
            if "replication" in wave.panels and "replication" in grids and replication is not None:
                rep.append(graph_for(grids["replication"], cfg=cfg, partition="replication", task=task,
                                     lock=replication, base_budget=main_a, amendments=amendments))
        if main:
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
    api_key = None if rehearsal() else load_api_key()
    locks = read_endpoint_locks(paths)
    drift: list[str] = []
    if not rehearsal():
        client = OpenRouterClient(api_key=api_key)
        try:
            drift = await verify_endpoint_drift(client, locks, cfg)
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
    # R02-8: the ledger gets the absolute provider-derived ceiling (retries/reconciliation may use the contingency);
    # the dispatcher holds NEW attempts to the configured soft admission ceiling itself.
    ceiling = active_ceiling(max_usd, ledger_settled(paths), snap)
    runtime = Runtime(cfg, paths, lock=primary, tasks=tasks, ceiling=ceiling, api_key=api_key)
    try:
        return await _run_main_phases(cfg, paths, max_usd, runtime=runtime, by_wave=by_wave, tasks=tasks,
                                      locks=locks, amendments=amendments, api_key=api_key, soft=soft,
                                      reserve=reserve, ceiling=ceiling)
    finally:
        await runtime.aclose()


async def _run_main_phases(cfg, paths: RunPaths, max_usd: Decimal, *, runtime: "Runtime", by_wave, tasks, locks,
                           amendments, api_key, soft: Decimal, reserve: Decimal, ceiling: Decimal) -> int:
    from agent_allocation.forecast import (
        fit_models, item_cost, observations_from_rows, per_item_costs, wave_admission, SAFETY_MULTIPLIER,
    )
    primary = locks[cfg["primary_model"]]
    if not by_wave["high"] and not by_wave["replication"]:
        # No high-A or replication partition (forced27b): nothing competes with main under the cut order, so every
        # main+extension wave is admitted at once (the dispatcher holds max_active_waves open and keeps each new
        # attempt under the soft admission ceiling itself); the forecast is recorded, not used to trim.
        summary = await runtime.run(by_wave["main"], allow_paid=not rehearsal(), max_usd=max_usd, ceiling=ceiling)
        record_decision(paths, phase="main_all_waves", waves=len(by_wave["main"]),
                        items=sum(len(w) for w in by_wave["main"]), summary=str(summary), ceiling=str(ceiling),
                        shard=(cfg.get("shard") or {}).get("id"))
        if stopped_early(summary):
            record_decision(paths, phase="main_all_waves", decision="stop_notify", reason=stopped_early(summary))
            return 3
        return 0
    summary = await runtime.run(by_wave["main"][:FORECAST_CHECK_WAVES], allow_paid=not rehearsal(), max_usd=max_usd,
                                ceiling=ceiling)
    record_decision(paths, phase="A", summary=str(summary), ceiling=str(ceiling))
    if stopped_early(summary):
        record_decision(paths, phase="A", decision="stop_notify", reason=stopped_early(summary))
        return 3

    # Phase B: re-forecast from main rows; admit remaining main waves and high-A waves under the cut order.
    rows = load_call_rows(paths, "main", {})
    models = fit_models(observations_from_rows([r for r in rows if r.spec.model == primary.model]))
    headroom = {b: headroom_for(cfg, b, amendments) for b in BENCHMARKS}
    costs = per_item_costs(models, grids_from_config(cfg), lock=primary, headroom_by_benchmark=headroom,
                           main_a=int(cfg["reasoning"]["main_a"]), high_a=cfg["reasoning"].get("high_a"))
    spent = ledger_settled(paths)
    snap = provider_snapshot(paths, "run_phase_b")
    ceiling = active_ceiling(max_usd, spent, snap)
    n_waves = len(by_wave["main"])
    ext_waves = cfg["panels"]["extension_items"] // 16
    admission = wave_admission(costs, n_waves=max(n_waves - FORECAST_CHECK_WAVES, 0),
                               extension_waves=max(ext_waves - FORECAST_CHECK_WAVES, 0),
                               high_waves=len(by_wave["high"]), available_usd=min(ceiling, soft) - spent)
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
    summary = await runtime.run(interleaved, allow_paid=not rehearsal(), max_usd=max_usd, ceiling=ceiling)
    record_decision(paths, phase="B", summary=str(summary))
    if stopped_early(summary):
        record_decision(paths, phase="B", decision="stop_notify", reason=stopped_early(summary))
        return 3

    # Phase C: optional Flash check only after core completion, if core + forecast + reserve fit.
    core_waves = by_wave["main"][:FORECAST_CHECK_WAVES] + main_rest + high
    if not by_wave["replication"]:
        record_decision(paths, phase="C", decision="skip_replication", reason="no replication partition")
        return 0
    if not all_terminal(paths, core_waves):
        record_decision(paths, phase="C", decision="skip_replication", reason="core incomplete")
        return 0
    replication_lock = locks[cfg["replication_model"]]
    flash_rows = ([r for r in load_call_rows(paths, "canary", {}) if r.spec.model == replication_lock.model]
                  if paths.jobs("canary").exists() else [])
    flash_models = fit_models(observations_from_rows(flash_rows))
    from agent_allocation.forecast import BenchmarkModel
    saturated = BenchmarkModel(root_prompt=2500.0, private_prompt=2500.0, packet_tokens=0.0, completion={})
    grids = grids_from_config(cfg)
    flash_items = sum(len(w) for w in by_wave["replication"])
    flash_forecast = Decimal("0")
    for wave in by_wave["replication"]:
        for graph in wave:
            model = flash_models.get(graph.nodes[0].benchmark) or next(iter(flash_models.values()), saturated)
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
    summary = await dispatch(cfg, paths, by_wave["replication"], lock=replication_lock, tasks=tasks,
                             allow_paid=not rehearsal(), max_usd=max_usd, ceiling=absolute - reserve, api_key=api_key)
    record_decision(paths, phase="C", summary=str(summary))
    return 3 if stopped_early(summary) else 0


def cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation run")
    common_args(parser)
    parser.add_argument("--resume", action="store_true", help="accepted for the spec's command contract; runs always resume")
    parser.add_argument("--rehearsal", action="store_true", help="fake transport end to end in a scratch run dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.rehearsal:
        if RunPaths.for_run(args.run_dir).run_dir.resolve() == RunPaths.for_run(None).run_dir.resolve():
            raise RunRefused("--rehearsal must use a scratch --run-dir, never the study run dir")
        os.environ[REHEARSAL_ENV] = "1"
    cfg, paths, max_usd = preflight(cfg, args)
    paths.ensure()
    return asyncio.run(run_main(cfg, paths, max_usd))


def cmd_seal(argv: list[str]) -> int:
    """Select and seal every checkpoint of the frozen partitions (after grading's public checks exist)."""
    parser = argparse.ArgumentParser(prog="allocation seal")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--partition", action="append", default=None)
    shard_arg(parser)
    args = parser.parse_args(argv)
    cfg, paths = resolve_run(load_config(args.config), args)
    tasks = load_tasks(paths)
    locks = read_endpoint_locks(paths)
    by_wave = graphs_by_wave(cfg, paths, tasks, locks, load_amendments(paths))
    out = {}
    grids = grids_from_config(cfg)
    enabled = ["main"] + [p for p in ("high", "replication") if p in grids]
    for partition in args.partition or enabled:
        graphs = [g for wave in by_wave[partition] for g in wave]
        out[partition] = select_and_seal(cfg, paths, partition, graphs, tasks)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


# ------------------------------------------------------------------------------------------------------------------
# Offline end-to-end (Q1 from agent 08; also the lead's seam test). Never paid: the transport must be a mock.
# ------------------------------------------------------------------------------------------------------------------

def _write_offline_manifests(paths: RunPaths, items: Sequence[Any], extension_ids: set[str]) -> None:
    from agent_allocation.contracts import PanelDraw, Wave
    from agent_allocation.manifest import write_atomic, write_jsonl
    paths.ensure()
    for benchmark in BENCHMARKS:
        chosen = [i for i in items if i.public.benchmark == benchmark]
        write_jsonl(paths.public_manifest(benchmark), [i.public for i in chosen])
        write_jsonl(paths.private_manifest(benchmark), [i.private for i in chosen], mode=0o600)
    draws = {b: PanelDraw(benchmark=b, dev_item_ids=(),
                          main_item_ids=tuple(i.public.item_id for i in items if i.public.benchmark == b),
                          reserve_item_ids=(), selection_seed=0, procedure="offline_given_items")
             for b in BENCHMARKS}
    write_atomic(paths.panels, json.dumps({b: dump_record(d) for b, d in draws.items()}, indent=1, sort_keys=True))
    ext = tuple(i.public.item_id for i in items if i.public.item_id in extension_ids)
    rest = tuple(i.public.item_id for i in items if i.public.item_id not in extension_ids)
    waves = [w for w in (Wave(0, ext, ("main", "extension")) if ext else None,
                         Wave(1 if ext else 0, rest, ("main",)) if rest else None) if w is not None]
    write_atomic(paths.waves, json.dumps([dump_record(w) for w in waves], indent=1, sort_keys=True))


def _grade_offline(paths: RunPaths, graphs: Sequence[ItemGraph], tasks, privates, code_grader, public_checker) -> dict:
    from agent_allocation.contracts import Candidate
    from agent_allocation.grading.text.grade import grade_text
    counts = {"public_checks": 0, "grades": 0}
    for graph in graphs:
        for node in graph.nodes:
            parsed = paths.parsed(node.node_id)
            if not parsed.exists():
                continue
            payload = json.loads(parsed.read_text())
            if not payload.get("candidate"):
                continue
            candidate = load_record(Candidate, payload["candidate"])
            private = privates[node.item_id]
            if node.benchmark == "mbpp_plus":
                if public_checker is None:
                    from agent_allocation.grading.code.public import run_public_checks as public_checker
                check = public_checker(candidate, tasks[node.item_id])
                target = paths.public_check(node.node_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(dump_record(check), sort_keys=True))
                counts["public_checks"] += 1
                if code_grader is None:
                    from agent_allocation.grading.code.private import grade_code as code_grader
                grade = code_grader(candidate, private, paths)
            else:
                grade = grade_text(candidate, private)
            target = paths.grade(node.node_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(dump_record(grade), sort_keys=True))
            counts["grades"] += 1
    return counts


async def run_offline_async(paths: RunPaths, config: Mapping[str, Any], items: Sequence[Any], transport, *,
                            extension_item_ids: Sequence[str] = (), code_grader=None, public_checker=None) -> dict:
    import httpx
    from agent_allocation.api.fake import fake_endpoint_lock
    if not isinstance(transport, httpx.MockTransport):
        raise RunRefused("run_offline only accepts a mock transport")
    extension_ids = set(extension_item_ids)
    _write_offline_manifests(paths, items, extension_ids)
    tasks = {i.public.item_id: i.public for i in items}
    privates = {i.public.item_id: i.private for i in items}
    lock = fake_endpoint_lock(config["primary_model"])
    grids = grids_from_config(config)
    graphs = [graph_for(grids["main_extension"] if i.public.item_id in extension_ids else grids["main"], cfg=config,
                        partition="main", task=i.public, lock=lock, base_budget=int(config["reasoning"]["main_a"]),
                        amendments={}) for i in items]
    write_jobs(paths, "main", graphs)
    ceiling = credit_ceiling(config)
    waves = [graphs[k:k + 16] for k in range(0, len(graphs), 16)]
    summary = await dispatch(config, paths, waves, lock=lock, tasks=tasks, allow_paid=False, max_usd=ceiling,
                             ceiling=ceiling, transport=transport, api_key=None)
    grading = _grade_offline(paths, graphs, tasks, privates, code_grader, public_checker)
    selection = select_and_seal(config, paths, "main", graphs, tasks)
    from agent_allocation.reporting.audit import cmd_audit
    from agent_allocation.reporting.cli import cmd_analyze
    analyze_rc = cmd_analyze(["--run-dir", str(paths.run_dir)])
    audit_rc = cmd_audit(["--run-dir", str(paths.run_dir)])
    return {"summary": summary, "grading": grading, "selection": selection, "analyze_rc": analyze_rc,
            "audit_rc": audit_rc, "stubbed_code_grader": code_grader is not None,
            "stubbed_public_checker": public_checker is not None,
            "nodes": sum(len(g.nodes) for g in graphs)}


def run_offline(paths: RunPaths, config: Mapping[str, Any], items: Sequence[Any], transport, *,
                extension_item_ids: Sequence[str] = (), code_grader=None, public_checker=None) -> dict:
    """Mocked four-benchmark pipeline for tests. Calling it again on the same ``paths`` is the resume path."""
    return asyncio.run(run_offline_async(paths, config, items, transport, extension_item_ids=extension_item_ids,
                                         code_grader=code_grader, public_checker=public_checker))
