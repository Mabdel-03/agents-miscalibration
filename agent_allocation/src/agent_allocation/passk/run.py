"""``allocation passk-run``: dispatch the passk_cont_v1 partitions of one provider shard (WP-G).

Modelled on :func:`agent_allocation.calibrate.collect`:

* loads the shard's public tasks, endpoint lock and amendments, and ``<artifacts>/manifests/splits.json``;
* builds the FULL frozen node list of every requested partition for this shard (:mod:`passk.banks`), writes
  ``manifests/jobs/<partition>.jsonl`` (always the whole list, so a resume never shrinks it; an identical file is
  left untouched) and asserts that its sha256 equals the one recorded at Gate 0 (``<study>/gates/gate0.json``); a
  PAID dispatch refuses unless that record exists, passed, and froze every requested partition of this shard;
* builds one :class:`live.Runtime` whose ledger ceiling is explicit per shard:
  ``min(live.active_ceiling(max_usd, settled, /key snapshot, account_auto_reload=...), shard credit slice)``;
* runs the waves (stage 1 then seeded stage-2 item-waves; ``passk.banks.plan_waves``). It NEVER grades inline:
  grading is the per-shard ``allocation grade --follow`` with the held-out fence (``passk.grade_scope``).

Resume: the state store and ledger journal of the shard dir carry every terminal node; rerunning the same command
re-plans the same frozen lists and dispatches only what is not terminal (``--resume`` is accepted for the command
contract; every run resumes).

``--plan-only`` prints per-partition counts, jobs shas and the unit-cost forecast without writing or dispatching.
``--rehearsal`` runs against the fake transport (scratch run dirs only). Exit 0 completed, 3 stopped early.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_allocation.passk import banks

GATE0_NAME = "gate0.json"
LEGACY_LOCK_REL = Path("locks") / "openrouter_endpoints.json"


class PasskRunRefused(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------
# Resolution
# ------------------------------------------------------------------------------------------------------------------

def resolve(raw_cfg: Mapping[str, Any], run_dir: str | None, shard_id: str | None):
    """``(cfg, paths)``: with ``--shard`` the run dir is the run root; a base config with a shard dir as ``--run-dir``
    (``.../shards/<id>``) infers the shard; a derived shard config runs in the given dir."""
    from agent_allocation.live import resolve_run
    from agent_allocation.shards import SHARDS_DIRNAME
    if shard_id is None and raw_cfg.get("shards") and raw_cfg.get("shard") is None and run_dir:
        candidate = Path(run_dir)
        if candidate.parent.name == SHARDS_DIRNAME:
            shard_id, run_dir = candidate.name, str(candidate.parent.parent)
    return resolve_run(raw_cfg, argparse.Namespace(run_dir=run_dir, shard=shard_id))


def shard_id_of(cfg: Mapping[str, Any]) -> str:
    return str((cfg.get("shard") or {}).get("id") or "unsharded")


def gate0_record(paths) -> dict[str, Any] | None:
    target = banks.study_root(paths) / "gates" / GATE0_NAME
    if not target.is_file():
        return None
    return json.loads(target.read_text())


def recorded_jobs_sha(paths, shard: str, partition: str) -> str | None:
    record = gate0_record(paths)
    if not record:
        return None
    return ((record.get("jobs_sha256") or {}).get(shard) or {}).get(partition)


def adopt_legacy_lock(cfg: Mapping[str, Any], paths) -> dict[str, Any]:
    """Copy the legacy forced_v2 shard's endpoint lock (same provider tag) into this shard dir when it has none, so
    the frozen node ids (which carry the lock hash) are known before any paid call. Write-once; refuses a tag
    mismatch. The canary then uses this lock instead of fetching a new one."""
    legacy_root = Path(cfg["passk"]["legacy_run_root"])
    shard = shard_id_of(cfg)
    source = legacy_root / "shards" / shard / LEGACY_LOCK_REL
    if paths.endpoint_lock.exists():
        return {"adopted": False, "reason": "lock exists", "path": str(paths.endpoint_lock)}
    if not source.is_file():
        raise PasskRunRefused(f"no legacy lock at {source}")
    data = json.loads(source.read_text())
    lock = data["locks"][cfg["primary_model"]]
    tag = cfg["provider"].get("endpoint_tag")
    if lock.get("provider_tag") != tag:
        raise PasskRunRefused(f"legacy lock tag {lock.get('provider_tag')!r} != shard tag {tag!r}")
    paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths.endpoint_lock)
    return {"adopted": True, "source": str(source), "path": str(paths.endpoint_lock),
            "lock_hash": data.get("lock_hashes", {}).get(cfg["primary_model"])}


# ------------------------------------------------------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------------------------------------------------------

def shard_plan(cfg: Mapping[str, Any], paths, partitions: Sequence[str], *, splits: Mapping[str, Any] | None = None,
               tasks=None, lock=None, amendments=None) -> dict[str, list]:
    """This shard's full frozen node list per partition."""
    from agent_allocation.live import load_amendments, load_tasks, read_endpoint_locks, shard_items
    tasks = load_tasks(paths) if tasks is None else tasks
    amendments = load_amendments(paths) if amendments is None else amendments
    if lock is None:
        locks = read_endpoint_locks(paths)
        if cfg["primary_model"] not in locks:
            raise PasskRunRefused(f"{paths.run_dir}: no endpoint lock (canary, or passk-run --adopt-legacy-lock)")
        lock = locks[cfg["primary_model"]]
    splits = banks.load_splits(paths) if splits is None else splits
    return banks.build_plan(cfg, splits, tasks, lock, amendments, partitions, allowed=shard_items(cfg, paths))


def plan_summary(cfg: Mapping[str, Any], paths, graphs: Mapping[str, Sequence]) -> dict[str, Any]:
    from agent_allocation.passk.forecast import forecast_graphs
    shard = shard_id_of(cfg)
    shas = {p: banks.jobs_text_sha(gs) for p, gs in graphs.items()}
    recorded = {p: recorded_jobs_sha(paths, shard, p) for p in graphs}
    waves = banks.plan_waves(graphs, int(cfg["scheduler"]["wave_items"]))
    out = {"shard": shard, "run_dir": str(paths.run_dir),
           "items": {p: len(gs) for p, gs in graphs.items()}, "nodes": banks.node_counts(graphs),
           "waves": len(waves), "jobs_sha256": shas,
           "gate0_jobs_sha256": recorded, "gate0_match": {p: (recorded[p] is None or recorded[p] == shas[p])
                                                          for p in graphs},
           "splits_sha256": banks.splits_sha256(paths) if banks.splits_path(paths).is_file() else None}
    if shard in ("mancer", "parasail"):
        out["forecast"] = forecast_graphs(shard, graphs, runs=1)
    return out


def require_gate0(cfg: Mapping[str, Any], paths, partitions: Sequence[str]) -> dict[str, Any]:
    """Paid dispatch needs a passed Gate 0 record that froze every requested partition of this shard."""
    record = gate0_record(paths)
    if not record or not record.get("passed") or not record.get("complete"):
        raise PasskRunRefused(f"no passed and complete (key + tests) Gate 0 record at "
                              f"{banks.study_root(paths) / 'gates' / GATE0_NAME}; nothing is sent")
    shard = shard_id_of(cfg)
    missing = [p for p in partitions if recorded_jobs_sha(paths, shard, p) is None]
    if missing:
        raise PasskRunRefused(f"Gate 0 froze no node list for {shard}/{missing}; nothing is sent")
    return record


def assert_frozen(cfg: Mapping[str, Any], paths, graphs: Mapping[str, Sequence]) -> dict[str, str]:
    shard = shard_id_of(cfg)
    shas = {}
    for partition, gs in graphs.items():
        sha = banks.jobs_text_sha(gs)
        recorded = recorded_jobs_sha(paths, shard, partition)
        if recorded is not None and recorded != sha:
            raise PasskRunRefused(f"{shard}/{partition}: node list sha {sha[:12]} != Gate 0 record {recorded[:12]}")
        shas[partition] = sha
    return shas


# ------------------------------------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------------------------------------

PROTOCOL_PARAMETERS = ("logit_bias", "top_k", "temperature", "top_p", "max_tokens")   # the forced protocol's needs
PRICE_TOLERANCE = Decimal("0.10")        # a price rise beyond +10% refuses; a smaller rise or any decrease is noted


def classify_endpoint_drift(lock, live) -> tuple[list[str], list[str]]:
    """(hard, soft) differences between the adopted lock and the live endpoint. Hard = protocol-relevant: the model,
    the provider pin (provider_tag carries the quantization, e.g. ``parasail/fp8``), a forced-protocol parameter
    dropped from ``supported_parameters``, a price rise beyond +10%, or a smaller context / completion cap. Everything
    else (other parameters, price decreases or small rises, a larger cap) is soft: logged, dispatch continues. The
    adopted lock stays the identity of the frozen node ids either way."""
    hard: list[str] = []
    soft: list[str] = []
    for field in ("model", "provider_tag", "provider_name"):
        if getattr(live, field) != getattr(lock, field):
            hard.append(f"{field}: {getattr(lock, field)!r} -> {getattr(live, field)!r}")
    live_params, lock_params = set(live.supported_parameters), set(lock.supported_parameters)
    dropped = sorted(p for p in PROTOCOL_PARAMETERS if p in lock_params and p not in live_params)
    if dropped:
        hard.append(f"supported_parameters: forced-protocol parameters dropped {dropped}")
    if live_params != lock_params and not dropped:
        soft.append(f"supported_parameters: +{sorted(live_params - lock_params)} -{sorted(lock_params - live_params)}")
    for field in ("input_price_per_token", "output_price_per_token", "internal_reasoning_price_per_token"):
        old, new = Decimal(str(getattr(lock, field) or "0")), Decimal(str(getattr(live, field) or "0"))
        if new == old:
            continue
        rise = new > old * (1 + PRICE_TOLERANCE) if old > 0 else new > 0
        (hard if rise else soft).append(f"{field}: {old} -> {new}")
    for field in ("context_length", "max_completion_tokens"):
        old, new = int(getattr(lock, field) or 0), int(getattr(live, field) or 0)
        if new != old:
            (hard if new < old else soft).append(f"{field}: {old} -> {new}")
    if live.service_tier != lock.service_tier:
        soft.append(f"service_tier: {lock.service_tier!r} -> {live.service_tier!r}")
    return hard, soft


async def passk_endpoint_drift(client, lock, cfg: Mapping[str, Any], *, attempts: int = 5,
                               delay_s: float = 60.0) -> tuple[list[str], list[str], Any]:
    """Probe the live listing (retried: a transient GET failure is not drift) and classify it against the lock.
    Returns (hard, soft, raw listing or None)."""
    from agent_allocation.api.endpoints import EndpointBlocked, select_standard_endpoint
    from agent_allocation.api.forced import forced_enabled
    model = cfg["primary_model"]
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            raw, parsed = await client.list_endpoints(model)
            break
        except Exception as exc:                   # noqa: BLE001 - transport/5xx: retried, then a distinct refusal
            last = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(delay_s)
    else:
        raise PasskRunRefused(f"endpoint drift probe unavailable after {attempts} attempts: {last!r}")
    try:
        live = select_standard_endpoint(parsed, model, preferred_tag=lock.provider_tag, forced=forced_enabled(cfg))
    except EndpointBlocked as exc:
        return [f"{model}: {exc}"], [], raw
    hard, soft = classify_endpoint_drift(lock, live)
    return [f"{model}: {d}" for d in hard], [f"{model}: {d}" for d in soft], raw


async def run_partitions(cfg: Mapping[str, Any], paths, partitions: Sequence[str], *, max_usd: Decimal,
                         transport=None, api_key: str | None = None, allow_paid: bool = True,
                         account_auto_reload: bool = True, wrap_client: Callable[[Any], Any] | None = None,
                         check_drift: bool | None = None) -> dict[str, Any]:
    """Write the frozen jobs files and dispatch the partitions' waves on this shard (resumable)."""
    from agent_allocation.live import (
        Runtime, active_ceiling, ledger_settled, load_amendments, load_tasks, provider_snapshot,
        read_endpoint_locks, record_decision, rehearsal, stopped_early, write_jobs,
    )
    tasks = load_tasks(paths)
    amendments = load_amendments(paths)
    locks = read_endpoint_locks(paths)
    if cfg["primary_model"] not in locks:
        raise PasskRunRefused(f"{paths.run_dir}: no endpoint lock (run the canary for this shard first)")
    lock = locks[cfg["primary_model"]]
    offline = transport is not None or rehearsal()
    if allow_paid and not offline:
        require_gate0(cfg, paths, partitions)
    if check_drift is None:
        check_drift = allow_paid and not offline
    if check_drift:
        from agent_allocation.api.client import OpenRouterClient
        client = OpenRouterClient(api_key=api_key)
        try:
            hard, soft, raw = await passk_endpoint_drift(client, lock, cfg)
        finally:
            await client.aclose()
        if hard or soft:
            snap_dir = paths.lock_snapshots_dir
            snap_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            snap = snap_dir / f"drift_probe_{shard_id_of(cfg) or 'shard'}_{stamp}.json"
            snap.write_bytes(raw if isinstance(raw, (bytes, bytearray)) else json.dumps(raw, default=str).encode())
            if soft:
                record_decision(paths, phase="passk_run", decision="drift_noted", detail=soft, snapshot=str(snap))
            if hard:
                record_decision(paths, phase="passk_run", decision="stop_notify", reason="endpoint drift",
                                detail=hard, snapshot=str(snap))
                raise PasskRunRefused(f"endpoint drift (protocol-relevant): {hard}; live listing saved to {snap}")
    graphs = shard_plan(cfg, paths, partitions, tasks=tasks, lock=lock, amendments=amendments)
    shas = assert_frozen(cfg, paths, graphs)
    for partition, gs in graphs.items():
        target = paths.jobs(partition)
        if target.is_file() and sha256(target.read_bytes()).hexdigest() == shas[partition]:
            continue                        # identical frozen list already on disk: never rewrite under the grader
        written = write_jobs(paths, partition, gs)
        if written != shas[partition]:
            raise PasskRunRefused(f"{partition}: written jobs sha differs from the planned list")
    settled = ledger_settled(paths)
    snap = None if offline else provider_snapshot(paths, "passk_run")
    slice_usd = Decimal(str(cfg["budget"]["credit_ceiling_usd"]))
    ceiling = min(active_ceiling(max_usd, settled, snap, account_auto_reload=account_auto_reload), slice_usd)
    waves = banks.plan_waves(graphs, int(cfg["scheduler"]["wave_items"]))
    record_decision(paths, phase="passk_run_start", partitions=list(partitions), nodes=banks.node_counts(graphs),
                    waves=len(waves), jobs_sha256=shas, ceiling=str(ceiling), settled=str(settled),
                    max_usd=str(max_usd), shard_slice=str(slice_usd), account_auto_reload=account_auto_reload,
                    limit_remaining=(snap or {}).get("key", {}).get("limit_remaining") if snap else None,
                    shard=shard_id_of(cfg))
    runtime = Runtime(cfg, paths, lock=lock, tasks=tasks, ceiling=ceiling, transport=transport, api_key=api_key)
    if wrap_client is not None:
        runtime.dispatcher.client = wrap_client(runtime.dispatcher.client)
    try:
        summary = await runtime.run(waves, allow_paid=allow_paid, max_usd=max_usd, ceiling=ceiling)
    finally:
        await runtime.aclose()
    reason = stopped_early(summary)
    record_decision(paths, phase="passk_run_end", partitions=list(partitions), summary=str(summary),
                    stop_reason=getattr(summary, "stop_reason", None), decision="stop_notify" if reason else "done",
                    shard=shard_id_of(cfg))
    return {"shard": shard_id_of(cfg), "partitions": list(partitions), "nodes": banks.node_counts(graphs),
            "waves": len(waves), "ceiling": str(ceiling), "summary": str(summary),
            "stop_reason": getattr(summary, "stop_reason", None), "stopped_early": reason}


# ------------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------------

def cmd_passk_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation passk-run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="shard dir (…/artifacts/shards/<id>), or the run root with --shard")
    parser.add_argument("--shard", default=None)
    parser.add_argument("--partitions", action="append", required=True,
                        help="partition names joined by '+' (or ','), repeatable")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-usd", default=None, help="cumulative spend cap of this shard's ledger")
    parser.add_argument("--account-auto-reload", action="store_true")
    parser.add_argument("--resume", action="store_true", help="accepted; every run resumes from the shard state")
    parser.add_argument("--plan-only", action="store_true", help="print counts, shas and forecast; no dispatch")
    parser.add_argument("--adopt-legacy-lock", action="store_true",
                        help="copy the legacy forced_v2 shard endpoint lock when this shard has none, then exit")
    parser.add_argument("--rehearsal", action="store_true", help="fake transport end to end (scratch run dirs only)")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config, validate_config
    from agent_allocation.live import REHEARSAL_ENV, RunRefused, checked_max_usd, rehearsal
    try:
        partitions = banks.parse_partitions(args.partitions)
    except banks.BankError as exc:
        print(f"refused: {exc}")
        return 2
    raw = load_config(args.config)
    if args.rehearsal:
        target = Path(args.run_dir or "").resolve()
        if "agents_scaling_results" in str(target):
            raise RunRefused("--rehearsal must use a scratch --run-dir, never a study results dir")
        os.environ[REHEARSAL_ENV] = "1"
    cfg, paths = resolve(raw, args.run_dir, args.shard)
    problems = validate_config(cfg)
    if problems:
        raise RunRefused(f"config invalid: {problems}")
    if args.adopt_legacy_lock:
        print(json.dumps(adopt_legacy_lock(cfg, paths), indent=1, sort_keys=True))
        return 0
    if args.plan_only:
        graphs = shard_plan(cfg, paths, partitions)
        summary = plan_summary(cfg, paths, graphs)
        print(json.dumps(summary, indent=1, sort_keys=True, default=str))
        return 0 if all(summary["gate0_match"].values()) else 3
    if args.max_usd is None:
        raise RunRefused("--max-usd is required to dispatch")
    if not args.allow_paid and not rehearsal():
        raise RunRefused("passk-run without --allow-paid: nothing is sent (use --plan-only)")
    max_usd = checked_max_usd(cfg, args.max_usd)
    paths.ensure()
    api_key = None
    if not rehearsal():
        from agent_allocation.apikey import load_api_key
        api_key = load_api_key()
    try:
        result = asyncio.run(run_partitions(cfg, paths, partitions, max_usd=max_usd, api_key=api_key,
                                            allow_paid=not rehearsal(),
                                            account_auto_reload=args.account_auto_reload))
    except PasskRunRefused as exc:
        print(f"refused: {exc}")
        return 2
    print(json.dumps(result, indent=1, sort_keys=True, default=str))
    return 3 if result["stopped_early"] else 0
