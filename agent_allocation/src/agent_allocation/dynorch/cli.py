"""WP-D: the ``dynorch-*`` verbs (contract §8), registered in ``agent_allocation/cli.py::VERBS``.

``dynorch-plan`` enumerates the planned cells of a stage (a dry run needs nothing but the config and the sealed base
run root; ``--write`` hands the cells to WP-B's ``plan.plan_runs`` and writes the manifest). ``dynorch-run`` is the
only paid verb: it refuses without ``--allow-paid``, refuses a ``--max-usd`` above the preset ceiling
(``config.PRESET_CEILINGS_USD``) or above the stage allowance unless ``--approval`` text is given and recorded through
``live.record_decision``, resolves the plan §4 endpoint rule once (canary), snapshots the account before and after
(``reconcile``), binds the metering proxy on ``127.0.0.1:0`` and writes ``<run_root>/state/proxy.json``, runs the
stage through WP-B's runner, and writes ``state/status.json``. ``dynorch-tables`` calls WP-C, ``dynorch-gates``
evaluates one gate, ``dynorch-reconcile`` and ``dynorch-doctor`` delegate to their modules. WP-A/B/C modules are
imported lazily inside the verbs; a missing entry point reports ``blocked`` (exit 2) with the requested name
(``docs/dynorch/requests/wpD.md``) instead of a traceback.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent_allocation.apikey import redact
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic

EXIT_OK, EXIT_STOP, EXIT_BLOCKED, EXIT_STOP_NOTIFY = 0, 1, 2, 3
STAGES: tuple[str, ...] = ("canary", "pilot", "main", "ext_e0", "ext_e1", "ext_e2", "ext_e3")
PAID_STAGES: frozenset[str] = frozenset(STAGES)          # every stage sends model calls
CORE_ARMS: tuple[str, ...] = ("H1", "H3", "H0")
REHEARSAL_ENV = "ALLOCATION_REHEARSAL"


class RunRefused(RuntimeError):
    """A verb refused before anything was sent or written."""


class Blocked(RuntimeError):
    """A WP-A/B/C entry point this verb codes against has not landed (see docs/dynorch/requests/wpD.md)."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print(obj: Any) -> None:
    print(redact(json.dumps(obj, indent=1, sort_keys=True, default=str)))


# ------------------------------------------------------------------------------------------------------------------
# Stage specification and cell enumeration (plan §1; the runner's plan_runs enumerates the same cells)
# ------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    arms: tuple[str, ...]
    rungs: tuple[int, ...]
    items: str                 # "main" | "dev" | "canary"
    per_benchmark: int | None  # dev items per benchmark (None = all of the panel)
    repeat: int = 1


@dataclass(frozen=True)
class StageSpec:
    stage: str
    partition: str
    blocks: tuple[Block, ...]
    allowance_usd: Decimal | None      # canary/pilot allowances from config.budget
    requires: tuple[str, ...]           # gate files that must record advance before this stage


def stage_spec(cfg: Mapping[str, Any], stage: str) -> StageSpec:
    """The frozen cell blocks of every stage (plan §1 / §8 / §9)."""
    dyn = cfg["dynorch"]
    rungs = tuple(int(r) for r in dyn["rungs"])
    budget = cfg["budget"]
    if stage == "canary":
        c = dyn["canary"]
        return StageSpec(stage, "canary", (
            Block(CORE_ARMS, (int(c["rung"]),), "canary", 1),
            Block(("H1", "H3"), (int(c["ceiling_probe_rung"]),), "canary", 1),
        ), Decimal(str(budget["canary_allowance_usd"])), ())
    if stage == "pilot":
        p = dyn["pilot"]
        return StageSpec(stage, "pilot", (
            Block(("H1", "H3"), tuple(int(r) for r in p["rungs"]), "dev", int(p["dev_items_per_benchmark"])),
            Block(("H0",), (int(p["h0_rung"]),), "dev", int(p["h0_items_per_benchmark"])),
        ), Decimal(str(budget["pilot_allowance_usd"])), ("gate1_canary",))
    if stage == "main":
        return StageSpec(stage, "main", (Block(CORE_ARMS, rungs, "main", None),), None, ("gate2_pilot",))
    if stage == "ext_e0":
        return StageSpec(stage, "ext_e0", (Block(("H1c", "H3c"), (4, 16), "main", None),), None, ("gate4_main",))
    if stage == "ext_e1":
        return StageSpec(stage, "ext_e1", (Block(("H1", "H3"), (16, 4, 25, 8, 2, 1), "main", None, repeat=2),), None, ("gate4_main",))
    if stage == "ext_e2":
        return StageSpec(stage, "ext_e2", (Block(("H1n", "H3n"), (4, 16), "main", None),), None, ("gate4_main",))
    if stage == "ext_e3":
        return StageSpec(stage, "ext_e3", (Block(("H2",), (4, 16), "main", None),), None, ("gate4_main",))
    raise RunRefused(f"unknown stage {stage!r}; expected one of {STAGES}")


def load_panels(base_run_root: Path) -> dict[str, Mapping[str, Any]]:
    """``manifests/panels.json`` of the sealed base run: ``{benchmark: {dev_item_ids, main_item_ids, ...}}``."""
    doc = json.loads((Path(base_run_root) / "manifests" / "panels.json").read_text())
    panels = {b: p for b, p in doc.items() if isinstance(p, Mapping) and "main_item_ids" in p}
    if not panels:
        raise RunRefused(f"{base_run_root}/manifests/panels.json has no benchmark panels")
    return panels


def block_items(cfg: Mapping[str, Any], panels: Mapping[str, Mapping[str, Any]], block: Block) -> list[tuple[str, str]]:
    """``[(benchmark, item_id)]`` for a block: the 96 main items, the first N dev items per benchmark, or the
    canary's one dev item per listed benchmark, always in panel order (never by outcome)."""
    benches = [b["id"] for b in cfg["benchmarks"]]
    out: list[tuple[str, str]] = []
    if block.items == "canary":
        wanted = [str(b) for b in cfg["dynorch"]["canary"]["items"]]
        for bench in wanted:
            if bench not in panels:
                raise RunRefused(f"canary benchmark {bench!r} has no panel")
            out.append((bench, str(panels[bench]["dev_item_ids"][0])))
        return out
    for bench in benches:
        if bench not in panels:
            raise RunRefused(f"benchmark {bench!r} has no panel")
        ids = panels[bench]["main_item_ids" if block.items == "main" else "dev_item_ids"]
        if block.per_benchmark is not None:
            ids = ids[: block.per_benchmark]
        out.extend((bench, str(i)) for i in ids)
    return out


def enumerate_cells(cfg: Mapping[str, Any], stage: str, *, arms: Sequence[str] | None = None,
                    rungs: Sequence[int] | None = None, base_run_root: Path | None = None) -> list[dict[str, Any]]:
    """Every planned cell ``{arm, rung, item_id, benchmark, repeat, partition}`` of a stage, optionally restricted
    to ``arms`` / ``rungs`` (a restriction never adds cells)."""
    spec = stage_spec(cfg, stage)
    panels = load_panels(Path(base_run_root or cfg["base_run_root"]))
    cells: list[dict[str, Any]] = []
    for block in spec.blocks:
        for arm in block.arms:
            if arms is not None and arm not in arms:
                continue
            for rung in block.rungs:
                if rungs is not None and rung not in rungs:
                    continue
                for bench, item in block_items(cfg, panels, block):
                    cells.append({"arm": arm, "rung": int(rung), "item_id": item, "benchmark": bench,
                                  "repeat": block.repeat, "partition": spec.partition})
    return cells


def cell_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cells:
        key = f"{c['arm']}/{c['rung']}" + (f"/r{c['repeat']}" if c["repeat"] > 1 else "")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------------------------------
# Shared argument handling
# ------------------------------------------------------------------------------------------------------------------

def _parse_list(text: str | None, cast: Callable[[str], Any]) -> tuple[Any, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(cast(part.strip()) for part in text.split(",") if part.strip())


def preset_ceiling(cfg: Mapping[str, Any]) -> Decimal:
    from agent_allocation.config import PRESET_CEILINGS_USD
    preset = str(cfg.get("preset"))
    if preset not in PRESET_CEILINGS_USD:
        raise RunRefused(f"config preset {preset!r} has no registered ceiling")
    return PRESET_CEILINGS_USD[preset]


def checked_max_usd(cfg: Mapping[str, Any], requested: str, *, stage: str, approval: str | None) -> tuple[Decimal, dict[str, Any] | None]:
    """The dollar ceiling for this invocation. Above the preset ceiling or the stage allowance the verb refuses
    unless ``approval`` text is given; the override is returned to be recorded via ``live.record_decision``."""
    value = Decimal(requested)
    if value <= 0:
        raise RunRefused(f"--max-usd {value} must be positive")
    ceiling = preset_ceiling(cfg)
    spec = stage_spec(cfg, stage)
    override: dict[str, Any] | None = None
    if value > ceiling:
        if not approval:
            raise RunRefused(f"--max-usd {value} exceeds the preset ceiling {ceiling}; pass --approval \"<owner text>\" to record an override")
        override = {"kind": "max_usd_override", "stage": stage, "max_usd": str(value), "preset_ceiling": str(ceiling), "approval": approval}
    elif spec.allowance_usd is not None and value > spec.allowance_usd:
        if not approval:
            raise RunRefused(f"--max-usd {value} exceeds the {stage} allowance {spec.allowance_usd}; pass --approval to record an override")
        override = {"kind": "stage_allowance_override", "stage": stage, "max_usd": str(value),
                    "allowance": str(spec.allowance_usd), "approval": approval}
    return value, override


def gate_advanced(paths: RunPaths, name: str) -> bool:
    """True when ``gates/<name>.json`` exists with ``decision == advance``."""
    target = paths.run_dir / "gates" / f"{name}.json"
    if not target.exists():
        return False
    doc = json.loads(target.read_text())
    return doc.get("decision") == "advance"


def owner_go_recorded(paths: RunPaths) -> bool:
    if not paths.run_decisions.exists():
        return False
    for line in paths.run_decisions.read_text().splitlines():
        if line.strip() and json.loads(line).get("kind") == "owner_go":
            return True
    return False


def stage_prerequisites(cfg: Mapping[str, Any], paths: RunPaths, stage: str) -> list[str]:
    """Why a stage may not start yet (empty = go): required gates advanced, the owner go recorded before main,
    the endpoint lock frozen after the canary."""
    spec = stage_spec(cfg, stage)
    problems = [f"gate {g} has not recorded advance" for g in spec.requires if not gate_advanced(paths, g)]
    if stage != "canary" and not paths.endpoint_lock.exists():
        problems.append("locks/openrouter_endpoints.json missing: the canary resolves the endpoint rule first")
    if stage == "main" and cfg.get("execution", {}).get("main_requires_passed_pilot_and_frozen_locks", True):
        if not owner_go_recorded(paths):
            problems.append("owner go/no-go not recorded (dynorch-gates --gate 2 --owner-go \"<text>\")")
    return problems


def rehearsal() -> bool:
    return os.environ.get(REHEARSAL_ENV) == "1"


def make_client(cfg: Mapping[str, Any]) -> Any:
    """The one OpenRouter client of a stage: a ``ForcedClient`` (raw ``/completions`` for the proxy's ``Upstream``
    plus the inherited ``/key`` / ``/credits`` / ``/generation`` / endpoint listing for the endpoint rule and the
    reconciliation). The real key outside rehearsal; the fake transport (no network, no key) inside it. Timeouts
    from the config's ``scheduler`` block (the 900 s read timeout covers a 660 s stalled reply)."""
    from agent_allocation.api.forced import make_generation_client
    scheduler = cfg["scheduler"]
    timeouts = {"connect_timeout": float(scheduler["connect_timeout_seconds"]),
                "read_timeout": float(scheduler["read_timeout_seconds"])}
    if rehearsal():
        from agent_allocation.api.fake import fake_openrouter_transport
        return make_generation_client(cfg, api_key="rehearsal-no-key", transport=fake_openrouter_transport(),
                                      **timeouts)
    from agent_allocation.apikey import load_api_key
    return make_generation_client(cfg, api_key=load_api_key(), **timeouts)


def write_proxy_state(paths: RunPaths, *, host: str, port: int, stage: str, pid: int | None = None) -> Path:
    """``<run_root>/state/proxy.json``: where the harnesses' base URL points for this job."""
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    doc = {"host": host, "port": int(port), "pid": os.getpid() if pid is None else pid, "stage": stage, "bound_at": utcnow(),
           "base_url_template": f"http://{host}:{int(port)}/r/<run_key>/v1"}
    target = paths.state_dir / "proxy.json"
    replace_atomic(target, json.dumps(doc, indent=1, sort_keys=True).encode())
    return target


def write_status(paths: RunPaths, doc: Mapping[str, Any]) -> Path:
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    replace_atomic(paths.status_json, (redact(json.dumps(doc, indent=1, sort_keys=True, default=str)) or "").encode())
    return paths.status_json


def stage_marker(paths: RunPaths, stage: str) -> Path:
    """``state/stage_<stage>_started``: its mtime anchors the Gate 1 "nothing written under $HOME" scan."""
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    target = paths.state_dir / f"stage_{stage}_started"
    target.write_text(utcnow())
    return target


# ------------------------------------------------------------------------------------------------------------------
# Endpoint rule at the canary
# ------------------------------------------------------------------------------------------------------------------

async def resolve_endpoint_lock(cfg: Mapping[str, Any], paths: RunPaths, *, client: Any, stage: str) -> str:
    """Freeze the plan §4 decision once (canary); later stages only read it. Returns the decision action."""
    from agent_allocation.dynorch import endpoint_rule
    if paths.endpoint_lock.exists():
        return endpoint_rule.read_decision(paths.run_dir)["action"]
    if stage != "canary":
        raise RunRefused("endpoint lock missing outside the canary")
    raw, parsed = await client.list_endpoints(str(cfg["primary_model"]))
    assignment = endpoint_rule.load_assignment(Path(cfg["base_run_root"]))
    fetched_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decision = endpoint_rule.freeze(paths.run_dir, parsed, raw, cfg, assignment=assignment, fetched_at=fetched_at)
    return decision.action


# ------------------------------------------------------------------------------------------------------------------
# Verbs
# ------------------------------------------------------------------------------------------------------------------

def cmd_dynorch_plan(argv: list[str]) -> int:
    """``allocation dynorch-plan --config CFG --stage S [--run-dir RUN] [--arms A,B] [--rungs 4,16] [--write]``.
    Dry run by default: prints the cell counts per (arm, rung) and the plan §8 ceiling; ``--write`` needs WP-B's
    ``plan.plan_runs`` and writes ``manifests/dynorch_plan_<stage>.jsonl``."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-plan")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--arms", default=None)
    parser.add_argument("--rungs", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    cfg = load_config(args.config)
    arms = _parse_list(args.arms, str)
    rungs = _parse_list(args.rungs, int)
    try:
        cells = enumerate_cells(cfg, args.stage, arms=arms, rungs=rungs)
    except RunRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_STOP
    spec = stage_spec(cfg, args.stage)
    out: dict[str, Any] = {"stage": args.stage, "partition": spec.partition, "cells": len(cells), "by_cell": cell_summary(cells),
                           "allowance_usd": None if spec.allowance_usd is None else str(spec.allowance_usd),
                           "requires": list(spec.requires), "dry_run": not args.write}
    try:
        from agent_allocation.dynorch import forecast as _forecast
        out["forecast"] = _forecast.stage_forecast(cfg, cells)          # requested in docs/dynorch/requests/wpD.md
    except (ImportError, AttributeError) as exc:
        out["forecast"] = f"unavailable: {exc}"
    if args.write:
        if not args.run_dir:
            print("refused: --write needs --run-dir", file=sys.stderr)
            return EXIT_STOP
        paths = RunPaths.for_run(args.run_dir)
        try:
            from agent_allocation.dynorch.plan import plan_runs
        except ImportError as exc:
            print(f"blocked: dynorch.plan.plan_runs not available ({exc})", file=sys.stderr)
            return EXIT_BLOCKED
        specs = plan_runs(cfg, stage=args.stage, arms=list(arms or sorted({c['arm'] for c in cells})),
                          rungs=list(rungs or sorted({c['rung'] for c in cells})), repeats=max(c["repeat"] for c in cells),
                          items=sorted({c["item_id"] for c in cells}))
        paths.manifests_dir.mkdir(parents=True, exist_ok=True)
        target = paths.manifests_dir / f"dynorch_plan_{args.stage}.jsonl"
        lines = [json.dumps(asdict(s), sort_keys=True, default=str) for s in specs]
        replace_atomic(target, ("\n".join(lines) + "\n").encode())
        out["written"] = str(target)
        out["run_specs"] = len(specs)
    _print(out)
    return EXIT_OK


def run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-usd", required=True)
    parser.add_argument("--approval", default=None, help="owner text recorded with a ceiling/allowance override")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--account-auto-reload", action="store_true",
                        help="the account auto-refills: the credit balance is not a ceiling (the key limit still is)")
    parser.add_argument("--arms", default=None)
    parser.add_argument("--rungs", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--rehearsal", action="store_true", help="fake transport end to end; never sends a request")


def cmd_dynorch_run(argv: list[str]) -> int:
    """``allocation dynorch-run --config CFG --run-dir RUN --stage S --allow-paid --max-usd N [...]`` (contract §8)."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-run")
    run_args(parser)
    args = parser.parse_args(argv)
    if args.rehearsal:
        os.environ[REHEARSAL_ENV] = "1"
    try:
        return asyncio.run(_run(args))
    except RunRefused as exc:
        print(f"refused: {redact(str(exc))}", file=sys.stderr)
        return EXIT_STOP
    except Blocked as exc:
        print(f"blocked: {redact(str(exc))}", file=sys.stderr)
        return EXIT_BLOCKED


async def _run(args: argparse.Namespace) -> int:
    from agent_allocation.config import load_config, validate_config
    from agent_allocation.live import active_ceiling, record_decision
    from agent_allocation.dynorch import reconcile
    cfg = load_config(args.config)
    problems = validate_config(cfg)
    if problems:
        raise RunRefused(f"config invalid: {problems}")
    if args.stage in PAID_STAGES and not args.allow_paid:
        raise RunRefused(f"stage {args.stage} sends paid model calls: nothing is sent without --allow-paid "
                         "(use `allocation dynorch-plan` for a dry run)")
    max_usd, override = checked_max_usd(cfg, args.max_usd, stage=args.stage, approval=args.approval)
    paths = RunPaths.for_run(args.run_dir)
    if args.run_dir and str(Path(args.run_dir).resolve()) == str(Path(cfg["base_run_root"]).resolve()):
        raise RunRefused("the sealed base run root is read-only; give dynorch its own --run-dir")
    prereqs = stage_prerequisites(cfg, paths, args.stage)
    if prereqs:
        raise RunRefused("stage prerequisites unmet: " + "; ".join(prereqs))
    paths.ensure()
    if override is not None:
        record_decision(paths, **override)
    arms = _parse_list(args.arms, str)
    rungs = _parse_list(args.rungs, int)
    cells = enumerate_cells(cfg, args.stage, arms=arms, rungs=rungs)
    if not cells:
        raise RunRefused("no cells to run after --arms/--rungs restriction")
    record_decision(paths, kind="dynorch_run_start", stage=args.stage, max_usd=str(max_usd),
                    account_auto_reload=bool(args.account_auto_reload), resume=bool(args.resume), cells=len(cells),
                    arms=sorted({c["arm"] for c in cells}), rungs=sorted({c["rung"] for c in cells}), rehearsal=rehearsal())
    client = make_client(cfg)
    status: dict[str, Any] = {"stage": args.stage, "started_at": utcnow(), "max_usd": str(max_usd), "cells": len(cells),
                              "by_cell": cell_summary(cells), "rehearsal": rehearsal(), "state": "starting"}
    try:
        action = await resolve_endpoint_lock(cfg, paths, client=client, stage=args.stage)
        status["endpoint_rule"] = action
        if action == "stop":
            status["state"] = "stop_notify"
            write_status(paths, status)
            print("stop_notify: both shard endpoints fail the plan §4 rule (locks/provider_decision.json)", file=sys.stderr)
            return EXIT_STOP_NOTIFY
        before = await reconcile.snapshot_account(client)
        reconcile.write_before(paths, args.stage, before)
        if before.payment_required:
            status["state"] = "stop_notify"
            write_status(paths, status)
            print("stop_notify: the account endpoints answered 402 before the stage started", file=sys.stderr)
            return EXIT_STOP_NOTIFY
        settled = Decimal(reconcile.ledger_totals(paths)["settled_usd"])
        snapshot = {"key": {"limit_remaining": before.key_limit_remaining_usd},
                    "account_balance_usd": str((before.total_credits_usd or Decimal(0)) - (before.total_usage_usd or Decimal(0)))}
        ceiling = active_ceiling(max_usd, settled, snapshot, account_auto_reload=bool(args.account_auto_reload))
        status["active_ceiling_usd"] = str(ceiling)
        stage_marker(paths, args.stage)
        write_status(paths, {**status, "state": "running"})
        summary = await launch_stage(cfg, paths, stage=args.stage, cells=cells, max_usd=max_usd, active_ceiling_usd=ceiling,
                                     resume=bool(args.resume), concurrency=args.concurrency, client=client,
                                     account_auto_reload=bool(args.account_auto_reload),
                                     on_bound=lambda host, port: write_proxy_state(paths, host=host, port=port, stage=args.stage))
        status["runner"] = summary
        from agent_allocation.dynorch import tables as _tables
        try:
            built = _tables.build_tables(paths.run_dir, cfg)        # every pre-registered stage (contract §7)
            status["tables"] = {"n_planned": built["n_planned"], "n_records": built["n_records"],
                                "reasoning_unknown_rows": built["reasoning_unknown_rows"]}
        except _tables.TablesError as exc:                          # recorded, re-run with `dynorch-tables`
            print(f"tables not built: {redact(str(exc))}", file=sys.stderr)
            status["tables"] = f"error: {redact(str(exc))}"
        population, source, upstream_402 = reconcile.load_population(paths, stage=args.stage)
        totals = reconcile.ledger_totals(paths)
        before_doc = reconcile.read_before(paths, args.stage)
        hint = reconcile.ledger_charged(totals) - Decimal(before_doc["ledger_charged_usd"])
        report = await reconcile.reconcile_after(cfg, paths, args.stage, client=client, population=population,
                                                 population_source=source, upstream_402=upstream_402 or bool(summary.get("upstream_402")),
                                                 key_tolerance_usd=reconcile.key_tolerance_for(args.stage, hint))
        status["reconciliation"] = {k: report[k] for k in ("decision", "key_delta_usd", "ledger_delta_usd", "problems")}
        status["state"] = "stop_notify" if report["decision"] == reconcile.STOP_NOTIFY else "finished"
        status["finished_at"] = utcnow()
        write_status(paths, status)
        _print(status)
        return EXIT_STOP_NOTIFY if status["state"] == "stop_notify" else EXIT_OK
    finally:
        await client.aclose()


async def launch_stage(cfg: Mapping[str, Any], paths: RunPaths, *, stage: str, cells: Sequence[Mapping[str, Any]],
                       max_usd: Decimal, active_ceiling_usd: Decimal, resume: bool, concurrency: int | None, client: Any,
                       account_auto_reload: bool, on_bound: Callable[[str, int], Any]) -> dict[str, Any]:
    """Bind the proxy and run the stage through WP-B's runner. Codes against the requested entry point
    ``dynorch.runner.launch_stage(cfg, paths, *, stage, cells, max_usd, active_ceiling_usd, resume, concurrency,
    client, account_auto_reload, proxy_host, proxy_port, on_bound) -> dict`` (docs/dynorch/requests/wpD.md); the
    proxy binds ``127.0.0.1:0`` and ``on_bound(host, port)`` writes ``state/proxy.json`` before the first harness."""
    try:
        from agent_allocation.dynorch import runner
    except ImportError as exc:
        raise Blocked(f"dynorch.runner not available ({exc})") from None
    entry = getattr(runner, "launch_stage", None)
    if entry is None:
        raise Blocked("dynorch.runner.launch_stage missing (see docs/dynorch/requests/wpD.md)")
    proxy = cfg["dynorch"]["proxy"]
    result = await entry(cfg, paths, stage=stage, cells=list(cells), max_usd=max_usd, active_ceiling_usd=active_ceiling_usd,
                         resume=resume, concurrency=concurrency, client=client, account_auto_reload=account_auto_reload,
                         proxy_host=str(proxy["host"]), proxy_port=int(proxy["port"]), on_bound=on_bound)
    if not isinstance(result, Mapping):
        raise Blocked(f"runner.launch_stage returned {type(result).__name__}, expected a summary mapping")
    return dict(result)


def cmd_dynorch_tables(argv: list[str]) -> int:
    """``allocation dynorch-tables --config CFG --run-dir RUN [--stage S]``: WP-C's
    ``tables.build_tables(run_root, cfg, stage=...)`` (every pre-registered stage unless ``--stage`` narrows it)."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-tables")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", default=None, choices=list(STAGES))
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    try:
        from agent_allocation.dynorch import tables
        build = tables.build_tables
    except (ImportError, AttributeError) as exc:
        print(f"blocked: dynorch.tables.build_tables not available ({exc})", file=sys.stderr)
        return EXIT_BLOCKED
    result = build(paths.run_dir, cfg, stage=args.stage)
    _print({"tables_dir": str(paths.tables_dir),
            "result": {k: v for k, v in result.items() if k in ("n_planned", "n_records", "reasoning_unknown_rows")},
            "rows": {k: int(len(v)) for k, v in result.items() if hasattr(v, "__len__") and not isinstance(v, (str, int))}})
    return EXIT_OK


# ---- gates ------------------------------------------------------------------------------------------------------

def load_tables(paths: RunPaths, *, stage: str) -> tuple[Any, Any]:
    """WP-C's runs/calls tables restricted to one partition (raises when absent)."""
    import pandas as pd
    runs_path = paths.tables_dir / "dynorch_runs.parquet"
    calls_path = paths.tables_dir / "dynorch_calls.parquet"
    if not runs_path.exists() or not calls_path.exists():
        raise Blocked(f"{runs_path.name} / {calls_path.name} missing: run `allocation dynorch-tables` first (WP-C)")
    runs = pd.read_parquet(runs_path)
    calls = pd.read_parquet(calls_path)
    if "partition" not in runs.columns:
        raise RunRefused("dynorch_runs.parquet has no 'partition' column")
    runs = runs[runs["partition"] == stage]
    calls = calls[calls["run_key"].isin(set(runs["run_key"]))]
    return runs, calls


def new_files_under_home(home: Path, since: float, *, exclude: Sequence[str] = (".claude", ".cursor-server", ".vscode-server", ".cache/uv"),
                         budget_seconds: float = 600.0, limit: int = 50) -> list[str] | None:
    """Files under ``home`` modified after ``since`` (epoch seconds), skipping the interactive tools' own state.
    ``None`` when the walk exceeds its time budget (reported as "not scanned", never as clean)."""
    started = time.monotonic()
    hits: list[str] = []
    skip = {str(home / e) for e in exclude}
    for dirpath, dirs, files in os.walk(home, onerror=lambda _e: None):
        if time.monotonic() - started > budget_seconds:
            return None
        dirs[:] = [d for d in dirs if str(Path(dirpath) / d) not in skip]
        for name in files:
            path = Path(dirpath) / name
            try:
                if path.lstat().st_mtime > since:
                    hits.append(str(path))
            except OSError:
                continue
            if len(hits) >= limit:
                return hits
    return hits


def key_hits_in_run_tree(run_root: Path) -> list[str]:
    """Key-shaped strings under the run root, gzip journals included (paths only, never the string)."""
    import gzip
    import re
    pattern = re.compile(r"sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{8,}")
    hits: list[str] = []
    for path in sorted(Path(run_root).rglob("*")):
        if not path.is_file():
            continue
        try:
            data = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
        except (OSError, EOFError):
            continue
        if pattern.search(data.decode("utf-8", errors="replace")):
            hits.append(str(path.relative_to(run_root)))
    return hits


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def run_gate0(cfg: Mapping[str, Any], paths: RunPaths, *, skip_tests: bool) -> Any:
    from agent_allocation.dynorch import doctor, gates
    checks = doctor.run_checks(doctor.DoctorContext(cfg=cfg, run_root=paths.run_dir))
    if skip_tests:
        rc, summary = None, "skipped (--skip-tests)"
    else:
        env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests/unit/dynorch", "-q", "-m", "not live", "-p", "no:cacheprovider"],
                              cwd=doctor.project_root(), capture_output=True, text=True, env=env, timeout=3600)
        rc, summary = proc.returncode, (proc.stdout or "").strip()
    smoke = _read_json(paths.run_dir / "gates" / "gate0_smoke.json")
    return gates.evaluate_gate0(checks, pytest_returncode=rc, pytest_summary=summary, smoke_report=smoke)


def run_gate1(cfg: Mapping[str, Any], paths: RunPaths, *, home_scan: bool) -> Any:
    from agent_allocation.dynorch import endpoint_rule, gates, reconcile
    runs, calls = load_tables(paths, stage="canary")
    recon = reconcile.read_report(paths, "canary") if reconcile.report_path(paths, "canary").exists() else None
    decision = endpoint_rule.read_decision(paths.run_dir) if paths.endpoint_lock.exists() else None
    marker = paths.state_dir / "stage_canary_started"
    home_new: list[str] | None = None
    if home_scan and marker.exists():
        home_new = new_files_under_home(Path.home(), marker.stat().st_mtime)
    result, overheads = gates.evaluate_gate1(cfg, runs, calls, recon, lock_decision=decision,
                                             key_hits=key_hits_in_run_tree(paths.run_dir), home_new_files=home_new)
    if overheads:
        paths.locks_dir.mkdir(parents=True, exist_ok=True)
        replace_atomic(paths.locks_dir / "harness_prompts.json",
                       json.dumps({"S_" + arm: doc for arm, doc in overheads.items()}, indent=1, sort_keys=True).encode())
    return result


def run_gate2(cfg: Mapping[str, Any], paths: RunPaths) -> Any:
    from agent_allocation.dynorch import gates, reconcile
    runs, calls = load_tables(paths, stage="pilot")
    recon = reconcile.read_report(paths, "pilot") if reconcile.report_path(paths, "pilot").exists() else None
    forecast: Mapping[str, Any] | None
    try:
        from agent_allocation.dynorch import forecast as _forecast
        forecast = _forecast.gate2_refit(cfg, runs, spent_usd=Decimal(reconcile.ledger_totals(paths)["settled_usd"]))
    except (ImportError, AttributeError):
        forecast = None
    result = gates.evaluate_gate2(cfg, runs, calls, recon, forecast=forecast)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    replace_atomic(paths.reports_dir / "PILOT_REPORT.md",
                   gates.render_pilot_report(cfg, runs, calls, result, recon, forecast).encode("utf-8"))
    return result


def regrade_mismatches(paths: RunPaths, runs: Any, base_run_root: Path) -> int:
    """Re-grade every VALID final answer through WP-B's ``grade.candidate_from_text`` + ``grade_text`` and count
    rows whose verdict differs from the table (Gate 4 ``grader_rerun_identical``)."""
    try:
        from agent_allocation.dynorch.grade import candidate_from_text
        from agent_allocation.grading.text.grade import grade_text
        from agent_allocation.grading.worker import load_tasks
    except ImportError as exc:
        raise Blocked(f"grader re-run needs dynorch.grade ({exc})") from None
    public, private = load_tasks(RunPaths.for_run(base_run_root, local_state_dir=paths.local_state_dir))
    mismatches = 0
    for rec in runs.to_dict("records"):
        if rec.get("parse_status") != "VALID" or not isinstance(rec.get("final_message"), str):
            continue
        task = public[rec["item_id"]]
        candidate = candidate_from_text(rec["final_message"], task, node_id=rec["run_key"])
        verdict = grade_text(candidate, private[rec["item_id"]])
        if verdict.semantic_correct != rec.get("correct"):
            mismatches += 1
    return mismatches


def run_gate4(cfg: Mapping[str, Any], paths: RunPaths, *, stage: str) -> Any:
    import pandas as pd
    from agent_allocation.dynorch import gates, reconcile
    outcomes_path = paths.tables_dir / "policy_outcomes_dynorch.parquet"
    if not outcomes_path.exists():
        raise Blocked(f"{outcomes_path.name} missing: run `allocation dynorch-tables` first (WP-C)")
    outcomes = pd.read_parquet(outcomes_path)
    partition = stage_spec(cfg, stage).partition
    if "partition_id" in outcomes.columns:
        outcomes = outcomes[outcomes["partition_id"] == partition]
    runs, _calls = load_tables(paths, stage=stage)
    cells = enumerate_cells(cfg, stage)
    recon = reconcile.read_report(paths, stage) if reconcile.report_path(paths, stage).exists() else None
    base = Path(cfg["base_run_root"])
    sealed = gates.seal_tables(paths.tables_dir)
    return gates.evaluate_gate4(cfg, cells, outcomes, recon, grader_rerun_mismatches=regrade_mismatches(paths, runs, base),
                                ladder_estimates=gates.ladder_estimates_from_base(base), sealed_tables=sealed, stage=stage)


def cmd_dynorch_gates(argv: list[str]) -> int:
    """``allocation dynorch-gates --config CFG --run-dir RUN --gate 0|1|2|4 [--stage S] [--owner-go TEXT |
    --owner-no-go TEXT] [--skip-tests] [--no-home-scan]``. Exit 0 advance / 3 stop_notify / 1 stop / 2 blocked."""
    parser = argparse.ArgumentParser(prog="allocation dynorch-gates")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gate", required=True, choices=["0", "1", "2", "4"])
    parser.add_argument("--stage", default=None, help="gate 4: the sealed stage (default main)")
    parser.add_argument("--owner-go", default=None, help="gate 2: record the owner's go decision text")
    parser.add_argument("--owner-no-go", default=None, help="gate 2: record the owner's no-go decision text")
    parser.add_argument("--skip-tests", action="store_true", help="gate 0: do not run the unit suite")
    parser.add_argument("--no-home-scan", action="store_true", help="gate 1: skip the $HOME walk (criterion then fails)")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    from agent_allocation.dynorch import gates
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    try:
        if args.gate == "0":
            result = run_gate0(cfg, paths, skip_tests=args.skip_tests)
        elif args.gate == "1":
            result = run_gate1(cfg, paths, home_scan=not args.no_home_scan)
        elif args.gate == "2":
            result = run_gate2(cfg, paths)
        else:
            result = run_gate4(cfg, paths, stage=args.stage or "main")
    except Blocked as exc:
        print(f"blocked: {redact(str(exc))}", file=sys.stderr)
        return EXIT_BLOCKED
    except RunRefused as exc:
        print(f"refused: {redact(str(exc))}", file=sys.stderr)
        return EXIT_STOP
    target = gates.record(paths, result)
    from agent_allocation.contracts import dump_record
    _print({"gate": result.gate, "stage": result.stage, "decision": result.decision, "written": str(target),
            "failing": [c.name for c in result.criteria if not c.passed]})
    print(redact(json.dumps(dump_record(result), indent=1, sort_keys=True)))
    if args.gate == "2" and (args.owner_go or args.owner_no_go):
        from agent_allocation.live import record_decision
        if args.owner_go and args.owner_no_go:
            print("refused: give --owner-go or --owner-no-go, not both", file=sys.stderr)
            return EXIT_STOP
        kind = "owner_go" if args.owner_go else "owner_no_go"
        record_decision(paths, kind=kind, gate="gate2", gate_decision=result.decision, text=args.owner_go or args.owner_no_go)
        print(f"recorded {kind} in {paths.run_decisions}")
    return gates.exit_code(result)


def cmd_dynorch_reconcile(argv: list[str]) -> int:
    from agent_allocation.dynorch.reconcile import cmd_dynorch_reconcile as impl
    return impl(argv)


def cmd_dynorch_doctor(argv: list[str]) -> int:
    from agent_allocation.dynorch.doctor import cmd_dynorch_doctor as impl
    return impl(argv)
