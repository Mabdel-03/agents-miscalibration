"""The ``audit-legacy`` verb: run every E0 leg and write ``<run_root>/e0/`` (spec 7.1-7.4).

Order matters and is fixed: the **inventory** runs first, so every later leg can say whether
its input was even present; the protocol map runs next, so a contrast row can state which
protocol it is comparing; then the three claim legs; then the contrast table; then the BCB
repair plan.  A leg whose sources are missing writes its ``BLOCKED_MISSING_SOURCE`` rows and
the pass continues -- one absent legacy artifact must not stop the audit of the others.

``--bcb-cells`` shards the behavioural repair across the cpu array
(``study.experiments.E0.bcb_array_cells``); with no shard index the verb writes the plan and
the input census without entering the sandbox, which is what makes the leg testable offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final, Sequence

from agents_scaling.nextdyn import cli, config as C
from agents_scaling.nextdyn.e0 import (
    ard36, atlas, bcb_inputgen, bcb_repair, claims, contrasts, inventory, protocol_map,
    registry as R, run0,
)

E0_DIR: Final[str] = "e0"


def out_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / E0_DIR


def bcb_plan(cfg: Any, sources: R.LegacySources, *, shard: int | None = None) -> dict[str, Any]:
    """The BCB repair plan: how many cpu cells, which tasks in this shard, what the policy is.

    ``study.experiments.E0.bcb_array_cells`` is the one declaration of the array size; the shard
    assignment is ``index % n_cells`` over the pinned parquet's file order, so a shard is a
    deterministic function of the frozen input and not of when it ran.
    """
    n_cells = int(cfg.experiments.E0.bcb_array_cells)
    settings = bcb_repair.settings(sources)
    raw = R.resolve(sources, "study_v4.raw_bcb", hash_files=False)
    plan: dict[str, Any] = {
        "n_cells": n_cells, "shard": shard, **settings,
        "raw_bcb": str(raw.path), "raw_bcb_present": raw.exists,
    }
    if not raw.exists:
        plan["status"] = R.BLOCKED_MISSING_SOURCE
        plan["reason"] = f"{raw.path} is absent"
        return plan
    try:
        from agents_scaling.study.data.bcb import load_bcb_rows

        rows = load_bcb_rows(raw.path, expected_rows=None)
    except Exception as exc:  # pragma: no cover - a corrupt pin is a finding
        plan["status"] = R.BLOCKED_MISSING_SOURCE
        plan["reason"] = f"{raw.path} is unreadable ({exc})"
        return plan
    study_seed = bytes.fromhex(cfg.seeds.study_seed_hex)
    selected = [row for index, row in enumerate(rows)
                if shard is None or index % n_cells == int(shard)]
    generated = [
        bcb_inputgen.generate(str(row.get("task_id") or index),
                              str(row.get("instruct_prompt") or ""), study_seed,
                              max_inputs=settings["max_inputs_per_task"])
        for index, row in enumerate(selected)
    ]
    plan["status"] = "ok"
    plan["n_tasks_total"] = len(rows)
    plan["n_tasks_in_shard"] = len(selected)
    plan["input_census"] = bcb_inputgen.census(generated)
    plan["note"] = ("the generator reads the public instruct_prompt only; hidden tests are read "
                    "by the audit stage after selection_seal.json exists")
    return plan


def run(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
        registry_path: str | os.PathLike | None = None, shard: int | None = None,
        n_resamples: int = 20_000) -> cli.Report:
    """Run every leg and write ``<run_root>/e0/``."""
    cfg = C.load_study(Path(config_path) if config_path else C.DEFAULT_STUDY_CONFIG)
    sources = R.load(registry_path)
    folder = out_dir(run_root)
    folder.mkdir(parents=True, exist_ok=True)

    inventory_rows = inventory.legacy_inventory(sources)
    inventory.write(folder, inventory_rows)
    blocked = inventory.blocked_sources(inventory_rows)

    protocol_map.write(folder, protocol_map.protocol_map(sources))

    claim_rows = claims.collect(sources, n_resamples=n_resamples)
    claims.write(folder, claim_rows)
    ard36.write_discordant(folder, ard36.discordant_audit(sources))

    contrasts.write(folder, contrasts.contrast_table(sources))
    contrasts.write_overlap(folder, contrasts.overlap_inventory(sources))

    repair_dir = folder / bcb_repair.REPAIR_DIR
    repair_dir.mkdir(parents=True, exist_ok=True)
    plan = bcb_plan(cfg, sources, shard=shard)
    (repair_dir / "plan.json").write_text(
        json.dumps(plan, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")

    census = claims.summary(claim_rows)
    (folder / "SUMMARY.json").write_text(json.dumps(
        {"claims": census, "blocked_sources": blocked, "bcb": plan.get("status"),
         "registry": str(sources.path)}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return cli.Report(
        completed=census.get("pass", 0) + census.get("ok", 0),
        failed=census.get("fail", 0),
        blocked=census.get(R.BLOCKED_MISSING_SOURCE, 0) + census.get("unavailable", 0),
        detail=f"{folder}: {census}")


def e0_worker(runner: Any, source_item_id: str) -> "tables.ItemFile":
    """The E0 cell worker named by ``kinds.KIND_TABLE[CellKind.E0].worker``.

    E0 generates nothing: it reads three sealed legacy run roots and writes ``<run_root>/e0/``.
    It is a cell rather than a login-node script so that the same census, the same resume rule
    and the same typed failure reporting cover it as cover every other stage -- ARD36's habit of
    running analyses by hand is what let a stage "succeed" while producing the wrong experiment.
    One cell does one leg, named by ``runner.cell.options['leg']`` (default: every leg).

    The returned ``ItemFile`` carries no rows: E0's outputs are its own ``e0/`` artifacts, and
    declaring zero rows keeps it out of every table whose columns it does not produce.
    """
    from agents_scaling.nextdyn import tables
    from agents_scaling.nextdyn.ids import CellKind

    cell = runner.cell
    options = dict(getattr(cell, "options", {}) or {})
    report = run(runner.run_root, config_path=getattr(runner, "config_path", None),
                 registry_path=options.get("registry_path"),
                 shard=options.get("bcb_shard"),
                 n_resamples=int(options.get("n_resamples", 20_000)))
    status = tables.ItemStatus.COMPLETE if report.failed == 0 else tables.ItemStatus.FAILED
    return tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=str(cell.cell_id),
        cell_kind=CellKind.E0, source_item_id=str(source_item_id), status=status,
        bundle_export_hash=None, rows={})


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("audit-legacy")
    parser.add_argument("--registry", default=None, help="legacy_sources.yaml (default: configs/nextdyn)")
    parser.add_argument("--bcb-shard", type=int, default=None, help="cpu array shard index")
    parser.add_argument("--resamples", type=int, default=20_000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(cli.resolve_run_root(args), config_path=args.config, registry_path=args.registry,
               shard=args.bcb_shard, n_resamples=args.resamples)


__all__ = ["E0_DIR", "bcb_plan", "e0_worker", "main", "out_dir", "run"]
