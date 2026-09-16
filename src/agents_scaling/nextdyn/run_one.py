"""``python -m agents_scaling.nextdyn.run_one`` -- one Slurm array task = one nextdyn cell.

Plan §3.3/§3.4, spec 15.3 (preemption: stop admitting work, commit what is finished).
Compatible with ``slurm/nextdyn_cell_array.sbatch.tmpl``::

    python -m agents_scaling.nextdyn.run_one --run-root … --manifest … --index $SLURM_ARRAY_TASK_ID

Exit codes: ``0`` done (or a clean stop), ``2`` no endpoint within the wait, ``3`` items left
incomplete after the bounded retries, ``4`` suspended (a harness defect; terminal until a
human removes ``SUSPENDED.json``), ``5`` :data:`~agents_scaling.nextdyn.runner.EXIT_DEPS_UNMET`
-- a ``cell:`` / ``bundle:`` / ``register:`` dependency is absent or tampered.  Exit 5 is
**not** a failure: no ``meta.json`` was written and no request was issued, so the array task
exits 0 and the lane driver re-dispatches the cell once the dependency lands.

The dependency gate is checked twice on purpose: ``dispatch`` checks it before submitting (to
avoid burning array slots) and the runner checks it again here, before any endpoint is opened,
because between those two moments a register can be revoked or a bundle can be found tampered
(plan §3.4: "``run_one.py`` re-checks before opening any endpoint").

The per-cell lock is a **non-blocking** ``flock``: a second runner on a locked cell exits 0
immediately rather than waiting, because the first one is already doing the work and a waiting
array task is a Slurm slot that could be running another cell
(``test_a_second_runner_on_a_locked_cell_exits_zero``).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import threading
from pathlib import Path
from typing import Sequence

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import kinds
from agents_scaling.nextdyn.errors import NextdynError
from agents_scaling.nextdyn.runner import EXIT_DEPS_UNMET, NextdynCellRunner, pool_run_root
from agents_scaling.study.run_one import install_stop_handler   # the frozen SIGUSR1/SIGTERM handler
from agents_scaling.study.types import EXIT_DONE, EXIT_SUSPENDED

LOCK_FILE = ".lock"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.nextdyn.run_one",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-root", default=None, help="the run root (or --run-id + --results-root)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--manifest", required=True, help="a block manifest (absolute, or under the run root)")
    p.add_argument("--index", required=True, type=int, help="index into the manifest (= $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--config", type=Path, default=C.DEFAULT_STUDY_CONFIG)
    p.add_argument("--server-run-root", default=None, help="the pool's sub-run-root (plan §3.2)")
    p.add_argument("--max-inflight", type=int, default=None)
    p.add_argument("--wait-s", type=float, default=None, help="endpoint wait before exit 2")
    p.add_argument("--dry-run", action="store_true", help="print the resolved cell and exit 0")
    return p


def resolve_run_root(args: argparse.Namespace) -> Path:
    if args.run_root:
        return Path(args.run_root)
    if args.run_id:
        return Path(args.results_root) / args.run_id
    raise NextdynError("pass --run-root or --run-id")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_root = resolve_run_root(args)
    except NextdynError as exc:
        sys.stderr.write(f"[run_one] {exc}\n")
        return EXIT_SUSPENDED
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = run_root / manifest
    try:
        cells = CE.load_manifest(manifest)
    except NextdynError as exc:
        sys.stderr.write(f"[run_one] {manifest}: {exc}\n")
        return EXIT_SUSPENDED
    if not 0 <= args.index < len(cells):
        sys.stderr.write(f"[run_one] index {args.index} outside 0..{len(cells) - 1}\n")
        return EXIT_SUSPENDED
    cell = cells[args.index]
    if args.dry_run:
        print(json.dumps({"index": args.index, "run_root": str(run_root), "cell": cell.to_dict()},
                         indent=2, sort_keys=True))
        return EXIT_DONE
    if cell.reuse_of:
        sys.stderr.write(f"[run_one] {cell.cell_id} reuses {cell.reuse_of}; nothing to dispatch\n")
        return EXIT_DONE

    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    server_run_root = Path(args.server_run_root) if args.server_run_root else (
        pool_run_root(run_root, cell.pool) if cell.pool is not None else run_root)

    cell_dir = run_root / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = open(cell_dir / LOCK_FILE, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.stderr.write(f"[run_one] cell {cell.cell_id} is already being run by another task; "
                         "exiting 0\n")
        return EXIT_DONE

    stop_event = threading.Event()
    install_stop_handler(stop_event)
    kwargs = {}
    if args.wait_s is not None:
        kwargs["wait_timeout_s"] = float(args.wait_s)
    if args.max_inflight is not None:
        kwargs["max_inflight"] = int(args.max_inflight)
    runner = NextdynCellRunner(cell, run_root, server_run_root, args.index, stop_event,
                               cfg=cfg, compute=compute, run_id=run_root.name, **kwargs)
    code = runner.run()
    needs_endpoint = kinds.spec_for(cell.kind).needs_endpoint
    sys.stderr.write(f"[run_one] cell {cell.cell_id} ({cell.kind.value}, endpoint="
                     f"{needs_endpoint}) exit {code}\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["EXIT_DEPS_UNMET", "LOCK_FILE", "build_parser", "main", "resolve_run_root"]
