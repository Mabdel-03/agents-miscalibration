"""``python -m agents_scaling.ard.run_one`` — one Slurm array task = one ARD36 cell (WP4).

Spec §10.4-style preemption handling (``SIGUSR1`` from ``--signal=B:USR1@1200`` sets the
stop event; the runner finishes in-flight items and exits 0 **without** ``meta.json`` so the
lane driver re-queues the cell), §12.3 (a controller-gated cell that cannot see its frozen
controller must not run).  Plan §10 (the new ``EXIT_DEPS_UNMET`` path).

Compatible with ``slurm/ard36_cell_array.sbatch.tmpl``::

    python -m agents_scaling.ard.run_one --run-id … --server-run-id … \
        --cells-file … --index $SLURM_ARRAY_TASK_ID

Exit codes: 0 done (or a stop), 2 no endpoint, 3 incomplete items, 4 suspended (terminal,
needs a human), **5 ``EXIT_DEPS_UNMET``** — a ``controller:``/``state:`` dependency is
absent or tampered; no ``meta.json`` is written and no request is generated, so the driver
re-queues the cell when the dependency lands.  The signal handler and the per-cell flock are
the frozen study ones (``study/run_one.py``), imported rather than copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from agents_scaling.ard.cells import load_ard_cells_file
from agents_scaling.ard.runner import DEPS_UNMET_FILE, run_ard_cell
from agents_scaling.ard.types import EXIT_DEPS_UNMET, EXIT_SUSPENDED
from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.study.run_one import install_stop_handler  # frozen SIGUSR1/SIGTERM handler
from agents_scaling.study.types import ProtocolError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.run_one",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--cells-file", required=True, help="absolute path or a name under the run root")
    p.add_argument("--index", required=True, type=int, help="index into the cells file (= $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--server-run-id", default=None, help="run id whose servers/ registry holds the endpoints")
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--max-inflight", type=int, default=None)
    p.add_argument("--wait-s", type=float, default=300.0, help="endpoint wait before exit 2")
    p.add_argument("--dry-run", action="store_true", help="print the resolved cell and exit 0")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    run_root = results_root / args.run_id
    server_run_root = results_root / (args.server_run_id or args.run_id)
    cells_path = Path(args.cells_file)
    if not cells_path.is_absolute():
        cells_path = run_root / cells_path
    try:
        cells = load_ard_cells_file(cells_path)
    except ProtocolError as exc:
        sys.stderr.write(f"[ard-run_one] {exc}\n")
        return EXIT_SUSPENDED
    if not 0 <= args.index < len(cells):
        sys.stderr.write(f"[ard-run_one] index {args.index} outside 0..{len(cells) - 1}\n")
        return EXIT_SUSPENDED
    cell = cells[args.index]
    if args.dry_run:
        print(json.dumps({"index": args.index, "run_root": str(run_root),
                          "server_run_root": str(server_run_root), "cell": cell.to_dict()}, indent=2))
        return 0
    # One runner per cell at a time (study-v4): overlapping arrays would otherwise generate
    # the same content-addressed requests twice.  The second runner exits 0 without meta.json.
    import fcntl

    cell_dir = run_root / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = open(cell_dir / ".lock", "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.stderr.write(f"[ard-run_one] cell {cell.cell_id} is already being run; exiting 0\n")
        return 0
    stop_event = threading.Event()
    install_stop_handler(stop_event)
    code = run_ard_cell(cell, run_root, server_run_root, args.index, stop_event,
                        max_inflight=args.max_inflight, wait_timeout_s=args.wait_s,
                        run_id=args.run_id)
    if code == EXIT_DEPS_UNMET:
        sys.stderr.write(
            f"[ard-run_one] cell {cell.cell_id}: dependencies unmet "
            f"(see {cell_dir / DEPS_UNMET_FILE}); no meta.json, no requests — the lane driver "
            f"re-queues this cell when the dependency lands\n")
    sys.stderr.write(f"[ard-run_one] cell {cell.cell_id} exit {code}\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
