"""``python -m agents_scaling.study.run_one`` — one Slurm array task = one cell (WP5).

Spec §10.4 (resumable idempotent worker; on preemption stop admitting work and commit
completed artifacts: ``SIGUSR1`` from ``--signal=B:USR1@1200`` sets the stop event and the
runner finishes in-flight items, exits 0 without ``meta.json`` so the driver re-queues).
Architecture §1.13, §3.  Compatible with ``slurm/study_cell_array.sbatch.tmpl``::

    exec python -m agents_scaling.study.run_one --run-id … --server-run-id … --cells-file … --index $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.study.cells import load_cells_file
from agents_scaling.study.runner import run_cell
from agents_scaling.study.types import EXIT_SUSPENDED, ProtocolError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.run_one", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--cells-file", required=True, help="absolute path or a name under the run root")
    p.add_argument("--index", required=True, type=int, help="index into the cells file (= $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--server-run-id", default=None, help="run id whose servers/ registry holds the endpoints (default: --run-id)")
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--max-inflight", type=int, default=None, help="override the cell's per-item request concurrency")
    p.add_argument("--wait-s", type=float, default=300.0, help="endpoint wait before exit 2")
    p.add_argument("--dry-run", action="store_true", help="print the resolved cell and exit 0")
    return p


def install_stop_handler(stop_event: threading.Event) -> None:
    """``SIGUSR1`` → set the event (architecture §3 step 8); also honours ``SIGTERM``."""

    def _handler(signum, _frame) -> None:  # noqa: ANN001
        sys.stderr.write(f"[run_one] signal {signum}: finishing in-flight items, then exiting without meta.json\n")
        stop_event.set()

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, _handler)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    run_root = results_root / args.run_id
    server_run_root = results_root / (args.server_run_id or args.run_id)
    cells_path = Path(args.cells_file)
    if not cells_path.is_absolute():
        cells_path = run_root / cells_path
    try:
        cells = load_cells_file(cells_path)
    except ProtocolError as exc:
        sys.stderr.write(f"[run_one] {exc}\n")
        return EXIT_SUSPENDED
    if not 0 <= args.index < len(cells):
        sys.stderr.write(f"[run_one] index {args.index} outside 0..{len(cells) - 1}\n")
        return EXIT_SUSPENDED
    cell = cells[args.index]
    if args.dry_run:
        print(json.dumps({"index": args.index, "run_root": str(run_root), "server_run_root": str(server_run_root), "cell": cell.to_dict()}, indent=2))
        return 0
    # study-v4: one runner per cell at a time.  Overlapping chunk arrays (driver re-submission,
    # manual one-shots) would otherwise generate the same content-addressed requests twice;
    # the second runner exits 0 without meta.json and the driver re-queues nothing while
    # the first is still listed.
    import fcntl
    cell_dir = run_root / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = open(cell_dir / ".lock", "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.stderr.write(f"[run_one] cell {cell.cell_id} is already being run by another task; exiting 0\n")
        return 0
    stop_event = threading.Event()
    install_stop_handler(stop_event)
    code = run_cell(
        cell, run_root, server_run_root, args.index, stop_event,
        max_inflight=args.max_inflight, wait_timeout_s=args.wait_s, run_id=args.run_id,
    )
    sys.stderr.write(f"[run_one] cell {cell.cell_id} exit {code}\n")
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "install_stop_handler", "main"]
