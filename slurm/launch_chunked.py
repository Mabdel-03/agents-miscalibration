"""Submit a large sweep as dependency-chained SLURM array CHUNKS.

A single array of all cells exceeds the cluster's MaxSubmitJobs association limit (500),
so we split the cell list into chunks of <= CHUNK_SIZE and submit one array job per chunk,
each gated on the previous chunk via ``--dependency=afterany`` so at most one chunk's
worth of tasks is ever queued/running. The per-cell runner is resumable (skips cells with
meta.json), so chunks that get preempted/killed are safely re-runnable.

This does NOT (re)launch servers — run ``launch_sweep.py --serve-only`` (and/or
``launch_server.py``) first so endpoints are registered. Cells discover servers via the
multi-endpoint registry and round-robin across them.

Usage:
  python slurm/launch_chunked.py --config configs/full_sweep.yaml --run-id full_sweep_v1 \
      --chunk-size 480 --throttle 480 --cell-partition mit_preemptable --cell-time 2-00:00:00
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment import io
from agents_scaling.experiment.sweep import load_sweep

REPO = Path(__file__).resolve().parent.parent
ARRAY_TMPL = REPO / "slurm" / "run_cell_array.sbatch.tmpl"


def _render(run_id, cells_file, lo, hi, throttle, partition, time_limit, log_dir) -> str:
    """Render one chunk array covering global indices [lo, hi] (inclusive)."""
    text = ARRAY_TMPL.read_text()
    # The template uses array 0-LAST%THROTTLE; for a chunk we want lo-hi%throttle.
    text = text.replace("--array=0-{LAST_INDEX}%{THROTTLE}", "--array={LO}-{HI}%{THROTTLE}")
    repl = {
        "RUN_ID": run_id,
        "CELLS_FILE": cells_file,
        "LO": str(lo),
        "HI": str(hi),
        "THROTTLE": str(throttle),
        "PARTITION": partition,
        "TIME": time_limit,
        "LOG_DIR": log_dir,
        "REPO": str(REPO),
    }
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a sweep as chained array chunks.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--chunk-size", type=int, default=400, help="cells per array chunk")
    ap.add_argument("--throttle", type=int, default=400, help="max concurrent tasks per chunk")
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="2-00:00:00")
    ap.add_argument("--max-chunks", type=int, default=None, help="cap chunks (debug)")
    ap.add_argument("--dry-run", action="store_true", help="render sbatches, don't submit")
    # Drive mode (default): submit one chunk at a time, waiting for the submitted-job count
    # to drop below the QOS submit cap before submitting the next. This works around
    # QOSMaxSubmitJobPerUserLimit (a whole array counts as N submitted jobs), which forbids
    # pre-queuing many chunks. The runner is resumable, so a killed driver can just re-run.
    ap.add_argument("--submit-cap", type=int, default=440,
                    help="max submitted jobs to keep under (QOS QOSMaxSubmitJobPerUserLimit)")
    ap.add_argument("--no-drive", action="store_true",
                    help="don't drive; just render chunk sbatches (use --dry-run to inspect)")
    ap.add_argument("--reshuffle", action="store_true",
                    help="regenerate cells.json with a fresh interleave (ONLY when no array is "
                         "mid-flight against the old ordering — resume is by cell_id so completed "
                         "work is safe, but in-flight array indices would point at different cells)")
    ap.add_argument("--poll-s", type=float, default=120.0, help="drive poll interval (s)")
    args = ap.parse_args()

    cells = load_sweep(args.config)
    run_root = io.run_dir(args.run_id)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Persist the canonical cell list the arrays index into (single source of truth).
    # IMPORTANT: cells are generated sorted by cell_id, which CLUSTERS them by model size
    # (e.g. the first chunk would be all 0.6B). With per-chunk concurrency that sends every
    # running cell to one size's server(s) while the rest sit idle. So we INTERLEAVE the
    # cell list deterministically (seeded shuffle) so each chunk spans all sizes and load
    # spreads across the whole fleet. Deterministic => array-index -> cell is stable across
    # re-runs; and the runner resumes by cell_id, so reordering never loses completed work.
    cells_file = run_root / "cells.json"
    if cells_file.exists() and not args.reshuffle:
        # Reuse the existing ordering so a running array's index->cell mapping is preserved.
        print(f"[chunked] reusing existing {cells_file} (pass --reshuffle to regenerate)")
    else:
        import random

        order = list(range(len(cells)))
        random.Random(1234).shuffle(order)  # fixed seed -> reproducible interleaving
        cells = [cells[i] for i in order]
        cells_file.write_text(json.dumps([c.to_dict() for c in cells], indent=2))
        print(f"[chunked] wrote interleaved {cells_file}")
    # Re-load from the file so the in-memory list matches exactly what array tasks index.
    cells = [ExperimentCell.from_dict(c) for c in json.loads(cells_file.read_text())]
    n = len(cells)
    print(f"[chunked] run_id={args.run_id}  cells={n}  chunk_size={args.chunk_size}  -> {cells_file}")

    chunks = [(lo, min(lo + args.chunk_size - 1, n - 1)) for lo in range(0, n, args.chunk_size)]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"[chunked] {len(chunks)} chunks, throttle {args.throttle}, partition {args.cell_partition}")

    # Render all chunk sbatches up front (cheap; also the audit trail).
    chunk_paths = []
    for ci, (lo, hi) in enumerate(chunks):
        text = _render(args.run_id, str(cells_file), lo, hi, args.throttle,
                       args.cell_partition, args.cell_time, str(log_dir))
        p = run_root / f"chunk_{ci:03d}_{lo}-{hi}.sbatch"
        p.write_text(text)
        chunk_paths.append((ci, lo, hi, p))
        if args.dry_run:
            print(f"  [dry-run] chunk {ci}: indices {lo}-{hi} -> {p}")
    if args.dry_run or args.no_drive:
        return

    _drive(run_root, chunk_paths, args.submit_cap, args.poll_s)


def _my_submitted_count() -> int:
    """Number of my jobs+array-tasks currently submitted (PD/R), which is what the QOS
    QOSMaxSubmitJobPerUserLimit counts. `squeue -r` expands array tasks to one line each."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"],
        capture_output=True, text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def _chunk_complete(run_root, lo: int, hi: int, cells: list) -> bool:
    """A chunk is done when every cell index in [lo,hi] has a meta.json (resumable)."""
    cdir = run_root / "cells"
    for idx in range(lo, hi + 1):
        cid = ExperimentCell.from_dict(cells[idx]).cell_id
        if not (cdir / cid / "meta.json").exists():
            return False
    return True


def _drive(run_root, chunk_paths, submit_cap: int, poll_s: float) -> None:
    """Submit chunks one at a time, keeping total submitted jobs under ``submit_cap``."""
    import time

    cells = json.loads((run_root / "cells.json").read_text())
    submitted_log: list[dict] = []
    for ci, lo, hi, path in chunk_paths:
        if _chunk_complete(run_root, lo, hi, cells):
            print(f"[drive] chunk {ci} ({lo}-{hi}) already complete; skipping")
            continue
        # Wait until there's headroom for this chunk's array tasks under the QOS cap.
        need = hi - lo + 1
        while True:
            cur = _my_submitted_count()
            if cur + need <= submit_cap:
                break
            print(f"[drive] waiting: {cur} submitted + {need} chunk > cap {submit_cap}; sleep {poll_s:.0f}s")
            time.sleep(poll_s)
        out = subprocess.run(["sbatch", "--parsable", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
        job_id = out.split(";")[0]
        submitted_log.append({"chunk": ci, "lo": lo, "hi": hi, "job_id": job_id})
        (run_root / "chunk_jobs.json").write_text(json.dumps(submitted_log, indent=2))
        print(f"[drive] chunk {ci:3d}: indices {lo:5d}-{hi:5d} -> job {job_id}  (submitted now: {_my_submitted_count()})")
        # brief settle so squeue reflects the new tasks before the next headroom check
        time.sleep(10)
    print(f"[drive] all {len(chunk_paths)} chunks dispatched; results under {run_root}/cells/")


if __name__ == "__main__":
    main()
