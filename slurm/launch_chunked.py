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


def _render(run_id, cells_file, lo, hi, throttle, partition, time_limit, log_dir,
            mem: str = "16G") -> str:
    """Render one chunk array covering global indices [lo, hi] (inclusive)."""
    text = ARRAY_TMPL.read_text()
    # The template uses array 0-LAST%THROTTLE; for a chunk we want lo-hi%throttle.
    text = text.replace("--array=0-{LAST_INDEX}%{THROTTLE}", "--array={LO}-{HI}%{THROTTLE}")
    # Run-scope the job name so concurrent sweeps don't count each other's cells (see
    # _my_submitted_count). The template ships a bare `asys-cells` name.
    text = text.replace("--job-name=asys-cells", f"--job-name=asys-cells-{run_id}")
    repl = {
        "RUN_ID": run_id,
        "CELLS_FILE": cells_file,
        "LO": str(lo),
        "HI": str(hi),
        "THROTTLE": str(throttle),
        "PARTITION": partition,
        "MEM": mem,
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
    ap.add_argument("--cell-mem", default="16G")
    ap.add_argument("--max-chunks", type=int, default=None, help="cap chunks (debug)")
    ap.add_argument("--dry-run", action="store_true", help="render sbatches, don't submit")
    # Drive mode (default): submit one chunk at a time, waiting for the submitted-job count
    # to drop below the QOS submit cap before submitting the next. This works around
    # QOSMaxSubmitJobPerUserLimit (a whole array counts as N submitted jobs), which forbids
    # pre-queuing many chunks. The runner is resumable, so a killed driver can just re-run.
    ap.add_argument("--submit-cap", type=int, default=400,
                    help="target number of CELL tasks to keep queued (counts asys-cells only). "
                         "The driver ALSO enforces --qos-limit on ALL jobs, so this can stay high.")
    ap.add_argument("--qos-limit", type=int, default=448,
                    help="absolute QOSMaxSubmitJobPerUserLimit across ALL my jobs (serve + "
                         "loops + cells). The driver waits until a whole chunk fits under this "
                         "alongside everything else, so growing the serve fleet never starves "
                         "cell dispatch. Set to the real per-user submit cap of the cell QOS.")
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
                       args.cell_partition, args.cell_time, str(log_dir), mem=args.cell_mem)
        p = run_root / f"chunk_{ci:03d}_{lo}-{hi}.sbatch"
        p.write_text(text)
        chunk_paths.append((ci, lo, hi, p))
        if args.dry_run:
            print(f"  [dry-run] chunk {ci}: indices {lo}-{hi} -> {p}")
    if args.dry_run or args.no_drive:
        return

    _drive(run_root, chunk_paths, args.submit_cap, args.poll_s,
           qos_limit=args.qos_limit, chunk_size=args.chunk_size, run_id=args.run_id)


def _my_submitted_count(run_id: str) -> int:
    """Number of THIS RUN's cell array-tasks currently submitted (PD/R). `squeue -r` expands
    array tasks to one line each.

    Scoped to ``asys-cells-<run_id>``, NOT a bare ``asys-cells`` match: when two sweeps run
    concurrently (e.g. full_sweep_v1 + full_sweep_agent_counts_v1) a name-only count makes
    each driver see the OTHER run's cells as its own, conclude it is at capacity, and never
    dispatch — starving one run indefinitely. Serve/loop jobs are excluded for the same
    reason as before (long-lived fixed overhead); the absolute --qos-limit gate in _drive()
    is what actually protects the shared per-user QOS cap."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j"],
        capture_output=True, text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if line.strip() == f"asys-cells-{run_id}")


def _my_total_count() -> int:
    """ALL my submitted jobs/array-tasks (PD/R), which is what the absolute QOS
    QOSMaxSubmitJobPerUserLimit actually counts — serve + loops + cells + anything else."""
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


def _drive(run_root, chunk_paths, submit_cap: int, poll_s: float, topup_min: int = 80,
           qos_limit: int = 448, chunk_size: int = 400, run_id: str = "") -> None:
    """Submit chunks to keep the cell-task queue TOPPED UP near ``submit_cap``.

    Chunks have no inter-chunk dependency (the array template carries none) and ``run_one``
    skips cells that already have meta.json, so overlapping chunks are safe — extra tasks on
    already-done cells are cheap no-ops, and SLURM's per-chunk ``%throttle`` plus the QOS cap
    bound real concurrency. So instead of waiting for a *full chunk* of headroom (which
    deadlocks: the slow tail of one chunk — 32B/unlimited-thinking — drains for hours while
    the whole fleet starves on those few cells), we submit the next pending chunk whenever
    there is at least ``topup_min`` free slots under the cap.

    TWO gates, both required, because a chunk is one whole array of ``chunk_size`` tasks:
      * cell gate: keep CELL tasks near ``submit_cap`` (cell-only headroom >= topup_min);
      * ABSOLUTE gate: the next chunk's ``chunk_size`` tasks must fit under the real
        ``qos_limit`` ALONGSIDE every other job I have (serve + loops + existing cells).
        Without this, growing the serve fleet (e.g. 6 -> 18 servers) silently pushes the
        baseline up until ``existing_total + chunk_size`` exceeds the QOS limit and EVERY
        submission is rejected — the cell array can no longer refill and throughput collapses
        even though cell-only headroom looks fine. We wait for genuine absolute room."""
    import time

    cells = json.loads((run_root / "cells.json").read_text())
    submitted_log: list[dict] = []
    for ci, lo, hi, path in chunk_paths:
        if _chunk_complete(run_root, lo, hi, cells):
            print(f"[drive] chunk {ci} ({lo}-{hi}) already complete; skipping")
            continue
        # Submit this chunk once there is headroom, RETRYING on the same chunk until it
        # actually lands. Three wait conditions all re-poll (never crash — that would break
        # the self-resubmit chain):
        while True:
            cur = _my_submitted_count(run_id)
            total = _my_total_count()
            headroom = submit_cap - cur
            abs_room = qos_limit - total
            if headroom < topup_min:
                print(f"[drive] topped up: {cur} cell-tasks queued, headroom {headroom} < {topup_min}; sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            if abs_room < chunk_size:
                # The whole chunk won't fit under the absolute QOS limit alongside my serve/
                # loop/cell jobs. Wait for cells to drain rather than spam sbatch rejections.
                print(f"[drive] abs cap: {total}/{qos_limit} jobs, room {abs_room} < chunk {chunk_size} "
                      f"(serve fleet grew?); sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            proc = subprocess.run(["sbatch", "--parsable", str(path)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[drive] sbatch rejected chunk {ci} (rc={proc.returncode}): "
                      f"{proc.stderr.strip()[:200]}; sleep {poll_s:.0f}s and retry")
                time.sleep(poll_s)
                continue
            job_id = proc.stdout.strip().split(";")[0]
            break
        submitted_log.append({"chunk": ci, "lo": lo, "hi": hi, "job_id": job_id})
        (run_root / "chunk_jobs.json").write_text(json.dumps(submitted_log, indent=2))
        print(f"[drive] chunk {ci:3d}: indices {lo:5d}-{hi:5d} -> job {job_id}  (cell-tasks now: {_my_submitted_count(run_id)})")
        # brief settle so squeue reflects the new tasks before the next headroom check
        time.sleep(10)
    print(f"[drive] all {len(chunk_paths)} chunks dispatched; results under {run_root}/cells/")


if __name__ == "__main__":
    main()
