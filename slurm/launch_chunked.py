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
import subprocess
from pathlib import Path

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
    ap.add_argument("--chunk-size", type=int, default=480, help="cells per array chunk (<=500)")
    ap.add_argument("--throttle", type=int, default=480, help="max concurrent tasks per chunk")
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="2-00:00:00")
    ap.add_argument("--max-chunks", type=int, default=None, help="cap chunks (debug)")
    ap.add_argument("--dry-run", action="store_true", help="render sbatches, don't submit")
    args = ap.parse_args()

    if args.chunk_size > 500:
        ap.error("--chunk-size must be <= 500 (MaxSubmitJobs association limit)")

    cells = load_sweep(args.config)
    run_root = io.run_dir(args.run_id)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Persist the canonical cell list the arrays index into (single source of truth).
    cells_file = run_root / "cells.json"
    cells_file.write_text(json.dumps([c.to_dict() for c in cells], indent=2))
    n = len(cells)
    print(f"[chunked] run_id={args.run_id}  cells={n}  chunk_size={args.chunk_size}  -> {cells_file}")

    chunks = [(lo, min(lo + args.chunk_size - 1, n - 1)) for lo in range(0, n, args.chunk_size)]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"[chunked] {len(chunks)} chunks, throttle {args.throttle}, partition {args.cell_partition}")

    prev_job: str | None = None
    submitted: list[str] = []
    for ci, (lo, hi) in enumerate(chunks):
        text = _render(
            args.run_id, str(cells_file), lo, hi, args.throttle,
            args.cell_partition, args.cell_time, str(log_dir),
        )
        sbatch_path = run_root / f"chunk_{ci:03d}_{lo}-{hi}.sbatch"
        sbatch_path.write_text(text)
        if args.dry_run:
            print(f"  [dry-run] chunk {ci}: indices {lo}-{hi} -> {sbatch_path}")
            continue
        cmd = ["sbatch", "--parsable"]
        if prev_job:
            cmd.append(f"--dependency=afterany:{prev_job}")
        cmd.append(str(sbatch_path))
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        job_id = out.split(";")[0]  # --parsable: "jobid[;cluster]"
        submitted.append(job_id)
        dep = f" (after {prev_job})" if prev_job else ""
        print(f"  chunk {ci:3d}: indices {lo:5d}-{hi:5d} -> job {job_id}{dep}")
        prev_job = job_id

    if not args.dry_run:
        (run_root / "chunk_jobs.json").write_text(json.dumps(submitted, indent=2))
        print(f"[chunked] submitted {len(submitted)} chunk jobs; ids -> {run_root/'chunk_jobs.json'}")


if __name__ == "__main__":
    main()
