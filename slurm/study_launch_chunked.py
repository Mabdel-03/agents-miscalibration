"""study_v4 per-lane chunk admission (WP5; fork of ``slurm/launch_chunked.py``).

Keeps ``_drive`` / ``_my_submitted_count`` / ``_my_total_count`` (the two gates + top-up)
and replaces the sweep config with a pre-built study cells manifest (``--cells-file``,
written by ``python -m agents_scaling.study.cells``).  ``_chunk_complete`` reads
``cells[idx]["cell_id"]`` from that manifest.  Arrays are named
``asys-study-<run_id>-<lane>`` and counted by that exact name (04_critic_corrections P0-7),
so several lanes (32B / 14B / 8B / 4B / eval) can be driven concurrently with different
throttles (P0-6: throttle ≈ 6 cells × 8 in-flight per live 32B endpoint).

Usage:
  python slurm/study_launch_chunked.py --allow-legacy-admission --run-id study_v4 \
      --cells-file cells_1_32B.json --lane 32B --chunk-size 100 --throttle 48 \
      --submit-cap 380 --qos-limit 460 --cell-partition mit_preemptable \
      --cell-time 1-00:00:00 --cell-mem 4G --cpus 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402

ARRAY_TMPL = REPO / "slurm" / "study_cell_array.sbatch.tmpl"
LANES = ("32B", "14B", "8B", "4B", "eval")


def job_name(run_id: str, lane: str) -> str:
    return f"asys-study-{run_id}-{lane}"


def _render(run_id, server_run_id, lane, cells_file, lo, hi, throttle, partition, time_limit, log_dir,
            mem: str = "4G", cpus: int = 1) -> str:
    """Render one chunk array covering global indices [lo, hi] (inclusive)."""
    text = ARRAY_TMPL.read_text()
    text = text.replace("--array=0-{LAST_INDEX}%{THROTTLE}", "--array={LO}-{HI}%{THROTTLE}")
    repl = {
        "RUN_ID": run_id,
        "SERVER_RUN_ID": server_run_id,
        "LANE": lane,
        "CELLS_FILE": str(cells_file),
        "LO": str(lo),
        "HI": str(hi),
        "THROTTLE": str(throttle),
        "PARTITION": partition,
        "MEM": mem,
        "TIME": time_limit,
        "CPUS": str(cpus),
        "LOG_DIR": log_dir,
        "REPO": str(REPO),
    }
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    if "{" in text and "}" in text and any(f"{{{k}}}" in text for k in ("LAST_INDEX", "LO", "HI", "MEM", "TIME", "CPUS")):
        raise RuntimeError("unrendered placeholder in study_cell_array.sbatch.tmpl")
    return text


def load_cells(cells_file: Path) -> list[dict]:
    data = json.loads(cells_file.read_text(encoding="utf-8"))
    cells = data["cells"] if isinstance(data, dict) else data
    if not isinstance(cells, list) or not all(isinstance(c, dict) and "cell_id" in c for c in cells):
        raise RuntimeError(f"{cells_file}: expected a study cells manifest with a 'cells' list")
    sidecar = cells_file.with_name(cells_file.name + ".sha256")
    if sidecar.exists():
        import hashlib

        expected = sidecar.read_text().split()[0]
        actual = hashlib.sha256(cells_file.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError(f"{cells_file}: sha256 {actual} != frozen {expected}")
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="study_v4 per-lane chunk admission (legacy-style driver)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells-file", required=True, help="study cells manifest (absolute or under the run root)")
    ap.add_argument("--lane", required=True, choices=LANES)
    ap.add_argument("--server-run-id", default=None, help="run id holding servers/ (default: --run-id)")
    ap.add_argument("--chunk-size", type=int, default=100, help="cells per array chunk")
    ap.add_argument("--throttle", type=int, default=48, help="max concurrent tasks per chunk (≈ 6 × live endpoints, P0-6)")
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="1-00:00:00")
    ap.add_argument("--cell-mem", default="4G")
    ap.add_argument("--cpus", type=int, default=1, help="CPUs per task (EVAL_BCB lane: 2)")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="render sbatches, don't submit")
    ap.add_argument("--submit-cap", type=int, default=380,
                    help="target number of THIS LANE's cell tasks to keep queued (counted by exact job name)")
    ap.add_argument("--qos-limit", type=int, default=460,
                    help="absolute QOSMaxSubmitJobPerUserLimit across ALL my jobs (serve + loops + every lane)")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--allow-legacy-admission", action="store_true",
                    help="explicit override retained from launch_chunked.py (per-lane drivers share the QOS cap)")
    args = ap.parse_args()

    if not args.dry_run and not args.no_drive and not args.allow_legacy_admission:
        ap.error("pass --allow-legacy-admission (per-lane drivers share one QOS cap; --qos-limit must be honest)")

    results_root = Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    run_root = results_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cells_file = Path(args.cells_file)
    if not cells_file.is_absolute():
        cells_file = run_root / cells_file
    cells = load_cells(cells_file)
    n = len(cells)
    print(f"[study-chunked] run_id={args.run_id} lane={args.lane} cells={n} chunk_size={args.chunk_size} <- {cells_file}")
    chunks = [(lo, min(lo + args.chunk_size - 1, n - 1)) for lo in range(0, n, args.chunk_size)]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"[study-chunked] {len(chunks)} chunks, throttle {args.throttle}, partition {args.cell_partition}, job {job_name(args.run_id, args.lane)}")

    chunk_paths = []
    for ci, (lo, hi) in enumerate(chunks):
        text = _render(args.run_id, args.server_run_id or args.run_id, args.lane, cells_file, lo, hi, args.throttle,
                       args.cell_partition, args.cell_time, str(log_dir), mem=args.cell_mem, cpus=args.cpus)
        p = run_root / f"chunk_{args.lane}_{cells_file.stem}_{ci:03d}_{lo}-{hi}.sbatch"
        p.write_text(text)
        chunk_paths.append((ci, lo, hi, p))
        if args.dry_run:
            print(f"  [dry-run] chunk {ci}: indices {lo}-{hi} -> {p}")
    if args.dry_run or args.no_drive:
        return

    _drive.cells_file = Path(args.cells_file).name  # manifest identity for the in-flight check (study-v4)
    _drive(run_root, chunk_paths, args.submit_cap, args.poll_s, qos_limit=args.qos_limit,
           chunk_size=args.chunk_size, run_id=args.run_id, lane=args.lane, cells=cells)


def _my_submitted_count(run_id: str, lane: str) -> int:
    """This run's + lane's cell array-tasks currently submitted (PD/R), by EXACT job name
    (``squeue -r`` expands array tasks to one line each) — P0-7."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j"],
        capture_output=True, text=True,
    ).stdout
    name = job_name(run_id, lane)
    return sum(1 for line in out.splitlines() if line.strip() == name)


def _my_total_count() -> int:
    """ALL my submitted jobs/array-tasks (PD/R): what QOSMaxSubmitJobPerUserLimit counts."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"],
        capture_output=True, text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def _chunk_complete(run_root, lo: int, hi: int, cells: list) -> bool:
    """A chunk is done when every cell index in [lo,hi] is terminal: ``meta.json`` (finished)
    or ``SUSPENDED.json`` (a harness defect, §10.4 — never re-queued; ``run_one`` exits 4 at
    once for it).  Suspended cells are printed loudly; remove the file after the fix to rerun.
    """
    cdir = run_root / "cells"
    suspended: list[str] = []
    complete = True
    for idx in range(lo, hi + 1):
        cid = cells[idx]["cell_id"]
        if (cdir / cid / "meta.json").exists():
            continue
        if (cdir / cid / "SUSPENDED.json").exists():
            suspended.append(cid)
            continue
        complete = False
    if suspended:
        print(f"[drive] SUSPENDED cells in indices {lo}-{hi} (terminal, not re-queued; delete SUSPENDED.json after the fix): "
              + ", ".join(suspended), file=sys.stderr, flush=True)
    return complete



def _chunk_job_in_flight(submitted_log: list, lo: int, hi: int, cells_file: str = "") -> str | None:
    """Job id of an earlier submission of exactly this [lo,hi] chunk that Slurm still lists
    (PD/R/CG in any state), else None.  Uses ``squeue -h -j <id>``; a job absent from squeue
    is terminal and the chunk may be re-submitted (the runner skips finished cells)."""
    for entry in submitted_log:
        if cells_file and entry.get("cells_file") and entry.get("cells_file") != cells_file:
            continue  # a different manifest's chunk with the same indices (study-v4)
        if entry.get("lo") == lo and entry.get("hi") == hi and entry.get("job_id"):
            jid = str(entry["job_id"])
            out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%i"], capture_output=True, text=True).stdout
            if out.strip():
                return jid
    return None


def _drive(run_root, chunk_paths, submit_cap: int, poll_s: float, topup_min: int = 40,
           qos_limit: int = 460, chunk_size: int = 100, run_id: str = "", lane: str = "", cells: list | None = None) -> None:
    """Submit chunks to keep this lane's cell-task queue TOPPED UP near ``submit_cap``.

    Verbatim logic of ``launch_chunked._drive`` (two gates + top-up): chunks carry no
    inter-chunk dependency and ``run_one`` skips cells with ``meta.json``, so overlapping
    chunks are cheap no-ops.  Gate 1: lane headroom ≥ ``topup_min``; gate 2: the whole chunk
    fits under the absolute ``qos_limit`` alongside every other job I have.
    """
    cells = cells if cells is not None else load_cells(run_root / "cells.json")
    log_path = run_root / f"chunk_jobs_{lane}.json"
    try:
        submitted_log: list[dict] = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:  # noqa: BLE001 — a corrupt log must not block dispatch
        submitted_log = []
    for ci, lo, hi, path in chunk_paths:
        if _chunk_complete(run_root, lo, hi, cells):
            print(f"[drive:{lane}] chunk {ci} ({lo}-{hi}) already complete; skipping")
            continue
        live_job = _chunk_job_in_flight(submitted_log, lo, hi, cells_file=str(getattr(_drive, "cells_file", "")))
        if live_job is not None:
            # study-v4: a previous invocation (or a one-shot dispatch) already submitted this
            # exact chunk and its array is still queued/running — never double-submit it
            # (run_one also holds a per-cell lock, but a duplicate array wastes GPU work).
            print(f"[drive:{lane}] chunk {ci} ({lo}-{hi}) in flight as job {live_job}; skipping")
            continue
        while True:
            cur = _my_submitted_count(run_id, lane)
            total = _my_total_count()
            headroom = submit_cap - cur
            abs_room = qos_limit - total
            if headroom < topup_min:
                print(f"[drive:{lane}] topped up: {cur} cell-tasks queued, headroom {headroom} < {topup_min}; sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            if abs_room < chunk_size:
                print(f"[drive:{lane}] abs cap: {total}/{qos_limit} jobs, room {abs_room} < chunk {chunk_size}; sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            proc = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[drive:{lane}] sbatch rejected chunk {ci} (rc={proc.returncode}): {proc.stderr.strip()[:200]}; sleep {poll_s:.0f}s and retry")
                time.sleep(poll_s)
                continue
            job_id = proc.stdout.strip().split(";")[0]
            break
        submitted_log.append({"chunk": ci, "lo": lo, "hi": hi, "job_id": job_id, "lane": lane, "cells_file": str(getattr(_drive, "cells_file", ""))})
        log_path.write_text(json.dumps(submitted_log, indent=2))
        print(f"[drive:{lane}] chunk {ci:3d}: indices {lo:5d}-{hi:5d} -> job {job_id}  (lane tasks now: {_my_submitted_count(run_id, lane)})")
        time.sleep(10)
    print(f"[drive:{lane}] all {len(chunk_paths)} chunks dispatched; results under {run_root}/cells/")


if __name__ == "__main__":
    main()
