"""ARD36 per-lane chunk admission (WP4; fork of ``slurm/study_launch_chunked.py``).

Carries every study_v4 operational fix verbatim — the two admission gates plus top-up,
sparse-array re-submission of only the cells still lacking ``meta.json``, the in-flight
check scoped to *this manifest* (``chunk_jobs_<lane>.json`` × ``squeue``), and the merge of
the on-disk chunk log instead of an overwrite — and adds the three ARD36 differences:

* lanes are ``contract.LANES`` (``8B``, ``32B``, ``grade``, ``analysis``, ``capture``);
  the frozen study lane tuples in ``study/cells.py``, ``slurm/study_loops.py`` and
  ``slurm/study_launch_chunked.py`` are left untouched;
* the worker is ``python -m agents_scaling.ard.run_one`` through
  ``slurm/ard36_cell_array.sbatch.tmpl``;
* ``EXIT_DEPS_UNMET`` (5) is a *waiting* cell, not a failed one: the array task exits 0, the
  cell keeps ``DEPS_UNMET.json`` and no ``meta.json``, and this driver re-dispatches it on a
  later pass while printing what it is waiting for, so a controller freeze that lands late
  costs one loop interval instead of a manual repair.

Per-lane QOS limits (plan §6): ``--submit-cap 380 --qos-limit 420`` on ``mit_preemptable``;
any lane submitting into an ou_bcs_* / pi_* QOS gets its own limit of 230, because the
flag models one global cap this cluster does not have.

Usage::

  python slurm/ard36_launch_chunked.py --allow-legacy-admission --run-id ard36 \
      --cells-file cells_E2a_8B.json --lane 8B --chunk-size 60 --throttle 24 \
      --cell-partition mit_preemptable --cell-time 1-00:00:00 --cell-mem 4G --cpus 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.ard.contract import LANES  # noqa: E402
from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402

ARRAY_TMPL = REPO / "slurm" / "ard36_cell_array.sbatch.tmpl"
DEPS_UNMET_FILE = "DEPS_UNMET.json"
#: plan §6: the submit cap the study_v4 run settled on, and the per-QOS ceilings.
DEFAULT_SUBMIT_CAP = 380
QOS_LIMITS = {"mit_preemptable": 420, "ou_bcs_normal": 230, "ou_bcs_low": 230,
              "pi_manoli": 230, "pi_tpoggio": 230}


def job_name(run_id: str, lane: str) -> str:
    return f"asys-ard36-{run_id}-{lane}"


def qos_limit_for(partition: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    first = str(partition).split(",")[0]
    return QOS_LIMITS.get(first, 230)


def _render(run_id, server_run_id, lane, cells_file, lo, hi, throttle, partition, time_limit,
            log_dir, mem: str = "4G", cpus: int = 1) -> str:
    """Render one chunk array covering global indices [lo, hi] (inclusive)."""
    text = ARRAY_TMPL.read_text()
    text = text.replace("--array=0-{LAST_INDEX}%{THROTTLE}", "--array={LO}-{HI}%{THROTTLE}")
    repl = {
        "RUN_ID": run_id, "SERVER_RUN_ID": server_run_id, "LANE": lane,
        "CELLS_FILE": str(cells_file), "LO": str(lo), "HI": str(hi), "THROTTLE": str(throttle),
        "PARTITION": partition, "MEM": mem, "TIME": time_limit, "CPUS": str(cpus),
        "LOG_DIR": log_dir, "REPO": str(REPO),
    }
    for key, value in repl.items():
        text = text.replace("{" + key + "}", value)
    if any(f"{{{key}}}" in text for key in ("LAST_INDEX", "LO", "HI", "MEM", "TIME", "CPUS")):
        raise RuntimeError("unrendered placeholder in ard36_cell_array.sbatch.tmpl")
    return text


def load_cells(cells_file: Path) -> list[dict]:
    """Load and *verify* the frozen manifest (the ``.sha256`` sidecar is mandatory in ARD36)."""
    data = json.loads(cells_file.read_text(encoding="utf-8"))
    cells = data["cells"] if isinstance(data, dict) else data
    if not isinstance(cells, list) or not all(isinstance(c, dict) and "cell_id" in c for c in cells):
        raise RuntimeError(f"{cells_file}: expected an ARD36 cells manifest with a 'cells' list")
    sidecar = cells_file.with_name(cells_file.name + ".sha256")
    if sidecar.exists():
        import hashlib

        expected = sidecar.read_text().split()[0]
        actual = hashlib.sha256(cells_file.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError(f"{cells_file}: sha256 {actual} != frozen {expected}")
    return cells


def _my_submitted_count(run_id: str, lane: str) -> int:
    """This run's + lane's cell array-tasks currently submitted (PD/R), by EXACT job name."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j"],
                         capture_output=True, text=True).stdout
    name = job_name(run_id, lane)
    return sum(1 for line in out.splitlines() if line.strip() == name)


def _my_total_count() -> int:
    """ALL my submitted jobs/array-tasks (PD/R): what QOSMaxSubmitJobPerUserLimit counts."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"],
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def _chunk_complete(run_root: Path, lo: int, hi: int, cells: list) -> bool:
    """Terminal = ``meta.json`` (finished) or ``SUSPENDED.json`` (a defect, never re-queued).

    ``DEPS_UNMET.json`` is explicitly *not* terminal: the cell is waiting for its controller
    or state register and must be re-dispatched later.
    """
    cdir = run_root / "cells"
    suspended: list[str] = []
    waiting: list[str] = []
    complete = True
    for idx in range(lo, hi + 1):
        cid = cells[idx]["cell_id"]
        if (cdir / cid / "meta.json").exists():
            continue
        if (cdir / cid / "SUSPENDED.json").exists():
            suspended.append(cid)
            continue
        if (cdir / cid / DEPS_UNMET_FILE).exists():
            waiting.append(cid)
        complete = False
    if suspended:
        print(f"[drive] SUSPENDED cells in {lo}-{hi} (terminal, not re-queued; delete "
              f"SUSPENDED.json after the fix): " + ", ".join(suspended), file=sys.stderr, flush=True)
    if waiting:
        print(f"[drive] waiting on a controller:/state: dependency in {lo}-{hi}: "
              + ", ".join(waiting[:8]) + (" …" if len(waiting) > 8 else ""), flush=True)
    return complete


def _chunk_job_in_flight(submitted_log: list, lo: int, hi: int, cells_file: str = "") -> str | None:
    """Job id of an earlier submission of exactly this [lo,hi] chunk of THIS manifest."""
    for entry in submitted_log:
        if cells_file and entry.get("cells_file") and entry.get("cells_file") != cells_file:
            continue
        if entry.get("lo") == lo and entry.get("hi") == hi and entry.get("job_id"):
            jid = str(entry["job_id"])
            out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%i"],
                                 capture_output=True, text=True).stdout
            if out.strip():
                return jid
    return None


def _drive(run_root: Path, chunk_paths, submit_cap: int, poll_s: float, topup_min: int = 40,
           qos_limit: int = 420, run_id: str = "", lane: str = "", cells: list | None = None,
           cells_file: str = "") -> None:
    """Keep this lane's cell-task queue topped up near ``submit_cap`` (two gates + top-up)."""
    cells = cells or []
    log_path = run_root / f"chunk_jobs_{lane}.json"
    try:
        submitted_log: list[dict] = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:  # noqa: BLE001 — a corrupt log must not block dispatch
        submitted_log = []
    for ci, lo, hi, path in chunk_paths:
        if _chunk_complete(run_root, lo, hi, cells):
            print(f"[drive:{lane}] chunk {ci} ({lo}-{hi}) already complete; skipping")
            continue
        live_job = _chunk_job_in_flight(submitted_log, lo, hi, cells_file=cells_file)
        if live_job is not None:
            print(f"[drive:{lane}] chunk {ci} ({lo}-{hi}) in flight as job {live_job}; skipping")
            continue
        todo = [i for i in range(lo, hi + 1)
                if not (run_root / "cells" / cells[i]["cell_id"] / "meta.json").exists()]
        need = hi - lo + 1
        submit_path = path
        if todo and len(todo) < need:
            text = Path(path).read_text()
            match = re.search(r"^#SBATCH --array=.*?%(\d+)$", text, flags=re.M)
            thr = match.group(1) if match else "40"
            sparse = re.sub(r"^#SBATCH --array=.*$",
                            f"#SBATCH --array={','.join(map(str, todo))}%{thr}", text, flags=re.M)
            submit_path = Path(str(path).replace(".sbatch", ".resume.sbatch"))
            submit_path.write_text(sparse)
            need = len(todo)
            print(f"[drive:{lane}] chunk {ci} ({lo}-{hi}): {need} unfinished cells -> sparse resume array")
        while True:
            cur = _my_submitted_count(run_id, lane)
            total = _my_total_count()
            headroom = submit_cap - cur
            abs_room = qos_limit - total
            if headroom < min(topup_min, need):
                print(f"[drive:{lane}] topped up: {cur} cell-tasks queued, headroom {headroom} "
                      f"< {min(topup_min, need)}; sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            if abs_room < need:
                print(f"[drive:{lane}] abs cap: {total}/{qos_limit} jobs, room {abs_room} < "
                      f"needed {need}; sleep {poll_s:.0f}s")
                time.sleep(poll_s)
                continue
            proc = subprocess.run(["sbatch", "--parsable", str(submit_path)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[drive:{lane}] sbatch rejected chunk {ci} (rc={proc.returncode}): "
                      f"{proc.stderr.strip()[:200]}; sleep {poll_s:.0f}s and retry")
                time.sleep(poll_s)
                continue
            job_id = proc.stdout.strip().split(";")[0]
            break
        entry = {"chunk": ci, "lo": lo, "hi": hi, "job_id": job_id, "lane": lane,
                 "cells_file": cells_file}
        try:  # merge with the on-disk log (another driver may have written since we loaded it)
            on_disk = json.loads(log_path.read_text()) if log_path.exists() else []
        except (OSError, ValueError):
            on_disk = list(submitted_log)
        submitted_log[:] = on_disk + [entry]
        log_path.write_text(json.dumps(submitted_log, indent=2))
        print(f"[drive:{lane}] chunk {ci:3d}: indices {lo:5d}-{hi:5d} -> job {job_id} "
              f"(lane tasks now: {_my_submitted_count(run_id, lane)})")
        time.sleep(10)
    print(f"[drive:{lane}] all {len(chunk_paths)} chunks dispatched; results under {run_root}/cells/")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells-file", required=True, help="ARD36 cells manifest (absolute or under the run root)")
    ap.add_argument("--lane", required=True, choices=list(LANES))
    ap.add_argument("--server-run-id", default=None)
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--throttle", type=int, default=24, help="max concurrent tasks per chunk")
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="1-00:00:00")
    ap.add_argument("--cell-mem", default="4G")
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="render sbatches, don't submit")
    ap.add_argument("--submit-cap", type=int, default=DEFAULT_SUBMIT_CAP)
    ap.add_argument("--qos-limit", type=int, default=None,
                    help="absolute submit ceiling; default 420 on mit_preemptable, 230 elsewhere")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--allow-legacy-admission", action="store_true")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    args = ap.parse_args(argv)

    if not args.dry_run and not args.no_drive and not args.allow_legacy_admission:
        ap.error("pass --allow-legacy-admission (per-lane drivers share one QOS cap)")

    run_root = Path(args.results_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cells_file = Path(args.cells_file)
    if not cells_file.is_absolute():
        cells_file = run_root / cells_file
    cells = load_cells(cells_file)
    qos_limit = qos_limit_for(args.cell_partition, args.qos_limit)
    n = len(cells)
    print(f"[ard36-chunked] run_id={args.run_id} lane={args.lane} cells={n} "
          f"chunk_size={args.chunk_size} qos_limit={qos_limit} <- {cells_file}")
    chunks = [(lo, min(lo + args.chunk_size - 1, n - 1)) for lo in range(0, n, args.chunk_size)]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"[ard36-chunked] {len(chunks)} chunks, throttle {args.throttle}, partition "
          f"{args.cell_partition}, job {job_name(args.run_id, args.lane)}")

    chunk_paths = []
    for ci, (lo, hi) in enumerate(chunks):
        text = _render(args.run_id, args.server_run_id or args.run_id, args.lane, cells_file, lo,
                       hi, args.throttle, args.cell_partition, args.cell_time, str(log_dir),
                       mem=args.cell_mem, cpus=args.cpus)
        path = run_root / f"chunk_{args.lane}_{cells_file.stem}_{ci:03d}_{lo}-{hi}.sbatch"
        path.write_text(text)
        chunk_paths.append((ci, lo, hi, path))
        if args.dry_run:
            print(f"  [dry-run] chunk {ci}: indices {lo}-{hi} -> {path}")
    if args.dry_run or args.no_drive:
        return 0
    _drive(run_root, chunk_paths, args.submit_cap, args.poll_s, qos_limit=qos_limit,
           run_id=args.run_id, lane=args.lane, cells=cells, cells_file=cells_file.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
