"""Orchestrate a full sweep on SLURM.

Steps:
  1. Expand the sweep YAML into canonical cells; write ``cells.json``.
  2. Launch one long-lived vLLM serving job per distinct model size (servers register
     their endpoint to the run's registry once healthy).
  3. Submit the cell array job (each task discovers its server via the registry and waits
     until it is ready, so we don't need an explicit SLURM dependency on the servers).

``--serve-only`` launches just the servers (useful for the smoke test / pilot). Pass
``--models`` to restrict which sizes to serve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.experiment import io
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest
from agents_scaling.experiment.sweep import load_sweep, models_in_sweep
from agents_scaling.serving.launch_server import submit as submit_server

ARRAY_TMPL = REPO / "slurm" / "run_cell_array.sbatch.tmpl"


def _render_array(
    run_id: str,
    cells_file: str,
    n_cells: int,
    partition: str,
    time_limit: str,
    throttle: int,
    log_dir: str,
    mem: str,
) -> str:
    text = ARRAY_TMPL.read_text()
    repl = {
        "RUN_ID": run_id,
        "CELLS_FILE": cells_file,
        "LAST_INDEX": str(n_cells - 1),
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
    ap = argparse.ArgumentParser(description="Launch a sweep (servers + cell array) on SLURM.")
    ap.add_argument("--config", default=str(REPO / "configs" / "pilot.yaml"))
    ap.add_argument("--run-id", default=f"run_{int(time.time())}")
    ap.add_argument("--serve-only", action="store_true", help="launch only the vLLM servers")
    ap.add_argument("--models", nargs="*", help="restrict served model sizes (default: all in sweep)")
    ap.add_argument("--serve-partition", default="pi_tpoggio")
    ap.add_argument("--serve-gpu-type", default="a100")
    ap.add_argument("--serve-time", default="7-00:00:00")  # servers must outlive the run
    # Cells are CPU-only HTTP clients -> a long-limit non-GPU partition. Preemption is safe
    # because the runner resumes (skips completed cells / answered qids on re-submit).
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="2-00:00:00")
    ap.add_argument("--cell-mem", default="2G")
    ap.add_argument("--throttle", type=int, default=16, help="max concurrent array tasks")
    ap.add_argument(
        "--allow-legacy-array",
        action="store_true",
        help="explicit emergency-only override; production cells use the global dispatcher",
    )
    args = ap.parse_args()

    if not args.serve_only and not args.allow_legacy_array:
        ap.error(
            "direct sweep arrays are retired: initialize/freeze the manifest and use "
            "slurm/launch_dispatcher.py; --serve-only remains supported"
        )

    cells = load_sweep(args.config)
    run_root = io.run_dir(args.run_id)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cells_file = run_root / "cells.json"
    if cells_file.exists():
        snapshot = load_manifest(run_root)
        cells = list(snapshot.cells)
    else:
        io.atomic_write_text(
            cells_file, json.dumps([c.to_dict() for c in cells], indent=2) + "\n"
        )
        freeze_manifest(run_root)
    print(f"[sweep] run_id={args.run_id}  cells={len(cells)}  -> {cells_file}")

    sizes = args.models or models_in_sweep(cells)
    print(f"[sweep] serving model sizes: {sizes}")
    for size in sizes:
        submit_server(size, str(run_root), args.serve_partition, args.serve_gpu_type, args.serve_time)

    if args.serve_only:
        print("[sweep] --serve-only: skipping cell array. Run the pilot/cells once servers are up.")
        return

    array_text = _render_array(
        args.run_id, str(cells_file), len(cells), args.cell_partition, args.cell_time,
        args.throttle, str(log_dir), args.cell_mem,
    )
    array_path = run_root / "run_cells.sbatch"
    io.atomic_write_text(array_path, array_text)
    import subprocess

    out = subprocess.run(["sbatch", str(array_path)], capture_output=True, text=True, check=True).stdout.strip()
    print(f"[sweep] cell array submitted: {out}  (sbatch: {array_path})")


if __name__ == "__main__":
    main()
