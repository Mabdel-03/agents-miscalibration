"""Launch the study_v4 keepalive + per-lane chunk-driver loop jobs (WP5; fork of
``slurm/launch_loops.py``).

Renders ``loop_keepalive.sbatch.tmpl`` UNCHANGED (submitted with
``--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1,ASYS_MODEL_CONTRACT=…``) and the new
``study_loop_driver.sbatch.tmpl`` which execs ``study_launch_chunked.py`` once per lane.
Both self-resubmit with ``afterany`` from inside the script; ``--requeue`` is never passed
(it duplicated the chain, see ``launch_loops.py``).  Never submits anything with
``--dry-run``.

Usage:
  python slurm/study_loops.py --allow-legacy-driver --run-id study_v4 \
      --spec "32B-long:3:pi_manoli:7-00:00:00,…" \
      --lane 32B=cells_1_32B.json:48 --lane 14B=cells_2_14B.json:40 --lane eval=cells_1-eval_eval.json:20:2:8G
  python slurm/study_loops.py --allow-legacy-driver --run-id study_v4 --spec … --keepalive-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402

LANES = ("32B", "14B", "8B", "4B", "eval")
DEFAULT_MODEL_CONTRACT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling/configs/model_contracts.v1.json"


def parse_lane(spec: str) -> dict:
    """``LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]]`` → dict (eval lane defaults to 2 CPUs / 8G)."""
    if "=" not in spec:
        raise ValueError(f"--lane expects LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]], got {spec!r}")
    lane, rest = spec.split("=", 1)
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
    parts = rest.split(":")
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"--lane {spec!r}: need CELLS_FILE:THROTTLE")
    cpus = int(parts[2]) if len(parts) > 2 else (2 if lane == "eval" else 1)
    mem = parts[3] if len(parts) > 3 else ("8G" if lane == "eval" else "4G")
    return {"lane": lane, "cells_file": parts[0], "throttle": int(parts[1]), "cpus": cpus, "mem": mem}


def lane_command(args, lane: dict) -> str:
    return (
        "python -u slurm/study_launch_chunked.py --allow-legacy-admission "
        f"--run-id \"{args.run_id}\" --server-run-id \"{args.server_run_id or args.run_id}\" "
        f"--cells-file \"{lane['cells_file']}\" --lane {lane['lane']} "
        f"--chunk-size {args.chunk_size} --throttle {lane['throttle']} --submit-cap {args.submit_cap} "
        f"--qos-limit {args.qos_limit} --cell-partition \"{args.cell_partition}\" --cell-time \"{args.cell_time}\" "
        f"--cell-mem \"{lane['mem']}\" --cpus {lane['cpus']} --poll-s {args.poll_s} &"
    )


def render(tmpl_name: str, repl: dict[str, str]) -> tuple[Path, str]:
    text = (REPO / "slurm" / tmpl_name).read_text()
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    sbatch_path = REPO / "slurm" / f"{tmpl_name.replace('.tmpl', '')}.{repl['RUN_ID']}"
    return sbatch_path, text


def _render_submit(tmpl_name: str, repl: dict[str, str], export: str, dry_run: bool) -> str:
    sbatch_path, text = render(tmpl_name, repl)
    sbatch_path.write_text(text)
    if dry_run:
        return f"dry-run:{sbatch_path}"
    out = subprocess.run(
        ["sbatch", "--parsable", export, str(sbatch_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(";")[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="study_v4 keepalive + per-lane driver loop launcher")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--server-run-id", default=None)
    ap.add_argument("--spec", required=True, help="keepalive replica spec (profile:count:partition:time,...)")
    ap.add_argument("--lane", action="append", default=[], help="LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]] (repeatable)")
    ap.add_argument("--keepalive-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--submit-cap", type=int, default=380)
    ap.add_argument("--qos-limit", type=int, default=460)
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="1-00:00:00")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--partition", default="mit_preemptable,mit_normal,mit_normal_gpu")
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--model-contract", default=os.environ.get("ASYS_MODEL_CONTRACT", DEFAULT_MODEL_CONTRACT))
    ap.add_argument("--dry-run", action="store_true", help="render the sbatch files; submit nothing")
    ap.add_argument("--allow-legacy-driver", action="store_true", help="explicit override retained from launch_loops.py")
    args = ap.parse_args()

    if not args.allow_legacy_driver and not args.dry_run:
        ap.error("pass --allow-legacy-driver (the retired per-run control loops are re-enabled for study_v4 only)")
    if not args.keepalive_only and not args.lane:
        ap.error("give at least one --lane LANE=CELLS_FILE:THROTTLE or --keepalive-only")

    results_root = os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)
    run_root = Path(results_root) / args.run_id
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    export = f"--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1,ASYS_MODEL_CONTRACT={args.model_contract}"
    common = {"REPO": str(REPO), "RUN_ID": args.run_id, "PARTITION": args.partition, "TIME": args.time, "LOG_DIR": str(log_dir)}

    keepalive_job = _render_submit("loop_keepalive.sbatch.tmpl", dict(common, SPEC=args.spec), export, args.dry_run)
    print(f"[study-loops] keepalive -> {keepalive_job}")
    if args.keepalive_only:
        return
    lanes = [parse_lane(s) for s in args.lane]
    if len({l["lane"] for l in lanes}) != len(lanes):
        ap.error("each lane at most once")
    commands = "\n".join(lane_command(args, lane) for lane in lanes)
    driver_job = _render_submit("study_loop_driver.sbatch.tmpl", dict(common, LANE_COMMANDS=commands), export, args.dry_run)
    print(f"[study-loops] driver ({', '.join(l['lane'] for l in lanes)}) -> {driver_job}")
    print("[study-loops] both maintained by in-script afterany self-resubmit (singleton chain)")


if __name__ == "__main__":
    main()
