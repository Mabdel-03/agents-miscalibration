"""Launch the ARD36 keepalive + per-lane 15-minute loop jobs (WP4).

Fork of ``slurm/study_loops.py``.  Renders ``loop_keepalive.sbatch.tmpl`` UNCHANGED
(submitted with ``--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1,ASYS_MODEL_CONTRACT=…``, the
legacy fleet path study_v4 ran) and the new ``ard36_loop_driver.sbatch.tmpl``, which runs
one independent 15-minute loop per ARD36 lane over ``ard36_launch_chunked.py`` and, when
asked, ``ard36_topup_lane.py``.  Both self-resubmit with ``afterany`` from inside the
script; ``--requeue`` is never passed (it duplicated the chain).

Lane throttles and per-QOS submit ceilings follow plan §6: ``--submit-cap 380
--qos-limit 420`` on ``mit_preemptable``, 230 for any lane submitting into an ou_bcs_* or
pi_* QOS.  ARD36 lanes are ``contract.LANES``; the frozen study lane tuples are untouched.

Usage::

  python slurm/ard36_loops.py --allow-legacy-driver --run-id ard36 \\
      --spec "ard36-32B:2:pi_manoli:2-00:00:00,ard36-8B:3:pi_manoli:2-00:00:00,…" \\
      --lane "8B=cells_E2a_8B.json:24" --lane "grade=cells_GRADE.json:32:2:4G" --topup
  python slurm/ard36_loops.py --allow-legacy-driver --run-id ard36 --spec … --keepalive-only
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

from agents_scaling.ard.contract import LANES  # noqa: E402
from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402

DEFAULT_MODEL_CONTRACT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling/configs/model_contracts.v1.json"
#: plan §6 / §7: one 15-minute pass per lane for the whole 36-hour window.
LOOP_INTERVAL_S = 900
KEEPALIVE_INTERVAL_S = 180  # ou_bcs_low replicas lived 12-17 min under preemption in study_v4
LOOP_ITERATIONS = 160
#: CPU shape per lane (plan §6): generation cells are 1 CPU HTTP clients; grading is
#: latency-critical with 2 CPUs; the analysis lane compacts and bootstraps.
LANE_DEFAULTS = {
    "8B": {"cpus": 1, "mem": "4G", "throttle": 24},
    "32B": {"cpus": 1, "mem": "4G", "throttle": 15},
    "grade": {"cpus": 2, "mem": "4G", "throttle": 32},
    "analysis": {"cpus": 4, "mem": "32G", "throttle": 4},
    "capture": {"cpus": 8, "mem": "60G", "throttle": 2},
}


def parse_lane(spec: str) -> dict:
    """``LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]]`` → dict (defaults from ``LANE_DEFAULTS``)."""
    if "=" not in spec:
        raise ValueError(f"--lane expects LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]], got {spec!r}")
    lane, rest = spec.split("=", 1)
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
    parts = rest.split(":")
    if not parts[0]:
        raise ValueError(f"--lane {spec!r}: need CELLS_FILE")
    defaults = LANE_DEFAULTS[lane]
    return {
        "lane": lane,
        "cells_file": parts[0],
        "throttle": int(parts[1]) if len(parts) > 1 and parts[1] else defaults["throttle"],
        "cpus": int(parts[2]) if len(parts) > 2 else defaults["cpus"],
        "mem": parts[3] if len(parts) > 3 else defaults["mem"],
    }


def lane_command(args, lane: dict) -> str:
    """One lane's 15-minute loop body: the chunk driver, then (optionally) the top-up tool."""
    driver = (
        "python -u slurm/ard36_launch_chunked.py --allow-legacy-admission "
        f"--run-id \"{args.run_id}\" --server-run-id \"{args.server_run_id or args.run_id}\" "
        f"--cells-file \"{lane['cells_file']}\" --lane {lane['lane']} "
        f"--chunk-size {args.chunk_size} --throttle {lane['throttle']} "
        f"--submit-cap {args.submit_cap} --qos-limit {args.qos_limit} "
        f"--cell-partition \"{args.cell_partition}\" --cell-time \"{args.cell_time}\" "
        f"--cell-mem \"{lane['mem']}\" --cpus {lane['cpus']} --poll-s {args.poll_s}"
    )
    body = driver
    if args.topup:
        body += (
            " ; python -u slurm/ard36_topup_lane.py "
            f"--run-id \"{args.run_id}\" --cells-file \"{lane['cells_file']}\" "
            f"--lane {lane['lane']} --throttle {lane['throttle']} --qos-limit {args.qos_limit}"
        )
    return f"lane_loop {lane['lane']} '{body}' &"


def render(tmpl_name: str, repl: dict[str, str]) -> tuple[Path, str]:
    text = (REPO / "slurm" / tmpl_name).read_text()
    for key, value in repl.items():
        text = text.replace("{" + key + "}", value)
    sbatch_path = REPO / "slurm" / f"{tmpl_name.replace('.tmpl', '')}.{repl['RUN_ID']}"
    return sbatch_path, text


def _render_submit(tmpl_name: str, repl: dict[str, str], export: str, dry_run: bool) -> str:
    sbatch_path, text = render(tmpl_name, repl)
    sbatch_path.write_text(text)
    if dry_run:
        return f"dry-run:{sbatch_path}"
    return subprocess.run(["sbatch", "--parsable", export, str(sbatch_path)],
                          capture_output=True, text=True, check=True).stdout.strip().split(";")[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--server-run-id", default=None)
    ap.add_argument("--spec", required=True, help="keepalive replica spec (profile:count:partition:time,...)")
    ap.add_argument("--lane", action="append", default=[], help="LANE=CELLS_FILE:THROTTLE[:CPUS[:MEM]] (repeatable)")
    ap.add_argument("--keepalive-only", action="store_true")
    ap.add_argument("--topup", action="store_true", help="also run ard36_topup_lane.py on every pass")
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--submit-cap", type=int, default=380)
    ap.add_argument("--qos-limit", type=int, default=420)
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="1-00:00:00")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--partition", default="mit_preemptable,mit_normal")
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--loop-interval-s", type=int, default=LOOP_INTERVAL_S)
    ap.add_argument("--keepalive-interval-s", type=int, default=KEEPALIVE_INTERVAL_S)
    ap.add_argument("--loop-iterations", type=int, default=LOOP_ITERATIONS)
    ap.add_argument("--model-contract", default=os.environ.get("ASYS_MODEL_CONTRACT", DEFAULT_MODEL_CONTRACT))
    ap.add_argument("--dry-run", action="store_true", help="render the sbatch files; submit nothing")
    ap.add_argument("--allow-legacy-driver", action="store_true")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    args = ap.parse_args(argv)

    if not args.allow_legacy_driver and not args.dry_run:
        ap.error("pass --allow-legacy-driver (the retired per-run control loops are re-enabled for ARD36)")
    if not args.keepalive_only and not args.lane:
        ap.error("give at least one --lane LANE=CELLS_FILE:THROTTLE or --keepalive-only")

    run_root = Path(args.results_root) / args.run_id
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    export = f"--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1,ASYS_MODEL_CONTRACT={args.model_contract}"
    common = {"REPO": str(REPO), "RUN_ID": args.run_id, "PARTITION": args.partition,
              "TIME": args.time, "LOG_DIR": str(log_dir)}

    keepalive_job = _render_submit("ard36_keepalive.sbatch.tmpl",
                                   dict(common, SPEC=args.spec,
                                        KEEPALIVE_INTERVAL_S=str(args.keepalive_interval_s)),
                                   export, args.dry_run)
    print(f"[ard36-loops] keepalive -> {keepalive_job}")
    if args.keepalive_only:
        return 0
    lanes = [parse_lane(spec) for spec in args.lane]
    if len({lane["lane"] for lane in lanes}) != len(lanes):
        ap.error("each lane at most once")
    commands = "\n".join(lane_command(args, lane) for lane in lanes)
    driver_job = _render_submit(
        "ard36_loop_driver.sbatch.tmpl",
        dict(common, LANE_COMMANDS=commands, LOOP_INTERVAL_S=str(args.loop_interval_s),
             LOOP_ITERATIONS=str(args.loop_iterations)),
        export, args.dry_run)
    print(f"[ard36-loops] driver ({', '.join(lane['lane'] for lane in lanes)}) -> {driver_job}")
    print("[ard36-loops] both maintained by in-script afterany self-resubmit (singleton chain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
