"""Launch the legacy chunk-driver + keepalive pair (disabled by default).

Login-node nohup processes (even with setsid+disown) repeatedly get reaped on this
cluster when the user's session ends. The fix: run the two long-lived loops as actual
SLURM jobs on compute nodes. They're CPU-only, tiny.

Durability is achieved by the in-script `afterany` self-resubmit (see loop_*.sbatch.tmpl):
at startup each job queues exactly ONE successor that runs when the current job ends for
ANY reason (walltime, preemption, cancel, success). This produces a clean singleton chain.

NOTE: do NOT also pass `--requeue` here. Both mechanisms together caused the chain to
duplicate every walltime cycle (observed: 1 -> 3 -> 7 concurrent loops over a few days).

Usage:
  python slurm/launch_loops.py --run-id full_sweep_v1 \
      --spec 0.6B:1:pi_tpoggio:7-00:00:00,1.7B:2:ou_bcs_low:1-00:00:00,...
  python slurm/launch_loops.py --run-id full_sweep_agent_counts_v1 \
      --config configs/full_sweep_agent_counts.yaml \
      --chunk-size 120 --throttle 96 --submit-cap 400 --qos-limit 448 --cell-mem 8G \
      --spec 0.6B:1:pi_tpoggio:7-00:00:00,...
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

from agents_scaling.config import DEFAULT_RESULTS_ROOT


def _render_submit(tmpl_name: str, repl: dict[str, str]) -> str:
    """Render an sbatch template into a STABLE-NAMED file under slurm/ (so the script's
    self-resubmit can reference itself), then submit it. Returns the job id."""
    text = (REPO / "slurm" / tmpl_name).read_text()
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    # Stable name keyed by run-id so the self-resubmit always finds itself.
    sbatch_path = REPO / "slurm" / f"{tmpl_name.replace('.tmpl', '')}.{repl['RUN_ID']}"
    sbatch_path.write_text(text)
    # NO --requeue: the in-script afterany self-resubmit is the ONLY durability mechanism
    # (see module docstring). Combining the two caused chain duplication.
    out = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1",
            str(sbatch_path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(";")[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LEGACY split driver+keepalive launcher; global dispatcher is production"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--spec", required=True,
                    help="replica spec for keepalive (size:count:partition:time,...)")
    ap.add_argument("--config", default="configs/full_sweep.yaml",
                    help="sweep config passed to launch_chunked.py")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--throttle", type=int, default=240)
    ap.add_argument("--submit-cap", type=int, default=440)
    ap.add_argument("--qos-limit", type=int, default=448)
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="2-00:00:00")
    ap.add_argument("--cell-mem", default="2G")
    ap.add_argument("--partition", default="mit_preemptable,mit_normal,mit_normal_gpu",
                    help="partition list for the loop jobs (CPU-only, long limit). "
                         "Comma-separated for SLURM partition fallback so a single-partition "
                         "outage (maintenance, QOS hold) doesn't paralyze the control plane.")
    ap.add_argument("--time", default="12:00:00",
                    help="walltime per loop job. 12h is the cap on mit_normal; the in-script "
                         "afterany successor makes the cycle count immaterial.")
    ap.add_argument(
        "--allow-legacy-driver",
        action="store_true",
        help="explicit emergency-only override for the retired per-run admission driver",
    )
    args = ap.parse_args()

    if not args.allow_legacy_driver:
        ap.error(
            "legacy per-run drivers are retired because they can recreate split-brain "
            "admission; launch slurm/launch_dispatcher.py and run keepalive separately"
        )

    results_root = os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)
    run_root = Path(results_root) / args.run_id
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "REPO": str(REPO),
        "RUN_ID": args.run_id,
        "PARTITION": args.partition,
        "TIME": args.time,
        "LOG_DIR": str(log_dir),
        "CONFIG": args.config,
        "CHUNK_SIZE": str(args.chunk_size),
        "THROTTLE": str(args.throttle),
        "SUBMIT_CAP": str(args.submit_cap),
        "QOS_LIMIT": str(args.qos_limit),
        "CELL_PARTITION": args.cell_partition,
        "CELL_TIME": args.cell_time,
        "CELL_MEM": args.cell_mem,
    }
    driver_job = _render_submit("loop_driver.sbatch.tmpl", dict(common))
    keepalive_job = _render_submit(
        "loop_keepalive.sbatch.tmpl", dict(common, SPEC=args.spec)
    )
    print(f"[loops] driver    -> job {driver_job}")
    print(f"[loops] keepalive -> job {keepalive_job}")
    print(f"[loops] both maintained by in-script afterany self-resubmit (singleton chain)")


if __name__ == "__main__":
    main()
