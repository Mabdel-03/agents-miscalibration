"""Launch chunk-driver + keepalive as SLURM batch jobs (durable across login sessions).

Login-node nohup processes (even with setsid+disown) repeatedly get reaped on this
cluster when the user's session ends. The fix: run the two long-lived loops as actual
SLURM jobs on compute nodes. They're CPU-only, tiny, and auto-requeue on preemption.

Usage:
  python slurm/launch_loops.py --run-id full_sweep_v1 \
      --spec 0.6B:1:pi_tpoggio:7-00:00:00,1.7B:2:ou_bcs_low:1-00:00:00,...
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from agents_scaling.config import DEFAULT_RESULTS_ROOT

REPO = Path(__file__).resolve().parent.parent


def _render_submit(tmpl_name: str, repl: dict[str, str]) -> str:
    """Render an sbatch template into a STABLE-NAMED file under slurm/ (so the script's
    self-resubmit can reference itself), then submit it. Returns the job id."""
    text = (REPO / "slurm" / tmpl_name).read_text()
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    # Stable name keyed by run-id so the self-resubmit always finds itself.
    sbatch_path = REPO / "slurm" / f"{tmpl_name.replace('.tmpl', '')}.{repl['RUN_ID']}"
    sbatch_path.write_text(text)
    # Use --requeue too (handles preemption); walltime expiry is handled by the
    # script's own dependency-based self-resubmit (see loop_*.sbatch.tmpl).
    out = subprocess.run(
        ["sbatch", "--requeue", "--parsable", str(sbatch_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(";")[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch driver+keepalive as SLURM jobs.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--spec", required=True,
                    help="replica spec for keepalive (size:count:partition:time,...)")
    ap.add_argument("--partition", default="mit_preemptable",
                    help="partition for the loop jobs (CPU-only, long limit)")
    ap.add_argument("--time", default="2-00:00:00",
                    help="walltime per loop job (--requeue restarts on expiry)")
    args = ap.parse_args()

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
    }
    driver_job = _render_submit("loop_driver.sbatch.tmpl", dict(common))
    keepalive_job = _render_submit(
        "loop_keepalive.sbatch.tmpl", dict(common, SPEC=args.spec)
    )
    print(f"[loops] driver    -> job {driver_job}")
    print(f"[loops] keepalive -> job {keepalive_job}")
    print(f"[loops] both auto-requeue on preemption/timeout (resumable code)")


if __name__ == "__main__":
    main()
