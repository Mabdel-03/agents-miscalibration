"""One-shot fleet scale-up: bring each model size up to a desired replica count.

Idempotent: for each ``size:count:partition:time`` target it counts how many endpoints
are already live (+ serve jobs in flight) and launches only the missing replicas, on the
target partition, with distinct replica indices (distinct ports). Re-running is safe — it
just tops up to the target. Use ``--dry-run`` to preview.

This complements keepalive.py (which maintains the count over time); scale_up does the
initial burst. They share the same spec format and the same replica-id logic.

Usage:
  python slurm/scale_up.py --run-id full_sweep_v1 \
      --spec 1.7B:2:ou_bcs_low:1-00:00:00,4B:3:ou_bcs_low:1-00:00:00,8B:4:ou_bcs_low:1-00:00:00,\
14B:5:ou_bcs_low:1-00:00:00,32B:6:ou_bcs_normal:1-00:00:00 [--dry-run]
"""

from __future__ import annotations

import argparse
import os

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving.launch_server import _port_for
from agents_scaling.serving.launch_server import submit as submit_server

# Reuse the keepalive spec parser + in-flight counter for consistency.
from keepalive import Target, _serve_jobs_in_flight, parse_spec  # type: ignore


def _live_endpoints(run_root: str, size: str) -> list:
    return [e for e in registry.list_servers(run_root, size) if healthcheck.is_alive(e.host, e.port)]


def _used_replica_ids(live: list, size: str) -> set[int]:
    used = set()
    base = _port_for(size, 0)
    for e in live:
        off = e.port - base
        if 0 <= off < 64:
            used.add(off)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description="Top up vLLM replicas to a target count (one-shot).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--spec", required=True, help="size:count:partition:time[,...]")
    ap.add_argument("--gpu-type", default="a100")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(
        os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT), args.run_id
    )
    targets = parse_spec(args.spec, args.gpu_type)
    print(f"[scale_up] run_root={run_root}")
    total_launch = 0
    for t in targets:
        live = _live_endpoints(run_root, t.size)
        # Serve-job count is authoritative (live endpoints are a subset of running jobs);
        # adding them would double-count a registered running server.
        have = _serve_jobs_in_flight(t.size)
        need = max(0, t.count - have)
        print(f"[scale_up] {t.size}: serve_jobs={have} (live_endpoints={len(live)}) target={t.count} -> launch {need} on {t.partition}")
        used = _used_replica_ids(live, t.size)
        rid = 0
        for _ in range(need):
            while rid in used:
                rid += 1
            used.add(rid)
            if args.dry_run:
                print(f"  [dry-run] would launch {t.size} r{rid} on {t.partition} "
                      f"(port {_port_for(t.size, rid)})")
            else:
                submit_server(t.size, run_root, t.partition, t.gpu_type, t.time_limit, replica=rid)
                total_launch += 1
            rid += 1
    if not args.dry_run:
        print(f"[scale_up] launched {total_launch} replica server(s)")


if __name__ == "__main__":
    main()
