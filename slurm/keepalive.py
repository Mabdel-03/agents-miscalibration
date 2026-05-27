"""Server keepalive: relaunch any vLLM server that has died, for a long-running sweep.

Servers have walltime limits (pi_tpoggio 7-day, ou_bcs 1-day), so over a multi-week run
they get killed and cells then block forever on ``wait_for_server``. This loop watches each
model size's endpoints and, when a size has NO live endpoint, prunes the stale registry
entry and resubmits a server for it on that size's designated partition.

Liveness = an actual HTTP ``/health`` 200 (not merely a registry file existing), so a
registry file left behind by a killed job is correctly treated as dead.

Run under nohup for the duration of the sweep; it is itself stateless/idempotent (reads
live SLURM + registry state each tick), so it can be killed and restarted freely.

Usage:
  python slurm/keepalive.py --run-id full_sweep_v1 \
      --pi-sizes 0.6B,1.7B,14B --ou-sizes 4B,8B,32B \
      --pi-partition pi_tpoggio --ou-partition ou_bcs_normal --interval 600
"""

from __future__ import annotations

import argparse
import subprocess
import time

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.models import REGISTRY
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving.launch_server import submit as submit_server


def _has_pending_or_running_server(model_size: str) -> bool:
    """Is there already a serve job for this size queued/running (so don't double-submit)?"""
    out = subprocess.run(
        ["squeue", "-u", __import__("os").environ.get("USER", ""), "-h", "-o", "%j %t"],
        capture_output=True, text=True,
    ).stdout
    name = f"asys-serve-{model_size}"
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == name and parts[-1] in ("R", "PD", "CF"):
            return True
    return False


def _live_endpoints(run_root: str, model_size: str) -> tuple[list, list]:
    """Return (live, dead) ServerEntry lists for a size, probing /health on each."""
    live, dead = [], []
    for e in registry.list_servers(run_root, model_size):
        (live if healthcheck.is_alive(e.host, e.port) else dead).append(e)
    return live, dead


def tick(run_root: str, size_partition: dict[str, tuple[str, str, str]]) -> None:
    """One keepalive pass over all managed sizes."""
    for size, (partition, gpu_type, time_limit) in size_partition.items():
        live, dead = _live_endpoints(run_root, size)
        # Always prune dead registry files so cells don't round-robin onto them.
        for e in dead:
            registry.prune_entry(run_root, e)
            print(f"[keepalive] pruned dead endpoint {size} @ {e.host}:{e.port}")
        if live:
            continue  # at least one healthy endpoint — nothing to do
        if _has_pending_or_running_server(size):
            print(f"[keepalive] {size}: no live endpoint yet but a serve job is PD/R; waiting")
            continue
        # No live endpoint and no job in flight -> relaunch.
        try:
            job = submit_server(size, run_root, partition, gpu_type, time_limit)
            print(f"[keepalive] RELAUNCHED {size} on {partition} -> job {job}")
        except Exception as exc:  # noqa: BLE001 — keepalive must never die on one failure
            print(f"[keepalive] FAILED to relaunch {size} on {partition}: {exc!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Relaunch dead vLLM servers for a long sweep.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", default=None, help="defaults to $ASYS_RESULTS_ROOT/<run-id>")
    ap.add_argument("--pi-sizes", default="0.6B,1.7B,14B")
    ap.add_argument("--ou-sizes", default="4B,8B,32B")
    ap.add_argument("--pi-partition", default="pi_tpoggio")
    ap.add_argument("--pi-time", default="7-00:00:00")
    ap.add_argument("--ou-partition", default="ou_bcs_normal")
    ap.add_argument("--ou-time", default="1-00:00:00")
    ap.add_argument("--gpu-type", default="a100")
    ap.add_argument("--interval", type=float, default=600.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="single pass then exit (for testing)")
    args = ap.parse_args()

    import os

    run_root = args.run_root or os.path.join(
        os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT), args.run_id
    )
    size_partition: dict[str, tuple[str, str, str]] = {}
    for s in [x for x in args.pi_sizes.split(",") if x]:
        size_partition[s] = (args.pi_partition, args.gpu_type, args.pi_time)
    for s in [x for x in args.ou_sizes.split(",") if x]:
        size_partition[s] = (args.ou_partition, args.gpu_type, args.ou_time)
    # Guard against typos: every managed size must be a known model.
    unknown = [s for s in size_partition if s not in REGISTRY]
    if unknown:
        ap.error(f"unknown model sizes: {unknown} (known: {sorted(REGISTRY)})")

    print(f"[keepalive] run_root={run_root}")
    print(f"[keepalive] managing: {size_partition}")
    while True:
        try:
            tick(run_root, size_partition)
        except Exception as exc:  # noqa: BLE001 — never let a transient error kill the loop
            print(f"[keepalive] tick error (continuing): {exc!r}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
