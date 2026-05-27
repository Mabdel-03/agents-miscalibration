"""Server keepalive: maintain a DESIRED REPLICA COUNT per model size for a long sweep.

Servers have walltime limits (pi_tpoggio 7-day, ou_bcs 1-day), so over a multi-week run
they get killed and cells then block/round-robin onto dead endpoints. This loop, each pass:
  1. probes a real HTTP /health on every registered endpoint (presence != liveness);
  2. prunes the registry file of any dead endpoint (so cells stop hitting it);
  3. for each managed size, if (live endpoints) + (serve jobs already PD/R) < desired count,
     relaunches enough replicas on that size's designated partition to reach the target.

The target is a **replica spec**: ``size:count:partition:time`` entries, e.g.
``0.6B:1:pi_tpoggio:7-00:00:00,32B:6:ou_bcs_low:1-00:00:00``. Replica indices are assigned
to keep ports distinct (see launch_server._port_for(size, replica)).

Stateless/idempotent (reads live SLURM + registry each tick) -> kill/restart freely.

Usage:
  python slurm/keepalive.py --run-id full_sweep_v1 --interval 600 \
      --spec 0.6B:1:pi_tpoggio:7-00:00:00,1.7B:2:ou_bcs_low:1-00:00:00,...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.models import REGISTRY
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving.launch_server import submit as submit_server


@dataclass
class Target:
    size: str
    count: int          # desired number of live endpoints for this size
    partition: str      # where to launch (re)replicas
    time_limit: str
    gpu_type: str = "a100"


def parse_spec(spec: str, default_gpu: str) -> list[Target]:
    """Parse 'size:count:partition:time[,...]' into Targets. Time may contain colons
    (e.g. 7-00:00:00), so split into at most 4 fields from the left."""
    targets: list[Target] = []
    for item in [x for x in spec.split(",") if x.strip()]:
        parts = item.split(":", 3)
        if len(parts) != 4:
            raise ValueError(f"bad spec entry {item!r}; expected size:count:partition:time")
        size, count, partition, time_limit = parts
        targets.append(Target(size, int(count), partition, time_limit, default_gpu))
    return targets


def _serve_jobs_in_flight(model_size: str) -> int:
    """Count serve JOBS for this size currently R/PD/CF.

    This is the authoritative "current + coming" server count: every server — running &
    registered, running but still loading, or pending — has exactly one squeue job. The
    live-endpoint set is a SUBSET of the running jobs, so we must NOT add live + jobs
    (that double-counts a registered running server). We launch ``count - this``.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %t"],
        capture_output=True, text=True,
    ).stdout
    name = f"asys-serve-{model_size}"
    n = 0
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == name and parts[-1] in ("R", "PD", "CF"):
            n += 1
    return n


def _live_dead(run_root: str, model_size: str) -> tuple[list, list]:
    live, dead = [], []
    for e in registry.list_servers(run_root, model_size):
        (live if healthcheck.is_alive(e.host, e.port) else dead).append(e)
    return live, dead


def _used_replica_ids(live: list) -> set[int]:
    """Infer which replica indices are live from their ports (port = base + replica)."""
    used = set()
    for e in live:
        from agents_scaling.serving.launch_server import _port_for

        base = _port_for(e.model_size, 0)
        off = e.port - base
        if 0 <= off < 64:  # plausible replica offset
            used.add(off)
    return used


def tick(run_root: str, targets: list[Target]) -> None:
    """One keepalive pass: prune dead endpoints, top up each size to its desired count."""
    for t in targets:
        live, dead = _live_dead(run_root, t.size)
        for e in dead:
            registry.prune_entry(run_root, e)
            print(f"[keepalive] pruned dead {t.size} @ {e.host}:{e.port}")
        # Job count is authoritative (live endpoints are a subset of running jobs).
        have = _serve_jobs_in_flight(t.size)
        if have >= t.count:
            continue
        need = t.count - have
        used = _used_replica_ids(live)
        # Pick the lowest free replica indices for the new servers (distinct ports).
        rid = 0
        for _ in range(need):
            while rid in used:
                rid += 1
            used.add(rid)
            try:
                job = submit_server(t.size, run_root, t.partition, t.gpu_type, t.time_limit, replica=rid)
                print(f"[keepalive] RELAUNCH {t.size} r{rid} on {t.partition} -> job {job} "
                      f"(have {have}/{t.count})")
            except Exception as exc:  # noqa: BLE001 — never die on one failure
                print(f"[keepalive] FAILED {t.size} r{rid} on {t.partition}: {exc!r}")
            rid += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Maintain desired vLLM replica counts per size.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", default=None, help="defaults to $ASYS_RESULTS_ROOT/<run-id>")
    ap.add_argument(
        "--spec", required=True,
        help="comma-list of size:count:partition:time, e.g. 0.6B:1:pi_tpoggio:7-00:00:00,32B:6:ou_bcs_low:1-00:00:00",
    )
    ap.add_argument("--gpu-type", default="a100")
    ap.add_argument("--interval", type=float, default=600.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="single pass then exit (for testing)")
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(
        os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT), args.run_id
    )
    targets = parse_spec(args.spec, args.gpu_type)
    unknown = [t.size for t in targets if t.size not in REGISTRY]
    if unknown:
        ap.error(f"unknown model sizes: {unknown} (known: {sorted(REGISTRY)})")

    print(f"[keepalive] run_root={run_root}")
    for t in targets:
        print(f"[keepalive] target: {t.size} x{t.count} on {t.partition} ({t.time_limit})")
    while True:
        try:
            tick(run_root, targets)
        except Exception as exc:  # noqa: BLE001 — never let a transient error kill the loop
            print(f"[keepalive] tick error (continuing): {exc!r}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
