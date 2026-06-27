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
    """Parse 'size:count:partition:time[:gpu_type][,...]' into Targets.

    Time contains colons (e.g. 7-00:00:00), so we FIRST peel an optional trailing
    ``:gpu_type`` — recognizable because a time field is digit-led ('7-00:...') while a GPU
    label is alpha-led ('h100','a100'). Then split the rest into exactly 4 fields from the
    left. This lets entries on different partitions request different GPUs (h100 on
    ou_bcs_high, a100 on pi_tpoggio); a 4-field entry stays backward compatible via
    ``default_gpu``."""
    targets: list[Target] = []
    for item in [x for x in spec.split(",") if x.strip()]:
        gpu = default_gpu
        head, _, last = item.rpartition(":")
        if head and last[:1].isalpha():  # trailing alpha token => gpu_type, not a time field
            gpu = last
            item = head
        parts = item.split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                f"bad spec entry {item!r}; expected size:count:partition:time[:gpu_type]"
            )
        size, count, partition, time_limit = parts
        targets.append(Target(size, int(count), partition, time_limit, gpu))
    return targets


def _serve_jobs_in_flight(model_size: str, partition: str | None = None) -> int:
    """Count serve JOBS for this size currently R/PD/CF, optionally scoped to ``partition``.

    This is the authoritative "current + coming" server count: every server — running &
    registered, running but still loading, or pending — has exactly one squeue job. The
    live-endpoint set is a SUBSET of the running jobs, so we must NOT add live + jobs
    (that double-counts a registered running server). We launch ``count - this``.

    ``partition`` scoping lets the SAME size be served from multiple partitions with
    independent target counts (e.g. 1 replica on pi_tpoggio + 2 bonus on ou_bcs_high): each
    target maintains only its own partition's jobs instead of the two targets fighting over
    one size-global count.
    """
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%t|%P"],
        capture_output=True, text=True,
    ).stdout
    name = f"asys-serve-{model_size}"
    n = 0
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jname, state, jpart = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if jname == name and state in ("R", "PD", "CF") and (partition is None or jpart == partition):
            n += 1
    return n


def _running_serve_nodes(model_size: str) -> list[str]:
    """Nodes of currently-RUNNING serve jobs for this size."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-t", "R", "-o", "%j %N"],
        capture_output=True, text=True,
    ).stdout
    name = f"asys-serve-{model_size}"
    nodes = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            nodes.append(parts[1])
    return nodes


def _reregister_running(run_root: str, model_size: str) -> int:
    """Self-heal: re-register live endpoints whose registry file was lost.

    A running serve job whose registry entry got pruned (e.g. a transient false-dead) would
    otherwise be stranded forever — job count >= target so no relaunch, yet not discoverable.
    For each running serve node, probe the replica ports and re-register any live endpoint
    not already in the registry.
    """
    import dataclasses as _dc
    import json as _json

    from agents_scaling.models import get_model
    from agents_scaling.serving.launch_server import _port_for
    from agents_scaling.serving.registry import ServerEntry, _size_dir

    known = {(e.host, e.port) for e in registry.list_servers(run_root, model_size)}
    restored = 0
    for node in _running_serve_nodes(model_size):
        for r in range(8):
            port = _port_for(model_size, r)
            if (node, port) in known:
                continue
            # FAST single probe (3s, no retries) so a tick can't stall on flaky/saturated
            # ports — a miss just retries next tick (eventually consistent). Registration
            # is additive and safe; we never remove here.
            if healthcheck.is_alive(node, port, timeout=3.0):
                e = ServerEntry(model_size=model_size, hf_id=get_model(model_size).hf_id,
                                host=node, port=port)
                (_size_dir(run_root, model_size) / f"{node}_{port}.json").write_text(
                    _json.dumps(_dc.asdict(e)))
                known.add((node, port))
                restored += 1
                print(f"[keepalive] re-registered live {model_size} @ {node}:{port}")
    return restored


def _robust_alive(host: str, port: int, attempts: int = 5, timeout: float = 10.0) -> bool:
    """True if /health responds 200 on ANY of several attempts.

    Two failure modes make a single probe unreliable: (1) a SATURATED but healthy vLLM is
    slow to answer /health; (2) the login node <-> server-node path is intermittently flaky
    (observed: 5 consecutive 15s timeouts, then 0.1s success). So we retry several times
    with backoff before declaring dead. Even so, a sustained blip can false-prune — the
    self-heal re-registration (_reregister_running) recovers a still-running server on the
    next tick, so a false prune is transient, not permanent.
    """
    for i in range(attempts):
        if healthcheck.is_alive(host, port, timeout=timeout):
            return True
        if i < attempts - 1:
            time.sleep(3.0)
    return False


def _live_dead(run_root: str, model_size: str) -> tuple[list, list]:
    live, dead = [], []
    for e in registry.list_servers(run_root, model_size):
        (live if _robust_alive(e.host, e.port) else dead).append(e)
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
    """One keepalive pass.

    Policy: trust the SLURM JOB as the liveness signal, NOT /health probes. A saturated
    healthy vLLM intermittently fails /health (and the login<->node path is flaky), so
    probe-based pruning caused thrashing/false prunes. Instead:
      * a RUNNING serve job => presumed healthy; ensure it's registered (self-heal) but
        never prune/cancel it on a probe failure alone;
      * only relaunch when the serve-JOB count is below target (a job truly ended).
    This is robust to flaky probes; the cost is we don't auto-recover a true in-job vLLM
    hang (rare; surfaces as that size's cells slowing — handle manually if it occurs).
    """
    # Replica IDs are per-SIZE (ports = base + replica), but a size may now be served from
    # several partitions. Track IDs handed out this tick per size so two same-size targets
    # (e.g. pi_tpoggio + ou_bcs_high) never collide on a replica index / port.
    assigned_rids: dict[str, set[int]] = {}
    for t in targets:
        # Self-heal: re-register any running server whose registry file was lost, so cells
        # can discover it. (Best-effort; uses a probe but only to ADD, never to remove.)
        _reregister_running(run_root, t.size)
        # Relaunch only when the JOB count is below target FOR THIS PARTITION (a server
        # actually ended). Partition-scoped so multi-partition specs for one size don't
        # fight over a single size-global count.
        have = _serve_jobs_in_flight(t.size, t.partition)
        if have >= t.count:
            continue
        need = t.count - have
        # Avoid collisions with live replicas AND any IDs already handed out this tick for
        # this size (covers pending servers on other partitions not yet in the registry).
        used = _used_replica_ids(_live_dead(run_root, t.size)[0]) | assigned_rids.get(t.size, set())
        rid = 0
        for _ in range(need):
            while rid in used:
                rid += 1
            used.add(rid)
            assigned_rids.setdefault(t.size, set()).add(rid)
            try:
                job = submit_server(t.size, run_root, t.partition, t.gpu_type, t.time_limit, replica=rid)
                print(f"[keepalive] RELAUNCH {t.size} r{rid} on {t.partition} ({t.gpu_type}) -> job {job} "
                      f"(have {have}/{t.count} on {t.partition})")
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
