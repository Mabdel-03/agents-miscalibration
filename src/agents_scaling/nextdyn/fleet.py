"""The serving fleet: spec parsing, census, scale-to-idle, cap preflight, keepalive, teardown.

Plan §3.2 (pools are sub-run-roots; scale each spec entry down to idle capacity with zero
preemption of other users; ``fleet up`` refuses any spec whose per-(partition, gpu_type) GPU
sum exceeds the MaxTRESPU captured in ``preflight/h0.json``), D20, spec 16.3.

Four ARD36/study_v4 operational defects are structural here:

* **The (profile, partition) census collision.**  ``census`` keys on
  ``(profile, partition, gpu_type)``, so an H100 replica and an A100 replica of one profile on
  one partition stop cancelling each other out
  (``test_census_keys_on_gpu_type_so_h100_and_a100_entries_do_not_fight``).
* **The keepalive relaunching a cancelled fleet.**  Every tick checks ``$RUN/FLEET_STOP``
  first and exits 0 without launching; ``fleet stop`` writes that file **before** any
  ``scancel``, and teardown uses exact job names from :func:`job_names` -- never ``pkill -f``.
* **The GPU sampler nobody sourced.**  ARD36's resources telemetry was NULL on all 67,105 rows
  because the serve template never sourced the sampler.  :func:`inject_gpu_sampler` and
  :func:`inject_sacct_capture` are anchor inserts into the *rendered* script and fail hard when
  an anchor is missing, so a template change breaks the render test rather than the telemetry.
* **Pools that shared one registry.**  ``serving.registry.servers_dir`` is hard-coded to
  ``<root>/servers``, so a pool is a **sub-run-root** ``$RUN/pools/<pool>/`` with its own
  registry and its own ``ASYS_PORT_OFFSET``.

``parse_spec`` is vendored from ``slurm/keepalive.py:172`` (which is not on the ``ARD_IMPORTS``
allowlist and lives in a 3,941-line module that reaches the scheduler at import) with a parity
test over a dozen spec strings.  Every number -- tick interval, launches per tick, per-GPU
``max_num_seqs``, caps -- comes from ``compute.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from agents_scaling.experiment import io
from agents_scaling.nextdyn import cli, config as C
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation, ManifestError
from agents_scaling.nextdyn.kinds import Pool
from agents_scaling.serving import registry as REG

FLEET_STOP_FILE: Final[str] = "FLEET_STOP"
POOLS_DIR: Final[str] = "pools"
FLEET_DIR: Final[str] = "fleet"
PREFLIGHT_FILE: Final[str] = "preflight/h0.json"
RESOURCES_DIR: Final[str] = "resources"
LOGS_DIR: Final[str] = "logs"
#: ``registry.servers_dir`` is hard-coded to ``<root>/servers`` (plan §3.2).
SERVERS_DIR: Final[str] = "servers"
#: The model contract the registrar loads, and the env var that may override it.
MODEL_CONTRACT_FILE: Final[str] = "model_contracts.v1.json"
MODEL_CONTRACT_ENV: Final[str] = "ASYS_MODEL_CONTRACT"
MODEL_CONTRACT_SHA_ENV: Final[str] = "ASYS_MODEL_CONTRACT_SHA256"
SBATCH_DIR: Final[str] = "sbatch"
JOB_PREFIX: Final[str] = "asys-nextdyn"
SAMPLER_SCRIPT: Final[str] = "slurm/nextdyn_gpu_sampler.sh"
#: <repo>/src/agents_scaling/nextdyn/fleet.py -> <repo> (no index literal: plan §3.3).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent

#: The rendered serve script's anchors (``slurm/serve_qwen.sbatch.tmpl``).  A missing anchor is
#: a hard failure: silently skipping the insert is precisely the ARD36 defect.
ANCHOR_VLLM_PID: Final[str] = "VLLM_PID=$!"
ANCHOR_EXIT_TRAP: Final[str] = "trap 'rm -rf \"$ASYS_LOCAL_CACHE\"' EXIT"
SAMPLER_CALL: Final[str] = "nextdyn_start_gpu_sampler"
SACCT_CALL: Final[str] = "nextdyn_capture_sacct"


# --------------------------------------------------------------------------- spec grammar


@dataclass(frozen=True)
class Target:
    """One spec entry: ``profile:count:partition:time[:gpu_type]`` (the legacy grammar)."""

    size: str
    count: int
    partition: str
    time_limit: str
    gpu_type: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.size, self.partition, self.gpu_type)

    def render(self) -> str:
        tail = f":{self.gpu_type}" if self.gpu_type else ""
        return f"{self.size}:{self.count}:{self.partition}:{self.time_limit}{tail}"


def parse_spec(spec: str, default_gpu: str) -> list[Target]:
    """Parse ``size:count:partition:time[:gpu_type][,...]`` (vendored, ``keepalive.py:172``).

    Time contains colons (``7-00:00:00``), so an optional trailing ``:gpu_type`` is peeled
    FIRST -- recognisable because a time field is digit-led and a GPU label is alpha-led --
    and the rest is split into exactly four fields from the left.  A four-field entry keeps
    working through ``default_gpu``.  ``test_spec_parity_with_legacy_parser`` holds this to
    the original on a dozen strings.
    """
    targets: list[Target] = []
    for item in [x for x in str(spec).split(",") if x.strip()]:
        gpu = default_gpu
        head, _, last = item.rpartition(":")
        if head and last[:1].isalpha():         # trailing alpha token => gpu type, not a time
            gpu = last
            item = head
        parts = item.split(":", TIME_SPLIT_MAX)
        if len(parts) != len(SPEC_FIELDS):
            raise ValueError(f"bad spec entry {item!r}; expected size:count:partition:time[:gpu_type]")
        size, count, partition, time_limit = parts
        targets.append(Target(size, int(count), partition, time_limit, gpu))
    return targets


#: ``size:count:partition:time`` -- four fields, so three splits from the left.
SPEC_FIELDS: Final[tuple[str, ...]] = ("size", "count", "partition", "time")
TIME_SPLIT_MAX: Final[int] = len(SPEC_FIELDS) - 1


def render_spec(targets: Sequence[Target]) -> str:
    return ",".join(t.render() for t in targets)


# --------------------------------------------------------------------------- pools


def pool_run_root(run_root: str | os.PathLike, pool: Pool | str) -> Path:
    """``$RUN/pools/<pool>`` -- a sub-run-root, because the registry hard-codes ``/servers``."""
    name = pool.value if isinstance(pool, Pool) else str(pool)
    return Path(run_root) / POOLS_DIR / name


def pool_block(compute: Any, pool: Pool | str) -> Any:
    name = pool.value if isinstance(pool, Pool) else str(pool)
    try:
        return getattr(compute.pools, name)
    except AttributeError as exc:
        raise ConfigError(f"compute.pools has no pool {name!r}") from exc


def port_offset(compute: Any, pool: Pool | str) -> int:
    return int(pool_block(compute, pool).port_offset)


def max_num_seqs_for(compute: Any, *, profile: str, gpu_type: str) -> int | None:
    """``compute.serving.max_num_seqs_by_gpu`` -- a scheduling knob, outside request identity."""
    table = compute.serving.max_num_seqs_by_gpu.get(str(gpu_type), {})
    for key, value in table.items():
        if str(profile).endswith(str(key)):
            return int(value)
    return None


# --------------------------------------------------------------------------- census


@dataclass(frozen=True)
class ServeJob:
    job_id: str
    name: str
    partition: str
    state: str
    gres: str
    nodes: str
    elapsed: str

    @property
    def gpu_type(self) -> str:
        """The GPU label Slurm reports in ``%b`` (``gpu:h100:2`` -> ``h100``)."""
        match = re.search(r"gpu:([A-Za-z0-9_]+)", self.gres or "")
        return match.group(1) if match else ""

    @property
    def gpus(self) -> int:
        match = re.search(r"gpu:[A-Za-z0-9_]*:?(\d+)", self.gres or "")
        return int(match.group(1)) if match else 1


def _sh(cmd: Sequence[str]) -> str:
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True).stdout
    except Exception:                                   # noqa: BLE001 - the scheduler is advisory
        return ""


def serving_job_name(run_root: str | os.PathLike, profile: str) -> str:
    return REG.serving_job_name(str(run_root), profile)


def census(run_root: str | os.PathLike, pool: Pool | str, profiles: Iterable[str], *,
           runner: Any = None) -> dict[tuple[str, str, str], list[ServeJob]]:
    """Live serve jobs keyed on ``(profile, partition, gpu_type)``.

    Keyed on the GPU type as well as the partition because one profile may hold H100 and A100
    replicas on one partition; ARD36's (profile, partition) key made those two entries fight
    over one count and the keepalive oscillated.
    """
    root = pool_run_root(run_root, pool)
    names = {serving_job_name(root, profile): profile for profile in profiles}
    out: dict[tuple[str, str, str], list[ServeJob]] = {}
    raw = (runner or _sh)(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o",
                           "%i %j %P %T %b %N %M"])
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < len(SERVE_FIELDS):
            continue
        job_id, name, partition, state, gres, nodes, elapsed = fields[:len(SERVE_FIELDS)]
        profile = names.get(name)
        if profile is None:
            continue
        job = ServeJob(job_id=job_id, name=name, partition=partition, state=state, gres=gres,
                       nodes=nodes, elapsed=elapsed)
        out.setdefault((profile, partition, job.gpu_type), []).append(job)
    return out


SERVE_FIELDS: Final[tuple[str, ...]] = ("job_id", "name", "partition", "state", "gres", "nodes",
                                        "elapsed")


def live_endpoint_count(run_root: str | os.PathLike, *, compute: Any,
                        pool: Pool | str | None = None) -> int:
    """Registered, live endpoints across the pools (what the lane throttle is computed from)."""
    del compute                          # the registry declares its own names; see _served_names
    total = 0
    pools = [pool] if pool is not None else list(Pool)
    for one in pools:
        root = pool_run_root(run_root, one)
        for name in _served_names(root):
            try:
                total += len(REG.list_live_servers(root, name))
            except Exception:                            # noqa: BLE001 - a monitor never raises
                continue
    return total


def live_endpoints_by_lane(run_root: str | os.PathLike, pool: Pool | str) -> dict[str, int]:
    """How many *probed-live* endpoints this pool has, per lane (``8B`` / ``32B``).

    Per lane, because a pool serves more than one: E2b puts its 8B and 32B cells in the same
    `low` pool, so "the pool has an endpoint" is not the question a 32B cell is asking.

    Probed, because a registry *directory* outlives the replica that made it.
    ``list_live_servers`` deliberately never deletes a shared record on a failed observation
    (keepalive owns cleanup), so after a preemption the directory still names the dead host --
    and a check that only listed directory names called the pool live, dispatched into it, and
    got ``exit 2`` from every cell.
    """
    root = pool_run_root(run_root, pool)
    out: dict[str, int] = {}
    for name in _served_names(root):
        try:
            live = len(REG.list_live_servers(root, name))
        except Exception:                                # noqa: BLE001 - a monitor never raises
            continue
        if live:
            lane = model_size_of(name)
            out[lane] = out.get(lane, 0) + live
    return out


def _served_names(server_root: Path) -> tuple[str, ...]:
    """The names this pool's registry is actually keyed by, read from the registry itself.

    A registry directory is named for the **serving profile** (``nd1-8B``), not for the served
    model name (``8B``).  Deriving the names from ``compute.serving.max_num_seqs_by_gpu``, whose
    tables are keyed by model size, meant this counted endpoints under ``servers/8B/`` while
    every replica had registered under ``servers/nd1-8B/`` -- so a pool with live replicas
    counted zero, and ``gates launch --required-replicas`` would have failed against a healthy
    fleet forever.  Reading the directory is also what makes an ad-hoc ``fleet up --spec`` with
    a profile that is in no pool spec count correctly.
    """
    base = server_root / SERVERS_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in base.iterdir() if entry.is_dir()))


# --------------------------------------------------------------------------- idle capacity


@dataclass(frozen=True)
class GpuCensus:
    partition: str
    gpu_type: str
    total: int
    used: int
    drained: tuple[str, ...] = ()

    @property
    def free(self) -> int:
        return max(self.total - self.used, 0)


def free_gpus(partition: str, gpu_type: str, *, exclude: Sequence[str] = (),
              runner: Any = None) -> GpuCensus:
    """Idle GPUs of one type on a partition (vendored ``scripts/ard36_preflight.py:77``)."""
    raw = (runner or _sh)(["sinfo", "-h", "-p", partition, "-N", "-O",
                           "NodeHost:20,Gres:60,GresUsed:60,StateLong:20"])
    total = used = 0
    drained: list[str] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < len(SINFO_FIELDS):
            continue
        host, gres, gres_used, state = parts[:len(SINFO_FIELDS)]
        if gpu_type not in gres:
            continue
        if "drain" in state.lower() or "down" in state.lower() or host in exclude:
            drained.append(host)
            continue
        total += _gres_count(gres, gpu_type)
        used += _gres_count(gres_used, gpu_type)
    return GpuCensus(partition=partition, gpu_type=gpu_type, total=total, used=used,
                     drained=tuple(drained))


SINFO_FIELDS: Final[tuple[str, ...]] = ("host", "gres", "gres_used", "state")


def _gres_count(field_text: str, gpu_type: str) -> int:
    for chunk in str(field_text).split(","):
        if chunk.startswith(f"gpu:{gpu_type}:"):
            token = chunk.split(":")[-1].split("(")[0]
            return int(token) if token.isdigit() else 0
    return 0


def scale_to_idle(targets: Sequence[Target], *, idle: Mapping[tuple[str, str], int],
                  gpus_per_replica: Mapping[str, int]) -> list[Target]:
    """Scale every entry down to what is idle, so the launch preempts nobody (plan D20).

    Entries are visited in spec order and each one consumes the idle pool it draws from, so
    two entries on one (partition, gpu_type) cannot both be told "there are four free".
    """
    remaining = {key: int(value) for key, value in idle.items()}
    out: list[Target] = []
    for target in targets:
        per_replica = max(int(gpus_per_replica.get(target.size, 1)), 1)
        pool_key = (target.partition, target.gpu_type)
        available = remaining.get(pool_key)
        if available is None:
            out.append(target)
            continue
        count = min(int(target.count), available // per_replica)
        remaining[pool_key] = available - count * per_replica
        if count > 0:
            out.append(replace(target, count=count))
    return out


# --------------------------------------------------------------------------- caps


def load_preflight(run_root: str | os.PathLike) -> Mapping[str, Any]:
    path = Path(run_root) / PREFLIGHT_FILE
    if not path.exists():
        raise ManifestError(f"{path} does not exist; run scripts/nextdyn_preflight.py before "
                            "`fleet up` (plan §3.5 S1)")
    return json.loads(path.read_text(encoding="utf-8"))


def caps_from_preflight(preflight: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    """``{(partition, gpu_type): MaxTRESPU gpus}`` as the preflight captured them live."""
    out: dict[tuple[str, str], int] = {}
    for row in preflight.get("qos", {}).get("caps", []):
        out[(str(row["partition"]), str(row["gpu_type"]))] = int(row["gpu"])
    return out


def planning_caps(compute: Any) -> dict[tuple[str, str], int]:
    """``compute.yaml``'s per-(partition, gpu_type) snapshot, plus each partition's total.

    The snapshot is not the authority -- ``preflight/h0.json`` is -- but it is what the plan
    was sized against, so a live cap *below* it is drift worth naming rather than silently
    obeying.  ``gpu_total`` is the separate ceiling on the partition as a whole (ou_bcs_normal
    admits 8 A100 **and** 8 H100, but only 16 GPUs in total).
    """
    out: dict[tuple[str, str], int] = {}
    for name, block in compute.partitions.items():
        for gpu_type, count in block.gpu.items():
            out[(name, gpu_type)] = int(count)
        if block.gpu_total is not None:
            out[(name, "")] = int(block.gpu_total)
    return out


def assert_walltimes(targets: Sequence[Target], compute: Any) -> None:
    """A spec entry may not ask for more walltime than its partition grants.

    Slurm refuses the job outright, so the failure mode is a fleet that silently never comes
    up: the keepalive submits, the scheduler rejects, the tick logs and moves on.
    """
    from agents_scaling.nextdyn import workload as WL

    for target in targets:
        block = compute.partitions.get(target.partition)
        if block is None:
            continue
        if WL.walltime_seconds(target.time_limit) > WL.walltime_seconds(block.max_walltime):
            raise ContractViolation(
                f"{target.render()}: {target.partition} grants at most {block.max_walltime}")


def assert_pool_preemption(pool: Pool | str, targets: Sequence[Target], compute: Any) -> None:
    """A non-preemptable pool never launches onto a preemptable partition (plan D20).

    The deadline pool is the one that matters: a preempted replica there is not a slow cell,
    it is a cell that dies mid-episode and leaves a half-written round.
    """
    if pool_block(compute, pool).preemptable_allowed:
        return
    offenders = [target.render() for target in targets
                 if _is_preemptable(compute, target.partition)]
    if offenders:
        name = pool.value if isinstance(pool, Pool) else str(pool)
        raise ContractViolation(
            f"pool {name!r} does not admit preemptable replicas but its spec names {offenders}")


def _is_preemptable(compute: Any, partition: str) -> bool:
    """Whether Slurm may requeue a job on this partition (``compute.partitions.*.preemptable``)."""
    block = compute.partitions.get(partition)
    return bool(block.preemptable) if block is not None else False


def assert_within_caps(targets: Sequence[Target], preflight: Mapping[str, Any], *,
                       gpus_per_replica: Mapping[str, int], compute: Any = None) -> None:
    """Refuse a spec whose per-(partition, gpu_type) GPU sum exceeds the live cap (plan D20).

    The *live* cap, from ``preflight/h0.json``, not the planning snapshot in ``compute.yaml``:
    a QOS that was tightened since the plan was written is the case this exists for.  When the
    compiled compute config is passed, the snapshot's caps and totals are checked too, so a
    spec that was over the plan's own numbers is caught even before a preflight exists.
    """
    caps = caps_from_preflight(preflight)
    if compute is not None:
        for key, value in planning_caps(compute).items():
            caps.setdefault(key, value)
    wanted: dict[tuple[str, str], int] = {}
    for target in targets:
        per_replica = max(int(gpus_per_replica.get(target.size, 1)), 1)
        gpus = int(target.count) * per_replica
        for key in ((target.partition, target.gpu_type), (target.partition, "")):
            wanted[key] = wanted.get(key, 0) + gpus
    over = {key: (value, caps[key]) for key, value in wanted.items()
            if key in caps and value > caps[key]}
    if over:
        lines = "; ".join(f"{partition}/{gpu}: {want} > cap {cap}"
                          for (partition, gpu), (want, cap) in sorted(over.items()))
        raise ContractViolation(
            f"fleet spec exceeds the captured MaxTRESPU caps ({lines}); scale the spec down "
            "rather than letting Slurm hold the surplus pending (plan D20)")


# --------------------------------------------------------------------------- serve injection


def inject_gpu_sampler(text: str, *, run_root: str | os.PathLike, repo: str | os.PathLike) -> str:
    """Start the per-replica sampler right after the server starts (anchor insert).

    ``launch_server`` renders the serve script; this rewrites the rendered text rather than the
    template, exactly as ``launch_server.py:703-711`` inserts ``ASYS_PORT_OFFSET``.  A missing
    anchor raises: ARD36's sampler was never sourced and every ``resources`` row came out NULL,
    which no test noticed because nothing asserted the insert happened.
    """
    if SAMPLER_CALL in text:
        return text                                      # idempotent: already prepared
    if ANCHOR_VLLM_PID not in text:
        raise ManifestError(f"the rendered serve script has no {ANCHOR_VLLM_PID!r} anchor; the "
                            "GPU sampler cannot be injected (spec 16.1 resources.parquet)")
    csv = f'"{Path(run_root) / RESOURCES_DIR}/gpu_$SLURM_JOB_ID.csv"'
    insert = (f'source "{Path(repo) / SAMPLER_SCRIPT}"\n'
              f'{SAMPLER_CALL} {csv}\n')
    return text.replace(ANCHOR_VLLM_PID, ANCHOR_VLLM_PID + "\n" + insert, 1)


def inject_sacct_capture(text: str, *, run_root: str | os.PathLike) -> str:
    """Extend the EXIT trap with the ``sacct`` capture for this replica (anchor insert)."""
    if SACCT_CALL in text:
        return text                                      # idempotent: already prepared
    if ANCHOR_EXIT_TRAP not in text:
        raise ManifestError(f"the rendered serve script has no EXIT trap anchor "
                            f"({ANCHOR_EXIT_TRAP!r}); the sacct capture cannot be injected")
    out = f'"{Path(run_root) / RESOURCES_DIR}/sacct_$SLURM_JOB_ID.json"'
    body = (f'{SACCT_CALL}() {{ mkdir -p "{Path(run_root) / RESOURCES_DIR}"; '
            f'sacct -j "$SLURM_JOB_ID" --json > {out} 2>/dev/null || true; }}\n')
    trap = ("trap 'rm -rf \"$ASYS_LOCAL_CACHE\"; nextdyn_stop_gpu_sampler || true; "
            f"{SACCT_CALL}' EXIT")
    return text.replace(ANCHOR_EXIT_TRAP, body + trap, 1)


def prepare_serve_script(text: str, *, run_root: str | os.PathLike,
                         repo: str | os.PathLike) -> str:
    """Both inserts, in the order the script runs them (sampler starts, trap stops)."""
    return inject_sacct_capture(inject_gpu_sampler(text, run_root=run_root, repo=repo),
                                run_root=run_root)


# --------------------------------------------------------------------------- job names


@dataclass(frozen=True)
class FleetJobNames:
    """The exact names ``fleet stop`` cancels, in teardown order (never ``pkill -f``)."""

    keepalives: tuple[str, ...]
    planner: str
    drivers: tuple[str, ...]
    status: str
    serve_prefix: str

    @property
    def ordered(self) -> tuple[str, ...]:
        return (*self.keepalives, self.planner, *self.drivers, self.status)


def job_names(run_id: str, *, pools: Sequence[Pool] | None = None,
              lanes: Sequence[str] | None = None) -> FleetJobNames:
    from agents_scaling.nextdyn import dispatch as D
    from agents_scaling.nextdyn.kinds import Lane

    pool_list = list(pools) if pools is not None else list(Pool)
    lane_list = list(lanes) if lanes is not None else [lane.value for lane in Lane]
    return FleetJobNames(
        keepalives=tuple(keepalive_job_name(run_id, pool) for pool in pool_list),
        planner=D.planner_job_name(run_id),
        drivers=tuple(D.driver_job_name(run_id, lane) for lane in lane_list),
        status=D.status_job_name(run_id),
        serve_prefix=f"asys-srv-{run_id}")


def keepalive_job_name(run_id: str, pool: Pool | str) -> str:
    name = pool.value if isinstance(pool, Pool) else str(pool)
    return f"{JOB_PREFIX}-keepalive-{run_id}-{name}"


# --------------------------------------------------------------------------- tick


@dataclass(frozen=True)
class TickReport:
    launched: tuple[str, ...]
    skipped: tuple[str, ...]
    stopped: bool
    detail: str = ""


def fleet_stop_path(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / FLEET_STOP_FILE


def is_stopped(run_root: str | os.PathLike) -> bool:
    return fleet_stop_path(run_root).exists()


def tick(run_root: str | os.PathLike, pool: Pool | str, targets: Sequence[Target], *,
         compute: Any, max_launch: int | None = None, submit: Any = None,
         runner: Any = None) -> TickReport:
    """One keepalive pass: launch what is missing, never prune, exit on ``FLEET_STOP``.

    Never prunes a registry record: a saturated healthy vLLM intermittently fails ``/health``
    and study_v4's probe-based pruning thrashed.  The Slurm job is the liveness signal, and a
    replica is relaunched only when its *job* is gone.
    """
    root = Path(run_root)
    if is_stopped(root):
        return TickReport(launched=(), skipped=tuple(t.render() for t in targets), stopped=True,
                          detail=f"{fleet_stop_path(root)} exists; launching nothing")
    budget = int(compute.keepalive.max_launch if max_launch is None else max_launch)
    profiles = sorted({t.size for t in targets})
    live = census(root, pool, profiles, runner=runner)
    launched: list[str] = []
    skipped: list[str] = []
    for target in targets:
        have = len(live.get(target.key, ()))
        need = int(target.count) - have
        if need <= 0:
            skipped.append(f"{target.render()} (have {have})")
            continue
        for replica in range(need):
            if len(launched) >= budget:
                skipped.append(f"{target.render()} (tick budget {budget} reached)")
                break
            job = _launch(root, pool, target, compute=compute, replica=have + replica,
                          submit=submit)
            launched.append(f"{target.render()}#{have + replica}={job}")
    return TickReport(launched=tuple(launched), skipped=tuple(skipped), stopped=False,
                      detail=f"{len(launched)} launched, {len(skipped)} skipped")


def model_size_of(profile: str) -> str:
    """The ``model_size`` the serving profile names (``nd1-8B`` -> ``8B``).

    ``launch_server._resolve_profile`` refuses a profile whose model size does not match the
    first argument, so the two cannot be the same string; a spec entry names the *profile*
    (plan §3.2) and this is the one place that projects it onto the model size.
    """
    block = _profiles().get(profile)
    if block is None:
        raise ConfigError(f"no serving profile named {profile!r}; the fleet spec names profiles "
                          f"(have {sorted(_profiles())})")
    return str(block.model_size)


def serve_script_path(server_root: Path, target: Target, replica: int) -> Path:
    """Where one launch attempt's prepared script is spooled.

    Intent-addressed like ``launch_server``'s production path, because the keepalive relaunches
    the *same* (profile, replica) every time a preemptable job dies and ``launch_server``
    refuses to overwrite a script it already spooled: a stable name would turn the second
    launch of a replica into a hard failure.
    """
    return (server_root / SBATCH_DIR /
            f"serve_{target.size}_r{int(replica)}.{uuid.uuid4().hex}.sbatch")


def _launch(run_root: Path, pool: Pool | str, target: Target, *, compute: Any, replica: int,
            submit: Any = None, repo: str | os.PathLike | None = None) -> str:
    """Render, **prepare**, spool and submit one replica into the pool's sub-run-root.

    The preparation is the point.  ``launch_server.submit`` renders, writes and ``sbatch``es in
    one call with no post-render hook, so calling it here would ship a serve script with no
    ``nvidia-smi`` sampler and no ``sacct`` trap -- which is precisely ARD36's defect, where
    every one of 67,105 ``resources`` rows came back NULL.  So this renders with
    :func:`launch_server.render_sbatch`, runs the text through :func:`prepare_serve_script`
    (which fails hard when an anchor has moved) and submits the prepared file.  ``submit``
    receives that file, so a test that stubs the submitter sees exactly the bytes Slurm would.
    """
    from agents_scaling.serving import launch_server as LS

    server_root = pool_run_root(run_root, pool)
    log_dir = server_root / LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ASYS_PORT_OFFSET"] = str(port_offset(compute, pool))
    rendered = LS.render_sbatch(
        model_size_of(target.size), str(server_root), target.partition, target.gpu_type,
        target.time_limit, str(log_dir), replica=replica, serving_profile=target.size,
        max_num_seqs=max_num_seqs_for(compute, profile=target.size, gpu_type=target.gpu_type),
        gpu_memory_utilization=float(compute.serving.gpu_memory_utilization),
        exclude_nodes=list(compute.serving.exclude_nodes),
        **model_contract_render_args())
    text = prepare_serve_script(rendered, run_root=run_root,
                                repo=REPO_ROOT if repo is None else repo)
    path = serve_script_path(server_root, target, replica)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tmp + fsync + rename, then read-only: Slurm spools the file it was handed, and the record
    # of what this replica was launched with has to survive as what actually ran.
    io.atomic_write_text(path, text)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if submit is not None:
        return str(submit(path, target, replica))
    return str(_sbatch(path))



def model_contract_path() -> Path:
    """The model contract the in-job registrar loads, resolved here rather than inherited."""
    override = os.environ.get(MODEL_CONTRACT_ENV)
    path = Path(override) if override else REPO_ROOT / "configs" / MODEL_CONTRACT_FILE
    if not path.exists():
        raise ManifestError(
            f"model contract {path} does not exist; the in-job registrar cannot start without it "
            f"(set {MODEL_CONTRACT_ENV} or restore configs/{MODEL_CONTRACT_FILE})")
    return path


def model_contract_render_args() -> dict[str, str]:
    """The contract's path and digest, from one read of one file.

    ``launch_server.render_sbatch`` defaults both of these to ambient ``ASYS_*`` environment
    variables, and the serve template scrubs every ``ASYS_*`` name before it runs -- so whatever
    the in-job registrar needs must be *rendered into* the script.  Launching from a shell that
    had exported the digest but not the path rendered ``ASYS_MODEL_CONTRACT=""`` beside a valid
    ``ASYS_MODEL_CONTRACT_SHA256``; the registrar resolved the empty path to the repo root and
    died with ``ModelContractError: missing regular model contract`` six seconds in, with the
    GPU already allocated.

    The path and the digest are one fact, so they are produced by one read here.  The digest is
    still a real check: it is taken at render time on the login node and re-verified inside the
    job on the compute node, so a contract edited between spool and start is caught.  An
    inherited ``ASYS_MODEL_CONTRACT_SHA256`` that disagrees with the file is refused rather than
    silently preferred -- disagreement means the environment is describing a different file than
    the one about to be shipped.
    """
    path = model_contract_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inherited = os.environ.get(MODEL_CONTRACT_SHA_ENV)
    if inherited and inherited != digest:
        raise ManifestError(
            f"{MODEL_CONTRACT_SHA_ENV}={inherited} disagrees with {path} (sha256 {digest}); the "
            f"environment names a different contract than the one being shipped")
    return {"model_contract_path": str(path), "model_contract_sha256": digest}


def _sbatch(path: Path) -> str:
    """``sbatch --parsable`` on a prepared serve script (the only place the fleet submits)."""
    out = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True,
                         check=True).stdout.strip()
    return out.split(";")[0] if out else ""


# --------------------------------------------------------------------------- the verb


def up(run_root: str | os.PathLike, pool: Pool | str, *, compute: Any, spec: str | None = None,
       preflight: Mapping[str, Any] | None = None, submit: Any = None, contract: Any = None,
       runner: Any = None) -> TickReport:
    """``fleet up``: scale the spec to idle capacity, refuse over-cap, launch the difference."""
    root = Path(run_root)
    block = pool_block(compute, pool)
    targets = parse_spec(spec if spec is not None else block.spec, DEFAULT_GPU_TYPE)
    assert_walltimes(targets, compute)
    assert_pool_preemption(pool, targets, compute)
    gpus = gpus_per_replica(contract)
    idle = {(t.partition, t.gpu_type): free_gpus(t.partition, t.gpu_type,
                                                 exclude=compute.serving.exclude_nodes,
                                                 runner=runner).free
            for t in targets}
    scaled = scale_to_idle(targets, idle=idle, gpus_per_replica=gpus)
    if preflight is not None:
        assert_within_caps(scaled, preflight, gpus_per_replica=gpus, compute=compute)
    report = tick(root, pool, scaled, compute=compute, submit=submit, runner=runner)
    _record_launch(root, pool, scaled, report)
    return report


DEFAULT_GPU_TYPE: Final[str] = "a100"


def gpus_per_replica(contract: Any = None) -> dict[str, int]:
    """``{serving profile: GPUs one replica holds}`` -- the model cell's tensor-parallel size.

    ``tp`` enters ``engine_digest`` and therefore every request id, so one profile has exactly
    one ``tp``: the 32B cell is TP2 on every GPU type (contract §3).  Read off the *model
    cells* when a contract is given, because that is where ``profile`` and ``tp`` are declared
    together, and off the serving profiles otherwise.
    """
    out: dict[str, int] = {}
    if contract is not None:
        for model in contract.models.values():
            out[model.profile] = int(model.tp)
    for name, block in _profiles().items():
        out.setdefault(name, int(getattr(block, "tp", 1) or 1))
    return out


def _profiles() -> Mapping[str, Any]:
    from agents_scaling.serving.profiles import SERVING_PROFILES

    return SERVING_PROFILES


def _record_launch(run_root: Path, pool: Pool | str, targets: Sequence[Target],
                   report: TickReport) -> Path:
    name = pool.value if isinstance(pool, Pool) else str(pool)
    path = run_root / FLEET_DIR / f"launch_{name}_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pool": name, "spec": render_spec(targets),
                                "launched": list(report.launched),
                                "skipped": list(report.skipped),
                                "stopped": report.stopped}, indent=2) + "\n", encoding="utf-8")
    return path


KEEPALIVE_TMPL: Final[str] = "nextdyn_keepalive.sbatch.tmpl"


def render_keepalive(run_root: str | os.PathLike, pool: Pool | str, *, compute: Any,
                     cfg: Any, repo: str | os.PathLike | None = None) -> Path:
    """Render one pool's keepalive job (2 CPU / 2 G / 12 h, afterany self-resubmit).

    ``ASYS_PORT_OFFSET`` is exported **after** ``common.sh``'s environment scrub -- the
    ``launch_server.py:703-711`` pattern -- because a scrubbed offset means the pool serves on
    the default ports and two pools collide on one socket.
    """
    root = Path(run_root)
    name = pool.value if isinstance(pool, Pool) else str(pool)
    repo_root = Path(repo) if repo is not None else REPO_ROOT
    keepalive = compute.keepalive
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = root / f"{keepalive_job_name(root.name, name)}.sbatch"
    values = {
        "JOB_NAME": keepalive_job_name(root.name, name),
        "PARTITION": ",".join(keepalive.partitions), "CPUS": str(int(keepalive.cpus)),
        "MEM": str(keepalive.mem), "TIME": str(keepalive.time), "LOG_DIR": str(log_dir),
        "REPO": str(repo_root), "RUN_ROOT": str(root), "POOL": name,
        "CONFIG": str(getattr(cfg, "path", C.DEFAULT_STUDY_CONFIG)),
        "PORT_OFFSET": str(port_offset(compute, name)), "TICK_S": str(int(keepalive.tick_s)),
        "SELF": str(path),
    }
    text = (repo_root / "slurm" / KEEPALIVE_TMPL).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unrendered = sorted(set(re.findall(r"\{([A-Z_]+)\}", text)))
    if unrendered:
        raise ManifestError(f"{KEEPALIVE_TMPL}: unrendered placeholder(s) {unrendered}")
    path.write_text(text, encoding="utf-8")
    return path


def stop(run_root: str | os.PathLike, *, run_id: str | None = None, cancel: Any = None
         ) -> cli.Report:
    """``fleet stop``: write ``FLEET_STOP`` first, then ``scancel -n`` the exact names.

    The order is the point: the keepalive reads the file at the top of every tick, so writing
    it before any cancellation is what stops a relaunch race (the study_v4 defect where the
    keepalive resurrected a fleet that had just been cancelled).
    """
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    fleet_stop_path(root).write_text(f"stopped at {time.time()}\n", encoding="utf-8")
    names = job_names(run_id or root.name)
    cancelled: list[str] = []
    for name in names.ordered:
        (cancel or _cancel)(name)
        cancelled.append(name)
    for profile in _profiles():
        for one in Pool:
            serve = serving_job_name(pool_run_root(root, one), profile)
            (cancel or _cancel)(serve)
            cancelled.append(serve)
    return cli.Report(completed=len(cancelled),
                      detail=f"FLEET_STOP written; scancel -n {len(cancelled)} exact names")


def _cancel(name: str) -> None:
    _sh(["scancel", "-n", name])


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("fleet", description="up / status / stop the serving fleet")
    parser.add_argument("action", choices=("up", "status", "stop"))
    parser.add_argument("--pool", default=Pool.DEADLINE.value,
                        choices=[p.value for p in Pool])
    parser.add_argument("--spec", default=None, help="override compute.yaml's pool spec")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    if args.action == "stop":
        return stop(run_root)
    if args.action == "status":
        profiles = sorted({t.size for t in parse_spec(pool_block(compute, args.pool).spec,
                                                      DEFAULT_GPU_TYPE)})
        live = census(run_root, args.pool, profiles)
        for (profile, partition, gpu_type), jobs in sorted(live.items()):
            print(f"{profile} {partition} {gpu_type}: {len(jobs)} live "
                  f"({[j.job_id for j in jobs]})")
        return cli.Report(completed=sum(len(v) for v in live.values()),
                          detail=f"pool {args.pool}: {len(live)} (profile, partition, gpu) groups")
    try:
        preflight = load_preflight(run_root)
    except ManifestError as exc:
        return cli.Report(blocked=1, detail=str(exc))
    try:
        report = up(run_root, args.pool, compute=compute, spec=args.spec, preflight=preflight,
                    contract=C.load_study(args.config).contract)
    except ContractViolation as exc:
        return cli.Report(blocked=1, detail=str(exc))
    return cli.Report(completed=len(report.launched), skipped=len(report.skipped),
                      blocked=1 if report.stopped else 0, detail=report.detail)


__all__ = [
    "ANCHOR_EXIT_TRAP", "ANCHOR_VLLM_PID", "DEFAULT_GPU_TYPE", "FLEET_STOP_FILE", "FleetJobNames",
    "GpuCensus", "POOLS_DIR", "PREFLIGHT_FILE", "REPO_ROOT", "SACCT_CALL", "SAMPLER_CALL",
    "SAMPLER_SCRIPT", "ServeJob", "TickReport",
    "Target", "assert_within_caps", "caps_from_preflight", "census", "fleet_stop_path",
    "free_gpus", "inject_gpu_sampler", "live_endpoints_by_lane", "inject_sacct_capture", "is_stopped", "job_names",
    "model_contract_path", "model_contract_render_args", "MODEL_CONTRACT_ENV",
    "MODEL_CONTRACT_FILE", "MODEL_CONTRACT_SHA_ENV",
    "keepalive_job_name", "live_endpoint_count", "load_preflight", "main", "max_num_seqs_for",
    "assert_pool_preemption", "assert_walltimes", "gpus_per_replica", "parse_spec",
    "planning_caps", "pool_block", "pool_run_root", "port_offset", "prepare_serve_script",
    "KEEPALIVE_TMPL", "render_keepalive", "render_spec", "scale_to_idle",
    "model_size_of", "serve_script_path", "serving_job_name", "stop", "tick", "up",
]
