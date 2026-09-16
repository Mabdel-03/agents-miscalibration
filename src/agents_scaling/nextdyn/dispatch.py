"""The lane driver: chunk admission, sparse resume, top-up, live throttle, caps.

Plan §3.3 (throttles and caps; "every numeric constant in ``configs/nextdyn/compute.yaml``"),
§3.4 (only cells whose dependencies are met are submitted), §3.5 S4, spec 17.5 (resumable;
reports completed / failed / skipped / blocked / reused).

One process per lane, run as a self-resubmitting Slurm job
(``slurm/nextdyn_lane_driver.sbatch.tmpl``).  Each pass:

1. **Two gates before anything is submitted** (the study_v4 operational fix ARD36 inherited):
   the per-lane headroom (this lane's own array tasks, counted by *exact* job name) and the
   global ``squeue -r`` count, which is what ``QOSMaxSubmitJobPerUserLimit`` actually counts.
   A pass that would exceed either does nothing and says so; it never submits and hopes.
2. **Only dependency-met cells.**  ``deps.dependency_status`` is the same reader the runner
   uses, so a cell waiting for a bundle costs no array slot; ``run_one`` re-checks anyway.
3. **Sparse resume.**  A chunk whose array has left the queue with cells still lacking
   ``meta.json`` is re-submitted as ``--array=3,7,11%N`` over exactly those absolute indices,
   never as the whole chunk again.
4. **Top-up inside a live array.**  ``ard36_launch_chunked`` gated on whole chunks, so a
   preempted straggler waited for its entire array to end -- hours, late in a run.  Cells that
   no live array task owns are re-dispatched immediately.
5. **The throttle tracks the fleet.**  ``throttle_for(lane, live_endpoints)`` is recomputed
   every pass from ``compute.dispatch.throttle_per_endpoint`` and applied to the live arrays
   with ``scontrol update ArrayTaskThrottle``: replicas that died take the throttle down with
   them instead of leaving 200 cells hammering two endpoints.

Nothing in this module is a number.  Throttles, per-lane and global caps, pass intervals and
the signal margin all come from ``compute.yaml``
(``test_no_numeric_dispatch_constant_lives_in_code``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import cli, config as C, contract as K, deps, kinds, plan as PLAN
from agents_scaling.nextdyn.cells import CellSpec
from agents_scaling.nextdyn.errors import ManifestError
from agents_scaling.nextdyn.kinds import Lane
from agents_scaling.nextdyn.runner import DEPS_UNMET_FILE, EXIT_DEPS_UNMET

#: <repo>/src/agents_scaling/nextdyn/dispatch.py -> <repo> (no index literal: the
#: dispatch modules carry no number at all, see the module docstring).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent.parent
SLURM_DIR: Final[str] = "slurm"
CELL_ARRAY_TMPL: Final[str] = "nextdyn_cell_array.sbatch.tmpl"
CPU_ARRAY_TMPL: Final[str] = "nextdyn_cpu_array.sbatch.tmpl"
#: The CAPTURE lane is the only one whose job holds a GPU of its own (plan §3.2): it replays
#: sealed decisions on a pinned checkpoint, so it needs ``--gres`` and the replay interpreter,
#: neither of which ``nextdyn_cell_array.sbatch.tmpl`` carries.
CAPTURE_TMPL: Final[str] = "nextdyn_capture.sbatch.tmpl"
LANE_DRIVER_TMPL: Final[str] = "nextdyn_lane_driver.sbatch.tmpl"
PLANNER_TMPL: Final[str] = "nextdyn_planner.sbatch.tmpl"
STATUS_TMPL: Final[str] = "nextdyn_status.sbatch.tmpl"

JOB_PREFIX: Final[str] = "asys-nextdyn"
CHUNK_LOG: Final[str] = "chunk_jobs"
#: Slurm states that still own a cell; terminal states are absent, so the cell is free.
LIVE_STATES: Final[str] = "PD,R,CG,CF,S,RS,RQ,RD,RF,RH"
ARRAY_RE: Final[re.Pattern[str]] = re.compile(r"^#SBATCH --array=.*$", re.M)
THROTTLE_RE: Final[re.Pattern[str]] = re.compile(r"^#SBATCH --array=.*?%(\d+)$", re.M)


# --------------------------------------------------------------------------- names


def job_name(run_id: str, lane: Lane | str) -> str:
    """The exact job name a lane's array tasks carry (teardown and counting both key on it)."""
    name = lane.value if isinstance(lane, Lane) else str(lane)
    return f"{JOB_PREFIX}-{run_id}-{name}"


def driver_job_name(run_id: str, lane: Lane | str) -> str:
    name = lane.value if isinstance(lane, Lane) else str(lane)
    return f"{JOB_PREFIX}-{run_id}-driver-{name}"


def planner_job_name(run_id: str) -> str:
    return f"{JOB_PREFIX}-{run_id}-planner"


def status_job_name(run_id: str) -> str:
    return f"{JOB_PREFIX}-{run_id}-status"


# --------------------------------------------------------------------------- squeue


def _sh(cmd: Sequence[str]) -> str:
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True).stdout
    except Exception:                                   # noqa: BLE001 - squeue is best effort
        return ""


def lane_task_count(run_id: str, lane: Lane | str, *, runner: Any = None) -> int:
    """This lane's submitted array tasks (PD/R/...), by EXACT job name."""
    out = (runner or _sh)(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%j"])
    name = job_name(run_id, lane)
    return sum(1 for line in out.splitlines() if line.strip() == name)


def total_task_count(*, runner: Any = None) -> int:
    """Every submitted job/array task of this user: what the association cap counts."""
    out = (runner or _sh)(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"])
    return sum(1 for line in out.splitlines() if line.strip())


def array_indices_live(job_id: str, *, runner: Any = None) -> set[int]:
    out = (runner or _sh)(["squeue", "-h", "-r", "-j", str(job_id), "-t", LIVE_STATES, "-o", "%K"])
    return {int(token) for token in out.split() if token.strip().isdigit()}


def job_is_live(job_id: str, *, runner: Any = None) -> bool:
    return bool((runner or _sh)(["squeue", "-h", "-j", str(job_id), "-o", "%i"]).strip())


# --------------------------------------------------------------------------- caps


@dataclass(frozen=True)
class Caps:
    """Every admission number of one pass, read from ``compute.yaml`` (plan §3.3)."""

    lane_submit_cap: int
    bank_submit_cap: int
    global_submit_cap: int
    association_cap: int
    throttle_per_endpoint: Mapping[str, int]

    @classmethod
    def of(cls, compute: Any, lane: Lane | str) -> "Caps":
        name = lane.value if isinstance(lane, Lane) else str(lane)
        dispatch = compute.dispatch
        try:
            lane_cap = int(dispatch.lane_submit_caps[name])
        except KeyError as exc:
            raise ManifestError(f"compute.dispatch.lane_submit_caps has no entry for lane "
                                f"{name!r}") from exc
        return cls(lane_submit_cap=lane_cap, bank_submit_cap=int(dispatch.bank_submit_cap),
                   global_submit_cap=int(dispatch.global_submit_cap),
                   association_cap=int(compute.association.max_submit_jobs),
                   throttle_per_endpoint=dict(dispatch.throttle_per_endpoint))

    def lane_cap_for(self, *, banks_only: bool) -> int:
        """Bank cells share the 8B lane with the episodes and must not crowd them out: the
        bank sub-cap is the separate ceiling ``compute.dispatch.bank_submit_cap`` sets."""
        return min(self.lane_submit_cap, self.bank_submit_cap) if banks_only else self.lane_submit_cap

    @property
    def chunk_size(self) -> int:
        """One array per pass, at most a lane's worth of tasks."""
        return self.lane_submit_cap


def throttle_for(lane: Lane | str, live_endpoints: int, *, caps: Caps,
                 kind: Any = None) -> int:
    """Cells per live endpoint, recomputed every pass (plan §3.3).

    The per-kind entry is the knob: an episode cell holds ``n`` requests in flight while a
    bank cell holds sixteen, so "how many cells may one replica serve" is a property of the
    kind, not of the lane.  With no live endpoint the throttle is 1, not 0: one cell may run
    and will exit 2 (no endpoint), which is how the driver learns the fleet is down instead
    of silently queueing nothing.
    """
    per_endpoint = _throttle_rate(lane, caps, kind)
    return max(1, int(per_endpoint) * max(int(live_endpoints), 0)) if live_endpoints else 1


def _throttle_rate(lane: Lane | str, caps: Caps, kind: Any) -> int:
    table = caps.throttle_per_endpoint
    if kind is not None:
        name = getattr(kind, "value", str(kind))
        if name in table:
            return int(table[name])
    lane_name = lane.value if isinstance(lane, Lane) else str(lane)
    if lane_name == Lane.AUDIT.value:
        return int(table.get(kinds.CellKind.SELECTION_AUDIT.value, 1))
    return int(table.get(kinds.CellKind.EPISODE.value, 1))


def set_array_throttle(job_id: str, n: int, *, runner: Any = None) -> bool:
    """``scontrol update JobId=<array> ArrayTaskThrottle=N`` (study_v4's live-throttle fix)."""
    out = (runner or _sh)(["scontrol", "update", f"JobId={job_id}", f"ArrayTaskThrottle={int(n)}"])
    return "error" not in out.lower()


# --------------------------------------------------------------------------- chunk state


@dataclass(frozen=True)
class ChunkState:
    """What one pass found on disk, per cell index."""

    done: tuple[int, ...]
    suspended: tuple[int, ...]
    waiting: tuple[int, ...]
    blocked: tuple[int, ...]
    reused: tuple[int, ...]
    todo: tuple[int, ...]

    @property
    def n_terminal(self) -> int:
        return len(self.done) + len(self.suspended) + len(self.reused)


def chunk_state(run_root: str | os.PathLike, cells: Sequence[CellSpec]) -> ChunkState:
    """Classify every cell of a manifest by what is on disk, never by what was submitted.

    ``DEPS_UNMET.json`` is *waiting*, not failed (its dependency may have landed since) and
    ``SUSPENDED.json`` is terminal until a human removes it -- re-queueing a suspended cell is
    how ARD36 turned one harness defect into thousands of wasted array tasks.
    """
    root = Path(run_root)
    done, suspended, waiting, blocked, reused, todo = [], [], [], [], [], []
    for index, cell in enumerate(cells):
        if cell.reuse_of:
            reused.append(index)
            continue
        directory = deps.cell_dir(root, cell.cell_id)
        if (directory / deps.META_FILE).exists():
            done.append(index)
            continue
        if (directory / deps.SUSPENDED_FILE).exists():
            suspended.append(index)
            continue
        if (directory / DEPS_UNMET_FILE).exists():
            waiting.append(index)
        if deps.unmet(root, cell.depends_on):
            blocked.append(index)
            continue
        todo.append(index)
    return ChunkState(done=tuple(done), suspended=tuple(suspended), waiting=tuple(waiting),
                      blocked=tuple(blocked), reused=tuple(reused), todo=tuple(todo))


# --------------------------------------------------------------------------- rendering


def render_array(template: Path, *, run_root: Path, manifest: Path, lane: Lane | str,
                 run_id: str, indices: Sequence[int], throttle: int, partition: str,
                 cpus: int, mem: str, time_limit: str, log_dir: Path, exclude: Sequence[str],
                 signal_s: int, config: Path) -> str:
    """Render one array over exactly ``indices`` (sparse when the resume is partial)."""
    text = template.read_text(encoding="utf-8")
    array = ",".join(str(i) for i in indices) if indices else str(0)
    exclude_line = f"#SBATCH --exclude={','.join(exclude)}\n" if exclude else ""
    values = {
        "JOB_NAME": job_name(run_id, lane), "PARTITION": partition, "CPUS": str(int(cpus)),
        "MEM": str(mem), "TIME": str(time_limit), "ARRAY": array, "THROTTLE": str(int(throttle)),
        "LOG_DIR": str(log_dir), "SIGNAL_SECONDS": str(int(signal_s)),
        "EXCLUDE_LINE": exclude_line, "REPO": str(REPO_ROOT), "RUN_ROOT": str(run_root),
        "MANIFEST": str(manifest), "LANE": lane.value if isinstance(lane, Lane) else str(lane),
        "CONFIG": str(config), "EXIT_DEPS_UNMET": str(EXIT_DEPS_UNMET),
    }
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unrendered = sorted(set(re.findall(r"\{([A-Z_]+)\}", text)))
    if unrendered:
        raise ManifestError(f"{template.name}: unrendered placeholder(s) {unrendered}")
    return text


def contiguous_runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    """``[3, 4, 5, 9]`` -> ``[(3, 5), (9, 9)]``.

    The capture job walks an inclusive index *range* rather than a Slurm array (one job, one
    GPU, cells run in sequence), so a chunk whose resume is sparse becomes one job per run
    instead of one job that would re-drive the cells in the gaps.
    """
    runs: list[tuple[int, int]] = []
    for index in sorted(set(int(i) for i in indices)):
        if runs and index == runs[-1][-1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def capture_target(compute: Any, *, runner: Any = None) -> tuple[str, str, int]:
    """``(partition, gpu_type, gpus_per_job)`` for a capture job, with the §3.2 fallback.

    The primary lane is ``compute.capture`` (``ou_bcs_low`` A100).  When it has no idle GPU of
    that type the job goes to ``compute.capture.fallback`` -- which is what
    :func:`capture_fallback` is for and what, until now, nothing called.
    """
    from agents_scaling.nextdyn import fleet as FL

    gpus = int(compute.capture.gpus_per_job)
    primary = (str(compute.capture.partition), str(compute.capture.gpu_type))
    fallback_partition, fallback_types = capture_fallback(compute)
    candidates = [primary] + [(fallback_partition, str(gpu)) for gpu in fallback_types]
    for partition, gpu_type in candidates:
        census = FL.free_gpus(partition, gpu_type, exclude=compute.serving.exclude_nodes,
                              runner=runner)
        if census.free >= gpus:
            return partition, gpu_type, gpus
    return (*primary, gpus)                # nothing idle anywhere: queue on the primary lane


def render_capture(template: Path, *, run_root: Path, manifest: Path, run_id: str,
                   first: int, last: int, partition: str, gpu_type: str, gpus_per_job: int,
                   cpus: int, mem: str, time_limit: str, log_dir: Path) -> str:
    """Render one capture job over the inclusive cell-index range ``first..last``.

    Separate from :func:`render_array` because the capture template is not an array: it is one
    job holding one GPU that calls ``run_one`` per index, and its placeholder set is its own
    (``GPU_TYPE`` / ``GPUS_PER_JOB`` / ``FIRST_INDEX`` / ``LAST_INDEX``, no ``ARRAY`` and no
    throttle).  The unrendered-placeholder refusal is the same one.
    """
    text = template.read_text(encoding="utf-8")
    values = {
        "RUN_ID": str(run_id), "PARTITION": str(partition), "GPU_TYPE": str(gpu_type),
        "GPUS_PER_JOB": str(int(gpus_per_job)), "CPUS": str(int(cpus)), "MEM": str(mem),
        "TIME": str(time_limit), "LOG_DIR": str(log_dir), "REPO": str(REPO_ROOT),
        "RUN_ROOT": str(run_root), "MANIFEST": str(manifest),
        "FIRST_INDEX": str(int(first)), "LAST_INDEX": str(int(last)),
    }
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unrendered = sorted(set(re.findall(r"\{([A-Z_]+)\}", text)))
    if unrendered:
        raise ManifestError(f"{template.name}: unrendered placeholder(s) {unrendered}")
    return text


def sparse_resume(text: str, indices: Sequence[int], throttle: int | None = None) -> str:
    """Rewrite an array line over exactly ``indices`` (ARD36's ``--array=3,7%24`` top-up)."""
    if not indices:
        raise ManifestError("a sparse resume needs at least one index")
    if throttle is None:
        match = THROTTLE_RE.search(text)
        throttle = int(match.group(1)) if match else len(indices)
    return ARRAY_RE.sub(f"#SBATCH --array={','.join(str(i) for i in indices)}%{int(throttle)}",
                        text, count=1)


# --------------------------------------------------------------------------- the pass


@dataclass(frozen=True)
class PassReport:
    """What one driver pass did (spec 17.5's five counters plus what it saw)."""

    lane: str
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    reused: int = 0
    submitted: tuple[str, ...] = ()
    throttle: int = 1
    detail: str = ""

    def as_report(self) -> cli.Report:
        return cli.Report(completed=self.completed, failed=self.failed, skipped=self.skipped,
                          blocked=self.blocked, reused=self.reused, detail=self.detail)


@dataclass
class Submitter:
    """``sbatch`` and the chunk log, injected so a test drives a whole pass with no cluster."""

    run_root: Path
    lane: str
    dry_run: bool = False
    submitted: list[dict[str, Any]] = field(default_factory=list)
    runner: Any = None

    @property
    def log_path(self) -> Path:
        return self.run_root / f"{CHUNK_LOG}_{self.lane}.json"

    def load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def record(self, entry: Mapping[str, Any]) -> None:
        on_disk = self.load()                            # merge: another driver may have written
        on_disk.append(dict(entry))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(on_disk, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")

    def submit(self, path: Path) -> str | None:
        if self.dry_run:
            return None
        out = (self.runner or _sh)(["sbatch", "--parsable", str(path)])
        return out.strip().split(";")[0] if out.strip() else None


def admit_chunks(state: ChunkState, *, caps: Caps, lane_count: int, total_count: int,
                 chunk_size: int, banks_only: bool = False) -> tuple[list[tuple[int, ...]], str]:
    """The two gates, then the chunks this pass may submit (plan §3.3).

    Returns ``(chunks, reason)``; an empty list with a reason is a *decision*, not a failure:
    the next pass will try again when the queue has drained.
    """
    todo = list(state.todo)
    if not todo:
        return [], "nothing to dispatch"
    lane_cap = caps.lane_cap_for(banks_only=banks_only)
    lane_headroom = lane_cap - int(lane_count)
    global_headroom = min(caps.global_submit_cap, caps.association_cap) - int(total_count)
    room = max(0, min(lane_headroom, global_headroom))
    if room <= 0:
        return [], (f"no headroom: lane {lane_count}/{lane_cap}, global "
                    f"{total_count}/{min(caps.global_submit_cap, caps.association_cap)}")
    admitted = todo[:room]
    chunks = [tuple(admitted[i:i + chunk_size]) for i in range(0, len(admitted), chunk_size)]
    reason = (f"admitted {len(admitted)}/{len(todo)} cells in {len(chunks)} chunk(s); "
              f"lane {lane_count}/{lane_cap}, global {total_count}/"
              f"{min(caps.global_submit_cap, caps.association_cap)}")
    return chunks, reason


def live_owned_indices(submitter: Submitter, manifest: Path, *, runner: Any = None) -> set[int]:
    """Cell indices a live job of *this* manifest already owns (never double-dispatch).

    An array job reports its live task ids and they *are* the cell indices.  The capture lane's
    job is not an array (one job, one GPU, an inclusive index range), so ``%K`` is ``N/A`` there
    and the indices it owns are the ones the chunk log recorded for it -- for as long as the job
    itself is live.  Without that fallback every capture pass would re-submit cells a running
    job already holds.
    """
    owned: set[int] = set()
    wanted = manifest.name
    for entry in submitter.load():
        if Path(str(entry.get("manifest", ""))).name != wanted:
            continue
        job_id = entry.get("job_id")
        if not job_id:
            continue
        tasks = array_indices_live(str(job_id), runner=runner)
        if tasks:
            owned |= tasks
        elif job_is_live(str(job_id), runner=runner):
            owned |= {int(i) for i in entry.get("indices", ())}
    return owned


def lane_pass(run_root: str | os.PathLike, lane: Lane | str, *, manifests: Sequence[Path],
              caps: Caps, live_endpoints: int, cfg: Any, compute: Any,
              run_id: str | None = None, dry_run: bool = False, squeue: Any = None,
              submitter: Any = None) -> PassReport:
    """One driver pass over every manifest of one lane (module docstring, steps 1-5)."""
    root = Path(run_root)
    lane_name = lane.value if isinstance(lane, Lane) else str(lane)
    run_id = run_id or root.name
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    counters = {"completed": 0, "failed": 0, "skipped": 0, "blocked": 0, "reused": 0}
    submitted: list[str] = []
    details: list[str] = []
    lane_count = lane_task_count(run_id, lane_name, runner=squeue)
    total_count = total_task_count(runner=squeue)
    throttle = throttle_for(lane_name, live_endpoints, caps=caps)

    for manifest in manifests:
        cells = CE.load_manifest(manifest)
        mine = [(index, cell) for index, cell in enumerate(cells)
                if cell.lane.value == lane_name]
        if not mine:
            continue
        state = chunk_state(root, [cell for _index, cell in mine])
        offset = {position: index for position, (index, _cell) in enumerate(mine)}
        counters["completed"] += len(state.done)
        counters["failed"] += len(state.suspended)
        counters["blocked"] += len(state.blocked)
        counters["reused"] += len(state.reused)
        sub = submitter or Submitter(run_root=root, lane=lane_name, dry_run=dry_run)
        owned = live_owned_indices(sub, manifest, runner=squeue)
        free = ChunkState(done=state.done, suspended=state.suspended, waiting=state.waiting,
                          blocked=state.blocked, reused=state.reused,
                          todo=tuple(p for p in state.todo if offset[p] not in owned))
        counters["skipped"] += len(state.todo) - len(free.todo)
        banks_only = all(kinds.is_bank(cell.kind) for _index, cell in mine)
        chunks, reason = admit_chunks(free, caps=caps, lane_count=lane_count,
                                      total_count=total_count, chunk_size=caps.chunk_size,
                                      banks_only=banks_only)
        details.append(f"{manifest.name}: {reason}")
        for chunk in chunks:
            indices = [offset[p] for p in chunk]
            cell = cells[indices[0]]
            shape = _shape(cell, compute, cfg)
            for text, spooled, owned_indices in _render_jobs(
                    cell, root=root, manifest=manifest, lane_name=lane_name, run_id=run_id,
                    indices=indices, throttle=throttle, shape=shape, compute=compute, cfg=cfg,
                    log_dir=log_dir, lane_count=lane_count, squeue=squeue):
                spooled.write_text(text, encoding="utf-8")
                job_id = sub.submit(spooled)
                sub.record({"manifest": str(manifest), "lane": lane_name,
                            "indices": list(owned_indices), "job_id": job_id, "at": time.time(),
                            "throttle": throttle})
                if job_id:
                    submitted.append(job_id)
            lane_count += len(indices)
            total_count += len(indices)
        for entry in sub.load():
            if entry.get("job_id") and job_is_live(str(entry["job_id"]), runner=squeue):
                set_array_throttle(str(entry["job_id"]), throttle, runner=squeue)
    return PassReport(lane=lane_name, completed=counters["completed"], failed=counters["failed"],
                      skipped=counters["skipped"], blocked=counters["blocked"],
                      reused=counters["reused"], submitted=tuple(submitted), throttle=throttle,
                      detail="; ".join(details))


def _render_jobs(cell: CellSpec, *, root: Path, manifest: Path, lane_name: str, run_id: str,
                 indices: Sequence[int], throttle: int, shape: Any, compute: Any, cfg: Any,
                 log_dir: Path, lane_count: int, squeue: Any = None
                 ) -> list[tuple[str, Path, tuple[int, ...]]]:
    """``[(script text, where to spool it, the indices it owns)]`` for one admitted chunk.

    One entry for every lane but CAPTURE, which yields one job per contiguous index run.
    """
    template = _template_for(cell)
    if cell.lane is Lane.CAPTURE:
        partition, gpu_type, gpus = capture_target(compute, runner=squeue)
        out = []
        for first, last in contiguous_runs(indices):
            text = render_capture(
                template, run_root=root, manifest=manifest, run_id=run_id, first=first,
                last=last, partition=partition, gpu_type=gpu_type, gpus_per_job=gpus,
                cpus=shape.cpus, mem=shape.mem, time_limit=shape.time, log_dir=log_dir)
            out.append((text, root / f"chunk_{lane_name}_{manifest.stem}_{first}.sbatch",
                        tuple(range(first, last + 1))))
        return out
    text = render_array(
        template, run_root=root, manifest=manifest, lane=lane_name, run_id=run_id,
        indices=indices, throttle=throttle,
        partition=worker_partition(cell, compute, queued=lane_count),
        cpus=shape.cpus, mem=shape.mem, time_limit=shape.time, log_dir=log_dir,
        exclude=compute.serving.exclude_nodes,
        signal_s=compute.dispatch.signal_seconds_before_walltime,
        config=Path(getattr(cfg, "path", C.DEFAULT_STUDY_CONFIG)))
    return [(text, root / f"chunk_{lane_name}_{manifest.stem}_{indices[0]}.sbatch",
             tuple(indices))]


def _template_for(cell: CellSpec) -> Path:
    """The sbatch template this cell's lane runs on.

    CAPTURE is not the cell array: that template allocates no GPU, so a capture cell rendered
    from it would land on a GPU partition with zero GPUs and the replay engine would fail on
    its first ``torch.cuda`` call.
    """
    if cell.lane is Lane.CAPTURE:
        return REPO_ROOT / SLURM_DIR / CAPTURE_TMPL
    template = CPU_ARRAY_TMPL if cell.lane is Lane.CPU else CELL_ARRAY_TMPL
    return REPO_ROOT / SLURM_DIR / template


def _shape(cell: CellSpec, compute: Any, cfg: Any) -> Any:
    contract = cfg.contract
    model = None if cell.model is None else contract.model(cell.model)
    return kinds.cell_shape(cell.kind, compute=compute, model=model,
                            thinking_cap=contract.tokens.thinking_cap, team=contract.team)


@dataclass(frozen=True)
class PartitionLimits:
    """What one partition grants a user: CPUs, memory and submitted jobs (``compute.yaml``).

    The GPU caps are the fleet's business; these three are the *worker* lane's.  ARD36 queued
    cell workers against ``mit_normal``'s 96-CPU / 386 G per-user ceiling without counting it,
    and the surplus sat PENDING for hours behind a reason nobody was reading.
    """

    partition: str
    cpus: int | None
    mem_gb: int | None
    max_submit: int
    overflow: str | None

    def workers(self, *, cpus_per_worker: int, mem_gb_per_worker: float) -> int:
        """How many cell workers this partition can hold at once."""
        limits = [self.max_submit]
        if self.cpus is not None:
            limits.append(int(self.cpus) // max(int(cpus_per_worker), 1))
        if self.mem_gb is not None:
            limits.append(int(float(self.mem_gb) / max(float(mem_gb_per_worker), 1.0)))
        return max(min(limits), 0)


def partition_limits(compute: Any, partition: str, *, deadline: bool = True) -> PartitionLimits:
    block = compute.partitions.get(partition)
    worker = compute.workers.deadline if deadline else compute.workers.loss_tolerant
    if block is None:
        return PartitionLimits(partition=partition, cpus=None, mem_gb=None,
                               max_submit=int(compute.association.max_submit_jobs),
                               overflow=worker.overflow_partition)
    return PartitionLimits(partition=partition, cpus=block.cpu, mem_gb=block.mem_gb,
                           max_submit=int(block.max_submit),
                           overflow=worker.overflow_partition)


def worker_partition(cell: CellSpec, compute: Any, *, queued: int = 0) -> str:
    """Where this cell's *worker* runs, with the overflow partition when the first is full.

    Plan §3.3: deadline cell workers live on the non-preemptable CPU partition and overflow to
    the preemptable one when the deadline fleet outgrows it -- which risks a re-dispatch, never
    data, because every item commits atomically.
    """
    spec = kinds.spec_for(cell.kind)
    primary = _partition_for(cell, compute)
    limits = partition_limits(compute, primary, deadline=spec.deadline_bearing)
    worker = compute.workers.deadline if spec.deadline_bearing else compute.workers.loss_tolerant
    room = limits.workers(cpus_per_worker=int(worker.cpus),
                          mem_gb_per_worker=_mem_gb(worker.mem))
    if queued < room or not limits.overflow:
        return primary
    return str(limits.overflow)


#: What a gigabyte *is*, not a quantity anyone schedules on: a unit conversion, exempt from the
#: "no dispatch number in code" guard by its ``_PER_`` name (plan §3.3).
MEGABYTES_PER_GIGABYTE: Final[float] = 1024.0


def _mem_gb(text: str) -> float:
    """``3G`` / ``512M`` -> gigabytes (the Slurm memory spelling ``compute.yaml`` uses)."""
    raw = str(text).strip().upper()
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == ".") or "1"
    value = float(digits)
    return value / MEGABYTES_PER_GIGABYTE if raw.endswith("M") else value


def capture_fallback(compute: Any) -> tuple[str, tuple[str, ...]]:
    """Where a capture cell goes when its own partition has no idle GPU (plan §3.2)."""
    fallback = compute.capture.fallback
    return str(fallback.partition), tuple(fallback.gpu_types)


def _partition_for(cell: CellSpec, compute: Any) -> str:
    """Where the *worker* runs (never where the server runs): a deadline lane's workers go to
    the non-preemptable worker partition and overflow to the preemptable one (plan §3.3)."""
    spec = kinds.spec_for(cell.kind)
    if not spec.needs_endpoint:
        block = compute.cells.kinds[cell.kind.value]
        return compute.capture.partition if cell.lane is Lane.CAPTURE else str(block.partition)
    worker = compute.workers.deadline if spec.deadline_bearing else compute.workers.loss_tolerant
    return worker.partition


# --------------------------------------------------------------------------- verbs


def _manifests_of(run_root: Path) -> list[Path]:
    directory = run_root.joinpath(*CE.MANIFESTS_DIR)
    return sorted(p for p in directory.glob("*.json")) if directory.exists() else []


def drive_main(argv: Sequence[str] | None = None) -> cli.Report:
    """``drive``: one lane, one pass (or a loop) -- the entry the lane-driver sbatch calls."""
    parser = cli.common_parser("drive", description="one lane driver pass")
    parser.add_argument("--lane", required=True, choices=[l.value for l in Lane])
    parser.add_argument("--passes", type=int, default=None, help="repeat this many passes")
    parser.add_argument("--dry-run", action="store_true", help="render sbatches, submit nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    caps = Caps.of(compute, args.lane)
    manifests = [Path(args.manifest)] if args.manifest else _manifests_of(run_root)
    if not manifests:
        return cli.Report(blocked=1, detail=f"no block manifests under {run_root}")
    from agents_scaling.nextdyn import fleet as FL

    report = None
    n_passes = int(args.passes) if args.passes else 1
    for index in range(n_passes):
        live = FL.live_endpoint_count(run_root, compute=compute)
        report = lane_pass(run_root, args.lane, manifests=manifests, caps=caps,
                           live_endpoints=live, cfg=cfg, compute=compute, dry_run=args.dry_run)
        print(f"[drive:{args.lane}] pass {index}: {report.as_report().line()} "
              f"throttle={report.throttle} {report.detail}")
        if index + 1 < n_passes:
            time.sleep(float(compute.dispatch.lane_driver_pass_s))
    return report.as_report() if report is not None else cli.Report(blocked=1)


def run_main(argv: Sequence[str] | None = None) -> cli.Report:
    """``run``: submit the planner, one driver per lane and the status job (plan §3.5 S4).

    It submits *control* jobs only; the cell arrays are the drivers' business.  Re-running is
    safe: a driver whose exact job name is already in the queue is not submitted again, which
    is what keeps a crashed-and-restarted ``run`` from duplicating a chain
    (``test_run_resubmit_does_not_duplicate_queued_arrays``).
    """
    parser = cli.common_parser("run", description="submit the planner, lane drivers and status")
    parser.add_argument("--lanes", default=None, help="comma list (default: every lane)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    from agents_scaling.nextdyn import freeze as FREEZE

    if cfg.study.mode == K.MODE_CONFIRMATION and not FREEZE.is_frozen(run_root):
        return cli.Report(blocked=1, detail=f"{run_root} is not frozen; run `freeze` first")
    lanes = [Lane(name) for name in (args.lanes.split(",") if args.lanes else
                                     [l.value for l in Lane])]
    run_id = run_root.name
    queued = _queued_job_names()
    submitted: list[str] = []
    skipped: list[str] = []
    for name, template, extra in _control_jobs(run_id, lanes):
        if name in queued:
            skipped.append(name)
            continue
        path = _render_control(run_root, cfg, compute, name=name, template=template, extra=extra)
        if args.dry_run:
            submitted.append(f"{name} (dry run -> {path.name})")
            continue
        out = _sh(["sbatch", "--parsable", str(path)])
        submitted.append(f"{name}={out.strip().split(';')[0]}" if out.strip() else f"{name}=?")
    return cli.Report(completed=len(submitted), skipped=len(skipped),
                      detail=f"submitted {submitted}; already queued {skipped}")


def _control_jobs(run_id: str, lanes: Sequence[Lane]) -> list[tuple[str, str, dict[str, str]]]:
    jobs = [(planner_job_name(run_id), PLANNER_TMPL, {}),
            (status_job_name(run_id), STATUS_TMPL, {})]
    jobs += [(driver_job_name(run_id, lane), LANE_DRIVER_TMPL, {"LANE": lane.value})
             for lane in lanes]
    return jobs


def _queued_job_names(*, runner: Any = None) -> set[str]:
    out = (runner or _sh)(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"])
    return {line.strip() for line in out.splitlines() if line.strip()}


def _render_control(run_root: Path, cfg: Any, compute: Any, *, name: str, template: str,
                    extra: Mapping[str, str]) -> Path:
    """Render a control job (planner / driver / status) into the run root."""
    text = (REPO_ROOT / SLURM_DIR / template).read_text(encoding="utf-8")
    keepalive = compute.keepalive
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "JOB_NAME": name, "PARTITION": ",".join(keepalive.partitions),
        "CPUS": str(int(keepalive.cpus)), "MEM": str(keepalive.mem), "TIME": str(keepalive.time),
        "LOG_DIR": str(log_dir), "REPO": str(REPO_ROOT), "RUN_ROOT": str(run_root),
        "CONFIG": str(getattr(cfg, "path", C.DEFAULT_STUDY_CONFIG)),
        "INTERVAL_S": str(int(compute.dispatch.lane_driver_pass_s)),
        "PLANNER_INTERVAL_S": str(int(compute.dispatch.planner_pass_s)),
        "CPU_INTERVAL_S": str(int(compute.dispatch.cpu_driver_pass_s)),
        "SELF": str(run_root / f"{name}.sbatch"), "LANE": "",
        "PASSES": str(_passes(keepalive.time, compute.dispatch.lane_driver_pass_s)),
        **dict(extra),
    }
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    unrendered = sorted(set(re.findall(r"\{([A-Z_]+)\}", text)))
    if unrendered:
        raise ManifestError(f"{template}: unrendered placeholder(s) {unrendered}")
    path = run_root / f"{name}.sbatch"
    path.write_text(text, encoding="utf-8")
    return path


def _passes(walltime: str, interval_s: Any) -> int:
    """How many passes fit in one driver job's walltime (the job then hands the baton on)."""
    from agents_scaling.nextdyn import workload as WL

    return max(1, int(WL.walltime_seconds(walltime) // max(float(interval_s), 1.0)))


def main(argv: Sequence[str] | None = None) -> cli.Report:    # pragma: no cover - alias
    return drive_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli.main(["drive", *sys.argv[1:]]))


__all__ = [
    "ARRAY_RE", "CAPTURE_TMPL", "CELL_ARRAY_TMPL", "CHUNK_LOG", "CPU_ARRAY_TMPL", "Caps",
    "ChunkState",
    "JOB_PREFIX", "LANE_DRIVER_TMPL", "LIVE_STATES", "PLANNER_TMPL", "PassReport", "STATUS_TMPL",
    "Submitter", "admit_chunks", "array_indices_live", "capture_fallback", "capture_target",
    "chunk_state", "contiguous_runs", "drive_main",
    "driver_job_name", "job_is_live", "job_name", "lane_pass", "lane_task_count",
    "live_owned_indices", "planner_job_name", "render_array", "render_capture", "run_main",
    "set_array_throttle",
    "sparse_resume", "status_job_name", "throttle_for", "total_task_count",
]
