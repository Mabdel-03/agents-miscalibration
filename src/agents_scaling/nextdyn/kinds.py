"""What each cell kind is: worker, lane, pool, deadline bearing, dependency prefixes (plan §3.3).

The one table a dispatcher asks "does this kind need a live endpoint, which lane and pool
does it use, what may it depend on, and what shape is one cell".  Two things differ from
``ard/kinds.py`` (its pattern is kept):

* **No number lives here.**  Items per cell, parallelism, in-flight, CPU/mem/time all come
  from ``configs/nextdyn/compute.yaml`` through :func:`cell_shape`, and
  ``test_no_numeric_dispatch_constant_lives_in_code`` pins it.  ``PRIORITY`` is an *order*
  (a tuple of kinds), not a set of integers.
* **Pool is a function of (kind, model, compute config)**, not a field: the same RECEIVER
  kind runs on the low pool for both models, and deadline-bearing kinds are refused a
  preemptable pool at load time (``test_deadline_bearing_kinds_never_get_a_low_pool``).

``KindSpec`` fields and their consumers (rule R2): :data:`KIND_FIELD_CONSUMERS` names, per
field, the *package module* that reads it, and
``test_every_kindspec_field_is_read_outside_kinds_py`` holds that module to it once it has
landed (pending until then; tests never count as readers).  A field nobody in the package
reads is the D-03 shape and is refused.

=================  ==================================================================
field              consumed by (package module)
=================  ==================================================================
``kind``           ``runner`` (``resolve_worker`` keys the worker on it)
``worker``         ``runner`` (``"module:callable"``, imported lazily)
``needs_endpoint`` ``run_one`` (open a pool or not)
``lane``           ``dispatch`` (lane drivers; ``None`` = follows the model's lane)
``deadline_bearing`` ``dispatch`` (never a preemptable pool / endpoint)
``dep_prefixes``   ``cells`` (which dependency kinds a cell of this kind may carry)
``doc``            ``plan`` (``--dry-run`` listing)
=================  ==================================================================

The WP that owns each worker is the ownership table in ``docs/nextdyn/00_contract.md`` §1,
not a field here (a field only tests read is not consumed).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn.config import CellBlock, ComputeConfig, ReferenceShape
from agents_scaling.nextdyn.errors import ConfigError, ManifestError
from agents_scaling.nextdyn.ids import BANK_KINDS, GENERATION_KINDS, CellKind, coerce_kind


class Lane(str, Enum):
    """Lane drivers (plan §3.5): one self-resubmitting Slurm job per lane."""

    SMALL = "8B"
    LARGE = "32B"
    AUDIT = "audit"
    CPU = "cpu"
    CAPTURE = "capture"


class Pool(str, Enum):
    """Server pools = sub-run-roots ``$RUN/pools/<pool>/servers`` (plan §3.2)."""

    DEADLINE = "deadline"
    LOW = "low"
    AUDIT = "audit"


_W = "agents_scaling.nextdyn."


@dataclass(frozen=True)
class KindSpec:
    kind: CellKind
    worker: str
    needs_endpoint: bool
    lane: Lane | None
    deadline_bearing: bool
    dep_prefixes: tuple[str, ...]
    doc: str

    def __post_init__(self) -> None:
        module, sep, attr = self.worker.partition(":")
        if not sep or not module.startswith(_W) or not attr:
            raise ManifestError(f"{self.kind.value}: worker must be 'agents_scaling.nextdyn.<module>:<callable>'")
        for prefix in self.dep_prefixes:
            if prefix not in deps.PREFIXES:
                raise ManifestError(f"{self.kind.value}: dependency prefix {prefix!r} not in {deps.PREFIXES}")
        if self.needs_endpoint and self.kind not in GENERATION_KINDS:
            raise ManifestError(f"{self.kind.value}: only generation kinds need an endpoint")


#: field -> the package module that reads it (R2; see the module docstring).
KIND_FIELD_CONSUMERS: Final[Mapping[str, str]] = {
    "kind": _W + "runner",
    "worker": _W + "runner",
    "needs_endpoint": _W + "run_one",
    "lane": _W + "dispatch",
    "deadline_bearing": _W + "dispatch",
    "dep_prefixes": _W + "cells",
    "doc": _W + "plan",
}

KIND_TABLE: Final[Mapping[CellKind, KindSpec]] = {
    CellKind.DIFFICULTY_BANK: KindSpec(
        kind=CellKind.DIFFICULTY_BANK, worker=_W + "banks:bank_worker", needs_endpoint=True, lane=None,
        deadline_bearing=False, dep_prefixes=(),
        doc="Fixed-count difficulty draws per (item, model, budget); never a donor or a root (spec 5.3)."),
    CellKind.DONOR_BANK: KindSpec(
        kind=CellKind.DONOR_BANK, worker=_W + "banks:bank_worker", needs_endpoint=True, lane=None,
        deadline_bearing=False, dep_prefixes=(deps.CELL_PREFIX,),
        doc="Donor draws for constructed teams; the extension stage depends on its GRADE cell."),
    CellKind.NATURAL_ROOT: KindSpec(
        kind=CellKind.NATURAL_ROOT, worker=_W + "banks:bank_worker", needs_endpoint=True, lane=None,
        deadline_bearing=False, dep_prefixes=(),
        doc="Natural root draws on natural-cohort items (spec 8.1)."),
    CellKind.EPISODE: KindSpec(
        kind=CellKind.EPISODE, worker=_W + "episode:episode_worker", needs_endpoint=True, lane=None,
        deadline_bearing=True, dep_prefixes=(deps.BUNDLE_PREFIX, deps.CELL_PREFIX),
        doc="R synchronous revisions of one VisibilitySpec on one root team (E1, E3a, bridge, protection)."),
    CellKind.RECEIVER: KindSpec(
        kind=CellKind.RECEIVER, worker=_W + "receiver:receiver_worker", needs_endpoint=True, lane=None,
        deadline_bearing=False, dep_prefixes=(deps.REGISTER_PREFIX,),
        doc="One-call (E2a) or two-phase (E2b) receiver episodes over fixed packets."),
    CellKind.SELECTION_AUDIT: KindSpec(
        kind=CellKind.SELECTION_AUDIT, worker=_W + "audit:audit_worker", needs_endpoint=True, lane=Lane.AUDIT,
        deadline_bearing=False, dep_prefixes=(deps.REGISTER_PREFIX,),
        doc="Pointwise audit of a sealed pool: one draw per unique packet (spec 6.5)."),
    CellKind.GRADE: KindSpec(
        kind=CellKind.GRADE, worker=_W + "grading:grade_worker", needs_endpoint=False, lane=Lane.CPU,
        deadline_bearing=False, dep_prefixes=(deps.CELL_PREFIX,),
        doc="Tri-state grading of committed answers into grades_private (writer side)."),
    CellKind.CONSTRUCT: KindSpec(
        kind=CellKind.CONSTRUCT, worker=_W + "construct:construct_worker", needs_endpoint=False, lane=Lane.CPU,
        deadline_bearing=False, dep_prefixes=(deps.CELL_PREFIX,),
        doc="Offline composition-driven team builder and label-free bundle export."),
    CellKind.CAPTURE: KindSpec(
        kind=CellKind.CAPTURE, worker=_W + "capture:capture_worker", needs_endpoint=False, lane=Lane.CAPTURE,
        deadline_bearing=True, dep_prefixes=(deps.REGISTER_PREFIX, deps.CELL_PREFIX),
        doc="Read-only teacher-forced replay capture of a sealed panel."),
    CellKind.E0: KindSpec(
        kind=CellKind.E0, worker=_W + "e0.main:e0_worker", needs_endpoint=False, lane=Lane.CPU,
        deadline_bearing=False, dep_prefixes=(),
        doc="Legacy reanalysis (ARD36 tables_v2, study_v4, Run0) and BCB behavioral repair."),
    CellKind.MATERIALIZE: KindSpec(
        kind=CellKind.MATERIALIZE, worker=_W + "materialize:materialize_worker", needs_endpoint=False,
        lane=Lane.CPU, deadline_bearing=False, dep_prefixes=(deps.CELL_PREFIX,),
        doc="Atomic parquet publish with key and referential assertions."),
}

if set(KIND_TABLE) != set(CellKind):  # pragma: no cover - import-time contract
    raise RuntimeError("KIND_TABLE must cover every CellKind")

#: Kinds that must never run against a preemptable endpoint (plan §3.3, D20).
DEADLINE_BEARING: Final[frozenset[CellKind]] = frozenset(k for k, s in KIND_TABLE.items() if s.deadline_bearing)

#: Dispatch order inside one lane pass: earlier first (plan §3.4: banks and natural roots at
#: t0, then everything that unblocks the DAG, then close-out).
PRIORITY: Final[tuple[CellKind, ...]] = (
    CellKind.DIFFICULTY_BANK, CellKind.DONOR_BANK, CellKind.NATURAL_ROOT, CellKind.GRADE,
    CellKind.CONSTRUCT, CellKind.EPISODE, CellKind.RECEIVER, CellKind.SELECTION_AUDIT,
    CellKind.CAPTURE, CellKind.E0, CellKind.MATERIALIZE,
)
if set(PRIORITY) != set(CellKind) or len(PRIORITY) != len(CellKind):  # pragma: no cover
    raise RuntimeError("PRIORITY must order every CellKind exactly once")

#: Spec 17.2 names -> (CellKind, phase).  ``revision`` is an EPISODE row at
#: ``phase="revision"``; the two receiver phases are RECEIVER rows; ``selection`` is the
#: audit kind; ``activation_replay`` is CAPTURE (plan D8).
CELL_KIND_COMPAT: Final[Mapping[str, tuple[CellKind, str | None]]] = {
    "difficulty_bank": (CellKind.DIFFICULTY_BANK, K.PHASE_BANK),
    "donor_bank": (CellKind.DONOR_BANK, K.PHASE_BANK),
    "natural_root": (CellKind.NATURAL_ROOT, K.PHASE_BANK),
    "revision": (CellKind.EPISODE, K.PHASE_REVISION),
    "receiver_plan": (CellKind.RECEIVER, K.PHASE_RECEIVER_PLAN),
    "receiver_decision": (CellKind.RECEIVER, K.PHASE_RECEIVER_DECISION),
    "selection": (CellKind.SELECTION_AUDIT, K.PHASE_AUDIT),
    "activation_replay": (CellKind.CAPTURE, None),
}


def batch_width(runner: Any, n_planned: int) -> int:
    """How many of one item's exchangeable requests may be in flight at once.

    Here rather than in :mod:`runner`, which is where ``Concurrency`` is compiled, because the
    SELECTION_AUDIT worker needs this too and ``runner`` reaches ``construct``: importing it
    from there widened the audit worker's import closure onto the gold-derived side and the R7
    firewall refused it, correctly.  ``KIND_TABLE`` below is where ``max_inflight`` is
    *declared*, so its reader belongs beside it (R1) and the module stays gold-free.

    The fallback is deliberately 1 rather than a guessed width: a runner that cannot say what
    it allows gets the old serial behaviour, visibly, instead of a number nobody chose.  That
    is the D-03 shape -- a declared knob that does not govern its own subject -- and it had
    already cost this package the bank and audit lanes once.
    """
    concurrency = getattr(runner, "concurrency", None)
    width = int(getattr(concurrency, "max_inflight", 1) or 1)
    return max(1, min(width, max(1, int(n_planned))))


def phases_of(kind: CellKind | str) -> tuple[str, ...]:
    """The ``decisions.phase`` values cells of this kind write, read off ``CELL_KIND_COMPAT``.

    Derived rather than declared again (R1): that map is already the one statement of which
    kind writes which phase, and a second list would be the thing that drifts.  A kind that
    writes no generating phase (CAPTURE, GRADE, …) returns empty.
    """
    kind = coerce_kind(kind)
    return tuple(dict.fromkeys(phase for cell_kind, phase in CELL_KIND_COMPAT.values()
                               if cell_kind is kind and phase is not None))


def spec_for(kind: CellKind | str) -> KindSpec:
    return KIND_TABLE[coerce_kind(kind)]


def priority_rank(kind: CellKind | str) -> int:
    return PRIORITY.index(coerce_kind(kind))


GENERATION_LANES: Final[tuple[Lane, ...]] = (Lane.SMALL, Lane.LARGE)


def lane_for(kind: CellKind | str, model: K.ModelCell | None) -> Lane:
    """Generation kinds follow their model's declared ``lane`` (``8B``/``32B``, a scale class
    independent of the served name so the Llama entry compiles onto the 8B lane, plan D11);
    the rest are fixed by the kind table."""
    spec = spec_for(kind)
    if spec.lane is not None:
        return spec.lane
    if model is None:
        raise ManifestError(f"{spec.kind.value} cells follow their model; pass the ModelCell")
    try:
        lane = Lane(model.lane)
    except ValueError:
        lane = None
    if lane not in GENERATION_LANES:
        raise ManifestError(f"model {model.name!r} declares lane {model.lane!r}, which is not a generation "
                            f"lane ({[l.value for l in GENERATION_LANES]})")
    return lane


def pool_for(kind: CellKind | str, model: K.ModelCell | None, compute: ComputeConfig) -> Pool | None:
    """The server pool of a generation kind from ``compute.pools.assignment``; ``None`` for
    CPU/capture kinds.  A deadline-bearing kind on a preemptable pool is a config error."""
    spec = spec_for(kind)
    if not spec.needs_endpoint:
        return None
    key = spec.kind.value if model is None else f"{spec.kind.value}@{model.name}"
    assignment = compute.pools.assignment
    name = assignment.get(key, assignment.get(spec.kind.value))
    if name is None:
        raise ConfigError(f"compute.pools.assignment has no entry for {spec.kind.value}")
    pool = Pool(name)
    if spec.deadline_bearing and getattr(compute.pools, pool.value).preemptable_allowed:
        raise ConfigError(f"{spec.kind.value} is deadline-bearing and may not use the preemptable "
                          f"pool {pool.value!r} (plan D20)")
    return pool


@dataclass(frozen=True)
class CellShape:
    items_per_cell: int
    parallel_items: int
    max_inflight: int
    cpus: int
    mem: str
    time: str
    partition: str


def items_per_cell(base: int, *, reference: ReferenceShape, n: int, revisions: int, thinking_cap: int) -> int:
    """Plan §2.12 sizing: ``clip(floor(base * (ref_n*ref_r)/(n*revisions) * ref_thinking/thinking), 1, base)``
    so a cell stays about the same wall time when N, R or the thinking cap change."""
    if base < 1 or n < 1 or revisions < 1 or thinking_cap < 1:
        raise ManifestError("items_per_cell needs positive base, n, revisions and thinking_cap")
    scaled = base * (reference.n_agents * reference.revision_rounds) / (n * revisions) \
        * reference.thinking_cap / thinking_cap
    return max(1, min(base, int(scaled)))


def cell_shape(kind: CellKind | str, *, compute: ComputeConfig, model: K.ModelCell | None = None,
               thinking_cap: int | None = None, team: K.TeamShape | None = None,
               grade_resources: Mapping[str, object] | None = None) -> CellShape:
    """The Slurm/worker shape of one cell of ``kind``, every number from ``compute.yaml``.

    Endpoint kinds take cpus/mem/time/partition from ``compute.workers.deadline`` or
    ``.loss_tolerant`` (by ``deadline_bearing``); GRADE takes them from the family's
    ``grade_resources`` when given (plan §1.10); CAPTURE takes its partition from
    ``compute.capture.partition`` (the capture lane's one declaration, R1); ``max_inflight:
    n_agents`` resolves from ``team``; EPISODE items scale with :func:`items_per_cell`.
    """
    spec = spec_for(kind)
    try:
        block: CellBlock = compute.cells.kinds[spec.kind.value]
    except KeyError as exc:
        raise ConfigError(f"compute.cells.kinds has no entry for {spec.kind.value}") from exc
    if spec.needs_endpoint:
        worker = compute.workers.deadline if spec.deadline_bearing else compute.workers.loss_tolerant
        cpus, mem, time, partition = worker.cpus, worker.mem, worker.time, worker.partition
    else:
        if block.cpus is None or block.mem is None or block.time is None:
            raise ConfigError(f"compute.cells.kinds.{spec.kind.value}: cpu kinds need cpus, mem, time")
        if spec.kind is CellKind.CAPTURE:
            if block.partition is not None:
                raise ConfigError("compute.cells.kinds.capture.partition must be null: the capture lane's "
                                  "partition is compute.capture.partition (one declaration, R1)")
            partition = compute.capture.partition
        elif block.partition is None:
            raise ConfigError(f"compute.cells.kinds.{spec.kind.value}: cpu kinds need a partition")
        else:
            partition = block.partition
        cpus, mem, time = block.cpus, block.mem, block.time
    items = block.items_per_cell
    if spec.kind is CellKind.GRADE and grade_resources is not None:
        cpus = int(grade_resources["cpus"])
        mem = str(grade_resources["mem"])
        time = str(grade_resources["time"])
        items = int(grade_resources["items_per_cell"])
    inflight = block.max_inflight
    if isinstance(inflight, str):
        if team is None:
            raise ManifestError(f"{spec.kind.value}: max_inflight is 'n_agents' but no TeamShape was given")
        inflight = team.n_agents
    if spec.kind is CellKind.EPISODE:
        if team is None or thinking_cap is None:
            raise ManifestError("EPISODE cell shape needs the TeamShape and the thinking cap")
        items = items_per_cell(items, reference=compute.cells.reference, n=team.n_agents,
                               revisions=team.revision_rounds, thinking_cap=thinking_cap)
    return CellShape(items_per_cell=items, parallel_items=block.parallel_items, max_inflight=int(inflight),
                     cpus=cpus, mem=mem, time=time, partition=partition)


def is_bank(kind: CellKind | str) -> bool:
    return coerce_kind(kind) in BANK_KINDS


__all__ = [
    "CELL_KIND_COMPAT", "CellShape", "DEADLINE_BEARING", "GENERATION_LANES", "KIND_FIELD_CONSUMERS",
    "KIND_TABLE", "KindSpec", "Lane", "PRIORITY",
    "Pool", "batch_width", "cell_shape", "is_bank", "items_per_cell", "lane_for", "pool_for", "phases_of", "priority_rank", "spec_for",
]
