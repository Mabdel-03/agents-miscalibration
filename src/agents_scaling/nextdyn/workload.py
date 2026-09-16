"""What the planned cells cost: calls, cache-unique requests, tokens, replica- and GPU-hours.

Plan §3.1 (the workload table ``plan --dry-run`` prints), §3.4 ("``plan.workload()`` reserves
audit, capture and selection cost before admitting episodes"), spec 16.1-16.4 (declared
budget, admission against the authorized ceiling).

Two things make this module more than arithmetic:

* **Reuse is counted, not assumed.**  ``raw_calls`` is what the manifests dispatch;
  ``unique_calls`` is what actually reaches a GPU after the common-random-numbers keying of
  plan D4 makes byte-identical prompts alias in the request store.  The aliasing fraction is
  not guessed here -- ``plan.reuse_census`` computes it mechanically (render the packet plan
  for one synthetic team per condition, hash the prompts, no model calls) and hands it in.
  ``status.py`` then compares the *live* alias rate against the same number, which is how a
  broken CRN keying (alias rate 0 in E3a) or an id collision (alias rate above plan) is
  caught while the run is still cheap to fix.
* **Reservation order.**  Selection audit, capture and grading are charged *first* and the
  episode lanes are admitted into what is left.  ARD36 admitted episodes against the whole
  ceiling and discovered the audit had no budget at the end, when the only remaining move was
  to drop the audit -- which is exactly the measurement the spec asks for.  Reserving first
  means a run that cannot afford its audit is reported at ``plan --dry-run``, before a single
  GPU-hour is spent, and STOP (run nothing) is always feasible (spec 16.4).

The throughput and length numbers below are **planning assumptions**, not experimental values:
they are the measured ARD36 rates carried into the plan's §3.1 table, and the pilot overwrites
every one of them into ``pilot/cost_profile.json`` (:func:`load_cost_profile`), which ``power``
and ``freeze`` then read.  No *experimental* quantity (panel size, replicate count, budget)
lives here -- those all arrive through the cells.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.cells import CellSpec
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane

COST_PROFILE_FILE: Final[str] = "cost_profile.json"
SECONDS_PER_HOUR: Final[float] = 3600.0


@dataclass(frozen=True)
class CostProfile:
    """Per-call token lengths and per-replica throughput, by lane (plan §3.1).

    Every field is an *operational* measurement: ARD36's realized rates at the 1024/256 caps,
    interpolated to this study's 2048/1024 contract.  ``from_measured`` replaces them with the
    pilot's own numbers, and ``freeze`` embeds whichever profile it admitted against.
    """

    generated_tokens_per_decision: float
    generated_tokens_per_receiver_call: float
    generated_tokens_per_audit_call: float
    prompt_tokens_per_decision: float
    tokens_per_second: Mapping[str, float]        # lane -> per-replica generation throughput
    gpus_per_replica: Mapping[str, int]           # lane -> GPUs one replica holds (tp)
    boundary_rate: float
    continuation_overhead: float                  # fraction of wall added by the phase-2 prefill
    bytes_per_request: float
    capture_tokens_per_replay: float
    capture_tokens_per_second: float
    grade_items_per_second: float
    measured: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens_per_second", dict(self.tokens_per_second))
        object.__setattr__(self, "gpus_per_replica", dict(self.gpus_per_replica))
        for lane, rate in self.tokens_per_second.items():
            if float(rate) <= 0.0:
                raise ContractViolation(f"cost profile: lane {lane} has throughput {rate!r}")

    def throughput(self, lane: Lane | str) -> float:
        name = lane.value if isinstance(lane, Lane) else str(lane)
        try:
            return float(self.tokens_per_second[name])
        except KeyError as exc:
            raise ContractViolation(
                f"cost profile has no throughput for lane {name!r}; measured lanes: "
                f"{sorted(self.tokens_per_second)}") from exc

    def gpus(self, lane: Lane | str) -> int:
        name = lane.value if isinstance(lane, Lane) else str(lane)
        return int(self.gpus_per_replica.get(name, 1))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostProfile":
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ContractViolation(f"cost profile carries unknown key(s) {unknown}")
        return cls(**dict(data))


#: Plan §3.1's assumptions, each with its measured basis.  Re-measured by the pilot.
PLANNING_PROFILE: Final[CostProfile] = CostProfile(
    generated_tokens_per_decision=1700.0,      # thinking 1350 + public JSON 350 + 2 delimiters
    generated_tokens_per_receiver_call=1100.0,  # E2b plan + decision average over both budgets
    generated_tokens_per_audit_call=820.0,     # pointwise audit at 1024/256
    prompt_tokens_per_decision=2300.0,         # PEER update prompt at a 512-token packet cap
    tokens_per_second={Lane.SMALL.value: 500.0, Lane.LARGE.value: 340.0, Lane.AUDIT.value: 340.0,
                       Lane.CAPTURE.value: 7700.0, Lane.CPU.value: 1.0},
    gpus_per_replica={Lane.SMALL.value: 1, Lane.LARGE.value: 2, Lane.AUDIT.value: 2,
                      Lane.CAPTURE.value: 1, Lane.CPU.value: 0},
    boundary_rate=0.28,                        # 8B at T=2048, between 34.7% @1024 and 21% @4096
    continuation_overhead=0.08,
    bytes_per_request=50_000.0,                # ARD36 store + native_reasoning_text + longer outputs
    capture_tokens_per_replay=4000.0,
    capture_tokens_per_second=7700.0,
    grade_items_per_second=66.0,
)


def walltime_seconds(text: str) -> float:
    """Slurm's ``[dd-]hh:mm:ss`` in seconds.

    Lives here rather than in a dispatch module because the dispatch modules carry no numbers
    at all, and ``60`` / ``3600`` are numbers -- even when they are units rather than knobs.
    """
    raw = str(text).strip()
    days = 0.0
    if "-" in raw:
        head, _, raw = raw.partition("-")
        days = float(head or 0)
    parts = [float(p or 0) for p in raw.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts[-3:]
    return days * 24 * 3600 + hours * 3600 + minutes * 60 + seconds


def cost_profile_path(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / COST_PROFILE_FILE


def load_cost_profile(run_root: str | os.PathLike | None) -> CostProfile:
    """The pilot's measured profile when it exists, else the planning one (plan §3.7)."""
    if run_root is None:
        return PLANNING_PROFILE
    path = cost_profile_path(run_root)
    if not path.exists():
        return PLANNING_PROFILE
    return CostProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_cost_profile(run_root: str | os.PathLike, profile: CostProfile) -> Path:
    path = cost_profile_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replace(profile, measured=True).to_dict(), indent=2, sort_keys=True)
                    + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- per-cell calls


def calls_per_item(cell: CellSpec, *, contract: K.StudyContract) -> int:
    """How many generation calls one item of ``cell`` issues, from the contract alone.

    Bank cells issue one call per draw; an episode issues ``n * R`` (round 0 is a reference to
    the bank draw, never a call); a one-call receiver issues one and a two-phase receiver two;
    an audit issues one per candidate.  Nothing here is a guess -- the counts are the shapes
    the engine actually runs, which is why ``test_14400_receiver_episodes_are_28800_generation
    _calls_per_model`` can be an arithmetic identity rather than a measurement.
    """
    banks = contract.banks
    team = contract.team
    if cell.kind is CellKind.DIFFICULTY_BANK:
        return int(banks.difficulty_draws)
    if cell.kind is CellKind.DONOR_BANK:
        stage = int(cell.option("extension_stage", 0))
        return int(banks.donor_initial) if stage == 0 else int(banks.donor_max - banks.donor_initial)
    if cell.kind is CellKind.NATURAL_ROOT:
        return int(cell.option("draws", banks.natural_root_draws))
    if cell.kind is CellKind.EPISODE:
        n = int(cell.option("n_agents", team.n_agents))
        revisions = int(cell.option("revisions", team.revision_rounds))
        return n * revisions
    if cell.kind is CellKind.RECEIVER:
        # E2a is one call (the decision, at full B); E2b is plan + decision (amendment A2).
        from agents_scaling.nextdyn.cells import ASSAY_E2A

        return 1 if cell.option("assay") == ASSAY_E2A else 2
    if cell.kind is CellKind.SELECTION_AUDIT:
        return int(cell.option("candidates", team.n_agents))
    return 0


def tokens_per_call(cell: CellSpec, *, profile: CostProfile) -> float:
    if cell.kind is CellKind.RECEIVER:
        return float(profile.generated_tokens_per_receiver_call)
    if cell.kind is CellKind.SELECTION_AUDIT:
        return float(profile.generated_tokens_per_audit_call)
    return float(profile.generated_tokens_per_decision)


# --------------------------------------------------------------------------- the table


@dataclass(frozen=True)
class LaneLoad:
    """One lane's share of the campaign."""

    lane: str
    cells: int
    items: int
    raw_calls: int
    unique_calls: int
    generated_tokens: float
    replica_hours: float
    gpu_hours: float
    storage_bytes: float

    def __add__(self, other: "LaneLoad") -> "LaneLoad":
        if other.lane != self.lane:
            raise ContractViolation(f"cannot add lane {other.lane!r} to {self.lane!r}")
        return LaneLoad(self.lane, self.cells + other.cells, self.items + other.items,
                        self.raw_calls + other.raw_calls, self.unique_calls + other.unique_calls,
                        self.generated_tokens + other.generated_tokens,
                        self.replica_hours + other.replica_hours,
                        self.gpu_hours + other.gpu_hours,
                        self.storage_bytes + other.storage_bytes)


@dataclass(frozen=True)
class BlockLoad:
    block: str
    cells: int
    raw_calls: int
    unique_calls: int
    generated_tokens: float
    gpu_hours: float


@dataclass(frozen=True)
class Workload:
    """The §3.1 table: per lane, per block, and the totals ``freeze`` admits against."""

    lanes: Mapping[str, LaneLoad]
    blocks: tuple[BlockLoad, ...]
    profile: CostProfile
    reuse_fraction: Mapping[str, float]

    @property
    def raw_calls(self) -> int:
        return sum(l.raw_calls for l in self.lanes.values())

    @property
    def unique_calls(self) -> int:
        return sum(l.unique_calls for l in self.lanes.values())

    @property
    def generated_tokens(self) -> float:
        return sum(l.generated_tokens for l in self.lanes.values())

    @property
    def gpu_hours(self) -> float:
        return sum(l.gpu_hours for l in self.lanes.values())

    @property
    def storage_bytes(self) -> float:
        return sum(l.storage_bytes for l in self.lanes.values())

    def lane(self, name: Lane | str) -> LaneLoad:
        key = name.value if isinstance(name, Lane) else str(name)
        return self.lanes.get(key, LaneLoad(key, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lanes": {name: asdict(load) for name, load in sorted(self.lanes.items())},
            "blocks": [asdict(b) for b in self.blocks],
            "reuse_fraction": dict(self.reuse_fraction),
            "totals": {"raw_calls": self.raw_calls, "unique_calls": self.unique_calls,
                       "generated_tokens": self.generated_tokens, "gpu_hours": self.gpu_hours,
                       "storage_bytes": self.storage_bytes},
            "profile": self.profile.to_dict(),
        }


def _lane_load(lane: str, cells: Sequence[CellSpec], *, contract: K.StudyContract,
               profile: CostProfile, reuse: Mapping[str, float]) -> LaneLoad:
    raw = unique = 0
    tokens = 0.0
    items = 0
    for cell in cells:
        per_item = calls_per_item(cell, contract=contract)
        cell_calls = per_item * cell.n_items
        fraction = 1.0 if cell.reuse_of else float(
            reuse.get(cell.display_label or "", reuse.get(cell.block, 0.0)))
        if not 0.0 <= fraction <= 1.0:
            raise ContractViolation(
                f"{cell.cell_id}: reuse fraction {fraction!r} outside [0, 1]")
        if fraction >= 1.0 and not cell.reuse_of:
            raise ContractViolation(
                f"{cell.cell_id}: the census says every one of this condition's requests aliases "
                "an already-planned one, but the cell declares no reuse_of; a fully equivalent "
                "condition is recorded (plan.reuse_of), never dispatched twice")
        cell_unique = int(round(cell_calls * (1.0 - fraction)))
        raw += cell_calls
        unique += cell_unique
        items += cell.n_items
        tokens += cell_unique * tokens_per_call(cell, profile=profile)
    if lane == Lane.CPU.value:
        replica_hours = items / max(profile.grade_items_per_second, 1e-9) / SECONDS_PER_HOUR
        gpu_hours = 0.0
    elif lane == Lane.CAPTURE.value:
        tokens = items * profile.capture_tokens_per_replay
        replica_hours = tokens / profile.capture_tokens_per_second / SECONDS_PER_HOUR
        gpu_hours = replica_hours * profile.gpus(lane)
    else:
        replica_hours = tokens / profile.throughput(lane) / SECONDS_PER_HOUR
        replica_hours *= (1.0 + profile.continuation_overhead)
        gpu_hours = replica_hours * profile.gpus(lane)
    return LaneLoad(lane=lane, cells=len(cells), items=items, raw_calls=raw, unique_calls=unique,
                    generated_tokens=tokens, replica_hours=replica_hours, gpu_hours=gpu_hours,
                    storage_bytes=unique * profile.bytes_per_request)


def calls(cells: Sequence[CellSpec], *, contract: K.StudyContract,
          profile: CostProfile | None = None,
          reuse_fraction: Mapping[str, float] | None = None) -> Workload:
    """The workload of a cell set (plan §3.1): per lane and per block."""
    profile = profile or PLANNING_PROFILE
    reuse = dict(reuse_fraction or {})
    by_lane: dict[str, list[CellSpec]] = {}
    by_block: dict[str, list[CellSpec]] = {}
    for cell in cells:
        by_lane.setdefault(cell.lane.value, []).append(cell)
        by_block.setdefault(cell.block, []).append(cell)
    lanes = {lane: _lane_load(lane, group, contract=contract, profile=profile, reuse=reuse)
             for lane, group in sorted(by_lane.items())}
    blocks = []
    for block, group in sorted(by_block.items()):
        sub = [_lane_load(lane, [c for c in group if c.lane.value == lane], contract=contract,
                          profile=profile, reuse=reuse)
               for lane in sorted({c.lane.value for c in group})]
        blocks.append(BlockLoad(block=block, cells=len(group),
                                raw_calls=sum(s.raw_calls for s in sub),
                                unique_calls=sum(s.unique_calls for s in sub),
                                generated_tokens=sum(s.generated_tokens for s in sub),
                                gpu_hours=sum(s.gpu_hours for s in sub)))
    return Workload(lanes=lanes, blocks=tuple(blocks), profile=profile, reuse_fraction=reuse)


# --------------------------------------------------------------------------- admission


#: The blocks whose cost is reserved *before* episodes are admitted (plan §3.4).  They are the
#: measurements the study cannot report without: the selection audit is spec 6.5's deployed
#: verifier, capture is spec 13, grading is every outcome.
RESERVED_BLOCKS: Final[tuple[str, ...]] = ("AUDIT", "CAPTURE", "GRADE", "E0", "MATERIALIZE")


@dataclass(frozen=True)
class Admission:
    """What ``plan --dry-run`` and ``freeze`` decide against the authorized ceiling."""

    ceiling_gpu_hours: float | None
    reserve_fraction: float
    reserved_gpu_hours: float
    episode_gpu_hours: float
    admitted: bool
    stop_feasible: bool
    storage_limit_bytes: int | None
    storage_bytes: float
    storage_admitted: bool
    detail: str

    @property
    def projected_gpu_hours(self) -> float:
        return self.reserved_gpu_hours + self.episode_gpu_hours

    @property
    def projected_with_reserve(self) -> float:
        return self.projected_gpu_hours * (1.0 + self.reserve_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "projected_gpu_hours": self.projected_gpu_hours,
                "projected_with_reserve": self.projected_with_reserve}


def admit(workload: Workload, *, budgets: Any) -> Admission:
    """Charge the reserved blocks first, then admit the episodes into what is left.

    ``STOP`` -- dispatch nothing -- is feasible by construction: it costs zero and is reported
    as such, so the planner never has to choose between "run everything" and "run nothing
    but do not say so" (spec 16.4, ``test_stop_is_available_in_every_action_policy``).
    """
    reserved = sum(b.gpu_hours for b in workload.blocks if b.block in RESERVED_BLOCKS)
    episodes = workload.gpu_hours - reserved
    ceiling = None if budgets.campaign_gpu_hours is None else float(budgets.campaign_gpu_hours)
    reserve_fraction = float(budgets.compute_reserve_fraction)
    storage_limit = None if budgets.storage_limit_bytes is None else int(budgets.storage_limit_bytes)
    storage = workload.storage_bytes * (1.0 + float(budgets.storage_reserve_fraction))
    projected = (reserved + episodes) * (1.0 + reserve_fraction)
    if ceiling is None:
        detail = ("no authorized ceiling: budgets.campaign_gpu_hours is null (spec 16.4 blocker; "
                  "freeze refuses it)")
        admitted = False
    elif reserved * (1.0 + reserve_fraction) > ceiling:
        detail = (f"the reserved blocks alone ({reserved:.1f} GPU-h x {1 + reserve_fraction:.2f}) "
                  f"exceed the authorized {ceiling:.1f}; reduce the audit scope before episodes")
        admitted = False
    elif projected > ceiling:
        detail = (f"projection {projected:.1f} GPU-h > authorized {ceiling:.1f}; reduce the tier "
                  "(audit scope, then the matched-slot diagnostic, then the bridge)")
        admitted = False
    else:
        detail = f"projection {projected:.1f} GPU-h within the authorized {ceiling:.1f}"
        admitted = True
    storage_admitted = storage_limit is None or storage <= storage_limit
    return Admission(ceiling_gpu_hours=ceiling, reserve_fraction=reserve_fraction,
                     reserved_gpu_hours=reserved, episode_gpu_hours=episodes, admitted=admitted,
                     stop_feasible=True, storage_limit_bytes=storage_limit, storage_bytes=storage,
                     storage_admitted=storage_admitted, detail=detail)


# --------------------------------------------------------------------------- projection


@dataclass(frozen=True)
class Projection:
    """Wall-clock the realized fleet implies (plan §3.1 "Lane time")."""

    lane_hours: Mapping[str, float]
    replicas: Mapping[str, int]
    critical_path_hours: float
    gpu_hours_with_reserve: float

    def to_dict(self) -> dict[str, Any]:
        return {"lane_hours": dict(self.lane_hours), "replicas": dict(self.replicas),
                "critical_path_hours": self.critical_path_hours,
                "gpu_hours_with_reserve": self.gpu_hours_with_reserve}


def projection(workload: Workload, *, replicas: Mapping[str, int], reserve_fraction: float) -> Projection:
    """Hours per lane at the given replica counts; the critical path is the slowest lane."""
    hours: dict[str, float] = {}
    for name, load in workload.lanes.items():
        n = max(int(replicas.get(name, 0)), 0)
        hours[name] = float("inf") if (n == 0 and load.replica_hours > 0) else (
            0.0 if load.replica_hours == 0 else load.replica_hours / n)
    critical = max(hours.values(), default=0.0)
    return Projection(lane_hours=hours, replicas=dict(replicas), critical_path_hours=critical,
                      gpu_hours_with_reserve=workload.gpu_hours * (1.0 + reserve_fraction))


def fleet_floor(workload: Workload, *, hours: float) -> dict[str, int]:
    """The replicas per lane a target wall clock implies (the "fleet floor" line of §3.1)."""
    if hours <= 0:
        raise ContractViolation("a fleet floor needs a positive target wall clock")
    out: dict[str, int] = {}
    for name, load in workload.lanes.items():
        if load.replica_hours <= 0:
            continue
        out[name] = max(1, int(-(-load.replica_hours // hours)))
    return out


def render_table(workload: Workload, admission: Admission | None = None) -> str:
    """The human table ``plan --dry-run`` prints (plan §3.1)."""
    lines = [f"{'block':<20}{'cells':>7}{'raw calls':>12}{'unique':>12}{'GPU-h':>10}"]
    for block in workload.blocks:
        lines.append(f"{block.block:<20}{block.cells:>7}{block.raw_calls:>12,}"
                     f"{block.unique_calls:>12,}{block.gpu_hours:>10.1f}")
    lines.append("-" * len(lines[0]))
    for name, load in sorted(workload.lanes.items()):
        lines.append(f"lane {name:<15}{load.cells:>7}{load.raw_calls:>12,}{load.unique_calls:>12,}"
                     f"{load.gpu_hours:>10.1f}")
    lines.append(f"{'TOTAL':<20}{'':>7}{workload.raw_calls:>12,}{workload.unique_calls:>12,}"
                 f"{workload.gpu_hours:>10.1f}")
    if admission is not None:
        lines.append(f"reserved (audit/capture/grade/E0/materialize): "
                     f"{admission.reserved_gpu_hours:.1f} GPU-h")
        lines.append(f"episodes admitted into the remainder: {admission.episode_gpu_hours:.1f} GPU-h")
        lines.append(f"admission: {'OK' if admission.admitted else 'REFUSED'} - {admission.detail}")
        lines.append(f"storage: {admission.storage_bytes / 1e9:.1f} GB vs limit "
                     f"{'null' if admission.storage_limit_bytes is None else admission.storage_limit_bytes / 1e9}"
                     f" GB -> {'OK' if admission.storage_admitted else 'REFUSED'}")
    return "\n".join(lines)


__all__ = [
    "Admission", "BlockLoad", "COST_PROFILE_FILE", "CostProfile", "LaneLoad", "PLANNING_PROFILE",
    "Projection", "RESERVED_BLOCKS", "Workload", "admit", "calls", "calls_per_item",
    "cost_profile_path", "fleet_floor", "load_cost_profile", "projection",
    "render_table", "tokens_per_call", "walltime_seconds", "write_cost_profile",
]
