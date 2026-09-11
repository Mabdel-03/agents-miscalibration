"""ARD36 runtime gates: the hour-2 launch gate and the hour-12/18/24 coverage assertions
(spec §16.3, §16.4; plan §7, §10 "Runtime gates").

Two gates live here.

**The launch gate** (:func:`launch_gate`, spec §16.3) is the hour-2 tier freeze.  It refuses to
approve a tier on projections alone: every field it scores is a *measured* one, and the
projection is cross-checked by running the bundle's own ``budget_check.py`` with those measured
rates (:func:`run_budget_check` - the script is executed out of the shipped bundle so the check
is the artifact the spec ships, not a paraphrase of it).  It asserts registered,
health-checked replicas per serving profile **x partition** through
``serving.registry.list_live_servers`` (a registry record is not a replica; the census counts
only endpoints that answered), parse validity, grader throughput, replay throughput, the HTTP
error rate, and throughput **stratified by call type** - ROOT, PEER/BALANCED revision, hub
leader versus worker, the 4,096-token budget and the receiver assay generate very different
prompt and generation lengths, so one aggregate tok/s would hide the lane that binds.

The tier is frozen *before* any new test outcome exists (spec §16.4), and the default is the
reduced core: the full core is restored only when the measured projection with the 20% reserve
finishes inside the plan's hour-30 mark.

**The coverage gate** (:func:`coverage_gate`) is the hour-12/18/24 reconciliation, asserted on
**decisions** and grouped by protocol x checkpoint x cohort (x split), because an aggregate
"90% complete" can be 100% of PEER and 0% of GATE.  Cell-level planned/completed/incomplete
accounting follows ``study/reconcile.py`` and reuses its concrete mechanism -
``study.runner.CellPaths`` and ``item_file_valid`` - through :func:`ard_cell_reconciliation`;
``reconcile.reconcile_cells`` itself cannot be called on an ARD manifest because it parses
cells with ``study.cells.CellSpec.from_dict``, which does not know the ARD fields.  Its
``judge_rates``/``vote_rates`` readouts are carried straight through when the legacy
directories exist.

Neither gate submits a job, launches a server, or writes into a cell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents_scaling.ard import contract as K
from agents_scaling.ard.types import ArdCellKind, ArdCellSpec
from agents_scaling.experiment import io
from agents_scaling.study import reconcile as SR
from agents_scaling.study.runner import CellPaths, item_file_valid

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ZIP = REPO_ROOT / "expermt_agent_specs" / "AgentsRoundV1" / "Agents_Round_Dynamics_36h_Bundle.zip"
BUNDLE_PREFIX = "Agents_Round_Dynamics_36h"

# --------------------------------------------------------------------------- thresholds

#: Root parse validity floor (plan §7 launch gate).
PARSE_VALIDITY_MIN: float = 0.95
#: The gate fails above 1% of requests (plan §7).
HTTP_ERROR_RATE_MAX: float = 0.01
#: Explicit uncertainty reserve (spec §16.3).
RESERVE_FRACTION: float = 0.20
#: Projected critical path (P0 -> P2 -> controller freeze) and total, in hours from T0.
CRITICAL_PATH_DEADLINE_H: float = 16.0
TOTAL_DEADLINE_H: float = 34.0
#: The full core is restored only if the reserved projection finishes by this hour (plan §9.1).
FULL_CORE_DEADLINE_H: float = 30.0
#: Hard capture deadline (plan §7, hours 24-30).
CAPTURE_DEADLINE_H: float = 30.0

#: Call types the hour-2 throughput measurement must cover (plan §7 launch gate).
CALL_TYPES: tuple[str, ...] = (
    "ROOT", "PEER_REVISE", "BALANCED_REVISE", "HUB_LEADER", "HUB_WORKER",
    "HIGH_BUDGET", "RECEIVER",
)
#: Fields every call-type row must carry (spec §16.3 "required measured fields").
CALL_TYPE_FIELDS: tuple[str, ...] = (
    "tokens_per_s", "latency_mean_s", "latency_p95_s", "prompt_tokens_mean",
    "generated_tokens_mean", "n_requests",
)

#: Decision coverage floors per split (plan §7: hour-18 reconcile, hour-24 check).  Hour 12 is
#: the plan's earlier checkpoint on the same series and is set at the sub-gate level of the
#: hour-14 train/val closure; all three are overridable by the caller.
COVERAGE_THRESHOLDS: Mapping[int, Mapping[str, float]] = {
    12: {"train": 0.90, "validation": 0.90, "test": 0.10},
    18: {"train": 0.99, "validation": 0.99, "test": 0.40},
    24: {"train": 0.99, "validation": 0.99, "test": 0.90},
}

#: Logical decisions one item of each kind owes.  ``None`` = the kind produces no episode
#: decisions (grading, compaction, capture and the CPU analysis lanes).
DECISIONS_PER_ITEM: Mapping[ArdCellKind, int | None] = {
    ArdCellKind.ROOTS: K.ROOT_DRAWS,
    ArdCellKind.EPISODE: K.N_AGENTS * K.UPDATE_ROUNDS,
    ArdCellKind.CHALLENGE: K.N_AGENTS * K.UPDATE_ROUNDS,
    ArdCellKind.FORK: K.N_AGENTS,
    ArdCellKind.SUFFIX: K.N_AGENTS,
    ArdCellKind.RECEIVER: None,
    ArdCellKind.SELECTOR_AUDIT: None,
    ArdCellKind.GRADE: None,
    ArdCellKind.E0: None,
    ArdCellKind.COMPACT: None,
    ArdCellKind.CAPTURE: None,
}


class GateError(RuntimeError):
    """A gate could not be evaluated (missing measurement, unreadable bundle, ...)."""


@dataclass(frozen=True)
class GateCheck:
    """One scored requirement.  ``blocking`` checks decide whether the gate passes."""

    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    note: str = ""
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- budget_check


def _extract_bundle(names: Sequence[str], into: Path, *, bundle_zip: Path = BUNDLE_ZIP) -> dict[str, Path]:
    if not Path(bundle_zip).exists():
        raise GateError(f"bundle not found at {bundle_zip}; the launch gate runs the shipped "
                        "budget_check.py, it does not re-implement it")
    out: dict[str, Path] = {}
    with zipfile.ZipFile(bundle_zip) as archive:
        for name in names:
            target = into / name
            target.write_bytes(archive.read(f"{BUNDLE_PREFIX}/{name}"))
            out[name] = target
    return out


def run_budget_check(
    *,
    small_tps_per_replica: float,
    large_tps_per_replica: float,
    replay_hours: float | None = None,
    hardware: str = "ten_gpus",
    config_path: str | os.PathLike | None = None,
    bundle_zip: Path = BUNDLE_ZIP,
    python: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run the bundle's ``budget_check.py`` with the **measured** rates and return its JSON.

    Spec §16.3: "measure representative batches ... raw decode-only benchmark TPS is not
    suitable".  The rates handed in here must therefore be effective output throughput over
    real batches, including prefill and both phases of split-budget generation.
    """
    if small_tps_per_replica <= 0 or large_tps_per_replica <= 0:
        raise GateError("measured throughput must be positive; the gate has no default rate")
    with tempfile.TemporaryDirectory(prefix="ard36_budget_") as tmp:
        tmp_path = Path(tmp)
        wanted = ["budget_check.py"] + ([] if config_path else ["run_plan.yaml"])
        files = _extract_bundle(wanted, tmp_path, bundle_zip=bundle_zip)
        config = Path(config_path) if config_path else files["run_plan.yaml"]
        argv = [python or sys.executable, str(files["budget_check.py"]),
                "--config", str(config), "--hardware", hardware,
                "--small-tps-per-replica", repr(float(small_tps_per_replica)),
                "--large-tps-per-replica", repr(float(large_tps_per_replica))]
        if replay_hours is not None:
            argv += ["--replay-hours", repr(float(replay_hours))]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False)
        if proc.returncode != 0:
            raise GateError(f"budget_check.py failed ({proc.returncode}): "
                            f"{proc.stderr.strip()[:400]}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - budget_check prints JSON
            raise GateError(f"budget_check.py produced no JSON: {exc}") from exc


# --------------------------------------------------------------------------- replicas


def live_replica_census(
    run_root: str | os.PathLike,
    profiles: Sequence[str] = (K.PROFILE_SMALL, K.PROFILE_LARGE),
    *,
    partition_of: Callable[[Any], str] | Mapping[str, str] | None = None,
    lister: Callable[..., Sequence[Any]] | None = None,
) -> dict[str, dict[str, int]]:
    """``{profile: {partition: live health-checked replica count}}``.

    Uses ``serving.registry.list_live_servers``, which validates each record (scheduler
    authority or repeated HTTP probes) instead of trusting the registry file - a stale record
    is exactly the failure the launch gate exists to catch.  ``ServerEntry`` carries no
    partition, so the partition is resolved by ``partition_of`` (a callable, or a
    ``host -> partition`` mapping); unresolved hosts land in ``"unknown"`` and are visible in
    the census rather than silently credited to a partition.
    """
    if lister is None:
        from agents_scaling.serving.registry import list_live_servers as lister  # noqa: PLW2901

    def resolve(entry: Any) -> str:
        if partition_of is None:
            return "unknown"
        if callable(partition_of):
            return str(partition_of(entry) or "unknown")
        return str(partition_of.get(getattr(entry, "host", ""), "unknown"))

    census: dict[str, dict[str, int]] = {}
    for profile in profiles:
        counts: dict[str, int] = defaultdict(int)
        for entry in lister(run_root, profile):
            counts[resolve(entry)] += 1
        census[profile] = dict(counts)
    return census


def _replica_checks(census: Mapping[str, Mapping[str, int]],
                    required: Mapping[str, Mapping[str, int]]) -> list[GateCheck]:
    checks: list[GateCheck] = []
    for profile, by_partition in required.items():
        for partition, need in by_partition.items():
            have = int(census.get(profile, {}).get(partition, 0))
            checks.append(GateCheck(
                name=f"replicas:{profile}:{partition}", passed=have >= int(need),
                value=have, threshold=int(need),
                note="registered AND health-checked replicas (registry records alone do not count)"))
    return checks


# --------------------------------------------------------------------------- launch gate


def _throughput_checks(measured: Mapping[str, Any],
                       call_types: Sequence[str]) -> list[GateCheck]:
    table = measured.get("throughput_by_call_type") or {}
    checks: list[GateCheck] = []
    for call_type in call_types:
        row = table.get(call_type)
        # A call type the frozen tier does not run is legitimately absent, not unmeasured.
        # HUB5 is dropped by §16.4 step 2 under the reduced tier, so demanding a measured
        # HUB_LEADER/HUB_WORKER batch would make an honest gate unpassable.  The caller must
        # say so explicitly -- silence still fails -- and the reason is carried into the report.
        if isinstance(row, Mapping) and row.get("not_in_tier"):
            checks.append(GateCheck(
                name=f"throughput:{call_type}", passed=True, value="not_in_tier",
                threshold=list(CALL_TYPE_FIELDS),
                note=str(row.get("reason") or "not run under the frozen tier (§16.4)")))
            continue
        if not isinstance(row, Mapping):
            checks.append(GateCheck(
                name=f"throughput:{call_type}", passed=False, value=None,
                threshold="measured row",
                note="no measured batch for this call type; one aggregate tok/s hides the "
                     "lane that binds (spec §16.3)"))
            continue
        missing = [f for f in CALL_TYPE_FIELDS if row.get(f) is None]
        positive = all(isinstance(row.get(f), (int, float)) and float(row[f]) > 0
                       for f in ("tokens_per_s", "n_requests") if row.get(f) is not None)
        checks.append(GateCheck(
            name=f"throughput:{call_type}", passed=not missing and positive,
            value={f: row.get(f) for f in CALL_TYPE_FIELDS}, threshold=list(CALL_TYPE_FIELDS),
            note=("missing fields: " + ", ".join(missing)) if missing else
                 "effective tokens/s including prefill and split-generation overhead"))
    return checks


def launch_gate(
    run_root: str | os.PathLike,
    *,
    measured: Mapping[str, Any],
    required_replicas: Mapping[str, Mapping[str, int]],
    partition_of: Callable[[Any], str] | Mapping[str, str] | None = None,
    lister: Callable[..., Sequence[Any]] | None = None,
    hardware: str = "ten_gpus",
    bundle_zip: Path = BUNDLE_ZIP,
    run_budget: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Score the hour-2 launch gate (spec §16.3) and choose the tier (spec §16.4).

    ``measured`` is the hour-2 measurement block; every field is a measurement, never a plan
    number.  Required keys: ``small_tps_per_replica``, ``large_tps_per_replica``,
    ``root_parse_validity``, ``http_error_rate``, ``grader_items_per_s``,
    ``required_grader_items_per_s``, ``replay_sequences_per_s``, ``replay_hours``,
    ``bf16_confirmed``, ``projected_critical_path_hours``, ``projected_total_hours`` and
    ``throughput_by_call_type``.
    """
    missing = [key for key in (
        "small_tps_per_replica", "large_tps_per_replica", "root_parse_validity",
        "http_error_rate", "grader_items_per_s", "replay_sequences_per_s",
        "projected_critical_path_hours", "projected_total_hours") if measured.get(key) is None]
    if missing:
        raise GateError(f"launch gate is missing measured field(s) {missing}; a gate scored on "
                        "plan numbers is not a gate (spec §16.3)")

    census = live_replica_census(run_root, tuple(required_replicas) or
                                 (K.PROFILE_SMALL, K.PROFILE_LARGE),
                                 partition_of=partition_of, lister=lister)
    checks: list[GateCheck] = list(_replica_checks(census, required_replicas))
    checks.append(GateCheck(
        name="bf16_confirmed", passed=bool(measured.get("bf16_confirmed")),
        value=measured.get("bf16_confirmed"), threshold=True,
        note=f"one precision throughout: {K.DTYPE} (spec §6.1)"))
    checks.append(GateCheck(
        name="root_parse_validity", passed=float(measured["root_parse_validity"]) >= PARSE_VALIDITY_MIN,
        value=float(measured["root_parse_validity"]), threshold=PARSE_VALIDITY_MIN,
        note="fraction of pilot root reports in contract.VALID_PARSE_STATUSES"))
    checks.append(GateCheck(
        name="http_error_rate", passed=float(measured["http_error_rate"]) <= HTTP_ERROR_RATE_MAX,
        value=float(measured["http_error_rate"]), threshold=HTTP_ERROR_RATE_MAX,
        note="4xx/5xx + retries + endpoint bans, as a fraction of requests"))
    required_grade_rate = float(measured.get("required_grader_items_per_s") or 0.0)
    grader_rate = float(measured["grader_items_per_s"])
    checks.append(GateCheck(
        name="grader_items_per_s", passed=grader_rate > 0 and grader_rate >= required_grade_rate,
        value=grader_rate, threshold=required_grade_rate,
        note="measured on a 500-answer pilot; below this, raise grade concurrency or switch "
             "non-primary tables to the string grader and record an amendment"))
    replay_rate = float(measured["replay_sequences_per_s"])
    replay_hours = measured.get("replay_hours")
    replay_ok = replay_rate > 0 and (replay_hours is None
                                     or float(replay_hours) <= CAPTURE_DEADLINE_H)
    checks.append(GateCheck(
        name="replay_sequences_per_s", passed=replay_ok, value=replay_rate,
        threshold={"positive": True, "replay_hours_max": CAPTURE_DEADLINE_H},
        note="replay throughput at p95 context; capture has a hard hour-30 deadline"))
    checks.extend(_throughput_checks(measured, CALL_TYPES))

    critical = float(measured["projected_critical_path_hours"]) * (1 + RESERVE_FRACTION)
    total = float(measured["projected_total_hours"]) * (1 + RESERVE_FRACTION)
    checks.append(GateCheck(
        name="projected_critical_path_h", passed=critical <= CRITICAL_PATH_DEADLINE_H,
        value=critical, threshold=CRITICAL_PATH_DEADLINE_H,
        note=f"P0->P2->controller freeze with the {int(RESERVE_FRACTION * 100)}% reserve"))
    checks.append(GateCheck(
        name="projected_total_h", passed=total <= TOTAL_DEADLINE_H, value=total,
        threshold=TOTAL_DEADLINE_H,
        note="a sum of GPU-hours does not prove the critical path fits (spec §16.3)"))

    budget: dict[str, Any] | None = None
    if run_budget:
        budget = run_budget_check(
            small_tps_per_replica=float(measured["small_tps_per_replica"]),
            large_tps_per_replica=float(measured["large_tps_per_replica"]),
            replay_hours=(float(replay_hours) if replay_hours is not None else None),
            hardware=hardware, bundle_zip=bundle_zip)
        for lane in ("small", "large"):
            passed = budget.get(f"{lane}_lane", {}).get("output_lane_gate_passed")
            checks.append(GateCheck(
                name=f"budget_check:{lane}_lane", passed=bool(passed),
                value=budget.get(f"{lane}_lane", {}).get("projected_hours_with_margin"),
                threshold=budget.get(f"{lane}_lane", {}).get("serving_window_hours"),
                note="the bundle's own ledger cross-check, fed the measured effective tok/s"))

    blocking = [c for c in checks if c.blocking]
    passed = all(c.passed for c in blocking)
    full_core = passed and total <= FULL_CORE_DEADLINE_H
    report = {
        "gate": "launch",
        "hour": 2,
        "study_id": K.STUDY_ID,
        "run_root": str(run_root),
        "passed": passed,
        "tier": "full_core" if full_core else "reduced_core",
        "tier_note": (
            "Full core restored: the measured projection with the reserve finishes by hour "
            f"{FULL_CORE_DEADLINE_H:g}." if full_core else
            "Reduced core (spec §16.4 steps 1-3): extensions off, selector audit on the root "
            "and GATE-final pools, HUB5 dropped, E2d 32->16 items, challenge cap 96->48, "
            "receiver 64->32 programs, all factor levels kept."),
        "replica_census": census,
        "required_replicas": {p: dict(v) for p, v in required_replicas.items()},
        "checks": [c.to_dict() for c in checks],
        "failed": [c.name for c in blocking if not c.passed],
        "budget_check": budget,
        "reserve_fraction": RESERVE_FRACTION,
        "note": ("The tier is frozen here, before any new test outcome exists; an arm is never "
                 "stopped because its results look poor (spec §16.4)."),
    }
    if write:
        io.write_json(Path(run_root) / "gates" / "launch_gate.json", report)
    return report


# --------------------------------------------------------------------------- coverage gate


def ard_cell_reconciliation(
    run_root: str | os.PathLike, cells: Sequence[ArdCellSpec]
) -> dict[str, dict[str, Any]]:
    """Planned vs completed vs incomplete items per protocol x checkpoint x cohort x split.

    ``study/reconcile.py:reconcile_cells`` line for line, with its ``(module, lane)`` group key
    replaced by the ARD axis and its ``load_cells_file`` step lifted out (an ARD manifest is
    parsed by ``ard.cells.load_ard_cells_file``; ``study.cells.CellSpec.from_dict`` does not
    know the ARD fields).  ``CellPaths`` and ``item_file_valid`` are the study functions,
    unmodified, so "completed" means exactly what it means for study_v4.
    """
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(lambda: {
        "planned_cells": 0, "cells_with_meta": 0, "suspended_cells": 0, "planned_items": 0,
        "completed_items": 0, "incomplete_items": 0, "aliased_requests": 0,
        "generated_requests": 0, "flops_total": 0, "incomplete": [], "suspended": [],
    })
    for cell in cells:
        key = (str(getattr(cell.protocol, "value", cell.protocol) or "-"),
               str(cell.checkpoint), str(getattr(cell.cohort, "value", cell.cohort)),
               str(cell.split))
        g = groups[key]
        paths = CellPaths.of(run_root, cell.cell_id)
        g["planned_cells"] += 1
        g["planned_items"] += len(cell.items)
        for sid in cell.items:
            try:
                if item_file_valid(paths.item(sid), cell, sid):
                    g["completed_items"] += 1
            except Exception as exc:  # noqa: BLE001 - study raises ProtocolError here
                g["incomplete"].append({"cell_id": cell.cell_id, "source_id": sid,
                                        "error": str(exc)[:300]})
        for sid in paths.incomplete_items():
            g["incomplete_items"] += 1
            g["incomplete"].append({"cell_id": cell.cell_id, "source_id": sid})
        if paths.suspended.exists():
            g["suspended_cells"] += 1
            g["suspended"].append(cell.cell_id)
        if paths.meta.exists():
            meta = json.loads(paths.meta.read_text(encoding="utf-8"))
            g["cells_with_meta"] += 1
            g["aliased_requests"] += int(meta.get("n_aliased_requests", 0))
            g["generated_requests"] += int(meta.get("n_generated_requests", 0))
            g["flops_total"] += int(meta.get("flops_total", 0))
    return {"|".join(key): value for key, value in sorted(groups.items())}


def decision_coverage(
    run_root: str | os.PathLike,
    cells: Sequence[ArdCellSpec],
    *,
    decisions_per_item: Mapping[ArdCellKind, int | None] = DECISIONS_PER_ITEM,
) -> list[dict[str, Any]]:
    """Planned vs recorded **decisions** per protocol x checkpoint x cohort x split.

    Coverage on items would pass a cell whose episodes each dropped a round; the spec's unit of
    completion is the logical decision, and no episode is ever short (early stopping on
    unanimity is disabled, ``contract.EARLY_STOP_ON_UNANIMITY is False``), so
    ``planned = items x decisions_per_item`` is exact.  A cell may override its own count with
    ``options["expected_decisions_per_item"]`` (E2d and SUFFIX arms).
    """
    from agents_scaling.ard.compact import load_episodes

    planned: dict[tuple[str, str, str, str], int] = defaultdict(int)
    observed: dict[tuple[str, str, str, str], int] = defaultdict(int)
    episodes_seen: dict[tuple[str, str, str, str], int] = defaultdict(int)
    cell_ids: list[str] = []
    key_of_cell: dict[str, tuple[str, str, str, str]] = {}
    for cell in cells:
        per_item = cell.options.get("expected_decisions_per_item") if cell.options else None
        if per_item is None:
            per_item = decisions_per_item.get(ArdCellKind(cell.kind))
        if per_item is None:
            continue
        key = (str(getattr(cell.protocol, "value", cell.protocol) or "-"),
               str(cell.checkpoint), str(getattr(cell.cohort, "value", cell.cohort)),
               str(cell.split))
        planned[key] += len(cell.items) * int(per_item)
        cell_ids.append(cell.cell_id)
        key_of_cell[cell.cell_id] = key
    for cell, _item_id, episode in load_episodes(run_root, cell_ids):
        key = key_of_cell.get(cell.cell_id)
        if key is None:
            continue
        observed[key] += len(episode.decisions)
        episodes_seen[key] += 1
    rows: list[dict[str, Any]] = []
    for key in sorted(set(planned) | set(observed)):
        protocol, checkpoint, cohort, split = key
        want = planned.get(key, 0)
        have = observed.get(key, 0)
        rows.append({
            "protocol": protocol, "checkpoint": checkpoint, "cohort": cohort, "split": split,
            "planned_decisions": want, "recorded_decisions": have,
            "episodes": episodes_seen.get(key, 0),
            "coverage": (have / want) if want else None,
        })
    return rows


def coverage_gate(
    run_root: str | os.PathLike,
    cells: Sequence[ArdCellSpec],
    *,
    hour: int,
    thresholds: Mapping[int, Mapping[str, float]] = COVERAGE_THRESHOLDS,
    write: bool = True,
) -> dict[str, Any]:
    """Assert decision coverage at hour 12, 18 or 24 (plan §7).

    Every protocol x checkpoint x cohort cell of the tier is scored separately: an aggregate
    "90% complete" can be 100% of PEER and 0% of GATE, and it is GATE that carries two of the
    three primaries.
    """
    if hour not in thresholds:
        raise GateError(f"no registered coverage thresholds for hour {hour}; "
                        f"have {sorted(thresholds)}")
    floors = thresholds[hour]
    rows = decision_coverage(run_root, cells)
    checks: list[GateCheck] = []
    for row in rows:
        floor = floors.get(row["split"])
        if floor is None:
            continue
        coverage = row["coverage"]
        name = (f"coverage:{row['protocol']}:{row['checkpoint']}:{row['cohort']}:"
                f"{row['split']}")
        checks.append(GateCheck(
            name=name, passed=coverage is not None and coverage >= floor,
            value=coverage, threshold=floor,
            note=f"{row['recorded_decisions']}/{row['planned_decisions']} logical decisions"))
    passed = all(c.passed for c in checks)
    report = {
        "gate": "coverage",
        "hour": int(hour),
        "study_id": K.STUDY_ID,
        "run_root": str(run_root),
        "passed": passed,
        "thresholds": dict(floors),
        "rows": rows,
        "checks": [c.to_dict() for c in checks],
        "failed": [c.name for c in checks if not c.passed],
        "cells": ard_cell_reconciliation(run_root, cells),
        "legacy_reconcile": _legacy_rates(Path(run_root)),
        "remediation": (
            "One re-dispatch of the failing shard; if still short at the next checkpoint, drop "
            "the shard from the training frame and log an amendment (plan §7).  At hour 24 a "
            "shortfall triggers §16.4 steps 4-5 now."),
    }
    if write:
        io.write_json(Path(run_root) / "gates" / f"coverage_gate_h{int(hour)}.json", report)
    return report


def _legacy_rates(run_root: Path) -> dict[str, Any]:
    """``study/reconcile.py``'s judge/vote readouts, carried through when they apply."""
    out: dict[str, Any] = {}
    try:
        out["judge"] = SR.judge_rates(run_root)
        out["vote"] = SR.vote_rates(run_root)
    except Exception as exc:  # noqa: BLE001 - a legacy-only readout must never fail the gate
        out["error"] = str(exc)[:200]
    return out


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.gates",
                                description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--run-root", required=True)
    sub = p.add_subparsers(dest="gate", required=True)
    launch = sub.add_parser("launch", help="hour-2 tier freeze (spec §16.3)")
    launch.add_argument("--measured", required=True, help="JSON file of the hour-2 measurements")
    launch.add_argument("--required-replicas", required=True,
                        help="JSON file: {profile: {partition: count}}")
    launch.add_argument("--partitions", default=None, help="JSON file: {host: partition}")
    launch.add_argument("--hardware", default="ten_gpus", choices=["ten_gpus", "twenty_gpus"])
    launch.add_argument("--no-budget-check", action="store_true")
    coverage = sub.add_parser("coverage", help="hour-12/18/24 decision coverage")
    coverage.add_argument("--hour", type=int, required=True, choices=sorted(COVERAGE_THRESHOLDS))
    coverage.add_argument("--cells-file", action="append", required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gate == "launch":
        measured = json.loads(Path(args.measured).read_text(encoding="utf-8"))
        required = json.loads(Path(args.required_replicas).read_text(encoding="utf-8"))
        partitions = (json.loads(Path(args.partitions).read_text(encoding="utf-8"))
                      if args.partitions else None)
        report = launch_gate(args.run_root, measured=measured, required_replicas=required,
                             partition_of=partitions, hardware=args.hardware,
                             run_budget=not args.no_budget_check)
    else:
        from agents_scaling.ard.cells import load_ard_cells_file

        cells: list[ArdCellSpec] = []
        for path in args.cells_file:
            cells.extend(load_ard_cells_file(path))
        report = coverage_gate(args.run_root, cells, hour=args.hour)
    print(json.dumps({"gate": report["gate"], "passed": report["passed"],
                      "failed": report["failed"],
                      **({"tier": report["tier"]} if "tier" in report else {})}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CALL_TYPES", "COVERAGE_THRESHOLDS", "DECISIONS_PER_ITEM", "GateCheck", "GateError",
    "ard_cell_reconciliation", "coverage_gate", "decision_coverage", "launch_gate",
    "live_replica_census", "run_budget_check",
]
