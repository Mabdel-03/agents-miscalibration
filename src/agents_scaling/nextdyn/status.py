"""The outcome-blind monitor: what a run is doing, never how it is turning out.

Plan §3.6 (the signal table), §3.5 S4, spec 17.5.  ``nextdyn status`` is the only thing a
human looks at while the run is in flight, so it is *structurally* incapable of showing an
arm-labelled accuracy: it reads a fixed allowlist of paths (``cells/*/meta.json``,
``cells/*/events.jsonl``, the pool registries, ``status/``, ``registers/``, ``plan/``) and its
import closure contains no module that can open ``eval/``, ``protected/`` or a table with a
grade column (``test_status_never_opens_eval_or_grades_private`` asserts both halves).

Why that matters more than it sounds: a dashboard that shows accuracy while the run is live
turns every operational decision -- top up this lane, cancel that one, extend the pilot -- into
an outcome-dependent one, and no amount of analysis discipline afterwards can undo it.

The alert rules are the ARD36/study_v4 stall detectors, each with the failure it caught:

=====================================  ==========================================================
rule                                   what it caught
=====================================  ==========================================================
running cells > 0, no completion       the "sacct PENDING loop": the queue was full of tasks that
  in ``stall_alert_min`` minutes       could never start and completions flatlined
writers == 0 while tasks are PENDING   the login-node reaper killing every worker
tasks PENDING > ``pending_alert_min``  ``QOSMaxSubmitJobPerUserLimit`` / ``QOSGrpGRES`` /
  with a reason                        ``QOSMaxMemPerUser`` / ``Resources``
live replicas < floor                  a keepalive that stopped launching (or ``FLEET_STOP``)
alias rate != the planned reuse        broken CRN keying (0 in E3a) or an id collision (> plan)
disk > the storage limit               the run filling the data filesystem
``SUSPENDED.json`` present             a harness defect that needs a human
a tampered register                    something edited a sealed set
=====================================  ==========================================================
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import cli, config as C, deps, fleet as FL
from agents_scaling.nextdyn.kinds import Lane

STATUS_DIR: Final[str] = "status"
#: Two spellings of one fact, held together by ``test_the_monitor_and_the_planner_agree_on_the
#: _names``: the monitor may not import ``plan`` (that module reaches the writer side through
#: its lazy imports, and a monitor one refactor away from a grade is not outcome-blind), so the
#: two path names it needs are spelled here and reconciled by a test.
PLAN_DIR: Final[str] = "plan"
DEPS_UNMET_FILE: Final[str] = "DEPS_UNMET.json"
DRY_RUN_GLOB: Final[str] = "dry_run_*.json"
ALERTS_FILE: Final[str] = "alerts.jsonl"
LATEST_FILE: Final[str] = "latest.json"
EVENTS_FILE: Final[str] = "events.jsonl"

#: The only path prefixes this module may open, relative to the run root (plan §3.6).
PATH_ALLOWLIST: Final[tuple[str, ...]] = (
    deps.CELLS_DIR, deps.REGISTERS_DIR, STATUS_DIR, PLAN_DIR, FL.POOLS_DIR, "logs",
)
#: Path segments this module must never open, whatever the caller asks for.
PATH_DENYLIST: Final[tuple[str, ...]] = ("eval", "protected", "gold", "tables", "derived")

SECONDS_PER_MINUTE: Final[float] = 60.0

ALERT_STALLED: Final[str] = "stalled"
ALERT_NO_WRITERS: Final[str] = "no_writers"
ALERT_PENDING: Final[str] = "pending_too_long"
ALERT_BELOW_FLOOR: Final[str] = "replicas_below_floor"
ALERT_HTTP: Final[str] = "http_error_rate"
ALERT_ALIAS: Final[str] = "alias_rate"
ALERT_DISK: Final[str] = "disk"
ALERT_SUSPENDED: Final[str] = "suspended_cells"
ALERT_TAMPERED: Final[str] = "tampered_register"
ALERT_INFLIGHT: Final[str] = "inflight_per_replica"


def assert_allowed(run_root: Path, path: Path) -> Path:
    """Refuse any path outside the allowlist (the monitor's firewall, R7)."""
    try:
        relative = path.resolve().relative_to(Path(run_root).resolve())
    except ValueError as exc:
        raise PermissionError(f"status may not read {path} (outside {run_root})") from exc
    parts = relative.parts
    if not parts or parts[0] not in PATH_ALLOWLIST or any(p in PATH_DENYLIST for p in parts):
        raise PermissionError(
            f"status may not read {relative}: the monitor is outcome-blind by construction "
            f"and reads only {list(PATH_ALLOWLIST)} (plan §3.6)")
    return path


# --------------------------------------------------------------------------- readings


@dataclass(frozen=True)
class LaneStatus:
    lane: str
    complete: int = 0
    running: int = 0
    waiting: int = 0
    suspended: int = 0
    planned: int = 0
    completions_recent: int = 0


@dataclass(frozen=True)
class Alert:
    rule: str
    detail: str
    at: float = field(default_factory=time.time)

    def line(self) -> str:
        return f"ALERT {self.rule}: {self.detail}"


@dataclass(frozen=True)
class Status:
    """One pass of the monitor, as JSON-able as it is printable."""

    run_root: str
    at: float
    lanes: Mapping[str, LaneStatus]
    live_endpoints: int
    writers: int
    pending_over_limit: tuple[Mapping[str, str], ...]
    http_error_rate: float
    alias_rate: float
    planned_alias_rate: float
    disk_bytes: int
    storage_limit_bytes: int | None
    suspended_cells: tuple[str, ...]
    tampered: tuple[str, ...]
    fleet_stopped: bool
    alerts: tuple[Alert, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_root": self.run_root, "at": self.at,
            "lanes": {name: asdict(lane) for name, lane in sorted(self.lanes.items())},
            "live_endpoints": self.live_endpoints, "writers": self.writers,
            "pending_over_limit": [dict(row) for row in self.pending_over_limit],
            "http_error_rate": self.http_error_rate, "alias_rate": self.alias_rate,
            "planned_alias_rate": self.planned_alias_rate, "disk_bytes": self.disk_bytes,
            "storage_limit_bytes": self.storage_limit_bytes,
            "suspended_cells": list(self.suspended_cells), "tampered": list(self.tampered),
            "fleet_stopped": self.fleet_stopped,
            "alerts": [asdict(a) for a in self.alerts],
        }

    def render(self) -> str:
        lines = [f"run {self.run_root}  live_endpoints={self.live_endpoints} "
                 f"writers={self.writers} fleet_stopped={self.fleet_stopped}"]
        for name, lane in sorted(self.lanes.items()):
            lines.append(f"  lane {name:<8} complete={lane.complete} running={lane.running} "
                         f"waiting={lane.waiting} suspended={lane.suspended} "
                         f"planned={lane.planned} recent={lane.completions_recent}")
        lines.append(f"  alias {self.alias_rate:.3f} (planned {self.planned_alias_rate:.3f})  "
                     f"http_errors {self.http_error_rate:.3f}  "
                     f"disk {self.disk_bytes / BYTES_PER_GB:.1f} GB")
        lines.extend("  " + alert.line() for alert in self.alerts)
        return "\n".join(lines)


BYTES_PER_GB: Final[float] = 1e9


def _sh(cmd: Sequence[str]) -> str:
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True).stdout
    except Exception:                                   # noqa: BLE001 - a monitor never raises
        return ""


def _read_meta(path: Path) -> Mapping[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _events(path: Path, *, since: float) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    out: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if float(event.get("t", 0)) >= since:
                out.append(event)
    except OSError:
        return []
    return out


def lane_statuses(run_root: Path, *, recent_since: float, now: float) -> dict[str, LaneStatus]:
    """Per-lane cell census from ``meta.json`` alone (never from a directory listing)."""
    counts: dict[str, dict[str, int]] = {}
    for directory in sorted((run_root / deps.CELLS_DIR).glob("*")) if (run_root / deps.CELLS_DIR).exists() else []:
        if not directory.is_dir():
            continue
        assert_allowed(run_root, directory)
        meta_path = directory / deps.META_FILE
        lane = Lane.CPU.value
        bucket = "planned"
        if meta_path.exists():
            meta = _read_meta(meta_path) or {}
            lane = str(meta.get("lane") or lane)
            bucket = "complete"
        elif (directory / deps.SUSPENDED_FILE).exists():
            bucket = "suspended"
        elif (directory / DEPS_UNMET_FILE).exists():
            bucket = "waiting"
        elif (directory / "items").exists() and any((directory / "items").glob("*.json")):
            bucket = "running"
        row = counts.setdefault(lane, {"complete": 0, "running": 0, "waiting": 0,
                                       "suspended": 0, "planned": 0, "recent": 0})
        row[bucket] += 1
        if bucket == "complete" and meta_path.stat().st_mtime >= recent_since:
            row["recent"] += 1
    del now
    return {lane: LaneStatus(lane=lane, complete=row["complete"], running=row["running"],
                            waiting=row["waiting"], suspended=row["suspended"],
                            planned=row["planned"], completions_recent=row["recent"])
            for lane, row in counts.items()}


def event_rates(run_root: Path, *, since: float) -> tuple[float, float, int]:
    """``(http error rate, alias rate, distinct writers)`` from the cells' event logs."""
    errors = requests = aliased = generated = 0
    writers: set[str] = set()
    cells_dir = run_root / deps.CELLS_DIR
    for directory in sorted(cells_dir.glob("*")) if cells_dir.exists() else []:
        events_path = directory / EVENTS_FILE
        if not events_path.exists():
            continue
        assert_allowed(run_root, events_path)
        for event in _events(events_path, since=since):
            kind = str(event.get("event", ""))
            if kind in ("request_error", "http_error", "retry", "banned"):
                errors += 1
            if kind in ("request_committed", "cache_hit", "request_generated"):
                requests += 1
            if kind == "cache_hit":
                aliased += 1
            if kind == "request_generated":
                generated += 1
            job = event.get("slurm_job_id") or event.get("host")
            if job:
                writers.add(str(job))
    total = aliased + generated
    error_rate = errors / requests if requests else 0.0
    alias_rate = aliased / total if total else 0.0
    return error_rate, alias_rate, len(writers)


def pending_over(run_root: Path, *, minutes: float, runner: Any = None) -> list[dict[str, str]]:
    """Array tasks PENDING longer than ``minutes``, with the Slurm reason (the loop detector)."""
    raw = (runner or _sh)(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-t", "PD",
                           "-o", "%i|%j|%M|%R"])
    out: list[dict[str, str]] = []
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) < len(PENDING_FIELDS):
            continue
        job_id, name, elapsed, reason = fields[:len(PENDING_FIELDS)]
        if _minutes(elapsed) >= minutes:
            out.append({"job_id": job_id, "name": name, "pending_for": elapsed,
                        "reason": reason.strip()})
    del run_root
    return out


PENDING_FIELDS: Final[tuple[str, ...]] = ("job_id", "name", "elapsed", "reason")
#: Reasons that mean "this will never start on its own" (plan §3.6).
BLOCKING_REASONS: Final[tuple[str, ...]] = (
    "QOSMaxSubmitJobPerUserLimit", "QOSGrpGRES", "QOSMaxMemPerUser", "QOSMaxGRESPerUser",
    "Resources", "AssocMaxSubmitJobLimit",
)


def _minutes(elapsed: str) -> float:
    """Slurm's ``%M`` (``[[dd-]hh:]mm:ss``) in minutes."""
    text = str(elapsed).strip()
    days = 0.0
    if "-" in text:
        head, _, text = text.partition("-")
        days = float(head or 0)
    parts = [float(p or 0) for p in text.split(":")]
    while len(parts) < len(TIME_PARTS):
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts[-len(TIME_PARTS):]
    return days * HOURS_PER_DAY * MINUTES_PER_HOUR + hours * MINUTES_PER_HOUR + minutes \
        + seconds / SECONDS_PER_MINUTE


TIME_PARTS: Final[tuple[str, ...]] = ("hours", "minutes", "seconds")
HOURS_PER_DAY: Final[float] = 24.0
MINUTES_PER_HOUR: Final[float] = 60.0


def inflight_band(compute: Any, lane: str) -> tuple[int, int] | None:
    """The healthy in-flight-per-replica band of one lane (plan §3.6's table).

    Out of band in *either* direction is a finding: below it the throttle or the endpoint
    filter is wrong and the fleet is idle; above it the replica is queueing requests it cannot
    serve and the next thing to happen is an APITimeout storm.
    """
    dispatch = compute.dispatch
    if lane == Lane.SMALL.value:
        return tuple(int(v) for v in dispatch.inflight_target_8b)
    if lane in (Lane.LARGE.value, Lane.AUDIT.value):
        return tuple(int(v) for v in dispatch.inflight_target_32b)
    return None


def tampered_registers(run_root: Path) -> list[str]:
    out: list[str] = []
    base = run_root / deps.REGISTERS_DIR
    for kind in deps.REGISTER_KINDS:
        directory = base / kind
        if not directory.exists():
            continue
        for sealed in sorted(directory.glob("*")):
            status = deps.verify_register(run_root, kind, sealed.name)
            if status.status == deps.STATUS_TAMPERED:
                out.append(f"register:{kind}/{sealed.name}: {status.detail}")
    return out


def disk_usage(run_root: Path) -> int:
    total = 0
    for path in run_root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


# --------------------------------------------------------------------------- the pass


def collect(run_root: str | os.PathLike, *, cfg: Any, compute: Any,
            planned_alias_rate: float | None = None, floor: int | None = None,
            now: float | None = None, squeue: Any = None) -> Status:
    """One monitor pass (module docstring).  Never raises: a monitor that dies is a blind run."""
    root = Path(run_root)
    now = time.time() if now is None else float(now)
    dispatch = compute.dispatch
    stall_minutes = float(dispatch.stall_alert_min)
    recent_since = now - stall_minutes * SECONDS_PER_MINUTE
    lanes = lane_statuses(root, recent_since=recent_since, now=now)
    error_rate, alias_rate, writers = event_rates(root, since=recent_since)
    pending = pending_over(root, minutes=float(dispatch.pending_alert_min), runner=squeue)
    live = FL.live_endpoint_count(root, compute=compute)
    suspended = tuple(sorted(
        d.name for d in (root / deps.CELLS_DIR).glob("*")
        if (d / deps.SUSPENDED_FILE).exists())) if (root / deps.CELLS_DIR).exists() else ()
    tampered = tuple(tampered_registers(root))
    planned = float(planned_alias_rate if planned_alias_rate is not None else 0.0)
    disk = disk_usage(root)
    limit = cfg.budgets.storage_limit_bytes

    alerts: list[Alert] = []
    running = sum(lane.running for lane in lanes.values())
    completions = sum(lane.completions_recent for lane in lanes.values())
    if running and not completions:
        alerts.append(Alert(ALERT_STALLED,
                            f"{running} cells running, 0 completions in {stall_minutes:.0f} min"))
    if pending and not writers:
        alerts.append(Alert(ALERT_NO_WRITERS,
                            f"0 writers while {len(pending)} task(s) are PENDING"))
    for row in pending:
        if any(reason in row["reason"] for reason in BLOCKING_REASONS):
            alerts.append(Alert(ALERT_PENDING,
                                f"{row['name']} pending {row['pending_for']}: {row['reason']}"))
    for lane_name, lane in sorted(lanes.items()):
        band = inflight_band(compute, lane_name)
        if band is None or not live or not lane.running:
            continue
        per_replica = lane.running / live
        if not band[0] <= per_replica <= band[-1]:
            alerts.append(Alert(ALERT_INFLIGHT,
                                f"lane {lane_name}: {per_replica:.0f} cells per live replica "
                                f"outside the healthy band {list(band)}"))
    if floor is not None and live < int(floor):
        alerts.append(Alert(ALERT_BELOW_FLOOR, f"{live} live endpoints < floor {floor}"))
    if error_rate > float(dispatch.http_error_rate_max):
        alerts.append(Alert(ALERT_HTTP, f"HTTP error rate {error_rate:.3f} > "
                                        f"{float(dispatch.http_error_rate_max):.3f}"))
    if planned and alias_rate == 0.0:
        alerts.append(Alert(ALERT_ALIAS, f"alias rate 0 while the mechanical census planned "
                                         f"{planned:.3f}; the CRN keying may be broken"))
    if limit is not None and disk > int(limit):
        alerts.append(Alert(ALERT_DISK, f"{disk / BYTES_PER_GB:.1f} GB > storage limit "
                                        f"{int(limit) / BYTES_PER_GB:.1f} GB"))
    if suspended:
        alerts.append(Alert(ALERT_SUSPENDED, f"{len(suspended)} suspended cell(s): "
                                             f"{list(suspended[:3])}"))
    for detail in tampered:
        alerts.append(Alert(ALERT_TAMPERED, detail))

    return Status(run_root=str(root), at=now, lanes=lanes, live_endpoints=live, writers=writers,
                  pending_over_limit=tuple(pending), http_error_rate=error_rate,
                  alias_rate=alias_rate, planned_alias_rate=planned, disk_bytes=disk,
                  storage_limit_bytes=None if limit is None else int(limit),
                  suspended_cells=suspended, tampered=tampered,
                  fleet_stopped=FL.is_stopped(root), alerts=tuple(alerts))


def write_status(run_root: str | os.PathLike, status: Status) -> Path:
    """``status/latest.json`` plus one appended line per alert (plan §3.6)."""
    directory = Path(run_root) / STATUS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / LATEST_FILE).write_text(json.dumps(status.to_dict(), indent=2, sort_keys=True)
                                         + "\n", encoding="utf-8")
    if status.alerts:
        with open(directory / ALERTS_FILE, "a", encoding="utf-8") as handle:
            for alert in status.alerts:
                handle.write(json.dumps(asdict(alert), sort_keys=True) + "\n")
    return directory / LATEST_FILE


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("status", description="the outcome-blind monitor")
    parser.add_argument("--json", action="store_true", help="print the status as JSON")
    parser.add_argument("--lane", default=None, choices=[l.value for l in Lane])
    parser.add_argument("--loop", type=float, default=None, help="seconds between passes")
    parser.add_argument("--floor", type=int, default=None, help="alert below this many replicas")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    planned = planned_alias_rate(run_root)
    while True:
        status = collect(run_root, cfg=cfg, compute=compute, planned_alias_rate=planned,
                         floor=args.floor)
        write_status(run_root, status)
        if args.json:
            print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        else:
            print(status.render() if args.lane is None else _one_lane(status, args.lane))
        if not args.loop:
            break
        time.sleep(float(args.loop))
    return cli.Report(completed=sum(lane.complete for lane in status.lanes.values()),
                      failed=len(status.suspended_cells), blocked=len(status.alerts),
                      detail=f"{len(status.alerts)} alert(s)")


def _one_lane(status: Status, lane: str) -> str:
    row = status.lanes.get(lane, LaneStatus(lane=lane))
    return (f"lane {lane}: complete={row.complete} running={row.running} waiting={row.waiting} "
            f"suspended={row.suspended} planned={row.planned}")


def planned_alias_rate(run_root: str | os.PathLike) -> float:
    """The mechanical reuse census, read from what ``plan --dry-run`` published (plan §1.3).

    Read as *data* from ``plan/``, not recomputed: recomputing it would mean importing the
    planner, and the planner reaches the writer side.  The number the monitor compares the
    live alias rate against is therefore exactly the number the planner printed -- which is
    also the only version a human has seen.
    """
    candidates = sorted((Path(run_root) / PLAN_DIR).glob(DRY_RUN_GLOB))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rate = payload.get("reuse_census", {}).get("overall_fraction")
        if rate is not None:
            return float(rate)
    return 0.0


__all__ = [
    "ALERTS_FILE", "ALERT_ALIAS", "ALERT_BELOW_FLOOR", "ALERT_DISK", "ALERT_HTTP",
    "ALERT_NO_WRITERS", "ALERT_PENDING", "ALERT_STALLED", "ALERT_SUSPENDED", "ALERT_TAMPERED",
    "ALERT_INFLIGHT", "Alert", "BLOCKING_REASONS", "LATEST_FILE", "LaneStatus",
    "PATH_ALLOWLIST", "PATH_DENYLIST", "inflight_band",
    "DEPS_UNMET_FILE", "DRY_RUN_GLOB", "PLAN_DIR", "STATUS_DIR", "Status", "assert_allowed",
    "collect", "disk_usage", "event_rates", "lane_statuses", "main", "pending_over",
    "planned_alias_rate", "tampered_registers", "write_status",
]
