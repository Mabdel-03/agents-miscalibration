"""The monitor: outcome-blind by path allowlist AND import closure, and its stall detectors.

Plan §3.6.  The first test is the one that matters: a dashboard that can show an arm-labelled
accuracy while the run is live turns every operational decision into an outcome-dependent one,
and no analysis discipline afterwards can undo it.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest

from agents_scaling.nextdyn import deps, status as ST, tables
from agents_scaling.nextdyn.kinds import Lane

PACKAGE = Path(ST.__file__).resolve().parent


def complete_cell(run_root: Path, cell_id: str, *, lane: str = "8B", at: float | None = None,
                  items: tuple[str, ...] = ("item0",)) -> Path:
    directory = deps.cell_dir(run_root, cell_id)
    (directory / tables.ITEMS_DIR).mkdir(parents=True, exist_ok=True)
    for item in items:
        (directory / tables.ITEMS_DIR / f"{item}.json").write_text(
            json.dumps({"cell_id": cell_id, "source_item_id": item, "status": "complete"}))
    meta = directory / deps.META_FILE
    meta.write_text(json.dumps({"cell_id": cell_id, "lane": lane, "n_items": len(items)}))
    if at is not None:
        import os

        os.utime(meta, (at, at))
    return directory


def events(run_root: Path, cell_id: str, rows: list[dict]) -> None:
    path = deps.cell_dir(run_root, cell_id) / ST.EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


# --------------------------------------------------------------------------- the firewall


def test_status_never_opens_eval_or_grades_private(tmp_run_root):
    """Two halves: the path allowlist and the import closure.

    The allowlist is what the module may open; the closure is what it could *reach* -- a
    monitor that imported ``grading`` or ``analysis`` would be one refactor away from showing
    an outcome, whatever its current code does.
    """
    for allowed in ("cells", "registers", "status", "plan", "pools", "logs"):
        ST.assert_allowed(tmp_run_root, tmp_run_root / allowed / "x.json")
    for refused in ("eval/scores.json", "protected/tables/grades_private.parquet",
                    "protected/gold/MATH.jsonl", "tables/decisions.parquet",
                    "derived/claim_ledger.csv"):
        with pytest.raises(PermissionError):
            ST.assert_allowed(tmp_run_root, tmp_run_root / refused)
    with pytest.raises(PermissionError):
        ST.assert_allowed(tmp_run_root, Path("/etc/passwd"))

    # The import closure: what ``import agents_scaling.nextdyn.status`` actually pulls in.
    # Module-level imports only, which is what a closure *is* -- a deferred import inside a
    # function body is a call-site decision, and the path allowlist above is what bounds those.
    forbidden = {"grading", "analysis", "outcomes", "e0", "items_build", "construct", "datasets",
                 "inventory", "graders", "banks", "episode", "receiver", "audit", "selectors"}
    seen: set[str] = set()
    frontier = ["status"]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        path = PACKAGE / f"{name}.py"
        if not path.exists():
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.startswith("agents_scaling.nextdyn"):
                tail = node.module.rsplit(".", 1)[-1]
                frontier += [alias.name for alias in node.names] + ([tail] if tail != "nextdyn" else [])
            elif isinstance(node, ast.Import):
                frontier += [alias.name.rsplit(".", 1)[-1] for alias in node.names
                             if alias.name.startswith("agents_scaling.nextdyn")]
    assert not (seen & forbidden), sorted(seen & forbidden)


def test_the_monitor_and_the_planner_agree_on_the_names(tmp_run_root):
    """The two spellings ``status`` keeps rather than importing the planner (see PLAN_DIR)."""
    from agents_scaling.nextdyn import plan as PLAN

    assert ST.PLAN_DIR == PLAN.PLAN_DIR
    assert ST.DEPS_UNMET_FILE == PLAN.DEPS_UNMET_FILE


def test_the_planned_alias_rate_is_read_from_what_the_planner_published(tmp_run_root):
    assert ST.planned_alias_rate(tmp_run_root) == 0.0
    directory = tmp_run_root / ST.PLAN_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dry_run_1000.json").write_text(json.dumps(
        {"reuse_census": {"overall_fraction": 0.21}}))
    (directory / "dry_run_2000.json").write_text(json.dumps(
        {"reuse_census": {"overall_fraction": 0.26}}))
    assert ST.planned_alias_rate(tmp_run_root) == pytest.approx(0.26)     # the newest wins


def test_the_denylist_is_refused_even_under_an_allowed_prefix(tmp_run_root):
    with pytest.raises(PermissionError):
        ST.assert_allowed(tmp_run_root, tmp_run_root / "cells" / "c" / "eval" / "x.json")


# --------------------------------------------------------------------------- readings


def test_the_lane_census_comes_from_meta_json_not_from_a_directory(tmp_run_root):
    now = time.time()
    complete_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0", at=now)
    running = deps.cell_dir(tmp_run_root, "episode.MATH.primary.a.b.pilot.r0.s0")
    (running / tables.ITEMS_DIR).mkdir(parents=True)
    (running / tables.ITEMS_DIR / "item0.json").write_text("{}")
    waiting = deps.cell_dir(tmp_run_root, "episode.MATH.primary.a.b.pilot.r0.s1")
    waiting.mkdir(parents=True)
    (waiting / ST.DEPS_UNMET_FILE).write_text("{}")
    suspended = deps.cell_dir(tmp_run_root, "episode.MATH.primary.a.b.pilot.r0.s2")
    suspended.mkdir(parents=True)
    (suspended / deps.SUSPENDED_FILE).write_text("{}")

    lanes = ST.lane_statuses(tmp_run_root, recent_since=now - 60, now=now)
    small = lanes[Lane.SMALL.value]
    assert small.complete == 1 and small.completions_recent == 1
    cpu = lanes[Lane.CPU.value]                     # no meta.json -> the lane is not yet known
    assert cpu.running == 1 and cpu.waiting == 1 and cpu.suspended == 1


def test_the_event_rates_are_read_from_the_cells_own_logs(tmp_run_root):
    now = time.time()
    events(tmp_run_root, "c0", [
        {"t": now, "event": "request_generated", "slurm_job_id": "1"},
        {"t": now, "event": "request_generated", "slurm_job_id": "1"},
        {"t": now, "event": "cache_hit", "slurm_job_id": "2"},
        {"t": now, "event": "request_error", "slurm_job_id": "2"},
        {"t": now, "event": "request_committed", "slurm_job_id": "2"},
        {"t": now - 10_000, "event": "request_generated", "slurm_job_id": "9"},   # too old
    ])
    error_rate, alias_rate, writers = ST.event_rates(tmp_run_root, since=now - 60)
    assert alias_rate == pytest.approx(1 / 3)
    assert writers == 2
    assert error_rate > 0


def test_pending_over_reads_the_slurm_reason(tmp_run_root):
    raw = "\n".join([
        "12_1|asys-nextdyn-nd1-8B|1:05:00|QOSMaxSubmitJobPerUserLimit",
        "12_2|asys-nextdyn-nd1-8B|0:03:00|Resources",
        "13_1|asys-nextdyn-nd1-cpu|2-01:00:00|QOSMaxMemPerUser",
    ])
    rows = ST.pending_over(tmp_run_root, minutes=60, runner=lambda cmd: raw)
    assert [r["job_id"] for r in rows] == ["12_1", "13_1"]
    assert any(r["reason"] in ST.BLOCKING_REASONS for r in rows)
    assert ST._minutes("1:05:00") == pytest.approx(65.0)
    assert ST._minutes("2-01:00:00") == pytest.approx(2 * 24 * 60 + 60)


# --------------------------------------------------------------------------- alerts


def test_a_stalled_lane_and_a_missing_writer_both_alert(tmp_run_root, study_cfg, compute_cfg):
    running = deps.cell_dir(tmp_run_root, "episode.MATH.primary.a.b.pilot.r0.s0")
    (running / tables.ITEMS_DIR).mkdir(parents=True)
    (running / tables.ITEMS_DIR / "item0.json").write_text("{}")
    pending = "12_1|asys-nextdyn-nd1-8B|3:00:00|QOSGrpGRES"
    status = ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg,
                        squeue=lambda cmd: pending)
    rules = {alert.rule for alert in status.alerts}
    assert ST.ALERT_STALLED in rules            # cells running, nothing completed
    assert ST.ALERT_NO_WRITERS in rules         # and nobody is writing
    assert ST.ALERT_PENDING in rules            # with a reason that never clears itself
    assert status.render().count("ALERT") == len(status.alerts)


def test_a_zero_alias_rate_against_a_planned_reuse_is_an_alert(tmp_run_root, study_cfg,
                                                               compute_cfg):
    status = ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg,
                        planned_alias_rate=0.25, squeue=lambda cmd: "")
    assert ST.ALERT_ALIAS in {a.rule for a in status.alerts}
    assert status.planned_alias_rate == 0.25


def test_a_suspended_cell_and_a_tampered_register_are_both_reported(tmp_run_root, study_cfg,
                                                                    compute_cfg):
    suspended = deps.cell_dir(tmp_run_root, "episode.MATH.primary.a.b.pilot.r0.s3")
    suspended.mkdir(parents=True)
    (suspended / deps.SUSPENDED_FILE).write_text("{}")
    sha = deps.write_register(tmp_run_root, deps.REGISTER_POOL, {"entries": [{"a": "b"}]})
    path = deps.register_path(tmp_run_root, deps.REGISTER_POOL, sha)
    path.write_text(path.read_text().replace('"b"', '"c"'))
    status = ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg, squeue=lambda cmd: "")
    rules = {a.rule for a in status.alerts}
    assert ST.ALERT_SUSPENDED in rules and ST.ALERT_TAMPERED in rules
    assert status.suspended_cells and status.tampered


def test_the_fleet_floor_and_the_stop_file_are_visible(tmp_run_root, study_cfg, compute_cfg):
    status = ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg, floor=12,
                        squeue=lambda cmd: "")
    assert ST.ALERT_BELOW_FLOOR in {a.rule for a in status.alerts}
    (tmp_run_root / "FLEET_STOP").write_text("stop\n")
    assert ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg,
                      squeue=lambda cmd: "").fleet_stopped


def test_the_status_is_written_where_the_overseer_reads_it(tmp_run_root, study_cfg, compute_cfg):
    status = ST.collect(tmp_run_root, cfg=study_cfg, compute=compute_cfg, floor=1,
                        squeue=lambda cmd: "")
    path = ST.write_status(tmp_run_root, status)
    assert path == tmp_run_root / ST.STATUS_DIR / ST.LATEST_FILE
    assert json.loads(path.read_text())["run_root"] == str(tmp_run_root)
    lines = (tmp_run_root / ST.STATUS_DIR / ST.ALERTS_FILE).read_text().splitlines()
    assert len(lines) == len(status.alerts)
    ST.write_status(tmp_run_root, status)                     # appends, never truncates
    assert len((tmp_run_root / ST.STATUS_DIR / ST.ALERTS_FILE).read_text().splitlines()) \
        == 2 * len(status.alerts)
