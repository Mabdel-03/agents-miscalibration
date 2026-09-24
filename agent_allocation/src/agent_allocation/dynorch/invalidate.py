"""Invalidate the records a stage completed under the pre-latch upstream-failure defect (contract §10.38).

Before the latch, an upstream failure (``UpstreamFailed``: an ambiguous outcome, a provider's "Stream interrupted"
error envelope, a 5xx after the upstream retries, ``forced_think_short``, an account refusal) settled the call's full
grant and returned 502 -- and both harnesses retry a 502. The retry found ``remaining < 64`` and got the B4 TERMINAL
call, so the run ended ``done`` with a no-thinking answer that was scored as a model outcome of an infrastructure
failure (main stage 2026-09-24: H1/16 omni_math_hard/3907 and /1257, H1/25 aime/2026/23, ...). H0 never retries, so
only the harness arms were biased.

``allocation dynorch-invalidate --config CFG --run-dir RUN --stage S --reason "<text>" [--dry-run]``:

1. finds every CURRENT record of stage ``S`` (``runs/S/...``) whose status is not ``transport_failed`` and whose
   attempt experienced an upstream failure, read from the attempt's call journal (a call payload whose ``error.code``
   is ``upstream_failed``), its ``calls_detail`` (a call returned 502) and its ``error`` (``upstream_failed:`` /
   ``infrastructure_stop:``) -- each finding lists its evidence and whether a TERMINAL call followed the failure;
2. refuses (nothing moved) while the stage's job may still be running: a record of ``S`` still ``running``, a ledger
   attempt of ``S`` still open, or ``state/status.json`` saying stage ``S`` is ``running`` (``--dry-run`` reports
   these blockers and still lists the cells; ``--ignore-status-file`` drops only the last check, for a job Slurm
   killed before it could rewrite ``status.json``);
3. per finding: copies the original record verbatim to
   ``superseded_executions/S/infra_invalidation_<utc>/originals/...``, writes the record -- rewritten to
   ``status = transport_failed``, ``rerun_reason = infra_invalidated``, the scored fields cleared and the original
   values kept under ``harness_reported["infra_invalidation"]`` -- to its attempt's ``runs_superseded/`` path, and
   removes it from ``runs/``; writes ``INVALIDATION.json`` (layout ``dynorch/infra_invalidation/v1``) and records
   ``live.record_decision(kind="infra_invalidation", cells=[...], reason=...)``.

The next ``dynorch-run --stage S --resume`` then re-executes exactly those cells: the runner takes a cell's highest
superseded attempt as its current record when ``runs/`` holds none, and ``infra_invalidated`` is an infrastructure
rerun -- attempt + 1 (a new run_key, so new ledger attempt ids and a new journal directory), the next execution
generation, a fresh budget, and no model rerun consumed. Until then the tables show the cell as ``unknown`` (y NaN):
the invalidated attempt is never the cell's scored attempt, and its dollars stay in ``retry_cost_usd``.

NEVER touched: the ledger journal (read only, for the open-attempt check), ``locks/``, the stage's plan manifest,
the call journals (``raw/``), the scratch work dirs, and every other stage.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.apikey import redact
from agent_allocation.dynorch.records import RunRecord, from_json, item_slug, to_json
from agent_allocation.dynorch.supersede import SUPERSEDED_DIR
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic

__all__ = ["INFRA_INVALIDATED", "InvalidateRefused", "LAYOUT", "Finding", "find_invalid", "invalidate_stage",
           "stage_blockers", "upstream_failures"]

LAYOUT = "dynorch/infra_invalidation/v1"
INFRA_INVALIDATED = "infra_invalidated"          # == runner.INFRA_INVALIDATED (kept import-light)
_SCORED_FIELDS = ("parse_status", "final_answer", "grade_status", "correct", "grade_detail")


class InvalidateRefused(RuntimeError):
    """The stage cannot be invalidated as it stands (nothing was moved)."""


@dataclass
class Finding:
    run_key: str
    arm: str
    rung: int
    repeat: int
    item_id: str
    attempt: int
    generation: int
    status: str
    rerun_reason: str | None
    parse_status: str | None
    correct: bool | None
    record_path: str
    failures: list[dict[str, Any]] = field(default_factory=list)
    terminal_after_failure: bool = False
    final_from_terminal: bool = False

    @property
    def cell(self) -> str:
        return f"{self.arm}/{self.rung}/r{self.repeat}/{self.item_id}@a{self.attempt}"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _journal_dir(run_root: Path, record: RunRecord) -> Path:
    spec = record.spec
    return (run_root / "raw" / spec.partition / spec.arm / str(spec.rung) / str(spec.repeat) / item_slug(spec.item_id)
            / f"a{spec.attempt}")


def upstream_failures(run_root: str | Path, record: RunRecord) -> list[dict[str, Any]]:
    """Every upstream failure of ``record``'s attempt, one entry per failed call (sorted by call number):
    ``{call_no, conversation_class, transport_status, detail, sources}``. Sources: ``journal`` (the call's journaled
    payload carries ``error.code == "upstream_failed"``), ``calls_detail`` (the call was returned with HTTP 502) and
    ``record_error`` (the record's last error is ``upstream_failed: ...`` / ``infrastructure_stop: ...``; attached to
    the last failed call, or on its own when neither other source saw one)."""
    found: dict[int, dict[str, Any]] = {}

    def note(call_no: int, source: str, **info: Any) -> None:
        entry = found.setdefault(int(call_no), {"call_no": int(call_no), "conversation_class": None,
                                                "transport_status": None, "detail": None, "sources": []})
        for key, value in info.items():
            if entry.get(key) is None and value is not None:
                entry[key] = value
        if source not in entry["sources"]:
            entry["sources"].append(source)

    journal = _journal_dir(Path(run_root), record)
    if journal.is_dir():
        for file in sorted(journal.glob("call_*.json.gz")):         # this attempt's own calls, not superseded/
            try:
                doc = json.loads(gzip.decompress(file.read_bytes()).decode("utf-8"))
            except (OSError, ValueError, EOFError):
                continue
            if doc.get("run_key") != record.spec.run_key:
                continue
            error = doc.get("error") or {}
            if isinstance(error, Mapping) and error.get("code") == "upstream_failed":
                call = doc.get("call") or {}
                note(int(doc.get("call_no") or call.get("call_no") or 0), "journal",
                     conversation_class=call.get("conversation_class"), transport_status=call.get("transport_status"),
                     detail=error.get("detail"))
    for call in record.calls_detail:
        if int(call.returned_status) == 502 and call.transport_status not in ("ok", "cache"):
            note(call.call_no, "calls_detail", conversation_class=call.conversation_class,
                 transport_status=call.transport_status)
    error = str(record.error or "")
    if error.startswith("upstream_failed:") or error.startswith("infrastructure_stop:"):
        if found:
            note(max(found), "record_error", detail=error.split(":", 1)[1].strip())
        else:
            note(0, "record_error", detail=error.split(":", 1)[1].strip())
    return [found[k] for k in sorted(found)]


def _finding(run_root: Path, record: RunRecord, path: Path) -> Finding | None:
    failures = upstream_failures(run_root, record)
    if not failures:
        return None
    first = min((f["call_no"] for f in failures if f["call_no"] > 0), default=0)
    terminal_calls = [c.call_no for c in record.calls_detail if c.forced_close_kind == "terminal"]
    after = [n for n in terminal_calls if n > first] if first else terminal_calls
    final_from_terminal = bool(after) and record.forced_close_kind == "terminal" and record.final_message is not None
    spec = record.spec
    return Finding(run_key=spec.run_key, arm=spec.arm, rung=int(spec.rung), repeat=int(spec.repeat),
                   item_id=spec.item_id, attempt=int(spec.attempt), generation=int(spec.generation),
                   status=record.status, rerun_reason=record.rerun_reason, parse_status=record.parse_status,
                   correct=record.correct, record_path=str(path.relative_to(run_root)), failures=failures,
                   terminal_after_failure=bool(after), final_from_terminal=final_from_terminal)


def _load(path: Path) -> RunRecord:
    return from_json(RunRecord, json.loads(path.read_text(encoding="utf-8")))


def find_invalid(run_root: str | Path, stage: str) -> list[Finding]:
    """The current records of ``stage`` that experienced an upstream failure yet did not end ``transport_failed``
    (read only). A ``running`` record is not judged (the stage blockers report it)."""
    root = Path(run_root)
    out: list[Finding] = []
    base = root / "runs" / stage
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("*.json")):
        record = _load(path)
        if record.spec.partition != stage or record.status in ("transport_failed", "running", "planned"):
            continue
        finding = _finding(root, record, path)
        if finding is not None:
            out.append(finding)
    return out


def stage_blockers(paths: RunPaths, stage: str, *, ignore_status_file: bool = False) -> list[str]:
    """Why the stage's job may still be running (empty = safe to move records). ``ignore_status_file`` drops only the
    ``state/status.json`` check (a job Slurm killed outright never rewrote it); running records and open ledger
    attempts always block."""
    root = Path(paths.run_dir)
    problems: list[str] = []
    running: list[str] = []
    for sub in ("runs", "runs_superseded"):
        base = root / sub / stage
        if base.is_dir():
            for path in sorted(base.rglob("*.json")):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if doc.get("status") == "running":
                    running.append(str((doc.get("spec") or {}).get("run_key")))
    if running:
        problems.append(f"{len(running)} record(s) of stage {stage} still running (e.g. {running[:3]})")
    if paths.ledger_journal.exists():
        from agent_allocation.runtime.ledger import journal_attempts
        open_ids = sorted(aid for aid, info in journal_attempts(paths.ledger_journal).items()
                          if info.partition_id == stage and info.resolution is None)
        if open_ids:
            problems.append(f"{len(open_ids)} open ledger attempt(s) of stage {stage} (e.g. {open_ids[:3]})")
    status_json = paths.status_json
    if status_json.exists() and not ignore_status_file:
        try:
            doc = json.loads(status_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
        if doc.get("stage") == stage and doc.get("state") in ("starting", "running"):
            problems.append(f"{status_json.relative_to(root)} says stage {stage} is {doc.get('state')} "
                            f"(started {doc.get('started_at')})")
    return problems


def _rewritten(record: RunRecord, reason: str, stamp: str, finding: Finding) -> RunRecord:
    original = {"status": record.status, "rerun_reason": record.rerun_reason, "error": record.error,
                **{name: getattr(record, name) for name in _SCORED_FIELDS}}
    reported = dict(record.harness_reported or {})
    reported["infra_invalidation"] = {"reason": reason, "recorded_at": stamp, "layout": LAYOUT, "original": original,
                                      "failures": finding.failures,
                                      "terminal_after_failure": finding.terminal_after_failure,
                                      "final_from_terminal": finding.final_from_terminal}
    record.harness_reported = reported
    record.status = "transport_failed"
    record.rerun_reason = INFRA_INVALIDATED
    record.error = f"{INFRA_INVALIDATED}: {reason}"
    for name in _SCORED_FIELDS:
        setattr(record, name, None)
    return record


def _superseded_path(run_root: Path, record: RunRecord) -> Path:
    spec = record.spec
    return (run_root / "runs_superseded" / spec.partition / spec.arm / str(spec.rung) / str(spec.repeat)
            / f"{item_slug(spec.item_id)}.a{spec.attempt}.json")


def invalidate_stage(paths: RunPaths, stage: str, *, reason: str, dry_run: bool = False,
                     ignore_status_file: bool = False, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """See the module docstring. Returns the report (``INVALIDATION.json`` for a real run)."""
    if not isinstance(reason, str) or not reason.strip():
        raise InvalidateRefused("--reason must be a non-empty text (it is recorded with the decision)")
    reason = reason.strip()
    root = Path(paths.run_dir)
    findings = find_invalid(root, stage)
    blockers = stage_blockers(paths, stage, ignore_status_file=ignore_status_file)
    report: dict[str, Any] = {
        "layout": LAYOUT, "stage": stage, "reason": reason, "dry_run": bool(dry_run), "blocked": blockers,
        "count": len(findings),
        "by_arm_rung": _count_by(findings),
        "cells": [{**asdict(f), "cell": f.cell} for f in findings],
        "untouched": ["state/ledger_journal.jsonl", "locks/", f"manifests/dynorch_plan_{stage}.jsonl", "raw/",
                      "scratch work dirs", "every other stage"], **dict(extra or {})}
    if dry_run:
        return report
    if blockers:
        raise InvalidateRefused("the stage's job may still be running: " + "; ".join(blockers))
    if not findings:
        report["moved"] = []
        return report
    stamp = _utc()
    target = root / SUPERSEDED_DIR / stage / f"infra_invalidation_{stamp}"
    if target.exists():
        raise InvalidateRefused(f"{target} already exists")
    # every destination is checked before anything moves
    for finding in findings:
        record = _load(root / finding.record_path)
        dest = _superseded_path(root, record)
        if dest.exists():
            raise InvalidateRefused(f"{dest.relative_to(root)} already exists (attempt {finding.attempt} of "
                                    f"{finding.cell} was superseded before); nothing moved")
    moved: list[dict[str, str]] = []
    for finding in findings:
        source = root / finding.record_path
        original_bytes = source.read_bytes()
        record = _rewritten(_load(source), reason, stamp, finding)
        dest = _superseded_path(root, record)
        keep = target / "originals" / finding.record_path
        keep.parent.mkdir(parents=True, exist_ok=True)
        replace_atomic(keep, original_bytes)                          # verbatim copy of what was scored
        data = json.dumps(to_json(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        replace_atomic(dest, (redact(data) or data).encode("utf-8"))  # the invalidated attempt, superseded
        source.unlink()                                               # the cell has no current record now
        moved.append({"cell": finding.cell, "from": finding.record_path, "to": str(dest.relative_to(root)),
                      "original": str(keep.relative_to(root))})
    report["moved"] = moved
    report["recorded_at"] = stamp
    report["target"] = str(target.relative_to(root))
    replace_atomic(target / "INVALIDATION.json",
                   json.dumps(report, indent=1, sort_keys=True, default=str).encode("utf-8"))
    from agent_allocation.live import record_decision
    record_decision(paths, kind="infra_invalidation", stage=stage, reason=reason, target=report["target"],
                    count=len(findings),
                    cells=[{"arm": f.arm, "rung": f.rung, "repeat": f.repeat, "item_id": f.item_id,
                            "attempt": f.attempt, "run_key": f.run_key} for f in findings])
    return report


def _count_by(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        key = f"{f.arm}/{f.rung}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))

