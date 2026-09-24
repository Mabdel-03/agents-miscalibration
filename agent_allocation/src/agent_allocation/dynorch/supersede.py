"""Supersede a stage execution that ran under a defect, so the stage can be re-executed (contract §10.35).

``dynorch-run --stage S --supersede "<reason>"`` calls ``supersede_stage`` before it launches the stage. The stage's
records and raw call journals were produced by code that has since been fixed (the canary of 2026-09-24 spent its
one grace on tool calls, contract §10.28), so they must neither be graded nor block the re-execution -- but they are
paid evidence and are never deleted. ``supersede_stage``:

1. refuses when the stage has a record still ``running`` or a ledger attempt still open (resolve them first with
   ``--resume``: an inherited SENT attempt must be settled from its call journal while the journal is in place);
2. picks the NEW execution generation: one above every generation the stage's records and the ledger journal's
   attempt ids (``{run_key}:g{gen}:...``, partition == the stage) have used, so the re-executed cells -- which keep
   their run_keys, their pre-registration and their record paths -- never reuse a ledger attempt id;
3. moves the stage's ``runs/<S>``, ``runs_superseded/<S>``, ``raw/<S>``, its ``state/`` files (reconciliation
   before/after, stage marker) and its gate records (``gates/*_<S>.json``) to
   ``superseded_executions/<S>/g<old>_<utc>/`` next to a ``SUPERSEDED.json`` carrying the reason, and the stage's
   scratch work dirs to ``<scratch>/work_superseded/<S>/g<old>_<utc>/`` (the runner APPENDS to a work dir's stdout, so
   a reused dir would feed the old run's output to the new parse);
4. writes the generation to ``state/stage_generations.json`` and records the decision (``live.record_decision``).

It NEVER touches the ledger journal (the cumulative record of real spend; it is only read), the endpoint lock
(``locks/``), the stage's pre-registration manifest (the planned cells are unchanged), or any other stage.
``tables.read_run_records`` reads ``runs/`` and ``runs_superseded/`` only, so a superseded execution is no row of
the tables; its dollars stay in the ledger and therefore in every reconciliation of the cumulative spend.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.dynorch.journal import parse_attempt_id
from agent_allocation.dynorch.records import RunRecord, RunSpec, from_json
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic

__all__ = ["GENERATIONS_FILE", "SUPERSEDED_DIR", "SupersedeRefused", "apply_generation", "execution_generation",
           "supersede_stage"]

GENERATIONS_FILE = "stage_generations.json"
SUPERSEDED_DIR = "superseded_executions"
LAYOUT = "dynorch/supersede/v1"


class SupersedeRefused(RuntimeError):
    """The stage cannot be superseded as it stands (nothing was moved)."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _generations_path(paths: RunPaths) -> Path:
    return paths.state_dir / GENERATIONS_FILE


def _read_generations(paths: RunPaths) -> dict[str, Any]:
    target = _generations_path(paths)
    if not target.exists():
        return {"layout": LAYOUT, "stages": {}}
    doc = json.loads(target.read_text("utf-8"))
    if doc.get("layout") != LAYOUT or not isinstance(doc.get("stages"), dict):
        raise SupersedeRefused(f"{target} is not a {LAYOUT} generations file")
    return doc


def execution_generation(run_root: str | Path, stage: str) -> int:
    """The execution generation the stage's attempt-1 cells run under: 1 unless the stage was superseded."""
    doc = _read_generations(RunPaths.for_run(run_root))
    entry = doc["stages"].get(stage)
    return int(entry["generation"]) if entry else 1


def apply_generation(specs: Sequence[RunSpec], generation: int) -> list[RunSpec]:
    """``plan_stage`` stamps ``generation = 1``; a superseded stage re-executes its cells under a higher one (the
    ledger attempt ids embed it). Only first attempts are planned, so the base generation is replaced outright."""
    if generation <= 1:
        return list(specs)
    return [dataclasses.replace(s, generation=int(generation)) if s.attempt == 1 else s for s in specs]


def _records(root: Path) -> list[RunRecord]:
    out: list[RunRecord] = []
    if root.is_dir():
        for file in sorted(root.rglob("*.json")):
            out.append(from_json(RunRecord, json.loads(file.read_text(encoding="utf-8"))))
    return out


def _stage_ledger_attempts(paths: RunPaths, stage: str) -> dict[str, Any]:
    """The ledger journal's attempts of this stage (READ ONLY; ``partition_id`` is the stage)."""
    from agent_allocation.runtime.ledger import journal_attempts
    if not paths.ledger_journal.exists():
        return {}
    return {aid: info for aid, info in journal_attempts(paths.ledger_journal).items() if info.partition_id == stage}


def _move(source: Path, target: Path, moved: list[str], base: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    moved.append(str(source.relative_to(base)) if source.is_relative_to(base) else str(source))


def supersede_stage(paths: RunPaths, stage: str, *, reason: str, scratch_root: str | Path | None = None,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """See the module docstring. Returns the ``SUPERSEDED.json`` document (also recorded as a decision)."""
    from agent_allocation.live import record_decision

    if not isinstance(reason, str) or not reason.strip():
        raise SupersedeRefused("--supersede needs a non-empty reason (it is recorded with the decision)")
    run_root = Path(paths.run_dir)
    records = _records(run_root / "runs" / stage) + _records(run_root / "runs_superseded" / stage)
    raw_stage = run_root / "raw" / stage
    if not records and not raw_stage.exists():
        raise SupersedeRefused(f"stage {stage} has no records or call journals to supersede")
    running = sorted(r.spec.run_key for r in records if r.status == "running")
    if running:
        raise SupersedeRefused(f"stage {stage} has {len(running)} record(s) still running (e.g. {running[:3]}): finish "
                               f"or --resume them first so their ledger attempts are resolved from the call journals")
    attempts = _stage_ledger_attempts(paths, stage)
    open_ids = sorted(aid for aid, info in attempts.items() if info.resolution is None)
    if open_ids:
        raise SupersedeRefused(f"stage {stage} has {len(open_ids)} open ledger attempt(s) (e.g. {open_ids[:3]}): "
                               f"--resume resolves them from the call journals before anything is moved")
    used = {int(r.spec.generation) for r in records}
    for aid in attempts:
        try:
            used.add(int(parse_attempt_id(aid)["generation"]))
        except ValueError:
            continue
    generations = _read_generations(paths)
    previous = int((generations["stages"].get(stage) or {}).get("generation", 1))
    used.add(previous)
    new_generation = max(used) + 1

    stamp = _utc()
    target = run_root / SUPERSEDED_DIR / stage / f"g{previous}_{stamp}"
    if target.exists():
        raise SupersedeRefused(f"{target} already exists")
    moved: list[str] = []
    for sub in ("runs", "runs_superseded", "raw"):
        _move(run_root / sub / stage, target / sub / stage, moved, run_root)
    for name in (f"reconcile_{stage}_before.json", f"reconciliation_{stage}.json", f"stage_{stage}_started"):
        _move(paths.state_dir / name, target / "state" / name, moved, run_root)
    gates_dir = run_root / "gates"
    if gates_dir.is_dir():
        for gate in sorted(gates_dir.glob(f"*_{stage}.json")):
            _move(gate, target / "gates" / gate.name, moved, run_root)
    scratch_moved = None
    if scratch_root is not None:
        work = Path(scratch_root) / "work" / stage
        if work.exists():
            scratch_target = Path(scratch_root) / "work_superseded" / stage / f"g{previous}_{stamp}"
            scratch_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(work), str(scratch_target))
            scratch_moved = str(scratch_target)

    doc = {"layout": LAYOUT, "stage": stage, "reason": reason.strip(), "recorded_at": stamp,
           "superseded_generation": previous, "new_generation": new_generation,
           "records": len(records), "run_keys": sorted({r.spec.run_key for r in records}),
           "ledger_attempts": len(attempts),
           "ledger_settled_usd": str(sum((info.resolved_usd or 0) for info in attempts.values())),
           "moved": moved, "scratch_work_moved_to": scratch_moved, "target": str(target.relative_to(run_root)),
           "untouched": ["state/ledger_journal.jsonl", "locks/", f"manifests/dynorch_plan_{stage}.jsonl",
                         "every other stage"], **dict(extra or {})}
    target.mkdir(parents=True, exist_ok=True)
    replace_atomic(target / "SUPERSEDED.json", json.dumps(doc, indent=1, sort_keys=True, default=str).encode("utf-8"))
    history = list((generations["stages"].get(stage) or {}).get("history", []))
    history.append({"superseded_generation": previous, "new_generation": new_generation, "reason": reason.strip(),
                    "at": stamp, "target": doc["target"]})
    generations["stages"][stage] = {"generation": new_generation, "history": history}
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    replace_atomic(_generations_path(paths), json.dumps(generations, indent=1, sort_keys=True).encode("utf-8"))
    record_decision(paths, kind="dynorch_supersede", stage=stage, reason=reason.strip(),
                    superseded_generation=previous, new_generation=new_generation, records=len(records),
                    ledger_attempts=len(attempts), target=doc["target"], moved=len(moved))
    return doc
