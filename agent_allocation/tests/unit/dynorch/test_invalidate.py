"""Contract §10.38: ``dynorch-invalidate`` over REAL journaled records of the main stage (2026-09-24), copied read-only
from the run root into ``tests/fixtures/dynorch/infra_invalidation/run_root``:

* H1/16 omni_math_hard/3907 a1 -- call 1 ``ambiguous_timeout`` (ReadError), full grant settled, call 2 TERMINAL,
  ``done`` / ``valid`` / correct;
* H1/16 omni_math_hard/1257 a1 -- call 1 ``error_envelope`` (Mancer: finish ``error`` without an error object),
  call 2 TERMINAL, ``done`` / ``valid``;
* H1/25 aime/2026/23 a1 -- call 1 ``error_envelope`` (Parasail "Stream interrupted", choice code 502), call 2
  TERMINAL, ``done`` / ``valid``;
* negatives: H1/16 aime/2025/28 (a clean ``done``), H0/4 omni_math_hard/1257 (a1 ``transport_failed`` already
  superseded, a2 ``done`` with no failure).
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from agent_allocation.dynorch import tables as T
from agent_allocation.dynorch.cli import cmd_dynorch_invalidate
from agent_allocation.dynorch.invalidate import (
    InvalidateRefused, find_invalid, invalidate_stage, stage_blockers, upstream_failures,
)
from agent_allocation.dynorch.journal import RunStore
from agent_allocation.dynorch.records import RunRecord, from_json
from agent_allocation.paths import RunPaths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "dynorch" / "infra_invalidation" / "run_root"
PROJECT = Path(__file__).resolve().parents[3]
INVALID = ["H1/16/r1/omni_math_hard/1257@a1", "H1/16/r1/omni_math_hard/3907@a1", "H1/25/r1/aime/2026/23@a1"]
REASON = "pre-latch upstream failure answered by a TERMINAL call (contract 10.38)"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "run_root"
    shutil.copytree(FIXTURE, target)
    return target


def _tree_digest(root: Path, *, exclude: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_file() and not rel.startswith(exclude):
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _record(path: Path) -> RunRecord:
    return from_json(RunRecord, json.loads(path.read_text()))


def test_finds_exactly_the_three_real_cells_with_their_evidence(root: Path):
    findings = find_invalid(root, "main")
    assert sorted(f.cell for f in findings) == INVALID
    by_item = {f.item_id: f for f in findings}
    assert by_item["omni_math_hard/3907"].failures[0]["transport_status"] == "ambiguous_timeout"
    assert by_item["omni_math_hard/1257"].failures[0]["transport_status"] == "error_envelope"
    parasail = by_item["aime/2026/23"].failures[0]
    assert parasail["transport_status"] == "error_envelope" and parasail["call_no"] == 1
    assert parasail["conversation_class"] == "root"
    for finding in findings:
        assert finding.status == "done" and finding.parse_status == "valid"
        assert finding.terminal_after_failure and finding.final_from_terminal
        assert set(finding.failures[0]["sources"]) == {"journal", "calls_detail", "record_error"}


def test_each_evidence_source_is_enough_on_its_own(root: Path, tmp_path: Path):
    path = root / "runs" / "main" / "H1" / "25" / "1" / "aime__2026__23.json"
    record = _record(path)
    # the journal alone (the record's calls_detail and error scrubbed)
    scrubbed = _record(path)
    scrubbed.calls_detail = [c for c in scrubbed.calls_detail if c.returned_status != 502]
    scrubbed.error = None
    assert [f["sources"] for f in upstream_failures(root, scrubbed)] == [["journal"]]
    # calls_detail alone (no journal)
    assert [f["sources"] for f in upstream_failures(tmp_path / "nowhere", record)] == [["calls_detail",
                                                                                       "record_error"]]
    clean = _record(root / "runs" / "main" / "H1" / "16" / "1" / "aime__2025__28.json")
    assert upstream_failures(root, clean) == []


def test_dry_run_lists_the_cells_and_moves_nothing(root: Path):
    before = _tree_digest(root, exclude=())
    report = invalidate_stage(RunPaths.for_run(root), "main", reason=REASON, dry_run=True)
    assert report["dry_run"] is True and report["blocked"] == [] and report["count"] == 3
    assert sorted(c["cell"] for c in report["cells"]) == INVALID
    assert report["by_arm_rung"] == {"H1/16": 2, "H1/25": 1}
    assert _tree_digest(root, exclude=()) == before


def test_invalidation_moves_exactly_those_records_and_touches_nothing_else(root: Path):
    paths = RunPaths.for_run(root)
    untouched_before = _tree_digest(root, exclude=("runs", "superseded_executions", "state/run_decisions.jsonl"))
    originals = {c: (root / "runs" / "main" / p).read_bytes() for c, p in [
        ("3907", "H1/16/1/omni_math_hard__3907.json"), ("1257", "H1/16/1/omni_math_hard__1257.json"),
        ("aime23", "H1/25/1/aime__2026__23.json")]}
    report = invalidate_stage(paths, "main", reason=REASON)
    assert report["count"] == 3 and len(report["moved"]) == 3
    # raw journals, the other stage files, the ledger, locks and manifests: byte-identical
    assert _tree_digest(root, exclude=("runs", "superseded_executions", "state/run_decisions.jsonl")) \
        == untouched_before
    # the two negatives stay current; the three are gone from runs/
    left = sorted(str(p.relative_to(root / "runs" / "main")) for p in (root / "runs" / "main").rglob("*.json"))
    assert left == ["H0/4/1/omni_math_hard__1257.json", "H1/16/1/aime__2025__28.json"]
    store = RunStore(root)
    for item, rung in [("omni_math_hard/3907", 16), ("omni_math_hard/1257", 16), ("aime/2026/23", 25)]:
        slug = item.replace("/", "__")
        moved = _record(root / "runs_superseded" / "main" / "H1" / str(rung) / "1" / f"{slug}.a1.json")
        assert moved.status == "transport_failed" and moved.rerun_reason == "infra_invalidated"
        assert moved.parse_status is None and moved.correct is None and moved.final_answer is None
        note = moved.harness_reported["infra_invalidation"]
        assert note["reason"] == REASON and note["original"]["status"] == "done"
        assert note["original"]["parse_status"] == "valid" and note["terminal_after_failure"] is True
        assert moved.cost_usd != "0" and moved.completion_total > 0          # the paid attempt keeps its accounting
        assert store.path(moved.spec).exists() is False
    target = root / report["target"]
    for key, rel in [("3907", "H1/16/1/omni_math_hard__3907.json"), ("aime23", "H1/25/1/aime__2026__23.json")]:
        assert (target / "originals" / "runs" / "main" / rel).read_bytes() == originals[key]
    doc = json.loads((target / "INVALIDATION.json").read_text())
    assert doc["layout"] == "dynorch/infra_invalidation/v1" and doc["reason"] == REASON and len(doc["moved"]) == 3
    decisions = [json.loads(l) for l in paths.run_decisions.read_text().splitlines()]
    assert len(decisions) == 1 and decisions[0]["kind"] == "infra_invalidation"
    assert decisions[0]["reason"] == REASON and decisions[0]["stage"] == "main"
    assert sorted((c["arm"], c["rung"], c["item_id"], c["attempt"]) for c in decisions[0]["cells"]) == [
        ("H1", 16, "omni_math_hard/1257", 1), ("H1", 16, "omni_math_hard/3907", 1), ("H1", 25, "aime/2026/23", 1)]
    # idempotent: nothing left to invalidate
    assert invalidate_stage(paths, "main", reason=REASON)["count"] == 0


def test_an_invalidated_attempt_is_never_the_scored_record(root: Path):
    """Before --resume re-executes it, the cell's current record is the invalidated attempt: y NaN / unknown, never
    its old valid answer; its dollars are the cell's cost."""
    invalidate_stage(RunPaths.for_run(root), "main", reason=REASON)
    records = T.read_run_records(root)
    cell = next(r for r in records if r["spec"]["item_id"] == "aime/2026/23")
    y, state = T.outcome_of(cell)
    assert math.isnan(y) and state == "unknown"
    clean = next(r for r in records if r["spec"]["item_id"] == "aime/2025/28")
    assert T.outcome_of(clean)[1] == "delivered"


@pytest.mark.parametrize("blocker", ["running_record", "open_ledger_attempt", "status_running"])
def test_refuses_while_the_stage_job_may_be_running(root: Path, blocker: str):
    paths = RunPaths.for_run(root)
    if blocker == "running_record":
        path = root / "runs" / "main" / "H1" / "16" / "1" / "aime__2025__28.json"
        doc = json.loads(path.read_text())
        doc["status"] = "running"
        path.write_text(json.dumps(doc))
    elif blocker == "open_ledger_attempt":
        import asyncio
        from decimal import Decimal
        from agent_allocation.runtime.ledger import CollectionLedger
        from agent_allocation.storage.journal import LedgerJournal

        async def reserve() -> None:
            journal = LedgerJournal(paths.ledger_journal, group_commit_ms=0)
            ledger = CollectionLedger(journal, ceiling_usd=Decimal("10"), max_unresolved_usd=Decimal("5"))
            await ledger.reserve(attempt_id="deadbeef:g1:1:think:1", node_id="deadbeef", partition_id="main",
                                 amount_usd=Decimal("0.01"))
            await journal.flush()
            journal.release_writer()

        asyncio.run(reserve())
    else:
        paths.state_dir.mkdir(parents=True, exist_ok=True)
        paths.status_json.write_text(json.dumps({"stage": "main", "state": "running", "started_at": "x"}))
    before = _tree_digest(root, exclude=("state",))
    assert stage_blockers(paths, "main")
    with pytest.raises(InvalidateRefused):
        invalidate_stage(paths, "main", reason=REASON)
    assert _tree_digest(root, exclude=("state",)) == before
    dry = invalidate_stage(paths, "main", reason=REASON, dry_run=True)       # the dry run still lists the cells
    assert dry["count"] == 3 and dry["blocked"]
    # another stage is not blocked by this one's job
    assert stage_blockers(paths, "pilot") == [] or blocker == "open_ledger_attempt"
    # --ignore-status-file lifts only the status.json check
    assert (stage_blockers(paths, "main", ignore_status_file=True) == []) is (blocker == "status_running")


def test_refuses_an_empty_reason(root: Path):
    with pytest.raises(InvalidateRefused):
        invalidate_stage(RunPaths.for_run(root), "main", reason="  ")


def test_the_verb_dry_run_prints_the_cells(root: Path, capsys):
    rc = cmd_dynorch_invalidate(["--config", str(PROJECT / "configs" / "dynorch_v1.json"), "--run-dir", str(root),
                                 "--stage", "main", "--reason", REASON, "--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and sorted(c["cell"] for c in out["cells"]) == INVALID
    assert (root / "runs" / "main" / "H1" / "25" / "1" / "aime__2026__23.json").exists()
