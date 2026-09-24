"""``dynorch-run --supersede "<reason>"`` (contract §10.35): a stage execution that ran under a defect is moved aside
and every planned cell is re-executed under a new execution generation. Driven through the verb in rehearsal (the
fake transport, $0, no key, no network) and against synthetic run roots for the refusals."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from agent_allocation.config import load_config
from agent_allocation.dynorch import cli as dcli
from agent_allocation.dynorch import supersede as sup
from agent_allocation.dynorch.journal import RunStore, parse_attempt_id
from agent_allocation.dynorch.plan import BaseInputs
from agent_allocation.dynorch.records import new_run_record
from agent_allocation.paths import RunPaths
from agent_allocation.runtime.ledger import journal_attempts

from .conftest import make_spec

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


@pytest.fixture(scope="module")
def base(cfg) -> BaseInputs:
    root = Path(cfg["base_run_root"])
    if not (root / "manifests" / "panels.json").is_file():
        pytest.skip("base run root not readable here")
    return BaseInputs.load(root, model=cfg["primary_model"])


def _config(tmp_path: Path) -> Path:
    doc = json.loads(CFG_PATH.read_text())
    doc["dynorch"]["scratch_root"] = str(tmp_path / "scratch")
    path = tmp_path / "dynorch_rehearsal.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


def _run(config: Path, run_dir: Path, *extra: str) -> int:
    return dcli.cmd_dynorch_run(["--config", str(config), "--run-dir", str(run_dir), "--stage", "canary",
                                 "--max-usd", "1", "--allow-paid", "--rehearsal", "--arms", "H0",
                                 "--concurrency", "2", *extra])


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(file.relative_to(root)).encode() + b"\0" + file.read_bytes())
    return h.hexdigest()


@pytest.mark.timeout(600)
def test_supersede_moves_the_stage_aside_and_reexecutes_it_under_a_new_generation(cfg, base, tmp_path, monkeypatch,
                                                                                    capsys):
    monkeypatch.setenv("ALLOCATION_LOCAL_STATE", str(tmp_path / "local"))
    monkeypatch.delenv(dcli.REHEARSAL_ENV, raising=False)
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    assert _run(config, run_dir) == 0, capsys.readouterr().err
    paths = RunPaths.for_run(run_dir)
    first = RunStore(run_dir).scan()
    assert len(first) == 2 and all(r.spec.generation == 1 and r.status == "done" for r in first.values())
    ledger_before = paths.ledger_journal.read_bytes()
    locks_before = _tree_digest(run_dir / "locks")
    manifest_before = (paths.manifests_dir / "dynorch_plan_canary.jsonl").read_bytes()
    # another stage's files must never be touched
    other_spec = make_spec(rung=2, partition="pilot")
    other_record = new_run_record(other_spec, status="done", started_at="2026-09-24T00:00:00Z")
    other = RunStore(run_dir).save(other_record)
    other_bytes = other.read_bytes()
    (run_dir / "raw" / "pilot").mkdir(parents=True)
    (run_dir / "raw" / "pilot" / "keep").write_text("x")
    old_attempts = set(journal_attempts(paths.ledger_journal))

    reason = "B3 grace offered tools (contract 10.28); canary re-executed under the fix"
    assert _run(config, run_dir, "--resume", "--supersede", reason) == 0, capsys.readouterr().err

    # the ledger journal is the cumulative record of real spend: only appended to, never rewritten
    ledger_after = paths.ledger_journal.read_bytes()
    assert ledger_after.startswith(ledger_before) and len(ledger_after) > len(ledger_before)
    assert _tree_digest(run_dir / "locks") == locks_before
    assert (paths.manifests_dir / "dynorch_plan_canary.jsonl").read_bytes() == manifest_before
    assert other.read_bytes() == other_bytes and (run_dir / "raw" / "pilot" / "keep").read_text() == "x"
    # the previous execution is kept, whole, with its reason
    moved = sorted((run_dir / sup.SUPERSEDED_DIR / "canary").glob("g1_*"))
    assert len(moved) == 1
    doc = json.loads((moved[0] / "SUPERSEDED.json").read_text())
    assert doc["reason"] == reason and doc["superseded_generation"] == 1 and doc["new_generation"] == 2
    assert doc["records"] == 2 and sorted(doc["run_keys"]) == sorted(first)
    assert len(list((moved[0] / "raw" / "canary").rglob("call_*.json.gz"))) == 2
    assert len(list((moved[0] / "runs" / "canary").rglob("*.json"))) == 2
    assert (moved[0] / "state" / "reconciliation_canary.json").is_file()
    assert sorted((tmp_path / "scratch" / "work_superseded" / "canary").glob("g1_*"))
    # every planned cell re-executed: same run_keys (same pre-registration), a new generation, new ledger ids
    second = {k: r for k, r in RunStore(run_dir).scan().items() if r.spec.partition == "canary"}
    assert set(second) == set(first)
    assert all(r.spec.generation == 2 and r.status == "done" and r.spec.attempt == 1 for r in second.values())
    new_attempts = set(journal_attempts(paths.ledger_journal)) - old_attempts
    assert new_attempts and not (new_attempts & old_attempts)
    assert {parse_attempt_id(a)["generation"] for a in new_attempts} == {2}
    # the decision is recorded, the stage's generation persisted, the status names the supersede
    decisions = [json.loads(line) for line in paths.run_decisions.read_text().splitlines()]
    supersedes = [d for d in decisions if d.get("kind") == "dynorch_supersede"]
    assert len(supersedes) == 1 and supersedes[0]["reason"] == reason and supersedes[0]["new_generation"] == 2
    assert sup.execution_generation(run_dir, "canary") == 2 and sup.execution_generation(run_dir, "pilot") == 1
    status = json.loads(paths.status_json.read_text())
    assert status["superseded"]["new_generation"] == 2 and status["reconciliation"]["decision"] == "clean"
    assert status["tables"]["n_records"] == 2          # the superseded execution is no row of the tables


def _record(tmp_path: Path, *, status: str, generation: int = 1, stage: str = "canary"):
    spec = dataclasses.replace(make_spec(rung=4, partition=stage), generation=generation)
    record = new_run_record(spec, status=status, started_at="2026-09-24T00:00:00Z")
    record.status = status
    RunStore(tmp_path).save(record)
    return spec


def _ledger(paths: RunPaths, events):
    from agent_allocation.contracts import LedgerEvent, LedgerEventKind
    from agent_allocation.storage.journal import LedgerJournal
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    journal = LedgerJournal(paths.ledger_journal)
    seq = sum(1 for _ in journal.replay())
    for kind, aid, amount, partition in events:
        seq += 1
        journal.append_sync(LedgerEvent(seq=seq, kind=LedgerEventKind(kind), attempt_id=aid, node_id=aid.split(":")[0],
                                        partition_id=partition, amount_usd=Decimal(amount),
                                        recorded_at="2026-09-24T00:00:00Z"))
    journal.release_writer()


def test_supersede_refuses_what_it_cannot_do_safely(tmp_path):
    paths = RunPaths.for_run(tmp_path, local_state_dir=tmp_path / "local")
    with pytest.raises(sup.SupersedeRefused, match="no records"):
        sup.supersede_stage(paths, "canary", reason="r")
    spec = _record(tmp_path, status="running")
    with pytest.raises(sup.SupersedeRefused, match="non-empty reason"):
        sup.supersede_stage(paths, "canary", reason="  ")
    with pytest.raises(sup.SupersedeRefused, match="still running"):
        sup.supersede_stage(paths, "canary", reason="r")
    _record(tmp_path, status="done")
    _ledger(paths, [("reserve", f"{spec.run_key}:g1:1:think:1", "0.1", "canary"),
                    ("sent", f"{spec.run_key}:g1:1:think:1", "0", "canary")])
    with pytest.raises(sup.SupersedeRefused, match="open ledger attempt"):
        sup.supersede_stage(paths, "canary", reason="r")
    assert (tmp_path / "runs" / "canary").exists()                    # nothing was moved by a refusal
    assert not (tmp_path / sup.SUPERSEDED_DIR).exists()


def test_the_new_generation_is_above_every_generation_the_stage_used(tmp_path):
    """Resumes bump the generation per attempt, and the ledger holds every id ever reserved: the re-execution's
    generation must be above all of them, so no attempt id can collide."""
    paths = RunPaths.for_run(tmp_path, local_state_dir=tmp_path / "local")
    spec = _record(tmp_path, status="done", generation=2)
    _ledger(paths, [("reserve", f"{spec.run_key}:g5:1:think:1", "0.1", "canary"),
                    ("sent", f"{spec.run_key}:g5:1:think:1", "0", "canary"),
                    ("settle", f"{spec.run_key}:g5:1:think:1", "0.05", "canary"),
                    ("reserve", "otherstage:g9:1:think:1", "0.1", "pilot"),
                    ("release_unsent", "otherstage:g9:1:think:1", "0", "pilot")])
    doc = sup.supersede_stage(paths, "canary", reason="defect")
    assert doc["new_generation"] == 6 and doc["superseded_generation"] == 1   # the pilot's g9 is not the canary's
    specs = sup.apply_generation([make_spec(partition="canary")], sup.execution_generation(tmp_path, "canary"))
    assert specs[0].generation == 6
    # superseding again (after the re-execution) climbs again
    _record(tmp_path, status="done", generation=6)
    assert sup.supersede_stage(paths, "canary", reason="again")["new_generation"] == 7
