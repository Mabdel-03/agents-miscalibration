"""Mutation-boundary tests for the semantic artifact repair audit."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.completion import CompletionState
from agents_scaling.experiment.manifest import load_manifest
from scripts import audit_repair_run as audit


def _cell() -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": "0.6B",
            "context_share_level": "artifact_only",
            "prompt_complexity_level": 0,
            "reasoning_level": "off",
            "topology": "single_agent",
            "benchmark": "gpqa",
            "n_agents": 1,
            "rounds": 1,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 1,
            "seed": 0,
        }
    )


def test_apply_counts_missing_cell_without_materializing_its_directory(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    cell = _cell()
    (run_root / "cells.json").write_text(
        json.dumps([cell.to_dict()]), encoding="utf-8"
    )
    snapshot = load_manifest(run_root)
    catalog = SimpleNamespace(
        snapshot=snapshot,
        frozen=SimpleNamespace(),
        sidecar_sha256="f" * 64,
        questions_for=lambda _cell: (SimpleNamespace(qid="q1"),),
    )
    monkeypatch.setattr(audit, "VerifiedQuestionCatalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(
        audit,
        "get_completion_status",
        lambda *_args, **_kwargs: SimpleNamespace(status=CompletionState.MISSING),
    )

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("a wholly missing cell has no repairable artifacts")

    monkeypatch.setattr(audit, "canonicalize_results", unexpected_mutation)
    monkeypatch.setattr(audit, "cell_lock", unexpected_mutation)

    report = audit.audit_run(run_root, apply=True)

    assert report["states_before"] == {"missing": 1}
    assert report["no_repairable_artifacts_skipped"] == 1
    assert not (run_root / "cells").exists()


def _repair_fixture(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    cell = _cell()
    (run_root / "cells.json").write_text(json.dumps([cell.to_dict()]), encoding="utf-8")
    snapshot = load_manifest(run_root)
    catalog = SimpleNamespace(
        snapshot=snapshot,
        frozen=SimpleNamespace(),
        sidecar_sha256="f" * 64,
        questions_for=lambda _cell: (SimpleNamespace(qid="q1"),),
    )
    monkeypatch.setattr(audit, "VerifiedQuestionCatalog", lambda *_args, **_kwargs: catalog)
    monkeypatch.setattr(
        audit,
        "get_completion_status",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=CompletionState.PARTIAL,
            errors=(),
        ),
    )
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    source = b"{malformed evidence\n"
    (cdir / "results.jsonl").write_bytes(source)
    record = {"qid": "q1", "answer": "A"}
    parsed = SimpleNamespace(
        records=(record,),
        needs_rewrite=True,
        raw_rows=1,
        malformed_lines=1,
        invalid_rows=0,
        duplicate_qids=(),
        unexpected_qids=(),
        out_of_order=False,
    )

    def canonicalize(*_args, write=False, **_kwargs):
        if write:
            (cdir / "results.jsonl").write_text(
                json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        return parsed

    monkeypatch.setattr(audit, "canonicalize_results", canonicalize)
    return run_root, cell, cdir, source


def test_dry_run_reports_exact_hash_plan_without_mutation(tmp_path, monkeypatch):
    run_root, cell, cdir, source = _repair_fixture(tmp_path, monkeypatch)

    report = audit.audit_run(run_root, apply=False)

    assert report["results_would_rewrite"] == 1
    assert report["metadata_would_quarantine"] == 0
    assert report["failure_would_quarantine"] == 0
    change = report["would_change_cells"][0]
    assert change["cell_id"] == cell.cell_id
    assert change["results"]["before_sha256"] == hashlib.sha256(source).hexdigest()
    assert change["results"]["after_sha256"] != change["results"]["before_sha256"]
    assert (cdir / "results.jsonl").read_bytes() == source
    assert not (run_root / audit.REPAIR_INCIDENT_FILENAME).exists()


def test_apply_archives_complete_preimage_before_rewrite(tmp_path, monkeypatch):
    run_root, cell, cdir, source = _repair_fixture(tmp_path, monkeypatch)
    source_sha = hashlib.sha256(source).hexdigest()

    report = audit.audit_run(run_root, apply=True)

    assert report["results_rewritten"] == 1
    archive = (
        run_root
        / audit.REPAIR_PREIMAGE_DIRECTORY
        / source_sha[:2]
        / f"{source_sha}.jsonl"
    )
    assert archive.read_bytes() == source
    assert archive.stat().st_mode & 0o222 == 0
    incident = json.loads((run_root / audit.REPAIR_INCIDENT_FILENAME).read_text())
    evidence = next(iter(incident["rewrites"].values()))
    assert evidence["cell_id"] == cell.cell_id
    assert evidence["source_results_sha256"] == source_sha
    assert hashlib.sha256((cdir / "results.jsonl").read_bytes()).hexdigest() == evidence[
        "canonical_results_sha256"
    ]
