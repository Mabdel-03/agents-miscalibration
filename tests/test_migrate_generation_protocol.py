"""Focused safety tests for the generation-protocol incident migration."""

from __future__ import annotations

import hashlib
import json

import pytest

from agents_scaling.experiment.completion import cell_lock
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest
from agents_scaling.experiment.result_schema import ARTIFACT_SCHEMA_VERSION
from scripts import migrate_generation_protocol as migration


def _cell(seed: int = 0) -> dict:
    return {
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
        "n_questions": 3,
        "seed": seed,
    }


def _frozen_run(tmp_path, *, name: str = "pilot", cells: list[dict] | None = None):
    run_root = tmp_path / name
    run_root.mkdir()
    (run_root / "cells.json").write_text(
        json.dumps(cells or [_cell()]), encoding="utf-8"
    )
    freeze_manifest(run_root)
    return run_root, load_manifest(run_root)


def _truncation_failure(cell) -> dict:
    return {
        "schema_version": 1,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "classification": "configuration",
        "disposition": "permanent",
        "attempts": 1,
        "first_failed_at": 1.0,
        "last_failed_at": 2.0,
        "last_error": {
            "type": "GenerationTruncationError",
            "message": "finish_reason=length",
        },
    }


def _write_target_artifacts(cell_dir, cell):
    legacy = {"qid": "q1", "answer": "legacy"}
    old = {"schema_version": 3, "qid": "q0", "answer": "old"}
    current = {"schema_version": ARTIFACT_SCHEMA_VERSION, "qid": "q2", "answer": "current"}
    results_text = "".join(
        [
            json.dumps(old, separators=(",", ":")) + "\n",
            json.dumps(legacy, separators=(",", ":")) + "\n",
            "{malformed pilot tail\n",
            json.dumps(current, separators=(",", ":")) + "\n",
        ]
    )
    (cell_dir / "results.jsonl").write_text(results_text, encoding="utf-8")
    metadata = {
        "schema_version": 3,
        "cell_id": cell.cell_id,
        "pilot": True,
    }
    failure = _truncation_failure(cell)
    (cell_dir / "meta.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (cell_dir / "failure.json").write_text(json.dumps(failure) + "\n", encoding="utf-8")
    return results_text, old, legacy, current, metadata, failure


def test_dry_run_identifies_targets_without_writing_or_touching_stale_dirs(
    tmp_path, monkeypatch
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    original = _write_target_artifacts(cdir, cell)[0]
    stale = run_root / "cells" / "unmanifested-pilot"
    stale.mkdir()
    stale_failure = stale / "failure.json"
    stale_failure.write_text(json.dumps(_truncation_failure(cell)), encoding="utf-8")
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))

    report = migration.migrate_run(run_root)

    assert report["applied"] is False
    assert report["candidate_cells"] == 1
    assert report["old_explicit_result_rows"] == 1
    assert report["old_explicit_metadata"] == 1
    assert report["generation_truncation_failures"] == 1
    assert report["malformed_result_lines_retained"] == 1
    assert report["stale_unmanifested_dirs"] == 1
    assert not (run_root / migration.INCIDENT_FILENAME).exists()
    assert not (cdir / ".cell.lock").exists()
    assert (cdir / "results.jsonl").read_text(encoding="utf-8") == original
    assert stale_failure.exists()


def test_apply_preserves_evidence_quarantines_pilot_state_and_is_idempotent(
    tmp_path, monkeypatch
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    original, old, legacy, current, metadata, failure = _write_target_artifacts(cdir, cell)
    source_results_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    source_meta_sha = hashlib.sha256((json.dumps(metadata) + "\n").encode()).hexdigest()
    source_failure_sha = hashlib.sha256((json.dumps(failure) + "\n").encode()).hexdigest()
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))

    report = migration.migrate_run(run_root, apply=True)

    assert not report["errors"]
    assert report["results_rewritten"] == 1
    assert report["metadata_quarantined"] == 1
    assert report["failures_quarantined"] == 1
    retained = (cdir / "results.jsonl").read_text(encoding="utf-8")
    assert retained == "".join(
        [
            json.dumps(legacy, separators=(",", ":")) + "\n",
            "{malformed pilot tail\n",
            json.dumps(current, separators=(",", ":")) + "\n",
        ]
    )
    assert not (cdir / "meta.json").exists()
    assert not (cdir / "failure.json").exists()
    quarantined_meta = cdir / migration._quarantine_relative_path("meta", source_meta_sha)
    quarantined_failure = cdir / migration._quarantine_relative_path(
        "failure", source_failure_sha
    )
    assert json.loads(quarantined_meta.read_text(encoding="utf-8")) == metadata
    assert json.loads(quarantined_failure.read_text(encoding="utf-8")) == failure

    incident_path = run_root / migration.INCIDENT_FILENAME
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["manifest"]["sha256"] == snapshot.sha256
    entry = incident["cells"][cell.cell_id]
    assert entry["inferred_next_missing_qid"] == "q0"
    assert entry["all_expected_qids_present_after_migration"] is False
    assert entry["removed_result_rows"][0]["row"] == old
    assert entry["removed_result_rows"][0]["raw_line"] == (
        json.dumps(old, separators=(",", ":")) + "\n"
    )
    assert entry["removed_result_rows"][0]["source_results_sha256"] == source_results_sha
    assert entry["quarantined_metadata"][0]["payload"] == metadata
    assert entry["quarantined_failures"][0]["payload"] == failure
    assert report["incident_sha256"] == hashlib.sha256(incident_path.read_bytes()).hexdigest()

    artifact_sha = hashlib.sha256((cdir / "results.jsonl").read_bytes()).hexdigest()
    incident_sha = hashlib.sha256(incident_path.read_bytes()).hexdigest()
    second = migration.migrate_run(run_root, apply=True)
    assert second["candidate_cells"] == 0
    assert second["results_rewritten"] == 0
    assert second["metadata_quarantined"] == 0
    assert second["failures_quarantined"] == 0
    assert hashlib.sha256((cdir / "results.jsonl").read_bytes()).hexdigest() == artifact_sha
    assert hashlib.sha256(incident_path.read_bytes()).hexdigest() == incident_sha


def test_schema4_incident_is_validated_as_sealed_history_and_never_appended(
    tmp_path, monkeypatch
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    _write_target_artifacts(cdir, cell)
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))
    migration.migrate_run(run_root, apply=True)

    historical_path = run_root / migration.INCIDENT_FILENAME
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical["current_artifact_schema_version"] = 4
    historical_path.write_text(json.dumps(historical, indent=2) + "\n", encoding="utf-8")
    historical_sha = hashlib.sha256(historical_path.read_bytes()).hexdigest()

    report = migration.migrate_run(run_root)

    assert not report["errors"]
    assert report["candidate_cells"] == 0
    assert report["sealed_historical_artifact_schema_version"] == 4
    assert report["sealed_historical_incident_sha256"] == historical_sha
    assert report["incident_sha256"] == historical_sha
    assert not (run_root / migration.CURRENT_INCIDENT_FILENAME).exists()
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == historical_sha


def test_new_target_after_schema4_history_uses_new_generation_incident(
    tmp_path, monkeypatch
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    _write_target_artifacts(cdir, cell)
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))
    migration.migrate_run(run_root, apply=True)
    historical_path = run_root / migration.INCIDENT_FILENAME
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical["current_artifact_schema_version"] = 4
    historical_path.write_text(json.dumps(historical, indent=2) + "\n", encoding="utf-8")
    historical_sha = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    with (cdir / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema_version": 3, "qid": "q0", "answer": "new"}) + "\n")

    report = migration.migrate_run(run_root, apply=True)

    current_path = run_root / migration.CURRENT_INCIDENT_FILENAME
    assert not report["errors"]
    assert report["results_rewritten"] == 1
    assert current_path.is_file()
    assert json.loads(current_path.read_text())["current_artifact_schema_version"] == (
        ARTIFACT_SCHEMA_VERSION
    )
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == historical_sha


def test_apply_skips_locked_cell_without_creating_incident(tmp_path, monkeypatch):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    original = _write_target_artifacts(cdir, cell)[0]
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))

    with cell_lock(cdir):
        report = migration.migrate_run(run_root, apply=True)

    assert report["active_or_locked_cells_skipped"] == 1
    assert report["candidate_cells"] == 0
    assert not (run_root / migration.INCIDENT_FILENAME).exists()
    assert (cdir / "results.jsonl").read_text(encoding="utf-8") == original
    assert (cdir / "meta.json").exists()
    assert (cdir / "failure.json").exists()


def test_current_and_schema_less_artifacts_are_out_of_scope(tmp_path, monkeypatch):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    rows = [
        {"qid": "q0", "legacy": True},
        {"schema_version": ARTIFACT_SCHEMA_VERSION, "qid": "q1"},
    ]
    (cdir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    legacy_meta = {"cell_id": cell.cell_id, "legacy": True}
    (cdir / "meta.json").write_text(json.dumps(legacy_meta), encoding="utf-8")
    unrelated_failure = _truncation_failure(cell)
    unrelated_failure["last_error"]["type"] = "ServerResponseProtocolError"
    (cdir / "failure.json").write_text(json.dumps(unrelated_failure), encoding="utf-8")
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))

    report = migration.migrate_run(run_root, apply=True)

    assert report["candidate_cells"] == 0
    assert not (run_root / migration.INCIDENT_FILENAME).exists()
    assert json.loads((cdir / "meta.json").read_text(encoding="utf-8")) == legacy_meta
    assert (cdir / "failure.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"classification": "context_capacity"},
        {"disposition": "retryable"},
    ],
)
def test_truncation_failure_migration_preserves_current_or_ambiguous_ledgers(
    tmp_path, monkeypatch, change
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    failure = _truncation_failure(cell) | change
    (cdir / "failure.json").write_text(
        json.dumps(failure) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2")
    )

    report = migration.migrate_run(run_root, apply=True)

    assert not report["errors"]
    assert report["candidate_cells"] == 0
    assert report["generation_truncation_failures"] == 0
    assert json.loads((cdir / "failure.json").read_text(encoding="utf-8")) == failure
    assert not (run_root / migration.INCIDENT_FILENAME).exists()


@pytest.mark.parametrize("schema_version", [1, ARTIFACT_SCHEMA_VERSION + 1])
def test_unknown_or_future_explicit_schema_fails_closed(
    tmp_path, monkeypatch, schema_version
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    payload = {"schema_version": schema_version, "qid": "q0"}
    original = json.dumps(payload) + "\n"
    (cdir / "results.jsonl").write_text(original, encoding="utf-8")
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0",))

    report = migration.migrate_run(run_root, apply=True)

    assert report["candidate_cells"] == 0
    assert report["unsafe_or_unreadable_cells_skipped"] == 1
    assert any("unknown explicit result schema" in error for error in report["errors"])
    assert (cdir / "results.jsonl").read_text(encoding="utf-8") == original
    assert not (run_root / migration.INCIDENT_FILENAME).exists()


def test_both_superseded_explicit_pilot_schemas_are_preserved_and_removed(
    tmp_path, monkeypatch
):
    run_root, snapshot = _frozen_run(tmp_path)
    cell = snapshot.cells[0]
    cdir = run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    rows = [
        {"schema_version": 2, "qid": "q0", "pilot": "serving-provenance"},
        {"schema_version": 3, "qid": "q1", "pilot": "multi-phase-generation"},
        {"qid": "q2", "legacy": True},
    ]
    (cdir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(migration, "expected_qids_for_cell", lambda _: ("q0", "q1", "q2"))

    report = migration.migrate_run(run_root, apply=True)

    assert not report["errors"]
    assert report["old_explicit_result_rows"] == 2
    assert [
        json.loads(line)
        for line in (cdir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ] == [rows[2]]
    incident = json.loads(
        (run_root / migration.INCIDENT_FILENAME).read_text(encoding="utf-8")
    )
    assert incident["target_artifact_schema_versions"] == [2, 3]
    preserved = incident["cells"][cell.cell_id]["removed_result_rows"]
    assert sorted(item["row"]["schema_version"] for item in preserved) == [2, 3]


def test_requires_checksum_frozen_manifest(tmp_path):
    run_root = tmp_path / "not-frozen"
    run_root.mkdir()
    (run_root / "cells.json").write_text(json.dumps([_cell()]), encoding="utf-8")

    with pytest.raises(migration.MigrationError, match="checksum"):
        migration.migrate_run(run_root)


def test_cli_reports_multiple_runs_and_returns_nonzero_if_any_run_fails(
    tmp_path, capsys
):
    _frozen_run(tmp_path, name="pilot-a")
    _frozen_run(tmp_path, name="pilot-b", cells=[_cell(seed=1)])

    exit_code = migration.main(
        [
            "--results-root",
            str(tmp_path),
            "--run-id",
            "pilot-a",
            "--run-id",
            "pilot-b",
        ]
    )
    reports = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [report["run_root"] for report in reports] == [
        str((tmp_path / "pilot-a").resolve()),
        str((tmp_path / "pilot-b").resolve()),
    ]
    assert all(report["applied"] is False for report in reports)

    exit_code = migration.main(
        [
            "--results-root",
            str(tmp_path),
            "--run-id",
            "pilot-a",
            "--run-id",
            "missing-run",
        ]
    )
    reports = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert not reports[0]["errors"]
    assert "checksum" in reports[1]["errors"][0]
