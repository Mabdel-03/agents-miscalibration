"""Safety and recovery tests for protocol-response incident archival."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.completion import cell_lock
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest
from agents_scaling.experiment.qid_checkpoint import CHECKPOINT_DIRECTORY
from scripts import archive_protocol_incidents as incidents


def _cell(seed: int = 0) -> ExperimentCell:
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
            "seed": seed,
        }
    )


def _frozen_run(tmp_path: Path, *, run_id: str = "sweep"):
    results_root = tmp_path / "results"
    run_root = results_root / run_id
    run_root.mkdir(parents=True)
    cell = _cell()
    (run_root / "cells.json").write_text(
        json.dumps([cell.to_dict()]), encoding="utf-8"
    )
    freeze_manifest(run_root)
    return results_root, run_root, load_manifest(run_root), cell


def _task(run_root: Path, snapshot, cell: ExperimentCell) -> dict:
    return {
        "run_id": run_root.name,
        "run_root": str(run_root.resolve()),
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": "b" * 64,
        "source_index": 0,
        "model_size": cell.model_size,
        "serving_profile": cell.model_size,
        "fanout_cost": 1,
    }


def _dispatcher_state(
    tmp_path: Path,
    task: dict,
    *,
    job_id: str = "42",
    include_protocol_traceback: bool = True,
) -> tuple[Path, Path, Path]:
    state = tmp_path / "dispatcher"
    logs = state / "logs"
    batches = state / "batches"
    logs.mkdir(parents=True)
    batches.mkdir()
    batch = {
        "schema_version": 1,
        "batch_id": "batch-a",
        "created_at": 1.0,
        "tasks": [task],
    }
    batch_path = batches / "batch-batch-a.json"
    batch_path.write_text(json.dumps(batch, sort_keys=True) + "\n", encoding="utf-8")
    job = {
        "job_id": job_id,
        "batch_id": "batch-a",
        "batch_manifest": str(batch_path.resolve()),
        "sbatch_path": str((batches / "batch-batch-a.sbatch").resolve()),
        "submitted_at": 1.0,
        "task_count": 1,
        "tasks": [task],
    }
    ledger = {
        "schema_version": 1,
        "jobs": {job_id: job},
        "runs": {},
        "cells": {},
        "intents": {},
        "fairness": {},
    }
    (state / "ledger.json").write_text(
        json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = logs / f"dispatch_{job_id}_0.out"
    if include_protocol_traceback:
        log_text = (
            "Traceback (most recent call last):\n"
            "agents_scaling.serving.client.ServerResponseProtocolError: "
            "completion_sha256=deadbeef\n"
        )
    else:
        log_text = "[run_cell] complete\n"
    log_path.write_text(log_text, encoding="utf-8")
    return state, log_path, batch_path


def _failure(cell: ExperimentCell) -> dict:
    return {
        "schema_version": 2,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "classification": "runtime",
        "disposition": "retryable",
        "attempts": 1,
        "first_failed_at": 1.0,
        "last_failed_at": 1.0,
        "last_error": {
            "type": "ServerResponseProtocolError",
            "message": "completion_sha256=deadbeef",
        },
        "next_eligible_at": 2.0,
        "serving_profile": "0.6B",
        "code_version": "old-protocol-v4",
        "server_pool_generation": "pool-a",
        "dormant": False,
    }


def _write_active_artifacts(run_root: Path, cell: ExperimentCell) -> dict[str, bytes]:
    cell_dir = run_root / "cells" / cell.cell_id
    checkpoint_dir = cell_dir / CHECKPOINT_DIRECTORY
    checkpoint_dir.mkdir(parents=True)
    payloads = {
        "results.jsonl": b'{"qid":"q0","schema_version":4}\n',
        "meta.json": b'{"schema_version":4,"complete":false}\n',
        "failure.json": (json.dumps(_failure(cell), sort_keys=True) + "\n").encode(),
        f"{CHECKPOINT_DIRECTORY}/qid.json": b'{"schema_version":1,"partial":true}\n',
    }
    for relative, payload in payloads.items():
        path = cell_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return payloads


def _run(
    run_root: Path,
    state: Path,
    *,
    apply: bool = False,
    selected: set[str] | None = None,
):
    return incidents.archive_reset_run(
        run_root,
        dispatcher_state_dirs=[state],
        apply=apply,
        selected_cell_ids=selected,
    )


def test_dry_run_joins_failure_and_dispatcher_without_any_writes(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    stale = run_root / "cells" / "unmanifested-stale"
    stale.mkdir()
    (stale / "failure.json").write_text(
        json.dumps(_failure(cell)), encoding="utf-8"
    )

    report = _run(run_root, state)

    assert report["applied"] is False
    assert report["affected_manifest_cells"] == 1
    assert report["stale_unmanifested_dirs"] == 1
    assert report["outcomes"]["candidate"] == 1
    assert report["cells"][0]["failure_evidence"] is True
    assert report["cells"][0]["dispatcher_event_count"] == 1
    assert report["would_change_cells"] == [cell.cell_id]
    plan = report["cells"][0]["mutation_plan"]
    assert plan["would_change"] is True
    assert plan["complete_preimages_archived"] is True
    changes = {row["path"]: row for row in plan["active_artifact_changes"]}
    assert set(changes) == {
        f"cells/{cell.cell_id}/{relative}" for relative in payloads
    }
    for relative, payload in payloads.items():
        change = changes[f"cells/{cell.cell_id}/{relative}"]
        assert change["before_sha256"] == hashlib.sha256(payload).hexdigest()
        assert change["after_sha256"] is None
        assert change["archived_path"].endswith(f"artifacts/{relative}")
    publications = plan["archive_publications"]
    assert any(row["operation"] == "create_incident_index" for row in publications)
    assert any(row["operation"] == "create_reset_marker" for row in publications)
    assert all(row["before_sha256"] is None for row in publications)
    assert all(len(row["after_sha256"]) == 64 for row in publications)
    assert not (run_root / "incidents").exists()
    assert not (run_root / "cells" / cell.cell_id / ".cell.lock").exists()
    for relative, payload in payloads.items():
        assert (run_root / "cells" / cell.cell_id / relative).read_bytes() == payload


def test_apply_preserves_exact_artifacts_and_dispatcher_evidence_then_resets(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, log_path, batch_path = _dispatcher_state(
        tmp_path, _task(run_root, snapshot, cell)
    )
    log_bytes = log_path.read_bytes()
    batch_bytes = batch_path.read_bytes()
    dry_plan = _run(run_root, state)["cells"][0]["mutation_plan"]

    report = _run(run_root, state, apply=True)

    assert not report["errors"]
    assert report["outcomes"]["reset"] == 1
    assert report["cells"][0]["mutation_plan"] == dry_plan
    archive = Path(report["cells"][0]["archive"])
    incident_path = archive / incidents.INCIDENT_FILENAME
    marker_path = archive / incidents.RESET_MARKER_FILENAME
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert incident["manifest"]["sha256"] == snapshot.sha256
    assert incident["cell"]["config_hash"] == cell.config_hash()
    assert incident["target_artifact_schema_version"] >= 5
    assert marker["incident_sha256"] == hashlib.sha256(
        incident_path.read_bytes()
    ).hexdigest()

    by_source = {
        item["source_relative_path"]: item
        for item in incident["active_artifacts"]
    }
    assert set(by_source) == set(payloads)
    for relative, original in payloads.items():
        row = by_source[relative]
        archived = archive / row["archived_path"]
        assert archived.read_bytes() == original
        assert row["sha256"] == hashlib.sha256(original).hexdigest()
        assert not (run_root / "cells" / cell.cell_id / relative).exists()

    event = incident["discovery"]["dispatcher_events"][0]
    assert (archive / event["log"]["archived_path"]).read_bytes() == log_bytes
    assert (archive / event["batch_manifest"]["archived_path"]).read_bytes() == batch_bytes
    assert event["log"]["sha256"] == hashlib.sha256(log_bytes).hexdigest()
    assert event["batch_manifest"]["sha256"] == hashlib.sha256(batch_bytes).hexdigest()
    assert event["ledger"]["sha256"] == hashlib.sha256(
        (state / "ledger.json").read_bytes()
    ).hexdigest()
    assert (run_root / "cells" / cell.cell_id / ".cell.lock").exists()
    assert not (run_root / "cells" / cell.cell_id / CHECKPOINT_DIRECTORY).exists()
    assert stat_mode(incident_path) == 0o444
    assert stat_mode(archive) == 0o555


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_completed_incident_never_touches_new_schema5_rerun_artifacts(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    first = _run(run_root, state, apply=True)
    assert first["outcomes"]["reset"] == 1
    cell_dir = run_root / "cells" / cell.cell_id
    new_results = b'{"qid":"q0","schema_version":5,"new":true}\n'
    (cell_dir / "results.jsonl").write_bytes(new_results)

    second = _run(run_root, state, apply=True)

    assert second["outcomes"]["already_reset"] == 1
    assert (cell_dir / "results.jsonl").read_bytes() == new_results


def test_post_reset_protocol_failure_requires_new_incident_without_mutation(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    first = _run(run_root, state, apply=True)
    assert first["outcomes"]["reset"] == 1
    cell_dir = run_root / "cells" / cell.cell_id
    new_results = b'{"qid":"q0","schema_version":5,"new":true}\n'
    new_failure = _failure(cell)
    new_failure["classification"] = "configuration"
    new_failure["disposition"] = "permanent"
    new_failure["next_eligible_at"] = None
    new_failure["code_version"] = "schema5-new-incident"
    (cell_dir / "results.jsonl").write_bytes(new_results)
    (cell_dir / "failure.json").write_text(
        json.dumps(new_failure, sort_keys=True) + "\n", encoding="utf-8"
    )
    failure_bytes = (cell_dir / "failure.json").read_bytes()

    second = _run(run_root, state, apply=True)

    assert second["outcomes"]["new_incident_required"] == 1
    assert "open a new operator incident" in second["errors"][0]
    assert (cell_dir / "results.jsonl").read_bytes() == new_results
    assert (cell_dir / "failure.json").read_bytes() == failure_bytes


def test_post_reset_new_dispatcher_traceback_requires_new_incident(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    _write_active_artifacts(run_root, cell)
    task = _task(run_root, snapshot, cell)
    state, _, _ = _dispatcher_state(tmp_path, task)
    first = _run(run_root, state, apply=True)
    assert first["outcomes"]["reset"] == 1
    cell_dir = run_root / "cells" / cell.cell_id
    new_results = b'{"qid":"q0","schema_version":5,"new":true}\n'
    (cell_dir / "results.jsonl").write_bytes(new_results)

    second_batch = {
        "schema_version": 1,
        "batch_id": "batch-b",
        "created_at": 2.0,
        "tasks": [task],
    }
    second_batch_path = state / "batches" / "batch-batch-b.json"
    second_batch_path.write_text(
        json.dumps(second_batch, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger_path = state / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["jobs"]["43"] = {
        "job_id": "43",
        "batch_id": "batch-b",
        "batch_manifest": str(second_batch_path.resolve()),
        "sbatch_path": str((state / "batches" / "batch-b.sbatch").resolve()),
        "submitted_at": 2.0,
        "task_count": 1,
        "tasks": [task],
    }
    ledger_path.write_text(
        json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8"
    )
    (state / "logs" / "dispatch_43_0.out").write_text(
        "agents_scaling.serving.client.ServerResponseProtocolError: new envelope error\n",
        encoding="utf-8",
    )

    second = _run(run_root, state, apply=True)

    assert second["outcomes"]["new_incident_required"] == 1
    assert "dispatcher evidence" in second["errors"][0]
    assert (cell_dir / "results.jsonl").read_bytes() == new_results


def test_apply_skips_a_cell_while_runner_owns_nonblocking_lock(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    cell_dir = run_root / "cells" / cell.cell_id

    with cell_lock(cell_dir):
        report = _run(run_root, state, apply=True)

    assert report["outcomes"]["locked"] == 1
    assert not (run_root / "incidents").exists()
    for relative, payload in payloads.items():
        assert (cell_dir / relative).read_bytes() == payload


def test_dispatcher_log_recovers_cell_after_failure_was_cleared(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    cell_dir = run_root / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)
    results = b'{"qid":"q0","schema_version":4,"accepted_retry":true}\n'
    (cell_dir / "results.jsonl").write_bytes(results)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))

    report = _run(run_root, state)

    assert report["affected_manifest_cells"] == 1
    assert report["cells"][0]["failure_evidence"] is False
    assert report["cells"][0]["dispatcher_event_count"] == 1


def test_unmappable_protocol_traceback_fails_before_any_cell_mutation(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, log_path, _ = _dispatcher_state(
        tmp_path, _task(run_root, snapshot, cell)
    )
    ledger = json.loads((state / "ledger.json").read_text(encoding="utf-8"))
    ledger["jobs"] = {}
    (state / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(incidents.IncidentError, match="no dispatcher ledger job mapping"):
        _run(run_root, state, apply=True)

    assert log_path.exists()
    assert not (run_root / "incidents").exists()
    for relative, payload in payloads.items():
        assert (run_root / "cells" / cell.cell_id / relative).read_bytes() == payload


@pytest.mark.parametrize("drift_field", ["manifest_sha256", "config_hash", "run_root"])
def test_dispatcher_identity_drift_is_rejected(tmp_path, drift_field):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    task = _task(run_root, snapshot, cell)
    task[drift_field] = (
        str(tmp_path / "other-run") if drift_field == "run_root" else "drift"
    )
    state, _, _ = _dispatcher_state(tmp_path, task)

    with pytest.raises(incidents.IncidentError, match="drift"):
        _run(run_root, state)


def test_failure_drift_after_discovery_is_rejected_under_cell_lock(tmp_path, monkeypatch):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(
        tmp_path,
        _task(run_root, snapshot, cell),
        include_protocol_traceback=False,
    )
    real_lock = incidents.cell_lock

    @contextmanager
    def drift_then_lock(cell_dir):
        with real_lock(cell_dir):
            path = Path(cell_dir) / "failure.json"
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["last_error"]["message"] = "different sampled response"
            path.write_text(json.dumps(changed), encoding="utf-8")
            yield

    monkeypatch.setattr(incidents, "cell_lock", drift_then_lock)

    report = _run(run_root, state, apply=True)

    assert report["outcomes"]["error"] == 1
    assert "changed after discovery" in report["errors"][0]
    assert not (run_root / "incidents").exists()
    assert (run_root / "cells" / cell.cell_id / "results.jsonl").read_bytes() == (
        payloads["results.jsonl"]
    )


def test_interrupted_published_archive_resumes_without_original_failure(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    payloads = _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    first = _run(run_root, state, apply=True)
    archive = Path(first["cells"][0]["archive"])

    # Recreate an interrupted post-publication/pre-marker state.  The deterministic
    # archive is intact, one original artifact remains, and the failure source is gone.
    archive.chmod(0o755)
    marker = archive / incidents.RESET_MARKER_FILENAME
    marker.chmod(0o644)
    marker.unlink()
    cell_dir = run_root / "cells" / cell.cell_id
    (cell_dir / "results.jsonl").write_bytes(payloads["results.jsonl"])
    (state / "logs" / "dispatch_42_0.out").write_text(
        "[old logs moved elsewhere]\n", encoding="utf-8"
    )

    report = _run(run_root, state, apply=True)

    assert report["outcomes"]["reset"] == 1
    assert not (cell_dir / "results.jsonl").exists()
    assert (archive / incidents.RESET_MARKER_FILENAME).exists()


def test_selected_cell_filter_cannot_add_unmanifested_or_unproven_target(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    state, _, _ = _dispatcher_state(
        tmp_path,
        _task(run_root, snapshot, cell),
        include_protocol_traceback=False,
    )

    with pytest.raises(incidents.IncidentError, match="not in frozen manifest"):
        _run(run_root, state, selected={"../escape"})
    with pytest.raises(incidents.IncidentError, match="lack.*evidence"):
        _run(run_root, state, selected={cell.cell_id})


def test_symlinked_active_artifact_is_never_archived_or_removed(tmp_path):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    cell_dir = run_root / "cells" / cell.cell_id
    cell_dir.mkdir(parents=True)
    outside = tmp_path / "outside-results"
    outside.write_text("do not touch", encoding="utf-8")
    (cell_dir / "results.jsonl").symlink_to(outside)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))

    report = _run(run_root, state, apply=True)

    assert report["outcomes"]["error"] == 1
    assert "non-regular active cell artifact" in report["errors"][0]
    assert outside.read_text(encoding="utf-8") == "do not touch"
    assert (cell_dir / "results.jsonl").is_symlink()
    assert not (run_root / "incidents").exists()


def test_cli_is_dry_by_default_and_reports_manifest_checksum_failure(tmp_path, capsys):
    results_root, run_root, snapshot, cell = _frozen_run(tmp_path)
    _write_active_artifacts(run_root, cell)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))

    exit_code = incidents.main(
        [
            "--results-root",
            str(results_root),
            "--run-id",
            run_root.name,
            "--dispatcher-state-dir",
            str(state),
        ]
    )
    report = json.loads(capsys.readouterr().out)[0]
    assert exit_code == 0
    assert report["applied"] is False
    assert not (run_root / "incidents").exists()

    (run_root / "cells.json").write_text("[]\n", encoding="utf-8")
    exit_code = incidents.main(
        [
            "--results-root",
            str(results_root),
            "--run-id",
            run_root.name,
            "--dispatcher-state-dir",
            str(state),
        ]
    )
    report = json.loads(capsys.readouterr().out)[0]
    assert exit_code == 1
    assert "checksum" in report["errors"][0]


def test_reset_utility_refuses_executable_schema_beyond_frozen_target(
    tmp_path, monkeypatch
):
    _, run_root, snapshot, cell = _frozen_run(tmp_path)
    state, _, _ = _dispatcher_state(tmp_path, _task(run_root, snapshot, cell))
    monkeypatch.setattr(incidents, "ARTIFACT_SCHEMA_VERSION", 6)

    with pytest.raises(incidents.IncidentError, match="pinned.*schema 5"):
        _run(run_root, state)
