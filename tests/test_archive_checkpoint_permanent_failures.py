"""Safety tests for exact obsolete checkpoint-failure archival."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.completion import cell_lock
from agents_scaling.experiment.manifest import freeze_manifest
from agents_scaling.experiment.qid_checkpoint import CHECKPOINT_DIRECTORY
from scripts import archive_checkpoint_permanent_failures as archive


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


def _run(tmp_path: Path, *, checkpoint_schema: int = 2):
    run = tmp_path / "test_run"
    run.mkdir()
    cell = _cell()
    (run / "cells.json").write_text(json.dumps([cell.to_dict()]), encoding="utf-8")
    freeze_manifest(run)
    cell_dir = run / "cells" / cell.cell_id
    checkpoints = cell_dir / CHECKPOINT_DIRECTORY
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / f"{'a' * 64}.json"
    checkpoint.write_text(
        json.dumps({"schema_version": checkpoint_schema, "qid": "q0"}) + "\n",
        encoding="utf-8",
    )
    failure = {
        "schema_version": 2,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "classification": "configuration",
        "disposition": "permanent",
        "attempts": 1,
        "last_error": {
            "type": "ExperimentConfigurationError",
            "message": (
                "durable QID checkpoint failed closed; refusing replacement sampling: "
                f"durable QID checkpoint /results/test_run/cells/{cell.cell_id}/"
                f"{CHECKPOINT_DIRECTORY}/{checkpoint.name} has the wrong root schema"
            ),
        },
    }
    failure_bytes = (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode()
    (cell_dir / "failure.json").write_bytes(failure_bytes)
    targets = {cell.cell_id: cell.config_hash()}
    return run, cell, cell_dir, targets, failure_bytes


def _archive(run: Path, targets: dict[str, str], *, apply: bool = False):
    return archive.archive_run(
        run,
        apply=apply,
        target_cell_hashes=targets,
        require_run_id=None,
    )


def test_dry_run_reports_exact_before_after_hashes_without_writes(tmp_path):
    run, cell, cell_dir, targets, failure_bytes = _run(tmp_path)

    report = _archive(run, targets)

    assert report["applied"] is False
    assert report["counts"] == {"would_reset": 1}
    assert report["would_change_cells"] == [cell.cell_id]
    row = report["cells"][0]
    assert row["cell_id"] == cell.cell_id
    assert row["before_sha256"] == hashlib.sha256(failure_bytes).hexdigest()
    assert row["after_sha256"] is None
    assert row["checkpoint_count"] == 1
    assert row["checkpoint_evidence"][0]["schema_version"] == 2
    assert len(row["incident_sha256"]) == 64
    assert (cell_dir / "failure.json").read_bytes() == failure_bytes
    assert not (run / "incidents").exists()
    assert not (cell_dir / ".cell.lock").exists()


def test_symlinked_run_root_is_rejected(tmp_path):
    run, _cell_value, _cell_dir, targets, _failure_bytes = _run(tmp_path)
    linked = tmp_path / "linked-run"
    linked.symlink_to(run, target_is_directory=True)

    with pytest.raises(archive.PermanentFailureArchiveError, match="is a symlink"):
        _archive(linked, targets)


def test_apply_archives_complete_preimage_removes_exact_failure_and_is_idempotent(tmp_path):
    run, cell, cell_dir, targets, failure_bytes = _run(tmp_path)

    report = _archive(run, targets, apply=True)

    assert report["counts"] == {"reset": 1}
    row = report["cells"][0]
    incident_dir = Path(row["archive"])
    assert not (cell_dir / "failure.json").exists()
    assert (incident_dir / archive.PREIMAGE_FILENAME).read_bytes() == failure_bytes
    incident_bytes = (incident_dir / archive.INCIDENT_FILENAME).read_bytes()
    incident = json.loads(incident_bytes)
    marker = json.loads((incident_dir / archive.RESET_MARKER_FILENAME).read_text())
    assert incident["mutation"] == {
        "before_sha256": hashlib.sha256(failure_bytes).hexdigest(),
        "after_sha256": None,
        "operation": "durable_remove_exact_archived_preimage",
    }
    assert marker["incident_sha256"] == hashlib.sha256(incident_bytes).hexdigest()
    assert row["incident_sha256"] == marker["incident_sha256"]
    assert row["checkpoint_evidence"] == incident["checkpoints_after_migration"]
    assert marker["active_failure_removed"] is True
    assert stat.S_IMODE(incident_dir.stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (incident_dir / archive.PREIMAGE_FILENAME).stat().st_mode
    ) == 0o444

    again = _archive(run, targets, apply=True)
    assert again["counts"] == {"already_reset": 1}
    assert (incident_dir / archive.PREIMAGE_FILENAME).read_bytes() == failure_bytes


def test_interrupted_after_removal_before_marker_resumes_without_source(tmp_path):
    run, _cell_value, cell_dir, targets, failure_bytes = _run(tmp_path)
    first = _archive(run, targets, apply=True)
    incident_dir = Path(first["cells"][0]["archive"])
    marker = incident_dir / archive.RESET_MARKER_FILENAME
    os.chmod(incident_dir, 0o755)
    marker.unlink()

    resumed = _archive(run, targets, apply=True)

    assert resumed["counts"] == {"reset": 1}
    assert not (cell_dir / "failure.json").exists()
    assert (incident_dir / archive.PREIMAGE_FILENAME).read_bytes() == failure_bytes
    assert marker.is_file()


def test_completed_incident_never_clears_a_new_failure(tmp_path):
    run, _cell_value, cell_dir, targets, failure_bytes = _run(tmp_path)
    _archive(run, targets, apply=True)
    new_failure = failure_bytes.replace(b'"attempts": 1', b'"attempts": 2')
    (cell_dir / "failure.json").write_bytes(new_failure)

    with pytest.raises(archive.PermanentFailureArchiveError, match="new failure"):
        _archive(run, targets, apply=True)

    assert (cell_dir / "failure.json").read_bytes() == new_failure


def test_completed_incident_rejects_coordinated_identity_tampering(tmp_path):
    run, _cell_value, _cell_dir, targets, _failure_bytes = _run(tmp_path)
    first = _archive(run, targets, apply=True)
    incident_dir = Path(first["cells"][0]["archive"])
    incident_path = incident_dir / archive.INCIDENT_FILENAME
    marker_path = incident_dir / archive.RESET_MARKER_FILENAME
    os.chmod(incident_dir, 0o755)
    os.chmod(incident_path, 0o644)
    os.chmod(marker_path, 0o644)
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    incident["run_id"] = "different-run"
    incident_bytes = (json.dumps(incident, indent=2, sort_keys=True) + "\n").encode()
    incident_path.write_bytes(incident_bytes)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["incident_sha256"] = hashlib.sha256(incident_bytes).hexdigest()
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    with pytest.raises(
        archive.PermanentFailureArchiveError,
        match="completed incident archive is inconsistent",
    ):
        _archive(run, targets)


def test_schema1_checkpoint_blocks_reset_without_archive(tmp_path):
    run, _cell_value, cell_dir, targets, failure_bytes = _run(
        tmp_path, checkpoint_schema=1
    )

    with pytest.raises(archive.PermanentFailureArchiveError, match="migration is incomplete"):
        _archive(run, targets, apply=True)

    assert (cell_dir / "failure.json").read_bytes() == failure_bytes
    assert not (run / "incidents").exists()


def test_dry_run_accepts_exact_projected_schema2_checkpoint_without_mutation(
    tmp_path: Path,
) -> None:
    run, cell, cell_dir, targets, failure_bytes = _run(
        tmp_path, checkpoint_schema=1
    )
    checkpoint = next((cell_dir / CHECKPOINT_DIRECTORY).iterdir())
    source = checkpoint.read_bytes()
    projection = {
        cell.cell_id: [
            {
                "cell_id": cell.cell_id,
                "checkpoint_relative_path": checkpoint.relative_to(run).as_posix(),
                "before_sha256": hashlib.sha256(source).hexdigest(),
                "after_sha256": "b" * 64,
                "before_size": len(source),
                "after_size": len(source) + 10,
                "after_schema_version": 2,
            }
        ]
    }

    report = archive.archive_run(
        run,
        target_cell_hashes=targets,
        require_run_id=None,
        projected_checkpoint_plans=projection,
    )

    row = report["cells"][0]
    assert row["status"] == "would_reset"
    assert row["checkpoint_projection_used"] is True
    assert row["before_sha256"] == hashlib.sha256(failure_bytes).hexdigest()
    assert row["after_sha256"] is None
    assert row["checkpoint_evidence"] == [
        {
            "relative_path": f"{CHECKPOINT_DIRECTORY}/{checkpoint.name}",
            "sha256": "b" * 64,
            "size": len(source) + 10,
            "schema_version": 2,
        }
    ]
    assert len(row["incident_sha256"]) == 64
    assert checkpoint.read_bytes() == source
    assert (cell_dir / "failure.json").read_bytes() == failure_bytes
    assert not (run / "incidents").exists()


def test_projected_checkpoint_must_bind_exact_source_hash(tmp_path: Path) -> None:
    run, cell, cell_dir, targets, _failure_bytes = _run(
        tmp_path, checkpoint_schema=1
    )
    checkpoint = next((cell_dir / CHECKPOINT_DIRECTORY).iterdir())
    source = checkpoint.read_bytes()
    projection = {
        cell.cell_id: [
            {
                "cell_id": cell.cell_id,
                "checkpoint_relative_path": checkpoint.relative_to(run).as_posix(),
                "before_sha256": "0" * 64,
                "after_sha256": "b" * 64,
                "before_size": len(source),
                "after_size": len(source),
                "after_schema_version": 2,
            }
        ]
    }

    with pytest.raises(archive.PermanentFailureArchiveError, match="hashes or schema"):
        archive.archive_run(
            run,
            target_cell_hashes=targets,
            require_run_id=None,
            projected_checkpoint_plans=projection,
        )

    assert checkpoint.read_bytes() == source
    assert not (run / "incidents").exists()


def test_busy_cell_is_not_mutated(tmp_path):
    run, _cell_value, cell_dir, targets, failure_bytes = _run(tmp_path)

    with cell_lock(cell_dir, blocking=False):
        with pytest.raises(archive.PermanentFailureArchiveError, match="cell is active"):
            _archive(run, targets, apply=True)

    assert (cell_dir / "failure.json").read_bytes() == failure_bytes
    assert not (run / "incidents").exists()
