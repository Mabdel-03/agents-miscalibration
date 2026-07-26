from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_schema5_r1_acceptance_receipts.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_schema5_r1_acceptance_receipts", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
receipts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipts)


def _write_json(path: Path, payload: object, *, read_only: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if read_only:
        path.chmod(0o444)
    return path


def _write_bytes(path: Path, payload: bytes, *, read_only: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if read_only:
        path.chmod(0o444)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence_fixture(tmp_path: Path) -> tuple[Path, Path]:
    recovery_root = tmp_path / "recovery"
    checkout = tmp_path / "r1"
    renderer = _write_bytes(
        checkout / "scripts" / "render_schema5_recovery_chain.py",
        b"#!/usr/bin/env python3\n",
        read_only=False,
    )
    stages = [
        "source_checkout",
        "release_materialize",
        "legacy_consolidate",
        "schema5_initialize",
        "production_resume",
    ]
    _write_json(
        recovery_root / receipts.R1_MANIFEST,
        {
            "first_legacy_mutation": "legacy_consolidate",
            "jobs": [{"name": stage} for stage in stages],
        },
    )
    job_rows = [
        {"name": stage, "job_id": str(1000 + index)}
        for index, stage in enumerate(stages)
    ]
    _write_json(recovery_root / receipts.R1_RECEIPT, {"jobs": job_rows})
    scheduler_rows = []
    for row in job_rows:
        stage = row["name"]
        scheduler_rows.append(
            {
                "job_id": row["job_id"],
                "job_name": stage,
                "state": (
                    "FAILED"
                    if stage == "release_materialize"
                    else "CANCELLED"
                ),
                "start": (
                    "2026-07-20T00:00:00"
                    if stage == "release_materialize"
                    else None
                ),
                "elapsed": (
                    "00:00:01"
                    if stage == "release_materialize"
                    else "00:00:00"
                ),
                "exit_code": "1:0" if stage == "release_materialize" else "0:15",
            }
        )
    _write_json(
        recovery_root / receipts.R1_FAILURE,
        {
            "classification": "requires_superseding_release",
            "failed_stage": "release_materialize",
            "scheduler_jobs": scheduler_rows,
        },
    )

    quarantine = recovery_root / "materialization_quarantines"
    inventory = _write_bytes(
        quarantine / "partial-job-18555913.sealed-inventory.txt",
        b"0" * 64 + b"  quarantined/file\n",
    )
    completion = _write_json(
        quarantine / "partial-job-18555913.complete.json",
        {"passed": True, "failed_job_id": "18555913"},
    )
    _write_json(
        quarantine / "partial-job-18555913.sealed.json",
        {
            "passed": True,
            "failed_job_id": "18555913",
            "inventory_sha256": _sha256(inventory),
            "quarantine_completion_sha256": _sha256(completion),
            "seal_id": "sealed-r1",
            "content_sha256": "1" * 64,
            "tree": {"files": 1},
        },
    )

    pre_repair = recovery_root / "pre_repair"
    _write_bytes(
        pre_repair / "SOURCE_INVENTORY.sha256",
        b"2" * 64 + b"  source/file\n",
    )
    _write_json(
        pre_repair / "SNAPSHOT_COMPLETE.json",
        {"passed": True, "inventory_sha256": "3" * 64},
    )
    _write_json(
        recovery_root / "pre_repair.attestation.json",
        {"passed": True, "snapshot": str(pre_repair)},
    )
    assert renderer.is_file()
    return recovery_root, checkout


def test_build_receipts_proves_two_repeats_and_zero_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery_root, checkout = _evidence_fixture(tmp_path)
    renderer = checkout / "scripts" / "render_schema5_recovery_chain.py"
    monkeypatch.setattr(
        receipts,
        "_renderer_identity",
        lambda path: (
            renderer,
            {
                "path": str(renderer),
                "sha256": _sha256(renderer),
                "size": renderer.stat().st_size,
                "checkout": str(path),
                "git_commit": receipts.R1_COMMIT,
                "git_tree": "4" * 40,
            },
        ),
    )
    calls: list[tuple[str, ...]] = []

    def run_json(argv: list[str], *, description: str) -> dict[str, object]:
        calls.append(tuple(argv))
        if argv[-1] == str(recovery_root / receipts.R1_MANIFEST):
            return {"passed": True, "protocol": "r1-native-verification"}
        return {
            "passed": True,
            "status": "already_quarantined",
            "seal_id": "sealed-r1",
        }

    monkeypatch.setattr(receipts, "_run_json", run_json)
    idempotency, zero_mutation = receipts.build_receipts(
        recovery_root, checkout
    )

    assert len(calls) == 3
    assert idempotency["passed"] is True
    assert idempotency["first_repeat"] == idempotency["second_repeat"]
    assert idempotency["first_repeat"]["status"] == "already_quarantined"
    assert zero_mutation["passed"] is True
    assert zero_mutation["legacy_result_mutation_count"] == 0
    assert zero_mutation["schema5_result_mutation_count"] == 0
    assert zero_mutation["started_result_mutating_stages"] == []
    assert [
        row["stage"] for row in zero_mutation["result_mutating_stage_states"]
    ] == ["legacy_consolidate", "schema5_initialize", "production_resume"]


def test_build_receipts_rejects_started_result_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery_root, checkout = _evidence_fixture(tmp_path)
    failure_path = recovery_root / receipts.R1_FAILURE
    failure_path.chmod(0o644)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    mutation = next(
        row
        for row in failure["scheduler_jobs"]
        if row["job_name"] == "legacy_consolidate"
    )
    mutation["start"] = "2026-07-20T00:01:00"
    mutation["elapsed"] = "00:00:01"
    _write_json(failure_path, failure)
    renderer = checkout / "scripts" / "render_schema5_recovery_chain.py"
    monkeypatch.setattr(
        receipts,
        "_renderer_identity",
        lambda _path: (
            renderer,
            {
                "path": str(renderer),
                "sha256": _sha256(renderer),
                "size": renderer.stat().st_size,
                "checkout": str(checkout),
                "git_commit": receipts.R1_COMMIT,
                "git_tree": "4" * 40,
            },
        ),
    )
    monkeypatch.setattr(
        receipts,
        "_run_json",
        lambda argv, *, description: (
            {"passed": True}
            if "verify" in argv
            else {"passed": True, "status": "already_quarantined"}
        ),
    )

    with pytest.raises(receipts.ReceiptError, match="stages started"):
        receipts.build_receipts(recovery_root, checkout)


def test_atomic_once_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    target = tmp_path / "receipts" / "receipt.json"
    payload = b'{"passed":true}\n'
    receipts._atomic_once(target, payload)
    first = target.stat()
    receipts._atomic_once(target, payload)

    assert target.read_bytes() == payload
    assert target.stat().st_ino == first.st_ino
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(receipts.ReceiptError, match="conflicting"):
        receipts._atomic_once(target, b'{"passed":false}\n')


def test_atomic_once_rejects_symlinked_output_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(actual, target_is_directory=True)

    with pytest.raises(receipts.ReceiptError, match="unsafe.*directory"):
        receipts._atomic_once(redirected / "receipt.json", b"{}\n")
    assert not (actual / "receipt.json").exists()


def test_renderer_identity_requires_exact_clean_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Schema Test"],
        cwd=checkout,
        check=True,
    )
    renderer = _write_bytes(
        checkout / "scripts" / "render_schema5_recovery_chain.py",
        b"#!/usr/bin/env python3\n",
        read_only=False,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "r1"], cwd=checkout, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    monkeypatch.setattr(receipts, "R1_COMMIT", commit)

    resolved, identity = receipts._renderer_identity(checkout)
    assert resolved == renderer
    assert identity["git_commit"] == commit

    renderer.write_text("# dirty\n", encoding="utf-8")
    with pytest.raises(receipts.ReceiptError, match="dirty"):
        receipts._renderer_identity(checkout)


def test_publish_is_marker_last_resumable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    checkout = tmp_path / "checkout"
    first = {"passed": True, "receipt_id": "a" * 64}
    second = {"passed": True, "receipt_id": "b" * 64}
    monkeypatch.setattr(
        receipts, "build_receipts", lambda *_args: (first, second)
    )
    original = receipts._atomic_once
    calls = 0

    def interrupt_after_first(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        original(path, payload)
        if calls == 1:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(receipts, "_atomic_once", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated crash"):
        receipts.publish(recovery_root, checkout, apply=True)

    output = recovery_root / receipts.OUTPUT_DIRECTORY
    assert (output / receipts.IDEMPOTENCY_RECEIPT).is_file()
    assert not (output / receipts.ZERO_MUTATION_RECEIPT).exists()

    monkeypatch.setattr(receipts, "_atomic_once", original)
    report = receipts.publish(recovery_root, checkout, apply=True)
    repeated = receipts.publish(recovery_root, checkout, apply=True)
    assert report == repeated
    assert (output / receipts.ZERO_MUTATION_RECEIPT).is_file()
    assert not any(output.glob("*.publishing"))
