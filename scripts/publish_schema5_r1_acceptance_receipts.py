#!/usr/bin/env python3
"""Publish deterministic r1 idempotency and zero-result-mutation receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any


R1_COMMIT = "a5cd9305e8fd741ba59bc14d9d296e3ecb5c5f96"
R1_MANIFEST = "RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
R1_RECEIPT = "RECOVERY_CHAIN_SCHEMA5_V1_1_R1_SUBMISSION.json"
R1_FAILURE = "FAILED_RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
OUTPUT_DIRECTORY = "r1_acceptance"
IDEMPOTENCY_RECEIPT = "QUARANTINE_IDEMPOTENCY_RECEIPT.json"
ZERO_MUTATION_RECEIPT = "ZERO_RESULT_MUTATION_RECEIPT.json"


class ReceiptError(RuntimeError):
    """The frozen r1 evidence does not prove the claimed acceptance property."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReceiptError(f"not a regular evidence file: {path}")
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ReceiptError(f"evidence changed while hashing: {path}")
    return digest.hexdigest()


def _file_record(path: Path, *, require_read_only: bool = True) -> dict[str, Any]:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ReceiptError(f"missing, symlinked, or non-canonical evidence: {path}")
    if require_read_only and stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ReceiptError(f"evidence remains writable: {path}")
    return {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"cannot parse evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError(f"evidence is not one JSON object: {path}")
    return payload


def _run_json(argv: list[str], *, description: str) -> dict[str, Any]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PIP_", "CONDA_"))
        and key
        not in {
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        }
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "CONDARC": os.devnull,
            "CONDA_NO_PLUGINS": "true",
        }
    )
    completed = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=600,
    )
    if completed.returncode != 0:
        raise ReceiptError(
            f"{description} failed ({completed.returncode}): "
            f"{completed.stderr.strip()[:1000]}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"{description} returned invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ReceiptError(f"{description} did not return one JSON object")
    return report


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if completed.returncode != 0:
        raise ReceiptError(
            f"Git identity query failed: {arguments!r}: "
            f"{completed.stderr.strip()[:500]}"
        )
    return completed.stdout.strip()


def _renderer_identity(checkout: Path) -> tuple[Path, dict[str, Any]]:
    checkout = checkout.expanduser().absolute()
    if checkout.is_symlink() or not checkout.is_dir():
        raise ReceiptError(f"exact r1 checkout is unavailable: {checkout}")
    if _git(checkout, "rev-parse", "HEAD") != R1_COMMIT:
        raise ReceiptError("r1 evidence publisher is not using exact a5cd930")
    if _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReceiptError("exact r1 checkout is dirty")
    renderer = checkout / "scripts" / "render_schema5_recovery_chain.py"
    record = _file_record(renderer, require_read_only=False)
    record.update(
        {
            "checkout": str(checkout),
            "git_commit": R1_COMMIT,
            "git_tree": _git(checkout, "rev-parse", f"{R1_COMMIT}^{{tree}}"),
        }
    )
    return renderer, record


def _atomic_once(path: Path, payload: bytes) -> None:
    path = path.expanduser().absolute()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise ReceiptError(f"unsafe acceptance-receipt directory: {parent}")
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
            or path.read_bytes() != payload
        ):
            raise ReceiptError(f"conflicting acceptance receipt: {path}")
        os.chmod(path, 0o444)
        return
    descriptor, name = tempfile.mkstemp(
        dir=parent, prefix=f".{path.name}.", suffix=".publishing"
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_receipts(
    recovery_root: Path,
    r1_checkout: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery_root = recovery_root.expanduser().absolute()
    if recovery_root.is_symlink() or not recovery_root.is_dir():
        raise ReceiptError(f"recovery root is unavailable: {recovery_root}")
    renderer, renderer_record = _renderer_identity(r1_checkout)
    manifest_path = recovery_root / R1_MANIFEST
    receipt_path = recovery_root / R1_RECEIPT
    failure_path = recovery_root / R1_FAILURE
    # Resolve and hash every command input before allowing the historical
    # executable to observe it.  In particular, never execute the r1 renderer
    # against a symlinked manifest and only reject that manifest afterwards.
    immutable_inputs = {
        "manifest": _file_record(manifest_path),
        "submission": _file_record(receipt_path),
        "failure": _file_record(failure_path),
    }
    manifest = _json(manifest_path)
    submission = _json(receipt_path)
    failure = _json(failure_path)

    native_verify = _run_json(
        [
            sys.executable,
            "-I",
            str(renderer),
            "verify",
            "--chain-manifest",
            str(manifest_path),
        ],
        description="exact a5cd930 native chain verification",
    )
    # This publisher is an acceptance probe, not a quarantine initiator.  Require
    # the marker-last sealed transaction before invoking the exact r1 command twice;
    # both calls must therefore take its read-only ``already_quarantined`` path.
    quarantine_root = recovery_root / "materialization_quarantines"
    seal = quarantine_root / "partial-job-18555913.sealed.json"
    inventory = quarantine_root / "partial-job-18555913.sealed-inventory.txt"
    completion = quarantine_root / "partial-job-18555913.complete.json"
    quarantine_inputs = {
        "seal": _file_record(seal),
        "inventory": _file_record(inventory),
        "completion": _file_record(completion),
    }
    seal_payload = _json(seal)
    if (
        stat.S_IMODE(seal.stat().st_mode) & 0o222
        or stat.S_IMODE(inventory.stat().st_mode) & 0o222
        or stat.S_IMODE(completion.stat().st_mode) & 0o222
        or seal_payload.get("passed") is not True
        or seal_payload.get("inventory_sha256") != _sha256(inventory)
        or seal_payload.get("quarantine_completion_sha256")
        != _sha256(completion)
        or seal_payload.get("failed_job_id") != "18555913"
    ):
        raise ReceiptError(
            "sealed quarantine must exist before the idempotency acceptance probe"
        )
    quarantine_argv = [
        sys.executable,
        "-I",
        str(renderer),
        "quarantine-materialization",
        "--chain-manifest",
        str(manifest_path),
        "--apply",
    ]
    first_repeat = _run_json(
        quarantine_argv, description="first repeated r1 quarantine"
    )
    second_repeat = _run_json(
        quarantine_argv, description="second repeated r1 quarantine"
    )
    if (
        first_repeat != second_repeat
        or first_repeat.get("status") != "already_quarantined"
        or first_repeat.get("passed") is not True
    ):
        raise ReceiptError("exact r1 quarantine is not deterministic and idempotent")

    if (
        seal_payload.get("passed") is not True
        or seal_payload.get("inventory_sha256") != _sha256(inventory)
        or seal_payload.get("quarantine_completion_sha256")
        != _sha256(completion)
        or seal_payload.get("failed_job_id") != "18555913"
    ):
        raise ReceiptError("sealed quarantine evidence is internally inconsistent")
    if {
        name: _file_record(Path(record["path"]))
        for name, record in immutable_inputs.items()
    } != immutable_inputs:
        raise ReceiptError("r1 chain evidence changed during acceptance probing")
    if {
        name: _file_record(Path(record["path"]))
        for name, record in quarantine_inputs.items()
    } != quarantine_inputs:
        raise ReceiptError("sealed quarantine changed during idempotency probing")
    _, renderer_after = _renderer_identity(r1_checkout)
    if renderer_after != renderer_record:
        raise ReceiptError("exact r1 checkout changed during acceptance probing")
    idempotency: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-r1-quarantine-idempotency-receipt-v1",
        "passed": True,
        "r1_renderer": renderer_record,
        "r1_manifest": immutable_inputs["manifest"],
        "r1_submission_receipt": immutable_inputs["submission"],
        "native_chain_verification": native_verify,
        "first_repeat": first_repeat,
        "second_repeat": second_repeat,
        "sealed_quarantine": {
            "completion": quarantine_inputs["completion"],
            "seal": quarantine_inputs["seal"],
            "inventory": quarantine_inputs["inventory"],
            "seal_id": seal_payload.get("seal_id"),
            "content_sha256": seal_payload.get("content_sha256"),
            "tree": seal_payload.get("tree"),
        },
        "sealed_tree_mutated": False,
    }
    idempotency["receipt_id"] = hashlib.sha256(_canonical(idempotency)).hexdigest()

    jobs = manifest.get("jobs")
    scheduler = failure.get("scheduler_jobs")
    if (
        native_verify.get("passed") is not True
        or manifest.get("first_legacy_mutation") != "legacy_consolidate"
        or not isinstance(jobs, list)
        or not isinstance(scheduler, list)
        or failure.get("classification") != "requires_superseding_release"
        or failure.get("failed_stage") != "release_materialize"
    ):
        raise ReceiptError("r1 native mutation-boundary evidence is incomplete")
    job_names = [row.get("name") for row in jobs if isinstance(row, dict)]
    mutation_index = job_names.index("legacy_consolidate")
    mutating_names = job_names[mutation_index:]
    scheduler_by_id = {
        str(row.get("job_id")): row for row in scheduler if isinstance(row, dict)
    }
    receipt_jobs = submission.get("jobs")
    if not isinstance(receipt_jobs, list):
        raise ReceiptError("r1 submission receipt lacks job bindings")
    receipt_by_name = {
        row.get("name"): row for row in receipt_jobs if isinstance(row, dict)
    }
    started_mutating: list[str] = []
    mutating_states: list[dict[str, Any]] = []
    for name in mutating_names:
        job = receipt_by_name.get(name)
        if not isinstance(job, dict):
            raise ReceiptError(f"missing r1 receipt binding for {name}")
        scheduler_row = scheduler_by_id.get(str(job.get("job_id")))
        if not isinstance(scheduler_row, dict):
            raise ReceiptError(f"missing sealed scheduler evidence for {name}")
        state_record = {
            key: scheduler_row.get(key)
            for key in ("job_id", "job_name", "state", "start", "elapsed", "exit_code")
        }
        mutating_states.append({"stage": name, **state_record})
        if (
            scheduler_row.get("start") not in {None, "None"}
            or scheduler_row.get("elapsed") not in {None, "00:00:00"}
        ):
            started_mutating.append(name)
    if started_mutating:
        raise ReceiptError(
            f"result-mutating r1 stages started unexpectedly: {started_mutating}"
        )
    source_inventory = recovery_root / "pre_repair" / "SOURCE_INVENTORY.sha256"
    snapshot_marker = recovery_root / "pre_repair" / "SNAPSHOT_COMPLETE.json"
    snapshot_attestation = recovery_root / "pre_repair.attestation.json"
    zero_mutation: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1",
        "passed": True,
        "proof": (
            "native a5cd930 verification authenticates every spooled body and DAG; "
            "sealed scheduler history proves the first legacy mutation and every "
            "schema5 successor had zero runtime after release materialization failed"
        ),
        "r1_renderer": renderer_record,
        "r1_manifest": immutable_inputs["manifest"],
        "r1_submission_receipt": immutable_inputs["submission"],
        "r1_failure_envelope": immutable_inputs["failure"],
        "native_chain_verification": native_verify,
        "failed_stage": "release_materialize",
        "first_result_mutating_stage": "legacy_consolidate",
        "result_mutating_stage_states": mutating_states,
        "started_result_mutating_stages": [],
        "pre_repair_source_inventory": _file_record(source_inventory),
        "pre_repair_source_inventory_entries": sum(
            1 for _line in source_inventory.open("rb")
        ),
        "pre_repair_snapshot_marker": _file_record(snapshot_marker),
        "pre_repair_snapshot_attestation": _file_record(snapshot_attestation),
        "legacy_result_mutation_count": 0,
        "schema5_result_mutation_count": 0,
    }
    zero_mutation["receipt_id"] = hashlib.sha256(
        _canonical(zero_mutation)
    ).hexdigest()
    return idempotency, zero_mutation


def publish(recovery_root: Path, r1_checkout: Path, *, apply: bool) -> dict[str, Any]:
    idempotency, zero_mutation = build_receipts(recovery_root, r1_checkout)
    output = recovery_root.expanduser().absolute() / OUTPUT_DIRECTORY
    paths = {
        "quarantine_idempotency": output / IDEMPOTENCY_RECEIPT,
        "zero_result_mutation": output / ZERO_MUTATION_RECEIPT,
    }
    if apply:
        _atomic_once(paths["quarantine_idempotency"], _canonical(idempotency))
        _atomic_once(paths["zero_result_mutation"], _canonical(zero_mutation))
    return {
        "status": "published" if apply else "dry_run",
        "passed": True,
        "receipts": {
            name: {
                "path": str(path),
                "receipt_id": (
                    idempotency if name == "quarantine_idempotency" else zero_mutation
                )["receipt_id"],
            }
            for name, path in paths.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--r1-checkout", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = publish(args.recovery_root, args.r1_checkout, apply=args.apply)
    except (OSError, ReceiptError, subprocess.SubprocessError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
