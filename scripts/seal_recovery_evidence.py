#!/usr/bin/env python3
"""Seal failed recovery artifacts without deleting or rewriting their contents.

The two operations in this module are intentionally narrow:

``seal-quarantine``
    Inventories a same-filesystem quarantined partial release, removes every write
    bit, verifies the content again, and publishes a marker outside the tree last.

``record-failure``
    Binds scheduler states and the already-sealed recovery artifacts into an
    immutable failure envelope.  It never infers that a failed deterministic stage
    is retryable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


class EvidenceError(RuntimeError):
    """Raised when forensic evidence cannot be sealed unambiguously."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    _atomic_bytes(path, _canonical_json(payload), mode=mode)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{description} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be a JSON object: {path}")
    return value


def _absolute(path: str | Path, *, description: str) -> Path:
    lexical = Path(path)
    if not lexical.is_absolute() or str(lexical) != os.path.normpath(str(lexical)):
        raise EvidenceError(f"{description} must be a canonical absolute path: {path}")
    return lexical


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError(f"unsafe evidence lock: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidenceError(f"another evidence sealer holds {path}") from exc
        yield
    finally:
        os.close(descriptor)


def _inventory_rows(root: Path, *, include_modes: bool) -> tuple[list[str], int, int]:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError(f"quarantine root is missing or unsafe: {root}")
    rows: list[str] = []
    file_count = 0
    total_bytes = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        metadata = os.lstat(candidate)
        mode = f"{stat.S_IMODE(metadata.st_mode):04o}|" if include_modes else ""
        if stat.S_ISREG(metadata.st_mode):
            digest = _sha256_file(candidate)
            rows.append(f"f|{mode}{metadata.st_size}|{digest}|{relative}")
            file_count += 1
            total_bytes += metadata.st_size
        elif stat.S_ISDIR(metadata.st_mode):
            rows.append(f"d|{mode}0|-|{relative}")
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(candidate)
            if "\n" in target or "\r" in target:
                raise EvidenceError(f"newline in symlink target: {relative}")
            rows.append(f"l|{mode}{len(target.encode('utf-8'))}|{_sha256_bytes(target.encode())}|{relative}|{target}")
        else:
            raise EvidenceError(f"unsupported quarantine entry type: {relative}")
    return rows, file_count, total_bytes


def _inventory_bytes(rows: Iterable[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode("utf-8")


def _content_identity(root: Path) -> tuple[str, int, int]:
    rows, file_count, total_bytes = _inventory_rows(root, include_modes=False)
    return _sha256_bytes(_inventory_bytes(rows)), file_count, total_bytes


def _remove_write_bits(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True)
    for candidate in entries:
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            continue
        os.chmod(candidate, stat.S_IMODE(metadata.st_mode) & ~0o222)
    os.chmod(root, stat.S_IMODE(os.lstat(root).st_mode) & ~0o222)
    _fsync_directory(root.parent)


def _writable_entries(root: Path) -> list[str]:
    values: list[str] = []
    for candidate in (root, *root.rglob("*")):
        metadata = os.lstat(candidate)
        if not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) & 0o222:
            values.append("." if candidate == root else candidate.relative_to(root).as_posix())
    return sorted(values)


def seal_quarantine(
    *,
    tree: str | Path,
    evidence_root: str | Path,
    release_id: str,
    failed_job_id: str,
    quarantine_completion: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    tree_path = _absolute(tree, description="quarantine tree")
    evidence = _absolute(evidence_root, description="quarantine evidence root")
    completion_path = _absolute(
        quarantine_completion, description="quarantine completion"
    )
    if not failed_job_id.isdigit() or not release_id:
        raise EvidenceError("invalid failed job or release identity")
    quarantine = _read_json(completion_path, description="quarantine completion")
    if (
        quarantine.get("passed") is not True
        or str(quarantine.get("destination")) != str(tree_path)
        or str(quarantine.get("materialize_job_id")) != failed_job_id
        or quarantine.get("release_id") != release_id
    ):
        raise EvidenceError("quarantine completion does not identify the requested tree")

    stem = f"partial-job-{failed_job_id}"
    intent_path = evidence / f"{stem}.seal-intent.json"
    inventory_path = evidence / f"{stem}.sealed-inventory.txt"
    marker_path = evidence / f"{stem}.sealed.json"
    lock_path = evidence / f".{stem}.seal.lock"
    guard = _exclusive_lock(lock_path) if apply else nullcontext()
    with guard:
        if marker_path.exists():
            marker = _read_json(marker_path, description="quarantine seal marker")
            rows, file_count, total_bytes = _inventory_rows(
                tree_path, include_modes=True
            )
            payload = _inventory_bytes(rows)
            if (
                marker.get("passed") is not True
                or marker.get("inventory_sha256") != _sha256_bytes(payload)
                or marker.get("file_count") != file_count
                or marker.get("total_bytes") != total_bytes
                or inventory_path.read_bytes() != payload
                or _writable_entries(tree_path)
            ):
                raise EvidenceError("sealed quarantine verification failed")
            return marker | {"status": "already_sealed"}

        content_sha, file_count, total_bytes = _content_identity(tree_path)
        tree_stat = os.lstat(tree_path)
        expected_intent = {
            "schema_version": 1,
            "protocol": "schema5-quarantined-materialization-seal-intent-v1",
            "release_id": release_id,
            "failed_job_id": failed_job_id,
            "tree": str(tree_path),
            "tree_device": tree_stat.st_dev,
            "tree_inode": tree_stat.st_ino,
            "content_sha256": content_sha,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "quarantine_completion": str(completion_path),
            "quarantine_completion_sha256": _sha256_file(completion_path),
        }
        if intent_path.exists():
            intent = _read_json(intent_path, description="quarantine seal intent")
            identity = dict(intent)
            intent_id = identity.pop("intent_id", None)
            created_at = identity.pop("created_at", None)
            if (
                identity != expected_intent
                or not isinstance(created_at, str)
                or intent_id
                != _sha256_bytes(
                    _canonical_json(expected_intent | {"created_at": created_at})
                )
            ):
                raise EvidenceError("quarantine seal intent drifted")
        else:
            intent = expected_intent | {"created_at": _utc_now()}
            intent["intent_id"] = _sha256_bytes(_canonical_json(intent))
            if apply:
                _atomic_json(intent_path, intent, mode=0o444)

        report = {
            "status": "dry_run" if not apply else "sealing",
            "tree": str(tree_path),
            "content_sha256": content_sha,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "intent": str(intent_path),
            "inventory": str(inventory_path),
            "marker": str(marker_path),
            "writable_entries": len(_writable_entries(tree_path)),
        }
        if not apply:
            return report

        _remove_write_bits(tree_path)
        final_content_sha, final_file_count, final_total_bytes = _content_identity(
            tree_path
        )
        if (
            final_content_sha != content_sha
            or final_file_count != file_count
            or final_total_bytes != total_bytes
            or _writable_entries(tree_path)
        ):
            raise EvidenceError("quarantine content changed while sealing")
        rows, file_count, total_bytes = _inventory_rows(tree_path, include_modes=True)
        inventory = _inventory_bytes(rows)
        _atomic_bytes(inventory_path, inventory, mode=0o444)
        marker = {
            "schema_version": 1,
            "protocol": "schema5-quarantined-materialization-seal-v1",
            "passed": True,
            "release_id": release_id,
            "failed_job_id": failed_job_id,
            "tree": str(tree_path),
            "tree_device": tree_stat.st_dev,
            "tree_inode": tree_stat.st_ino,
            "content_sha256": content_sha,
            "inventory": str(inventory_path),
            "inventory_sha256": _sha256_bytes(inventory),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "intent": str(intent_path),
            "intent_sha256": _sha256_file(intent_path),
            "quarantine_completion": str(completion_path),
            "quarantine_completion_sha256": _sha256_file(completion_path),
            "sealed_at": _utc_now(),
        }
        marker["seal_id"] = _sha256_bytes(_canonical_json(marker))
        _atomic_json(marker_path, marker, mode=0o444)
        return marker | {"status": "sealed"}


def _scheduler_states(job_ids: Sequence[str]) -> list[dict[str, str]]:
    if not job_ids or any(not value.isdigit() for value in job_ids):
        raise EvidenceError("scheduler job IDs must be numeric")
    proc = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,JobName,State,Elapsed,Start,End,ExitCode,NodeList",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        raise EvidenceError(f"sacct failed: {proc.stderr.strip()[:500]}")
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 8:
            raise EvidenceError(f"unexpected sacct row: {line!r}")
        row = dict(
            zip(
                (
                    "job_id",
                    "job_name",
                    "state",
                    "elapsed",
                    "start",
                    "end",
                    "exit_code",
                    "node_list",
                ),
                fields,
            )
        )
        row["state"] = row["state"].split("+", 1)[0].upper()
        rows.append(row)
    by_id = {row["job_id"]: row for row in rows}
    if set(by_id) != set(job_ids):
        raise EvidenceError(
            f"sacct did not return the exact job set: missing={sorted(set(job_ids)-set(by_id))}"
        )
    if any(row["state"] not in _TERMINAL_STATES for row in rows):
        raise EvidenceError("recovery failure cannot be recorded while a job is active")
    return [by_id[value] for value in job_ids]


def record_failure(
    *,
    output: str | Path,
    chain_manifest: str | Path,
    submission_receipt: str | Path,
    snapshot_marker: str | Path,
    snapshot_attestation: str | Path,
    materialization_log: str | Path,
    quarantine_seal: str | Path,
    incident: str | Path,
    job_ids: Sequence[str],
    apply: bool = False,
) -> dict[str, Any]:
    output_path = _absolute(output, description="failure envelope")
    inputs = {
        "chain_manifest": _absolute(chain_manifest, description="chain manifest"),
        "submission_receipt": _absolute(
            submission_receipt, description="submission receipt"
        ),
        "snapshot_marker": _absolute(snapshot_marker, description="snapshot marker"),
        "snapshot_attestation": _absolute(
            snapshot_attestation, description="snapshot attestation"
        ),
        "materialization_log": _absolute(
            materialization_log, description="materialization log"
        ),
        "quarantine_seal": _absolute(
            quarantine_seal, description="quarantine seal"
        ),
        "incident": _absolute(incident, description="operational incident"),
    }
    for name, path in inputs.items():
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"{name} is missing or unsafe: {path}")
    log_root = inputs["materialization_log"].parent
    if log_root.is_symlink() or not log_root.is_dir():
        raise EvidenceError(f"r1 scheduler log root is missing or unsafe: {log_root}")
    scheduler_logs: list[dict[str, Any]] = []
    for candidate in sorted(log_root.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError(f"unsafe entry in r1 scheduler logs: {candidate}")
        scheduler_logs.append(
            {
                "relative_path": candidate.relative_to(log_root).as_posix(),
                "sha256": _sha256_file(candidate),
                "size": candidate.stat().st_size,
            }
        )
    expected_logs = {
        "source_checkout_18555909.out",
        "maintenance_preflight_18555910.out",
        "pre_repair_snapshot_18555911.out",
        "pre_repair_snapshot_verify_18555912.out",
        "release_materialize_18555913.out",
    }
    observed_logs = {row["relative_path"] for row in scheduler_logs}
    if observed_logs != expected_logs:
        raise EvidenceError(
            "r1 scheduler log set drifted: "
            f"missing={sorted(expected_logs - observed_logs)}, "
            f"unexpected={sorted(observed_logs - expected_logs)}"
        )
    states = _scheduler_states(job_ids)
    if sum(row["state"] == "FAILED" for row in states) != 1:
        raise EvidenceError("expected exactly one failed r1 recovery job")
    if not any(
        row["job_name"] == "asys-s5v11r1-materialize"
        and row["state"] == "FAILED"
        for row in states
    ):
        raise EvidenceError("r1 materialization failure is absent")
    artifacts = {
        name: {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size}
        for name, path in inputs.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-recovery-chain-failure-v1",
        "status": "failed_closed",
        "classification": "requires_superseding_release",
        "retry_same_generation": False,
        "superseded_by": "sweep-recovery-schema5-v1.2",
        "failed_stage": "release_materialize",
        "scheduler_jobs": states,
        "scheduler_log_root": str(log_root),
        "scheduler_logs": scheduler_logs,
        "artifacts": artifacts,
        "recorded_at": _utc_now(),
    }
    payload["failure_id"] = _sha256_bytes(_canonical_json(payload))
    if output_path.exists():
        existing = _read_json(output_path, description="failure envelope")
        stable_existing = dict(existing)
        stable_payload = dict(payload)
        stable_existing.pop("recorded_at", None)
        stable_payload.pop("recorded_at", None)
        stable_existing.pop("failure_id", None)
        stable_payload.pop("failure_id", None)
        if stable_existing != stable_payload:
            raise EvidenceError("existing failure envelope drifted")
        return existing | {"status": "already_recorded"}
    report = payload | {"status": "dry_run", "output": str(output_path)}
    if not apply:
        return report
    _atomic_json(output_path, payload, mode=0o444)
    _atomic_bytes(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{_sha256_file(output_path)}  {output_path.name}\n".encode("utf-8"),
        mode=0o444,
    )
    return payload | {"status": "recorded", "output": str(output_path)}


def _metadata_version(prefix: Path, distribution: str) -> tuple[str, str]:
    site_packages = prefix / "lib" / "python3.11" / "site-packages"
    matches = sorted(site_packages.glob(f"{distribution}-*.dist-info/METADATA"))
    if len(matches) != 1:
        raise EvidenceError(
            f"{prefix} has {len(matches)} {distribution} METADATA records"
        )
    version = None
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
            break
    if not version:
        raise EvidenceError(f"missing Version in {matches[0]}")
    return version, str(matches[0])


def record_conda_reconciliation_incident(
    *,
    output: str | Path,
    harness_prefix: str | Path,
    serving_prefix: str | Path,
    recovered_harness_record: str | Path,
    failed_materialization_log: str | Path,
    observed_at: str,
    apply: bool = False,
) -> dict[str, Any]:
    output_path = _absolute(output, description="incident output")
    harness = _absolute(harness_prefix, description="harness prefix")
    serving = _absolute(serving_prefix, description="serving prefix")
    recovered = _absolute(
        recovered_harness_record, description="recovered harness conda record"
    )
    failed_log = _absolute(
        failed_materialization_log, description="failed materialization log"
    )
    record_name = "setuptools-82.0.1-pyh332efcf_0.json"
    live_harness_record = harness / "conda-meta" / record_name
    live_serving_record = serving / "conda-meta" / record_name
    if live_harness_record.exists() or live_harness_record.is_symlink():
        raise EvidenceError(
            "harness stale Setuptools conda record is unexpectedly present"
        )
    if any(path.is_symlink() or not path.is_file() for path in (recovered, failed_log)):
        raise EvidenceError("incident preimage/log input is missing or unsafe")
    if live_serving_record.is_symlink():
        raise EvidenceError("serving stale Setuptools record is symlinked")
    recovered_payload = _read_json(
        recovered, description="recovered harness Setuptools record"
    )
    serving_payload = (
        _read_json(live_serving_record, description="serving Setuptools record")
        if live_serving_record.is_file()
        else None
    )
    expected_identity = {
        "name": "setuptools",
        "version": "82.0.1",
        "build": "pyh332efcf_0",
    }
    checked_records = [("recovered harness", recovered_payload)]
    if serving_payload is not None:
        checked_records.append(("live serving", serving_payload))
    for description, payload in checked_records:
        if {key: payload.get(key) for key in expected_identity} != expected_identity:
            raise EvidenceError(f"{description} Setuptools record identity drifted")
    harness_version, harness_metadata = _metadata_version(harness, "setuptools")
    serving_version, serving_metadata = _metadata_version(serving, "setuptools")
    if harness_version != "81.0.0" or serving_version != "81.0.0":
        raise EvidenceError("live Setuptools runtime is not the observed pip 81.0.0")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-conda-pip-reconciliation-incident-v1",
        "classification": "source_metadata_reconciled_by_conda_pip_interop",
        "observed_at": observed_at,
        "recorded_at": _utc_now(),
        "intended_action": "read_only_environment_inventory",
        "mutation_intended": False,
        "live_sources_must_not_be_queried_by_conda": True,
        "harness_prefix": str(harness),
        "serving_prefix": str(serving),
        "harness_stale_conda_record_present": False,
        "serving_stale_conda_record_present": serving_payload is not None,
        "runtime_owner": {
            "distribution": "setuptools",
            "version": "81.0.0",
            "harness_metadata": harness_metadata,
            "harness_metadata_sha256": _sha256_file(Path(harness_metadata)),
            "serving_metadata": serving_metadata,
            "serving_metadata_sha256": _sha256_file(Path(serving_metadata)),
        },
        "superseded_conda_record": {
            **expected_identity,
            "artifact_sha256": "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1",
            "recovered_harness_path": str(recovered),
            "recovered_harness_sha256": _sha256_file(recovered),
            "serving_path": (
                str(live_serving_record) if serving_payload is not None else None
            ),
            "serving_sha256": (
                _sha256_file(live_serving_record)
                if serving_payload is not None
                else None
            ),
        },
        "failed_materialization_log": str(failed_log),
        "failed_materialization_log_sha256": _sha256_file(failed_log),
        "resolution": "capture_live_bytes_without_conda_then_normalize_only_immutable_seeds",
    }
    payload["incident_id"] = _sha256_bytes(_canonical_json(payload))
    if output_path.exists():
        existing = _read_json(output_path, description="Conda incident")
        stable_existing = dict(existing)
        stable_payload = dict(payload)
        for value in (stable_existing, stable_payload):
            value.pop("recorded_at", None)
            value.pop("incident_id", None)
        if stable_existing != stable_payload:
            raise EvidenceError("existing Conda incident drifted")
        return existing | {"status": "already_recorded"}
    if not apply:
        return payload | {"status": "dry_run", "output": str(output_path)}
    _atomic_json(output_path, payload, mode=0o444)
    _atomic_bytes(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{_sha256_file(output_path)}  {output_path.name}\n".encode(),
        mode=0o444,
    )
    return payload | {"status": "recorded", "output": str(output_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-quarantine")
    seal.add_argument("--tree", type=Path, required=True)
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--release-id", required=True)
    seal.add_argument("--failed-job-id", required=True)
    seal.add_argument("--quarantine-completion", type=Path, required=True)
    seal.add_argument("--apply", action="store_true")

    failure = subparsers.add_parser("record-failure")
    failure.add_argument("--output", type=Path, required=True)
    failure.add_argument("--chain-manifest", type=Path, required=True)
    failure.add_argument("--submission-receipt", type=Path, required=True)
    failure.add_argument("--snapshot-marker", type=Path, required=True)
    failure.add_argument("--snapshot-attestation", type=Path, required=True)
    failure.add_argument("--materialization-log", type=Path, required=True)
    failure.add_argument("--quarantine-seal", type=Path, required=True)
    failure.add_argument("--incident", type=Path, required=True)
    failure.add_argument("--job-id", action="append", required=True)
    failure.add_argument("--apply", action="store_true")

    incident = subparsers.add_parser("record-conda-incident")
    incident.add_argument("--output", type=Path, required=True)
    incident.add_argument("--harness-prefix", type=Path, required=True)
    incident.add_argument("--serving-prefix", type=Path, required=True)
    incident.add_argument("--recovered-harness-record", type=Path, required=True)
    incident.add_argument("--failed-materialization-log", type=Path, required=True)
    incident.add_argument("--observed-at", required=True)
    incident.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal-quarantine":
            result = seal_quarantine(
                tree=args.tree,
                evidence_root=args.evidence_root,
                release_id=args.release_id,
                failed_job_id=args.failed_job_id,
                quarantine_completion=args.quarantine_completion,
                apply=args.apply,
            )
        elif args.command == "record-failure":
            result = record_failure(
                output=args.output,
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                snapshot_marker=args.snapshot_marker,
                snapshot_attestation=args.snapshot_attestation,
                materialization_log=args.materialization_log,
                quarantine_seal=args.quarantine_seal,
                incident=args.incident,
                job_ids=args.job_id,
                apply=args.apply,
            )
        else:
            result = record_conda_reconciliation_incident(
                output=args.output,
                harness_prefix=args.harness_prefix,
                serving_prefix=args.serving_prefix,
                recovered_harness_record=args.recovered_harness_record,
                failed_materialization_log=args.failed_materialization_log,
                observed_at=args.observed_at,
                apply=args.apply,
            )
    except (EvidenceError, OSError) as exc:
        print(f"[schema5-evidence] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
