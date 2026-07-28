#!/usr/bin/env python3
"""Build artifact-backed schema-5 readiness evidence, failing closed on drift.

The schema-5 controller intentionally accepts only tiny, typed gate envelopes.  This
tool is the corresponding evidence producer.  It never accepts operator-supplied
metrics: every value is derived from a sealed snapshot, the legacy consolidation
marker, an exact context audit, or live scheduler/registry/HTTP observations.

Each successful command writes its source/wrapper artifacts first and publishes the
schema-2 outer envelope last using a same-directory fsync + ``os.replace``.  The
candidate envelope is passed through the controller's own validator before it becomes
visible at ``--output``.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import (  # noqa: E402
    VerifiedQuestionCatalog,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.serving.fleet_contract import (  # noqa: E402
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.launch_server import _port_for  # noqa: E402
from agents_scaling.serving.model_contracts import (  # noqa: E402
    FrozenModelContracts,
    load_model_contracts,
)
from agents_scaling.serving.registry import (  # noqa: E402
    ServerEntry,
    entry_matches_frozen_provenance,
    server_pool_id,
)
from agents_scaling.serving import fleet_transactions as fleet_tx  # noqa: E402
from agents_scaling.serving import protected_capacity, scheduler_safety  # noqa: E402
from scripts import audit_context_capacity as context_audit  # noqa: E402
from scripts.audit_context_capacity import AuditFilters, selected_cells  # noqa: E402
from scripts import schema5_email_ack  # noqa: E402
from scripts.verify_schema5_recovery_evidence import (  # noqa: E402
    EvidenceVerificationError,
    R8_PROTOCOL,
    verify_recovery_evidence,
)
from slurm import keepalive  # noqa: E402
from slurm import schema5_control as control_plane  # noqa: E402


READINESS_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 1
EXPECTED_CONTEXT_MARGIN = 1_287
EXPECTED_DENSE_CELLS = 216
EXPECTED_DENSE_REQUESTS = 43_092
EXPECTED_SEVEN_CELLS = 288
EXPECTED_SEVEN_REQUESTS = 57_456
CAPACITY_TRANSIENT_PROTOCOL = (
    "schema5-v1.2-r8-fleet-capacity-transient-receipt"
)
CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r8-fleet-capacity-transient-evidence"
)
CAPACITY_TRANSIENT_EVIDENCE_NAME = "FLEET_CAPACITY_TRANSIENT_EVIDENCE.json"
CAPACITY_TRANSIENT_MARKER_NAME = "CAPACITY_TRANSIENT_COMPLETE.json"
CAPACITY_TRANSIENT_LOCK_NAME = ".fleet-capacity-transient.lock"
CAPACITY_TRANSIENT_REASONS = frozenset({"Priority", "Resources"})
CAPACITY_TRANSIENT_EXPECTED_REPLICAS = 22
CAPACITY_TRANSIENT_EXPECTED_GPUS = 24
CAPACITY_TRANSIENT_BOUNDARY_SECONDS = 36_000
CAPACITY_TRANSIENT_BOUNDARY_TOLERANCE_SECONDS = 300
CAPACITY_TRANSIENT_MAX_EVIDENCE_AGE_SECONDS = 600
CAPACITY_TRANSIENT_INCIDENT_DIRECTORY = "incidents"
CAPACITY_PREIMAGE_ROOT_NAME = "sealed-preimages"
CAPACITY_PREIMAGE_MANIFEST_NAME = "PREIMAGE_MANIFEST.json"
CAPACITY_PREIMAGE_INVENTORY_NAME = "PREIMAGE_INVENTORY.sha256"
CAPACITY_PREIMAGE_COMPLETE_NAME = "PREIMAGE_ARCHIVE_COMPLETE.json"
CAPACITY_PREIMAGE_PROTOCOL = "schema5-v1.2-r8-capacity-preimage-archive"


class EvidenceError(RuntimeError):
    """A readiness fact cannot be proved from its authoritative source."""


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_json(path: Path) -> dict[str, Any]:
    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise EvidenceError(f"expected a regular evidence file: {supplied}")
    path = supplied.resolve()
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"cannot parse evidence JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence must contain one JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def _artifact(name: str, path: Path) -> dict[str, str]:
    supplied = path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise EvidenceError(f"artifact {name!r} is missing or a symlink: {supplied}")
    path = supplied.resolve()
    return {"name": name, "path": str(path), "sha256": _sha256(path)}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one JSON artifact and durably publish its directory entry."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_read_only_regular(path: Path, *, description: str) -> Path:
    supplied = path.expanduser().absolute()
    try:
        metadata = supplied.lstat()
    except OSError as exc:
        raise EvidenceError(f"{description} is unavailable: {supplied}: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o222
    ):
        raise EvidenceError(
            f"{description} must be one read-only regular non-symlink file: {supplied}"
        )
    return supplied.resolve()


class _CapacityReceiptLock:
    """One non-blocking, cross-node publication fence for marker-last evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.descriptor: int | None = None

    def __enter__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise EvidenceError(
                f"capacity-transient receipt root is unsafe: {self.root}"
            )
        lock = self.root / CAPACITY_TRANSIENT_LOCK_NAME
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock, flags, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            try:
                os.close(descriptor)
            except UnboundLocalError:
                pass
            raise EvidenceError(
                "another capacity-transient receipt publication is active"
            ) from exc
        except OSError as exc:
            raise EvidenceError(
                f"cannot lock capacity-transient receipt root {self.root}: {exc}"
            ) from exc
        self.descriptor = descriptor

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.descriptor is None:
            return
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None


def _prepared_snapshot_context(
    control: Mapping[str, Any], *, state_dir: Path | None
) -> Any | None:
    if state_dir is None:
        return None
    snapshot_record = control.get(control_plane.SNAPSHOT_ATTESTATION_STATE_KEY)
    if not isinstance(snapshot_record, dict):
        return None
    control_plane.refresh_snapshot_integrity_lease(state_dir, control)
    seal = control_plane.validate_snapshot_integrity_attestation(
        control, state_dir=state_dir, verify_lease=True
    )
    allowed_members = [
        member
        for binding in snapshot_record.get("member_bindings", {}).values()
        if isinstance(binding, dict)
        for member in binding.get("members", [])
        if isinstance(member, dict)
    ]
    return control_plane._SnapshotValidationContext(
        full=False,
        seal=seal,
        allow_inventory_lookup=True,
        allowed_members=allowed_members,
    )


def _publish_outer(
    *,
    control: Mapping[str, Any],
    gate: str,
    metrics: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, str]],
    output: Path,
    now: float | None = None,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a temporary candidate, then make the gate envelope visible last."""

    payload: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "gate": gate,
        "passed": True,
        "immutable_sha256": control["immutable_sha256"],
        "metrics": dict(metrics),
        "artifacts": list(artifacts),
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_name = tempfile.mkstemp(
        prefix=f".{output.name}.candidate.", suffix=".json", dir=output.parent
    )
    candidate = Path(candidate_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        candidate.chmod(0o444)
        digest = _sha256(candidate)
        validation_control = control
        snapshot_context = None
        if state_dir is not None and isinstance(
            control.get(control_plane.SNAPSHOT_ATTESTATION_STATE_KEY), dict
        ):
            validation_control = control_plane.load_control(state_dir)
            snapshot_context = _prepared_snapshot_context(
                validation_control, state_dir=state_dir
            )
        if snapshot_context is None:
            snapshot_context = control_plane._SnapshotValidationContext(full=True)
        control_plane._validate_attestation(  # type: ignore[attr-defined]
            validation_control,
            gate,
            candidate,
            digest,
            now=time.time() if now is None else float(now),
            snapshot_context=snapshot_context,
        )
        os.replace(candidate, output)
        _fsync_directory(output.parent)
        output_digest = _sha256(output)
        if gate == "snapshot" and state_dir is not None and snapshot_context.full:
            control_plane.prepare_snapshot_integrity_preseal(
                state_dir,
                evidence_path=output,
                evidence_sha256=output_digest,
                validation_context=snapshot_context,
                now=time.time() if now is None else float(now),
            )
    finally:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    return payload | {"path": str(output), "sha256": _sha256(output)}


def _wrapper(
    *,
    name: str,
    immutable_sha256: str,
    metrics: Mapping[str, Any],
    references: Sequence[Mapping[str, str]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": name,
        "passed": True,
        "immutable_sha256": immutable_sha256,
        "metrics": dict(metrics),
        "referenced_artifacts": list(references),
    }
    if extra:
        payload.update(extra)
    return payload


def _artifact_path(output: Path, name: str) -> Path:
    return output.expanduser().resolve().parent / "artifacts" / f"{name}.json"


def _require_exact_metrics(
    observed: Any, expected: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if not isinstance(observed, dict) or observed != dict(expected):
        raise EvidenceError(
            f"{context} metrics drifted: expected {dict(expected)!r}, "
            f"observed {observed!r}"
        )
    return dict(observed)


def _verify_recursive_reference(
    reference: Mapping[str, Any],
    *,
    context: str,
    snapshot_context: Any | None = None,
) -> None:
    """Use the controller's recursive checksum verifier on one source graph."""

    try:
        control_plane._validate_referenced_artifact(  # type: ignore[attr-defined]
            reference,
            context=context,
            verified=set(),
            active=set(),
            snapshot_context=snapshot_context,
        )
    except control_plane.ReadinessError as exc:
        raise EvidenceError(str(exc)) from exc


def build_snapshot_gate(
    control: Mapping[str, Any],
    *,
    pre_repair_attestation: Path,
    legacy_consolidated_attestation: Path,
    output: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    sources = (
        ("pre_repair_external_attestation", pre_repair_attestation),
        ("legacy_consolidated_external_attestation", legacy_consolidated_attestation),
    )
    artifacts: list[dict[str, str]] = []
    roots: set[str] = set()
    snapshot_ids: set[str] = set()
    for name, path in sources:
        payload = _read_json(path)
        root_value = payload.get("snapshot_root")
        if not isinstance(root_value, str) or not Path(root_value).is_absolute():
            raise EvidenceError(f"snapshot attestation root is not absolute: {path}")
        roots.add(str(Path(root_value).resolve()))
        snapshot_ids.add(str(payload.get("snapshot_id")))
        artifacts.append(_artifact(name, path))
    if len(roots) != 2 or len(snapshot_ids) != 2:
        raise EvidenceError("pre-repair and consolidated snapshots must have distinct identities")
    # Publication invokes the controller's recursive validator exactly once.  That
    # validator rehashes every inventory payload and derives ID/count/bytes from the
    # sealed controls before the candidate envelope is renamed into place.
    return _publish_outer(
        control=control,
        gate="snapshot",
        metrics={
            "snapshot_count": 2,
            "pre_repair_verified": True,
            "legacy_consolidated_verified": True,
        },
        artifacts=artifacts,
        output=output,
        state_dir=state_dir,
    )


LEGACY_ARTIFACTS = {
    "response_incident_archive_report": "consolidated_response_report",
    "checkpoint_migration_report": "consolidated_checkpoint_report",
    "permanent_ledger_archive_report": "consolidated_permanent_report",
    "legacy_semantic_audit_report": "consolidated_semantic_report",
}

LEGACY_REPORT_FILENAMES = {
    "response_incident_archive_report": "response_incident_archive_report.json",
    "checkpoint_migration_report": "checkpoint_migration_report.json",
    "permanent_ledger_archive_report": "permanent_ledger_archive_report.json",
    "legacy_semantic_audit_report": "legacy_semantic_audit_report.json",
}

MIGRATION_COMPONENT_METRICS = {
    "response_incident_archive_report": {
        "protocol_incidents_total": 22,
        "protocol_already_reset": 22,
        "sealed_incident_qids": 1_064,
    },
    "checkpoint_migration_report": {
        "migrated_checkpoints": 47,
        "remaining_schema1_checkpoints": 0,
        "coordinates_preserved": True,
    },
    "permanent_ledger_archive_report": {
        "permanent_ledgers_archived": 3,
        "unresolved_permanent_ledgers": 0,
    },
}

MIGRATION_OUTER_METRICS = {
    "protocol_incidents_total": 22,
    "protocol_already_reset": 22,
    "sealed_incident_qids": 1_064,
    "migrated_checkpoints": 47,
    "remaining_schema1_checkpoints": 0,
    "permanent_ledgers_archived": 3,
    "unresolved_permanent_ledgers": 0,
}

SEMANTIC_METRICS = {
    "complete_cells": 740,
    "active_validated_qids": 888_068,
    "corrupt_cells": 0,
    "permanent_cells": 0,
    "malformed_lines": 0,
    "duplicate_qids": 0,
    "unexpected_qids": 0,
    "repair_count": 0,
}

EVIDENCE_ACCOUNTING = {
    "newly_sealed_incidents": 8,
    "newly_sealed_qids": 1_064,
    "preexisting_historical_incidents": 14,
    "preexisting_historical_qids": 355,
    "all_sealed_incident_qids": 1_419,
    "frozen_baseline_validated_qids": 889_132,
}


def _load_cleanup_sources(
    cleanup_marker: Path,
    *,
    snapshot_context: Any | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Path, dict[str, Any], dict[str, str]]],
    dict[str, str],
]:
    """Load cleanup evidence only through its sealed consolidated snapshot.

    The live cleanup reports remain useful operator conveniences, but their parent is
    writable.  The authoritative source is the byte-for-byte copy authenticated by
    ``legacy_consolidated.attestation.json``.  Live and sealed copies must still match
    so any post-snapshot drift fails closed instead of silently selecting one side.
    """

    supplied_marker = cleanup_marker.expanduser()
    if supplied_marker.is_symlink() or not supplied_marker.is_file():
        raise EvidenceError(
            f"legacy cleanup marker is missing or a symlink: {supplied_marker}"
        )
    marker_path = supplied_marker.resolve()
    recovery_root = marker_path.parent
    snapshot_root = recovery_root / "legacy_consolidated"
    attestation_path = recovery_root / "legacy_consolidated.attestation.json"
    attestation_reference = _artifact(
        "legacy_consolidated_external_attestation", attestation_path
    )
    _verify_recursive_reference(
        attestation_reference,
        context="legacy consolidated snapshot",
        snapshot_context=snapshot_context,
    )
    attestation = _read_json(attestation_path)
    if Path(str(attestation.get("snapshot_root", ""))).resolve() != snapshot_root:
        raise EvidenceError("legacy cleanup attestation addresses another snapshot")

    catalog = _read_json(snapshot_root / "SNAPSHOT_CATALOG.json")
    results_root = recovery_root.parent.parent
    expected_sources = [
        {"name": run_id, "path": str((results_root / run_id).resolve())}
        for run_id in (
            "full_sweep_v1",
            "full_sweep_agent_counts_v1",
            "full_sweep_agent_count_7_v1",
        )
    ] + [
        {
            "name": "dispatcher_v3",
            "path": str((results_root / ".dispatcher-v3").resolve()),
        },
        {
            "name": "legacy_cleanup_evidence",
            "path": str(
                (recovery_root / "operations" / "legacy_consolidation").resolve()
            ),
        },
        {
            "name": "legacy_cleanup_complete",
            "path": str(marker_path),
        },
    ]
    if catalog.get("sources") != expected_sources:
        raise EvidenceError(
            "legacy consolidated snapshot does not cover the exact cleanup sources"
        )

    sealed_marker_path = snapshot_root / "legacy_cleanup_complete"
    if _sha256(marker_path) != _sha256(sealed_marker_path):
        raise EvidenceError("live cleanup marker drifted from its sealed snapshot")
    marker = _read_json(sealed_marker_path)
    if (
        marker.get("schema_version") != 1
        or marker.get("status") != "complete"
        or marker.get("passed") is not True
        or not isinstance(marker.get("snapshot_id"), str)
        or not marker["snapshot_id"]
    ):
        raise EvidenceError("legacy cleanup marker is not a completed schema-1 marker")
    if marker.get("evidence_accounting") != EVIDENCE_ACCOUNTING:
        raise EvidenceError("legacy cleanup marker evidence accounting drifted")
    records = marker.get("artifacts")
    if not isinstance(records, list):
        raise EvidenceError("legacy cleanup marker artifacts must be an array")
    names = [row.get("name") if isinstance(row, dict) else None for row in records]
    if len(names) != len(set(names)) or set(names) != set(LEGACY_ARTIFACTS):
        raise EvidenceError(
            f"legacy cleanup marker must bind exactly {sorted(LEGACY_ARTIFACTS)}"
        )
    live_operations = (recovery_root / "operations" / "legacy_consolidation").resolve()
    sealed_operations = snapshot_root / "legacy_cleanup_evidence"
    loaded: dict[str, tuple[Path, dict[str, Any], dict[str, str]]] = {}

    def verify_member_graph(
        *, live_path: Path, sealed_path: Path, expected_sha256: str, seen: set[Path]
    ) -> dict[str, Any]:
        if sealed_path in seen:
            raise EvidenceError(f"duplicate/cyclic sealed cleanup report: {sealed_path}")
        seen.add(sealed_path)
        if (
            live_path.is_symlink()
            or not live_path.is_file()
            or sealed_path.is_symlink()
            or not sealed_path.is_file()
            or _sha256(live_path) != expected_sha256
            or _sha256(sealed_path) != expected_sha256
        ):
            raise EvidenceError(
                f"live/sealed cleanup evidence differs: {live_path}"
            )
        payload = _read_json(sealed_path)
        references = payload.get("referenced_artifacts")
        if references is None:
            return payload
        if not isinstance(references, list):
            raise EvidenceError(f"cleanup report references are invalid: {sealed_path}")
        names: set[str] = set()
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {
                "name",
                "path",
                "sha256",
            }:
                raise EvidenceError(f"cleanup report reference is invalid: {sealed_path}")
            name = reference.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise EvidenceError(
                    f"cleanup report reference names are invalid: {sealed_path}"
                )
            names.add(name)
            nested_live = Path(str(reference.get("path", "")))
            try:
                relative = nested_live.resolve().relative_to(live_operations)
            except (OSError, ValueError) as exc:
                raise EvidenceError(
                    f"cleanup report reference escapes operation root: {nested_live}"
                ) from exc
            verify_member_graph(
                live_path=nested_live,
                sealed_path=sealed_operations / relative,
                expected_sha256=str(reference.get("sha256", "")),
                seen=seen,
            )
        return payload

    seen: set[Path] = set()
    for row in records:
        assert isinstance(row, dict)
        if set(row) != {"name", "path", "sha256"}:
            raise EvidenceError("legacy cleanup artifact reference has wrong fields")
        name = str(row["name"])
        live_path = live_operations / LEGACY_REPORT_FILENAMES[name]
        if row.get("path") != str(live_path):
            raise EvidenceError(f"legacy cleanup artifact path drifted: {name}")
        sealed_path = sealed_operations / LEGACY_REPORT_FILENAMES[name]
        raw = verify_member_graph(
            live_path=live_path,
            sealed_path=sealed_path,
            expected_sha256=str(row.get("sha256", "")),
            seen=seen,
        )
        loaded[name] = (
            sealed_path,
            raw,
            {
                "snapshot_id": str(attestation["snapshot_id"]),
                "snapshot_root": str(snapshot_root),
                "logical_path": sealed_path.relative_to(snapshot_root).as_posix(),
                "sha256": _sha256(sealed_path),
            },
        )
    return marker, loaded, attestation_reference


def _sealed_source_proxy(
    raw: Mapping[str, Any],
    *,
    membership: Mapping[str, str],
    attestation_reference: Mapping[str, str],
) -> dict[str, Any]:
    """Replace mutable nested paths with one recursive sealed-snapshot proof."""

    proxy = dict(raw)
    proxy["sealed_snapshot_member"] = dict(membership)
    proxy["referenced_artifacts"] = [dict(attestation_reference)]
    return proxy


def build_migrations_gate(
    control: Mapping[str, Any],
    *,
    cleanup_marker: Path,
    output: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    snapshot_context = _prepared_snapshot_context(control, state_dir=state_dir)
    marker, sources, snapshot_reference = _load_cleanup_sources(
        cleanup_marker, snapshot_context=snapshot_context
    )
    _require_exact_metrics(
        marker.get("migration_metrics"),
        MIGRATION_OUTER_METRICS,
        context="legacy cleanup marker migration",
    )
    artifacts: list[dict[str, str]] = []
    for name, expected in MIGRATION_COMPONENT_METRICS.items():
        _raw_path, raw, membership = sources[name]
        if raw.get("schema_version") != 1 or raw.get("passed") is not True:
            raise EvidenceError(
                f"legacy consolidation source did not pass: {_raw_path}"
            )
        observed = {key: raw.get(key) for key in expected}
        _require_exact_metrics(observed, expected, context=name)
        sealed_source_path = _artifact_path(output, f"{name}.sealed-source")
        _write_json_atomic(
            sealed_source_path,
            _sealed_source_proxy(
                raw,
                membership=membership,
                attestation_reference=snapshot_reference,
            ),
        )
        wrapper_path = _artifact_path(output, name)
        _write_json_atomic(
            wrapper_path,
            _wrapper(
                name=name,
                immutable_sha256=str(control["immutable_sha256"]),
                metrics=expected,
                references=[
                    _artifact(LEGACY_ARTIFACTS[name], sealed_source_path)
                ],
            ),
        )
        artifacts.append(_artifact(name, wrapper_path))
    return _publish_outer(
        control=control,
        gate="migrations",
        metrics=MIGRATION_OUTER_METRICS,
        artifacts=artifacts,
        output=output,
        state_dir=state_dir,
    )


def build_semantic_gate(
    control: Mapping[str, Any],
    *,
    cleanup_marker: Path,
    output: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    snapshot_context = _prepared_snapshot_context(control, state_dir=state_dir)
    marker, sources, snapshot_reference = _load_cleanup_sources(
        cleanup_marker, snapshot_context=snapshot_context
    )
    _require_exact_metrics(
        marker.get("semantic_metrics"), SEMANTIC_METRICS, context="cleanup semantic"
    )
    _raw_path, raw, membership = sources["legacy_semantic_audit_report"]
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "legacy_semantic_audit"
        or raw.get("passed") is not True
        or raw.get("manifest_cells") != 22_680
        or raw.get("invalid_rows") != 0
    ):
        raise EvidenceError("legacy semantic source identity/acceptance drifted")
    _require_exact_metrics(raw.get("metrics"), SEMANTIC_METRICS, context="semantic source")
    name = "legacy_semantic_audit_report"
    sealed_source_path = _artifact_path(output, f"{name}.sealed-source")
    _write_json_atomic(
        sealed_source_path,
        _sealed_source_proxy(
            raw,
            membership=membership,
            attestation_reference=snapshot_reference,
        ),
    )
    wrapper_path = _artifact_path(output, name)
    _write_json_atomic(
        wrapper_path,
        _wrapper(
            name=name,
            immutable_sha256=str(control["immutable_sha256"]),
            metrics=SEMANTIC_METRICS,
            references=[_artifact(LEGACY_ARTIFACTS[name], sealed_source_path)],
        ),
    )
    return _publish_outer(
        control=control,
        gate="semantic_audit",
        metrics=SEMANTIC_METRICS,
        artifacts=[_artifact(name, wrapper_path)],
        output=output,
        state_dir=state_dir,
    )


def _run(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=False, timeout=timeout
    )


def _registry_records(pool_root: Path) -> dict[str, tuple[ServerEntry, Path]]:
    servers = pool_root / "servers"
    if not servers.is_dir() or servers.is_symlink():
        raise EvidenceError(f"canonical server registry is missing: {servers}")
    records: dict[str, tuple[ServerEntry, Path]] = {}
    for path in sorted(servers.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"unsafe server registry artifact: {path}")
        payload = _read_json(path)
        try:
            entry = ServerEntry(**payload)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"malformed registry record {path}: {exc}") from exc
        replica_id = entry.replica_id
        if not isinstance(replica_id, str) or not replica_id:
            raise EvidenceError(f"registry record lacks exact replica identity: {path}")
        if replica_id in records:
            raise EvidenceError(f"duplicate registry records for replica {replica_id}")
        records[replica_id] = (entry, path.resolve())
    return records


def _http_probe(entry: ServerEntry, expected_model: str, timeout: float) -> dict[str, Any]:
    started = time.time()

    def fetch(path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(  # noqa: S310 - frozen in-cluster endpoint
                f"http://{entry.host}:{entry.port}{path}", timeout=timeout
            ) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), b""
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return 0, b""

    health_status, _ = fetch("/health")
    models_status, body = fetch("/v1/models")
    model_ids: list[str] = []
    if models_status == 200:
        try:
            payload = json.loads(body.decode("utf-8"))
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                model_ids = sorted(
                    str(row["id"])
                    for row in data
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                )
        except (UnicodeError, json.JSONDecodeError):
            model_ids = []
    completed = time.time()
    return {
        "health_status": health_status,
        "models_status": models_status,
        "model_ids": model_ids,
        "expected_model": expected_model,
        "probe_started_timestamp": started,
        "probe_completed_timestamp": completed,
        "healthy": (
            health_status == 200
            and models_status == 200
            and expected_model in model_ids
        ),
    }


def _combine_http_probe_rounds(
    rounds: Sequence[Mapping[str, Any]], *, expected_model: str
) -> dict[str, Any]:
    """Collapse two successful endpoint observations into the schema-2 evidence shape."""

    if len(rounds) != 2:
        raise EvidenceError("fleet readiness requires exactly two HTTP probe rounds")
    required = {
        "health_status",
        "models_status",
        "model_ids",
        "expected_model",
        "probe_started_timestamp",
        "probe_completed_timestamp",
        "healthy",
    }
    model_sets: list[set[str]] = []
    starts: list[float] = []
    completions: list[float] = []
    for index, result in enumerate(rounds, start=1):
        if not isinstance(result, Mapping) or set(result) != required:
            raise EvidenceError(f"fleet HTTP probe round {index} has invalid fields")
        started = result["probe_started_timestamp"]
        completed = result["probe_completed_timestamp"]
        models = result["model_ids"]
        if (
            result["expected_model"] != expected_model
            or not isinstance(started, (int, float))
            or isinstance(started, bool)
            or not isinstance(completed, (int, float))
            or isinstance(completed, bool)
            or not math.isfinite(float(started))
            or not math.isfinite(float(completed))
            or float(started) > float(completed)
            or not isinstance(models, list)
            or any(not isinstance(model, str) or not model for model in models)
        ):
            raise EvidenceError(f"fleet HTTP probe round {index} is malformed")
        starts.append(float(started))
        completions.append(float(completed))
        model_sets.append(set(models))
    common_models = sorted(model_sets[0] & model_sets[1])
    healthy = all(
        result["healthy"] is True
        and result["health_status"] == 200
        and result["models_status"] == 200
        for result in rounds
    ) and expected_model in common_models
    return {
        "health_status": 200 if all(
            result["health_status"] == 200 for result in rounds
        ) else 0,
        "models_status": 200 if all(
            result["models_status"] == 200 for result in rounds
        ) else 0,
        "model_ids": common_models,
        "expected_model": expected_model,
        "probe_started_timestamp": min(starts),
        "probe_completed_timestamp": max(completions),
        "healthy": healthy,
    }


def _paused_next_generation_runtime_environment(
    control: Mapping[str, Any], *, state_dir: Path
) -> dict[str, str]:
    """Create/verify the exact g+1 runtime proof used by the paused fleet.

    Fleet readiness precedes production ``resume``.  Its servers nevertheless become
    the generation-one production fleet, so their immutable sbatch bytes must carry the
    same attestation and lease that resume will later adopt.  This helper mirrors the
    renderer bootstrap under the controller lock without changing desired state.
    """

    resolved_state = state_dir.expanduser().resolve()
    try:
        with control_plane.control_lock(resolved_state):
            live = control_plane.load_control(resolved_state, verify_files=True)
            if live.get("immutable_sha256") != control.get("immutable_sha256"):
                raise EvidenceError(
                    "fleet readiness control changed after immutable verification"
                )
            if live.get("desired_state") != "paused":
                raise EvidenceError(
                    "fleet readiness requires paused, non-draining schema-5 control"
                )
            if live.get("drain_requested") is not False:
                try:
                    control_plane.validate_capacity_readiness_pending_state(
                        live,
                        require_fleet_gate=False,
                        # load_control(..., verify_files=True) immediately above has
                        # already verified the active overlay and marker.
                        verify_files=False,
                    )
                except control_plane.ControlError as exc:
                    raise EvidenceError(
                        "fleet readiness requires paused, non-draining control or "
                        f"an exact capacity readiness transition: {exc}"
                    ) from exc
            generation = live.get("rollout_generation")
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 0
            ):
                raise EvidenceError("fleet readiness control generation is invalid")
            next_generation = generation + 1
            attestation = control_plane.ensure_runtime_integrity_attestation(
                resolved_state,
                live,
                generation=next_generation,
                force_full=False,
            )
            projected = dict(live)
            projected["rollout_generation"] = next_generation
            projected[control_plane.RUNTIME_ATTESTATION_STATE_KEY] = attestation
            control_plane.validate_runtime_integrity_attestation(
                projected, verify_metadata=True
            )
            environment = control_plane.production_environment(projected)
    except (control_plane.ControlError, OSError) as exc:
        raise EvidenceError(f"fleet runtime integrity proof failed: {exc}") from exc
    required = {
        "ASYS_RUNTIME_ATTESTATION",
        "ASYS_RUNTIME_ATTESTATION_SHA256",
        "ASYS_RUNTIME_INTEGRITY_LEASE",
        "ASYS_IMMUTABLE_PINS_SHA256",
        "ASYS_ROLLOUT_GENERATION",
    }
    if not required.issubset(environment):
        raise EvidenceError("fleet runtime integrity environment is incomplete")
    return {name: str(environment[name]) for name in required}


def _validate_capacity_wave_admission_fence(
    control: Mapping[str, Any],
    *,
    fleet_binding: Mapping[str, Any],
    contract: protected_capacity.ProtectedCapacityContract,
) -> dict[str, Any]:
    """Join the fleet authority to its configured ceiling and saturation cut.

    The 384-client value is a Slurm admission ceiling, not a promise of 384
    simultaneous endpoint leases.  The signed WDRR selection is the independently
    certified saturation target that qualification keeps filled and refilled.
    A sub-ceiling target is valid in every generation and is not itself a reason
    to pause admission or initiate a capacity transition.
    """

    del control
    authority = fleet_binding.get("protected_capacity")
    summary = {
        "capacity_generation": contract.capacity_generation,
        "path": str(contract.static_feasibility_certificate_path),
        "sha256": contract.static_feasibility_certificate_sha256,
        "certificate_id": contract.static_feasibility_certificate_id,
        "effective_fleet_contract_sha256": (
            contract.effective_fleet_contract_sha256
        ),
        "wave_passed": contract.static_feasibility_wave_passed,
        "selected_cell_count": (
            contract.static_feasibility_selected_cell_count
        ),
        "target_cell_count": contract.static_feasibility_target_cell_count,
        "shortfall_cells": contract.static_feasibility_shortfall_cells,
        "configured_client_ceiling": (
            contract.static_feasibility_configured_client_ceiling
        ),
        "certified_saturation_target": (
            contract.static_feasibility_certified_saturation_target
        ),
    }
    if (
        not isinstance(authority, Mapping)
        or type(fleet_binding.get("capacity_generation")) is not int
        or fleet_binding.get("capacity_generation")
        != contract.capacity_generation
        or fleet_binding.get("sha256")
        != contract.effective_fleet_contract_sha256
        or authority.get("static_feasibility_certificate_path")
        != summary["path"]
        or authority.get("static_feasibility_certificate_sha256")
        != summary["sha256"]
        or authority.get("static_feasibility_certificate_id")
        != summary["certificate_id"]
        or authority.get("effective_fleet_contract_sha256")
        != summary["effective_fleet_contract_sha256"]
        or type(summary["selected_cell_count"]) is not int
        or type(summary["target_cell_count"]) is not int
        or type(summary["shortfall_cells"]) is not int
        or not 0 < summary["selected_cell_count"] <= 384
        or summary["target_cell_count"] != 384
        or summary["configured_client_ceiling"] != 384
        or summary["configured_client_ceiling"]
        != summary["target_cell_count"]
        or summary["certified_saturation_target"]
        != summary["selected_cell_count"]
        or summary["shortfall_cells"]
        != summary["target_cell_count"] - summary["selected_cell_count"]
        or summary["wave_passed"]
        is not (summary["shortfall_cells"] == 0)
    ):
        raise EvidenceError(
            "fleet readiness capacity generation/certificate/wave binding drifted"
        )
    return summary


def _verify_registry(
    *,
    pool_root: Path,
    fleet: FrozenFleetContract,
    models: FrozenModelContracts,
    pins: Mapping[str, Any],
    scheduler: Mapping[
        str, tuple[keepalive.FleetQueueRow, keepalive.SpooledServingProvenance]
    ],
    replica_ids: set[str] | None = None,
    records: dict[str, tuple[ServerEntry, Path]] | None = None,
) -> dict[str, tuple[ServerEntry, Path]]:
    records = _registry_records(pool_root) if records is None else dict(records)
    expected_ids = (
        {replica.replica_id for replica in fleet.replicas}
        if replica_ids is None
        else set(replica_ids)
    )
    if set(records) != expected_ids:
        raise EvidenceError(
            "registry replica set drifted: "
            f"missing={sorted(expected_ids - set(records))}, "
            f"unexpected={sorted(set(records) - expected_ids)}"
        )
    for replica in (
        replica for replica in fleet.replicas if replica.replica_id in expected_ids
    ):
        entry, path = records[replica.replica_id]
        row, provenance = scheduler[replica.replica_id]
        identity = models.for_size(replica.model_size)
        if not entry_matches_frozen_provenance(
            entry,
            replica.serving_profile,
            release_id=str(pins["release_id"]),
            environment_hash=str(pins["serving_environment_sha256"]),
            model_revision=identity.model_revision,
            tokenizer_id=identity.tokenizer_id,
            tokenizer_revision=identity.tokenizer_revision,
            model_contract_sha256=str(pins["model_contract_sha256"]),
            fleet_contract_sha256=str(pins["fleet_contract_sha256"]),
            expected_server_pool_id=fleet.fleet_id,
        ):
            raise EvidenceError(f"frozen registry provenance drift for {path}")
        expected_path = (
            pool_root
            / "servers"
            / replica.serving_profile
            / f"{entry.host}_{entry.port}.json"
        ).resolve()
        if (
            entry.replica_index != replica.replica_index
            or entry.model_size != replica.model_size
            or entry.slurm_job_id != row.job_id
            or entry.host != row.node
            or entry.port != _port_for(replica.serving_profile, replica.replica_index)
            or path != expected_path
            or provenance.run_root != str(pool_root)
            or provenance.server_pool_id != replica.pool_id
            or provenance.replica_id != replica.replica_id
            or provenance.replica_index != replica.replica_index
            or provenance.release_id != pins["release_id"]
            or provenance.environment_hash != pins["serving_environment_sha256"]
            or entry.model_revision != provenance.model_revision
            or entry.tokenizer_id != provenance.tokenizer_id
            or entry.tokenizer_revision != provenance.tokenizer_revision
            or provenance.model_contract_sha256 != pins["model_contract_sha256"]
            or provenance.fleet_contract_sha256 != pins["fleet_contract_sha256"]
        ):
            raise EvidenceError(f"registry/scheduler/spool mismatch for {replica.replica_id}")
        if (
            not isinstance(entry.started_at, (int, float))
            or isinstance(entry.started_at, bool)
            or not math.isfinite(float(entry.started_at))
            or float(entry.started_at) <= 0
        ):
            raise EvidenceError(f"invalid registration timestamp for {replica.replica_id}")
    return records


def _successful_capacity_handoff_history(
    *,
    current_ledger: Mapping[str, Any],
    logical_allocations: Sequence[Any],
) -> set[str]:
    """Return terminal predecessor IDs from fully sealed successful handoffs.

    Repeated 10-hour capacity waits can cross the fleet supervisor's 12-hour warm
    handoff boundary.  Such a handoff leaves its exact predecessor as a terminal,
    ``retiring`` attempt.  It is safe history only when every predecessor has one
    promoted, dual-probed successor and the chain ends at the sole active allocation.
    """

    active_by_replica = {
        allocation.replica_id: allocation for allocation in logical_allocations
    }
    allowed_terminal_ids: set[str] = set()
    replicas = current_ledger.get("replicas")
    if not isinstance(replicas, Mapping) or set(replicas) != set(active_by_replica):
        raise EvidenceError("current fleet ledger replica identity drifted")
    for replica_id, record in replicas.items():
        attempts = record.get("attempts") if isinstance(record, Mapping) else None
        active = active_by_replica[replica_id]
        if not isinstance(attempts, list) or not attempts:
            raise EvidenceError(
                f"current fleet generation lacks attempts for {replica_id}"
            )
        active_matches = [
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and attempt.get("intent_token")
            == active.attempt.get("intent_token")
        ]
        if len(active_matches) != 1:
            raise EvidenceError(
                f"current logical allocation is not uniquely ledger-bound: {replica_id}"
            )
        active_attempt = active_matches[0]
        if (
            active_attempt.get("state") != "committed"
            or active_attempt.get("job_id") != str(active.row.job_id)
            or active_attempt.get("committed_at") is None
            or active_attempt.get("last_error") is not None
            or active_attempt.get("retire_error") is not None
            or active_attempt.get("lifecycle") not in {"primary", "promoted"}
        ):
            raise EvidenceError(
                f"current logical allocation is not a clean commit: {replica_id}"
            )
        by_job: dict[str, Mapping[str, Any]] = {}
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise EvidenceError(
                    f"fleet attempt is malformed for {replica_id}"
                )
            job_id = str(attempt.get("job_id") or "")
            if not job_id.isdigit() or job_id in by_job:
                raise EvidenceError(
                    f"fleet attempt job identity is ambiguous for {replica_id}"
                )
            by_job[job_id] = attempt
        terminal = [
            attempt for attempt in attempts if attempt is not active_attempt
        ]
        terminal_ids = {str(attempt["job_id"]) for attempt in terminal}
        for predecessor in terminal:
            predecessor_id = str(predecessor["job_id"])
            successors = [
                candidate
                for candidate in attempts
                if candidate.get("launch_kind") == "handoff"
                and str(candidate.get("predecessor_job_id") or "")
                == predecessor_id
            ]
            if (
                predecessor.get("state") != "terminal"
                or predecessor.get("lifecycle") != "retiring"
                or predecessor.get("terminal_at") is None
                or predecessor.get("last_error") is not None
                or predecessor.get("retire_requested_at") is None
                or not isinstance(predecessor.get("retire_attempts"), int)
                or isinstance(predecessor.get("retire_attempts"), bool)
                or predecessor["retire_attempts"] < 1
                or predecessor.get("last_retire_attempt_at") is None
                or predecessor.get("retire_error") is not None
                or len(successors) != 1
            ):
                raise EvidenceError(
                    "current generation contains failed/orphaned fleet history for "
                    f"{replica_id} job {predecessor_id}"
                )
            successor = successors[0]
            if (
                successor.get("state") not in {"committed", "terminal"}
                or successor.get("lifecycle") not in {"promoted", "retiring"}
                or successor.get("promoted_at") is None
                or not isinstance(successor.get("ready_probe_count"), int)
                or isinstance(successor.get("ready_probe_count"), bool)
                or successor["ready_probe_count"] < 2
                or successor.get("last_error") is not None
                or successor.get("retire_error") is not None
                ):
                    raise EvidenceError(
                        "fleet handoff successor is failed/orphaned or not "
                        f"sealed/healthy for {replica_id}"
                    )
            allowed_terminal_ids.add(predecessor_id)

        # Every historical chain must reach the exact live tail, not a disjoint
        # promoted/retiring component.
        active_job_id = str(active_attempt["job_id"])
        for predecessor_id in terminal_ids:
            visited: set[str] = set()
            current = predecessor_id
            while current != active_job_id:
                if current in visited:
                    raise EvidenceError(
                        f"fleet handoff history cycles for {replica_id}"
                    )
                visited.add(current)
                successors = [
                    str(candidate["job_id"])
                    for candidate in attempts
                    if candidate.get("launch_kind") == "handoff"
                    and str(candidate.get("predecessor_job_id") or "") == current
                ]
                if len(successors) != 1:
                    raise EvidenceError(
                        f"fleet handoff history is disconnected for {replica_id}"
                    )
                current = successors[0]
                if current not in by_job:
                    raise EvidenceError(
                        f"fleet handoff successor is absent for {replica_id}"
                    )
    return allowed_terminal_ids


def _validate_capacity_registry_membership(
    records: Mapping[str, tuple[ServerEntry, Path]],
    *,
    running_ids: set[str],
    pending_ids: set[str],
) -> None:
    stale_pending_registry = sorted(set(records) & pending_ids)
    unexpected_registry = sorted(set(records) - running_ids)
    missing_running_registry = sorted(running_ids - set(records))
    if stale_pending_registry or unexpected_registry or missing_running_registry:
        raise EvidenceError(
            "capacity-transient registry is not the exact RUNNING allocation set: "
            f"pending_pointers={stale_pending_registry}, "
            f"unexpected={unexpected_registry}, missing={missing_running_registry}"
        )


def _scope_capacity_scheduler_rows(
    rows: Sequence[fleet_tx.SchedulerRow],
    *,
    fleet: FrozenFleetContract,
) -> tuple[tuple[fleet_tx.SchedulerRow, ...], tuple[str, ...]]:
    """Separate isolated canary history from the exact production namespace."""

    expected_by_replica = {
        replica.replica_id: replica for replica in fleet.replicas
    }
    expected_names = {
        replica.scheduler_job_name for replica in fleet.replicas
    }
    production: list[fleet_tx.SchedulerRow] = []
    isolated_foreign: list[str] = []
    for row in rows:
        parsed = fleet_tx.parse_intent_comment(row.comment)
        if parsed is None:
            claims_production = (
                row.job_name in expected_names
                or f"pool={fleet.fleet_id}" in row.comment
                or f"fleet={fleet.sha256}" in row.comment
                or any(
                    f"replica={replica_id}" in row.comment
                    for replica_id in expected_by_replica
                )
            )
            if claims_production:
                raise EvidenceError(
                    "malformed scheduler row collides with the production fleet "
                    f"namespace: job {row.job_id}"
                )
            isolated_foreign.append(row.job_id)
            continue
        replica = expected_by_replica.get(parsed["replica"])
        exact = (
            parsed["pool"] == fleet.fleet_id
            and parsed["fleet"] == fleet.sha256
            and replica is not None
            and parsed["profile"] == replica.serving_profile
            and row.job_name == replica.scheduler_job_name
        )
        foreign = (
            parsed["pool"] != fleet.fleet_id
            and parsed["fleet"] != fleet.sha256
            and replica is None
            and row.job_name not in expected_names
        )
        if exact:
            production.append(row)
        elif foreign:
            isolated_foreign.append(row.job_id)
        else:
            raise EvidenceError(
                "scheduler row partially collides with production fleet identity: "
                f"job {row.job_id}"
            )
    return (
        tuple(sorted(production, key=lambda row: int(row.job_id))),
        tuple(sorted(isolated_foreign, key=int)),
    )


def build_fleet_gate(
    control: Mapping[str, Any],
    *,
    output: Path,
    state_dir: Path,
    probe_timeout: float = 5.0,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None = None,
    probe: Callable[[ServerEntry, str, float], Mapping[str, Any]] = _http_probe,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if not 0 < probe_timeout <= 60:
        raise EvidenceError("probe timeout must be in (0, 60] seconds")
    runtime_environment = _paused_next_generation_runtime_environment(
        control, state_dir=state_dir
    )
    generation = int(runtime_environment["ASYS_ROLLOUT_GENERATION"])
    pins = control["immutable"]
    fleet_binding = control_plane.effective_fleet_contract_binding(
        control, verify_files=True
    )
    effective_pins = dict(pins)
    effective_pins.update(
        {
            "fleet_contract_path": fleet_binding["path"],
            "fleet_contract_sha256": fleet_binding["sha256"],
        }
    )
    pool_root = Path(str(pins["server_pool_root"])).expanduser().resolve()
    if server_pool_id(pool_root) != "schema5-v1":
        raise EvidenceError(f"server pool root has the wrong identity: {pool_root}")
    models = load_model_contracts(
        pins["model_contract_path"], expected_sha256=pins["model_contract_sha256"]
    )
    fleet = load_fleet_contract(
        fleet_binding["path"],
        model_contracts=models,
        expected_sha256=fleet_binding["sha256"],
        allow_capacity_layout=True,
    )
    fleet.verify_pool_root(pool_root)
    if (
        len(fleet.replicas) != fleet_binding["logical_replicas"]
        or sum(replica.gpus_per_replica for replica in fleet.replicas)
        != fleet_binding["allocated_gpus"]
    ):
        raise EvidenceError("fleet readiness differs from its capacity generation")
    if int(fleet_binding["capacity_generation"]) == 1:
        try:
            initial_capacity = (
                control_plane.load_effective_protected_capacity_contract(
                    control, verify_files=True
                )
            )
        except (
            control_plane.ControlError,
            control_plane.ImmutablePinError,
            protected_capacity.ProtectedCapacityError,
        ) as exc:
            raise EvidenceError(
                f"prequalification capacity baseline is invalid: {exc}"
            ) from exc
        if (
            fleet_binding["logical_replicas"] != 22
            or fleet_binding["allocated_gpus"] != 24
            or initial_capacity.effective_active_logical_replicas != 22
            or initial_capacity.effective_active_gpus != 24
            or initial_capacity.additive_reserved_logical_replicas != 0
            or initial_capacity.additive_reserved_gpus != 0
            or initial_capacity.retained_warm_turnover_job_elements != 3
            or initial_capacity.retained_warm_turnover_gpus != 4
            or initial_capacity.attested_total_gpus != 28
            or initial_capacity.job_element_accounting.get(
                "controller_monitor_other_held_job_elements"
            )
            != 39
            or initial_capacity.effective_fleet_contract_sha256
            != initial_capacity.base_fleet_contract_sha256
        ):
            raise EvidenceError(
                "prequalification fleet readiness must use exactly 22 logical "
                "replicas/24 active GPUs, zero additive replicas, 3 warm jobs/"
                "4 warm GPUs, 28 attested GPUs, and 39 held non-cell slots"
            )

    transaction_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None
    control_scheduler_reader = None
    if scheduler_runner is not None:
        def adapted_scheduler_runner(argv, **kwargs):
            return scheduler_runner(argv, float(kwargs.get("timeout", 15.0)))

        transaction_runner = adapted_scheduler_runner

        def control_scheduler_reader():
            return control_plane.query_scheduler(
                runner=lambda argv: scheduler_runner(list(argv), 15.0),
                now=float(now()),
            )

    try:
        fleet_snapshot = keepalive.reconcile_fleet_read_only(
            str(pool_root),
            fleet,
            current_generation=generation,
            scheduler_runner=transaction_runner,
            scheduler_now=float(now()),
        )
        trusted_scientific_provenance = (
            control_plane.reconcile_trusted_scientific_job_provenance(
                state_dir,
                fleet_bindings=keepalive.trusted_scientific_fleet_bindings(
                    fleet_snapshot
                ),
                fleet_contract_sha256=fleet.sha256,
                fleet_generation=generation,
                scheduler_reader=control_scheduler_reader,
                now=float(now()),
                allow_exact_cell_quiescence=True,
            )
        )
    except (
        control_plane.ControlError,
        keepalive.FleetContractError,
        OSError,
    ) as exc:
        raise EvidenceError(
            f"transactional scientific occupancy reconciliation failed: {exc}"
        ) from exc

    try:
        transport_binding = (
            scheduler_safety.validate_transport_uncertainty_binding(
                pins["transport_uncertainty_binding"]
            )
        )
        transport_binding_sha256 = (
            scheduler_safety.transport_uncertainty_binding_sha256(
                transport_binding
            )
        )
        if (
            pins.get("transport_uncertainty_binding_sha256")
            != transport_binding_sha256
        ):
            raise scheduler_safety.SchedulerSafetyError(
                "immutable transport binding digest drifted"
            )
        partition_time_requirements = (
            scheduler_safety.fleet_partition_time_requirements(
                fleet.replicas
            )
        )
        scheduler_evidence = (
            scheduler_safety.capture_scheduler_safety_evidence(
                list(partition_time_requirements),
                runner=_transaction_runner(scheduler_runner),
                captured_timestamp=float(now()),
            )
        )
        scheduler_policy = (
            scheduler_safety.validate_scheduler_safety_evidence(
                scheduler_evidence,
                expected_partitions=list(partition_time_requirements),
                required_time_limits_seconds=partition_time_requirements,
            )
        )
        client_capacity_summary = (
            control_plane.client_capacity_contract_from_state(state_dir)
        )
        protected_contract = (
            control_plane.load_effective_protected_capacity_contract(
                control, verify_files=True
            )
        )
        _validate_capacity_wave_admission_fence(
            control,
            fleet_binding=fleet_binding,
            contract=protected_contract,
        )
        protected_capacity.authorize_client(
            protected_contract,
            partition=str(client_capacity_summary["partition"]),
            qos=str(client_capacity_summary["qos"]),
            required_slots=int(
                client_capacity_summary["authorized_cell_ceiling"]
            ),
            required_reserve_jobs=int(
                client_capacity_summary["reserve_jobs"]
            ),
        )
        client_capacity = protected_capacity.capture_live_client_capacity(
            protected_contract,
            partition=str(client_capacity_summary["partition"]),
            qos=str(client_capacity_summary["qos"]),
            required_time_limit_seconds=(
                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
            ),
            trusted_scientific_job_provenance=(
                trusted_scientific_provenance
            ),
            runner=_transaction_runner(scheduler_runner),
            captured_timestamp=float(now()),
        )
        client_capacity_summary = (
            protected_capacity.validate_live_client_capacity_evidence(
                protected_contract,
                client_capacity,
                partition=str(client_capacity_summary["partition"]),
                qos=str(client_capacity_summary["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
            )
        )
    except (
        scheduler_safety.SchedulerSafetyError,
        protected_capacity.ProtectedCapacityError,
    ) as exc:
        raise EvidenceError(
            f"non-preemptible client partition/QOS readiness failed: {exc}"
        ) from exc

    scheduler = {
        allocation.replica_id: (
            allocation.row,
            allocation.spooled_provenance,
        )
        for allocation in fleet_snapshot.allocations
    }
    exact_scheduler_proofs: dict[str, dict[str, Any]] = {}
    exact_spooled_proofs: dict[str, dict[str, Any]] = {}
    allocation_by_replica = {
        allocation.replica_id: allocation
        for allocation in fleet_snapshot.allocations
    }
    if (
        fleet_snapshot.current_generation != generation
        or set(allocation_by_replica)
        != {replica.replica_id for replica in fleet.replicas}
    ):
        raise EvidenceError(
            "ordinary fleet readiness ledger generation/replica set drifted"
        )
    for replica in fleet.replicas:
        allocation = allocation_by_replica[replica.replica_id]
        parsed_comment = fleet_tx.parse_intent_comment(allocation.row.comment)
        if (
            allocation.ledger_generation != generation
            or allocation.attempt_state != "committed"
            or parsed_comment is None
            or parsed_comment.get("pool") != fleet.fleet_id
            or parsed_comment.get("profile") != replica.serving_profile
            or parsed_comment.get("replica") != replica.replica_id
            or parsed_comment.get("generation") != str(generation)
            or parsed_comment.get("intent") != allocation.intent_token
            or parsed_comment.get("fleet") != fleet.sha256
        ):
            raise EvidenceError(
                f"ordinary fleet readiness intent/ledger drift for "
                f"{replica.replica_id}"
            )
        script_path = _require_read_only_regular(
            Path(allocation.sbatch_path),
            description=f"fleet readiness script for {replica.replica_id}",
        )
        if _sha256(script_path) != allocation.sbatch_sha256:
            raise EvidenceError(
                f"fleet readiness local script hash drift for {replica.replica_id}"
            )
        scontrol_proof, reason = _exact_capacity_job_state(
            row=allocation.row,
            script_path=script_path,
            scheduler_runner=scheduler_runner,
            include_raw_output=True,
        )
        if reason is not None or scontrol_proof["job_state"] != "RUNNING":
            raise EvidenceError(
                f"ordinary fleet readiness requires RUNNING job {allocation.row.job_id}"
            )
        exact_scheduler_proofs[replica.replica_id] = scontrol_proof
        exact_spooled_proofs[replica.replica_id] = _exact_spooled_script_proof(
            job_id=str(allocation.row.job_id),
            script_path=script_path,
            scheduler_runner=scheduler_runner,
        )
    records = _verify_registry(
        pool_root=pool_root,
        fleet=fleet,
        models=models,
        pins=effective_pins,
        scheduler=scheduler,
    )

    def probe_one(replica_id: str) -> tuple[str, Mapping[str, Any]]:
        entry, _ = records[replica_id]
        replica = next(item for item in fleet.replicas if item.replica_id == replica_id)
        profile = fleet.by_profile[replica.serving_profile]
        del profile  # profile existence was frozen by the fleet loader
        from agents_scaling.serving.profiles import get_serving_profile

        expected_model = get_serving_profile(replica.serving_profile).served_model_name
        rounds = [
            probe(entry, expected_model, probe_timeout),
            probe(entry, expected_model, probe_timeout),
        ]
        return replica_id, _combine_http_probe_rounds(
            rounds, expected_model=expected_model
        )

    with ThreadPoolExecutor(max_workers=len(fleet.replicas)) as executor:
        probes = dict(executor.map(probe_one, sorted(records)))
    captured = float(now())
    scheduler_age = captured - float(fleet_snapshot.captured_at)
    if not 0 <= scheduler_age <= 600:
        raise EvidenceError("joined fleet scheduler evidence is already stale")
    unhealthy = sorted(
        replica_id
        for replica_id, result in probes.items()
        if result.get("healthy") is not True
    )
    if unhealthy:
        raise EvidenceError(f"fleet HTTP health failed for replicas: {unhealthy}")
    completed = [result.get("probe_completed_timestamp") for result in probes.values()]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) <= captured
        for value in completed
    ):
        raise EvidenceError("fleet probes have invalid completion timestamps")
    max_probe_age = max(captured - float(value) for value in completed)
    if not 0 <= max_probe_age <= 600:
        raise EvidenceError("fleet live-probe evidence is already stale")

    counts = Counter(replica.serving_profile for replica in fleet.replicas)
    profile_counts = {name: counts[name] for name in control_plane.EXPECTED_FLEET_PROFILES}
    metrics = {
        "logical_replicas": len(fleet.replicas),
        "allocated_gpus": sum(replica.gpus_per_replica for replica in fleet.replicas),
        "healthy_replicas": len(probes),
        "unhealthy_replicas": 0,
        "profile_replicas": profile_counts,
        "revision_mismatches": 0,
        "missing_profiles": 0,
        "stale_registrations": 0,
        "max_heartbeat_age_seconds": max_probe_age,
        "captured_timestamp": captured,
        "model_contract_sha256": models.sha256,
        "fleet_contract_sha256": fleet.sha256,
        "client_capacity_evidence_id": client_capacity["evidence_id"],
        "client_partition": client_capacity_summary["partition"],
        "client_qos": client_capacity_summary["qos"],
        "client_cpu_limit": client_capacity_summary["cpu_limit"],
        "client_memory_limit_mib": client_capacity_summary["memory_limit_mib"],
        "client_max_submit_jobs": client_capacity_summary["max_submit_jobs"],
        "transport_censor_protocol_version": transport_binding[
            "transport_censor_protocol_version"
        ],
        "transport_censor_protocol_hash": transport_binding[
            "transport_censor_protocol_hash"
        ],
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
        "scheduler_safety_evidence_id": scheduler_evidence["evidence_id"],
        "scheduler_safety_policy_id": scheduler_policy["policy_id"],
        "scheduler_safety_policy_contract_id": scheduler_policy[
            "policy_contract_id"
        ],
        "scheduler_preemptible_partitions": scheduler_policy[
            "preemptible_partitions"
        ],
        "scheduler_partition_time_requirements_seconds": (
            partition_time_requirements
        ),
    }
    detail = []
    registration_refs: list[dict[str, str]] = []
    for replica in fleet.replicas:
        row, provenance = scheduler[replica.replica_id]
        entry, registry_path = records[replica.replica_id]
        detail.append(
            {
                "replica_id": replica.replica_id,
                "serving_profile": replica.serving_profile,
                "ledger_generation": allocation_by_replica[
                    replica.replica_id
                ].ledger_generation,
                "intent_token": allocation_by_replica[
                    replica.replica_id
                ].intent_token,
                "attempt_state": allocation_by_replica[
                    replica.replica_id
                ].attempt_state,
                "slurm_job_id": row.job_id,
                "comment": row.comment,
                "job_name": row.job_name,
                "partition": row.partition,
                "node": row.node,
                "host": entry.host,
                "port": entry.port,
                "spooled_provenance": asdict(provenance),
                "local_script_path": allocation_by_replica[
                    replica.replica_id
                ].sbatch_path,
                "local_script_sha256": allocation_by_replica[
                    replica.replica_id
                ].sbatch_sha256,
                "spooled_script_sha256": exact_spooled_proofs[
                    replica.replica_id
                ]["observed_sha256"],
                "spooled_script_proof": exact_spooled_proofs[
                    replica.replica_id
                ],
                "scontrol": exact_scheduler_proofs[replica.replica_id],
                "http": dict(probes[replica.replica_id]),
                "probe_transport_provenance": {
                    "transport_censor_protocol_version": transport_binding[
                        "transport_censor_protocol_version"
                    ],
                    "transport_censor_protocol_hash": transport_binding[
                        "transport_censor_protocol_hash"
                    ],
                    "transport_uncertainty_binding_sha256": (
                        transport_binding_sha256
                    ),
                    "source_tree_sha256": pins["source_tree_sha256"],
                },
                "registry_path": str(registry_path),
                "registry_sha256": _sha256(registry_path),
            }
        )
        registration_refs.append(
            _artifact(f"registry_{replica.replica_id}", registry_path)
        )
    name = "fleet_health_report"
    health_path = _artifact_path(output, name)
    raw_name = "raw_fleet_health_probe"
    raw_path = output.expanduser().resolve().parent / "artifacts" / f"{raw_name}.json"
    _write_json_atomic(
        raw_path,
        {
            "schema_version": 3,
            "kind": raw_name,
            "passed": True,
            "immutable_sha256": control["immutable_sha256"],
            "server_pool_root": str(pool_root),
            "rollout_generation": generation,
            "isolated_foreign_scheduler_job_ids": list(
                fleet_snapshot.isolated_foreign_job_ids
            ),
            "sealed_successful_handoff_terminal_job_ids": list(
                fleet_snapshot.sealed_successful_handoff_terminal_job_ids
            ),
            "ignored_terminal_job_ids": list(
                fleet_snapshot.ignored_terminal_job_ids
            ),
            "client_capacity_contract": client_capacity,
            "fleet_scheduler_safety_evidence": scheduler_evidence,
            "fleet_scheduler_safety_policy": scheduler_policy,
            "captured_timestamp": captured,
            "replicas": detail,
            "referenced_artifacts": registration_refs,
        },
    )
    _validate_raw_fleet_readiness(
        raw_path,
        control=control,
        state_dir=state_dir,
        pool_root=pool_root,
        expected_replica_ids={
            replica.replica_id for replica in fleet.replicas
        },
        scheduler_runner=scheduler_runner,
    )
    wrapper = _wrapper(
        name=name,
        immutable_sha256=str(control["immutable_sha256"]),
        metrics=metrics,
        references=[_artifact(raw_name, raw_path)],
        extra={"server_pool_root": str(pool_root)},
    )
    _write_json_atomic(health_path, wrapper)
    return _publish_outer(
        control=control,
        gate="fleet",
        metrics=metrics,
        artifacts=[
            _artifact(name, health_path),
            _artifact("fleet_contract", Path(str(fleet_binding["path"]))),
        ],
        output=output,
        now=captured,
        state_dir=state_dir,
    )


def _invoke_scheduler(
    runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] | None,
    argv: Sequence[str],
    *,
    timeout: float,
    description: str,
) -> subprocess.CompletedProcess[str]:
    try:
        process = (
            _run(argv, timeout=timeout)
            if runner is None
            else runner(list(argv), float(timeout))
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"{description} failed: {exc}") from exc
    if process.returncode != 0:
        raise EvidenceError(
            f"{description} failed rc={process.returncode}: "
            f"{process.stderr.strip()[:500]}"
        )
    return process


def _parse_scontrol_record(
    output: str,
    *,
    expected_job_id: str,
    description: str,
) -> tuple[dict[str, str], str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise EvidenceError(f"{description} must return exactly one Slurm record")
    try:
        tokens = shlex.split(lines[0], posix=True)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"cannot parse {description}: {exc}") from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise EvidenceError(f"{description} contains duplicate Slurm field {key!r}")
        fields[key] = value.strip('"')
    if fields.get("JobId") != str(expected_job_id):
        raise EvidenceError(f"{description} changed exact JobId")
    return fields, lines[0] + "\n"


def _slurm_duration_seconds(value: str) -> int:
    normalized = str(value).strip()
    if not normalized:
        raise EvidenceError("Slurm RunTime is absent")
    day = 0
    clock = normalized
    if "-" in normalized:
        day_text, clock = normalized.split("-", 1)
        if not day_text.isdigit():
            raise EvidenceError(f"invalid Slurm RunTime {value!r}")
        day = int(day_text)
    fields = clock.split(":")
    if not 1 <= len(fields) <= 3 or any(not field.isdigit() for field in fields):
        raise EvidenceError(f"invalid Slurm RunTime {value!r}")
    numbers = [int(field) for field in fields]
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    else:
        hours, minutes, seconds = 0, 0, numbers[0]
    if minutes >= 60 or seconds >= 60:
        raise EvidenceError(f"invalid Slurm RunTime {value!r}")
    return day * 86_400 + hours * 3_600 + minutes * 60 + seconds


def _validate_pending_capacity_scontrol(
    fields: Mapping[str, str],
    *,
    job_id: str,
    expected_comment: str,
    expected_job_name: str,
    script_path: Path,
) -> str:
    reason = fields.get("Reason")
    if (
        fields.get("JobId") != str(job_id)
        or fields.get("JobState") != "PENDING"
        or reason not in CAPACITY_TRANSIENT_REASONS
        or fields.get("Requeue") != "0"
        or fields.get("Comment") != expected_comment
        or fields.get("JobName") != expected_job_name
        or not fleet_tx.command_binds_sbatch(
            fields.get("Command", ""), str(script_path)
        )
    ):
        raise EvidenceError(
            f"pending fleet job {job_id} is not solely blocked by "
            "Priority/Resources or its exact Slurm identity drifted"
        )
    return str(reason)


def _validate_running_capacity_scontrol(
    fields: Mapping[str, str],
    *,
    job_id: str,
    expected_comment: str,
    expected_job_name: str,
    expected_node: str,
    script_path: Path,
) -> None:
    if (
        fields.get("JobId") != str(job_id)
        or fields.get("JobState") != "RUNNING"
        or fields.get("Requeue") != "0"
        or fields.get("Comment") != expected_comment
        or fields.get("JobName") != expected_job_name
        or fields.get("NodeList") != expected_node
        or not fleet_tx.command_binds_sbatch(
            fields.get("Command", ""), str(script_path)
        )
    ):
        raise EvidenceError(
            f"running fleet job {job_id} exact Slurm identity/Requeue=0 drifted"
        )


def _exact_capacity_job_state(
    *,
    row: keepalive.FleetQueueRow,
    script_path: Path,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
    include_raw_output: bool = False,
) -> tuple[dict[str, Any], str | None]:
    command = ["scontrol", "show", "job", "-o", str(row.job_id)]
    process = _invoke_scheduler(
        scheduler_runner,
        command,
        timeout=30.0,
        description=f"exact active fleet job {row.job_id} query",
    )
    fields, normalized_output = _parse_scontrol_record(
        process.stdout,
        expected_job_id=str(row.job_id),
        description=f"exact active fleet job {row.job_id} query",
    )
    state = row.state.upper()
    reason: str | None = None
    if state == "PENDING":
        reason = _validate_pending_capacity_scontrol(
            fields,
            job_id=str(row.job_id),
            expected_comment=row.comment,
            expected_job_name=row.job_name,
            script_path=script_path,
        )
    elif state == "RUNNING":
        _validate_running_capacity_scontrol(
            fields,
            job_id=str(row.job_id),
            expected_comment=row.comment,
            expected_job_name=row.job_name,
            expected_node=row.node,
            script_path=script_path,
        )
    else:
        raise EvidenceError(
            f"exact active fleet job has forbidden state {state}: {row.job_id}"
        )
    proof: dict[str, Any] = {
            "argv": command,
            "job_id": str(row.job_id),
            "job_state": state,
            "reason": reason,
            "comment": row.comment,
            "job_name": row.job_name,
            "node": None if state == "PENDING" else row.node,
            "command": str(script_path),
            "effective_requeue": 0,
            "raw_output_sha256": hashlib.sha256(
                normalized_output.encode("utf-8")
            ).hexdigest(),
        }
    if include_raw_output:
        proof["raw_output"] = normalized_output
    return proof, reason


def _exact_spooled_script_proof(
    *,
    job_id: str,
    script_path: Path,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
) -> dict[str, Any]:
    command = ["scontrol", "write", "batch_script", str(job_id), "-"]
    process = _invoke_scheduler(
        scheduler_runner,
        command,
        timeout=30.0,
        description=f"exact spooled fleet script {job_id} query",
    )
    try:
        local = script_path.read_bytes()
        observed = process.stdout.encode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(
            f"cannot compare exact spooled fleet script {job_id}: {exc}"
        ) from exc
    if observed != local:
        raise EvidenceError(
            f"Slurm spooled script differs from immutable local bytes for job {job_id}"
        )
    local_sha256 = hashlib.sha256(local).hexdigest()
    observed_sha256 = hashlib.sha256(observed).hexdigest()
    return {
        "argv": command,
        "local_path": str(script_path),
        "local_sha256": local_sha256,
        "observed_sha256": observed_sha256,
        "observed_bytes": len(observed),
        "exact_match": True,
    }


def _validate_raw_fleet_readiness(
    path: Path,
    *,
    control: Mapping[str, Any],
    state_dir: Path,
    pool_root: Path,
    expected_replica_ids: set[str],
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
) -> dict[str, Any]:
    """Revalidate every effective Slurm/spool/probe fact before publication."""

    payload = _read_json(
        _require_read_only_regular(path, description="raw fleet readiness")
    )
    rows = payload.get("replicas")
    references = payload.get("referenced_artifacts")
    required = {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "server_pool_root",
        "rollout_generation",
        "isolated_foreign_scheduler_job_ids",
        "sealed_successful_handoff_terminal_job_ids",
        "ignored_terminal_job_ids",
        "client_capacity_contract",
        "fleet_scheduler_safety_evidence",
        "fleet_scheduler_safety_policy",
        "captured_timestamp",
        "replicas",
        "referenced_artifacts",
    }
    row_fields = {
        "replica_id",
        "serving_profile",
        "ledger_generation",
        "intent_token",
        "attempt_state",
        "slurm_job_id",
        "comment",
        "job_name",
        "partition",
        "node",
        "host",
        "port",
        "spooled_provenance",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_proof",
        "scontrol",
        "http",
        "probe_transport_provenance",
        "registry_path",
        "registry_sha256",
    }
    scontrol_fields = {
        "argv",
        "job_id",
        "job_state",
        "reason",
        "comment",
        "job_name",
        "node",
        "command",
        "effective_requeue",
        "raw_output",
        "raw_output_sha256",
    }
    spool_fields = {
        "argv",
        "local_path",
        "local_sha256",
        "observed_sha256",
        "observed_bytes",
        "exact_match",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 3
        or payload.get("kind") != "raw_fleet_health_probe"
        or payload.get("passed") is not True
        or payload.get("immutable_sha256") != control["immutable_sha256"]
        or payload.get("server_pool_root") != str(pool_root)
        or not isinstance(payload.get("rollout_generation"), int)
        or isinstance(payload.get("rollout_generation"), bool)
        or payload["rollout_generation"] < 1
        or payload.get("ignored_terminal_job_ids") != []
        or not isinstance(payload.get("captured_timestamp"), (int, float))
        or isinstance(payload.get("captured_timestamp"), bool)
        or not math.isfinite(float(payload["captured_timestamp"]))
        or float(payload["captured_timestamp"]) <= 0
        or not isinstance(rows, list)
        or len(rows) != len(expected_replica_ids)
        or not isinstance(references, list)
        or len(references) != len(expected_replica_ids)
    ):
        raise EvidenceError("raw fleet readiness envelope is invalid")
    pins = control["immutable"]
    fleet_binding = control_plane.effective_fleet_contract_binding(
        control, verify_files=True
    )
    models = load_model_contracts(
        pins["model_contract_path"],
        expected_sha256=pins["model_contract_sha256"],
    )
    fleet = load_fleet_contract(
        fleet_binding["path"],
        model_contracts=models,
        expected_sha256=fleet_binding["sha256"],
        allow_capacity_layout=True,
    )
    fleet.verify_pool_root(pool_root)
    if int(fleet_binding["capacity_generation"]) == 1:
        try:
            initial_capacity = (
                control_plane.load_effective_protected_capacity_contract(
                    control, verify_files=True
                )
            )
        except (
            control_plane.ControlError,
            control_plane.ImmutablePinError,
            protected_capacity.ProtectedCapacityError,
        ) as exc:
            raise EvidenceError(
                f"raw prequalification capacity baseline is invalid: {exc}"
            ) from exc
        if (
            len(fleet.replicas) != 22
            or sum(replica.gpus_per_replica for replica in fleet.replicas)
            != 24
            or initial_capacity.effective_active_logical_replicas != 22
            or initial_capacity.effective_active_gpus != 24
            or initial_capacity.additive_reserved_logical_replicas != 0
            or initial_capacity.additive_reserved_gpus != 0
            or initial_capacity.retained_warm_turnover_job_elements != 3
            or initial_capacity.retained_warm_turnover_gpus != 4
            or initial_capacity.attested_total_gpus != 28
            or initial_capacity.job_element_accounting.get(
                "controller_monitor_other_held_job_elements"
            )
            != 39
            or initial_capacity.effective_fleet_contract_sha256
            != initial_capacity.base_fleet_contract_sha256
        ):
            raise EvidenceError(
                "raw prequalification fleet evidence is not the exact "
                "22-logical/24-active + 3-warm/4-GPU, 28-GPU-attested baseline"
            )
    try:
        transport_binding = (
            scheduler_safety.validate_transport_uncertainty_binding(
                pins["transport_uncertainty_binding"]
            )
        )
        transport_binding_sha256 = (
            scheduler_safety.transport_uncertainty_binding_sha256(
                transport_binding
            )
        )
        if (
            pins.get("transport_uncertainty_binding_sha256")
            != transport_binding_sha256
        ):
            raise scheduler_safety.SchedulerSafetyError(
                "immutable transport binding digest drifted"
            )
        partition_time_requirements = (
            scheduler_safety.fleet_partition_time_requirements(
                fleet.replicas
            )
        )
        stored_scheduler_evidence = payload.get(
            "fleet_scheduler_safety_evidence"
        )
        if not isinstance(stored_scheduler_evidence, Mapping):
            raise scheduler_safety.SchedulerSafetyError(
                "stored fleet scheduler evidence is absent"
            )
        stored_scheduler_policy = (
            scheduler_safety.validate_scheduler_safety_evidence(
                stored_scheduler_evidence,
                expected_partitions=list(partition_time_requirements),
                required_time_limits_seconds=partition_time_requirements,
            )
        )
        if (
            payload.get("fleet_scheduler_safety_policy")
            != stored_scheduler_policy
        ):
            raise scheduler_safety.SchedulerSafetyError(
                "stored fleet scheduler policy differs from raw evidence"
            )
        fresh_scheduler_evidence = (
            scheduler_safety.capture_scheduler_safety_evidence(
                list(partition_time_requirements),
                runner=_transaction_runner(scheduler_runner),
            )
        )
        fresh_scheduler_policy = (
            scheduler_safety.validate_scheduler_safety_evidence(
                fresh_scheduler_evidence,
                expected_partitions=list(partition_time_requirements),
                required_time_limits_seconds=partition_time_requirements,
            )
        )
        if (
            fresh_scheduler_policy["policy_contract_id"]
            != stored_scheduler_policy["policy_contract_id"]
        ):
            raise scheduler_safety.SchedulerSafetyError(
                "fleet scheduler policy contract differs from live truth"
            )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise EvidenceError(
            f"raw fleet scheduler-safety proof failed: {exc}"
        ) from exc
    stored_client_capacity = payload.get("client_capacity_contract")
    current_client_capacity = (
        control_plane.client_capacity_contract_from_state(state_dir)
    )
    try:
        protected_contract = (
            control_plane.load_effective_protected_capacity_contract(
                control, verify_files=True
            )
        )
        _validate_capacity_wave_admission_fence(
            control,
            fleet_binding=fleet_binding,
            contract=protected_contract,
        )
        if not isinstance(stored_client_capacity, Mapping):
            raise protected_capacity.ProtectedCapacityError(
                "stored client-capacity contract is absent"
            )
        stored_client_summary = (
            protected_capacity.validate_live_client_capacity_evidence(
                protected_contract,
                stored_client_capacity,
                partition=str(current_client_capacity["partition"]),
                qos=str(current_client_capacity["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
            )
        )
        transaction_runner = (
            None
            if scheduler_runner is None
            else lambda argv, **kwargs: scheduler_runner(
                list(argv), float(kwargs.get("timeout", 15.0))
            )
        )
        fresh_fleet_snapshot = keepalive.reconcile_fleet_read_only(
            str(pool_root),
            fleet,
            current_generation=int(payload["rollout_generation"]),
            scheduler_runner=transaction_runner,
        )
        fresh_now = time.time()
        fresh_trusted_provenance = (
            control_plane.reconcile_trusted_scientific_job_provenance(
                state_dir,
                fleet_bindings=keepalive.trusted_scientific_fleet_bindings(
                    fresh_fleet_snapshot
                ),
                fleet_contract_sha256=fleet.sha256,
                fleet_generation=int(payload["rollout_generation"]),
                scheduler_reader=(
                    None
                    if scheduler_runner is None
                    else lambda: control_plane.query_scheduler(
                        runner=lambda argv: scheduler_runner(
                            list(argv), 15.0
                        ),
                        now=fresh_now,
                    )
                ),
                now=fresh_now,
                allow_exact_cell_quiescence=True,
            )
        )
        fresh_client_capacity = (
            protected_capacity.capture_live_client_capacity(
                protected_contract,
                partition=str(current_client_capacity["partition"]),
                qos=str(current_client_capacity["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
                trusted_scientific_job_provenance=(
                    fresh_trusted_provenance
                ),
                runner=_transaction_runner(scheduler_runner)
            )
        )
        fresh_client_summary = (
            protected_capacity.validate_live_client_capacity_evidence(
                protected_contract,
                fresh_client_capacity,
                partition=str(current_client_capacity["partition"]),
                qos=str(current_client_capacity["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
            )
        )
    except (
        control_plane.ControlError,
        keepalive.FleetContractError,
        scheduler_safety.SchedulerSafetyError,
        protected_capacity.ProtectedCapacityError,
    ) as exc:
        raise EvidenceError(
            f"raw fleet client partition/QOS proof failed: {exc}"
        ) from exc
    if {
        key: value
        for key, value in stored_client_summary.items()
        if key not in {"evidence_id", "scheduler_policy_id"}
    } != {
        key: value
        for key, value in fresh_client_summary.items()
        if key not in {"evidence_id", "scheduler_policy_id"}
    }:
        raise EvidenceError(
            "raw fleet client partition/QOS proof differs from live scheduler"
        )
    expected_by_id = {
        replica.replica_id: replica for replica in fleet.replicas
    }
    if set(expected_by_id) != expected_replica_ids:
        raise EvidenceError(
            "raw fleet readiness replica set differs from the immutable fleet"
        )
    generation = int(payload["rollout_generation"])
    expected_probe_transport = {
        "transport_censor_protocol_version": transport_binding[
            "transport_censor_protocol_version"
        ],
        "transport_censor_protocol_hash": transport_binding[
            "transport_censor_protocol_hash"
        ],
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
        "source_tree_sha256": pins["source_tree_sha256"],
    }
    try:
        fresh_snapshot = keepalive.reconcile_fleet_read_only(
            str(pool_root),
            fleet,
            current_generation=generation,
            scheduler_runner=_transaction_runner(scheduler_runner),
            scheduler_now=time.time(),
        )
    except (keepalive.FleetContractError, OSError) as exc:
        raise EvidenceError(
            f"independent raw fleet reconciliation failed: {exc}"
        ) from exc
    fresh_by_id = {
        allocation.replica_id: allocation
        for allocation in fresh_snapshot.allocations
    }
    if (
        fresh_snapshot.current_generation != generation
        or set(fresh_by_id) != expected_replica_ids
        or tuple(fresh_snapshot.ignored_terminal_job_ids) != ()
        or list(fresh_snapshot.isolated_foreign_job_ids)
        != payload["isolated_foreign_scheduler_job_ids"]
        or list(fresh_snapshot.sealed_successful_handoff_terminal_job_ids)
        != payload["sealed_successful_handoff_terminal_job_ids"]
    ):
        raise EvidenceError(
            "raw fleet readiness differs from independent scheduler/ledger truth"
        )
    observed_ids = [
        row.get("replica_id") for row in rows if isinstance(row, Mapping)
    ]
    active_job_ids = {
        str(row.get("slurm_job_id"))
        for row in rows
        if isinstance(row, Mapping)
    }
    for field in (
        "isolated_foreign_scheduler_job_ids",
        "sealed_successful_handoff_terminal_job_ids",
    ):
        values = payload.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.isdigit() for value in values)
            or values != sorted(set(values), key=int)
            or active_job_ids.intersection(values)
        ):
            raise EvidenceError(f"raw fleet readiness {field} is invalid")
    if set(payload["isolated_foreign_scheduler_job_ids"]).intersection(
        payload["sealed_successful_handoff_terminal_job_ids"]
    ):
        raise EvidenceError(
            "raw fleet readiness terminal/canary scheduler identities overlap"
        )
    if (
        len(observed_ids) != len(expected_replica_ids)
        or any(not isinstance(value, str) for value in observed_ids)
        or set(observed_ids) != expected_replica_ids
    ):
        raise EvidenceError("raw fleet readiness replica identity drifted")
    rows_by_id = {
        str(row["replica_id"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_references = {
        (
            f"registry_{replica_id}",
            str(Path(str(row["registry_path"])).expanduser().resolve()),
            str(row["registry_sha256"]),
        )
        for replica_id, row in rows_by_id.items()
    }
    observed_references = {
        (
            str(reference.get("name")),
            str(reference.get("path")),
            str(reference.get("sha256")),
        )
        for reference in references
        if isinstance(reference, Mapping)
    }
    if (
        len(observed_references) != len(references)
        or observed_references != expected_references
    ):
        raise EvidenceError(
            "raw fleet readiness registry references are not exact"
        )
    effective_pins = dict(pins)
    effective_pins["fleet_contract_path"] = str(fleet_binding["path"])
    effective_pins["fleet_contract_sha256"] = str(fleet_binding["sha256"])
    registry_records = _verify_registry(
        pool_root=pool_root,
        fleet=fleet,
        models=models,
        pins=effective_pins,
        scheduler={
            replica_id: (
                allocation.row,
                allocation.spooled_provenance,
            )
            for replica_id, allocation in fresh_by_id.items()
        },
    )
    from agents_scaling.serving.profiles import get_serving_profile

    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise EvidenceError("raw fleet readiness row fields drifted")
        job_id = row["slurm_job_id"]
        scontrol = row["scontrol"]
        spool = row["spooled_script_proof"]
        http = row["http"]
        parsed_comment = fleet_tx.parse_intent_comment(str(row.get("comment", "")))
        if not all(
            isinstance(value, Mapping)
            for value in (
                scontrol,
                spool,
                http,
                row.get("spooled_provenance"),
            )
        ):
            raise EvidenceError(
                f"raw fleet nested proof is malformed for {row.get('replica_id')}"
            )
        replica_id = str(row.get("replica_id", ""))
        replica = expected_by_id.get(replica_id)
        allocation = fresh_by_id.get(replica_id)
        if replica is None or allocation is None:
            raise EvidenceError(
                f"raw fleet readiness has an unknown replica {replica_id!r}"
            )
        script_path = _require_read_only_regular(
            Path(str(row.get("local_script_path", ""))),
            description=f"raw fleet local sbatch for {replica_id}",
        )
        registry_entry, registry_path = registry_records[replica_id]
        try:
            registry_payload = _read_json(registry_path)
            stored_provenance = keepalive.SpooledServingProvenance(
                **dict(row.get("spooled_provenance", {}))
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceError(
                f"raw fleet provenance is malformed for {replica_id}: {exc}"
            ) from exc
        expected_provenance = allocation.spooled_provenance
        identity = models.for_size(replica.model_size)
        profile = get_serving_profile(replica.serving_profile)
        raw_output = (
            scontrol.get("raw_output")
            if isinstance(scontrol, Mapping)
            else None
        )
        if not isinstance(raw_output, str):
            raise EvidenceError(
                f"raw fleet scheduler output is missing for {replica_id}"
            )
        parsed_scontrol, normalized_scontrol = _parse_scontrol_record(
            raw_output,
            expected_job_id=str(row.get("slurm_job_id", "")),
            description=f"persisted raw fleet scheduler proof for {replica_id}",
        )
        _validate_running_capacity_scontrol(
            parsed_scontrol,
            job_id=str(row.get("slurm_job_id", "")),
            expected_comment=str(row.get("comment", "")),
            expected_job_name=str(row.get("job_name", "")),
            expected_node=str(row.get("node", "")),
            script_path=script_path,
        )
        fresh_scontrol, fresh_reason = _exact_capacity_job_state(
            row=allocation.row,
            script_path=script_path,
            scheduler_runner=scheduler_runner,
        )
        fresh_spool = _exact_spooled_script_proof(
            job_id=str(row.get("slurm_job_id", "")),
            script_path=script_path,
            scheduler_runner=scheduler_runner,
        )
        stable_scontrol_fields = {
            key: value
            for key, value in scontrol.items()
            if key not in {"raw_output", "raw_output_sha256"}
        }
        fresh_stable_scontrol = {
            key: value
            for key, value in fresh_scontrol.items()
            if key != "raw_output_sha256"
        }
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or row.get("ledger_generation") != payload["rollout_generation"]
            or not isinstance(row.get("intent_token"), str)
            or row.get("attempt_state") != "committed"
            or parsed_comment is None
            or parsed_comment.get("pool") != "schema5-v1"
            or parsed_comment.get("profile") != row.get("serving_profile")
            or parsed_comment.get("replica") != row.get("replica_id")
            or parsed_comment.get("generation")
            != str(payload["rollout_generation"])
            or parsed_comment.get("intent") != row.get("intent_token")
            or parsed_comment.get("fleet") != fleet.sha256
            or allocation.ledger_generation != generation
            or allocation.intent_token != row.get("intent_token")
            or allocation.attempt_state != "committed"
            or allocation.row.job_id != job_id
            or allocation.row.comment != row.get("comment")
            or allocation.row.job_name != row.get("job_name")
            or allocation.row.partition != row.get("partition")
            or allocation.row.node != row.get("node")
            or allocation.sbatch_path != str(script_path)
            or allocation.sbatch_sha256 != row.get("local_script_sha256")
            or script_path.parent
            != (
                fleet_tx.state_directory(pool_root)
                / "sbatch"
                / f"g{generation:06d}"
            ).resolve()
            or script_path.name
            != (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", replica.replica_id)
                + f".{row.get('intent_token')}.sbatch"
            )
            or asdict(stored_provenance) != asdict(expected_provenance)
            or stored_provenance.run_root != str(pool_root)
            or stored_provenance.server_pool_id != fleet.fleet_id
            or stored_provenance.replica_id != replica.replica_id
            or stored_provenance.replica_index != replica.replica_index
            or stored_provenance.release_id != pins["release_id"]
            or stored_provenance.environment_hash
            != pins["serving_environment_sha256"]
            or stored_provenance.model_revision != identity.model_revision
            or stored_provenance.tokenizer_id != identity.tokenizer_id
            or stored_provenance.tokenizer_revision
            != identity.tokenizer_revision
            or stored_provenance.model_contract_sha256
            != pins["model_contract_sha256"]
            or stored_provenance.fleet_contract_sha256 != fleet.sha256
            or stored_provenance.release_fleet_contract_sha256
            != pins["fleet_contract_sha256"]
            or stored_provenance.capacity_generation
            != fleet_binding["capacity_generation"]
            or stored_provenance.rollout_generation != generation
            or stored_provenance.spooled_script_sha256
            != row.get("spooled_script_sha256")
            or not isinstance(scontrol, Mapping)
            or set(scontrol) != scontrol_fields
            or scontrol.get("argv")
            != ["scontrol", "show", "job", "-o", job_id]
            or scontrol.get("job_id") != job_id
            or scontrol.get("job_state") != "RUNNING"
            or scontrol.get("reason") is not None
            or scontrol.get("comment") != row["comment"]
            or scontrol.get("job_name") != row["job_name"]
            or scontrol.get("node") != row["node"]
            or scontrol.get("command") != row["local_script_path"]
            or scontrol.get("effective_requeue") != 0
            or normalized_scontrol != raw_output
            or scontrol.get("raw_output_sha256")
            != hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            or fresh_reason is not None
            or stable_scontrol_fields != fresh_stable_scontrol
            or not isinstance(spool, Mapping)
            or set(spool) != spool_fields
            or spool.get("argv")
            != ["scontrol", "write", "batch_script", job_id, "-"]
            or spool.get("local_path") != row["local_script_path"]
            or spool.get("local_sha256") != row["local_script_sha256"]
            or spool.get("observed_sha256") != row["spooled_script_sha256"]
            or row["spooled_script_sha256"] != row["local_script_sha256"]
            or not isinstance(spool.get("observed_bytes"), int)
            or isinstance(spool.get("observed_bytes"), bool)
            or spool["observed_bytes"] < 1
            or spool.get("exact_match") is not True
            or dict(spool) != fresh_spool
            or _sha256(script_path) != row.get("local_script_sha256")
            or script_path.stat().st_size != spool.get("observed_bytes")
            or registry_path
            != Path(str(row.get("registry_path", ""))).expanduser().resolve()
            or _sha256(registry_path) != row.get("registry_sha256")
            or registry_payload != asdict(registry_entry)
            or registry_entry.replica_id != replica.replica_id
            or registry_entry.replica_index != replica.replica_index
            or registry_entry.release_fleet_contract_sha256
            != pins["fleet_contract_sha256"]
            or registry_entry.fleet_contract_sha256 != fleet.sha256
            or registry_entry.capacity_generation
            != fleet_binding["capacity_generation"]
            or registry_entry.rollout_generation != generation
            or registry_entry.serving_profile != replica.serving_profile
            or registry_entry.max_model_len
            != profile.max_model_len
            or registry_entry.tp_size != profile.tp_size
            or row.get("host") != registry_entry.host
            or row.get("port") != registry_entry.port
            or not isinstance(http, Mapping)
            or set(http)
            != {
                "health_status",
                "models_status",
                "model_ids",
                "expected_model",
                "probe_started_timestamp",
                "probe_completed_timestamp",
                "healthy",
            }
            or http.get("healthy") is not True
            or http.get("health_status") != 200
            or http.get("models_status") != 200
            or http.get("expected_model") != profile.served_model_name
            or http.get("expected_model") not in http.get("model_ids", [])
            or not isinstance(http.get("probe_started_timestamp"), (int, float))
            or isinstance(http.get("probe_started_timestamp"), bool)
            or not isinstance(
                http.get("probe_completed_timestamp"), (int, float)
            )
            or isinstance(http.get("probe_completed_timestamp"), bool)
            or not math.isfinite(float(http["probe_started_timestamp"]))
            or not math.isfinite(float(http["probe_completed_timestamp"]))
            or float(http["probe_started_timestamp"])
            > float(http["probe_completed_timestamp"])
            or float(http["probe_completed_timestamp"])
            > float(payload["captured_timestamp"])
            or row.get("probe_transport_provenance")
            != expected_probe_transport
        ):
            raise EvidenceError(
                f"raw fleet readiness proof drifted for {row.get('replica_id')}"
            )
    return payload


def _transaction_runner(
    runner: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] | None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def invoke(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _invoke_scheduler(
            runner,
            argv,
            timeout=float(kwargs.get("timeout", 15.0)),
            description=f"{argv[0]} scheduler query",
        )

    return invoke


def _verify_capacity_allocation_provenance(
    provenance: keepalive.SpooledServingProvenance,
    *,
    replica: Any,
    pool_root: Path,
    pins: Mapping[str, Any],
    models: FrozenModelContracts,
    fleet_sha256: str,
) -> None:
    identity = models.for_size(replica.model_size)
    if (
        provenance.run_root != str(pool_root)
        or provenance.server_pool_id != replica.pool_id
        or provenance.replica_id != replica.replica_id
        or provenance.replica_index != replica.replica_index
        or provenance.release_id != pins["release_id"]
        or provenance.environment_hash != pins["serving_environment_sha256"]
        or provenance.model_revision != identity.model_revision
        or provenance.tokenizer_id != identity.tokenizer_id
        or provenance.tokenizer_revision != identity.tokenizer_revision
        or provenance.model_contract_sha256 != pins["model_contract_sha256"]
        or provenance.fleet_contract_sha256 != fleet_sha256
    ):
        raise EvidenceError(
            f"spooled serving provenance drift for {replica.replica_id}"
        )


def _collect_fleet_capacity_transient(
    control: Mapping[str, Any],
    *,
    state_dir: Path,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
    probe: Callable[[ServerEntry, str, float], Mapping[str, Any]],
    probe_timeout: float,
    now: Callable[[], float],
) -> dict[str, Any]:
    """Prove the one narrow fleet-capacity state that may exit readiness with 75."""

    if not 0 < probe_timeout <= 60:
        raise EvidenceError("capacity-transient probe timeout must be in (0, 60]")
    runtime_environment = _paused_next_generation_runtime_environment(
        control, state_dir=state_dir
    )
    generation = int(runtime_environment["ASYS_ROLLOUT_GENERATION"])
    pins = control["immutable"]
    fleet_binding = control_plane.effective_fleet_contract_binding(
        control, verify_files=True
    )
    if (
        fleet_binding["logical_replicas"] != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or fleet_binding["allocated_gpus"] != CAPACITY_TRANSIENT_EXPECTED_GPUS
    ):
        raise EvidenceError(
            "capacity-transient repair is restricted to the exact 22-replica/"
            "24-GPU production fleet"
        )
    effective_pins = dict(pins)
    effective_pins.update(
        {
            "fleet_contract_path": fleet_binding["path"],
            "fleet_contract_sha256": fleet_binding["sha256"],
        }
    )
    pool_root = Path(str(pins["server_pool_root"])).expanduser().resolve()
    if server_pool_id(pool_root) != "schema5-v1":
        raise EvidenceError(f"server pool root has the wrong identity: {pool_root}")
    models = load_model_contracts(
        pins["model_contract_path"], expected_sha256=pins["model_contract_sha256"]
    )
    fleet = load_fleet_contract(
        fleet_binding["path"],
        model_contracts=models,
        expected_sha256=fleet_binding["sha256"],
        allow_capacity_layout=True,
    )
    fleet.verify_pool_root(pool_root)
    if (
        len(fleet.replicas) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or sum(replica.gpus_per_replica for replica in fleet.replicas)
        != CAPACITY_TRANSIENT_EXPECTED_GPUS
    ):
        raise EvidenceError("capacity-transient fleet cardinality drifted")

    invoke = _transaction_runner(scheduler_runner)
    replica_ids = [replica.replica_id for replica in fleet.replicas]
    replica_by_id = {replica.replica_id: replica for replica in fleet.replicas}
    try:
        with fleet_tx.read_transaction_lock(pool_root) as directory:
            ledgers = fleet_tx.read_generation_ledgers(
                directory,
                pool_root=pool_root,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                current_generation=generation,
                replica_ids=replica_ids,
            )
            scheduler_snapshot = fleet_tx.query_scheduler(
                runner=invoke, now=float(now())
            )
            production_rows, isolated_foreign_job_ids = (
                _scope_capacity_scheduler_rows(
                    scheduler_snapshot.rows,
                    fleet=fleet,
                )
            )
            reconciled = fleet_tx.reconcile_scheduler_rows(
                production_rows,
                ledgers,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                replica_profiles={
                    replica.replica_id: replica.serving_profile
                    for replica in fleet.replicas
                },
                replica_job_names={
                    replica.replica_id: replica.scheduler_job_name
                    for replica in fleet.replicas
                },
                replica_qos={
                    replica.replica_id: replica.qos
                    for replica in fleet.replicas
                },
            )
            physical = reconciled.active_allocations
            logical = reconciled.logical_allocations
            if len(physical) != len(logical):
                raise EvidenceError("capacity-transient fleet has a handoff overlap")
            active_ids = [allocation.replica_id for allocation in logical]
            missing = sorted(set(replica_ids) - set(active_ids))
            unexpected = sorted(set(active_ids) - set(replica_ids))
            if (
                missing
                or unexpected
                or len(active_ids) != len(set(active_ids))
                or len(active_ids) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
            ):
                raise EvidenceError(
                    "capacity-transient fleet is incomplete/duplicated: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            current_ledger = next(
                (
                    ledger
                    for ledger in ledgers
                    if ledger["rollout_generation"] == generation
                ),
                None,
            )
            if current_ledger is None:
                raise EvidenceError("current exact fleet generation ledger is absent")
            current_pointer_path = (
                directory / fleet_tx.CURRENT_FILENAME
            ).resolve()
            generation_ledger_path = fleet_tx.ledger_path(
                directory, generation
            ).resolve()
            if (
                current_pointer_path.is_symlink()
                or not current_pointer_path.is_file()
                or generation_ledger_path.is_symlink()
                or not generation_ledger_path.is_file()
            ):
                raise EvidenceError(
                    "current fleet pointer/ledger source is unsafe"
                )
            current_pointer_sha256 = _sha256(current_pointer_path)
            generation_ledger_sha256 = _sha256(generation_ledger_path)
            allowed_terminal_ids = _successful_capacity_handoff_history(
                current_ledger=current_ledger,
                logical_allocations=logical,
            )
            ignored_terminal_ids = set(reconciled.ignored_terminal_job_ids)
            if ignored_terminal_ids:
                # Successfully retired handoff predecessors are ledger-mapped
                # allocations, never "ignored" rows.  Any ignored row that remains
                # after exact production scoping is therefore an unmapped collision.
                raise EvidenceError(
                    "capacity-transient production scheduler truth contains "
                    "unmapped terminal rows: "
                    f"{sorted(ignored_terminal_ids, key=int)}"
                )
            current_terminal: list[str] = []
            for row in production_rows:
                parsed = fleet_tx.parse_intent_comment(row.comment)
                if (
                    parsed is not None
                    and parsed["pool"] == fleet.fleet_id
                    and parsed["fleet"] == fleet.sha256
                    and parsed["generation"] == str(generation)
                    and fleet_tx.terminal_state(row.state)
                    and (
                        row.job_id not in allowed_terminal_ids
                        or row.state.upper().split("+", 1)[0].split(" ", 1)[0]
                        not in {"CANCELLED", "COMPLETED"}
                    )
                ):
                    current_terminal.append(row.job_id)
            if current_terminal:
                raise EvidenceError(
                    "capacity-transient generation contains unsealed/failed terminal "
                    "scheduler rows: "
                    f"{sorted(current_terminal, key=int)}"
                )

            verified: dict[
                str,
                tuple[
                    Any,
                    keepalive.SpooledServingProvenance,
                    Path,
                    dict[str, Any],
                    dict[str, Any],
                ],
            ] = {}
            for allocation in logical:
                replica = replica_by_id[allocation.replica_id]
                attempt = dict(allocation.attempt)
                row = allocation.row
                if (
                    allocation.ledger_generation != generation
                    or attempt.get("state") != "committed"
                    or attempt.get("job_id") != str(row.job_id)
                    or attempt.get("committed_at") is None
                    or attempt.get("last_error") is not None
                ):
                    raise EvidenceError(
                        f"fleet intent is not a clean current-generation commit: "
                        f"{replica.replica_id}"
                    )
                script_path = _require_read_only_regular(
                    Path(str(attempt["sbatch_path"])),
                    description=f"fleet script for {replica.replica_id}",
                )
                script = script_path.read_text(encoding="utf-8")
                keepalive._validate_fleet_script_contract(  # noqa: SLF001
                    script,
                    replica=replica,
                    rollout_generation=generation,
                    fleet_sha256=fleet.sha256,
                    standby=attempt.get("launch_kind") == "handoff",
                )
                provenance = keepalive._validate_scheduler_row(  # noqa: SLF001
                    row,
                    replica=replica,
                    attempt=attempt,
                    fleet=fleet,
                    run_root=str(pool_root),
                    expected_script=script,
                    scheduler_runner=invoke,
                )
                if provenance is None:
                    raise EvidenceError(
                        f"active fleet job lacks immutable spooled provenance: "
                        f"{row.job_id}"
                    )
                _verify_capacity_allocation_provenance(
                    provenance,
                    replica=replica,
                    pool_root=pool_root,
                    pins=effective_pins,
                    models=models,
                    fleet_sha256=fleet.sha256,
                )
                scontrol_proof, _reason = _exact_capacity_job_state(
                    row=row,
                    script_path=script_path,
                    scheduler_runner=scheduler_runner,
                )
                spooled_script_proof = _exact_spooled_script_proof(
                    job_id=str(row.job_id),
                    script_path=script_path,
                    scheduler_runner=scheduler_runner,
                )
                verified[replica.replica_id] = (
                    allocation,
                    provenance,
                    script_path,
                    scontrol_proof,
                    spooled_script_proof,
                )
    except (
        fleet_tx.FleetTransactionError,
        keepalive.FleetContractError,
        OSError,
        UnicodeError,
    ) as exc:
        raise EvidenceError(
            f"capacity-transient fleet reconciliation failed: {exc}"
        ) from exc

    running_ids: set[str] = set()
    pending_ids: set[str] = set()
    pending_detail: list[dict[str, Any]] = []
    for replica in fleet.replicas:
        (
            allocation,
            provenance,
            script_path,
            scontrol_proof,
            spooled_script_proof,
        ) = verified[replica.replica_id]
        row = allocation.row
        state = row.state.upper()
        if state == "RUNNING":
            if (
                not row.node
                or row.node in {"(null)", "N/A", "None", "None assigned"}
                or not isinstance(allocation.health, Mapping)
                or allocation.health.get("job_id") != str(row.job_id)
                or allocation.health.get("observer_generation") != generation
                or allocation.health.get("cancel_state") is not None
                or allocation.health.get("cancel_error") is not None
                or allocation.health.get("consecutive_failures") != 0
            ):
                raise EvidenceError(
                    f"running fleet allocation is unhealthy/hung: {replica.replica_id}"
                )
            running_ids.add(replica.replica_id)
            continue
        if state != "PENDING":
            raise EvidenceError(
                f"capacity-transient allocation has forbidden state {state}: "
                f"{replica.replica_id}"
            )
        if (
            row.node
            and row.node not in {"(null)", "N/A", "None", "None assigned"}
        ):
            raise EvidenceError(
                f"pending fleet allocation unexpectedly owns a node: {row.job_id}"
            )
        if allocation.health is not None:
            raise EvidenceError(
                f"pending fleet allocation has stale health state: {replica.replica_id}"
            )
        reason = scontrol_proof["reason"]
        pending_ids.add(replica.replica_id)
        pending_detail.append(
            {
                "replica_id": replica.replica_id,
                "serving_profile": replica.serving_profile,
                "allocated_gpus": replica.gpus_per_replica,
                "job_id": str(row.job_id),
                "state": "PENDING",
                "reason": reason,
                "partition": row.partition,
                "comment": row.comment,
                "intent_token": allocation.attempt["intent_token"],
                "ledger_generation": allocation.ledger_generation,
                "local_script_path": str(script_path),
                "local_script_sha256": _sha256(script_path),
                "spooled_script_sha256": spooled_script_proof[
                    "observed_sha256"
                ],
                "spooled_script_proof": spooled_script_proof,
                "spooled_provenance": asdict(provenance),
                "scontrol": scontrol_proof,
            }
        )
    if not pending_ids:
        raise EvidenceError(
            "capacity-transient receipt requires at least one Priority/Resources "
            "pending fleet allocation"
        )
    if not running_ids:
        raise EvidenceError(
            "capacity-transient receipt requires a healthy RUNNING remainder"
        )
    if len(running_ids) + len(pending_ids) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS:
        raise EvidenceError("capacity-transient running/pending partition is incomplete")

    registry_snapshot = _registry_records(pool_root)
    _validate_capacity_registry_membership(
        registry_snapshot,
        running_ids=running_ids,
        pending_ids=pending_ids,
    )
    scheduler_map = {
        replica_id: (verified[replica_id][0].row, verified[replica_id][1])
        for replica_id in running_ids
    }
    records = _verify_registry(
        pool_root=pool_root,
        fleet=fleet,
        models=models,
        pins=effective_pins,
        scheduler=scheduler_map,
        replica_ids=running_ids,
        records=registry_snapshot,
    )

    def probe_one(replica_id: str) -> tuple[str, dict[str, Any]]:
        entry, _path = records[replica_id]
        replica = replica_by_id[replica_id]
        from agents_scaling.serving.profiles import get_serving_profile

        expected_model = get_serving_profile(
            replica.serving_profile
        ).served_model_name
        rounds = [
            probe(entry, expected_model, probe_timeout),
            probe(entry, expected_model, probe_timeout),
        ]
        return replica_id, _combine_http_probe_rounds(
            rounds, expected_model=expected_model
        )

    with ThreadPoolExecutor(max_workers=len(running_ids)) as executor:
        probes = dict(executor.map(probe_one, sorted(running_ids)))
    captured = float(now())
    scheduler_age = captured - float(scheduler_snapshot.captured_at)
    if not 0 <= scheduler_age <= 600:
        raise EvidenceError("capacity-transient scheduler observation is stale")
    running_detail: list[dict[str, Any]] = []
    for replica in fleet.replicas:
        if replica.replica_id not in running_ids:
            continue
        (
            allocation,
            provenance,
            script_path,
            scontrol_proof,
            spooled_script_proof,
        ) = verified[replica.replica_id]
        entry, registry_path = records[replica.replica_id]
        http = probes[replica.replica_id]
        completed = http.get("probe_completed_timestamp")
        if (
            http.get("healthy") is not True
            or not isinstance(completed, (int, float))
            or isinstance(completed, bool)
            or not math.isfinite(float(completed))
            or not 0 <= captured - float(completed) <= 600
        ):
            raise EvidenceError(
                f"running fleet endpoint failed dual health proof: {replica.replica_id}"
            )
        running_detail.append(
            {
                "replica_id": replica.replica_id,
                "serving_profile": replica.serving_profile,
                "allocated_gpus": replica.gpus_per_replica,
                "job_id": str(allocation.row.job_id),
                "state": "RUNNING",
                "partition": allocation.row.partition,
                "node": allocation.row.node,
                "comment": allocation.row.comment,
                "intent_token": allocation.attempt["intent_token"],
                "ledger_generation": allocation.ledger_generation,
                "local_script_path": str(script_path),
                "local_script_sha256": _sha256(script_path),
                "spooled_script_sha256": spooled_script_proof[
                    "observed_sha256"
                ],
                "spooled_script_proof": spooled_script_proof,
                "spooled_provenance": asdict(provenance),
                "scontrol": scontrol_proof,
                "registry_path": str(registry_path),
                "registry_sha256": _sha256(registry_path),
                "http": dict(http),
            }
        )
    return {
        "control_immutable_sha256": control["immutable_sha256"],
        "rollout_generation": generation,
        "server_pool_root": str(pool_root),
        "fleet_id": fleet.fleet_id,
        "fleet_contract_path": str(Path(fleet_binding["path"]).resolve()),
        "fleet_contract_sha256": fleet.sha256,
        "model_contract_sha256": models.sha256,
        "current_pointer_path": str(current_pointer_path),
        "current_pointer_sha256": current_pointer_sha256,
        "generation_ledger_path": str(generation_ledger_path),
        "generation_ledger_sha256": generation_ledger_sha256,
        "scheduler_captured_timestamp": float(scheduler_snapshot.captured_at),
        "scheduler_age_seconds": scheduler_age,
        "scheduler_sources": {"squeue": True, "sacct": True},
        "isolated_foreign_scheduler_job_ids": list(
            isolated_foreign_job_ids
        ),
        "logical_replicas": len(verified),
        "allocated_gpus": sum(
            replica.gpus_per_replica for replica in fleet.replicas
        ),
        "running_replicas": len(running_detail),
        "pending_replicas": len(pending_detail),
        "ignored_current_terminal_job_ids": [],
        "sealed_successful_handoff_terminal_job_ids": sorted(
            allowed_terminal_ids, key=int
        ),
        "overlap_replicas": [],
        "running": running_detail,
        "pending": pending_detail,
        "captured_timestamp": captured,
    }


def _verified_capacity_chain_binding(
    *,
    chain_manifest: Path,
    submission_receipt: Path,
    readiness_job_id: str,
) -> dict[str, Any]:
    try:
        verified = verify_recovery_evidence(chain_manifest, submission_receipt)
    except EvidenceVerificationError as exc:
        raise EvidenceError(f"capacity-transient chain evidence is invalid: {exc}") from exc
    if verified["chain_protocol"] != R8_PROTOCOL:
        raise EvidenceError(
            "capacity-transient receipt accepts only the active r8 wire protocol"
        )
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs") if isinstance(receipt, Mapping) else None
    if not isinstance(manifest_jobs, list) or not isinstance(receipt_jobs, list):
        raise EvidenceError("capacity-transient chain lacks exact jobs")
    manifest_rows = [
        row for row in manifest_jobs
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    receipt_rows = [
        row for row in receipt_jobs
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    if len(manifest_rows) != 1 or len(receipt_rows) != 1:
        raise EvidenceError("capacity-transient chain has ambiguous fleet_readiness")
    manifest_row = manifest_rows[0]
    receipt_row = receipt_rows[0]
    generation = receipt_row.get("generation", 0)
    if (
        str(receipt_row.get("job_id")) != str(readiness_job_id)
        or not str(readiness_job_id).isdigit()
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
        or receipt_row.get("comment") is None
        or manifest_row.get("job_name") is None
    ):
        raise EvidenceError(
            "capacity-transient receipt is not bound to the exact fleet readiness job"
        )
    return {
        "verified": verified,
        "manifest_row": manifest_row,
        "receipt_row": receipt_row,
        "chain_generation": generation,
    }


def _capacity_boundary_proof(
    *,
    job_id: str,
    expected_comment: str,
    expected_job_name: str,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
    boundary_seconds: int,
) -> dict[str, Any]:
    command = ["scontrol", "show", "job", "-o", str(job_id)]
    process = _invoke_scheduler(
        scheduler_runner,
        command,
        timeout=30.0,
        description="exact fleet-readiness boundary query",
    )
    fields, normalized = _parse_scontrol_record(
        process.stdout,
        expected_job_id=job_id,
        description="exact fleet-readiness boundary query",
    )
    runtime_seconds = _slurm_duration_seconds(fields.get("RunTime", ""))
    if (
        fields.get("JobState") != "RUNNING"
        or fields.get("Requeue") != "0"
        or fields.get("Comment") != expected_comment
        or fields.get("JobName") != expected_job_name
        or runtime_seconds
        < boundary_seconds - CAPACITY_TRANSIENT_BOUNDARY_TOLERANCE_SECONDS
    ):
        raise EvidenceError(
            "fleet-readiness job has not reached its exact no-requeue 10-hour "
            "capacity boundary or its scheduler identity drifted"
        )
    return {
        "argv": command,
        "job_id": job_id,
        "comment": expected_comment,
        "job_name": expected_job_name,
        "job_state": "RUNNING",
        "effective_requeue": 0,
        "runtime_seconds": runtime_seconds,
        "raw_output_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _validate_capacity_transient_evidence(
    path: Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = _require_read_only_regular(
        path, description="fleet capacity-transient evidence"
    )
    payload = _read_json(path)
    identity = dict(payload)
    evidence_id = identity.pop("evidence_id", None)
    verified = binding["verified"]
    receipt = verified["submission_receipt"]
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "chain_generation",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "fleet_readiness_job_id",
        "fleet_readiness_comment",
        "boundary_seconds",
        "boundary_proof",
        "fleet",
        "preimage_archive",
        "observed_at",
        "observed_timestamp",
        "evidence_id",
    }
    fleet = payload.get("fleet")
    preimage_archive = payload.get("preimage_archive")
    boundary = payload.get("boundary_proof")
    pending_count = (
        fleet.get("pending_replicas") if isinstance(fleet, Mapping) else None
    )
    running_count = (
        fleet.get("running_replicas") if isinstance(fleet, Mapping) else None
    )
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("protocol") != CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL
        or payload.get("passed") is not True
        or payload.get("chain_protocol") != R8_PROTOCOL
        or payload.get("chain_id") != verified["manifest"]["chain_id"]
        or payload.get("chain_generation") != binding["chain_generation"]
        or payload.get("manifest") != verified["manifest_path"]
        or payload.get("manifest_sha256") != verified["manifest_sha256"]
        or payload.get("submission_receipt")
        != verified["submission_receipt_path"]
        or payload.get("submission_receipt_sha256")
        != verified["submission_receipt_sha256"]
        or payload.get("submission_receipt_id") != receipt["receipt_id"]
        or payload.get("fleet_readiness_job_id")
        != str(binding["receipt_row"]["job_id"])
        or payload.get("fleet_readiness_comment")
        != binding["receipt_row"]["comment"]
        or payload.get("boundary_seconds") != CAPACITY_TRANSIENT_BOUNDARY_SECONDS
        or not isinstance(boundary, Mapping)
        or set(boundary)
        != {
            "argv",
            "job_id",
            "comment",
            "job_name",
            "job_state",
            "effective_requeue",
            "runtime_seconds",
            "raw_output_sha256",
        }
        or boundary.get("argv")
        != [
            "scontrol",
            "show",
            "job",
            "-o",
            str(binding["receipt_row"]["job_id"]),
        ]
        or boundary.get("job_id") != str(binding["receipt_row"]["job_id"])
        or boundary.get("comment") != binding["receipt_row"]["comment"]
        or boundary.get("job_name") != binding["manifest_row"]["job_name"]
        or boundary.get("job_state") != "RUNNING"
        or boundary.get("effective_requeue") != 0
        or not isinstance(boundary.get("runtime_seconds"), int)
        or isinstance(boundary.get("runtime_seconds"), bool)
        or boundary["runtime_seconds"]
        < (
            CAPACITY_TRANSIENT_BOUNDARY_SECONDS
            - CAPACITY_TRANSIENT_BOUNDARY_TOLERANCE_SECONDS
        )
        or re.fullmatch(
            r"[0-9a-f]{64}", str(boundary.get("raw_output_sha256", ""))
        )
        is None
        or not isinstance(fleet, Mapping)
        or not isinstance(preimage_archive, Mapping)
        or set(fleet)
        != {
            "control_immutable_sha256",
            "rollout_generation",
            "server_pool_root",
            "fleet_id",
            "fleet_contract_path",
            "fleet_contract_sha256",
            "model_contract_sha256",
            "current_pointer_path",
            "current_pointer_sha256",
            "generation_ledger_path",
            "generation_ledger_sha256",
            "scheduler_captured_timestamp",
            "scheduler_age_seconds",
            "scheduler_sources",
            "isolated_foreign_scheduler_job_ids",
            "logical_replicas",
            "allocated_gpus",
            "running_replicas",
            "pending_replicas",
            "ignored_current_terminal_job_ids",
            "sealed_successful_handoff_terminal_job_ids",
            "overlap_replicas",
            "running",
            "pending",
            "captured_timestamp",
        }
        or fleet.get("fleet_id") != "schema5-v1"
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fleet.get("control_immutable_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fleet.get("fleet_contract_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fleet.get("model_contract_sha256", ""))
        )
        is None
        or not isinstance(fleet.get("current_pointer_path"), str)
        or not Path(fleet["current_pointer_path"]).is_absolute()
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fleet.get("current_pointer_sha256", ""))
        )
        is None
        or not isinstance(fleet.get("generation_ledger_path"), str)
        or not Path(fleet["generation_ledger_path"]).is_absolute()
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fleet.get("generation_ledger_sha256", ""))
        )
        is None
        or not isinstance(fleet.get("rollout_generation"), int)
        or isinstance(fleet.get("rollout_generation"), bool)
        or fleet["rollout_generation"] < 1
        or fleet.get("scheduler_sources") != {"squeue": True, "sacct": True}
        or not isinstance(
            fleet.get("isolated_foreign_scheduler_job_ids"), list
        )
        or any(
            not isinstance(job_id, str) or not job_id.isdigit()
            for job_id in fleet.get("isolated_foreign_scheduler_job_ids", [])
        )
        or len(set(fleet.get("isolated_foreign_scheduler_job_ids", [])))
        != len(fleet.get("isolated_foreign_scheduler_job_ids", []))
        or fleet.get("logical_replicas") != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or fleet.get("allocated_gpus") != CAPACITY_TRANSIENT_EXPECTED_GPUS
        or not isinstance(pending_count, int)
        or isinstance(pending_count, bool)
        or pending_count < 1
        or not isinstance(running_count, int)
        or isinstance(running_count, bool)
        or running_count < 1
        or running_count + pending_count
        != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or fleet.get("ignored_current_terminal_job_ids") != []
        or not isinstance(
            fleet.get("sealed_successful_handoff_terminal_job_ids"), list
        )
        or any(
            not isinstance(job_id, str) or not job_id.isdigit()
            for job_id in fleet.get(
                "sealed_successful_handoff_terminal_job_ids", []
            )
        )
        or len(
            set(fleet.get("sealed_successful_handoff_terminal_job_ids", []))
        )
        != len(fleet.get("sealed_successful_handoff_terminal_job_ids", []))
        or fleet.get("overlap_replicas") != []
        or not isinstance(payload.get("observed_timestamp"), (int, float))
        or isinstance(payload.get("observed_timestamp"), bool)
        or not math.isfinite(float(payload["observed_timestamp"]))
        or payload.get("observed_at")
        != _utc(float(payload["observed_timestamp"]))
        or not isinstance(fleet.get("captured_timestamp"), (int, float))
        or isinstance(fleet.get("captured_timestamp"), bool)
        or not 0
        <= float(payload["observed_timestamp"])
        - float(fleet["captured_timestamp"])
        <= CAPACITY_TRANSIENT_MAX_EVIDENCE_AGE_SECONDS
        or not isinstance(evidence_id, str)
        or evidence_id != _identity_sha256(identity)
    ):
        raise EvidenceError("fleet capacity-transient evidence identity is invalid")
    _validate_capacity_preimage_archive(
        preimage_archive,
        fleet=fleet,
        chain_id=str(payload["chain_id"]),
        chain_generation=int(payload["chain_generation"]),
        readiness_job_id=str(payload["fleet_readiness_job_id"]),
    )
    pending = fleet.get("pending")
    running = fleet.get("running")
    pending_fields = {
        "replica_id",
        "serving_profile",
        "allocated_gpus",
        "job_id",
        "state",
        "reason",
        "partition",
        "comment",
        "intent_token",
        "ledger_generation",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_proof",
        "spooled_provenance",
        "scontrol",
    }
    running_fields = {
        "replica_id",
        "serving_profile",
        "allocated_gpus",
        "job_id",
        "state",
        "partition",
        "node",
        "comment",
        "intent_token",
        "ledger_generation",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_proof",
        "spooled_provenance",
        "scontrol",
        "registry_path",
        "registry_sha256",
        "http",
    }
    all_rows = (
        [*pending, *running]
        if isinstance(pending, list) and isinstance(running, list)
        else []
    )
    replica_ids = [
        row.get("replica_id") for row in all_rows if isinstance(row, Mapping)
    ]
    job_ids = [
        row.get("job_id") for row in all_rows if isinstance(row, Mapping)
    ]
    allocated_gpus = [
        row.get("allocated_gpus")
        for row in all_rows
        if isinstance(row, Mapping)
    ]
    scontrol_fields = {
        "argv",
        "job_id",
        "job_state",
        "reason",
        "comment",
        "job_name",
        "node",
        "command",
        "effective_requeue",
        "raw_output_sha256",
    }
    spool_fields = {
        "argv",
        "local_path",
        "local_sha256",
        "observed_sha256",
        "observed_bytes",
        "exact_match",
    }
    provenance_fields = {
        "run_root",
        "server_pool_id",
        "replica_id",
        "replica_index",
        "release_id",
        "environment_hash",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_contract_sha256",
        "fleet_contract_sha256",
    }
    http_fields = {
        "health_status",
        "models_status",
        "model_ids",
        "expected_model",
        "probe_started_timestamp",
        "probe_completed_timestamp",
        "healthy",
    }
    if (
        not isinstance(pending, list)
        or not isinstance(running, list)
        or len(pending) != pending_count
        or len(running) != running_count
        or len(replica_ids) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or any(not isinstance(value, str) or not value for value in replica_ids)
        or len(set(replica_ids)) != len(replica_ids)
        or len(job_ids) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or any(not isinstance(value, str) or not value.isdigit() for value in job_ids)
        or len(set(job_ids)) != len(job_ids)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in allocated_gpus
        )
        or sum(allocated_gpus) != CAPACITY_TRANSIENT_EXPECTED_GPUS
        or any(
            not isinstance(row, Mapping)
            or set(row) != pending_fields
            or row.get("state") != "PENDING"
            or row.get("reason") not in CAPACITY_TRANSIENT_REASONS
            or row.get("ledger_generation") != fleet["rollout_generation"]
            or re.fullmatch(r"[0-9a-f]{32}", str(row.get("intent_token", "")))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("local_script_sha256", ""))
            )
            is None
            or row.get("spooled_script_sha256")
            != row.get("local_script_sha256")
            or not isinstance(row.get("spooled_script_proof"), Mapping)
            or set(row["spooled_script_proof"]) != spool_fields
            or row["spooled_script_proof"].get("argv")
            != [
                "scontrol",
                "write",
                "batch_script",
                row.get("job_id"),
                "-",
            ]
            or row["spooled_script_proof"].get("local_path")
            != row.get("local_script_path")
            or row["spooled_script_proof"].get("local_sha256")
            != row.get("local_script_sha256")
            or row["spooled_script_proof"].get("observed_sha256")
            != row.get("spooled_script_sha256")
            or not isinstance(
                row["spooled_script_proof"].get("observed_bytes"), int
            )
            or isinstance(
                row["spooled_script_proof"].get("observed_bytes"), bool
            )
            or row["spooled_script_proof"]["observed_bytes"] < 1
            or row["spooled_script_proof"].get("exact_match") is not True
            or not isinstance(row.get("spooled_provenance"), Mapping)
            or set(row["spooled_provenance"]) != provenance_fields
            or row["spooled_provenance"].get("replica_id")
            != row.get("replica_id")
            or row["spooled_provenance"].get("server_pool_id")
            != fleet.get("fleet_id")
            or row["spooled_provenance"].get("fleet_contract_sha256")
            != fleet.get("fleet_contract_sha256")
            or row["spooled_provenance"].get("model_contract_sha256")
            != fleet.get("model_contract_sha256")
            or not isinstance(row.get("scontrol"), Mapping)
            or set(row["scontrol"]) != scontrol_fields
            or row["scontrol"].get("argv")
            != ["scontrol", "show", "job", "-o", row.get("job_id")]
            or row["scontrol"].get("job_id") != row.get("job_id")
            or row["scontrol"].get("job_state") != "PENDING"
            or row["scontrol"].get("reason") != row.get("reason")
            or row["scontrol"].get("comment") != row.get("comment")
            or row["scontrol"].get("job_name") is None
            or row["scontrol"].get("node") is not None
            or row["scontrol"].get("command") != row.get("local_script_path")
            or row["scontrol"].get("effective_requeue") != 0
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(row["scontrol"].get("raw_output_sha256", "")),
            )
            is None
            for row in pending
        )
        or any(
            not isinstance(row, Mapping)
            or set(row) != running_fields
            or row.get("state") != "RUNNING"
            or row.get("ledger_generation") != fleet["rollout_generation"]
            or re.fullmatch(r"[0-9a-f]{32}", str(row.get("intent_token", "")))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("local_script_sha256", ""))
            )
            is None
            or row.get("spooled_script_sha256")
            != row.get("local_script_sha256")
            or not isinstance(row.get("spooled_script_proof"), Mapping)
            or set(row["spooled_script_proof"]) != spool_fields
            or row["spooled_script_proof"].get("argv")
            != [
                "scontrol",
                "write",
                "batch_script",
                row.get("job_id"),
                "-",
            ]
            or row["spooled_script_proof"].get("local_path")
            != row.get("local_script_path")
            or row["spooled_script_proof"].get("local_sha256")
            != row.get("local_script_sha256")
            or row["spooled_script_proof"].get("observed_sha256")
            != row.get("spooled_script_sha256")
            or not isinstance(
                row["spooled_script_proof"].get("observed_bytes"), int
            )
            or isinstance(
                row["spooled_script_proof"].get("observed_bytes"), bool
            )
            or row["spooled_script_proof"]["observed_bytes"] < 1
            or row["spooled_script_proof"].get("exact_match") is not True
            or re.fullmatch(
                r"[0-9a-f]{64}", str(row.get("registry_sha256", ""))
            )
            is None
            or not isinstance(row.get("spooled_provenance"), Mapping)
            or set(row["spooled_provenance"]) != provenance_fields
            or row["spooled_provenance"].get("replica_id")
            != row.get("replica_id")
            or row["spooled_provenance"].get("server_pool_id")
            != fleet.get("fleet_id")
            or row["spooled_provenance"].get("fleet_contract_sha256")
            != fleet.get("fleet_contract_sha256")
            or row["spooled_provenance"].get("model_contract_sha256")
            != fleet.get("model_contract_sha256")
            or not isinstance(row.get("scontrol"), Mapping)
            or set(row["scontrol"]) != scontrol_fields
            or row["scontrol"].get("argv")
            != ["scontrol", "show", "job", "-o", row.get("job_id")]
            or row["scontrol"].get("job_id") != row.get("job_id")
            or row["scontrol"].get("job_state") != "RUNNING"
            or row["scontrol"].get("reason") is not None
            or row["scontrol"].get("comment") != row.get("comment")
            or row["scontrol"].get("job_name") is None
            or row["scontrol"].get("node") != row.get("node")
            or row["scontrol"].get("command") != row.get("local_script_path")
            or row["scontrol"].get("effective_requeue") != 0
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(row["scontrol"].get("raw_output_sha256", "")),
            )
            is None
            or not isinstance(row.get("http"), Mapping)
            or set(row["http"]) != http_fields
            or row["http"].get("healthy") is not True
            or row["http"].get("health_status") != 200
            or row["http"].get("models_status") != 200
            or row["http"].get("expected_model")
            not in row["http"].get("model_ids", [])
            for row in running
        )
    ):
        raise EvidenceError("fleet capacity-transient state is not narrowly valid")
    return payload


def validate_capacity_transient_receipt(
    marker_path: Path,
    *,
    chain_manifest: Path,
    submission_receipt: Path,
    readiness_job_id: str,
) -> dict[str, Any]:
    """Validate the marker-last receipt without consulting mutable fleet state."""

    binding = _verified_capacity_chain_binding(
        chain_manifest=chain_manifest,
        submission_receipt=submission_receipt,
        readiness_job_id=readiness_job_id,
    )
    marker_path = _require_read_only_regular(
        marker_path, description="fleet capacity-transient completion marker"
    )
    marker = _read_json(marker_path)
    identity = dict(marker)
    receipt_id = identity.pop("receipt_id", None)
    evidence_path = marker_path.parent / CAPACITY_TRANSIENT_EVIDENCE_NAME
    evidence = _validate_capacity_transient_evidence(
        evidence_path, binding=binding
    )
    published_timestamp = marker.get("published_timestamp")
    evidence_timestamp = evidence.get("observed_timestamp")
    verified = binding["verified"]
    required = {
        "schema_version",
        "protocol",
        "passed",
        "capacity_transient_root",
        "chain_id",
        "chain_generation",
        "fleet_readiness_job_id",
        "fleet_readiness_comment",
        "evidence",
        "evidence_sha256",
        "evidence_id",
        "preimage_archive",
        "published_at",
        "published_timestamp",
        "receipt_id",
    }
    if (
        marker_path.name != CAPACITY_TRANSIENT_MARKER_NAME
        or set(marker) != required
        or marker.get("schema_version") != 1
        or marker.get("protocol") != CAPACITY_TRANSIENT_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("capacity_transient_root") is not True
        or marker.get("chain_id") != verified["manifest"]["chain_id"]
        or marker.get("chain_generation") != binding["chain_generation"]
        or marker.get("fleet_readiness_job_id") != readiness_job_id
        or marker.get("fleet_readiness_comment")
        != binding["receipt_row"]["comment"]
        or marker.get("evidence") != str(evidence_path.resolve())
        or marker.get("evidence_sha256") != _sha256(evidence_path)
        or marker.get("evidence_id") != evidence["evidence_id"]
        or marker.get("preimage_archive") != evidence.get("preimage_archive")
        or not isinstance(published_timestamp, (int, float))
        or isinstance(published_timestamp, bool)
        or not math.isfinite(float(published_timestamp))
        or marker.get("published_at") != _utc(float(published_timestamp))
        or not isinstance(evidence_timestamp, (int, float))
        or isinstance(evidence_timestamp, bool)
        or not 0
        <= float(published_timestamp) - float(evidence_timestamp)
        <= CAPACITY_TRANSIENT_MAX_EVIDENCE_AGE_SECONDS
        or not isinstance(receipt_id, str)
        or receipt_id != _identity_sha256(identity)
    ):
        raise EvidenceError("fleet capacity-transient completion marker is invalid")
    return marker | {
        "marker_path": str(marker_path),
        "marker_sha256": _sha256(marker_path),
    }


def _write_capacity_preimage_bytes(path: Path, payload: bytes) -> tuple[str, int]:
    """Write one real, fsynced, read-only file without clone/hardlink shortcuts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise EvidenceError(f"capacity preimage parent is unsafe: {path.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise EvidenceError(f"cannot create capacity preimage {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise EvidenceError(f"short write while archiving {path}")
            digest.update(view[offset : offset + written])
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return digest.hexdigest(), len(payload)


def _copy_capacity_preimage(
    source: Path, destination: Path, *, expected_sha256: str
) -> tuple[str, int]:
    source = source.expanduser().absolute()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise EvidenceError(f"capacity preimage source is unavailable: {source}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceError(f"capacity preimage source is unsafe: {source}")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read capacity preimage source {source}: {exc}") from exc
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise EvidenceError(
            f"capacity preimage source drifted: {source}: "
            f"expected {expected_sha256}, got {observed}"
        )
    digest, size = _write_capacity_preimage_bytes(destination, payload)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (
        source_stat.st_dev == destination_stat.st_dev
        and source_stat.st_ino == destination_stat.st_ino
    ):
        raise EvidenceError(
            f"capacity preimage copy shares an inode with its source: {source}"
        )
    return digest, size


def _capacity_archive_binding(
    *,
    root: Path,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    completion_path = root / CAPACITY_PREIMAGE_COMPLETE_NAME
    return {
        "schema_version": 1,
        "protocol": CAPACITY_PREIMAGE_PROTOCOL,
        "root": str(root),
        "completion": str(completion_path),
        "completion_sha256": _sha256(completion_path),
        "archive_id": completion["archive_id"],
        "manifest": completion["manifest"],
        "manifest_sha256": completion["manifest_sha256"],
        "manifest_id": completion["manifest_id"],
        "inventory": completion["inventory"],
        "inventory_sha256": completion["inventory_sha256"],
        "file_count": completion["file_count"],
        "total_bytes": completion["total_bytes"],
    }


def _validate_capacity_preimage_ledger(
    *,
    fleet: Mapping[str, Any],
    ledger: Mapping[str, Any],
    archive_records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind the sealed generation ledger to every admitted capacity row.

    The after-any sentinel cannot consult mutable fleet state.  The producer must
    therefore enforce the same committed-attempt and health contract before it
    publishes a repairable receipt, using only bytes that were copied into the
    sealed preimage archive.
    """

    rows = [*fleet.get("pending", []), *fleet.get("running", [])]
    replica_ids = [str(row.get("replica_id", "")) for row in rows]
    if (
        len(replica_ids) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or any(not replica_id for replica_id in replica_ids)
        or len(set(replica_ids)) != len(replica_ids)
    ):
        raise EvidenceError(
            "capacity preimage ledger has an invalid replica identity set"
        )
    ledger_path = Path(str(fleet["generation_ledger_path"]))
    transaction_directory = ledger_path.parent.parent
    try:
        fleet_tx._validate_ledger_material(  # noqa: SLF001
            dict(ledger),
            directory=transaction_directory,
            canonical_root=Path(str(fleet["server_pool_root"]))
            .expanduser()
            .resolve(),
            pool_id=str(fleet["fleet_id"]),
            fleet_sha256=str(fleet["fleet_contract_sha256"]),
            rollout_generation=int(fleet["rollout_generation"]),
            replica_ids=replica_ids,
        )
    except fleet_tx.FleetTransactionError as exc:
        raise EvidenceError(
            f"capacity preimage generation ledger is invalid: {exc}"
        ) from exc
    if (
        not isinstance(ledger.get("created_at"), (int, float))
        or isinstance(ledger.get("created_at"), bool)
        or not math.isfinite(float(ledger["created_at"]))
        or not isinstance(ledger.get("updated_at"), (int, float))
        or isinstance(ledger.get("updated_at"), bool)
        or not math.isfinite(float(ledger["updated_at"]))
    ):
        raise EvidenceError("capacity preimage ledger timestamps are invalid")
    archived_registry = {
        str(record["replica_id"]): Path(str(record["archive_path"]))
        for record in archive_records
        if record.get("kind") == "registry"
    }
    ledger_replicas = ledger["replicas"]
    observed_tokens: set[str] = set()
    observed_jobs: set[str] = set()
    for row in rows:
        replica_id = str(row["replica_id"])
        record = ledger_replicas[replica_id]
        attempts = record["attempts"]
        for candidate in attempts:
            token = str(candidate.get("intent_token") or "")
            job_id = candidate.get("job_id")
            if token in observed_tokens or (
                isinstance(job_id, str) and job_id in observed_jobs
            ):
                raise EvidenceError(
                    "capacity preimage ledger reuses an intent/job identity: "
                    f"{replica_id}"
                )
            observed_tokens.add(token)
            if isinstance(job_id, str):
                observed_jobs.add(job_id)
            for field in (
                "created_at",
                "submit_started_at",
                "submitted_at",
                "committed_at",
                "terminal_at",
                "last_seen_at",
                "missing_since",
                "predecessor_end_at",
                "scheduler_start_at",
                "scheduler_end_at",
                "scheduler_time_limit_seconds",
                "last_ready_probe_at",
                "promoted_at",
                "retire_requested_at",
                "last_retire_attempt_at",
            ):
                value = candidate.get(field)
                if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise EvidenceError(
                        "capacity preimage attempt timestamp is invalid: "
                        f"{replica_id}:{field}"
                    )
        active = [
            attempt
            for attempt in attempts
            if attempt["state"]
            in {"prepared", "submitting", "submitted", "committed", "missing"}
        ]
        if len(active) != 1:
            raise EvidenceError(
                "capacity preimage ledger has an ambiguous active attempt: "
                f"{replica_id}"
            )
        attempt = active[0]
        if (
            attempt.get("state") != "committed"
            or attempt.get("intent_token") != row.get("intent_token")
            or attempt.get("job_id") != row.get("job_id")
            or attempt.get("sbatch_path") != row.get("local_script_path")
            or attempt.get("sbatch_sha256") != row.get("local_script_sha256")
            or attempt.get("scheduler_comment") != row.get("comment")
            or attempt.get("allocated_gpus") != row.get("allocated_gpus")
            or not isinstance(attempt.get("committed_at"), (int, float))
            or isinstance(attempt.get("committed_at"), bool)
            or not math.isfinite(float(attempt["committed_at"]))
            or attempt.get("last_error") is not None
            or attempt.get("lifecycle") not in {"primary", "promoted"}
            or (
                attempt.get("lifecycle") == "primary"
                and (
                    attempt.get("launch_kind") != "primary"
                    or attempt.get("predecessor_job_id") is not None
                    or attempt.get("promoted_at") is not None
                )
            )
            or (
                attempt.get("lifecycle") == "promoted"
                and (
                    attempt.get("launch_kind") != "handoff"
                    or not str(attempt.get("predecessor_job_id") or "").isdigit()
                    or not isinstance(attempt.get("promoted_at"), (int, float))
                    or isinstance(attempt.get("promoted_at"), bool)
                    or not math.isfinite(float(attempt["promoted_at"]))
                    or attempt.get("ready_probe_count", 0) < 2
                )
            )
        ):
            raise EvidenceError(
                "capacity preimage row is not bound to a clean committed "
                f"attempt: {replica_id}"
            )
        for predecessor in (
            candidate for candidate in attempts if candidate is not attempt
        ):
            predecessor_job = str(predecessor.get("job_id") or "")
            successors = [
                candidate
                for candidate in attempts
                if candidate.get("launch_kind") == "handoff"
                and str(candidate.get("predecessor_job_id") or "")
                == predecessor_job
            ]
            if (
                predecessor.get("state") != "terminal"
                or predecessor.get("lifecycle") != "retiring"
                or not isinstance(predecessor.get("terminal_at"), (int, float))
                or isinstance(predecessor.get("terminal_at"), bool)
                or not math.isfinite(float(predecessor["terminal_at"]))
                or predecessor.get("last_error") is not None
                or not isinstance(
                    predecessor.get("retire_requested_at"), (int, float)
                )
                or isinstance(predecessor.get("retire_requested_at"), bool)
                or not math.isfinite(
                    float(predecessor["retire_requested_at"])
                )
                or predecessor.get("retire_attempts", 0) < 1
                or not isinstance(
                    predecessor.get("last_retire_attempt_at"), (int, float)
                )
                or isinstance(predecessor.get("last_retire_attempt_at"), bool)
                or not math.isfinite(
                    float(predecessor["last_retire_attempt_at"])
                )
                or predecessor.get("retire_error") is not None
                or len(successors) != 1
            ):
                raise EvidenceError(
                    "capacity preimage contains unsealed handoff history: "
                    f"{replica_id}"
                )
            successor = successors[0]
            if (
                successor.get("state") not in {"committed", "terminal"}
                or successor.get("lifecycle") not in {"promoted", "retiring"}
                or not isinstance(successor.get("promoted_at"), (int, float))
                or isinstance(successor.get("promoted_at"), bool)
                or not math.isfinite(float(successor["promoted_at"]))
                or successor.get("ready_probe_count", 0) < 2
                or successor.get("last_error") is not None
                or successor.get("retire_error") is not None
            ):
                raise EvidenceError(
                    "capacity preimage handoff successor is invalid: "
                    f"{replica_id}"
                )
            attempts_by_job = {
                str(candidate["job_id"]): candidate
                for candidate in attempts
                if isinstance(candidate.get("job_id"), str)
            }
            visited: set[str] = set()
            current_job = predecessor_job
            active_job = str(attempt.get("job_id"))
            while current_job != active_job:
                if current_job in visited:
                    raise EvidenceError(
                        "capacity preimage handoff history contains a cycle: "
                        f"{replica_id}"
                    )
                visited.add(current_job)
                next_jobs = [
                    str(candidate.get("job_id"))
                    for candidate in attempts
                    if candidate.get("launch_kind") == "handoff"
                    and str(candidate.get("predecessor_job_id") or "")
                    == current_job
                ]
                if (
                    len(next_jobs) != 1
                    or next_jobs[0] not in attempts_by_job
                ):
                    raise EvidenceError(
                        "capacity preimage handoff history is disconnected: "
                        f"{replica_id}"
                    )
                current_job = next_jobs[0]
        health = record["health"]
        if row.get("state") == "PENDING":
            if health is not None:
                raise EvidenceError(
                    "capacity preimage pending allocation has stale health: "
                    f"{replica_id}"
                )
            continue
        if (
            not isinstance(health, Mapping)
            or health.get("job_id") != row.get("job_id")
            or health.get("observer_generation")
            != fleet.get("rollout_generation")
            or health.get("consecutive_failures") != 0
            or health.get("cancel_state") is not None
            or health.get("cancel_error") is not None
            or not isinstance(health.get("endpoint"), str)
            or not health["endpoint"]
        ):
            raise EvidenceError(
                f"capacity preimage running health is invalid: {replica_id}"
            )
        for field in (
            "first_failure_at",
            "last_failure_at",
            "last_probe_at",
            "cancel_requested_at",
            "cancel_completed_at",
            "last_cancel_attempt_at",
            "next_cancel_eligible_at",
        ):
            value = health.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise EvidenceError(
                    "capacity preimage health timestamp is invalid: "
                    f"{replica_id}:{field}"
                )
        registry_path = archived_registry.get(replica_id)
        if registry_path is None:
            raise EvidenceError(
                f"capacity preimage omitted running registry: {replica_id}"
            )
        registry = _read_json(registry_path)
        host = registry.get("host")
        port = registry.get("port")
        if (
            not isinstance(host, str)
            or not host
            or not isinstance(port, int)
            or isinstance(port, bool)
            or port < 1
            or port > 65_535
            or health["endpoint"] != f"{host}:{port}"
        ):
            raise EvidenceError(
                "capacity preimage health/registry endpoint drifted: "
                f"{replica_id}"
            )


def _validate_capacity_preimage_archive(
    binding: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    chain_id: str,
    chain_generation: int,
    readiness_job_id: str,
) -> dict[str, Any]:
    """Validate the self-contained receipt archive without reopening live state."""

    required_binding = {
        "schema_version",
        "protocol",
        "root",
        "completion",
        "completion_sha256",
        "archive_id",
        "manifest",
        "manifest_sha256",
        "manifest_id",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
    }
    if set(binding) != required_binding:
        raise EvidenceError("capacity preimage archive binding fields drifted")
    root = Path(str(binding.get("root", "")))
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise EvidenceError("capacity preimage archive root is invalid")
    completion_path = _require_read_only_regular(
        Path(str(binding["completion"])),
        description="capacity preimage completion",
    )
    manifest_path = _require_read_only_regular(
        Path(str(binding["manifest"])),
        description="capacity preimage manifest",
    )
    inventory_path = _require_read_only_regular(
        Path(str(binding["inventory"])),
        description="capacity preimage inventory",
    )
    if (
        completion_path != root / CAPACITY_PREIMAGE_COMPLETE_NAME
        or manifest_path != root / CAPACITY_PREIMAGE_MANIFEST_NAME
        or inventory_path != root / CAPACITY_PREIMAGE_INVENTORY_NAME
        or _sha256(completion_path) != binding["completion_sha256"]
        or _sha256(manifest_path) != binding["manifest_sha256"]
        or _sha256(inventory_path) != binding["inventory_sha256"]
    ):
        raise EvidenceError("capacity preimage archive path/hash binding drifted")
    completion = _read_json(completion_path)
    completion_identity = dict(completion)
    archive_id = completion_identity.pop("archive_id", None)
    manifest = _read_json(manifest_path)
    manifest_identity = dict(manifest)
    manifest_id = manifest_identity.pop("manifest_id", None)
    completion_fields = {
        "schema_version",
        "protocol",
        "passed",
        "root",
        "manifest",
        "manifest_sha256",
        "manifest_id",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "archive_id",
    }
    manifest_fields = {
        "schema_version",
        "protocol",
        "chain_id",
        "chain_generation",
        "fleet_readiness_job_id",
        "rollout_generation",
        "fleet_contract_sha256",
        "files",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "manifest_id",
    }
    if (
        set(completion) != completion_fields
        or completion.get("schema_version") != 1
        or completion.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or completion.get("passed") is not True
        or completion.get("root") != str(root)
        or completion.get("manifest") != str(manifest_path)
        or completion.get("manifest_sha256") != _sha256(manifest_path)
        or completion.get("manifest_id") != manifest_id
        or completion.get("inventory") != str(inventory_path)
        or completion.get("inventory_sha256") != _sha256(inventory_path)
        or not isinstance(archive_id, str)
        or archive_id != _identity_sha256(completion_identity)
        or archive_id != binding.get("archive_id")
        or set(manifest) != manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or manifest.get("chain_id") != chain_id
        or manifest.get("chain_generation") != chain_generation
        or manifest.get("fleet_readiness_job_id") != readiness_job_id
        or manifest.get("rollout_generation") != fleet.get("rollout_generation")
        or manifest.get("fleet_contract_sha256")
        != fleet.get("fleet_contract_sha256")
        or manifest.get("inventory") != str(inventory_path)
        or manifest.get("inventory_sha256") != _sha256(inventory_path)
        or not isinstance(manifest_id, str)
        or manifest_id != _identity_sha256(manifest_identity)
        or manifest_id != binding.get("manifest_id")
    ):
        raise EvidenceError("capacity preimage archive identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise EvidenceError("capacity preimage manifest files are invalid")
    record_fields = {
        "kind",
        "replica_id",
        "job_id",
        "logical_path",
        "archive_path",
        "source_path",
        "source_argv",
        "sha256",
        "bytes",
    }
    inventory_lines: list[str] = []
    kinds: Counter[str] = Counter()
    logical_paths: set[str] = set()
    total_bytes = 0
    for record in files:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise EvidenceError("capacity preimage manifest record fields drifted")
        logical = record.get("logical_path")
        archived = Path(str(record.get("archive_path", "")))
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(logical, str)
            or not logical
            or logical.startswith("/")
            or ".." in Path(logical).parts
            or logical in logical_paths
            or archived != root / logical
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise EvidenceError("capacity preimage manifest record is invalid")
        archived = _require_read_only_regular(
            archived, description=f"capacity preimage {logical}"
        )
        if _sha256(archived) != digest or archived.stat().st_size != size:
            raise EvidenceError(f"capacity preimage bytes drifted: {logical}")
        source_path = record.get("source_path")
        if source_path is not None:
            if not isinstance(source_path, str) or not Path(source_path).is_absolute():
                raise EvidenceError(f"capacity preimage source path is invalid: {logical}")
            source = Path(source_path)
            if source.exists() and not source.is_symlink():
                source_stat = source.stat()
                archived_stat = archived.stat()
                if (
                    source_stat.st_dev == archived_stat.st_dev
                    and source_stat.st_ino == archived_stat.st_ino
                ):
                    raise EvidenceError(
                        f"capacity preimage shares a live inode: {logical}"
                    )
        logical_paths.add(logical)
        kinds[str(record.get("kind"))] += 1
        total_bytes += size
        inventory_lines.append(f"{digest}  {logical}\n")
    expected_rows = [
        *fleet.get("pending", []),
        *fleet.get("running", []),
    ]
    expected_by_key: dict[tuple[str, str | None], dict[str, Any]] = {
        ("current_pointer", None): {
            "job_id": None,
            "source_path": fleet.get("current_pointer_path"),
            "source_argv": None,
            "sha256": fleet.get("current_pointer_sha256"),
        },
        ("generation_ledger", None): {
            "job_id": None,
            "source_path": fleet.get("generation_ledger_path"),
            "source_argv": None,
            "sha256": fleet.get("generation_ledger_sha256"),
        },
    }
    for row in expected_rows:
        replica_id = str(row["replica_id"])
        job_id = str(row["job_id"])
        expected_by_key[("local_script", replica_id)] = {
            "job_id": job_id,
            "source_path": row["local_script_path"],
            "source_argv": None,
            "sha256": row["local_script_sha256"],
        }
        expected_by_key[("spooled_script", replica_id)] = {
            "job_id": job_id,
            "source_path": None,
            "source_argv": [
                "scontrol",
                "write",
                "batch_script",
                job_id,
                "-",
            ],
            "sha256": row["spooled_script_sha256"],
        }
    for row in fleet.get("running", []):
        replica_id = str(row["replica_id"])
        expected_by_key[("registry", replica_id)] = {
            "job_id": str(row["job_id"]),
            "source_path": row["registry_path"],
            "source_argv": None,
            "sha256": row["registry_sha256"],
        }
    observed_keys: set[tuple[str, str | None]] = set()
    for record in files:
        key = (
            str(record["kind"]),
            None
            if record["replica_id"] is None
            else str(record["replica_id"]),
        )
        expected_record = expected_by_key.get(key)
        if (
            expected_record is None
            or key in observed_keys
            or record["job_id"] != expected_record["job_id"]
            or record["source_path"] != expected_record["source_path"]
            or record["source_argv"] != expected_record["source_argv"]
            or record["sha256"] != expected_record["sha256"]
        ):
            raise EvidenceError(
                f"capacity preimage is not bound to fleet row {key}"
            )
        observed_keys.add(key)
    expected_running = len(fleet.get("running", []))
    expected_counts = {
        "current_pointer": 1,
        "generation_ledger": 1,
        "local_script": len(expected_rows),
        "spooled_script": len(expected_rows),
        "registry": expected_running,
    }
    inventory_payload = "".join(
        f"{record['sha256']}  {record['logical_path']}\n"
        for record in sorted(
            files, key=lambda item: str(item["logical_path"])
        )
    )
    expected_archive_files = {
        completion_path,
        manifest_path,
        inventory_path,
        *(Path(str(record["archive_path"])) for record in files),
    }
    observed_archive_files: set[Path] = set()
    for member in root.rglob("*"):
        metadata = member.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError(f"capacity preimage archive contains a symlink: {member}")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise EvidenceError(
                    f"capacity preimage archive directory is writable: {member}"
                )
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise EvidenceError(
                f"capacity preimage archive member is unsafe/writable: {member}"
            )
        observed_archive_files.add(member)
    if (
        kinds != Counter(expected_counts)
        or observed_keys != set(expected_by_key)
        or len(files) != sum(expected_counts.values())
        or manifest.get("file_count") != len(files)
        or manifest.get("total_bytes") != total_bytes
        or binding.get("file_count") != len(files)
        or binding.get("total_bytes") != total_bytes
        or completion.get("file_count") != len(files)
        or completion.get("total_bytes") != total_bytes
        or inventory_path.read_text(encoding="utf-8") != inventory_payload
        or observed_archive_files != expected_archive_files
        or stat.S_IMODE(root.stat().st_mode) & 0o222
    ):
        raise EvidenceError("capacity preimage archive cardinality/inventory drifted")
    current_record = next(
        record for record in files if record["kind"] == "current_pointer"
    )
    ledger_record = next(
        record for record in files if record["kind"] == "generation_ledger"
    )
    current_payload = _read_json(Path(str(current_record["archive_path"])))
    ledger_payload = _read_json(Path(str(ledger_record["archive_path"])))
    current_fields = {
        "schema_version",
        "pool_root",
        "pool_id",
        "fleet_sha256",
        "current_generation",
        "ledger_path",
        "ledger_sha256",
        "updated_at",
    }
    if (
        set(current_payload) != current_fields
        or current_payload.get("schema_version")
        != fleet_tx.STATE_SCHEMA_VERSION
        or current_payload.get("pool_root") != fleet.get("server_pool_root")
        or current_payload.get("pool_id") != fleet.get("fleet_id")
        or current_payload.get("fleet_sha256")
        != fleet.get("fleet_contract_sha256")
        or current_payload.get("current_generation")
        != fleet.get("rollout_generation")
        or current_payload.get("ledger_path")
        != fleet.get("generation_ledger_path")
        or current_payload.get("ledger_sha256") != ledger_record["sha256"]
        or not isinstance(current_payload.get("updated_at"), (int, float))
        or isinstance(current_payload.get("updated_at"), bool)
        or not math.isfinite(float(current_payload["updated_at"]))
        or ledger_payload.get("schema_version") != fleet_tx.STATE_SCHEMA_VERSION
        or ledger_payload.get("pool_root") != fleet.get("server_pool_root")
        or ledger_payload.get("pool_id") != fleet.get("fleet_id")
        or ledger_payload.get("fleet_sha256")
        != fleet.get("fleet_contract_sha256")
        or ledger_payload.get("rollout_generation")
        != fleet.get("rollout_generation")
    ):
        raise EvidenceError(
            "capacity preimage CURRENT/generation-ledger semantics drifted"
        )
    _validate_capacity_preimage_ledger(
        fleet=fleet,
        ledger=ledger_payload,
        archive_records=files,
    )
    return dict(binding)


def _build_capacity_preimage_archive(
    *,
    parent: Path,
    fleet: Mapping[str, Any],
    chain_id: str,
    chain_generation: int,
    readiness_job_id: str,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None,
) -> dict[str, Any]:
    """Capture all mutable/live capacity proof inputs into one sealed archive."""

    parent = parent.expanduser().absolute()
    archive_key = _identity_sha256(
        {
            "chain_id": chain_id,
            "chain_generation": chain_generation,
            "fleet_readiness_job_id": readiness_job_id,
            "rollout_generation": fleet.get("rollout_generation"),
            "current_pointer_sha256": fleet.get("current_pointer_sha256"),
            "generation_ledger_sha256": fleet.get("generation_ledger_sha256"),
            "scheduler_captured_timestamp": fleet.get(
                "scheduler_captured_timestamp"
            ),
            "captured_timestamp": fleet.get("captured_timestamp"),
        }
    )
    root = parent / f"{CAPACITY_PREIMAGE_ROOT_NAME}-{archive_key[:24]}"
    completion_path = root / CAPACITY_PREIMAGE_COMPLETE_NAME
    if completion_path.exists() and not completion_path.is_symlink():
        completion = _read_json(completion_path)
        return _validate_capacity_preimage_archive(
            _capacity_archive_binding(root=root, completion=completion),
            fleet=fleet,
            chain_id=chain_id,
            chain_generation=chain_generation,
            readiness_job_id=readiness_job_id,
        )
    if root.exists() or root.is_symlink():
        incident_root = parent / CAPACITY_TRANSIENT_INCIDENT_DIRECTORY
        incident_root.mkdir(mode=0o750, exist_ok=True)
        if incident_root.is_symlink() or not incident_root.is_dir():
            raise EvidenceError(
                f"capacity preimage incident root is unsafe: {incident_root}"
            )
        partial = incident_root / (
            f"{CAPACITY_PREIMAGE_ROOT_NAME}.partial-{os.getpid()}-{time.time_ns()}"
        )
        os.rename(root, partial)
        _fsync_directory(parent)
        _fsync_directory(incident_root)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{CAPACITY_PREIMAGE_ROOT_NAME}.",
            dir=parent,
        )
    )
    final_records: list[dict[str, Any]] = []

    def archive_source(
        *,
        kind: str,
        logical_path: str,
        source: Path,
        expected_sha256: str,
        replica_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        destination = temporary / logical_path
        digest, size = _copy_capacity_preimage(
            source, destination, expected_sha256=expected_sha256
        )
        final_records.append(
            {
                "kind": kind,
                "replica_id": replica_id,
                "job_id": job_id,
                "logical_path": logical_path,
                "archive_path": str(root / logical_path),
                "source_path": str(source.expanduser().absolute()),
                "source_argv": None,
                "sha256": digest,
                "bytes": size,
            }
        )

    try:
        archive_source(
            kind="current_pointer",
            logical_path="fleet-state/CURRENT.json",
            source=Path(str(fleet["current_pointer_path"])),
            expected_sha256=str(fleet["current_pointer_sha256"]),
        )
        archive_source(
            kind="generation_ledger",
            logical_path="fleet-state/generation-ledger.json",
            source=Path(str(fleet["generation_ledger_path"])),
            expected_sha256=str(fleet["generation_ledger_sha256"]),
        )
        all_rows = [*fleet["pending"], *fleet["running"]]
        for row in sorted(all_rows, key=lambda item: str(item["replica_id"])):
            replica_id = str(row["replica_id"])
            job_id = str(row["job_id"])
            prefix = f"jobs/{replica_id}"
            archive_source(
                kind="local_script",
                logical_path=f"{prefix}/local.sbatch",
                source=Path(str(row["local_script_path"])),
                expected_sha256=str(row["local_script_sha256"]),
                replica_id=replica_id,
                job_id=job_id,
            )
            command = ["scontrol", "write", "batch_script", job_id, "-"]
            completed = _invoke_scheduler(
                scheduler_runner,
                command,
                timeout=15.0,
                description=f"capacity preimage spooled script for {job_id}",
            )
            spooled_payload = completed.stdout.encode("utf-8")
            observed_sha256 = hashlib.sha256(spooled_payload).hexdigest()
            if observed_sha256 != row["spooled_script_sha256"]:
                raise EvidenceError(
                    f"capacity spooled script drifted during archive: {job_id}"
                )
            logical = f"{prefix}/spooled.sbatch"
            digest, size = _write_capacity_preimage_bytes(
                temporary / logical, spooled_payload
            )
            final_records.append(
                {
                    "kind": "spooled_script",
                    "replica_id": replica_id,
                    "job_id": job_id,
                    "logical_path": logical,
                    "archive_path": str(root / logical),
                    "source_path": None,
                    "source_argv": command,
                    "sha256": digest,
                    "bytes": size,
                }
            )
        for row in sorted(
            fleet["running"], key=lambda item: str(item["replica_id"])
        ):
            replica_id = str(row["replica_id"])
            archive_source(
                kind="registry",
                logical_path=f"jobs/{replica_id}/registry.json",
                source=Path(str(row["registry_path"])),
                expected_sha256=str(row["registry_sha256"]),
                replica_id=replica_id,
                job_id=str(row["job_id"]),
            )
        final_records.sort(key=lambda record: str(record["logical_path"]))
        inventory_payload = "".join(
            f"{record['sha256']}  {record['logical_path']}\n"
            for record in final_records
        ).encode("utf-8")
        inventory_path = temporary / CAPACITY_PREIMAGE_INVENTORY_NAME
        inventory_sha256, _ = _write_capacity_preimage_bytes(
            inventory_path, inventory_payload
        )
        total_bytes = sum(int(record["bytes"]) for record in final_records)
        manifest_path = temporary / CAPACITY_PREIMAGE_MANIFEST_NAME
        manifest = {
            "schema_version": 1,
            "protocol": CAPACITY_PREIMAGE_PROTOCOL,
            "chain_id": chain_id,
            "chain_generation": chain_generation,
            "fleet_readiness_job_id": readiness_job_id,
            "rollout_generation": fleet["rollout_generation"],
            "fleet_contract_sha256": fleet["fleet_contract_sha256"],
            "files": final_records,
            "inventory": str(root / CAPACITY_PREIMAGE_INVENTORY_NAME),
            "inventory_sha256": inventory_sha256,
            "file_count": len(final_records),
            "total_bytes": total_bytes,
        }
        manifest["manifest_id"] = _identity_sha256(manifest)
        _write_json_atomic(manifest_path, manifest)
        manifest_sha256 = _sha256(manifest_path)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        os.rename(temporary, root)
        _fsync_directory(parent)
        completion = {
            "schema_version": 1,
            "protocol": CAPACITY_PREIMAGE_PROTOCOL,
            "passed": True,
            "root": str(root),
            "manifest": str(root / CAPACITY_PREIMAGE_MANIFEST_NAME),
            "manifest_sha256": manifest_sha256,
            "manifest_id": manifest["manifest_id"],
            "inventory": str(root / CAPACITY_PREIMAGE_INVENTORY_NAME),
            "inventory_sha256": inventory_sha256,
            "file_count": len(final_records),
            "total_bytes": total_bytes,
        }
        completion["archive_id"] = _identity_sha256(completion)
        _write_json_atomic(root / CAPACITY_PREIMAGE_COMPLETE_NAME, completion)
        root.chmod(0o555)
        _fsync_directory(parent)
        return _validate_capacity_preimage_archive(
            _capacity_archive_binding(root=root, completion=completion),
            fleet=fleet,
            chain_id=chain_id,
            chain_generation=chain_generation,
            readiness_job_id=readiness_job_id,
        )
    except BaseException:
        if temporary.exists():
            # Preserve the complete partial transfer for postmortem; never delete it.
            incident_root = parent / CAPACITY_TRANSIENT_INCIDENT_DIRECTORY
            incident_root.mkdir(mode=0o750, exist_ok=True)
            partial = incident_root / (
                f"{CAPACITY_PREIMAGE_ROOT_NAME}.partial-{os.getpid()}-{time.time_ns()}"
            )
            os.rename(temporary, partial)
            _fsync_directory(parent)
            _fsync_directory(incident_root)
        raise


def _archive_stale_capacity_evidence(
    evidence_path: Path, evidence: Mapping[str, Any]
) -> Path:
    """Move an orphaned pre-marker proof aside without discarding its preimage."""

    evidence_path = _require_read_only_regular(
        evidence_path, description="stale fleet capacity-transient evidence"
    )
    evidence_id = evidence.get("evidence_id")
    if (
        not isinstance(evidence_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_id) is None
    ):
        raise EvidenceError(
            "stale capacity-transient evidence lacks a valid evidence ID"
        )
    digest = _sha256(evidence_path)
    incident_root = evidence_path.parent / CAPACITY_TRANSIENT_INCIDENT_DIRECTORY
    incident_root.mkdir(mode=0o750, exist_ok=True)
    if incident_root.is_symlink() or not incident_root.is_dir():
        raise EvidenceError(
            f"capacity-transient incident root is unsafe: {incident_root}"
        )
    archived = incident_root / (
        f"{evidence_path.stem}.stale-{evidence_id}-{digest[:16]}.json"
    )
    if archived.exists() or archived.is_symlink():
        raise EvidenceError(
            "stale capacity-transient preimage archive already exists while the "
            "canonical evidence path is still present"
        )
    os.replace(evidence_path, archived)
    _fsync_directory(incident_root)
    _fsync_directory(evidence_path.parent)
    return archived


def build_fleet_capacity_transient_receipt(
    control: Mapping[str, Any],
    *,
    output: Path,
    state_dir: Path,
    chain_manifest: Path,
    submission_receipt: Path,
    readiness_job_id: str,
    boundary_seconds: int = CAPACITY_TRANSIENT_BOUNDARY_SECONDS,
    probe_timeout: float = 10.0,
    scheduler_runner: Callable[
        [Sequence[str], float], subprocess.CompletedProcess[str]
    ]
    | None = None,
    probe: Callable[[ServerEntry, str, float], Mapping[str, Any]] = _http_probe,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Publish the sole deterministic exit-75 fleet-capacity receipt marker-last."""

    if boundary_seconds != CAPACITY_TRANSIENT_BOUNDARY_SECONDS:
        raise EvidenceError("capacity-transient boundary must be exactly 36,000 seconds")
    output = output.expanduser().absolute()
    if output.name != CAPACITY_TRANSIENT_MARKER_NAME:
        raise EvidenceError(
            f"capacity-transient output must be named {CAPACITY_TRANSIENT_MARKER_NAME}"
        )
    binding = _verified_capacity_chain_binding(
        chain_manifest=chain_manifest,
        submission_receipt=submission_receipt,
        readiness_job_id=readiness_job_id,
    )
    with _CapacityReceiptLock(output.parent):
        if output.exists() or output.is_symlink():
            return validate_capacity_transient_receipt(
                output,
                chain_manifest=chain_manifest,
                submission_receipt=submission_receipt,
                readiness_job_id=readiness_job_id,
            ) | {"status": "already_complete"}
        evidence_path = output.parent / CAPACITY_TRANSIENT_EVIDENCE_NAME
        evidence: dict[str, Any] | None = None
        if evidence_path.exists() or evidence_path.is_symlink():
            evidence = _validate_capacity_transient_evidence(
                evidence_path, binding=binding
            )
            recovery_timestamp = float(now())
            observed_timestamp = float(evidence["observed_timestamp"])
            if (
                not math.isfinite(recovery_timestamp)
                or recovery_timestamp < observed_timestamp
                or recovery_timestamp - observed_timestamp
                > CAPACITY_TRANSIENT_MAX_EVIDENCE_AGE_SECONDS
            ):
                _archive_stale_capacity_evidence(evidence_path, evidence)
                evidence = None
        if evidence is None:
            boundary = _capacity_boundary_proof(
                job_id=readiness_job_id,
                expected_comment=str(binding["receipt_row"]["comment"]),
                expected_job_name=str(binding["manifest_row"]["job_name"]),
                scheduler_runner=scheduler_runner,
                boundary_seconds=boundary_seconds,
            )
            verified = binding["verified"]
            archive_errors: list[str] = []
            for archive_attempt in range(1, 4):
                fleet = _collect_fleet_capacity_transient(
                    control,
                    state_dir=state_dir,
                    scheduler_runner=scheduler_runner,
                    probe=probe,
                    probe_timeout=probe_timeout,
                    now=now,
                )
                try:
                    archive = _build_capacity_preimage_archive(
                        parent=output.parent,
                        fleet=fleet,
                        chain_id=str(verified["manifest"]["chain_id"]),
                        chain_generation=int(binding["chain_generation"]),
                        readiness_job_id=readiness_job_id,
                        scheduler_runner=scheduler_runner,
                    )
                except EvidenceError as exc:
                    archive_errors.append(
                        f"attempt {archive_attempt}: {exc}"
                    )
                    if archive_attempt == 3:
                        raise EvidenceError(
                            "capacity preimage capture did not stabilize after "
                            "three complete live reconciliations: "
                            + " | ".join(archive_errors)
                        ) from exc
                    continue
                break
            timestamp = float(now())
            if not math.isfinite(timestamp):
                raise EvidenceError("capacity-transient clock is non-finite")
            receipt = verified["submission_receipt"]
            evidence = {
                "schema_version": 1,
                "protocol": CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL,
                "passed": True,
                "chain_protocol": R8_PROTOCOL,
                "chain_id": verified["manifest"]["chain_id"],
                "chain_generation": binding["chain_generation"],
                "manifest": verified["manifest_path"],
                "manifest_sha256": verified["manifest_sha256"],
                "submission_receipt": verified["submission_receipt_path"],
                "submission_receipt_sha256": verified["submission_receipt_sha256"],
                "submission_receipt_id": receipt["receipt_id"],
                "fleet_readiness_job_id": readiness_job_id,
                "fleet_readiness_comment": binding["receipt_row"]["comment"],
                "boundary_seconds": boundary_seconds,
                "boundary_proof": boundary,
                "fleet": fleet,
                "preimage_archive": archive,
                "observed_at": _utc(timestamp),
                "observed_timestamp": timestamp,
            }
            evidence["evidence_id"] = _identity_sha256(evidence)
            _write_json_atomic(evidence_path, evidence)
            evidence = _validate_capacity_transient_evidence(
                evidence_path, binding=binding
            )
        timestamp = float(now())
        if (
            not math.isfinite(timestamp)
            or timestamp < float(evidence["observed_timestamp"])
            or timestamp - float(evidence["observed_timestamp"])
            > CAPACITY_TRANSIENT_MAX_EVIDENCE_AGE_SECONDS
        ):
            _archive_stale_capacity_evidence(evidence_path, evidence)
            raise EvidenceError(
                "capacity-transient evidence became stale before marker "
                "publication; its preimage was archived and live proof must rerun"
            )
        marker = {
            "schema_version": 1,
            "protocol": CAPACITY_TRANSIENT_PROTOCOL,
            "passed": True,
            "capacity_transient_root": True,
            "chain_id": binding["verified"]["manifest"]["chain_id"],
            "chain_generation": binding["chain_generation"],
            "fleet_readiness_job_id": readiness_job_id,
            "fleet_readiness_comment": binding["receipt_row"]["comment"],
            "evidence": str(evidence_path.resolve()),
            "evidence_sha256": _sha256(evidence_path),
            "evidence_id": evidence["evidence_id"],
            "preimage_archive": evidence["preimage_archive"],
            "published_at": _utc(timestamp),
            "published_timestamp": timestamp,
        }
        marker["receipt_id"] = _identity_sha256(marker)
        _write_json_atomic(output, marker)
        validated = validate_capacity_transient_receipt(
            output,
            chain_manifest=chain_manifest,
            submission_receipt=submission_receipt,
            readiness_job_id=readiness_job_id,
        )
        return validated | {"status": "complete"}


CONTEXT_SPECS: dict[str, dict[str, Any]] = {
    "dense_peer_context_audit": {
        "source_name": "raw_dense_peer_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["b2048", "b8192", "unlimited"],
            "prompt_levels": [3],
            "topologies": ["decentralized"],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": EXPECTED_DENSE_CELLS,
        "audited_requests": EXPECTED_DENSE_REQUESTS,
    },
    "seven_agent_context_audit": {
        "source_name": "raw_seven_agent_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["unlimited"],
            "prompt_levels": [],
            "topologies": [],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": EXPECTED_SEVEN_CELLS,
        "audited_requests": EXPECTED_SEVEN_REQUESTS,
    },
}


def _run_pin(control: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    matches = [run for run in control["immutable"]["runs"] if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise EvidenceError(f"immutable pins do not uniquely define run {run_id}")
    return matches[0]


def _recompute_context_source(
    *,
    snapshot: Any,
    catalog: VerifiedQuestionCatalog,
    benchmark_loader: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-render every selected request from frozen inputs.

    A source report is not an authority merely because it is checksummed.  Rebuilding
    it here independently binds routing, prompt bytes, chat-template tokenization,
    peer-context truncation, requested output, and every per-cell/aggregate margin to
    the exact manifest and benchmark contracts accepted by the control plane.
    """

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    filters = AuditFilters(
        n_agents=frozenset(spec["filters"]["n_agents"]),
        reasoning=frozenset(spec["filters"]["reasoning"]),
        prompt_levels=frozenset(spec["filters"]["prompt_levels"]),
        topologies=frozenset(spec["filters"]["topologies"]),
        context_levels=frozenset(spec["filters"]["context_levels"]),
    )
    return context_audit.audit_snapshot(
        snapshot,
        run_id=str(spec["run_id"]),
        filters=filters,
        question_catalog=catalog,
        benchmark_loader=benchmark_loader,
        failure_example_limit=100,
        all_routed_profiles=True,
        prompt_root=REPO / "configs" / "prompts",
    )


def _validate_context_source(
    control: Mapping[str, Any],
    *,
    name: str,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = CONTEXT_SPECS[name]
    report = _read_json(path)
    run_pin = _run_pin(control, spec["run_id"])
    required_report_fields = {
        "schema_version",
        "audit",
        "run_id",
        "manifest",
        "benchmark_contracts",
        "filters",
        "all_routed_profiles",
        "assumptions",
        "summary",
        "groups",
        "failure_groups",
        "failure_examples",
        "cells",
    }
    if (
        report.get("schema_version") != 3
        or report.get("audit") != "all_routed_profiles_context_capacity"
        or report.get("run_id") != spec["run_id"]
        or report.get("all_routed_profiles") is not True
        or report.get("filters") != spec["filters"]
    ):
        raise EvidenceError(f"context audit identity/filters drifted: {path}")
    manifest = report.get("manifest")
    if not isinstance(manifest, dict) or manifest != {
        "path": str(Path(str(run_pin["manifest_path"])).resolve()),
        "sha256": run_pin["manifest_sha256"],
        "cells": run_pin["cell_count"],
    }:
        raise EvidenceError(f"context audit manifest identity drifted: {path}")
    benchmark_contracts = report.get("benchmark_contracts")
    if not isinstance(benchmark_contracts, dict) or benchmark_contracts != {
        "path": str(
            Path(str(run_pin["benchmark_contract_path"])).resolve()
        ),
        "sha256": run_pin["benchmark_contract_sha256"],
        "verified": True,
    }:
        raise EvidenceError(
            f"context audit frozen benchmark-contract identity drifted: {path}"
        )
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise EvidenceError(f"context audit summary is missing: {path}")
    expected_summary = {
        "passed": True,
        "selected_cells": spec["selected_cells"],
        "audited_requests": spec["audited_requests"],
        "failed_requests": 0,
        "failed_cells": 0,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise EvidenceError(
                f"context audit {name} {field} drifted: "
                f"expected {expected!r}, observed {summary.get(field)!r}"
            )
    source_margin = summary.get("minimum_context_headroom_tokens")
    if (
        not isinstance(source_margin, int)
        or isinstance(source_margin, bool)
        or source_margin < EXPECTED_CONTEXT_MARGIN
    ):
        raise EvidenceError(
            f"context audit {name} minimum_context_headroom_tokens is below "
            f"{EXPECTED_CONTEXT_MARGIN}: observed {source_margin!r}"
        )
    if report.get("failure_groups") != [] or report.get("failure_examples") != []:
        raise EvidenceError(f"context audit {name} retains failure evidence")
    if set(report) != required_report_fields:
        raise EvidenceError(f"context audit {name} fields drifted: {path}")

    snapshot = load_manifest(Path(str(run_pin["run_root"])), verify_frozen=True)
    filters = AuditFilters(
        n_agents=frozenset(spec["filters"]["n_agents"]),
        reasoning=frozenset(spec["filters"]["reasoning"]),
        prompt_levels=frozenset(spec["filters"]["prompt_levels"]),
        topologies=frozenset(spec["filters"]["topologies"]),
        context_levels=frozenset(spec["filters"]["context_levels"]),
    )
    selected = selected_cells(snapshot, filters, all_routed_profiles=True)
    if len(selected) != spec["selected_cells"]:
        raise EvidenceError(f"manifest-derived context selection drifted for {name}")
    expected_ids = {cell.cell_id for cell in selected}
    rows = report.get("cells")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise EvidenceError(f"context audit cell records are incomplete for {name}")
    observed_ids = [row.get("cell_id") if isinstance(row, dict) else None for row in rows]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise EvidenceError(f"context audit cell identities drifted for {name}")

    benchmark_loader = context_audit.CacheOnlyBenchmarkLoader()
    catalog = VerifiedQuestionCatalog(
        Path(str(run_pin["run_root"])),
        snapshot=snapshot,
        benchmark_loader=benchmark_loader,
    )
    selected_by_id = {cell.cell_id: cell for cell in selected}
    request_total = 0
    margins: list[int] = []
    request_truncations = 0
    for row in rows:
        assert isinstance(row, dict)
        cell = selected_by_id[str(row["cell_id"])]
        expected_requests = len(catalog.questions_for(cell))
        expected_question_contract = catalog.frozen.contract_for_cell(cell)[
            "question_contract_sha256"
        ]
        margin = row.get("minimum_context_headroom_tokens")
        if (
            row.get("config_hash") != cell.config_hash()
            or row.get("question_contract_sha256")
            != expected_question_contract
            or row.get("audited_requests") != expected_requests
            or row.get("failed_requests") != 0
            or row.get("fits") is not True
            or not isinstance(margin, int)
            or isinstance(margin, bool)
            or margin <= 0
        ):
            raise EvidenceError(f"invalid context cell evidence for {cell.cell_id}")
        minimum_request = row.get("minimum_headroom_request")
        if (
            not isinstance(minimum_request, dict)
            or minimum_request.get("context_preflight_fits") is not True
            or minimum_request.get("output_capacity_headroom_tokens") != margin
        ):
            raise EvidenceError(f"context request evidence drifted for {cell.cell_id}")
        if minimum_request.get("context_preflight_fits") is not True:
            request_truncations += 1
        request_total += expected_requests
        margins.append(margin)
    if (
        request_total != spec["audited_requests"]
        or min(margins) < EXPECTED_CONTEXT_MARGIN
        or request_truncations != 0
    ):
        raise EvidenceError(f"context audit derived aggregate drifted for {name}")
    rebuilt = _recompute_context_source(
        snapshot=snapshot,
        catalog=catalog,
        benchmark_loader=benchmark_loader,
        spec=spec,
    )
    if report != rebuilt:
        raise EvidenceError(
            f"context audit {name} is not the exact independently rerendered "
            "frozen-input report"
        )
    wrapper_metrics = {
        "selected_cells": spec["selected_cells"],
        "audited_requests": spec["audited_requests"],
        "failed_requests": 0,
        "failed_cells": 0,
        "minimum_context_headroom_tokens": min(margins),
        "truncation_incidents": 0,
    }
    return report, wrapper_metrics


def build_context_gate(
    control: Mapping[str, Any],
    *,
    dense_peer_audit: Path,
    seven_agent_audit: Path,
    output: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    inputs = {
        "dense_peer_context_audit": dense_peer_audit,
        "seven_agent_context_audit": seven_agent_audit,
    }
    artifacts: list[dict[str, str]] = []
    total_failures = total_truncations = 0
    margins: list[int] = []
    for name, path in inputs.items():
        _, metrics = _validate_context_source(control, name=name, path=path)
        spec = CONTEXT_SPECS[name]
        wrapper_path = _artifact_path(output, name)
        _write_json_atomic(
            wrapper_path,
            _wrapper(
                name=name,
                immutable_sha256=str(control["immutable_sha256"]),
                metrics=metrics,
                references=[_artifact(str(spec["source_name"]), path)],
                extra={
                    "run_id": spec["run_id"],
                    "filters": spec["filters"],
                    "all_routed_profiles": True,
                },
            ),
        )
        artifacts.append(_artifact(name, wrapper_path))
        total_failures += int(metrics["failed_requests"])
        total_truncations += int(metrics["truncation_incidents"])
        margins.append(int(metrics["minimum_context_headroom_tokens"]))
    outer_metrics = {
        "dense_peer_requests": EXPECTED_DENSE_REQUESTS,
        "seven_agent_requests": EXPECTED_SEVEN_REQUESTS,
        "failed_preflights": total_failures,
        "truncation_incidents": total_truncations,
        "minimum_context_margin_tokens": min(margins),
    }
    return _publish_outer(
        control=control,
        gate="context_audit",
        metrics=outer_metrics,
        artifacts=artifacts,
        output=output,
        state_dir=state_dir,
    )


def build_email_request(
    control: Mapping[str, Any],
    *,
    output: Path,
    chain_id: str,
    challenge_generation: int,
    release_tag: str,
    release_git_commit: str,
    acknowledgement_script: Path,
    apply: bool,
    mail_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: Callable[[], float] = time.time,
    random_bytes: Callable[[int], bytes] = secrets.token_bytes,
    sleeper: Callable[[float], None] = time.sleep,
    retry_delays: Sequence[float] = (0.0, 5.0, 20.0, 60.0),
    challenge_ttl_seconds: float = 28_800.0,
    acknowledgement_python: Path | None = None,
) -> dict[str, Any]:
    """Deliver and activate a cryptographically random one-time challenge.

    The raw token exists only in this live process and the submitted mail body.
    Delivery retries reuse it only inside this call.  A later invocation creates a
    fresh request, then atomically switches ``output`` to supersede the old pending
    request.  A consumed challenge can never be superseded or replayed.
    """

    if not apply:
        raise EvidenceError("email request requires --apply; no synthetic receipt is valid")
    recipient = control.get("alert_email")
    immutable = control.get("immutable_sha256")
    immutable_pins = control.get("immutable")
    if (
        not isinstance(recipient, str)
        or not recipient
        or any(character in recipient for character in "\r\n")
    ):
        raise EvidenceError("control alert_email is missing")
    if (
        not isinstance(immutable, str)
        or re.fullmatch(r"[0-9a-f]{64}", immutable) is None
    ):
        raise EvidenceError("control immutable identity is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", chain_id) is None:
        raise EvidenceError("email challenge chain ID is invalid")
    if (
        not isinstance(challenge_generation, int)
        or isinstance(challenge_generation, bool)
        or challenge_generation < 0
    ):
        raise EvidenceError("email challenge generation is invalid")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", release_tag) is None:
        raise EvidenceError("email challenge release tag is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", release_git_commit) is None:
        raise EvidenceError("email challenge release commit is invalid")
    if isinstance(immutable_pins, Mapping) and (
        immutable_pins.get("git_commit") != release_git_commit
    ):
        raise EvidenceError(
            "email challenge release commit differs from immutable control"
        )
    if not retry_delays or len(retry_delays) > 8 or any(
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(float(delay))
        or float(delay) < 0
        or float(delay) > 300
        for delay in retry_delays
    ):
        raise EvidenceError("email delivery retry policy is invalid")
    if (
        isinstance(challenge_ttl_seconds, bool)
        or not isinstance(challenge_ttl_seconds, (int, float))
        or not math.isfinite(float(challenge_ttl_seconds))
        or not 600 <= float(challenge_ttl_seconds) <= 86_400
    ):
        raise EvidenceError("email challenge lifetime is invalid")

    output = output.expanduser().absolute()
    root = output.parent
    try:
        schema5_email_ack.require_safe_directory(
            root.parent, description="email challenge parent"
        )
        if not root.exists() and not root.is_symlink():
            root.mkdir(mode=0o750)
        schema5_email_ack.require_safe_directory(
            root, description="email challenge root"
        )
    except (schema5_email_ack.AcknowledgementError, OSError) as exc:
        raise EvidenceError(f"email challenge root is unsafe: {exc}") from exc
    if (
        output.name != "CURRENT.json"
        or output.is_symlink()
    ):
        raise EvidenceError(
            "email active-challenge output must be a safe CURRENT.json path"
        )
    try:
        schema5_email_ack.require_safe_directory(
            acknowledgement_script.expanduser().absolute().parent,
            description="schema-5 email acknowledgement tool parent",
        )
        acknowledgement_script = _require_read_only_regular(
            acknowledgement_script,
            description="schema-5 email acknowledgement tool",
        )
    except schema5_email_ack.AcknowledgementError as exc:
        raise EvidenceError(str(exc)) from exc
    if acknowledgement_python is None:
        acknowledgement_python = Path(sys.executable)
    acknowledgement_python = acknowledgement_python.expanduser().absolute()
    try:
        schema5_email_ack.require_safe_directory(
            acknowledgement_python.parent,
            description="email acknowledgement interpreter parent",
        )
        resolved_acknowledgement_python = acknowledgement_python.resolve(
            strict=True
        )
        schema5_email_ack.require_safe_directory(
            resolved_acknowledgement_python.parent,
            description="resolved email acknowledgement interpreter parent",
        )
    except (schema5_email_ack.AcknowledgementError, OSError) as exc:
        raise EvidenceError(
            f"email acknowledgement interpreter is unsafe: {exc}"
        ) from exc
    if (
        not resolved_acknowledgement_python.is_file()
        or resolved_acknowledgement_python
        != Path(sys.executable).resolve(strict=True)
        or any(character in str(acknowledgement_python) for character in "\r\n")
    ):
        raise EvidenceError(
            "email acknowledgement interpreter differs from the running "
            "immutable harness"
        )

    def secure_blob(description: str) -> bytes:
        try:
            value = random_bytes(32)
        except Exception as exc:
            raise EvidenceError(
                f"cannot generate {description} entropy"
            ) from exc
        if not isinstance(value, bytes) or len(value) != 32:
            raise EvidenceError(
                f"{description} entropy source must return exactly 32 bytes"
            )
        return value

    try:
        lock_context = schema5_email_ack.challenge_lock(output)
        lock_context.__enter__()
    except (schema5_email_ack.AcknowledgementError, OSError) as exc:
        raise EvidenceError(f"cannot acquire email challenge lock: {exc}") from exc
    try:
        prior: dict[str, Any] | None = None
        prior_request: dict[str, Any] | None = None
        prior_request_path: Path | None = None
        if output.exists():
            prior = _read_json(output)
            try:
                prior_request_path = Path(str(prior["request"])).expanduser().absolute()
                prior_request = _read_json(prior_request_path)
                schema5_email_ack.validate_request_payload(prior_request)
                schema5_email_ack.validate_active_challenge(
                    prior,
                    active_path=output,
                    request=prior_request,
                    request_path=prior_request_path,
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                schema5_email_ack.AcknowledgementError,
            ) as exc:
                raise EvidenceError(
                    "existing active email challenge is invalid"
                ) from exc
            if (
                prior["immutable_sha256"] != immutable
                or prior["chain_id"] != chain_id
                or prior["release_tag"] != release_tag
                or prior["release_git_commit"] != release_git_commit
                or prior["recipient"] != recipient
            ):
                raise EvidenceError(
                    "existing active email challenge belongs to another chain "
                    "or immutable release"
                )
            if int(prior["challenge_generation"]) > challenge_generation:
                raise EvidenceError(
                    "email challenge generation rollback is forbidden"
                )
            prior_ack = Path(str(prior["acknowledgement"])).expanduser().absolute()
            if prior_ack.exists() or prior_ack.is_symlink():
                if int(prior["challenge_generation"]) != challenge_generation:
                    raise EvidenceError(
                        "a consumed email challenge cannot be reused by a repair "
                        "generation"
                    )
                return prior | {
                    "status": "already_acknowledged",
                    "path": str(output),
                    "sha256": _sha256(output),
                }

        token = base64.urlsafe_b64encode(
            secure_blob("email token")
        ).rstrip(b"=").decode("ascii")
        salt = secure_blob("email salt").hex()
        nonce = secure_blob("email nonce").hex()
        request_entropy = secure_blob("email request").hex()
        request_id = hashlib.sha256(
            (
                "schema5-email-request-v2\0"
                f"{immutable}\0{chain_id}\0{release_tag}\0"
                f"{release_git_commit}\0{recipient}\0{challenge_generation}\0"
                f"{request_entropy}"
            ).encode("utf-8")
        ).hexdigest()
        generation_root = root / f"g{challenge_generation:04d}"
        request_path = generation_root / "requests" / f"{request_id}.json"
        acknowledgement_path = (
            generation_root / "acknowledgements" / f"{request_id}.json"
        )
        try:
            for directory in (
                generation_root,
                request_path.parent,
                acknowledgement_path.parent,
            ):
                schema5_email_ack.require_safe_directory(
                    directory.parent,
                    description=f"{directory.name} parent",
                )
                if not directory.exists() and not directory.is_symlink():
                    directory.mkdir(mode=0o750)
                schema5_email_ack.require_safe_directory(
                    directory,
                    description=f"email challenge directory {directory.name}",
                )
        except (schema5_email_ack.AcknowledgementError, OSError) as exc:
            raise EvidenceError(
                f"email challenge generation ancestry is unsafe: {exc}"
            ) from exc
        if (
            request_path.exists()
            or request_path.is_symlink()
            or acknowledgement_path.exists()
            or acknowledgement_path.is_symlink()
        ):
            raise EvidenceError("fresh email request identity unexpectedly exists")

        request_binding: dict[str, Any] = {
            "immutable_sha256": immutable,
            "chain_id": chain_id,
            "request_id": request_id,
            "release_tag": release_tag,
            "release_git_commit": release_git_commit,
            "recipient": recipient,
            "challenge_generation": challenge_generation,
            "challenge_nonce": nonce,
            "challenge_salt": salt,
        }
        request_binding["challenge_id"] = schema5_email_ack.challenge_id_for(
            request_binding
        )
        acknowledgement_command = shlex.join(
            [
                str(acknowledgement_python),
                "-I",
                str(acknowledgement_script),
                "--request",
                str(request_path),
                "--output",
                str(acknowledgement_path),
                "--token",
                token,
                "--apply",
            ]
        )
        body = (
            f"The schema-5 production chain {release_tag} requires proof that this alert "
            "was received. The challenge is valid only for the exact immutable "
            "identity below and may be consumed once.\n\n"
            f"One-time token:\n{token}\n\n"
            f"Run exactly once:\n{acknowledgement_command}\n\n"
            f"Chain ID: {chain_id}\n"
            f"Request ID: {request_id}\n"
            f"Release tag: {release_tag}\n"
            f"Release commit: {release_git_commit}\n"
            f"Challenge generation: {challenge_generation}\n"
            f"Recipient: {recipient}\n"
            f"Immutable control: {immutable}\n"
        )
        attempts: list[dict[str, Any]] = []
        delivered = False
        for attempt, delay in enumerate(retry_delays, start=1):
            if float(delay):
                sleeper(float(delay))
            timed_out = False
            try:
                proc = mail_runner(
                    [
                        "mail",
                        "-s",
                        schema5_email_ack.EMAIL_CHALLENGE_SUBJECT,
                        recipient,
                    ],
                    input=body,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30.0,
                )
                returncode = int(proc.returncode)
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = 124
            except OSError:
                returncode = 127
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": returncode,
                    "timed_out": timed_out,
                }
            )
            if returncode == 0 and not timed_out:
                delivered = True
                break
        if not delivered:
            # No durable request is published: the plaintext token dies with this
            # process and a repair must create a fresh challenge.
            raise EvidenceError(
                "mail submission did not succeed within the bounded retry policy"
            )

        submitted = float(now())
        request_payload = {
            "schema_version": schema5_email_ack.EMAIL_CHALLENGE_SCHEMA_VERSION,
            "kind": "schema5_email_ack_request",
            "passed": True,
            **request_binding,
            "challenge_verifier_algorithm": (
                schema5_email_ack.EMAIL_CHALLENGE_VERIFIER_ALGORITHM
            ),
            "challenge_verifier": schema5_email_ack.challenge_verifier_for(
                token, request_binding
            ),
            "active_challenge": str(output.resolve()),
            "acknowledgement": str(acknowledgement_path.resolve()),
            "delivery_succeeded": True,
            "returncode": 0,
            "delivery_attempts": attempts,
            "submitted_timestamp": submitted,
            "expires_timestamp": submitted + float(challenge_ttl_seconds),
            "subject": schema5_email_ack.EMAIL_CHALLENGE_SUBJECT,
            "body_template": schema5_email_ack.EMAIL_CHALLENGE_BODY_TEMPLATE,
            "acknowledgement_tool": {
                "path": str(acknowledgement_script),
                "sha256": _sha256(acknowledgement_script),
            },
        }
        try:
            schema5_email_ack.validate_request_payload(request_payload)
        except schema5_email_ack.AcknowledgementError as exc:
            raise EvidenceError(f"generated email request is invalid: {exc}") from exc
        _write_json_atomic(request_path, request_payload)

        supersedes = None
        if prior is not None and prior_request_path is not None:
            supersedes = {
                "challenge_generation": prior["challenge_generation"],
                "challenge_id": prior["challenge_id"],
                "request_id": prior["request_id"],
                "request": str(prior_request_path.resolve()),
                "request_sha256": _sha256(prior_request_path),
            }
        active_stable = {
            "schema_version": schema5_email_ack.EMAIL_CHALLENGE_SCHEMA_VERSION,
            "kind": "schema5_email_active_challenge",
            "passed": True,
            "immutable_sha256": immutable,
            "chain_id": chain_id,
            "request_id": request_id,
            "release_tag": release_tag,
            "release_git_commit": release_git_commit,
            "recipient": recipient,
            "challenge_generation": challenge_generation,
            "challenge_id": request_payload["challenge_id"],
            "request": str(request_path.resolve()),
            "request_sha256": _sha256(request_path),
            "acknowledgement": str(acknowledgement_path.resolve()),
            "supersedes": supersedes,
            "activated_timestamp": submitted,
        }
        active = active_stable | {
            "active_challenge_id": schema5_email_ack.identity_sha256(
                active_stable
            )
        }
        try:
            schema5_email_ack.validate_active_challenge(
                active,
                active_path=output,
                request=request_payload,
                request_path=request_path,
            )
        except schema5_email_ack.AcknowledgementError as exc:
            raise EvidenceError(
                f"generated active email challenge is invalid: {exc}"
            ) from exc
        # This replace is the supersession commit point.  Before it, only the old
        # request is active; after it, the old emailed token can no longer consume.
        _write_json_atomic(output, active)
        return active | {
            "status": "requested",
            "path": str(output),
            "sha256": _sha256(output),
        }
    finally:
        lock_context.__exit__(None, None, None)


def build_email_gate(
    control: Mapping[str, Any],
    *,
    output: Path,
    active_challenge: Path,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    active_challenge = active_challenge.expanduser().absolute()
    try:
        schema5_email_ack.require_safe_directory(
            active_challenge.parent, description="email challenge root"
        )
    except schema5_email_ack.AcknowledgementError as exc:
        raise EvidenceError(str(exc)) from exc
    active_payload = _read_json(active_challenge)
    try:
        request = Path(str(active_payload["request"])).expanduser().absolute()
        acknowledgement = Path(
            str(active_payload["acknowledgement"])
        ).expanduser().absolute()
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("active email challenge paths are invalid") from exc
    request_payload = _read_json(request)
    ack_payload = _read_json(acknowledgement)
    immutable = control.get("immutable_sha256")
    recipient = control.get("alert_email")
    try:
        schema5_email_ack.validate_request_payload(request_payload)
        schema5_email_ack.validate_active_challenge(
            active_payload,
            active_path=active_challenge,
            request=request_payload,
            request_path=request,
        )
    except schema5_email_ack.AcknowledgementError as exc:
        raise EvidenceError(f"email request identity is invalid: {exc}") from exc
    if (
        request_payload.get("immutable_sha256") != immutable
        or request_payload.get("recipient") != recipient
        or request_payload.get("acknowledgement") != str(acknowledgement.resolve())
    ):
        raise EvidenceError("email request differs from immutable control")
    tool = request_payload["acknowledgement_tool"]
    try:
        tool_path = _require_read_only_regular(
            Path(tool["path"]),
            description="email acknowledgement tool",
        )
        schema5_email_ack.require_safe_directory(
            tool_path.parent, description="email acknowledgement tool parent"
        )
    except (EvidenceError, schema5_email_ack.AcknowledgementError) as exc:
        raise EvidenceError(f"email acknowledgement tool is unsafe: {exc}") from exc
    if _sha256(tool_path) != tool["sha256"]:
        raise EvidenceError("email acknowledgement tool hash drifted")
    try:
        schema5_email_ack.validate_acknowledgement(
            ack_payload,
            acknowledgement_path=acknowledgement,
            request=request_payload,
            request_path=request,
        )
    except schema5_email_ack.AcknowledgementError as exc:
        raise EvidenceError(
            f"email acknowledgement identity is invalid: {exc}"
        ) from exc
    receipt_metrics = {
        "recipient": recipient,
        "delivery_succeeded": True,
        "returncode": 0,
        "acknowledged": True,
        "chain_id": request_payload["chain_id"],
        "request_id": request_payload["request_id"],
        "release_tag": request_payload["release_tag"],
        "release_git_commit": request_payload["release_git_commit"],
        "challenge_generation": request_payload["challenge_generation"],
        "challenge_id": request_payload["challenge_id"],
        "challenge_verifier": request_payload["challenge_verifier"],
        "acknowledged_at": ack_payload["acknowledged_at"],
    }
    name = "email_delivery_receipt"
    wrapper_path = _artifact_path(output, name)
    _write_json_atomic(
        wrapper_path,
        _wrapper(
            name=name,
            immutable_sha256=str(control["immutable_sha256"]),
            metrics=receipt_metrics,
            references=[
                _artifact("active_email_challenge", active_challenge),
                _artifact("email_ack_request", request),
                _artifact("email_acknowledgement", acknowledgement),
            ],
        ),
    )
    return _publish_outer(
        control=control,
        gate="email_test",
        metrics=receipt_metrics,
        artifacts=[_artifact(name, wrapper_path)],
        output=output,
        state_dir=state_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="gate", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--pre-repair-attestation", required=True, type=Path)
    snapshot.add_argument("--legacy-consolidated-attestation", required=True, type=Path)

    for name in ("migrations", "semantic-audit"):
        command = subparsers.add_parser(name)
        command.add_argument("--cleanup-marker", required=True, type=Path)

    fleet = subparsers.add_parser("fleet")
    fleet.add_argument("--probe-timeout", type=float, default=5.0)
    capacity = subparsers.add_parser("fleet-capacity-transient")
    capacity.add_argument("--probe-timeout", type=float, default=10.0)
    capacity.add_argument(
        "--boundary-seconds",
        type=int,
        default=CAPACITY_TRANSIENT_BOUNDARY_SECONDS,
    )
    capacity.add_argument("--chain-manifest", required=True, type=Path)
    capacity.add_argument("--submission-receipt", required=True, type=Path)
    capacity.add_argument("--readiness-job-id", required=True)

    context = subparsers.add_parser("context-audit")
    context.add_argument("--dense-peer-audit", required=True, type=Path)
    context.add_argument("--seven-agent-audit", required=True, type=Path)

    email_request = subparsers.add_parser("email-request")
    email_request.add_argument("--chain-id", required=True)
    email_request.add_argument("--challenge-generation", required=True, type=int)
    email_request.add_argument("--release-tag", required=True)
    email_request.add_argument("--release-git-commit", required=True)
    email_request.add_argument(
        "--acknowledgement-script", required=True, type=Path
    )
    email_request.add_argument("--acknowledgement-python", type=Path)
    email_request.add_argument("--apply", action="store_true")
    email = subparsers.add_parser("email-test")
    email.add_argument("--active-challenge", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        current = control_plane.load_control(
            args.state_dir.expanduser().resolve(), verify_files=True
        )
        if args.gate == "snapshot":
            report = build_snapshot_gate(
                current,
                pre_repair_attestation=args.pre_repair_attestation,
                legacy_consolidated_attestation=args.legacy_consolidated_attestation,
                output=args.output,
                state_dir=args.state_dir,
            )
        elif args.gate == "migrations":
            report = build_migrations_gate(
                current,
                cleanup_marker=args.cleanup_marker,
                output=args.output,
                state_dir=args.state_dir,
            )
        elif args.gate == "semantic-audit":
            report = build_semantic_gate(
                current,
                cleanup_marker=args.cleanup_marker,
                output=args.output,
                state_dir=args.state_dir,
            )
        elif args.gate == "fleet":
            report = build_fleet_gate(
                current,
                output=args.output,
                state_dir=args.state_dir,
                probe_timeout=args.probe_timeout,
            )
        elif args.gate == "fleet-capacity-transient":
            report = build_fleet_capacity_transient_receipt(
                current,
                output=args.output,
                state_dir=args.state_dir,
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                readiness_job_id=args.readiness_job_id,
                boundary_seconds=args.boundary_seconds,
                probe_timeout=args.probe_timeout,
            )
        elif args.gate == "context-audit":
            report = build_context_gate(
                current,
                dense_peer_audit=args.dense_peer_audit,
                seven_agent_audit=args.seven_agent_audit,
                output=args.output,
                state_dir=args.state_dir,
            )
        elif args.gate == "email-request":
            report = build_email_request(
                current,
                output=args.output,
                chain_id=args.chain_id,
                challenge_generation=args.challenge_generation,
                release_tag=args.release_tag,
                release_git_commit=args.release_git_commit,
                acknowledgement_script=args.acknowledgement_script,
                acknowledgement_python=args.acknowledgement_python,
                apply=args.apply,
            )
        elif args.gate == "email-test":
            report = build_email_gate(
                current,
                output=args.output,
                active_challenge=args.active_challenge,
                state_dir=args.state_dir,
            )
        else:  # pragma: no cover - argparse closes the command set
            raise AssertionError(args.gate)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "type": type(exc).__name__, "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
