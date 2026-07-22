#!/usr/bin/env python3
"""Evidence-preservingly clear three obsolete checkpoint-schema failure ledgers.

After the schema-1 QID checkpoints have been migrated to schema 2, three cells in
``full_sweep_v1`` still carry the permanent ``ExperimentConfigurationError`` raised by
the old executable.  This tool has an exact cell/config allowlist and will clear no
other failure.  It requires every checkpoint in the target cell to be schema 2 and the
active failure ledger to match the obsolete error contract exactly.

Dry-run is the default and reports exact before/after hashes.  ``--apply`` acquires the
normal nonblocking cell lock, archives the complete ``failure.json`` preimage with an
incident record and hashes, durably removes only that exact active preimage, and writes
``reset_complete.json`` last.  An interrupted transaction is resumable; a completed
incident never touches a later failure ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402
from agents_scaling.experiment import io  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    FAILURE_FILENAME,
    CellLockUnavailable,
    cell_lock,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.experiment.qid_checkpoint import (  # noqa: E402
    CHECKPOINT_DIRECTORY,
    CHECKPOINT_SCHEMA_VERSION,
)


INCIDENT_SCHEMA_VERSION = 1
INCIDENT_KIND = "checkpoint_schema_permanent_failure_v1"
INCIDENT_ROOT = Path("incidents") / INCIDENT_KIND
PREIMAGE_FILENAME = "failure.preimage.json"
INCIDENT_FILENAME = "incident.json"
RESET_MARKER_FILENAME = "reset_complete.json"
TARGET_RUN_ID = "full_sweep_v1"
TARGET_CELL_CONFIG_HASHES: dict[str, str] = {
    "32B_centralized_artifact_only_p3_rb8192_mmlu_pro_s1": "c9fd4387afaf",
    "4B_single_agent_artifact_only_p3_runlimited_math_s2": "e66fd8fceb9a",
    "1.7B_single_agent_artifact_only_p3_roff_gpqa_s2": "79a9fd591269",
}
_CHECKPOINT_NAME_RE = re.compile(r"[0-9a-f]{64}\.json")
_OBSOLETE_MESSAGE_RE = re.compile(
    r"^durable QID checkpoint failed closed; refusing replacement sampling: "
    r"durable QID checkpoint .+ has the wrong root schema$"
)


class PermanentFailureArchiveError(RuntimeError):
    """The tool cannot prove that a narrow archive/reset is safe."""


class _InvalidJSON(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _InvalidJSON(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise _InvalidJSON(f"non-finite JSON number {value!r}")


def _strict_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, _InvalidJSON) as exc:
        raise PermanentFailureArchiveError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PermanentFailureArchiveError(f"{label} must contain one JSON object")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermanentFailureArchiveError(f"missing or unsafe {label}: {path}")
    return path.read_bytes()


def _validate_failure(
    payload: bytes, *, cell_id: str, config_hash: str
) -> dict[str, Any]:
    value = _strict_object(payload, label=f"failure ledger for {cell_id}")
    expected = {
        "schema_version": 2,
        "cell_id": cell_id,
        "config_hash": config_hash,
        "classification": "configuration",
        "disposition": "permanent",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise PermanentFailureArchiveError(
                f"{cell_id} failure {field} is {value.get(field)!r}, expected {wanted!r}"
            )
    error = value.get("last_error")
    if (
        not isinstance(error, dict)
        or error.get("type") != "ExperimentConfigurationError"
        or not isinstance(error.get("message"), str)
        or _OBSOLETE_MESSAGE_RE.fullmatch(error["message"]) is None
    ):
        raise PermanentFailureArchiveError(
            f"{cell_id} failure is not the exact obsolete checkpoint-schema error"
        )
    checkpoint_fragment = f"/{cell_id}/{CHECKPOINT_DIRECTORY}/"
    if checkpoint_fragment not in error["message"]:
        raise PermanentFailureArchiveError(
            f"{cell_id} failure message points to a different cell/checkpoint"
        )
    return value


def _checkpoint_evidence(cell_dir: Path) -> list[dict[str, Any]]:
    checkpoint_root = cell_dir / CHECKPOINT_DIRECTORY
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise PermanentFailureArchiveError(
            f"migrated checkpoint directory is missing: {checkpoint_root}"
        )
    paths = sorted(checkpoint_root.iterdir(), key=lambda item: item.name)
    if not paths:
        raise PermanentFailureArchiveError(
            f"target cell has no migrated checkpoints: {checkpoint_root}"
        )
    evidence: list[dict[str, Any]] = []
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or _CHECKPOINT_NAME_RE.fullmatch(path.name) is None
        ):
            raise PermanentFailureArchiveError(f"unsafe checkpoint artifact: {path}")
        payload = path.read_bytes()
        value = _strict_object(payload, label=f"checkpoint {path}")
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise PermanentFailureArchiveError(
                f"checkpoint migration is incomplete for {path}: "
                f"schema={value.get('schema_version')!r}, expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        evidence.append(
            {
                "relative_path": path.relative_to(cell_dir).as_posix(),
                "sha256": _sha256(payload),
                "size": len(payload),
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
            }
        )
    return evidence


def _incident_payload(
    *,
    run_id: str,
    manifest_sha256: str,
    manifest_cell_count: int,
    cell_id: str,
    config_hash: str,
    failure_payload: bytes,
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "incident_schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_kind": INCIDENT_KIND,
        "run_id": run_id,
        "manifest": {
            "sha256": manifest_sha256,
            "cell_count": manifest_cell_count,
        },
        "cell": {"cell_id": cell_id, "config_hash": config_hash},
        "preimage": {
            "source_relative_path": f"cells/{cell_id}/{FAILURE_FILENAME}",
            "archived_filename": PREIMAGE_FILENAME,
            "sha256": _sha256(failure_payload),
            "size": len(failure_payload),
        },
        "checkpoints_after_migration": checkpoints,
        "mutation": {
            "before_sha256": _sha256(failure_payload),
            "after_sha256": None,
            "operation": "durable_remove_exact_archived_preimage",
        },
    }


def _validate_completed_archive(
    archive: Path,
    *,
    run_id: str,
    manifest_sha256: str,
    manifest_cell_count: int,
    cell_id: str,
    expected_config_hash: str,
) -> dict[str, Any]:
    preimage = _read_regular(archive / PREIMAGE_FILENAME, label="archived preimage")
    incident_bytes = _read_regular(archive / INCIDENT_FILENAME, label="incident record")
    incident = _strict_object(incident_bytes, label=f"incident for {cell_id}")
    marker = _strict_object(
        _read_regular(archive / RESET_MARKER_FILENAME, label="reset marker"),
        label=f"reset marker for {cell_id}",
    )
    _validate_failure(
        preimage, cell_id=cell_id, config_hash=expected_config_hash
    )
    preimage_sha256 = _sha256(preimage)
    expected_preimage = {
        "source_relative_path": f"cells/{cell_id}/{FAILURE_FILENAME}",
        "archived_filename": PREIMAGE_FILENAME,
        "sha256": preimage_sha256,
        "size": len(preimage),
    }
    expected_mutation = {
        "before_sha256": preimage_sha256,
        "after_sha256": None,
        "operation": "durable_remove_exact_archived_preimage",
    }
    if (
        set(incident)
        != {
            "incident_schema_version",
            "incident_kind",
            "run_id",
            "manifest",
            "cell",
            "preimage",
            "checkpoints_after_migration",
            "mutation",
        }
        or incident.get("incident_schema_version") != INCIDENT_SCHEMA_VERSION
        or incident.get("incident_kind") != INCIDENT_KIND
        or incident.get("run_id") != run_id
        or incident.get("manifest")
        != {"sha256": manifest_sha256, "cell_count": manifest_cell_count}
        or incident.get("cell")
        != {"cell_id": cell_id, "config_hash": expected_config_hash}
        or incident.get("preimage") != expected_preimage
        or incident.get("mutation") != expected_mutation
    ):
        raise PermanentFailureArchiveError(
            f"completed incident archive is inconsistent: {archive}"
        )
    checkpoints = incident.get("checkpoints_after_migration")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise PermanentFailureArchiveError(
            f"completed incident checkpoint evidence is invalid: {archive}"
        )
    observed_paths: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "relative_path",
            "sha256",
            "size",
            "schema_version",
        }:
            raise PermanentFailureArchiveError(
                f"completed incident checkpoint evidence is invalid: {archive}"
            )
        relative = checkpoint.get("relative_path")
        checksum = checkpoint.get("sha256")
        size = checkpoint.get("size")
        if (
            not isinstance(relative, str)
            or relative in observed_paths
            or Path(relative).parent.as_posix() != CHECKPOINT_DIRECTORY
            or _CHECKPOINT_NAME_RE.fullmatch(Path(relative).name) is None
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        ):
            raise PermanentFailureArchiveError(
                f"completed incident checkpoint evidence is invalid: {archive}"
            )
        observed_paths.add(relative)
    expected_marker = {
        "incident_schema_version": INCIDENT_SCHEMA_VERSION,
        "incident_kind": INCIDENT_KIND,
        "cell_id": cell_id,
        "preimage_sha256": preimage_sha256,
        "incident_sha256": _sha256(incident_bytes),
        "after_sha256": None,
        "active_failure_removed": True,
    }
    if marker != expected_marker:
        raise PermanentFailureArchiveError(
            f"completed reset marker is inconsistent: {archive}"
        )
    return {
        "cell_id": cell_id,
        "status": "already_reset",
        "before_sha256": preimage_sha256,
        "after_sha256": None,
        "archive": str(archive),
    }


def _publish_or_validate_archive(
    archive: Path,
    *,
    incident: Mapping[str, Any],
    failure_payload: bytes,
) -> tuple[bytes, bytes]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_symlink():
        raise PermanentFailureArchiveError(f"incident archive is a symlink: {archive}")
    if archive.exists() and not archive.is_dir():
        raise PermanentFailureArchiveError(f"incident archive is unsafe: {archive}")
    archive.mkdir(mode=0o700, exist_ok=True)
    os.chmod(archive, stat.S_IMODE(archive.stat().st_mode) | 0o700)
    incident_bytes = _json_text(incident).encode("utf-8")
    expected = {
        archive / PREIMAGE_FILENAME: failure_payload,
        archive / INCIDENT_FILENAME: incident_bytes,
    }
    for path, payload in expected.items():
        if path.exists():
            if _read_regular(path, label=path.name) != payload:
                raise PermanentFailureArchiveError(
                    f"conflicting incomplete incident preimage: {path}"
                )
        else:
            io.atomic_write_text(path, payload.decode("utf-8"))
        os.chmod(path, 0o444)
    return failure_payload, incident_bytes


def _apply_one(
    *,
    run_root: Path,
    manifest_sha256: str,
    manifest_cell_count: int,
    cell_id: str,
    config_hash: str,
) -> dict[str, Any]:
    cell_dir = run_root / "cells" / cell_id
    archive = run_root / INCIDENT_ROOT / cell_id
    marker = archive / RESET_MARKER_FILENAME
    failure_path = cell_dir / FAILURE_FILENAME
    if marker.exists():
        completed = _validate_completed_archive(
            archive,
            run_id=run_root.name,
            manifest_sha256=manifest_sha256,
            manifest_cell_count=manifest_cell_count,
            cell_id=cell_id,
            expected_config_hash=config_hash,
        )
        if failure_path.exists():
            raise PermanentFailureArchiveError(
                f"a new failure exists after completed incident for {cell_id}; refusing it"
            )
        return completed

    try:
        with cell_lock(cell_dir, blocking=False):
            if marker.exists():
                completed = _validate_completed_archive(
                    archive,
                    run_id=run_root.name,
                    manifest_sha256=manifest_sha256,
                    manifest_cell_count=manifest_cell_count,
                    cell_id=cell_id,
                    expected_config_hash=config_hash,
                )
                if failure_path.exists():
                    raise PermanentFailureArchiveError(
                        f"a new failure exists after completed incident for {cell_id}"
                    )
                return completed

            preimage_path = archive / PREIMAGE_FILENAME
            if failure_path.exists():
                failure_payload = _read_regular(failure_path, label="active failure")
            elif preimage_path.exists():
                # Recovery after the exact removal but before marker publication.
                failure_payload = _read_regular(preimage_path, label="archived preimage")
            else:
                raise PermanentFailureArchiveError(
                    f"failure ledger is absent with no incident preimage: {cell_id}"
                )
            _validate_failure(
                failure_payload, cell_id=cell_id, config_hash=config_hash
            )
            checkpoints = _checkpoint_evidence(cell_dir)
            incident = _incident_payload(
                run_id=run_root.name,
                manifest_sha256=manifest_sha256,
                manifest_cell_count=manifest_cell_count,
                cell_id=cell_id,
                config_hash=config_hash,
                failure_payload=failure_payload,
                checkpoints=checkpoints,
            )
            preimage, incident_bytes = _publish_or_validate_archive(
                archive, incident=incident, failure_payload=failure_payload
            )
            if failure_path.exists():
                current = _read_regular(failure_path, label="active failure")
                if current != preimage:
                    raise PermanentFailureArchiveError(
                        f"active failure changed after archival: {cell_id}"
                    )
                io.remove_file(failure_path)
            if failure_path.exists():
                raise PermanentFailureArchiveError(
                    f"active failure removal was not durable: {cell_id}"
                )
            reset = {
                "incident_schema_version": INCIDENT_SCHEMA_VERSION,
                "incident_kind": INCIDENT_KIND,
                "cell_id": cell_id,
                "preimage_sha256": _sha256(preimage),
                "incident_sha256": _sha256(incident_bytes),
                "after_sha256": None,
                "active_failure_removed": True,
            }
            io.atomic_write_text(marker, _json_text(reset))
            os.chmod(marker, 0o444)
            os.chmod(archive, 0o555)
            return {
                "cell_id": cell_id,
                "status": "reset",
                "before_sha256": _sha256(preimage),
                "after_sha256": None,
                "archive": str(archive),
            }
    except CellLockUnavailable as exc:
        raise PermanentFailureArchiveError(f"cell is active: {cell_id}") from exc


def archive_run(
    run_root: Path,
    *,
    apply: bool = False,
    target_cell_hashes: Mapping[str, str] = TARGET_CELL_CONFIG_HASHES,
    require_run_id: str | None = TARGET_RUN_ID,
) -> dict[str, Any]:
    supplied_root = Path(run_root)
    if supplied_root.is_symlink():
        raise PermanentFailureArchiveError(
            f"run root is a symlink: {supplied_root}"
        )
    root = supplied_root.resolve()
    if not root.is_dir():
        raise PermanentFailureArchiveError(f"run root is missing or unsafe: {run_root}")
    if require_run_id is not None and root.name != require_run_id:
        raise PermanentFailureArchiveError(
            f"refusing run {root.name!r}; exact target is {require_run_id!r}"
        )
    try:
        snapshot = load_manifest(root, verify_frozen=True)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PermanentFailureArchiveError(f"cannot load frozen manifest: {exc}") from exc
    manifest = {cell.cell_id: cell for cell in snapshot.cells}
    missing = sorted(set(target_cell_hashes) - set(manifest))
    if missing:
        raise PermanentFailureArchiveError(f"target cells are absent from manifest: {missing}")

    cells: list[dict[str, Any]] = []
    for cell_id, expected_hash in target_cell_hashes.items():
        cell = manifest[cell_id]
        if cell.config_hash() != expected_hash:
            raise PermanentFailureArchiveError(
                f"manifest config hash drift for {cell_id}: {cell.config_hash()}"
            )
        archive = root / INCIDENT_ROOT / cell_id
        marker = archive / RESET_MARKER_FILENAME
        failure_path = root / "cells" / cell_id / FAILURE_FILENAME
        if marker.exists():
            row = _validate_completed_archive(
                archive,
                run_id=root.name,
                manifest_sha256=snapshot.sha256,
                manifest_cell_count=len(snapshot.cells),
                cell_id=cell_id,
                expected_config_hash=expected_hash,
            )
            if failure_path.exists():
                raise PermanentFailureArchiveError(
                    f"new failure exists after completed incident: {cell_id}"
                )
            cells.append(row)
            continue
        if failure_path.exists():
            payload = _read_regular(failure_path, label="active failure")
        elif (archive / PREIMAGE_FILENAME).exists():
            payload = _read_regular(archive / PREIMAGE_FILENAME, label="incident preimage")
        else:
            raise PermanentFailureArchiveError(
                f"target failure is absent without reset evidence: {cell_id}"
            )
        _validate_failure(payload, cell_id=cell_id, config_hash=expected_hash)
        checkpoints = _checkpoint_evidence(root / "cells" / cell_id)
        row = {
            "cell_id": cell_id,
            "status": "would_reset" if not apply else "pending",
            "before_sha256": _sha256(payload),
            "after_sha256": None,
            "archive": str(archive),
            "checkpoint_count": len(checkpoints),
        }
        if apply:
            row = _apply_one(
                run_root=root,
                manifest_sha256=snapshot.sha256,
                manifest_cell_count=len(snapshot.cells),
                cell_id=cell_id,
                config_hash=expected_hash,
            )
        cells.append(row)
    counts: dict[str, int] = {}
    for row in cells:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "run_id": root.name,
        "manifest_sha256": snapshot.sha256,
        "applied": apply,
        "target_count": len(target_cell_hashes),
        "counts": counts,
        "cells": cells,
        "would_change_cells": sorted(
            row["cell_id"] for row in cells if row["status"] == "would_reset"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(DEFAULT_RESULTS_ROOT) / TARGET_RUN_ID,
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = archive_run(args.run_root, apply=args.apply)
    except PermanentFailureArchiveError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
