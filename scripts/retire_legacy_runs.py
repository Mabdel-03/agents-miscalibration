#!/usr/bin/env python3
"""Retire the three legacy sweep roots and dispatcher state as read-only evidence.

The operation is deliberately last in legacy consolidation.  It requires a verified
``legacy_consolidated`` snapshot and external snapshot attestation, writes an intent
marker into each exact allowlisted root, removes write bits without following symlinks,
verifies every node, and publishes one external completion record last.  Dry-run is the
default; completed retirement is idempotently revalidated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable, Mapping

try:
    from scripts.create_recovery_snapshot import verify_snapshot
except ModuleNotFoundError:  # direct execution
    from create_recovery_snapshot import verify_snapshot  # type: ignore[no-redef]


LEGACY_ROOT_NAMES = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
    ".dispatcher-v3",
)
ROOT_MARKER = "RETIRED_SCHEMA5_V1.json"
COMPLETE_MARKER = "LEGACY_RETIREMENT_COMPLETE.json"
LEGACY_CLEANUP_MARKER = "LEGACY_CLEANUP_COMPLETE.json"


class RetirementError(RuntimeError):
    """Legacy evidence cannot be safely or completely retired."""


def _snapshot_source_name(root_name: str) -> str:
    return root_name.lstrip(".").replace("-", "_").replace(".", "_") or "dispatcher"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_snapshot_attestation(
    path: Path, snapshot_root: Path, *, expected_snapshot_id: str
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot read legacy snapshot attestation: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "recovery_snapshot_external_attestation"
        or payload.get("passed") is not True
        or Path(str(payload.get("snapshot_root", ""))).resolve() != snapshot_root
        or payload.get("snapshot_id") != expected_snapshot_id
    ):
        raise RetirementError("legacy snapshot attestation has the wrong identity")
    controls = payload.get("control_artifacts")
    if not isinstance(controls, dict) or len(controls) != 5:
        raise RetirementError("legacy snapshot attestation has an incomplete control inventory")
    for filename, record in controls.items():
        candidate = snapshot_root / filename
        if (
            not isinstance(record, dict)
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != record.get("size")
            or _sha256(candidate) != record.get("sha256")
        ):
            raise RetirementError(f"legacy snapshot control artifact drifted: {filename}")
    return payload


def _walk_nodes(root: Path) -> list[Path]:
    nodes = [root]
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                raise RetirementError(f"refusing symlink in legacy evidence: {path}")
            nodes.append(path)
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RetirementError(f"refusing unsafe legacy evidence node: {path}")
            nodes.append(path)
    return nodes


def _verify_read_only(roots: Iterable[Path]) -> int:
    count = 0
    for root in roots:
        for path in _walk_nodes(root):
            count += 1
            if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222:
                raise RetirementError(f"retired evidence remains writable: {path}")
    return count


def _verify_consolidated_snapshot_sources(
    *, results_root: Path, recovery_root: Path, snapshot_root: Path, snapshot_id: str
) -> None:
    try:
        catalog = json.loads(
            (snapshot_root / "SNAPSHOT_CATALOG.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot read consolidated snapshot catalog: {exc}") from exc
    expected_sources = [
        {
            "name": _snapshot_source_name(name),
            "path": str((results_root / name).resolve()),
        }
        for name in LEGACY_ROOT_NAMES
    ] + [
        {
            "name": "legacy_cleanup_evidence",
            "path": str((recovery_root / "operations" / "legacy_consolidation").resolve()),
        },
        {
            "name": "legacy_cleanup_complete",
            "path": str((recovery_root / LEGACY_CLEANUP_MARKER).resolve()),
        },
    ]
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or catalog.get("snapshot_id") != snapshot_id
        or catalog.get("copy_contract")
        != "independent_regular_files_no_hardlinks_no_symlinks"
        or catalog.get("sources") != expected_sources
    ):
        raise RetirementError(
            "legacy_consolidated snapshot does not cover the exact retired roots and cleanup evidence"
        )


def retire(
    *, results_root: Path, recovery_root: Path, apply: bool = False
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    recovery_root = recovery_root.expanduser().resolve()
    roots = tuple(results_root / name for name in LEGACY_ROOT_NAMES)
    for name, root in zip(LEGACY_ROOT_NAMES, roots, strict=True):
        if root.name != name or root.is_symlink() or not root.is_dir():
            raise RetirementError(f"missing or unsafe exact legacy root: {root}")
    snapshot_root = recovery_root / "legacy_consolidated"
    verified_snapshot = verify_snapshot(snapshot_root)
    _verify_consolidated_snapshot_sources(
        results_root=results_root,
        recovery_root=recovery_root,
        snapshot_root=snapshot_root,
        snapshot_id=str(verified_snapshot["snapshot_id"]),
    )
    attestation_path = recovery_root / "legacy_consolidated.attestation.json"
    attestation = _load_snapshot_attestation(
        attestation_path,
        snapshot_root,
        expected_snapshot_id=str(verified_snapshot["snapshot_id"]),
    )
    complete_path = recovery_root / COMPLETE_MARKER
    if complete_path.is_file():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") != verified_snapshot["snapshot_id"]:
            raise RetirementError("retirement marker references another snapshot")
        count = _verify_read_only(roots)
        return payload | {"status": "already_retired", "verified_nodes": count}

    node_counts = {root.name: len(_walk_nodes(root)) for root in roots}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if not apply else "retired",
        "results_root": str(results_root),
        "roots": [str(root) for root in roots],
        "node_counts": node_counts,
        "snapshot_id": verified_snapshot["snapshot_id"],
        "snapshot_attestation_sha256": _sha256(attestation_path),
    }
    if not apply:
        return report

    for root in roots:
        intent = {
            "schema_version": 1,
            "status": "retirement_intent",
            "root": str(root),
            "snapshot_id": verified_snapshot["snapshot_id"],
            "snapshot_attestation_sha256": report["snapshot_attestation_sha256"],
        }
        marker = root / ROOT_MARKER
        if not marker.exists():
            _atomic_json(marker, intent)
        # Files first, then deepest directories, and the exact root last.
        nodes = _walk_nodes(root)
        for path in sorted(nodes, key=lambda item: len(item.parts), reverse=True):
            info = path.stat(follow_symlinks=False)
            os.chmod(path, stat.S_IMODE(info.st_mode) & ~0o222)
    report["verified_nodes"] = _verify_read_only(roots)
    _atomic_json(complete_path, report)
    return report | {"completion_marker": str(complete_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = retire(
            results_root=args.results_root,
            recovery_root=args.recovery_root,
            apply=args.apply,
        )
    except (OSError, ValueError, RetirementError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
