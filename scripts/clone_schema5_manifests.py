#!/usr/bin/env python3
"""Clone the three frozen scientific contracts into empty schema-5 run roots.

Only the immutable manifest and benchmark-contract files are copied.  Result rows,
cell directories, batch files, logs, and legacy dispatcher state are never imported.
The four scientific-contract files are copied byte-for-byte and verified against their
recorded checksums.  Deterministic lineage and artifact-policy sidecars bind each new
run to an immutable release, two immutable environments, and the frozen model contract.

Dry-run is the default.  ``--apply`` builds each run in a same-filesystem staging
directory, verifies it, publishes the initialized marker last, and atomically renames
the empty run root into place.  Existing initialized roots are verification-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.serving.model_contracts import (  # noqa: E402
    FrozenModelContracts,
    load_model_contracts,
)


CLONE_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 5
LINEAGE_FILENAME = "lineage.schema5-v1.json"
LINEAGE_CHECKSUM_FILENAME = "lineage.schema5-v1.sha256"
POLICY_FILENAME = "artifact_policy.schema5-v1.json"
POLICY_CHECKSUM_FILENAME = "artifact_policy.schema5-v1.sha256"
INITIALIZED_FILENAME = "SCHEMA5_RUN_INITIALIZED.json"
STATIC_CONTRACT_FILENAMES = (
    "cells.json",
    "cells.sha256",
    "benchmark_contracts.v1.json",
    "benchmark_contracts.v1.sha256",
)
REQUIRED_METADATA_FIELDS = (
    "release_id",
    "environment_hash",
    "model_revision",
    "tokenizer_revision",
    "model_contract_sha256",
    "serving_profile",
    "endpoint_generation",
    "effective_context",
    "rollout_generation",
)
RUN_CLONES = (
    ("full_sweep_v1", "full_sweep_schema5_v1", 4_680),
    (
        "full_sweep_agent_counts_v1",
        "full_sweep_agent_counts_schema5_v1",
        14_400,
    ),
    (
        "full_sweep_agent_count_7_v1",
        "full_sweep_agent_count_7_schema5_v1",
        3_600,
    ),
)
EXPECTED_TOTAL_CELLS = 22_680
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class CloneError(RuntimeError):
    """The new run cannot be proven to be an empty exact scientific clone."""


@dataclass(frozen=True)
class ReleasePins:
    release_id: str
    git_commit: str
    source_tree_sha256: str
    harness_sha256: str
    serving_sha256: str

    def validate(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id:
            raise CloneError("release_id must be non-empty text")
        if _GIT_COMMIT_RE.fullmatch(self.git_commit) is None:
            raise CloneError("git_commit must be an exact 40-character commit")
        for label, digest in (
            ("source_tree_sha256", self.source_tree_sha256),
            ("harness_sha256", self.harness_sha256),
            ("serving_sha256", self.serving_sha256),
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise CloneError(f"{label} must be a lowercase SHA-256")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _domain_id(domain: str, payload: Mapping[str, Any]) -> str:
    return _sha256(_canonical_bytes({"domain": domain, "payload": payload}))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".clone-tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CloneError(f"missing or unsafe {label}: {path}")
    return path.read_bytes()


def _verified_checksum(path: Path, filename: str, payload: bytes) -> str:
    raw = _read_regular(path, label=f"{filename} checksum")
    try:
        fields = raw.decode("utf-8").strip().split()
    except UnicodeError as exc:
        raise CloneError(f"checksum is not UTF-8: {path}") from exc
    observed = _sha256(payload)
    if (
        len(fields) != 2
        or _SHA256_RE.fullmatch(fields[0]) is None
        or fields[1] != filename
        or fields[0] != observed
    ):
        raise CloneError(f"checksum does not bind exact {filename} bytes: {path}")
    return observed


def _source_contract(
    source_root: Path, *, expected_cell_count: int
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]], tuple[int, ...]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise CloneError(f"source run root is missing or unsafe: {source_root}")
    payloads = {
        filename: _read_regular(source_root / filename, label=filename)
        for filename in STATIC_CONTRACT_FILENAMES
    }
    manifest_sha = _verified_checksum(
        source_root / "cells.sha256", "cells.json", payloads["cells.json"]
    )
    benchmark_sha = _verified_checksum(
        source_root / "benchmark_contracts.v1.sha256",
        "benchmark_contracts.v1.json",
        payloads["benchmark_contracts.v1.json"],
    )
    try:
        snapshot = load_manifest(source_root, verify_frozen=True)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CloneError(f"cannot validate source manifest {source_root}: {exc}") from exc
    if len(snapshot.cells) != expected_cell_count:
        raise CloneError(
            f"source {source_root.name} has {len(snapshot.cells)} cells; "
            f"expected {expected_cell_count}"
        )
    if snapshot.sha256 != manifest_sha:
        raise CloneError("loaded manifest digest disagrees with exact-byte checksum")
    try:
        benchmark_payload = json.loads(payloads["benchmark_contracts.v1.json"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CloneError(f"cannot parse frozen benchmark contracts: {exc}") from exc
    if not isinstance(benchmark_payload, dict):
        raise CloneError("benchmark contracts must contain one JSON object")
    if benchmark_payload.get("manifest_sha256") != manifest_sha:
        raise CloneError("benchmark contracts do not bind the source manifest")
    identities = {
        filename: {"sha256": _sha256(payload), "size": len(payload)}
        for filename, payload in payloads.items()
    }
    if identities["benchmark_contracts.v1.json"]["sha256"] != benchmark_sha:
        raise CloneError("benchmark sidecar digest changed during source validation")
    return payloads, identities, tuple(cell.n_agents for cell in snapshot.cells)


def _lineage(
    *,
    source_run_id: str,
    target_run_id: str,
    identities: Mapping[str, dict[str, Any]],
    cell_count: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CLONE_SCHEMA_VERSION,
        "clone_mode": "byte_for_byte_scientific_contract",
        "source_run_id": source_run_id,
        "target_run_id": target_run_id,
        "source_artifacts": dict(identities),
        "target_artifacts": dict(identities),
        "manifest_cell_count": cell_count,
        "imported_result_rows": 0,
        "transformations": [],
    }
    payload["lineage_id"] = _domain_id(
        "agents_scaling.lineage.schema5-v1", payload
    )
    return payload


def _policy(
    *,
    run_id: str,
    manifest_sha256: str,
    benchmark_sha256: str,
    model_contracts: FrozenModelContracts,
    pins: ReleasePins,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CLONE_SCHEMA_VERSION,
        "run_id": run_id,
        "authoritative": True,
        "required_artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "accepted_manifest_sha256": manifest_sha256,
        "accepted_benchmark_contracts_sha256": benchmark_sha256,
        "accepted_model_contract_sha256": model_contracts.sha256,
        "release": {
            "release_id": pins.release_id,
            "git_commit": pins.git_commit,
            "source_tree_sha256": pins.source_tree_sha256,
        },
        "environment": {
            "harness_sha256": pins.harness_sha256,
            "serving_sha256": pins.serving_sha256,
        },
        "required_metadata_fields": list(REQUIRED_METADATA_FIELDS),
        "legacy_result_import_allowed": False,
    }
    payload["policy_id"] = _domain_id(
        "agents_scaling.artifact_policy.schema5-v1", payload
    )
    return payload


def _checksum_bytes(filename: str, payload: bytes) -> bytes:
    return f"{_sha256(payload)}  {filename}\n".encode("utf-8")


def _static_sidecars(
    *,
    source_run_id: str,
    target_run_id: str,
    payloads: Mapping[str, bytes],
    identities: Mapping[str, dict[str, Any]],
    expected_cell_count: int,
    model_contracts: FrozenModelContracts,
    pins: ReleasePins,
) -> dict[str, bytes]:
    lineage_bytes = _json_bytes(
        _lineage(
            source_run_id=source_run_id,
            target_run_id=target_run_id,
            identities=identities,
            cell_count=expected_cell_count,
        )
    )
    policy_bytes = _json_bytes(
        _policy(
            run_id=target_run_id,
            manifest_sha256=_sha256(payloads["cells.json"]),
            benchmark_sha256=_sha256(payloads["benchmark_contracts.v1.json"]),
            model_contracts=model_contracts,
            pins=pins,
        )
    )
    return {
        LINEAGE_FILENAME: lineage_bytes,
        LINEAGE_CHECKSUM_FILENAME: _checksum_bytes(LINEAGE_FILENAME, lineage_bytes),
        POLICY_FILENAME: policy_bytes,
        POLICY_CHECKSUM_FILENAME: _checksum_bytes(POLICY_FILENAME, policy_bytes),
    }


def _verify_initialized(
    target_root: Path,
    *,
    expected_static: Mapping[str, bytes],
    allow_populated_cells: bool,
) -> dict[str, Any]:
    marker_path = target_root / INITIALIZED_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise CloneError(f"schema-5 initialized marker is missing: {marker_path}")
    for filename, expected in expected_static.items():
        observed = _read_regular(target_root / filename, label=filename)
        if observed != expected:
            raise CloneError(f"initialized run static artifact drift: {filename}")
    cells = target_root / "cells"
    if cells.is_symlink() or not cells.is_dir():
        raise CloneError(f"initialized run has no safe cells directory: {cells}")
    if not allow_populated_cells and any(cells.iterdir()):
        raise CloneError(f"new authoritative run imported cell artifacts: {cells}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloneError(f"cannot parse initialized marker {marker_path}: {exc}") from exc
    expected_hashes = {
        filename: _sha256(payload) for filename, payload in expected_static.items()
    }
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != CLONE_SCHEMA_VERSION
        or marker.get("static_artifact_sha256s") != expected_hashes
        or marker.get("legacy_result_rows_imported") != 0
        or marker.get("cells_directory_empty_at_initialization") is not True
    ):
        raise CloneError(f"initialized marker does not bind static artifacts: {marker_path}")
    return marker


def clone_run(
    *,
    source_root: Path,
    target_root: Path,
    expected_cell_count: int,
    model_contracts: FrozenModelContracts,
    pins: ReleasePins,
    apply: bool = False,
) -> dict[str, Any]:
    pins.validate()
    payloads, identities, agent_counts = _source_contract(
        source_root, expected_cell_count=expected_cell_count
    )
    static = dict(payloads)
    static.update(
        _static_sidecars(
            source_run_id=source_root.name,
            target_run_id=target_root.name,
            payloads=payloads,
            identities=identities,
            expected_cell_count=expected_cell_count,
            model_contracts=model_contracts,
            pins=pins,
        )
    )
    if target_root.exists():
        marker = _verify_initialized(
            target_root, expected_static=static, allow_populated_cells=True
        )
        return {
            "status": "already_initialized",
            "source_run_id": source_root.name,
            "target_run_id": target_root.name,
            "cell_count": expected_cell_count,
            "marker": marker,
            "agent_counts": sorted(set(agent_counts)),
        }
    report = {
        "status": "dry_run",
        "source_run_id": source_root.name,
        "target_run_id": target_root.name,
        "cell_count": expected_cell_count,
        "manifest_sha256": _sha256(payloads["cells.json"]),
        "benchmark_contracts_sha256": _sha256(
            payloads["benchmark_contracts.v1.json"]
        ),
        "model_contract_sha256": model_contracts.sha256,
        "agent_counts": sorted(set(agent_counts)),
        "would_import_result_rows": 0,
    }
    if not apply:
        return report
    target_root.parent.mkdir(parents=True, exist_ok=True)
    staging = target_root.parent / f".{target_root.name}.clone-incomplete"
    if staging.exists():
        raise CloneError(
            f"staging root already exists; inspect before retrying: {staging}"
        )
    staging.mkdir(mode=0o700)
    try:
        for filename, payload in static.items():
            _atomic_bytes(staging / filename, payload, mode=0o444)
            os.chmod(staging / filename, 0o444)
        (staging / "cells").mkdir(mode=0o755)
        static_hashes = {
            filename: _sha256(payload) for filename, payload in static.items()
        }
        marker = {
            "schema_version": CLONE_SCHEMA_VERSION,
            "run_id": target_root.name,
            "static_artifact_sha256s": static_hashes,
            "legacy_result_rows_imported": 0,
            "cells_directory_empty_at_initialization": True,
        }
        _atomic_bytes(staging / INITIALIZED_FILENAME, _json_bytes(marker), mode=0o444)
        os.chmod(staging / INITIALIZED_FILENAME, 0o444)
        _verify_initialized(
            staging, expected_static=static, allow_populated_cells=False
        )
        os.replace(staging, target_root)
        _fsync_directory(target_root.parent)
    finally:
        # Fail closed without deleting a staging preimage; a non-empty staging root
        # requires explicit operator inspection before a retry.
        pass
    report["status"] = "initialized"
    return report


def clone_all(
    *,
    results_root: Path,
    model_contract_path: Path,
    pins: ReleasePins,
    apply: bool = False,
) -> dict[str, Any]:
    model_contracts = load_model_contracts(model_contract_path)
    reports: list[dict[str, Any]] = []
    all_agent_counts: set[int] = set()
    total = 0
    for source_id, target_id, expected_count in RUN_CLONES:
        report = clone_run(
            source_root=results_root / source_id,
            target_root=results_root / target_id,
            expected_cell_count=expected_count,
            model_contracts=model_contracts,
            pins=pins,
            apply=apply,
        )
        reports.append(report)
        total += int(report["cell_count"])
        all_agent_counts.update(int(value) for value in report["agent_counts"])
    if total != EXPECTED_TOTAL_CELLS:
        raise CloneError(f"schema-5 clone cardinality is {total}, expected 22680")
    if all_agent_counts != set(range(1, 8)):
        raise CloneError(
            f"schema-5 clone agent-count coverage is {sorted(all_agent_counts)}, expected 1..7"
        )
    return {
        "status": "initialized" if apply else "dry_run",
        "results_root": str(results_root),
        "total_cells": total,
        "agent_counts": sorted(all_agent_counts),
        "model_contract_sha256": model_contracts.sha256,
        "runs": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--model-contract",
        type=Path,
        default=REPO / "configs" / "model_contracts.v1.json",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--harness-env-sha256", required=True)
    parser.add_argument("--serving-env-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pins = ReleasePins(
        release_id=args.release_id,
        git_commit=args.git_commit,
        source_tree_sha256=args.source_tree_sha256,
        harness_sha256=args.harness_env_sha256,
        serving_sha256=args.serving_env_sha256,
    )
    try:
        report = clone_all(
            results_root=args.results_root.resolve(),
            model_contract_path=args.model_contract.resolve(),
            pins=pins,
            apply=args.apply,
        )
    except CloneError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
