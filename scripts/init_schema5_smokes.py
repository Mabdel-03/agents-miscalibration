#!/usr/bin/env python3
"""Initialize the three immutable, estimand-excluded schema-5 smoke suites.

Each suite is generated from a configuration frozen in the release, receives the same
authoritative artifact-schema-5 policy as production, and is published empty through a
same-filesystem staging rename.  Smoke rows can therefore validate the complete runtime
provenance contract while remaining structurally unable to enter primary analysis.
Dry-run is the default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.experiment.artifact_policy import (  # noqa: E402
    POLICY_DOMAIN,
    POLICY_FILENAME,
    POLICY_CHECKSUM_FILENAME,
    REQUIRED_METADATA_FIELDS,
    load_artifact_policy,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.experiment.sweep import load_sweep  # noqa: E402
from agents_scaling.serving.model_contracts import load_model_contracts  # noqa: E402
from agents_scaling.serving.profiles import serving_profile_for_cell  # noqa: E402
from scripts.init_run_manifest import initialize  # noqa: E402


SMOKE_SUITES = (
    (
        "schema5_smoke_32b_long_v1",
        Path("configs/long_context_protocol_smoke.yaml"),
        15,
    ),
    (
        "schema5_smoke_selective_long_v1",
        Path("configs/selective_long_profiles_smoke.yaml"),
        20,
    ),
    (
        "schema5_smoke_standard_canaries_v1",
        Path("configs/standard_profile_canaries.yaml"),
        6,
    ),
)
LINEAGE_FILENAME = "smoke_lineage.schema5-v1.json"
LINEAGE_CHECKSUM_FILENAME = "smoke_lineage.schema5-v1.sha256"
MARKER_FILENAME = "SCHEMA5_SMOKE_INITIALIZED.json"
ATTEMPT_BINDING_PROTOCOL = "schema5-v1.2-r11-smoke-attempt-binding-v1"
_ATTEMPT_BINDING_FIELDS = {
    "protocol",
    "attempt_id",
    "attempt_ordinal",
    "immutable_sha256",
    "capacity_generation",
    "rollout_generation",
    "fleet_contract_sha256",
    "release_fleet_contract_sha256",
    "trusted_catalog_id",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class SmokeInitializationError(RuntimeError):
    """A smoke suite cannot be proven immutable, empty, and estimand-excluded."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, value: bytes, *, mode: int = 0o444) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _checksum(filename: str, payload: bytes) -> bytes:
    return f"{_sha256_bytes(payload)}  {filename}\n".encode("utf-8")


def _policy_payload(
    *,
    run_id: str,
    manifest_sha256: str,
    benchmark_sha256: str,
    model_contract_sha256: str,
    release_id: str,
    git_commit: str,
    source_tree_sha256: str,
    harness_environment_sha256: str,
    serving_environment_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "authoritative": True,
        "required_artifact_schema_version": 5,
        "accepted_manifest_sha256": manifest_sha256,
        "accepted_benchmark_contracts_sha256": benchmark_sha256,
        "accepted_model_contract_sha256": model_contract_sha256,
        "release": {
            "release_id": release_id,
            "git_commit": git_commit,
            "source_tree_sha256": source_tree_sha256,
        },
        "environment": {
            "harness_sha256": harness_environment_sha256,
            "serving_sha256": serving_environment_sha256,
        },
        "required_metadata_fields": list(REQUIRED_METADATA_FIELDS),
        "legacy_result_import_allowed": False,
    }
    payload["policy_id"] = _sha256_bytes(
        _canonical_bytes({"domain": POLICY_DOMAIN, "payload": payload})
    )
    return payload


def _lineage_payload(
    *,
    run_id: str,
    config_path: Path,
    config_sha256: str,
    expected_cells: int,
    manifest_sha256: str,
    benchmark_sha256: str,
    release_id: str,
    source_tree_sha256: str,
    routes: Mapping[str, int],
    attempt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "schema5_operational_smoke",
        "run_id": run_id,
        "estimand_excluded": True,
        "primary_analysis_eligible": False,
        "configuration": {
            "release_relative_path": config_path.as_posix(),
            "sha256": config_sha256,
        },
        "expected_cells": expected_cells,
        "manifest_sha256": manifest_sha256,
        "benchmark_contracts_sha256": benchmark_sha256,
        "release_id": release_id,
        "source_tree_sha256": source_tree_sha256,
        "serving_profile_counts": dict(sorted(routes.items())),
        "smoke_attempt": (
            None if attempt_binding is None else dict(attempt_binding)
        ),
    }
    payload["lineage_id"] = _sha256_bytes(
        _canonical_bytes({"domain": "agents_scaling.schema5_smoke_lineage.v1", "payload": payload})
    )
    return payload


def _verify_suite(
    root: Path,
    *,
    config_path: Path,
    expected_cells: int,
    expected_policy_sha256: str | None = None,
    expected_release_id: str | None = None,
    expected_git_commit: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_harness_environment_sha256: str | None = None,
    expected_serving_environment_sha256: str | None = None,
    expected_model_contract_sha256: str | None = None,
    expected_attempt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = tuple(load_sweep(config_path))
    if len(expected) != expected_cells:
        raise SmokeInitializationError(
            f"smoke config {config_path} has {len(expected)} cells, expected {expected_cells}"
        )
    snapshot = load_manifest(root, verify_frozen=True)
    if snapshot.cells != expected:
        raise SmokeInitializationError(f"smoke manifest drifted from release config: {root}")
    catalog = VerifiedQuestionCatalog(root, snapshot=snapshot)
    policy = load_artifact_policy(
        root, required=True, expected_file_sha256=expected_policy_sha256
    )
    assert policy is not None
    expected_identity = {
        "release_id": (policy.release.release_id, expected_release_id),
        "git_commit": (policy.release.git_commit, expected_git_commit),
        "source_tree_sha256": (
            policy.release.source_tree_sha256,
            expected_source_tree_sha256,
        ),
        "harness_environment_sha256": (
            policy.environment.harness_sha256,
            expected_harness_environment_sha256,
        ),
        "serving_environment_sha256": (
            policy.environment.serving_sha256,
            expected_serving_environment_sha256,
        ),
        "model_contract_sha256": (
            policy.accepted_model_contract_sha256,
            expected_model_contract_sha256,
        ),
    }
    for label, (observed, wanted) in expected_identity.items():
        if wanted is not None and observed != wanted:
            raise SmokeInitializationError(
                f"smoke {label} drifted: expected {wanted}, observed {observed}"
            )
    lineage_path = root / LINEAGE_FILENAME
    marker_path = root / MARKER_FILENAME
    if not lineage_path.is_file() or not marker_path.is_file():
        raise SmokeInitializationError(f"smoke suite publication is incomplete: {root}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checksum = (root / LINEAGE_CHECKSUM_FILENAME).read_text(encoding="utf-8")
    if checksum != _checksum(LINEAGE_FILENAME, lineage_path.read_bytes()).decode("utf-8"):
        raise SmokeInitializationError(f"smoke lineage checksum is invalid: {root}")
    if (
        lineage.get("estimand_excluded") is not True
        or lineage.get("primary_analysis_eligible") is not False
        or marker.get("estimand_excluded") is not True
        or marker.get("manifest_sha256") != snapshot.sha256
        or marker.get("artifact_policy_sha256") != policy.file_sha256
        or lineage.get("configuration", {}).get("sha256") != _sha256_file(config_path)
        or lineage.get("smoke_attempt")
        != (
            None
            if expected_attempt_binding is None
            else dict(expected_attempt_binding)
        )
        or marker.get("smoke_attempt")
        != (
            None
            if expected_attempt_binding is None
            else dict(expected_attempt_binding)
        )
    ):
        raise SmokeInitializationError(f"smoke exclusion/policy marker is invalid: {root}")
    routes = Counter(serving_profile_for_cell(cell).name for cell in snapshot.cells)
    if lineage.get("serving_profile_counts") != dict(sorted(routes.items())):
        raise SmokeInitializationError(f"smoke serving-route census drifted: {root}")
    return {
        "run_id": root.name,
        "cell_count": len(snapshot.cells),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "artifact_policy_sha256": policy.file_sha256,
        "lineage_sha256": _sha256_file(lineage_path),
        "serving_profile_counts": dict(sorted(routes.items())),
        "estimand_excluded": True,
        "smoke_attempt": (
            None
            if lineage.get("smoke_attempt") is None
            else dict(lineage["smoke_attempt"])
        ),
    }


def initialize_all(
    *,
    results_root: Path,
    release_worktree: Path,
    model_contract_path: Path,
    release_id: str,
    git_commit: str,
    source_tree_sha256: str,
    harness_environment_sha256: str,
    serving_environment_sha256: str,
    apply: bool,
    attempt_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    release_worktree = release_worktree.expanduser().resolve()
    model_contracts = load_model_contracts(model_contract_path.expanduser().resolve())
    if attempt_binding is not None:
        if (
            set(attempt_binding) != _ATTEMPT_BINDING_FIELDS
            or attempt_binding.get("protocol") != ATTEMPT_BINDING_PROTOCOL
            or not isinstance(attempt_binding.get("attempt_id"), str)
            or not attempt_binding["attempt_id"]
            or not isinstance(attempt_binding.get("attempt_ordinal"), int)
            or isinstance(attempt_binding["attempt_ordinal"], bool)
            or int(attempt_binding["attempt_ordinal"]) < 1
            or any(
                not isinstance(attempt_binding.get(field), int)
                or isinstance(attempt_binding[field], bool)
                or int(attempt_binding[field]) < 1
                for field in ("capacity_generation", "rollout_generation")
            )
            or any(
                _SHA256_RE.fullmatch(str(attempt_binding.get(field, ""))) is None
                for field in (
                    "immutable_sha256",
                    "fleet_contract_sha256",
                    "release_fleet_contract_sha256",
                    "trusted_catalog_id",
                )
            )
        ):
            raise SmokeInitializationError(
                "smoke attempt binding is malformed"
            )
    if _GIT_RE.fullmatch(git_commit) is None:
        raise SmokeInitializationError("git_commit must be an exact 40-character SHA")
    for label, value in (
        ("source_tree_sha256", source_tree_sha256),
        ("harness_environment_sha256", harness_environment_sha256),
        ("serving_environment_sha256", serving_environment_sha256),
    ):
        if _SHA256_RE.fullmatch(value) is None:
            raise SmokeInitializationError(f"{label} is not a lowercase SHA-256")
    reports: list[dict[str, Any]] = []
    for run_id, relative_config, expected_cells in SMOKE_SUITES:
        config_path = release_worktree / relative_config
        if not config_path.is_file():
            raise SmokeInitializationError(f"frozen smoke config is missing: {config_path}")
        cells = tuple(load_sweep(config_path))
        if len(cells) != expected_cells:
            raise SmokeInitializationError(
                f"{run_id} config has {len(cells)} cells, expected {expected_cells}"
            )
        routes = Counter(serving_profile_for_cell(cell).name for cell in cells)
        target = results_root / run_id
        if target.exists():
            reports.append(
                {"status": "already_initialized"}
                | _verify_suite(
                    target,
                    config_path=config_path,
                    expected_cells=expected_cells,
                    expected_release_id=release_id,
                    expected_git_commit=git_commit,
                    expected_source_tree_sha256=source_tree_sha256,
                    expected_harness_environment_sha256=harness_environment_sha256,
                    expected_serving_environment_sha256=serving_environment_sha256,
                    expected_model_contract_sha256=model_contracts.sha256,
                    expected_attempt_binding=attempt_binding,
                )
            )
            continue
        dry = {
            "status": "dry_run",
            "run_id": run_id,
            "cell_count": expected_cells,
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "serving_profile_counts": dict(sorted(routes.items())),
            "estimand_excluded": True,
        }
        if not apply:
            reports.append(dry)
            continue
        results_root.mkdir(parents=True, exist_ok=True)
        stage_parent = results_root / f".{run_id}.schema5-smoke-incomplete"
        if stage_parent.exists() or stage_parent.is_symlink():
            raise SmokeInitializationError(
                f"incomplete smoke staging root requires inspection: {stage_parent}"
            )
        stage_parent.mkdir(mode=0o700)
        staged = stage_parent / run_id
        try:
            initialize(config=config_path, run_root=staged, expected_cells=expected_cells)
            snapshot = load_manifest(staged, verify_frozen=True)
            catalog = VerifiedQuestionCatalog(staged, snapshot=snapshot)
            policy = _policy_payload(
                run_id=run_id,
                manifest_sha256=snapshot.sha256,
                benchmark_sha256=catalog.sidecar_sha256,
                model_contract_sha256=model_contracts.sha256,
                release_id=release_id,
                git_commit=git_commit,
                source_tree_sha256=source_tree_sha256,
                harness_environment_sha256=harness_environment_sha256,
                serving_environment_sha256=serving_environment_sha256,
            )
            policy_bytes = _json_bytes(policy)
            lineage_bytes = _json_bytes(
                _lineage_payload(
                    run_id=run_id,
                    config_path=relative_config,
                    config_sha256=_sha256_file(config_path),
                    expected_cells=expected_cells,
                    manifest_sha256=snapshot.sha256,
                    benchmark_sha256=catalog.sidecar_sha256,
                    release_id=release_id,
                    source_tree_sha256=source_tree_sha256,
                    routes=routes,
                    attempt_binding=attempt_binding,
                )
            )
            _atomic_bytes(staged / POLICY_FILENAME, policy_bytes)
            _atomic_bytes(
                staged / POLICY_CHECKSUM_FILENAME,
                _checksum(POLICY_FILENAME, policy_bytes),
            )
            _atomic_bytes(staged / LINEAGE_FILENAME, lineage_bytes)
            _atomic_bytes(
                staged / LINEAGE_CHECKSUM_FILENAME,
                _checksum(LINEAGE_FILENAME, lineage_bytes),
            )
            (staged / "cells").mkdir(mode=0o755)
            marker = {
                "schema_version": 1,
                "run_id": run_id,
                "cell_count": expected_cells,
                "manifest_sha256": snapshot.sha256,
                "benchmark_contracts_sha256": catalog.sidecar_sha256,
                "artifact_policy_sha256": _sha256_bytes(policy_bytes),
                "smoke_lineage_sha256": _sha256_bytes(lineage_bytes),
                "estimand_excluded": True,
                "cells_directory_empty_at_initialization": True,
                "smoke_attempt": (
                    None if attempt_binding is None else dict(attempt_binding)
                ),
            }
            _atomic_bytes(staged / MARKER_FILENAME, _json_bytes(marker))
            _verify_suite(
                staged,
                config_path=config_path,
                expected_cells=expected_cells,
                expected_policy_sha256=_sha256_bytes(policy_bytes),
                expected_release_id=release_id,
                expected_git_commit=git_commit,
                expected_source_tree_sha256=source_tree_sha256,
                expected_harness_environment_sha256=harness_environment_sha256,
                expected_serving_environment_sha256=serving_environment_sha256,
                expected_model_contract_sha256=model_contracts.sha256,
                expected_attempt_binding=attempt_binding,
            )
            if target.exists() or target.is_symlink():
                raise SmokeInitializationError(f"smoke target appeared during staging: {target}")
            os.replace(staged, target)
            _fsync_directory(results_root)
            stage_parent.rmdir()
        except Exception:
            # Preserve every incomplete preimage for explicit operator inspection.
            raise
        reports.append(
            {"status": "initialized"}
            | _verify_suite(
                target,
                config_path=config_path,
                expected_cells=expected_cells,
                expected_release_id=release_id,
                expected_git_commit=git_commit,
                expected_source_tree_sha256=source_tree_sha256,
                expected_harness_environment_sha256=harness_environment_sha256,
                expected_serving_environment_sha256=serving_environment_sha256,
                expected_model_contract_sha256=model_contracts.sha256,
                expected_attempt_binding=attempt_binding,
            )
        )
    return {
        "status": "initialized" if apply else "dry_run",
        "total_cells": sum(int(report["cell_count"]) for report in reports),
        "model_contract_sha256": model_contracts.sha256,
        "runs": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--release-worktree", required=True, type=Path)
    parser.add_argument("--model-contract", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--harness-environment-sha256", required=True)
    parser.add_argument("--serving-environment-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = initialize_all(
            results_root=args.results_root,
            release_worktree=args.release_worktree,
            model_contract_path=args.model_contract,
            release_id=args.release_id,
            git_commit=args.git_commit,
            source_tree_sha256=args.source_tree_sha256,
            harness_environment_sha256=args.harness_environment_sha256,
            serving_environment_sha256=args.serving_environment_sha256,
            apply=args.apply,
        )
    except (OSError, ValueError, SmokeInitializationError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
