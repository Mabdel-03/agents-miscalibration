#!/usr/bin/env python3
"""Materialize the initial production schema-5 effective fleet.

The frozen release contains the immutable 22-replica/24-GPU base fleet.  The
initial protected-capacity and throughput-qualification generation must use that
fleet exactly: its additive delta is zero and the effective contract is
byte-for-byte identical to the tagged base contract.  Additional replicas may be
introduced only after a sealed throughput-qualification failure through the
controlled capacity-transition workflow.

This tool constructs the initial effective-fleet transaction from the tagged base;
operators must never hand-author it.  Publication is marker-first/marker-last and
create-once:

``EFFECTIVE_FLEET_INTENT.json``
    freezes the exact release, inputs, output bytes, and fixed delta;
``schema5_fleet.effective.v1.json`` and its ``.sha256`` sidecar
    are published as read-only regular files; and
``EFFECTIVE_FLEET_COMPLETE.json``
    is published last after the additive and runtime fleet validators pass.

Dry-run is the default.  Replaying ``materialize --apply`` revalidates and adopts
the exact transaction without rewriting completed artifacts.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.serving.fleet_contract import (  # noqa: E402
    EXPECTED_COUNTS,
    FleetContractError,
    expected_replica_id,
    expected_scheduler_job_name,
    load_fleet_contract,
)
from agents_scaling.serving.model_contracts import (  # noqa: E402
    ModelContractError,
    load_model_contracts,
)
from slurm import schema5_control as control  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r7-effective-fleet-materialization-v1"
RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r7"
CHAIN_NAMESPACE = "schema5-v1.2-r7"

BASE_FILENAME = "schema5_fleet.v1.json"
MODEL_FILENAME = "model_contracts.v1.json"
OUTPUT_FILENAME = "schema5_fleet.effective.v1.json"
CHECKSUM_FILENAME = "schema5_fleet.effective.v1.sha256"
INTENT_FILENAME = "EFFECTIVE_FLEET_INTENT.json"
COMPLETE_FILENAME = "EFFECTIVE_FLEET_COMPLETE.json"

FIXED_ADDITIVE_PROFILE_DELTA: Mapping[str, int] = {
    "0.6B": 0,
    "1.7B": 0,
    "4B": 0,
    "8B": 0,
    "14B": 0,
    "32B": 0,
    "0.6B-long": 0,
    "1.7B-long": 0,
    "4B-long": 0,
    "8B-long": 0,
    "14B-long": 0,
    "32B-long": 0,
}
EXPECTED_EFFECTIVE_COUNTS: Mapping[str, int] = {
    profile: EXPECTED_COUNTS[profile] + FIXED_ADDITIVE_PROFILE_DELTA[profile]
    for profile in EXPECTED_COUNTS
}
EXPECTED_BASE_LOGICAL_REPLICAS = 22
EXPECTED_BASE_GPUS = 24
EXPECTED_ADDITIVE_LOGICAL_REPLICAS = 0
EXPECTED_ADDITIVE_GPUS = 0
EXPECTED_EFFECTIVE_LOGICAL_REPLICAS = 22
EXPECTED_EFFECTIVE_GPUS = 24

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")


class EffectiveFleetMaterializationError(RuntimeError):
    """The effective fleet cannot be derived or published truthfully."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise EffectiveFleetMaterializationError(
            f"cannot self-hash an object already containing {field!r}"
        )
    result = copy.deepcopy(dict(value))
    result[field] = _sha256_bytes(_canonical_bytes(result))
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EffectiveFleetMaterializationError(
                f"JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def _decode_object(raw: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EffectiveFleetMaterializationError(
                    f"{description} contains non-finite number {token!r}"
                )
            ),
        )
    except EffectiveFleetMaterializationError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EffectiveFleetMaterializationError(
            f"{description} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise EffectiveFleetMaterializationError(
            f"{description} must contain one JSON object"
        )
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _canonical_existing_path(
    path: Path,
    *,
    description: str,
    kind: str,
) -> Path:
    supplied = Path(path).expanduser()
    lexical = _lexical_absolute(supplied)
    if not supplied.is_absolute() or supplied != lexical:
        raise EffectiveFleetMaterializationError(
            f"{description} must be a canonical absolute path"
        )
    try:
        resolved = lexical.resolve(strict=True)
        metadata = lexical.stat(follow_symlinks=False)
    except (OSError, RuntimeError) as exc:
        raise EffectiveFleetMaterializationError(
            f"{description} is unavailable: {exc}"
        ) from exc
    if resolved != lexical:
        raise EffectiveFleetMaterializationError(
            f"{description} traverses a symlink"
        )
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise EffectiveFleetMaterializationError(
            f"{description} is not a regular file"
        )
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise EffectiveFleetMaterializationError(
            f"{description} is not a directory"
        )
    return lexical


def _stable_sealed_bytes(path: Path, *, description: str) -> bytes:
    canonical = _canonical_existing_path(
        path, description=description, kind="file"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise EffectiveFleetMaterializationError(
            f"cannot open {description}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    current = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) & 0o222
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(current) != identity(after)
    ):
        raise EffectiveFleetMaterializationError(
            f"{description} is mutable, linked, or changed while read"
        )
    return b"".join(chunks)


def _git(
    worktree: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            check=False,
            capture_output=True,
            text=text,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EffectiveFleetMaterializationError(
            f"cannot verify release Git identity: {exc}"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise EffectiveFleetMaterializationError(
            "cannot verify release Git identity: " + str(stderr).strip()
        )
    return result.stdout


def _verify_release(
    *,
    release_worktree: Path,
    release_git_commit: str,
    release_tag_object: str,
    base_fleet_contract: Path,
    model_contract: Path,
) -> dict[str, str]:
    if (
        _GIT_OBJECT_RE.fullmatch(release_git_commit) is None
        or _GIT_OBJECT_RE.fullmatch(release_tag_object) is None
    ):
        raise EffectiveFleetMaterializationError(
            "release commit and annotated-tag object must be lowercase 40-hex IDs"
        )
    root = _canonical_existing_path(
        release_worktree,
        description="release worktree",
        kind="directory",
    )
    expected_paths = {
        "configs/schema5_fleet.v1.json": base_fleet_contract,
        "configs/schema5_fleet.v1.sha256": (
            base_fleet_contract.with_suffix(".sha256")
        ),
        "configs/model_contracts.v1.json": model_contract,
        "configs/model_contracts.v1.sha256": (
            model_contract.with_suffix(".sha256")
        ),
        "scripts/materialize_schema5_effective_fleet.py": Path(__file__),
        "slurm/dispatch_sweeps.py": (
            root / "slurm" / "dispatch_sweeps.py"
        ),
        "scripts/run_schema5_throughput_qualification.py": (
            root
            / "scripts"
            / "run_schema5_throughput_qualification.py"
        ),
    }
    observed_paths: dict[str, Path] = {}
    for relative, supplied in expected_paths.items():
        expected = (root / relative).resolve()
        observed = _canonical_existing_path(
            Path(supplied),
            description=f"release artifact {relative}",
            kind="file",
        )
        if observed != expected:
            raise EffectiveFleetMaterializationError(
                f"release artifact is not the exact tagged path: {relative}"
            )
        observed_paths[relative] = observed

    def git_text(*arguments: str) -> str:
        output = _git(root, *arguments, text=True)
        assert isinstance(output, str)
        return output.strip()

    head = git_text("rev-parse", "HEAD")
    top_level = Path(git_text("rev-parse", "--show-toplevel")).resolve()
    tag_object = git_text("rev-parse", f"refs/tags/{RELEASE_TAG}")
    tag_type = git_text("cat-file", "-t", tag_object)
    peeled = git_text("rev-parse", f"{tag_object}^{{commit}}")
    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    tagged_hashes: dict[str, str] = {}
    for relative, local_path in observed_paths.items():
        blob_id = git_text("rev-parse", f"{peeled}:{relative}")
        if (
            _GIT_OBJECT_RE.fullmatch(blob_id) is None
            or git_text("cat-file", "-t", blob_id) != "blob"
        ):
            raise EffectiveFleetMaterializationError(
                f"tagged release artifact is not one Git blob: {relative}"
            )
        raw = _git(root, "cat-file", "blob", blob_id, text=False)
        assert isinstance(raw, bytes)
        local_raw = _stable_sealed_bytes(
            local_path, description=f"sealed release artifact {relative}"
        )
        if raw != local_raw:
            raise EffectiveFleetMaterializationError(
                f"local release artifact differs from annotated tag: {relative}"
            )
        tagged_hashes[relative] = _sha256_bytes(raw)
    try:
        source_tree_sha256 = control.sha256_tree(root)
    except (OSError, control.ControlError) as exc:
        raise EffectiveFleetMaterializationError(
            f"cannot hash release source tree: {exc}"
        ) from exc
    if (
        head != release_git_commit
        or peeled != release_git_commit
        or tag_object != release_tag_object
        or tag_type != "tag"
        or top_level != root
        or status
        or git_text("rev-parse", "HEAD") != head
        or git_text("rev-parse", f"refs/tags/{RELEASE_TAG}") != tag_object
        or git_text("cat-file", "-t", tag_object) != tag_type
        or git_text("rev-parse", f"{tag_object}^{{commit}}") != peeled
        or control.sha256_tree(root) != source_tree_sha256
    ):
        raise EffectiveFleetMaterializationError(
            "release worktree is dirty, moving, or not the exact annotated tag"
        )
    return {
        "release_worktree": str(root),
        "release_git_commit": head,
        "release_tag_object": tag_object,
        "source_tree_sha256": source_tree_sha256,
        "publisher_source_sha256": tagged_hashes[
            "scripts/materialize_schema5_effective_fleet.py"
        ],
        "dispatcher_source_sha256": tagged_hashes[
            "slurm/dispatch_sweeps.py"
        ],
        "qualification_runner_source_sha256": tagged_hashes[
            "scripts/run_schema5_throughput_qualification.py"
        ],
        "base_fleet_source_sha256": tagged_hashes[
            "configs/schema5_fleet.v1.json"
        ],
        "model_contract_source_sha256": tagged_hashes[
            "configs/model_contracts.v1.json"
        ],
    }


def _require_checksum_sidecar(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> None:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise EffectiveFleetMaterializationError(
            f"{description} expected SHA-256 is malformed"
        )
    raw = _stable_sealed_bytes(
        path.with_suffix(".sha256"),
        description=f"{description} checksum sidecar",
    )
    if raw != f"{expected_sha256}  {path.name}\n".encode("ascii"):
        raise EffectiveFleetMaterializationError(
            f"{description} checksum sidecar does not bind its exact bytes"
        )


def _construct_effective_payload(base: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(base))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise EffectiveFleetMaterializationError(
            "base fleet profiles are malformed"
        )
    seen: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise EffectiveFleetMaterializationError(
                "base fleet contains a malformed profile"
            )
        name = profile.get("serving_profile")
        replicas = profile.get("replicas")
        if (
            not isinstance(name, str)
            or name not in FIXED_ADDITIVE_PROFILE_DELTA
            or name in seen
            or not isinstance(replicas, list)
            or not replicas
            or len(replicas) != EXPECTED_COUNTS[name]
        ):
            raise EffectiveFleetMaterializationError(
                f"base profile cannot be extended safely: {name!r}"
            )
        seen.add(name)
        template = copy.deepcopy(replicas[-1])
        for index in range(
            len(replicas),
            len(replicas) + FIXED_ADDITIVE_PROFILE_DELTA[name],
        ):
            replica = copy.deepcopy(template)
            replica["replica_index"] = index
            replica["replica_id"] = expected_replica_id(name, index)
            replica["scheduler_job_name"] = expected_scheduler_job_name(
                name, index
            )
            replicas.append(replica)
    if seen != set(FIXED_ADDITIVE_PROFILE_DELTA):
        raise EffectiveFleetMaterializationError(
            "base fleet profile set differs from the fixed production design"
        )
    payload["logical_replica_count"] = EXPECTED_EFFECTIVE_LOGICAL_REPLICAS
    payload["allocated_gpu_count"] = EXPECTED_EFFECTIVE_GPUS
    return payload


def _validate_candidate(
    *,
    base_path: Path,
    model_path: Path,
    model_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> None:
    try:
        models = load_model_contracts(
            model_path, expected_sha256=model_sha256
        )
        base = load_fleet_contract(
            base_path,
            model_contracts=models,
            expected_sha256=_sha256_bytes(
                _stable_sealed_bytes(
                    base_path, description="base fleet contract"
                )
            ),
            allow_capacity_layout=False,
        )
        effective = load_fleet_contract(
            candidate_path,
            model_contracts=models,
            expected_sha256=candidate_sha256,
            allow_capacity_layout=True,
        )
        control._assert_additive_capacity_contract(  # noqa: SLF001
            base_path, candidate_path
        )
    except (
        FleetContractError,
        ModelContractError,
        control.ControlError,
        OSError,
        ValueError,
    ) as exc:
        raise EffectiveFleetMaterializationError(
            f"effective fleet validator rejected the candidate: {exc}"
        ) from exc
    base_counts = {
        profile: len(base.by_profile[profile])
        for profile in sorted(base.by_profile)
    }
    effective_counts = {
        profile: len(effective.by_profile[profile])
        for profile in sorted(effective.by_profile)
    }
    delta = {
        profile: effective_counts[profile] - base_counts[profile]
        for profile in base_counts
    }
    if (
        base_counts != dict(EXPECTED_COUNTS)
        or effective_counts != dict(EXPECTED_EFFECTIVE_COUNTS)
        or delta != dict(FIXED_ADDITIVE_PROFILE_DELTA)
        or len(base.replicas) != EXPECTED_BASE_LOGICAL_REPLICAS
        or sum(row.gpus_per_replica for row in base.replicas)
        != EXPECTED_BASE_GPUS
        or len(effective.replicas)
        != EXPECTED_EFFECTIVE_LOGICAL_REPLICAS
        or sum(row.gpus_per_replica for row in effective.replicas)
        != EXPECTED_EFFECTIVE_GPUS
    ):
        raise EffectiveFleetMaterializationError(
            "initial effective fleet differs from the exact zero-delta 22/24 design"
        )


def _prepare_transaction(
    *,
    release_worktree: Path,
    release_git_commit: str,
    release_tag_object: str,
    base_fleet_contract: Path,
    base_fleet_contract_sha256: str,
    model_contract: Path,
    model_contract_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    root = _canonical_existing_path(
        release_worktree,
        description="release worktree",
        kind="directory",
    )
    base_path = _canonical_existing_path(
        base_fleet_contract,
        description="base fleet contract",
        kind="file",
    )
    model_path = _canonical_existing_path(
        model_contract,
        description="model contract",
        kind="file",
    )
    base_raw = _stable_sealed_bytes(
        base_path, description="base fleet contract"
    )
    model_raw = _stable_sealed_bytes(
        model_path, description="model contract"
    )
    if (
        _sha256_bytes(base_raw) != base_fleet_contract_sha256
        or _sha256_bytes(model_raw) != model_contract_sha256
    ):
        raise EffectiveFleetMaterializationError(
            "base fleet or model-contract hash differs from the supplied authority"
        )
    _require_checksum_sidecar(
        base_path,
        expected_sha256=base_fleet_contract_sha256,
        description="base fleet contract",
    )
    _require_checksum_sidecar(
        model_path,
        expected_sha256=model_contract_sha256,
        description="model contract",
    )
    release = _verify_release(
        release_worktree=root,
        release_git_commit=release_git_commit,
        release_tag_object=release_tag_object,
        base_fleet_contract=base_path,
        model_contract=model_path,
    )
    if (
        release["base_fleet_source_sha256"]
        != base_fleet_contract_sha256
        or release["model_contract_source_sha256"]
        != model_contract_sha256
    ):
        raise EffectiveFleetMaterializationError(
            "supplied base/model hashes differ from their exact tagged blobs"
        )
    base_payload = _decode_object(
        base_raw, description="base fleet contract"
    )
    candidate_payload = _construct_effective_payload(base_payload)
    if candidate_payload != base_payload:
        raise EffectiveFleetMaterializationError(
            "zero-delta effective fleet changed tagged base semantics"
        )
    candidate_raw = base_raw
    candidate_sha256 = _sha256_bytes(candidate_raw)
    checksum_raw = (
        f"{candidate_sha256}  {OUTPUT_FILENAME}\n".encode("ascii")
    )

    supplied_output_root = Path(output_root).expanduser()
    lexical_output_root = _lexical_absolute(supplied_output_root)
    if (
        not supplied_output_root.is_absolute()
        or supplied_output_root != lexical_output_root
    ):
        raise EffectiveFleetMaterializationError(
            "output root must be a canonical absolute path"
        )
    parent = _canonical_existing_path(
        lexical_output_root.parent,
        description="effective fleet output parent",
        kind="directory",
    )
    if lexical_output_root.parent != parent:
        raise EffectiveFleetMaterializationError(
            "effective fleet output parent is aliased"
        )
    if lexical_output_root.exists() or lexical_output_root.is_symlink():
        _canonical_existing_path(
            lexical_output_root,
            description="effective fleet output root",
            kind="directory",
        )
    paths = {
        "intent": str(lexical_output_root / INTENT_FILENAME),
        "effective_fleet_contract": str(
            lexical_output_root / OUTPUT_FILENAME
        ),
        "effective_fleet_contract_checksum": str(
            lexical_output_root / CHECKSUM_FILENAME
        ),
        "completion_marker": str(
            lexical_output_root / COMPLETE_FILENAME
        ),
    }
    shared = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "release_git_commit": release_git_commit,
        "release_tag_object": release_tag_object,
        "chain_namespace": CHAIN_NAMESPACE,
        "release_worktree": str(root),
        "source_tree_sha256": release["source_tree_sha256"],
        "publisher_source_sha256": release[
            "publisher_source_sha256"
        ],
        "dispatcher_source_sha256": release[
            "dispatcher_source_sha256"
        ],
        "qualification_runner_source_sha256": release[
            "qualification_runner_source_sha256"
        ],
        "base_fleet_contract": {
            "path": str(base_path),
            "sha256": base_fleet_contract_sha256,
        },
        "model_contract": {
            "path": str(model_path),
            "sha256": model_contract_sha256,
        },
        "fixed_additive_profile_delta": dict(
            FIXED_ADDITIVE_PROFILE_DELTA
        ),
        "base_profile_replicas": dict(EXPECTED_COUNTS),
        "effective_profile_replicas": dict(
            EXPECTED_EFFECTIVE_COUNTS
        ),
        "base_logical_replicas": EXPECTED_BASE_LOGICAL_REPLICAS,
        "base_allocated_gpus": EXPECTED_BASE_GPUS,
        "additive_logical_replicas": (
            EXPECTED_ADDITIVE_LOGICAL_REPLICAS
        ),
        "additive_allocated_gpus": EXPECTED_ADDITIVE_GPUS,
        "effective_logical_replicas": (
            EXPECTED_EFFECTIVE_LOGICAL_REPLICAS
        ),
        "effective_allocated_gpus": EXPECTED_EFFECTIVE_GPUS,
        "effective_fleet_contract": {
            "path": paths["effective_fleet_contract"],
            "sha256": candidate_sha256,
            "size": len(candidate_raw),
            "checksum_path": paths[
                "effective_fleet_contract_checksum"
            ],
            "checksum_sha256": _sha256_bytes(checksum_raw),
        },
    }
    intent = _identity(
        {
            **shared,
            "output_root": str(lexical_output_root),
            "publication_order": [
                INTENT_FILENAME,
                OUTPUT_FILENAME,
                CHECKSUM_FILENAME,
                COMPLETE_FILENAME,
            ],
        },
        "intent_id",
    )
    completion = _identity(
        {
            **shared,
            "passed": True,
            "intent_id": intent["intent_id"],
            "intent_path": paths["intent"],
            "additive_overlay_contract": {
                "path": paths["effective_fleet_contract"],
                "sha256": candidate_sha256,
            },
            "runbook_inputs": {
                "base_fleet_contract": str(base_path),
                "base_fleet_contract_sha256": (
                    base_fleet_contract_sha256
                ),
                "effective_fleet_contract": paths[
                    "effective_fleet_contract"
                ],
                "effective_fleet_contract_sha256": (
                    candidate_sha256
                ),
                "additive_overlay_contract": paths[
                    "effective_fleet_contract"
                ],
                "additive_overlay_contract_sha256": (
                    candidate_sha256
                ),
                "model_contract": str(model_path),
                "model_contract_sha256": model_contract_sha256,
                "source_tree_sha256": release[
                    "source_tree_sha256"
                ],
                "dispatcher_source": str(
                    root / "slurm" / "dispatch_sweeps.py"
                ),
                "dispatcher_source_sha256": release[
                    "dispatcher_source_sha256"
                ],
                "qualification_runner_source": str(
                    root
                    / "scripts"
                    / "run_schema5_throughput_qualification.py"
                ),
                "qualification_runner_source_sha256": release[
                    "qualification_runner_source_sha256"
                ],
            },
        },
        "marker_id",
    )

    # Exercise the production loaders against the exact bytes before returning even
    # a dry-run report.
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="schema5-effective-fleet-validation-"
    ) as temporary:
        candidate_path = Path(temporary) / OUTPUT_FILENAME
        sidecar_path = candidate_path.with_suffix(".sha256")
        candidate_path.write_bytes(candidate_raw)
        sidecar_path.write_bytes(checksum_raw)
        candidate_path.chmod(0o444)
        sidecar_path.chmod(0o444)
        _validate_candidate(
            base_path=base_path,
            model_path=model_path,
            model_sha256=model_contract_sha256,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
        )
    return {
        "release": release,
        "root": lexical_output_root,
        "paths": paths,
        "candidate_raw": candidate_raw,
        "checksum_raw": checksum_raw,
        "intent": intent,
        "intent_raw": _canonical_bytes(intent),
        "completion": completion,
        "completion_raw": _canonical_bytes(completion),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise EffectiveFleetMaterializationError(
            f"cannot durably fsync publication directory {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def _open_publication_lock(path: Path) -> int:
    """Open one non-aliased sibling lock and bind the pathname to its inode."""

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise EffectiveFleetMaterializationError(
            f"cannot safely open effective-fleet publication lock {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o077
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
            )
            != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_nlink,
            )
        ):
            raise EffectiveFleetMaterializationError(
                "effective-fleet publication lock is aliased, linked, or unsafe"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            (
                locked.st_dev,
                locked.st_ino,
                locked.st_mode,
                locked.st_nlink,
            )
            != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_nlink,
            )
            or locked.st_nlink != 1
        ):
            raise EffectiveFleetMaterializationError(
                "effective-fleet publication lock changed while being acquired"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_exact_sealed_file(
    path: Path,
    expected: bytes,
    *,
    description: str,
) -> None:
    observed = _stable_sealed_bytes(path, description=description)
    if observed != expected:
        raise EffectiveFleetMaterializationError(
            f"{description} conflicts with the frozen transaction"
        )


def _publish_once(
    path: Path,
    value: bytes,
    *,
    transaction_id: str,
    description: str,
) -> None:
    if path.exists() or path.is_symlink():
        _require_exact_sealed_file(
            path, value, description=description
        )
        return
    temporary = path.parent / (
        f".{path.name}.{transaction_id}.pending"
    )
    if temporary.exists() or temporary.is_symlink():
        try:
            metadata = temporary.stat(follow_symlinks=False)
        except OSError as exc:
            raise EffectiveFleetMaterializationError(
                f"cannot inspect interrupted {description}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or temporary.is_symlink()
        ):
            raise EffectiveFleetMaterializationError(
                f"interrupted {description} path is unsafe"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o222 == 0:
            # A kill after the temp file's fchmod/fsync but before rename leaves
            # a fully sealed preimage. Adopt only the exact bytes; never make a
            # conflicting sealed pending artifact writable.
            _require_exact_sealed_file(
                temporary,
                value,
                description=f"interrupted sealed {description}",
            )
            os.replace(temporary, path)
            _fsync_directory(path.parent)
            _require_exact_sealed_file(
                path, value, description=description
            )
            return
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    _require_exact_sealed_file(path, value, description=description)


def _verify_complete(
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(transaction["root"])
    expected = {
        INTENT_FILENAME: transaction["intent_raw"],
        OUTPUT_FILENAME: transaction["candidate_raw"],
        CHECKSUM_FILENAME: transaction["checksum_raw"],
        COMPLETE_FILENAME: transaction["completion_raw"],
    }
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise EffectiveFleetMaterializationError(
            "effective fleet output root is missing or unsafe"
        )
    entries = {entry.name for entry in root.iterdir()}
    if entries != set(expected):
        raise EffectiveFleetMaterializationError(
            "effective fleet output root contains missing or unexpected artifacts: "
            f"{sorted(entries ^ set(expected))}"
        )
    for name, raw in expected.items():
        _require_exact_sealed_file(
            root / name,
            raw,
            description=f"effective fleet artifact {name}",
        )
    marker = _decode_object(
        transaction["completion_raw"],
        description="expected effective fleet completion marker",
    )
    if (
        marker.get("marker_id")
        != _identity(
            {
                key: value
                for key, value in marker.items()
                if key != "marker_id"
            },
            "marker_id",
        )["marker_id"]
    ):
        raise EffectiveFleetMaterializationError(
            "effective fleet completion marker self-hash is invalid"
        )
    _validate_candidate(
        base_path=Path(
            marker["base_fleet_contract"]["path"]
        ),
        model_path=Path(marker["model_contract"]["path"]),
        model_sha256=str(marker["model_contract"]["sha256"]),
        candidate_path=root / OUTPUT_FILENAME,
        candidate_sha256=str(
            marker["effective_fleet_contract"]["sha256"]
        ),
    )
    return {
        "status": "complete",
        "passed": True,
        "protocol": PROTOCOL,
        "intent_id": transaction["intent"]["intent_id"],
        "marker_id": transaction["completion"]["marker_id"],
        "completion_marker": str(root / COMPLETE_FILENAME),
        "effective_fleet_contract": str(root / OUTPUT_FILENAME),
        "effective_fleet_contract_sha256": _sha256_bytes(
            transaction["candidate_raw"]
        ),
        "additive_overlay_contract": str(root / OUTPUT_FILENAME),
        "additive_overlay_contract_sha256": _sha256_bytes(
            transaction["candidate_raw"]
        ),
        "runbook_inputs": copy.deepcopy(
            transaction["completion"]["runbook_inputs"]
        ),
    }


def materialize(
    *,
    release_worktree: Path,
    release_git_commit: str,
    release_tag_object: str,
    base_fleet_contract: Path,
    base_fleet_contract_sha256: str,
    model_contract: Path,
    model_contract_sha256: str,
    output_root: Path,
    apply: bool,
) -> dict[str, Any]:
    transaction = _prepare_transaction(
        release_worktree=release_worktree,
        release_git_commit=release_git_commit,
        release_tag_object=release_tag_object,
        base_fleet_contract=base_fleet_contract,
        base_fleet_contract_sha256=base_fleet_contract_sha256,
        model_contract=model_contract,
        model_contract_sha256=model_contract_sha256,
        output_root=output_root,
    )
    if not apply:
        return {
            "status": "dry_run",
            "passed": True,
            "would_publish": [
                transaction["paths"]["intent"],
                transaction["paths"]["effective_fleet_contract"],
                transaction["paths"][
                    "effective_fleet_contract_checksum"
                ],
                transaction["paths"]["completion_marker"],
            ],
            "intent_id": transaction["intent"]["intent_id"],
            "marker_id": transaction["completion"]["marker_id"],
            "effective_fleet_contract_sha256": _sha256_bytes(
                transaction["candidate_raw"]
            ),
            "base_profile_replicas": dict(EXPECTED_COUNTS),
            "fixed_additive_profile_delta": dict(
                FIXED_ADDITIVE_PROFILE_DELTA
            ),
            "effective_profile_replicas": dict(
                EXPECTED_EFFECTIVE_COUNTS
            ),
            "runbook_inputs": copy.deepcopy(
                transaction["completion"]["runbook_inputs"]
            ),
        }
    root = Path(transaction["root"])
    lock_path = root.parent / f".{root.name}.materialize.lock"
    descriptor = _open_publication_lock(lock_path)
    try:
        if not root.exists():
            os.mkdir(root, 0o750)
            _fsync_directory(root.parent)
        _canonical_existing_path(
            root,
            description="effective fleet output root",
            kind="directory",
        )
        # The first plan necessarily precedes lock acquisition. Rebuild it under
        # the sibling lock so neither a completed replay nor a fresh publication
        # can accept source/input drift from that interval.
        locked = _prepare_transaction(
            release_worktree=release_worktree,
            release_git_commit=release_git_commit,
            release_tag_object=release_tag_object,
            base_fleet_contract=base_fleet_contract,
            base_fleet_contract_sha256=base_fleet_contract_sha256,
            model_contract=model_contract,
            model_contract_sha256=model_contract_sha256,
            output_root=output_root,
        )
        for key in (
            "intent_raw",
            "candidate_raw",
            "checksum_raw",
            "completion_raw",
        ):
            if locked[key] != transaction[key]:
                raise EffectiveFleetMaterializationError(
                    "release or fleet inputs changed before publication lock"
                )
        transaction = locked
        transaction_id = str(transaction["intent"]["intent_id"])
        completion_path = root / COMPLETE_FILENAME
        if completion_path.exists() or completion_path.is_symlink():
            result = _verify_complete(transaction)
            result["status"] = "already_complete"
            return result
        _publish_once(
            root / INTENT_FILENAME,
            transaction["intent_raw"],
            transaction_id=transaction_id,
            description="effective fleet intent",
        )
        _publish_once(
            root / OUTPUT_FILENAME,
            transaction["candidate_raw"],
            transaction_id=transaction_id,
            description="effective fleet contract",
        )
        _publish_once(
            root / CHECKSUM_FILENAME,
            transaction["checksum_raw"],
            transaction_id=transaction_id,
            description="effective fleet checksum",
        )
        # Rebuild the trust binding immediately before the marker-last commit.
        fresh = _prepare_transaction(
            release_worktree=release_worktree,
            release_git_commit=release_git_commit,
            release_tag_object=release_tag_object,
            base_fleet_contract=base_fleet_contract,
            base_fleet_contract_sha256=base_fleet_contract_sha256,
            model_contract=model_contract,
            model_contract_sha256=model_contract_sha256,
            output_root=output_root,
        )
        for key in (
            "intent_raw",
            "candidate_raw",
            "checksum_raw",
            "completion_raw",
        ):
            if fresh[key] != transaction[key]:
                raise EffectiveFleetMaterializationError(
                    "release or fleet inputs changed before marker-last publication"
                )
        _publish_once(
            root / COMPLETE_FILENAME,
            transaction["completion_raw"],
            transaction_id=transaction_id,
            description="effective fleet completion marker",
        )
        return _verify_complete(transaction)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def verify(
    **arguments: Any,
) -> dict[str, Any]:
    transaction = _prepare_transaction(**arguments)
    return _verify_complete(transaction)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-worktree", required=True, type=Path)
    parser.add_argument("--release-git-commit", required=True)
    parser.add_argument("--release-tag-object", required=True)
    parser.add_argument(
        "--base-fleet-contract", required=True, type=Path
    )
    parser.add_argument("--base-fleet-contract-sha256", required=True)
    parser.add_argument("--model-contract", required=True, type=Path)
    parser.add_argument("--model-contract-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "materialize",
        help="derive and optionally publish the fixed effective fleet",
    )
    _add_common_arguments(publish)
    publish.add_argument("--apply", action="store_true")
    verify_parser = commands.add_parser(
        "verify",
        help="recompute and verify the marker-last publication",
    )
    _add_common_arguments(verify_parser)
    verify_parser.add_argument(
        "--format",
        choices=("json", "runbook-nul"),
        default="json",
        help=(
            "emit JSON or ten fixed-order NUL-delimited runbook inputs; "
            "runbook-nul is valid only after complete verification"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "release_worktree": args.release_worktree,
        "release_git_commit": args.release_git_commit,
        "release_tag_object": args.release_tag_object,
        "base_fleet_contract": args.base_fleet_contract,
        "base_fleet_contract_sha256": (
            args.base_fleet_contract_sha256
        ),
        "model_contract": args.model_contract,
        "model_contract_sha256": args.model_contract_sha256,
        "output_root": args.output_root,
    }
    try:
        if args.command == "materialize":
            result = materialize(**common, apply=args.apply)
        else:
            result = verify(**common)
    except EffectiveFleetMaterializationError as exc:
        print(f"effective fleet materialization failed closed: {exc}", file=sys.stderr)
        return 2
    if (
        args.command == "verify"
        and args.format == "runbook-nul"
    ):
        inputs = result["runbook_inputs"]
        ordered = (
            inputs["effective_fleet_contract"],
            inputs["effective_fleet_contract_sha256"],
            inputs["additive_overlay_contract"],
            inputs["additive_overlay_contract_sha256"],
            inputs["source_tree_sha256"],
            inputs["dispatcher_source"],
            inputs["dispatcher_source_sha256"],
            inputs["qualification_runner_source"],
            inputs["qualification_runner_source_sha256"],
            result["marker_id"],
        )
        if any(
            not isinstance(value, str) or "\x00" in value
            for value in ordered
        ):
            raise EffectiveFleetMaterializationError(
                "verified runbook input cannot be emitted safely"
            )
        sys.stdout.buffer.write(
            b"\x00".join(value.encode("utf-8") for value in ordered)
            + b"\x00"
        )
    else:
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
