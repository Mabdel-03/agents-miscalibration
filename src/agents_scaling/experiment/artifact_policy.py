"""Strict run-root policy for the homogeneous schema-5 production sweep.

Legacy runs have no policy sidecar and remain readable under their historical artifact
contracts.  The presence of ``artifact_policy.schema5-v1.json`` switches a run to the
fail-closed production contract: exact schema 5, immutable manifest/benchmark/model
pins, a frozen release and pair of environments, and the complete metadata provenance
required for every authoritative cell.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


POLICY_SCHEMA_VERSION = 1
POLICY_FILENAME = "artifact_policy.schema5-v1.json"
POLICY_CHECKSUM_FILENAME = "artifact_policy.schema5-v1.sha256"
POLICY_DOMAIN = "agents_scaling.artifact_policy.schema5-v1"
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

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "run_id",
        "authoritative",
        "required_artifact_schema_version",
        "accepted_manifest_sha256",
        "accepted_benchmark_contracts_sha256",
        "accepted_model_contract_sha256",
        "release",
        "environment",
        "required_metadata_fields",
        "legacy_result_import_allowed",
    }
)
_RELEASE_FIELDS = frozenset({"release_id", "git_commit", "source_tree_sha256"})
_ENVIRONMENT_FIELDS = frozenset({"harness_sha256", "serving_sha256"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArtifactPolicyError(RuntimeError):
    """A schema-5 run policy is absent, malformed, or has drifted."""


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


@dataclass(frozen=True)
class ReleasePins:
    release_id: str
    git_commit: str
    source_tree_sha256: str


@dataclass(frozen=True)
class EnvironmentPins:
    harness_sha256: str
    serving_sha256: str


@dataclass(frozen=True)
class ArtifactPolicy:
    path: Path
    checksum_path: Path
    file_sha256: str
    policy_id: str
    run_id: str
    accepted_manifest_sha256: str
    accepted_benchmark_contracts_sha256: str
    accepted_model_contract_sha256: str
    release: ReleasePins
    environment: EnvironmentPins
    required_metadata_fields: tuple[str, ...]
    required_artifact_schema_version: int = 5
    authoritative: bool = True
    legacy_result_import_allowed: bool = False


def _read_checksum(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ArtifactPolicyError(f"missing regular artifact-policy checksum: {path}")
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise ArtifactPolicyError(f"cannot read {path}: {exc}") from exc
    if (
        len(fields) != 2
        or not _is_sha256(fields[0])
        or fields[1] != POLICY_FILENAME
    ):
        raise ArtifactPolicyError(f"invalid artifact-policy checksum record: {path}")
    return str(fields[0])


def load_artifact_policy(
    run_root: str | Path,
    *,
    required: bool = False,
    expected_file_sha256: str | None = None,
) -> ArtifactPolicy | None:
    """Load the frozen policy, or return ``None`` only for a wholly legacy run."""

    root = Path(run_root)
    path = root / POLICY_FILENAME
    checksum_path = root / POLICY_CHECKSUM_FILENAME
    if not path.exists() and not checksum_path.exists():
        if required:
            raise ArtifactPolicyError(f"missing schema-5 artifact policy: {path}")
        return None
    if path.is_symlink() or not path.is_file():
        raise ArtifactPolicyError(f"artifact policy must be a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactPolicyError(f"cannot read {path}: {exc}") from exc
    file_sha256 = hashlib.sha256(raw).hexdigest()
    recorded_sha256 = _read_checksum(checksum_path)
    if file_sha256 != recorded_sha256:
        raise ArtifactPolicyError(
            f"artifact-policy checksum mismatch: expected {recorded_sha256}, "
            f"observed {file_sha256}"
        )
    if expected_file_sha256 is not None and expected_file_sha256 != file_sha256:
        raise ArtifactPolicyError(
            f"dispatcher artifact-policy pin {expected_file_sha256} does not match "
            f"{file_sha256}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPolicyError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _ROOT_FIELDS:
        raise ArtifactPolicyError("artifact policy has the wrong root fields")
    if payload["schema_version"] != POLICY_SCHEMA_VERSION:
        raise ArtifactPolicyError(
            f"unsupported artifact-policy schema {payload['schema_version']!r}"
        )
    if payload["run_id"] != root.name:
        raise ArtifactPolicyError(
            f"artifact policy run_id {payload['run_id']!r} does not match {root.name!r}"
        )
    if payload["authoritative"] is not True:
        raise ArtifactPolicyError("schema-5 production policy must be authoritative")
    if payload["required_artifact_schema_version"] != 5:
        raise ArtifactPolicyError("production policy must require artifact schema 5")
    if payload["legacy_result_import_allowed"] is not False:
        raise ArtifactPolicyError("authoritative schema-5 runs forbid legacy result import")
    for field in (
        "accepted_manifest_sha256",
        "accepted_benchmark_contracts_sha256",
        "accepted_model_contract_sha256",
    ):
        if not _is_sha256(payload[field]):
            raise ArtifactPolicyError(f"artifact policy {field} must be lowercase SHA-256")

    release = payload["release"]
    if not isinstance(release, dict) or set(release) != _RELEASE_FIELDS:
        raise ArtifactPolicyError("artifact policy release has the wrong fields")
    if (
        not isinstance(release["release_id"], str)
        or not release["release_id"]
        or not isinstance(release["git_commit"], str)
        or not release["git_commit"]
        or not _is_sha256(release["source_tree_sha256"])
    ):
        raise ArtifactPolicyError("artifact policy release pins are invalid")

    environment = payload["environment"]
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ArtifactPolicyError("artifact policy environment has the wrong fields")
    if not all(_is_sha256(environment[field]) for field in _ENVIRONMENT_FIELDS):
        raise ArtifactPolicyError("artifact policy environment pins must be SHA-256")

    required_fields = payload["required_metadata_fields"]
    if not isinstance(required_fields, list) or tuple(required_fields) != REQUIRED_METADATA_FIELDS:
        raise ArtifactPolicyError(
            "artifact policy required_metadata_fields do not match the schema-5 contract"
        )

    policy_payload: Mapping[str, Any] = {
        key: value for key, value in payload.items() if key != "policy_id"
    }
    expected_policy_id = hashlib.sha256(
        _canonical_bytes({"domain": POLICY_DOMAIN, "payload": policy_payload})
    ).hexdigest()
    if payload["policy_id"] != expected_policy_id:
        raise ArtifactPolicyError(
            f"artifact policy id mismatch: expected {expected_policy_id}, "
            f"observed {payload['policy_id']!r}"
        )

    return ArtifactPolicy(
        path=path.resolve(),
        checksum_path=checksum_path.resolve(),
        file_sha256=file_sha256,
        policy_id=expected_policy_id,
        run_id=root.name,
        accepted_manifest_sha256=str(payload["accepted_manifest_sha256"]),
        accepted_benchmark_contracts_sha256=str(
            payload["accepted_benchmark_contracts_sha256"]
        ),
        accepted_model_contract_sha256=str(payload["accepted_model_contract_sha256"]),
        release=ReleasePins(**release),
        environment=EnvironmentPins(**environment),
        required_metadata_fields=tuple(required_fields),
    )
