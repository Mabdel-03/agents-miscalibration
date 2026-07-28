#!/usr/bin/env python3
"""Publish the schema-5 protected-capacity prerequisite, fail closed.

This tool is a marker publisher, not a capacity simulator or job submitter.  It
consumes two independently sealed, self-hashed JSON objects and, immediately
before publication, re-executes every recorded read-only scheduler capture:

* scheduler evidence containing all scientific server/client placements, their
  partition and QOS ``PreemptMode`` values, capacity totals, and complete raw
  ``squeue``/``sacct`` captures; and
* canary evidence binding those exact placements and scheduler-evidence identity
  to effective ``Requeue=0`` observations.

Both objects must carry the exact schema-5 v1.2-r11 release and chain namespace.
The release commit and annotated-tag object are also supplied as explicit trust
anchors so that two consistently substituted evidence files cannot authorize a
different release.  The scheduler evidence additionally binds the exact canary
element accounting: 384 clients plus an inclusive 64-job non-cell reserve made up
  of the frozen 22-replica/24-GPU base with zero prelaunch additive replicas,
  three warm-turnover allocations (four GPUs), and 39 held
  controller/monitor/other placeholders.  Thus
  the canary is executable under the minimum accepted 448-job submit limit rather
  than requiring 64 slots in addition to its serving allocations.

The only mutation is an atomic, create-once, mode-0444 publication of
``recovery_root/PROTECTED_CAPACITY_COMPLETE.json`` when ``--apply`` is supplied.
The completion marker is the sole output artifact and is therefore marker-last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from agents_scaling.serving import protected_capacity as runtime_capacity


SCHEMA_VERSION = 4
RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r11"
CHAIN_NAMESPACE = "schema5-v1.2-r11"
PROTOCOL = "schema5-v1.2-r11-protected-capacity-v4"
SCHEDULER_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r11-protected-capacity-scheduler-evidence-v4"
)
CANARY_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r11-protected-capacity-canary-evidence-v4"
)
CAPACITY_SOURCE = (
    "sealed_protected_canary+partition_inventory+association"
)
MARKER_FILENAME = "PROTECTED_CAPACITY_COMPLETE.json"
# Public aliases match the names used by the downstream chain renderer.
PROTECTED_CAPACITY_PROTOCOL = PROTOCOL
PROTECTED_CAPACITY_MARKER_NAME = MARKER_FILENAME

BASE_ACTIVE_GPUS = 24
BASE_LOGICAL_REPLICAS = 22
PRODUCTION_ACTIVE_GPUS = 24
RETAINED_WARM_TURNOVER_GPUS = 4
PRODUCTION_ATTESTED_GPUS = (
    PRODUCTION_ACTIVE_GPUS + RETAINED_WARM_TURNOVER_GPUS
)
MIN_CLIENT_SLOTS = 384
MIN_CLIENT_CPUS = 384
MIN_CLIENT_MEMORY_MIB = 1_572_864
MIN_RESERVE_JOBS = 64
MIN_SUBMIT_HEADROOM = 448
CLIENT_MEMORY_MIB_PER_SLOT = 4_096
EXPECTED_WARM_TURNOVER_JOB_ELEMENTS = 3
EXPECTED_TOTAL_JOB_ELEMENTS = MIN_CLIENT_SLOTS + MIN_RESERVE_JOBS
MIN_SCIENTIFIC_WALL_SECONDS = 86_400
CLIENT_WALL_SECONDS = 43_200

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PLACEMENT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_BINDING_FIELDS = {
    "release_id",
    "release_tag",
    "release_git_commit",
    "release_tag_object",
    "chain_namespace",
}
_SOURCE_TRUST_FIELDS = {
    "source_tree_sha256",
    "dispatcher_source_sha256",
    "qualification_runner_source_sha256",
}
_SCHEDULER_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    *_BINDING_FIELDS,
    *_SOURCE_TRUST_FIELDS,
    "observed_at",
    "preempt_type",
    "capacity_source",
    "scheduler_cluster",
    "scheduler_account",
    "scheduler_user",
    "scheduler_max_jobs",
    "scheduler_max_submit_jobs",
    "running_scientific_jobs",
    "minimum_scientific_wall_seconds",
    "scientific_qos_contracts",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "capacity_generation",
    "base_fleet_contract_path",
    "base_fleet_contract_sha256",
    "effective_fleet_contract_path",
    "effective_fleet_contract_sha256",
    "additive_overlay_contract_path",
    "additive_overlay_contract_sha256",
    "static_feasibility_certificate",
    "base_active_logical_replicas",
    "base_active_gpus",
    "base_active_topology",
    "base_active_topology_sha256",
    "additive_reserved_logical_replicas",
    "additive_reserved_gpus",
    "additive_reserved_tp1_replicas",
    "additive_reserved_tp2_replicas",
    "additive_reserved_topology",
    "additive_reserved_topology_sha256",
    "effective_active_logical_replicas",
    "effective_active_gpus",
    "effective_active_topology",
    "effective_active_topology_sha256",
    "retained_warm_turnover_job_elements",
    "retained_warm_turnover_gpus",
    "retained_warm_turnover_tp1_allocations",
    "retained_warm_turnover_tp2_allocations",
    "retained_warm_turnover_topology",
    "retained_warm_turnover_topology_sha256",
    "attested_total_gpus",
    "fleet_contract_sha256",
    "active_fleet_topology_sha256",
    "builder_source_sha256",
    "publisher_source_sha256",
    "expected_total_job_elements",
    "job_element_accounting",
    "occupancy_preflight",
    "scheduler_configuration",
    "partition_configuration",
    "qos_configuration",
    "association_configuration",
    "base_fleet_contract",
    "effective_fleet_contract",
    "additive_overlay_contract",
    "static_feasibility_certificate_source",
    "builder_source",
    "publisher_source",
    "scientific_server_placements",
    "scientific_client_placements",
    "squeue",
    "sacct",
    "evidence_id",
}
_CANARY_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    *_BINDING_FIELDS,
    "completed_at",
    "scheduler_evidence_id",
    "scientific_server_placements",
    "scientific_client_placements",
    "squeue_complete",
    "sacct_complete",
    "canary_id",
}
_SERVER_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "base_active_gpus",
    "reserved_additive_gpus",
    "effective_active_gpus",
    "retained_warm_turnover_gpus",
    "attested_total_gpus",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "partition_nodes",
}
_CLIENT_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "slots",
    "cpus",
    "memory_mib",
    "reserve_jobs",
    "submit_headroom",
}
_CANARY_PLACEMENT_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "effective_requeue",
}
_SOURCE_FIELDS = {
    "complete",
    "argv",
    "raw_output",
    "raw_output_sha256",
    "record_count",
}
_JOB_ELEMENT_ACCOUNTING_FIELDS = {
    "cell_job_elements",
    "active_server_job_elements",
    "warm_turnover_job_elements",
    "controller_monitor_other_held_job_elements",
    "total_non_cell_reserve_job_elements",
    "total_canary_job_elements",
}
_MARKER_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    *_BINDING_FIELDS,
    *_SOURCE_TRUST_FIELDS,
    "capacity_generation",
    "base_fleet_contract_path",
    "base_fleet_contract_sha256",
    "effective_fleet_contract_path",
    "effective_fleet_contract_sha256",
    "additive_overlay_contract_path",
    "additive_overlay_contract_sha256",
    "static_feasibility_certificate",
    "static_feasibility_wave_passed",
    "static_feasibility_selected_cell_count",
    "static_feasibility_target_cell_count",
    "static_feasibility_shortfall_cells",
    "static_feasibility_configured_client_ceiling",
    "static_feasibility_certified_saturation_target",
    "base_active_logical_replicas",
    "base_active_gpus",
    "base_active_topology",
    "base_active_topology_sha256",
    "additive_reserved_logical_replicas",
    "additive_reserved_gpus",
    "additive_reserved_tp1_replicas",
    "additive_reserved_tp2_replicas",
    "additive_reserved_topology",
    "additive_reserved_topology_sha256",
    "effective_active_logical_replicas",
    "effective_active_gpus",
    "effective_active_topology",
    "effective_active_topology_sha256",
    "retained_warm_turnover_job_elements",
    "retained_warm_turnover_gpus",
    "retained_warm_turnover_tp1_allocations",
    "retained_warm_turnover_tp2_allocations",
    "retained_warm_turnover_topology",
    "retained_warm_turnover_topology_sha256",
    "attested_total_gpus",
    "job_element_accounting",
    "active_gpus",
    "warm_headroom_gpus",
    "cell_ceiling",
    "reserve_jobs",
    "submit_headroom",
    "cpu",
    "memory_mib",
    "preempt_type",
    "capacity_source",
    "scheduler_cluster",
    "scheduler_account",
    "scheduler_user",
    "scheduler_max_jobs",
    "scheduler_max_submit_jobs",
    "running_scientific_jobs",
    "minimum_scientific_wall_seconds",
    "scientific_qos_contracts",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "fleet_contract_sha256",
    "active_fleet_topology_sha256",
    "scientific_server_preempt_mode",
    "scientific_client_preempt_mode",
    "scientific_server_placements",
    "scientific_client_placements",
    "scheduler_evidence_id",
    "scheduler_evidence_sha256",
    "canary_id",
    "canary_evidence_sha256",
    "squeue_complete",
    "sacct_complete",
    "marker_id",
}
_QOS_CONTRACT_FIELDS = {
    "qos",
    "max_wall_seconds",
    "max_jobs_per_user",
    "max_submit_jobs_per_user",
    "required_wall_seconds",
    "required_running_jobs",
    "required_submit_jobs",
}


class ProtectedCapacityError(RuntimeError):
    """Sealed evidence cannot prove the protected-capacity contract."""


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical representation used by every self-hash."""

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


def identity_sha256(value: Mapping[str, Any]) -> str:
    """Hash one JSON identity using the schema-5 canonical representation."""

    return hashlib.sha256(canonical_bytes(dict(value))).hexdigest()


def with_self_hash(value: Mapping[str, Any], *, identity_field: str) -> dict[str, Any]:
    """Return a copy with its canonical self-hash appended."""

    if identity_field in value:
        raise ProtectedCapacityError(
            f"cannot self-hash an identity containing {identity_field!r}"
        )
    result = dict(value)
    result[identity_field] = identity_sha256(result)
    return result


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _path_without_symlinks(
    path: Path,
    *,
    description: str,
    kind: str,
) -> Path:
    lexical = _lexical_absolute(path)
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise ProtectedCapacityError(f"{description} path is unsafe: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtectedCapacityError(
            f"{description} is missing or unsafe: {lexical}: {exc}"
        ) from exc
    if resolved != lexical:
        raise ProtectedCapacityError(f"{description} traverses a symlink: {lexical}")
    if kind == "file" and not lexical.is_file():
        raise ProtectedCapacityError(f"{description} is not a regular file: {lexical}")
    if kind == "directory" and not lexical.is_dir():
        raise ProtectedCapacityError(f"{description} is not a directory: {lexical}")
    return lexical


def _read_sealed_bytes(path: Path, *, description: str) -> tuple[Path, bytes]:
    sealed = _path_without_symlinks(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(sealed, flags)
    except OSError as exc:
        raise ProtectedCapacityError(
            f"cannot open {description} {sealed}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) & 0o222:
            raise ProtectedCapacityError(
                f"{description} must be a read-only regular file: {sealed}"
            )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ProtectedCapacityError(
            f"{description} changed while being read: {sealed}"
        )
    try:
        current = sealed.stat(follow_symlinks=False)
    except OSError as exc:
        raise ProtectedCapacityError(
            f"{description} disappeared after reading: {sealed}: {exc}"
        ) from exc
    if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise ProtectedCapacityError(
            f"{description} was replaced while being read: {sealed}"
        )
    return sealed, b"".join(chunks)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtectedCapacityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ProtectedCapacityError(f"non-finite JSON number {token!r}")


def _decode_json_object(raw: bytes, *, description: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ProtectedCapacityError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtectedCapacityError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtectedCapacityError(
            f"{description} must contain exactly one JSON object"
        )
    return payload


def read_sealed_json(path: Path, *, description: str) -> tuple[Path, dict[str, Any]]:
    """Read one immutable JSON object without following symlinks."""

    sealed, raw = _read_sealed_bytes(path, description=description)
    payload = _decode_json_object(raw, description=description)
    return sealed, payload


def _read_sealed_json_with_sha256(
    path: Path,
    *,
    description: str,
) -> tuple[Path, dict[str, Any], str]:
    sealed, raw = _read_sealed_bytes(path, description=description)
    return (
        sealed,
        _decode_json_object(raw, description=description),
        hashlib.sha256(raw).hexdigest(),
    )


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    description: str,
) -> None:
    if set(value) != expected:
        drift = sorted(set(value) ^ expected)
        raise ProtectedCapacityError(f"{description} fields drifted: {drift}")


def _require_positive_timestamp(value: Any, *, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ProtectedCapacityError(
            f"{description} must be one positive finite timestamp"
        )
    return float(value)


def _require_integer(
    value: Any,
    *,
    minimum: int,
    description: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProtectedCapacityError(f"{description} must be an integer >= {minimum}")
    return value


def _validate_release_binding(
    evidence: Mapping[str, Any],
    *,
    expected_release_git_commit: str,
    expected_release_tag_object: str,
    description: str,
) -> dict[str, str]:
    if (
        _GIT_OBJECT_RE.fullmatch(expected_release_git_commit) is None
        or _GIT_OBJECT_RE.fullmatch(expected_release_tag_object) is None
    ):
        raise ProtectedCapacityError(
            "expected release commit/tag object must be lowercase 40-hex Git IDs"
        )
    binding = {field: evidence.get(field) for field in _BINDING_FIELDS}
    if (
        binding.get("release_id") != RELEASE_ID
        or binding.get("release_tag") != RELEASE_TAG
        or binding.get("release_git_commit") != expected_release_git_commit
        or binding.get("release_tag_object") != expected_release_tag_object
        or binding.get("chain_namespace") != CHAIN_NAMESPACE
    ):
        raise ProtectedCapacityError(
            f"{description} is not bound to the exact release/tag/chain"
        )
    return {field: str(binding[field]) for field in sorted(_BINDING_FIELDS)}


def _validate_source_trust_anchors(
    evidence: Mapping[str, Any],
    *,
    expected_source_tree_sha256: str | None,
    expected_dispatcher_source_sha256: str | None,
    expected_qualification_runner_source_sha256: str | None,
    description: str,
) -> dict[str, str]:
    """Require three caller-supplied hashes from one immutable release authority."""

    expected = {
        "source_tree_sha256": expected_source_tree_sha256,
        "dispatcher_source_sha256": expected_dispatcher_source_sha256,
        "qualification_runner_source_sha256": (
            expected_qualification_runner_source_sha256
        ),
    }
    if any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in expected.values()
    ):
        raise ProtectedCapacityError(
            "frozen release source-tree, dispatcher, and qualification-runner "
            "SHA-256 trust anchors are all required"
        )
    if any(evidence.get(field) != value for field, value in expected.items()):
        raise ProtectedCapacityError(
            f"{description} source trust anchors differ from the immutable release"
        )
    return {field: str(value) for field, value in sorted(expected.items())}


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    description: str,
) -> str:
    identity = dict(value)
    observed = identity.pop(identity_field, None)
    if (
        not isinstance(observed, str)
        or _SHA256_RE.fullmatch(observed) is None
        or observed != identity_sha256(identity)
    ):
        raise ProtectedCapacityError(
            f"{description} self-hash {identity_field} is invalid"
        )
    return observed


def _placement_identity(
    row: Mapping[str, Any],
    *,
    role: str,
    preempt_type: str,
) -> tuple[str, str]:
    partition = row.get("partition")
    qos = row.get("qos")
    if (
        not isinstance(partition, str)
        or _PLACEMENT_NAME_RE.fullmatch(partition) is None
        or not isinstance(qos, str)
        or _PLACEMENT_NAME_RE.fullmatch(qos) is None
    ):
        raise ProtectedCapacityError(
            f"scientific {role} partition/QOS identity is malformed"
        )
    identity = (partition, qos)
    label = f"{partition}/{qos}"
    partition_mode = row.get("partition_preempt_mode")
    qos_mode = row.get("qos_preempt_mode")
    if partition_mode != "OFF":
        raise ProtectedCapacityError(
            f"scientific {role} placement {label} has partition "
            f"PreemptMode={partition_mode!r}; effective protection requires OFF"
        )
    if qos_mode != "OFF" and not (
        preempt_type == "preempt/partition_prio" and qos_mode == "cluster"
    ):
        raise ProtectedCapacityError(
            f"scientific {role} placement {label} has QOS "
            f"PreemptMode={qos_mode!r}, incompatible with PreemptType={preempt_type!r}"
        )
    return identity


def _validate_scheduler_server_rows(
    rows: Any,
    *,
    preempt_type: str,
    capacity_generation: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(rows, list) or not rows:
        raise ProtectedCapacityError(
            "scheduler evidence has no scientific server placements"
        )
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    totals = {
        "base_active_gpus": 0,
        "reserved_additive_gpus": 0,
        "effective_active_gpus": 0,
        "retained_warm_turnover_gpus": 0,
        "attested_total_gpus": 0,
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtectedCapacityError(
                f"scientific server placement {index} is malformed"
            )
        _require_exact_fields(
            raw,
            _SERVER_FIELDS,
            description=f"scientific server placement {index}",
        )
        identity = _placement_identity(
            raw, role="server", preempt_type=preempt_type
        )
        if identity in identities:
            raise ProtectedCapacityError(
                f"duplicate scientific server placement {identity[0]}/{identity[1]}"
            )
        identities.add(identity)
        for field in totals:
            totals[field] += _require_integer(
                raw.get(field),
                minimum=0,
                description=f"server placement {identity} {field}",
            )
        if (
            raw["base_active_gpus"] + raw["reserved_additive_gpus"]
            != raw["effective_active_gpus"]
            or raw["effective_active_gpus"]
            + raw["retained_warm_turnover_gpus"]
            > raw["attested_total_gpus"]
        ):
            raise ProtectedCapacityError(
                f"server placement {identity} double-counts active/turnover GPUs"
            )
        normalized.append(dict(raw))
    if totals["base_active_gpus"] != BASE_ACTIVE_GPUS:
        raise ProtectedCapacityError(
            "base active serving GPU total differs from the frozen 24-GPU fleet"
        )
    initial_generation = capacity_generation == 1
    if (
        len(normalized) != 1
        or totals["retained_warm_turnover_gpus"]
        != RETAINED_WARM_TURNOVER_GPUS
        or totals["attested_total_gpus"]
        != totals["effective_active_gpus"]
        + totals["retained_warm_turnover_gpus"]
        or (
            initial_generation
            and (
                totals["effective_active_gpus"] != PRODUCTION_ACTIVE_GPUS
                or totals["attested_total_gpus"]
                != PRODUCTION_ATTESTED_GPUS
            )
        )
        or (
            not initial_generation
            and totals["effective_active_gpus"] <= BASE_ACTIVE_GPUS
        )
    ):
        raise ProtectedCapacityError(
            "server capacity placement does not match the generation-scoped "
            "active fleet plus four retained warm-turnover GPUs"
        )
    normalized.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["qos"]),
            canonical_bytes(row),
        )
    )
    return normalized, totals


def _validate_scheduler_client_rows(
    rows: Any,
    *,
    preempt_type: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(rows, list) or not rows:
        raise ProtectedCapacityError(
            "scheduler evidence has no scientific client placements"
        )
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    totals = {
        "cell_ceiling": 0,
        "cpu": 0,
        "memory_mib": 0,
        "reserve_jobs": 0,
        "submit_headroom": 0,
    }
    field_map = {
        "slots": "cell_ceiling",
        "cpus": "cpu",
        "memory_mib": "memory_mib",
        "reserve_jobs": "reserve_jobs",
        "submit_headroom": "submit_headroom",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtectedCapacityError(
                f"scientific client placement {index} is malformed"
            )
        _require_exact_fields(
            raw,
            _CLIENT_FIELDS,
            description=f"scientific client placement {index}",
        )
        identity = _placement_identity(
            raw, role="client", preempt_type=preempt_type
        )
        if identity in identities:
            raise ProtectedCapacityError(
                f"duplicate scientific client placement {identity[0]}/{identity[1]}"
            )
        identities.add(identity)
        for source, target in field_map.items():
            totals[target] += _require_integer(
                raw.get(source),
                minimum=0,
                description=f"client placement {identity} {source}",
            )
        normalized.append(dict(raw))
    minima = {
        "cell_ceiling": MIN_CLIENT_SLOTS,
        "cpu": MIN_CLIENT_CPUS,
        "memory_mib": MIN_CLIENT_MEMORY_MIB,
        "reserve_jobs": MIN_RESERVE_JOBS,
        "submit_headroom": MIN_SUBMIT_HEADROOM,
    }
    for field, minimum in minima.items():
        if totals[field] < minimum:
            raise ProtectedCapacityError(
                f"client {field} {totals[field]} < required {minimum}"
            )
    if totals["cpu"] < totals["cell_ceiling"]:
        raise ProtectedCapacityError(
            "client CPU capacity cannot place every attested client slot"
        )
    required_memory = totals["cell_ceiling"] * CLIENT_MEMORY_MIB_PER_SLOT
    if totals["memory_mib"] < required_memory:
        raise ProtectedCapacityError(
            "client memory cannot provide 4096 MiB for every attested slot"
        )
    if totals["submit_headroom"] < totals["cell_ceiling"] + totals["reserve_jobs"]:
        raise ProtectedCapacityError(
            "submit headroom cannot cover the client ceiling plus reserve"
        )
    directly_usable = [
        row
        for row in normalized
        if row["slots"] >= MIN_CLIENT_SLOTS
        and row["cpus"] >= MIN_CLIENT_CPUS
        and row["memory_mib"] >= MIN_CLIENT_MEMORY_MIB
        and row["reserve_jobs"] >= MIN_RESERVE_JOBS
        and row["submit_headroom"] >= MIN_SUBMIT_HEADROOM
        and row["cpus"] >= row["slots"]
        and row["memory_mib"]
        >= row["slots"] * CLIENT_MEMORY_MIB_PER_SLOT
        and row["submit_headroom"]
        >= row["slots"] + row["reserve_jobs"]
    ]
    if len(directly_usable) != 1:
        raise ProtectedCapacityError(
            "protected capacity must contain exactly one scientific client "
            "partition/QOS row that itself authorizes 384 cells plus the "
            "64-job reserve"
        )
    normalized.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["qos"]),
            canonical_bytes(row),
        )
    )
    return normalized, totals


def _expected_job_element_accounting(
    active_server_job_elements: int,
) -> dict[str, int]:
    residual = (
        MIN_RESERVE_JOBS
        - active_server_job_elements
        - EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
    )
    if (
        active_server_job_elements < BASE_LOGICAL_REPLICAS
        or residual <= 0
        or active_server_job_elements
        + EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
        + residual
        != MIN_RESERVE_JOBS
        or MIN_CLIENT_SLOTS + MIN_RESERVE_JOBS
        != EXPECTED_TOTAL_JOB_ELEMENTS
    ):
        raise ProtectedCapacityError(
            "protected-capacity residual reserve formula is invalid"
        )
    return {
        "cell_job_elements": MIN_CLIENT_SLOTS,
        "active_server_job_elements": active_server_job_elements,
        "warm_turnover_job_elements": EXPECTED_WARM_TURNOVER_JOB_ELEMENTS,
        "controller_monitor_other_held_job_elements": residual,
        "total_non_cell_reserve_job_elements": MIN_RESERVE_JOBS,
        "total_canary_job_elements": EXPECTED_TOTAL_JOB_ELEMENTS,
    }


def _validate_job_element_accounting(
    value: Any,
    *,
    active_server_job_elements: int,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProtectedCapacityError(
            "scheduler job-element accounting is malformed"
        )
    _require_exact_fields(
        value,
        _JOB_ELEMENT_ACCOUNTING_FIELDS,
        description="scheduler job-element accounting",
    )
    normalized = {
        field: _require_integer(
            value.get(field),
            minimum=0,
            description=f"scheduler job-element accounting {field}",
        )
        for field in _JOB_ELEMENT_ACCOUNTING_FIELDS
    }
    expected = _expected_job_element_accounting(active_server_job_elements)
    if normalized != expected:
        raise ProtectedCapacityError(
            "scheduler job-element accounting does not realize exactly 384 "
            "clients plus the inclusive 64-job non-cell reserve"
        )
    return normalized


def _validate_occupancy_preflight(
    value: Any,
    *,
    scheduler_account: str,
    scheduler_max_jobs: int | None,
    scheduler_max_submit_jobs: int,
    qos_contracts: Sequence[Mapping[str, Any]],
    expected_running_jobs: int,
) -> dict[str, Any]:
    """Validate the historical residual-capacity proof made before canary submit."""

    required = {
        "protocol",
        "plan_id",
        "observation_interval_seconds",
        "scheduler_account",
        "scientific_qos",
        "association_max_jobs",
        "qos_max_jobs",
        "effective_max_jobs",
        "association_max_submit_jobs",
        "qos_max_submit_jobs",
        "effective_max_submit_jobs",
        "existing_job_elements",
        "existing_association_job_elements",
        "existing_qos_job_elements",
        "existing_association_running_job_elements",
        "existing_qos_running_job_elements",
        "required_new_running_job_elements",
        "required_new_job_elements",
        "first_observation",
        "second_observation",
        "preflight_id",
    }
    if not isinstance(value, Mapping):
        raise ProtectedCapacityError("occupancy preflight is absent")
    _require_exact_fields(
        value,
        required,
        description="occupancy preflight",
    )
    identity = dict(value)
    preflight_id = identity.pop("preflight_id")
    qos_contract_by_name = {
        str(row["qos"]): row for row in qos_contracts
    }
    scientific_qos = value.get("scientific_qos")
    qos_contract = qos_contract_by_name.get(str(scientific_qos))
    interval = value.get("observation_interval_seconds")
    if (
        value.get("protocol")
        != "schema5-v1.2-r11-protected-capacity-occupancy-preflight-v3"
        or _SHA256_RE.fullmatch(str(value.get("plan_id", ""))) is None
        or preflight_id
        != hashlib.sha256(canonical_bytes(identity)).hexdigest()
        or value.get("scheduler_account") != scheduler_account
        or not isinstance(scientific_qos, str)
        or qos_contract is None
        or not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(float(interval))
        or float(interval) < 0
        or value.get("association_max_jobs") != scheduler_max_jobs
        or value.get("qos_max_jobs")
        != qos_contract["max_jobs_per_user"]
        or value.get("association_max_submit_jobs")
        != scheduler_max_submit_jobs
        or value.get("qos_max_submit_jobs")
        != qos_contract["max_submit_jobs_per_user"]
        or value.get("required_new_running_job_elements")
        != expected_running_jobs
        or value.get("required_new_job_elements")
        != EXPECTED_TOTAL_JOB_ELEMENTS
    ):
        raise ProtectedCapacityError(
            "occupancy preflight identity/limit binding is invalid"
        )
    finite_max_jobs = [
        limit
        for limit in (
            scheduler_max_jobs,
            qos_contract["max_jobs_per_user"],
        )
        if limit is not None
    ]
    finite_max_submit = [
        limit
        for limit in (
            scheduler_max_submit_jobs,
            qos_contract["max_submit_jobs_per_user"],
        )
        if limit is not None
    ]
    if (
        value.get("effective_max_jobs")
        != (min(finite_max_jobs) if finite_max_jobs else None)
        or value.get("effective_max_submit_jobs")
        != min(finite_max_submit)
    ):
        raise ProtectedCapacityError(
            "occupancy preflight effective limits are invalid"
        )
    observations: list[Mapping[str, Any]] = []
    for field in ("first_observation", "second_observation"):
        observation = value.get(field)
        if not isinstance(observation, Mapping):
            raise ProtectedCapacityError(
                f"occupancy preflight {field} is malformed"
            )
        _require_exact_fields(
            observation,
            {
                "observed_at",
                "job_elements",
                "jobs",
                "squeue_sha256",
                "sacct_sha256",
                "observation_id",
            },
            description=f"occupancy preflight {field}",
        )
        observation_identity = dict(observation)
        observation_id = observation_identity.pop("observation_id")
        jobs = observation.get("jobs")
        if (
            not isinstance(observation.get("observed_at"), (int, float))
            or isinstance(observation.get("observed_at"), bool)
            or not math.isfinite(float(observation["observed_at"]))
            or not isinstance(jobs, list)
            or observation.get("job_elements") != len(jobs)
            or _SHA256_RE.fullmatch(
                str(observation.get("squeue_sha256", ""))
            )
            is None
            or _SHA256_RE.fullmatch(
                str(observation.get("sacct_sha256", ""))
            )
            is None
            or observation_id
            != hashlib.sha256(
                canonical_bytes(observation_identity)
            ).hexdigest()
        ):
            raise ProtectedCapacityError(
                f"occupancy preflight {field} envelope is invalid"
            )
        identities: set[str] = set()
        for row in jobs:
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"job_id", "state", "account", "qos", "comment"}
                or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", str(row.get("job_id", "")))
                is None
                or str(row["job_id"]) in identities
                or str(row.get("state", ""))
                not in {
                    "PENDING",
                    "RUNNING",
                    "CONFIGURING",
                    "COMPLETING",
                    "RESIZING",
                    "SUSPENDED",
                }
                or _PLACEMENT_NAME_RE.fullmatch(str(row.get("account", "")))
                is None
                or _PLACEMENT_NAME_RE.fullmatch(str(row.get("qos", "")))
                is None
                or not isinstance(row.get("comment"), str)
            ):
                raise ProtectedCapacityError(
                    f"occupancy preflight {field} job row is invalid"
                )
            identities.add(str(row["job_id"]))
        observations.append(observation)
    second_jobs = observations[1]["jobs"]
    max_jobs_states = {
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
    association_running = sum(
        row["state"] in max_jobs_states
        and row["account"] == scheduler_account
        for row in second_jobs
    )
    qos_running = sum(
        row["state"] in max_jobs_states
        and row["account"] == scheduler_account
        and row["qos"] == scientific_qos
        for row in second_jobs
    )
    existing = len(second_jobs)
    association_existing = sum(
        row["account"] == scheduler_account for row in second_jobs
    )
    qos_existing = sum(
        row["account"] == scheduler_account
        and row["qos"] == scientific_qos
        for row in second_jobs
    )
    if (
        observations[0]["jobs"] != second_jobs
        or observations[0]["job_elements"] != existing
        or float(observations[1]["observed_at"])
        - float(observations[0]["observed_at"])
        < float(interval)
        or value.get("existing_job_elements") != existing
        or value.get("existing_association_job_elements")
        != association_existing
        or value.get("existing_qos_job_elements") != qos_existing
        or value.get("existing_association_running_job_elements")
        != association_running
        or value.get("existing_qos_running_job_elements") != qos_running
        or association_existing + EXPECTED_TOTAL_JOB_ELEMENTS
        > scheduler_max_submit_jobs
        or (
            qos_contract["max_submit_jobs_per_user"] is not None
            and qos_existing + int(qos_contract["required_submit_jobs"])
            > int(qos_contract["max_submit_jobs_per_user"])
        )
        or (
            scheduler_max_jobs is not None
            and association_running + expected_running_jobs
            > scheduler_max_jobs
        )
        or (
            qos_contract["max_jobs_per_user"] is not None
            and qos_running + int(qos_contract["required_running_jobs"])
            > qos_contract["max_jobs_per_user"]
        )
    ):
        raise ProtectedCapacityError(
            "occupancy preflight does not prove residual submit/MaxJobs "
            "capacity around contemporaneous unrelated work"
        )
    return dict(value)


def _validate_scheduler_source(
    source: Any,
    *,
    command: str,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ProtectedCapacityError(f"{command} evidence is malformed")
    _require_exact_fields(
        source,
        _SOURCE_FIELDS,
        description=f"{command} evidence",
    )
    argv = source.get("argv")
    raw_output = source.get("raw_output")
    if (
        source.get("complete") is not True
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or command
        not in {
            Path(item).name
            for item in argv
        }
        and command not in argv
        or not isinstance(raw_output, str)
        or "\x00" in raw_output
        or not raw_output.strip()
    ):
        raise ProtectedCapacityError(
            f"{command} evidence is incomplete or has an invalid capture"
        )
    raw_digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    rows = [line for line in raw_output.splitlines() if line.strip()]
    if (
        source.get("raw_output_sha256") != raw_digest
        or not isinstance(source.get("record_count"), int)
        or isinstance(source.get("record_count"), bool)
        or source.get("record_count") != len(rows)
        or len(rows) < 1
    ):
        raise ProtectedCapacityError(
            f"{command} raw capture/hash/cardinality is inconsistent"
        )
    return dict(source)


def _pipe_rows(
    source: Mapping[str, Any],
    *,
    command: str,
    width: int,
    description: str,
) -> list[list[str]]:
    validated = _validate_scheduler_source(source, command=command)
    rows = [
        line.strip().split("|")
        for line in str(validated["raw_output"]).splitlines()
        if line.strip()
    ]
    if any(
        len(row) != width
        or any("\n" in field or "\r" in field for field in row)
        for row in rows
    ):
        raise ProtectedCapacityError(
            f"{description} does not use the closed {width}-field protocol"
        )
    return rows


def _parse_nonnegative_field(value: str, *, description: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ProtectedCapacityError(
            f"{description} is not an integer"
        ) from exc
    if parsed < 0:
        raise ProtectedCapacityError(f"{description} must be nonnegative")
    return parsed


def _parse_scheduler_configuration(
    source: Mapping[str, Any],
) -> str:
    rows = _pipe_rows(
        source,
        command="scontrol",
        width=2,
        description="scheduler configuration",
    )
    if len(rows) != 1 or rows[0][0] != "PreemptType":
        raise ProtectedCapacityError(
            "scheduler configuration must contain exactly one PreemptType fact"
        )
    preempt_type = rows[0][1]
    if preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
        raise ProtectedCapacityError(
            "scheduler configuration has unsupported PreemptType"
        )
    return preempt_type


def _parse_partition_configuration(
    source: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = _pipe_rows(
        source,
        command="scontrol",
        width=8,
        description="partition configuration",
    )
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        partition, mode, state = row[:3]
        if (
            _PLACEMENT_NAME_RE.fullmatch(partition) is None
            or partition in parsed
            or mode != "OFF"
            or state != "UP"
        ):
            raise ProtectedCapacityError(
                "partition configuration is duplicated, unavailable, or "
                "preemptible"
            )
        parsed[partition] = {
            "preempt_mode": mode,
            "state": state,
            "max_time_seconds": _parse_nonnegative_field(
                row[3], description=f"{partition} MaxTime"
            ),
            "cpus": _parse_nonnegative_field(
                row[4], description=f"{partition} CPUs"
            ),
            "memory_mib": _parse_nonnegative_field(
                row[5], description=f"{partition} memory"
            ),
            "gpus": _parse_nonnegative_field(
                row[6], description=f"{partition} GPUs"
            ),
            "nodes": _parse_nonnegative_field(
                row[7], description=f"{partition} nodes"
            ),
        }
        if parsed[partition]["nodes"] < 1:
            raise ProtectedCapacityError(
                f"partition {partition} must contain at least one node"
            )
    return parsed


def _parse_optional_limit(value: str, *, description: str) -> int | None:
    if value == "-":
        return None
    return _parse_nonnegative_field(value, description=description)


def _parse_qos_configuration(
    source: Mapping[str, Any],
    *,
    preempt_type: str,
) -> dict[str, dict[str, Any]]:
    rows = _pipe_rows(
        source,
        command="sacctmgr",
        width=5,
        description="QOS configuration",
    )
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qos, mode = row[:2]
        safe_mode = mode == "OFF" or (
            preempt_type == "preempt/partition_prio" and mode == "cluster"
        )
        if (
            _PLACEMENT_NAME_RE.fullmatch(qos) is None
            or qos in parsed
            or not safe_mode
        ):
            raise ProtectedCapacityError(
                "QOS configuration is duplicated or effectively preemptible"
            )
        parsed[qos] = {
            "preempt_mode": mode,
            "max_jobs_per_user": _parse_optional_limit(
                row[2], description=f"{qos} MaxJobsPerUser"
            ),
            "max_submit_jobs_per_user": _parse_optional_limit(
                row[3], description=f"{qos} MaxSubmitJobsPerUser"
            ),
            "max_wall_seconds": _parse_optional_limit(
                row[4], description=f"{qos} MaxWall"
            ),
        }
    return parsed


def _parse_association_configuration(
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _pipe_rows(
        source,
        command="sacctmgr",
        width=6,
        description="association configuration",
    )
    parsed: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for cluster, account, user, qos_csv, max_jobs_raw, max_submit_raw in rows:
        identity = (cluster, account, user)
        qoses = sorted(set(filter(None, qos_csv.split(","))))
        if (
            any(
                _PLACEMENT_NAME_RE.fullmatch(value) is None
                for value in identity
            )
            or identity in identities
            or not qoses
            or any(_PLACEMENT_NAME_RE.fullmatch(qos) is None for qos in qoses)
        ):
            raise ProtectedCapacityError(
                "scheduler association identity/QOS list is malformed"
            )
        identities.add(identity)
        parsed.append(
            {
                "cluster": cluster,
                "account": account,
                "user": user,
                "qos": qoses,
                "max_jobs": _parse_optional_limit(
                    max_jobs_raw,
                    description=f"association {identity} MaxJobs",
                ),
                "max_submit_jobs": _parse_optional_limit(
                    max_submit_raw,
                    description=f"association {identity} MaxSubmitJobs",
                ),
            }
        )
    return parsed


def _fleet_memory_mib(value: Any, *, description: str) -> int:
    if not isinstance(value, str):
        raise ProtectedCapacityError(f"{description} is not a memory string")
    match = re.fullmatch(r"([0-9]+)([MGT])", value)
    if match is None:
        raise ProtectedCapacityError(f"{description} is malformed")
    amount = int(match.group(1))
    factor = {"M": 1, "G": 1024, "T": 1024**2}[match.group(2)]
    return amount * factor


def _parse_fleet_contract(
    source: Mapping[str, Any],
    *,
    command: str,
    require_base: bool,
) -> dict[str, Any]:
    validated = _validate_scheduler_source(
        source,
        command=command,
    )
    raw = str(validated["raw_output"]).encode("utf-8")
    fleet = _decode_json_object(raw, description="frozen fleet contract")
    required_root = {
        "schema_version",
        "fleet_id",
        "release_id",
        "model_contract_sha256",
        "offline_environment",
        "server_pool",
        "logical_replica_count",
        "allocated_gpu_count",
        "profiles",
    }
    if (
        set(fleet) != required_root
        or fleet.get("schema_version") != 1
        or fleet.get("fleet_id") != "schema5-v1"
        or fleet.get("release_id") != RELEASE_ID
        or not isinstance(fleet.get("logical_replica_count"), int)
        or isinstance(fleet.get("logical_replica_count"), bool)
        or fleet["logical_replica_count"] < BASE_LOGICAL_REPLICAS
        or not isinstance(fleet.get("allocated_gpu_count"), int)
        or isinstance(fleet.get("allocated_gpu_count"), bool)
        or fleet["allocated_gpu_count"] < BASE_ACTIVE_GPUS
        or _SHA256_RE.fullmatch(str(fleet.get("model_contract_sha256", "")))
        is None
        or not isinstance(fleet.get("profiles"), list)
    ):
        raise ProtectedCapacityError(
            "frozen fleet contract identity/cardinality is invalid"
        )
    topology: list[dict[str, Any]] = []
    identities: set[str] = set()
    for raw_profile in fleet["profiles"]:
        if not isinstance(raw_profile, Mapping):
            raise ProtectedCapacityError("frozen fleet profile is malformed")
        profile = raw_profile.get("serving_profile")
        replicas = raw_profile.get("replicas")
        tp = raw_profile.get("tensor_parallel_size")
        if (
            not isinstance(profile, str)
            or _PLACEMENT_NAME_RE.fullmatch(profile) is None
            or not isinstance(replicas, list)
            or not isinstance(tp, int)
            or isinstance(tp, bool)
            or tp not in {1, 2}
            or raw_profile.get("gpus_per_replica") != tp
        ):
            raise ProtectedCapacityError(
                "frozen fleet profile topology is malformed"
            )
        for replica in replicas:
            if not isinstance(replica, Mapping):
                raise ProtectedCapacityError(
                    "frozen fleet replica is malformed"
                )
            replica_id = replica.get("replica_id")
            cpus = replica.get("cpus_per_task")
            time_limit = replica.get("time_limit")
            if (
                not isinstance(replica_id, str)
                or _PLACEMENT_NAME_RE.fullmatch(replica_id) is None
                or replica_id in identities
                or not isinstance(cpus, int)
                or isinstance(cpus, bool)
                or cpus < 1
                or time_limit != "1-00:00:00"
            ):
                raise ProtectedCapacityError(
                    "frozen fleet replica identity/resources are malformed"
                )
            identities.add(replica_id)
            topology.append(
                {
                    "shape_id": replica_id,
                    "serving_profile": profile,
                    "tasks": 1,
                    "cpus": cpus,
                    "memory_mib": _fleet_memory_mib(
                        replica.get("memory"),
                        description=f"{replica_id} memory",
                    ),
                    "gpus": tp,
                    "time_limit_seconds": MIN_SCIENTIFIC_WALL_SECONDS,
                }
            )
    if (
        len(topology) != fleet["logical_replica_count"]
        or sum(row["gpus"] for row in topology)
        != fleet["allocated_gpu_count"]
        or (
            require_base
            and (
                len(topology) != BASE_LOGICAL_REPLICAS
                or sum(row["gpus"] for row in topology)
                != BASE_ACTIVE_GPUS
                or sorted(row["gpus"] for row in topology)
                != [1] * 20 + [2] * 2
            )
        )
    ):
        raise ProtectedCapacityError(
            "fleet contract aggregate/topology cardinality is invalid"
        )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "topology": topology,
        "topology_sha256": hashlib.sha256(canonical_bytes(topology)).hexdigest(),
        "logical_replicas": len(topology),
        "allocated_gpus": sum(int(row["gpus"]) for row in topology),
        "profile_replicas": {
            str(profile["serving_profile"]): len(profile["replicas"])
            for profile in fleet["profiles"]
        },
    }


def _parse_canary_jobs(
    source: Mapping[str, Any],
    *,
    command: str,
) -> list[dict[str, Any]]:
    rows = _pipe_rows(
        source,
        command=command,
        width=13,
        description=f"{command} protected canary jobs",
    )
    parsed: list[dict[str, Any]] = []
    identities: set[str] = set()
    allowed_roles = {"server_active", "server_warm", "client", "reserve"}
    for row in rows:
        job_id, role, state, partition, qos = row[:5]
        shape_id = row[11]
        spooled_script_sha256 = row[12]
        if (
            re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None
            or job_id in identities
            or role not in allowed_roles
            or _PLACEMENT_NAME_RE.fullmatch(partition) is None
            or _PLACEMENT_NAME_RE.fullmatch(qos) is None
            or (
                shape_id != "-"
                and _PLACEMENT_NAME_RE.fullmatch(shape_id) is None
            )
            or _SHA256_RE.fullmatch(spooled_script_sha256) is None
        ):
            raise ProtectedCapacityError(
                f"{command} protected canary job identity is malformed"
            )
        identities.add(job_id)
        parsed.append(
            {
                "job_id": job_id,
                "role": role,
                "state": state,
                "partition": partition,
                "qos": qos,
                "tasks": _parse_nonnegative_field(
                    row[5], description=f"{job_id} tasks"
                ),
                "cpus": _parse_nonnegative_field(
                    row[6], description=f"{job_id} CPUs"
                ),
                "memory_mib": _parse_nonnegative_field(
                    row[7], description=f"{job_id} memory"
                ),
                "gpus": _parse_nonnegative_field(
                    row[8], description=f"{job_id} GPUs"
                ),
                "effective_requeue": _parse_nonnegative_field(
                    row[9], description=f"{job_id} Requeue"
                ),
                "time_limit_seconds": _parse_nonnegative_field(
                    row[10], description=f"{job_id} TimeLimit"
                ),
                "shape_id": shape_id,
                "spooled_script_sha256": spooled_script_sha256,
            }
        )
    if any(row["effective_requeue"] != 0 for row in parsed):
        raise ProtectedCapacityError(
            f"{command} protected canary contains a requeue-enabled job"
        )
    return parsed


def _derive_capacity_from_raw_sources(
    evidence: Mapping[str, Any],
    *,
    servers: Sequence[Mapping[str, Any]],
    clients: Sequence[Mapping[str, Any]],
    expected_source_tree_sha256: str,
    expected_dispatcher_source_sha256: str,
    expected_qualification_runner_source_sha256: str,
) -> dict[str, Any]:
    preempt_type = _parse_scheduler_configuration(
        evidence["scheduler_configuration"]
    )
    partitions = _parse_partition_configuration(
        evidence["partition_configuration"]
    )
    qoses = _parse_qos_configuration(
        evidence["qos_configuration"],
        preempt_type=preempt_type,
    )
    associations = _parse_association_configuration(
        evidence["association_configuration"]
    )
    base_fleet = _parse_fleet_contract(
        evidence["base_fleet_contract"],
        command="base-fleet-contract",
        require_base=True,
    )
    effective_fleet = _parse_fleet_contract(
        evidence["effective_fleet_contract"],
        command="effective-fleet-contract",
        require_base=False,
    )
    overlay_source = _validate_scheduler_source(
        evidence["additive_overlay_contract"],
        command="additive-overlay-contract",
    )
    certificate_source = _validate_scheduler_source(
        evidence["static_feasibility_certificate_source"],
        command="static-feasibility-certificate",
    )
    base_by_id = {
        str(row["shape_id"]): row for row in base_fleet["topology"]
    }
    effective_by_id = {
        str(row["shape_id"]): row for row in effective_fleet["topology"]
    }
    if (
        not set(base_by_id).issubset(effective_by_id)
        or any(effective_by_id[key] != row for key, row in base_by_id.items())
    ):
        raise ProtectedCapacityError(
            "effective fleet is not an exact additive extension of the base fleet"
        )
    additive_topology = [
        row
        for row in effective_fleet["topology"]
        if str(row["shape_id"]) not in base_by_id
    ]
    certificate_binding = evidence.get("static_feasibility_certificate")
    if (
        not isinstance(certificate_binding, Mapping)
        or set(certificate_binding)
        != {"path", "sha256", "certificate_id"}
    ):
        raise ProtectedCapacityError(
            "scheduler evidence static feasibility binding is malformed"
        )
    try:
        certificate = runtime_capacity.load_static_feasibility_certificate(
            str(certificate_binding["path"]),
            expected_sha256=str(certificate_binding["sha256"]),
            expected_certificate_id=str(certificate_binding["certificate_id"]),
            expected_capacity_generation=int(evidence["capacity_generation"]),
            expected_base_fleet_contract_sha256=base_fleet["sha256"],
            expected_effective_fleet_contract_sha256=effective_fleet["sha256"],
            expected_additive_overlay_contract_sha256=effective_fleet["sha256"],
            expected_release_git_commit=str(evidence["release_git_commit"]),
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_dispatcher_source_sha256=(
                expected_dispatcher_source_sha256
            ),
            expected_qualification_runner_source_sha256=(
                expected_qualification_runner_source_sha256
            ),
        )
    except (runtime_capacity.ProtectedCapacityError, TypeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"static feasibility certificate is invalid: {exc}"
        ) from exc
    if (
        evidence.get("base_fleet_contract_sha256") != base_fleet["sha256"]
        or evidence.get("effective_fleet_contract_sha256")
        != effective_fleet["sha256"]
        or evidence.get("additive_overlay_contract_sha256")
        != effective_fleet["sha256"]
        or evidence.get("fleet_contract_sha256")
        != effective_fleet["sha256"]
        or evidence.get("active_fleet_topology_sha256")
        != effective_fleet["topology_sha256"]
        or overlay_source["raw_output_sha256"]
        != effective_fleet["sha256"]
        or certificate_source["raw_output_sha256"]
        != certificate.sha256
    ):
        raise ProtectedCapacityError(
            "claimed base/effective/overlay/certificate bytes differ from "
            "their parsed protected-capacity bindings"
        )
    for field in ("builder_source", "publisher_source"):
        source = _validate_scheduler_source(
            evidence[field],
            command="release-source",
        )
        if source["raw_output_sha256"] != evidence.get(f"{field}_sha256"):
            raise ProtectedCapacityError(
                f"claimed {field} hash differs from tagged source bytes"
            )
    queued = _parse_canary_jobs(evidence["squeue"], command="squeue")
    accounted = _parse_canary_jobs(evidence["sacct"], command="sacct")
    placement_identities = {
        (str(row["partition"]), str(row["qos"]))
        for row in [*servers, *clients]
    }
    for partition, qos in placement_identities:
        if partition not in partitions or qos not in qoses:
            raise ProtectedCapacityError(
                f"claimed placement {partition}/{qos} is absent from raw "
                "partition/QOS configuration"
            )
    required_qoses = {qos for _partition, qos in placement_identities}
    expected_running_jobs = (
        MIN_CLIENT_SLOTS
        + int(effective_fleet["logical_replicas"])
        + EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
    )
    matching_associations = [
        association
        for association in associations
        if required_qoses.issubset(set(association["qos"]))
        and association["max_submit_jobs"] is not None
        and association["max_submit_jobs"] >= MIN_SUBMIT_HEADROOM
        and (
            association["max_jobs"] is None
            or association["max_jobs"] >= expected_running_jobs
        )
    ]
    if len(matching_associations) != 1:
        raise ProtectedCapacityError(
            "raw association configuration does not yield one exact user/account "
            "authority for every claimed QOS and sufficient submit headroom"
        )
    association = matching_associations[0]
    queued_by_role = {
        role: [row for row in queued if row["role"] == role]
        for role in ("server_active", "server_warm", "client", "reserve")
    }
    accounted_by_role = {
        role: [row for row in accounted if row["role"] == role]
        for role in ("server_active", "server_warm", "client", "reserve")
    }
    if any(
        not queued_by_role[role] or not accounted_by_role[role]
        for role in ("server_active", "server_warm", "client", "reserve")
    ):
        raise ProtectedCapacityError(
            "raw scheduler truth lacks the exact server/client/reserve canary rows"
        )

    def aggregate_role(role: str) -> dict[str, Any]:
        queue_rows = {
            str(row["job_id"]): row for row in queued_by_role[role]
        }
        accounting_rows = {
            str(row["job_id"]): row for row in accounted_by_role[role]
        }
        if (
            len(queue_rows) != len(queued_by_role[role])
            or len(accounting_rows) != len(accounted_by_role[role])
            or set(queue_rows) != set(accounting_rows)
        ):
            raise ProtectedCapacityError(
                f"squeue and sacct {role} canary identities disagree"
            )
        placements: set[tuple[str, str]] = set()
        totals = {
            "tasks": 0,
            "cpus": 0,
            "memory_mib": 0,
            "gpus": 0,
        }
        states: set[str] = set()
        shape_rows: list[dict[str, Any]] = []
        for job_id in sorted(queue_rows):
            queue_row = queue_rows[job_id]
            accounting_row = accounting_rows[job_id]
            if any(
                queue_row[field] != accounting_row[field]
                for field in (
                    "job_id",
                    "role",
                    "partition",
                    "qos",
                    "tasks",
                    "cpus",
                    "memory_mib",
                    "gpus",
                    "effective_requeue",
                    "time_limit_seconds",
                    "shape_id",
                    "spooled_script_sha256",
                )
            ):
                raise ProtectedCapacityError(
                    f"squeue and sacct {role} canary facts disagree"
                )
            placements.add(
                (str(queue_row["partition"]), str(queue_row["qos"]))
            )
            states.add(str(queue_row["state"]))
            for field in totals:
                totals[field] += int(queue_row[field])
            shape_rows.append(
                {
                    "shape_id": str(queue_row["shape_id"]),
                    "tasks": int(queue_row["tasks"]),
                    "cpus": int(queue_row["cpus"]),
                    "memory_mib": int(queue_row["memory_mib"]),
                    "gpus": int(queue_row["gpus"]),
                    "time_limit_seconds": int(
                        queue_row["time_limit_seconds"]
                    ),
                }
            )
        if len(placements) != 1:
            raise ProtectedCapacityError(
                f"raw {role} canary spans multiple placements"
            )
        partition, qos = next(iter(placements))
        return {
            "partition": partition,
            "qos": qos,
            "states": states,
            "shape_rows": shape_rows,
            **totals,
        }

    server_active = aggregate_role("server_active")
    server_warm = aggregate_role("server_warm")
    client = aggregate_role("client")
    reserve = aggregate_role("reserve")
    observed_job_element_accounting = {
        "cell_job_elements": client["tasks"],
        "active_server_job_elements": server_active["tasks"],
        "warm_turnover_job_elements": server_warm["tasks"],
        "controller_monitor_other_held_job_elements": reserve["tasks"],
        "total_non_cell_reserve_job_elements": (
            server_active["tasks"]
            + server_warm["tasks"]
            + reserve["tasks"]
        ),
        "total_canary_job_elements": (
            client["tasks"]
            + server_active["tasks"]
            + server_warm["tasks"]
            + reserve["tasks"]
        ),
    }
    expected_job_element_accounting = _expected_job_element_accounting(
        int(effective_fleet["logical_replicas"])
    )
    if (
        evidence.get("expected_total_job_elements")
        != EXPECTED_TOTAL_JOB_ELEMENTS
        or evidence.get("job_element_accounting")
        != expected_job_element_accounting
        or observed_job_element_accounting
        != expected_job_element_accounting
    ):
        raise ProtectedCapacityError(
            "raw scheduler canary, expected total, and category accounting do "
            "not realize the exact 448-element contract"
        )
    observed_active = {
        row["shape_id"]: row for row in server_active["shape_rows"]
    }
    expected_active = {
        row["shape_id"]: {
            key: row[key]
            for key in (
                "shape_id",
                "tasks",
                "cpus",
                "memory_mib",
                "gpus",
                "time_limit_seconds",
            )
        }
        for row in effective_fleet["topology"]
    }
    if (
        len(observed_active) != len(server_active["shape_rows"])
        or observed_active != expected_active
    ):
        raise ProtectedCapacityError(
            "active GPU canaries do not exactly realize the effective fleet topology"
        )
    warm_shapes = sorted(
        (
            row["tasks"],
            row["cpus"],
            row["memory_mib"],
            row["gpus"],
            row["time_limit_seconds"],
        )
        for row in server_warm["shape_rows"]
    )
    if warm_shapes != sorted(
        [
            (1, 8, 120 * 1024, 1, MIN_SCIENTIFIC_WALL_SECONDS),
            (1, 8, 120 * 1024, 1, MIN_SCIENTIFIC_WALL_SECONDS),
            (1, 16, 240 * 1024, 2, MIN_SCIENTIFIC_WALL_SECONDS),
        ]
    ):
        raise ProtectedCapacityError(
            "warm GPU canaries do not prove two TP=1 and one TP=2 turnover shapes"
        )
    if (
        not client["states"].issubset({"RUNNING", "MIXED"})
        or not reserve["states"].issubset({"PENDING", "HELD"})
        or client["tasks"] != MIN_CLIENT_SLOTS
        or client["cpus"] != MIN_CLIENT_CPUS
        or client["memory_mib"] != MIN_CLIENT_MEMORY_MIB
        or client["gpus"] != 0
        or any(
            row["time_limit_seconds"] != CLIENT_WALL_SECONDS
            for row in client["shape_rows"]
        )
        or reserve["tasks"]
        != expected_job_element_accounting[
            "controller_monitor_other_held_job_elements"
        ]
        or reserve["cpus"]
        != expected_job_element_accounting[
            "controller_monitor_other_held_job_elements"
        ]
        or reserve["memory_mib"]
        != expected_job_element_accounting[
            "controller_monitor_other_held_job_elements"
        ]
        * 1024
        or reserve["gpus"] != 0
        or any(
            row["time_limit_seconds"] != CLIENT_WALL_SECONDS
            for row in reserve["shape_rows"]
        )
    ):
        raise ProtectedCapacityError(
            "raw client/reserve canary does not prove the full protected envelope"
        )
    if (
        not server_active["states"].issubset({"RUNNING", "MIXED"})
        or not server_warm["states"].issubset({"RUNNING", "MIXED"})
        or server_active["gpus"] != effective_fleet["allocated_gpus"]
        or server_warm["gpus"] != RETAINED_WARM_TURNOVER_GPUS
        or server_active["tasks"] != effective_fleet["logical_replicas"]
        or server_warm["tasks"] != EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
        or any(
            row["time_limit_seconds"] != MIN_SCIENTIFIC_WALL_SECONDS
            for row in server_active["shape_rows"]
        )
        or any(
            row["time_limit_seconds"] != MIN_SCIENTIFIC_WALL_SECONDS
            for row in server_warm["shape_rows"]
        )
    ):
        raise ProtectedCapacityError(
            "raw GPU canary does not separately prove the effective active fleet "
            "and four retained warm-turnover GPUs"
        )
    derived_server = {
        "partition": server_active["partition"],
        "qos": server_active["qos"],
        "base_active_gpus": base_fleet["allocated_gpus"],
        "reserved_additive_gpus": (
            effective_fleet["allocated_gpus"] - base_fleet["allocated_gpus"]
        ),
        "effective_active_gpus": server_active["gpus"],
        "retained_warm_turnover_gpus": server_warm["gpus"],
        "attested_total_gpus": server_active["gpus"] + server_warm["gpus"],
        "partition_cpus": partitions[server_active["partition"]]["cpus"],
        "partition_memory_mib": partitions[
            server_active["partition"]
        ]["memory_mib"],
        "partition_gpus": partitions[server_active["partition"]]["gpus"],
        "partition_nodes": partitions[server_active["partition"]]["nodes"],
    }
    if (
        (server_warm["partition"], server_warm["qos"])
        != (derived_server["partition"], derived_server["qos"])
    ):
        raise ProtectedCapacityError(
            "active and warm GPU canaries used different placements"
        )
    derived_client = {
        "partition": client["partition"],
        "qos": client["qos"],
        "slots": client["tasks"],
        "cpus": client["cpus"],
        "memory_mib": client["memory_mib"],
        "reserve_jobs": observed_job_element_accounting[
            "total_non_cell_reserve_job_elements"
        ],
        "submit_headroom": observed_job_element_accounting[
            "total_canary_job_elements"
        ],
    }
    if (reserve["partition"], reserve["qos"]) != (
        derived_client["partition"],
        derived_client["qos"],
    ):
        raise ProtectedCapacityError(
            "client and reserve canaries used different placements"
        )
    required_running_by_qos: dict[str, int] = {}
    required_submit_by_qos: dict[str, int] = {}
    required_wall_by_qos: dict[str, int] = {}
    for role_record, running, required_wall in (
        (server_active, True, MIN_SCIENTIFIC_WALL_SECONDS),
        (server_warm, True, MIN_SCIENTIFIC_WALL_SECONDS),
        (client, True, CLIENT_WALL_SECONDS),
        (reserve, False, CLIENT_WALL_SECONDS),
    ):
        qos_name = str(role_record["qos"])
        required_submit_by_qos[qos_name] = (
            required_submit_by_qos.get(qos_name, 0)
            + int(role_record["tasks"])
        )
        if running:
            required_running_by_qos[qos_name] = (
                required_running_by_qos.get(qos_name, 0)
                + int(role_record["tasks"])
            )
        required_wall_by_qos[qos_name] = max(
            required_wall_by_qos.get(qos_name, 0),
            required_wall,
        )
    if (
        sum(required_running_by_qos.values())
        != expected_running_jobs
        or sum(required_submit_by_qos.values())
        != EXPECTED_TOTAL_JOB_ELEMENTS
        or max(required_wall_by_qos.values(), default=0)
        != MIN_SCIENTIFIC_WALL_SECONDS
    ):
        raise ProtectedCapacityError(
            "raw scheduler canary does not derive the exact running/submit/"
            "walltime scientific envelope"
        )
    qos_contracts: list[dict[str, Any]] = []
    for qos_name in sorted(required_submit_by_qos):
        qos_policy = qoses[qos_name]
        required_running = required_running_by_qos.get(qos_name, 0)
        required_submit = required_submit_by_qos[qos_name]
        required_wall = required_wall_by_qos[qos_name]
        if (
            qos_policy["max_jobs_per_user"] is not None
            and qos_policy["max_jobs_per_user"] < required_running
        ) or (
            qos_policy["max_submit_jobs_per_user"] is not None
            and qos_policy["max_submit_jobs_per_user"] < required_submit
        ) or (
            qos_policy["max_wall_seconds"] is not None
            and qos_policy["max_wall_seconds"] < required_wall
        ):
            raise ProtectedCapacityError(
                f"QOS {qos_name} limits cannot sustain its exact "
                "running/submit/walltime canary envelope"
            )
        qos_contracts.append(
            {
                "qos": qos_name,
                "max_wall_seconds": qos_policy["max_wall_seconds"],
                "max_jobs_per_user": qos_policy["max_jobs_per_user"],
                "max_submit_jobs_per_user": qos_policy[
                    "max_submit_jobs_per_user"
                ],
                "required_wall_seconds": required_wall,
                "required_running_jobs": required_running,
                "required_submit_jobs": required_submit,
            }
        )
    required_by_partition: dict[str, dict[str, int]] = {}
    for partition_name, cpus, memory_mib, gpus, wall_seconds in (
        (
            derived_server["partition"],
            server_active["cpus"] + server_warm["cpus"],
            server_active["memory_mib"] + server_warm["memory_mib"],
            server_active["gpus"] + server_warm["gpus"],
            MIN_SCIENTIFIC_WALL_SECONDS,
        ),
        (
            derived_client["partition"],
            derived_client["cpus"],
            derived_client["memory_mib"],
            0,
            CLIENT_WALL_SECONDS,
        ),
    ):
        requirements = required_by_partition.setdefault(
            partition_name,
            {"cpus": 0, "memory_mib": 0, "gpus": 0, "wall_seconds": 0},
        )
        requirements["cpus"] += cpus
        requirements["memory_mib"] += memory_mib
        requirements["gpus"] += gpus
        requirements["wall_seconds"] = max(
            requirements["wall_seconds"], wall_seconds
        )
    if any(
        partitions[partition_name]["max_time_seconds"]
        < requirements["wall_seconds"]
        or partitions[partition_name]["cpus"] < requirements["cpus"]
        or partitions[partition_name]["memory_mib"]
        < requirements["memory_mib"]
        or partitions[partition_name]["gpus"] < requirements["gpus"]
        for partition_name, requirements in required_by_partition.items()
    ):
        raise ProtectedCapacityError(
            "raw partition inventory cannot sustain the derived canary envelope"
        )
    expected_server = next(
        (
            row
            for row in servers
            if (row["partition"], row["qos"])
            == (derived_server["partition"], derived_server["qos"])
        ),
        None,
    )
    expected_client = next(
        (
            row
            for row in clients
            if (row["partition"], row["qos"])
            == (derived_client["partition"], derived_client["qos"])
        ),
        None,
    )
    if (
        expected_server is None
        or expected_client is None
        or any(
            expected_server[field] != derived_server[field]
            for field in (
                "base_active_gpus",
                "reserved_additive_gpus",
                "effective_active_gpus",
                "retained_warm_turnover_gpus",
                "attested_total_gpus",
                "partition_cpus",
                "partition_memory_mib",
                "partition_gpus",
                "partition_nodes",
            )
        )
        or any(
            expected_client[field] != derived_client[field]
            for field in (
                "slots",
                "cpus",
                "memory_mib",
                "reserve_jobs",
                "submit_headroom",
            )
        )
    ):
        raise ProtectedCapacityError(
            "claimed placement/capacity rows differ from independently parsed "
            "raw scheduler canaries"
        )
    return {
        "preempt_type": preempt_type,
        "capacity_source": CAPACITY_SOURCE,
        "scheduler_cluster": association["cluster"],
        "scheduler_account": association["account"],
        "scheduler_user": association["user"],
        "scheduler_max_jobs": association["max_jobs"],
        "scheduler_max_submit_jobs": association["max_submit_jobs"],
        "running_scientific_jobs": expected_running_jobs,
        "minimum_scientific_wall_seconds": (
            MIN_SCIENTIFIC_WALL_SECONDS
        ),
        "scientific_qos_contracts": qos_contracts,
        "partition_cpus": partitions[derived_client["partition"]]["cpus"],
        "partition_memory_mib": partitions[derived_client["partition"]][
            "memory_mib"
        ],
        "partition_gpus": partitions[derived_client["partition"]]["gpus"],
        "active_gpus": derived_server["effective_active_gpus"],
        "warm_headroom_gpus": derived_server[
            "retained_warm_turnover_gpus"
        ],
        "cell_ceiling": derived_client["slots"],
        "cpu": derived_client["cpus"],
        "memory_mib": derived_client["memory_mib"],
        "reserve_jobs": derived_client["reserve_jobs"],
        "submit_headroom": derived_client["submit_headroom"],
        "capacity_generation": certificate.capacity_generation,
        "base_fleet_contract_path": str(
            evidence["base_fleet_contract_path"]
        ),
        "base_fleet_contract_sha256": base_fleet["sha256"],
        "effective_fleet_contract_path": str(
            evidence["effective_fleet_contract_path"]
        ),
        "effective_fleet_contract_sha256": effective_fleet["sha256"],
        "additive_overlay_contract_path": str(
            evidence["additive_overlay_contract_path"]
        ),
        "additive_overlay_contract_sha256": effective_fleet["sha256"],
        "static_feasibility_certificate": dict(certificate_binding),
        "static_feasibility_wave_passed": certificate.wave_passed,
        "static_feasibility_selected_cell_count": (
            certificate.selected_cell_count
        ),
        "static_feasibility_target_cell_count": certificate.target_cell_count,
        "static_feasibility_shortfall_cells": certificate.shortfall_cells,
        "static_feasibility_configured_client_ceiling": (
            certificate.target_cell_count
        ),
        "static_feasibility_certified_saturation_target": (
            certificate.selected_cell_count
        ),
        "base_active_logical_replicas": base_fleet["logical_replicas"],
        "base_active_gpus": base_fleet["allocated_gpus"],
        "base_active_topology": base_fleet["topology"],
        "base_active_topology_sha256": base_fleet["topology_sha256"],
        "additive_reserved_logical_replicas": len(additive_topology),
        "additive_reserved_gpus": sum(
            int(row["gpus"]) for row in additive_topology
        ),
        "additive_reserved_tp1_replicas": sum(
            int(row["gpus"]) == 1 for row in additive_topology
        ),
        "additive_reserved_tp2_replicas": sum(
            int(row["gpus"]) == 2 for row in additive_topology
        ),
        "additive_reserved_topology": additive_topology,
        "additive_reserved_topology_sha256": hashlib.sha256(
            canonical_bytes(additive_topology)
        ).hexdigest(),
        "effective_active_logical_replicas": effective_fleet[
            "logical_replicas"
        ],
        "effective_active_gpus": effective_fleet["allocated_gpus"],
        "effective_active_topology": effective_fleet["topology"],
        "effective_active_topology_sha256": effective_fleet[
            "topology_sha256"
        ],
        "retained_warm_turnover_job_elements": (
            EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
        ),
        "retained_warm_turnover_gpus": RETAINED_WARM_TURNOVER_GPUS,
        "retained_warm_turnover_tp1_allocations": 2,
        "retained_warm_turnover_tp2_allocations": 1,
        "retained_warm_turnover_topology": [
            {
                "shape_id": str(row["shape_id"]),
                "serving_profile": (
                    "warm-tp2" if int(row["gpus"]) == 2 else "warm-tp1"
                ),
                "tasks": int(row["tasks"]),
                "cpus": int(row["cpus"]),
                "memory_mib": int(row["memory_mib"]),
                "gpus": int(row["gpus"]),
                "time_limit_seconds": int(row["time_limit_seconds"]),
            }
            for row in server_warm["shape_rows"]
        ],
        "retained_warm_turnover_topology_sha256": hashlib.sha256(
            canonical_bytes(
                [
                    {
                        "shape_id": str(row["shape_id"]),
                        "serving_profile": (
                            "warm-tp2"
                            if int(row["gpus"]) == 2
                            else "warm-tp1"
                        ),
                        "tasks": int(row["tasks"]),
                        "cpus": int(row["cpus"]),
                        "memory_mib": int(row["memory_mib"]),
                        "gpus": int(row["gpus"]),
                        "time_limit_seconds": int(
                            row["time_limit_seconds"]
                        ),
                    }
                    for row in server_warm["shape_rows"]
                ]
            )
        ).hexdigest(),
        "attested_total_gpus": (
            effective_fleet["allocated_gpus"] + RETAINED_WARM_TURNOVER_GPUS
        ),
        "fleet_contract_sha256": effective_fleet["sha256"],
        "active_fleet_topology_sha256": effective_fleet[
            "topology_sha256"
        ],
        "source_tree_sha256": expected_source_tree_sha256,
        "dispatcher_source_sha256": expected_dispatcher_source_sha256,
        "qualification_runner_source_sha256": (
            expected_qualification_runner_source_sha256
        ),
        "builder_source_sha256": str(evidence["builder_source_sha256"]),
        "publisher_source_sha256": str(evidence["publisher_source_sha256"]),
        "expected_total_job_elements": EXPECTED_TOTAL_JOB_ELEMENTS,
        "job_element_accounting": observed_job_element_accounting,
    }


def validate_scheduler_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_release_git_commit: str,
    expected_release_tag_object: str,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and summarize one sealed scheduler evidence object."""

    _require_exact_fields(
        evidence,
        _SCHEDULER_FIELDS,
        description="scheduler evidence",
    )
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("protocol") != SCHEDULER_EVIDENCE_PROTOCOL
        or evidence.get("passed") is not True
    ):
        raise ProtectedCapacityError(
            "scheduler evidence envelope is invalid or not passed"
        )
    binding = _validate_release_binding(
        evidence,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        description="scheduler evidence",
    )
    source_trust = _validate_source_trust_anchors(
        evidence,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
        description="scheduler evidence",
    )
    observed_at = _require_positive_timestamp(
        evidence.get("observed_at"),
        description="scheduler observed_at",
    )
    preempt_type = evidence.get("preempt_type")
    if preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
        raise ProtectedCapacityError(
            "scheduler evidence has unsupported Slurm PreemptType"
        )
    evidence_id = _validate_self_hash(
        evidence,
        identity_field="evidence_id",
        description="scheduler evidence",
    )
    effective_logical_replicas = _require_integer(
        evidence.get("effective_active_logical_replicas"),
        minimum=BASE_LOGICAL_REPLICAS,
        description="scheduler effective active logical replicas",
    )
    capacity_generation = _require_integer(
        evidence.get("capacity_generation"),
        minimum=1,
        description="scheduler capacity generation",
    )
    job_element_accounting = _validate_job_element_accounting(
        evidence.get("job_element_accounting"),
        active_server_job_elements=effective_logical_replicas,
    )
    if (
        _require_integer(
            evidence.get("expected_total_job_elements"),
            minimum=EXPECTED_TOTAL_JOB_ELEMENTS,
            description="scheduler expected total job elements",
        )
        != EXPECTED_TOTAL_JOB_ELEMENTS
    ):
        raise ProtectedCapacityError(
            "scheduler expected total job elements must be exactly 448"
        )
    servers, server_totals = _validate_scheduler_server_rows(
        evidence.get("scientific_server_placements"),
        preempt_type=str(preempt_type),
        capacity_generation=capacity_generation,
    )
    clients, client_totals = _validate_scheduler_client_rows(
        evidence.get("scientific_client_placements"),
        preempt_type=str(preempt_type),
    )
    derived = _derive_capacity_from_raw_sources(
        evidence,
        servers=servers,
        clients=clients,
        expected_source_tree_sha256=source_trust["source_tree_sha256"],
        expected_dispatcher_source_sha256=source_trust[
            "dispatcher_source_sha256"
        ],
        expected_qualification_runner_source_sha256=source_trust[
            "qualification_runner_source_sha256"
        ],
    )
    occupancy_preflight = _validate_occupancy_preflight(
        evidence.get("occupancy_preflight"),
        scheduler_account=str(derived["scheduler_account"]),
        scheduler_max_jobs=derived["scheduler_max_jobs"],
        scheduler_max_submit_jobs=int(
            derived["scheduler_max_submit_jobs"]
        ),
        qos_contracts=derived["scientific_qos_contracts"],
        expected_running_jobs=int(derived["running_scientific_jobs"]),
    )
    if (
        derived["preempt_type"] != preempt_type
        or derived["capacity_source"] != evidence.get("capacity_source")
        or derived["scheduler_cluster"] != evidence.get("scheduler_cluster")
        or derived["scheduler_account"] != evidence.get("scheduler_account")
        or derived["scheduler_user"] != evidence.get("scheduler_user")
        or derived["scheduler_max_jobs"]
        != evidence.get("scheduler_max_jobs")
        or derived["scheduler_max_submit_jobs"]
        != evidence.get("scheduler_max_submit_jobs")
        or derived["running_scientific_jobs"]
        != evidence.get("running_scientific_jobs")
        or derived["minimum_scientific_wall_seconds"]
        != evidence.get("minimum_scientific_wall_seconds")
        or derived["scientific_qos_contracts"]
        != evidence.get("scientific_qos_contracts")
        or derived["partition_cpus"] != evidence.get("partition_cpus")
        or derived["partition_memory_mib"]
        != evidence.get("partition_memory_mib")
        or derived["partition_gpus"] != evidence.get("partition_gpus")
        or any(
            derived[derived_field] != server_totals[placement_field]
            for placement_field, derived_field in (
                ("base_active_gpus", "base_active_gpus"),
                ("reserved_additive_gpus", "additive_reserved_gpus"),
                ("effective_active_gpus", "effective_active_gpus"),
                (
                    "retained_warm_turnover_gpus",
                    "retained_warm_turnover_gpus",
                ),
                ("attested_total_gpus", "attested_total_gpus"),
            )
        )
        or any(derived[field] != client_totals[field] for field in client_totals)
        or derived["fleet_contract_sha256"]
        != evidence.get("fleet_contract_sha256")
        or derived["active_fleet_topology_sha256"]
        != evidence.get("active_fleet_topology_sha256")
        or derived["builder_source_sha256"]
        != evidence.get("builder_source_sha256")
        or derived["publisher_source_sha256"]
        != evidence.get("publisher_source_sha256")
        or any(
            derived[field] != evidence.get(field)
            for field in _SOURCE_TRUST_FIELDS
        )
        or derived["expected_total_job_elements"]
        != evidence.get("expected_total_job_elements")
        or derived["job_element_accounting"] != job_element_accounting
        or any(
            derived[field] != evidence.get(field)
            for field in (
                "capacity_generation",
                "base_fleet_contract_path",
                "base_fleet_contract_sha256",
                "effective_fleet_contract_path",
                "effective_fleet_contract_sha256",
                "additive_overlay_contract_path",
                "additive_overlay_contract_sha256",
                "static_feasibility_certificate",
                "base_active_logical_replicas",
                "base_active_gpus",
                "base_active_topology",
                "base_active_topology_sha256",
                "additive_reserved_logical_replicas",
                "additive_reserved_gpus",
                "additive_reserved_tp1_replicas",
                "additive_reserved_tp2_replicas",
                "additive_reserved_topology",
                "additive_reserved_topology_sha256",
                "effective_active_logical_replicas",
                "effective_active_gpus",
                "effective_active_topology",
                "effective_active_topology_sha256",
                "retained_warm_turnover_job_elements",
                "retained_warm_turnover_gpus",
                "retained_warm_turnover_tp1_allocations",
                "retained_warm_turnover_tp2_allocations",
                "retained_warm_turnover_topology",
                "retained_warm_turnover_topology_sha256",
                "attested_total_gpus",
            )
        )
    ):
        raise ProtectedCapacityError(
            "scheduler evidence claims differ from independently parsed raw "
            "scheduler/canary capacity"
        )
    squeue = _validate_scheduler_source(evidence.get("squeue"), command="squeue")
    sacct = _validate_scheduler_source(evidence.get("sacct"), command="sacct")
    return {
        "binding": binding,
        "observed_at": observed_at,
        "evidence_id": evidence_id,
        "preempt_type": preempt_type,
        "capacity_source": derived["capacity_source"],
        "scheduler_cluster": derived["scheduler_cluster"],
        "scheduler_account": derived["scheduler_account"],
        "scheduler_user": derived["scheduler_user"],
        "scheduler_max_jobs": derived["scheduler_max_jobs"],
        "scheduler_max_submit_jobs": derived[
            "scheduler_max_submit_jobs"
        ],
        "running_scientific_jobs": derived[
            "running_scientific_jobs"
        ],
        "minimum_scientific_wall_seconds": derived[
            "minimum_scientific_wall_seconds"
        ],
        "scientific_qos_contracts": derived[
            "scientific_qos_contracts"
        ],
        "partition_cpus": derived["partition_cpus"],
        "partition_memory_mib": derived["partition_memory_mib"],
        "partition_gpus": derived["partition_gpus"],
        **{
            field: derived[field]
            for field in (
                "capacity_generation",
                "base_fleet_contract_path",
                "base_fleet_contract_sha256",
                "effective_fleet_contract_path",
                "effective_fleet_contract_sha256",
                "additive_overlay_contract_path",
                "additive_overlay_contract_sha256",
                "static_feasibility_certificate",
                "static_feasibility_wave_passed",
                "static_feasibility_selected_cell_count",
                "static_feasibility_target_cell_count",
                "static_feasibility_shortfall_cells",
                "static_feasibility_configured_client_ceiling",
                "static_feasibility_certified_saturation_target",
                "base_active_logical_replicas",
                "base_active_gpus",
                "base_active_topology",
                "base_active_topology_sha256",
                "additive_reserved_logical_replicas",
                "additive_reserved_gpus",
                "additive_reserved_tp1_replicas",
                "additive_reserved_tp2_replicas",
                "additive_reserved_topology",
                "additive_reserved_topology_sha256",
                "effective_active_logical_replicas",
                "effective_active_gpus",
                "effective_active_topology",
                "effective_active_topology_sha256",
                "retained_warm_turnover_job_elements",
                "retained_warm_turnover_gpus",
                "retained_warm_turnover_tp1_allocations",
                "retained_warm_turnover_tp2_allocations",
                "retained_warm_turnover_topology",
                "retained_warm_turnover_topology_sha256",
                "attested_total_gpus",
            )
        },
        "servers": servers,
        "clients": clients,
        "squeue": squeue,
        "sacct": sacct,
        "fleet_contract_sha256": derived["fleet_contract_sha256"],
        "active_fleet_topology_sha256": derived[
            "active_fleet_topology_sha256"
        ],
        "builder_source_sha256": derived["builder_source_sha256"],
        "publisher_source_sha256": derived["publisher_source_sha256"],
        **source_trust,
        "expected_total_job_elements": derived[
            "expected_total_job_elements"
        ],
        "job_element_accounting": derived["job_element_accounting"],
        "occupancy_preflight": occupancy_preflight,
        **server_totals,
        **client_totals,
    }


def _validate_canary_rows(
    rows: Any,
    *,
    expected: Sequence[Mapping[str, Any]],
    role: str,
    preempt_type: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ProtectedCapacityError(
            f"canary scientific {role} placement cardinality drifted"
        )
    normalized: list[dict[str, Any]] = []
    observed_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ProtectedCapacityError(
                f"canary scientific {role} placement {index} is malformed"
            )
        _require_exact_fields(
            raw,
            _CANARY_PLACEMENT_FIELDS,
            description=f"canary scientific {role} placement {index}",
        )
        identity = _placement_identity(
            raw, role=role, preempt_type=preempt_type
        )
        if identity in observed_by_identity:
            raise ProtectedCapacityError(
                f"canary duplicates scientific {role} placement "
                f"{identity[0]}/{identity[1]}"
            )
        if raw.get("effective_requeue") != 0:
            raise ProtectedCapacityError(
                f"scientific {role} canary placement "
                f"{identity[0]}/{identity[1]} did not prove Requeue=0"
            )
        observed_by_identity[identity] = dict(raw)
        normalized.append(dict(raw))
    expected_by_identity = {
        (str(row["partition"]), str(row["qos"])): row for row in expected
    }
    if set(observed_by_identity) != set(expected_by_identity):
        raise ProtectedCapacityError(
            f"canary scientific {role} placements differ from scheduler evidence"
        )
    for identity, raw in observed_by_identity.items():
        scheduler = expected_by_identity[identity]
        for field in ("partition_preempt_mode", "qos_preempt_mode"):
            if raw[field] != scheduler[field]:
                raise ProtectedCapacityError(
                    f"canary scientific {role} placement "
                    f"{identity[0]}/{identity[1]} policy drifted"
                )
    return normalized


def validate_canary_evidence(
    evidence: Mapping[str, Any],
    *,
    scheduler: Mapping[str, Any],
    expected_release_git_commit: str,
    expected_release_tag_object: str,
) -> dict[str, Any]:
    """Validate a canary and its exact cross-binding to scheduler evidence."""

    _require_exact_fields(
        evidence,
        _CANARY_FIELDS,
        description="canary evidence",
    )
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("protocol") != CANARY_EVIDENCE_PROTOCOL
        or evidence.get("passed") is not True
    ):
        raise ProtectedCapacityError(
            "canary evidence envelope is invalid or not passed"
        )
    binding = _validate_release_binding(
        evidence,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        description="canary evidence",
    )
    if binding != scheduler.get("binding"):
        raise ProtectedCapacityError(
            "scheduler and canary release/tag/chain bindings differ"
        )
    completed_at = _require_positive_timestamp(
        evidence.get("completed_at"),
        description="canary completed_at",
    )
    if completed_at < float(scheduler["observed_at"]):
        raise ProtectedCapacityError(
            "canary completion predates its bound scheduler observation"
        )
    canary_id = _validate_self_hash(
        evidence,
        identity_field="canary_id",
        description="canary evidence",
    )
    if evidence.get("scheduler_evidence_id") != scheduler.get("evidence_id"):
        raise ProtectedCapacityError(
            "canary is not bound to the exact scheduler evidence ID"
        )
    _validate_canary_rows(
        evidence.get("scientific_server_placements"),
        expected=scheduler["servers"],
        role="server",
        preempt_type=str(scheduler["preempt_type"]),
    )
    _validate_canary_rows(
        evidence.get("scientific_client_placements"),
        expected=scheduler["clients"],
        role="client",
        preempt_type=str(scheduler["preempt_type"]),
    )
    if (
        evidence.get("squeue_complete") is not True
        or evidence.get("sacct_complete") is not True
        or scheduler["squeue"].get("complete") is not True
        or scheduler["sacct"].get("complete") is not True
    ):
        raise ProtectedCapacityError("complete joined squeue+sacct truth is required")
    return {
        "binding": binding,
        "completed_at": completed_at,
        "canary_id": canary_id,
    }


def build_marker(
    scheduler_evidence: Mapping[str, Any],
    canary_evidence: Mapping[str, Any],
    *,
    scheduler_evidence_sha256: str,
    canary_evidence_sha256: str,
    expected_release_git_commit: str,
    expected_release_tag_object: str,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact downstream marker from two validated evidence objects."""

    scheduler = validate_scheduler_evidence(
        scheduler_evidence,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
    )
    canary = validate_canary_evidence(
        canary_evidence,
        scheduler=scheduler,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
    )
    if (
        _SHA256_RE.fullmatch(scheduler_evidence_sha256) is None
        or _SHA256_RE.fullmatch(canary_evidence_sha256) is None
    ):
        raise ProtectedCapacityError(
            "protected-capacity source evidence hashes must be lowercase SHA-256"
        )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "passed": True,
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "release_git_commit": expected_release_git_commit,
        "release_tag_object": expected_release_tag_object,
        "chain_namespace": CHAIN_NAMESPACE,
        **{
            field: scheduler[field]
            for field in sorted(_SOURCE_TRUST_FIELDS)
        },
        **{
            field: scheduler[field]
            for field in (
                "capacity_generation",
                "base_fleet_contract_path",
                "base_fleet_contract_sha256",
                "effective_fleet_contract_path",
                "effective_fleet_contract_sha256",
                "additive_overlay_contract_path",
                "additive_overlay_contract_sha256",
                "static_feasibility_certificate",
                "static_feasibility_wave_passed",
                "static_feasibility_selected_cell_count",
                "static_feasibility_target_cell_count",
                "static_feasibility_shortfall_cells",
                "static_feasibility_configured_client_ceiling",
                "static_feasibility_certified_saturation_target",
                "base_active_logical_replicas",
                "base_active_gpus",
                "base_active_topology",
                "base_active_topology_sha256",
                "additive_reserved_logical_replicas",
                "additive_reserved_gpus",
                "additive_reserved_tp1_replicas",
                "additive_reserved_tp2_replicas",
                "additive_reserved_topology",
                "additive_reserved_topology_sha256",
                "effective_active_logical_replicas",
                "effective_active_gpus",
                "effective_active_topology",
                "effective_active_topology_sha256",
                "retained_warm_turnover_job_elements",
                "retained_warm_turnover_gpus",
                "retained_warm_turnover_tp1_allocations",
                "retained_warm_turnover_tp2_allocations",
                "retained_warm_turnover_topology",
                "retained_warm_turnover_topology_sha256",
                "attested_total_gpus",
                "job_element_accounting",
            )
        },
        "active_gpus": scheduler["effective_active_gpus"],
        "warm_headroom_gpus": scheduler[
            "retained_warm_turnover_gpus"
        ],
        "cell_ceiling": scheduler["cell_ceiling"],
        "reserve_jobs": scheduler["reserve_jobs"],
        "submit_headroom": scheduler["submit_headroom"],
        "cpu": scheduler["cpu"],
        "memory_mib": scheduler["memory_mib"],
        "preempt_type": scheduler["preempt_type"],
        "capacity_source": scheduler["capacity_source"],
        "scheduler_cluster": scheduler["scheduler_cluster"],
        "scheduler_account": scheduler["scheduler_account"],
        "scheduler_user": scheduler["scheduler_user"],
        "scheduler_max_jobs": scheduler["scheduler_max_jobs"],
        "scheduler_max_submit_jobs": scheduler[
            "scheduler_max_submit_jobs"
        ],
        "running_scientific_jobs": scheduler[
            "running_scientific_jobs"
        ],
        "minimum_scientific_wall_seconds": scheduler[
            "minimum_scientific_wall_seconds"
        ],
        "scientific_qos_contracts": scheduler[
            "scientific_qos_contracts"
        ],
        "partition_cpus": scheduler["partition_cpus"],
        "partition_memory_mib": scheduler["partition_memory_mib"],
        "partition_gpus": scheduler["partition_gpus"],
        "fleet_contract_sha256": scheduler["fleet_contract_sha256"],
        "active_fleet_topology_sha256": scheduler[
            "active_fleet_topology_sha256"
        ],
        "scientific_server_preempt_mode": "OFF",
        "scientific_client_preempt_mode": "OFF",
        "scientific_server_placements": scheduler["servers"],
        "scientific_client_placements": scheduler["clients"],
        "scheduler_evidence_id": scheduler["evidence_id"],
        "scheduler_evidence_sha256": scheduler_evidence_sha256,
        "canary_id": canary["canary_id"],
        "canary_evidence_sha256": canary_evidence_sha256,
        "squeue_complete": True,
        "sacct_complete": True,
    }
    return with_self_hash(marker, identity_field="marker_id")


def _validate_scientific_qos_contracts(
    value: Any,
    *,
    expected_running_jobs: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ProtectedCapacityError(
            "protected-capacity scientific QOS contracts are absent"
        )
    normalized: list[dict[str, Any]] = []
    prior_qos = ""
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ProtectedCapacityError(
                f"scientific QOS contract {index} is malformed"
            )
        _require_exact_fields(
            raw,
            _QOS_CONTRACT_FIELDS,
            description=f"scientific QOS contract {index}",
        )
        qos = raw.get("qos")
        if (
            not isinstance(qos, str)
            or _PLACEMENT_NAME_RE.fullmatch(qos) is None
            or qos <= prior_qos
        ):
            raise ProtectedCapacityError(
                "scientific QOS contracts are duplicated or not canonical"
            )
        prior_qos = qos
        optional: dict[str, int | None] = {}
        for field in (
            "max_wall_seconds",
            "max_jobs_per_user",
            "max_submit_jobs_per_user",
        ):
            observed = raw.get(field)
            if observed is None:
                optional[field] = None
            else:
                optional[field] = _require_integer(
                    observed,
                    minimum=1,
                    description=f"{qos} {field}",
                )
        required_wall = _require_integer(
            raw.get("required_wall_seconds"),
            minimum=CLIENT_WALL_SECONDS,
            description=f"{qos} required walltime",
        )
        required_running = _require_integer(
            raw.get("required_running_jobs"),
            minimum=0,
            description=f"{qos} required running jobs",
        )
        required_submit = _require_integer(
            raw.get("required_submit_jobs"),
            minimum=1,
            description=f"{qos} required submit jobs",
        )
        if (
            optional["max_wall_seconds"] is not None
            and optional["max_wall_seconds"] < required_wall
        ) or (
            optional["max_jobs_per_user"] is not None
            and optional["max_jobs_per_user"] < required_running
        ) or (
            optional["max_submit_jobs_per_user"] is not None
            and optional["max_submit_jobs_per_user"] < required_submit
        ):
            raise ProtectedCapacityError(
                f"scientific QOS contract {qos} cannot sustain its bound load"
            )
        normalized.append(dict(raw))
    if (
        sum(row["required_running_jobs"] for row in normalized)
        != expected_running_jobs
        or sum(row["required_submit_jobs"] for row in normalized)
        != EXPECTED_TOTAL_JOB_ELEMENTS
        or max(row["required_wall_seconds"] for row in normalized)
        != MIN_SCIENTIFIC_WALL_SECONDS
    ):
        raise ProtectedCapacityError(
            "scientific QOS contracts do not realize the dynamic running/"
            "448-submit/24-hour envelope"
        )
    return normalized


def validate_marker_payload(
    marker: Mapping[str, Any],
    *,
    expected_release_git_commit: str | None = None,
    expected_release_tag_object: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact marker schema used by the r11 chain renderer."""

    _require_exact_fields(
        marker,
        _MARKER_FIELDS,
        description="protected-capacity marker",
    )
    commit = marker.get("release_git_commit")
    tag_object = marker.get("release_tag_object")
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("passed") is not True
        or marker.get("release_id") != RELEASE_ID
        or marker.get("release_tag") != RELEASE_TAG
        or marker.get("chain_namespace") != CHAIN_NAMESPACE
        or not isinstance(commit, str)
        or _GIT_OBJECT_RE.fullmatch(commit) is None
        or not isinstance(tag_object, str)
        or _GIT_OBJECT_RE.fullmatch(tag_object) is None
        or (
            expected_release_git_commit is not None
            and commit != expected_release_git_commit
        )
        or (
            expected_release_tag_object is not None
            and tag_object != expected_release_tag_object
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker release/tag/chain binding is invalid"
        )
    source_trust = _validate_source_trust_anchors(
        marker,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
        description="protected-capacity marker",
    )
    _validate_self_hash(
        marker,
        identity_field="marker_id",
        description="protected-capacity marker",
    )
    effective_count = _require_integer(
        marker.get("effective_active_logical_replicas"),
        minimum=BASE_LOGICAL_REPLICAS,
        description="protected-capacity effective logical replicas",
    )
    effective_gpus = _require_integer(
        marker.get("effective_active_gpus"),
        minimum=BASE_ACTIVE_GPUS,
        description="protected-capacity effective active GPUs",
    )
    capacity_generation = _require_integer(
        marker.get("capacity_generation"),
        minimum=1,
        description="protected-capacity capacity generation",
    )
    expected_running_jobs = (
        MIN_CLIENT_SLOTS
        + effective_count
        + EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
    )
    minima = {
        "active_gpus": effective_gpus,
        "warm_headroom_gpus": RETAINED_WARM_TURNOVER_GPUS,
        "cell_ceiling": MIN_CLIENT_SLOTS,
        "reserve_jobs": MIN_RESERVE_JOBS,
        "submit_headroom": MIN_SUBMIT_HEADROOM,
        "cpu": MIN_CLIENT_CPUS,
        "memory_mib": MIN_CLIENT_MEMORY_MIB,
    }
    values = {
        field: _require_integer(
            marker.get(field),
            minimum=minimum,
            description=f"protected-capacity marker {field}",
        )
        for field, minimum in minima.items()
    }
    servers, server_totals = _validate_scheduler_server_rows(
        marker.get("scientific_server_placements"),
        preempt_type=str(marker.get("preempt_type")),
        capacity_generation=capacity_generation,
    )
    clients, client_totals = _validate_scheduler_client_rows(
        marker.get("scientific_client_placements"),
        preempt_type=str(marker.get("preempt_type")),
    )
    qos_contracts = _validate_scientific_qos_contracts(
        marker.get("scientific_qos_contracts"),
        expected_running_jobs=expected_running_jobs,
    )
    qos_contract_by_name = {
        str(row["qos"]): row for row in qos_contracts
    }
    scheduler_max_jobs = marker.get("scheduler_max_jobs")
    if scheduler_max_jobs is not None:
        scheduler_max_jobs = _require_integer(
            scheduler_max_jobs,
            minimum=expected_running_jobs,
            description="protected-capacity scheduler MaxJobs",
        )
    _require_integer(
        marker.get("scheduler_max_submit_jobs"),
        minimum=MIN_SUBMIT_HEADROOM,
        description="protected-capacity scheduler MaxSubmitJobs",
    )
    if (
        marker.get("scientific_server_placements") != servers
        or marker.get("scientific_client_placements") != clients
    ):
        raise ProtectedCapacityError(
            "protected-capacity placement rows are not canonically sorted"
        )
    source_identities = (
        "scheduler_evidence_id",
        "scheduler_evidence_sha256",
        "canary_id",
        "canary_evidence_sha256",
        "base_fleet_contract_sha256",
        "effective_fleet_contract_sha256",
        "additive_overlay_contract_sha256",
        "base_active_topology_sha256",
        "additive_reserved_topology_sha256",
        "effective_active_topology_sha256",
        "retained_warm_turnover_topology_sha256",
        "fleet_contract_sha256",
        "active_fleet_topology_sha256",
    )
    certificate_binding = marker.get("static_feasibility_certificate")
    if (
        not isinstance(certificate_binding, Mapping)
        or set(certificate_binding)
        != {"path", "sha256", "certificate_id"}
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker static feasibility binding is malformed"
        )
    try:
        certificate = runtime_capacity.load_static_feasibility_certificate(
            str(certificate_binding["path"]),
            expected_sha256=str(certificate_binding["sha256"]),
            expected_certificate_id=str(certificate_binding["certificate_id"]),
            expected_capacity_generation=int(marker["capacity_generation"]),
            expected_base_fleet_contract_sha256=str(
                marker["base_fleet_contract_sha256"]
            ),
            expected_effective_fleet_contract_sha256=str(
                marker["effective_fleet_contract_sha256"]
            ),
            expected_additive_overlay_contract_sha256=str(
                marker["additive_overlay_contract_sha256"]
            ),
            expected_release_git_commit=str(commit),
            expected_source_tree_sha256=source_trust[
                "source_tree_sha256"
            ],
            expected_dispatcher_source_sha256=source_trust[
                "dispatcher_source_sha256"
            ],
            expected_qualification_runner_source_sha256=source_trust[
                "qualification_runner_source_sha256"
            ],
        )
    except (runtime_capacity.ProtectedCapacityError, TypeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"protected-capacity static feasibility binding failed: {exc}"
        ) from exc
    base_count = _require_integer(
        marker.get("base_active_logical_replicas"),
        minimum=BASE_LOGICAL_REPLICAS,
        description="protected-capacity base logical replicas",
    )
    base_gpus = _require_integer(
        marker.get("base_active_gpus"),
        minimum=BASE_ACTIVE_GPUS,
        description="protected-capacity base GPUs",
    )
    additive_count = _require_integer(
        marker.get("additive_reserved_logical_replicas"),
        minimum=0,
        description="protected-capacity additive logical replicas",
    )
    additive_gpus = _require_integer(
        marker.get("additive_reserved_gpus"),
        minimum=0,
        description="protected-capacity additive GPUs",
    )
    additive_tp1 = _require_integer(
        marker.get("additive_reserved_tp1_replicas"),
        minimum=0,
        description="protected-capacity additive TP1 replicas",
    )
    additive_tp2 = _require_integer(
        marker.get("additive_reserved_tp2_replicas"),
        minimum=0,
        description="protected-capacity additive TP2 replicas",
    )
    warm_count = _require_integer(
        marker.get("retained_warm_turnover_job_elements"),
        minimum=1,
        description="protected-capacity retained warm jobs",
    )
    warm_gpus = _require_integer(
        marker.get("retained_warm_turnover_gpus"),
        minimum=1,
        description="protected-capacity retained warm GPUs",
    )
    topology_specs = (
        ("base_active_topology", base_count, base_gpus),
        ("additive_reserved_topology", additive_count, additive_gpus),
        ("effective_active_topology", effective_count, effective_gpus),
        ("retained_warm_turnover_topology", warm_count, warm_gpus),
    )
    parsed_topologies: dict[str, tuple[Mapping[str, Any], ...]] = {}
    try:
        for field, count, gpus in topology_specs:
            parsed_topologies[field] = runtime_capacity._validated_topology(  # noqa: SLF001
                marker.get(field),
                field=field,
                expected_count=count,
                expected_gpus=gpus,
            )
    except runtime_capacity.ProtectedCapacityError as exc:
        raise ProtectedCapacityError(str(exc)) from exc
    base_by_id = {
        str(row["shape_id"]): row
        for row in parsed_topologies["base_active_topology"]
    }
    additive_by_id = {
        str(row["shape_id"]): row
        for row in parsed_topologies["additive_reserved_topology"]
    }
    effective_by_id = {
        str(row["shape_id"]): row
        for row in parsed_topologies["effective_active_topology"]
    }
    expected_held = (
        MIN_RESERVE_JOBS
        - effective_count
        - EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
    )
    expected_accounting = _expected_job_element_accounting(effective_count)
    for path_field, hash_field in (
        ("base_fleet_contract_path", "base_fleet_contract_sha256"),
        ("effective_fleet_contract_path", "effective_fleet_contract_sha256"),
        ("additive_overlay_contract_path", "additive_overlay_contract_sha256"),
    ):
        raw_path = marker.get(path_field)
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ProtectedCapacityError(
                f"protected-capacity marker {path_field} is not absolute"
            )
        _bound_path, _bound_payload, observed_sha256 = (
            _read_sealed_json_with_sha256(
                Path(raw_path),
                description=f"protected-capacity {path_field}",
            )
        )
        if observed_sha256 != marker.get(hash_field):
            raise ProtectedCapacityError(
                f"protected-capacity marker {path_field} bytes drifted"
            )
    if (
        values["cpu"] < values["cell_ceiling"]
        or values["memory_mib"] < values["cell_ceiling"] * CLIENT_MEMORY_MIB_PER_SLOT
        or values["submit_headroom"] < values["cell_ceiling"] + values["reserve_jobs"]
        or marker.get("running_scientific_jobs")
        != expected_running_jobs
        or marker.get("minimum_scientific_wall_seconds")
        != MIN_SCIENTIFIC_WALL_SECONDS
        or {
            str(row["qos"]) for row in qos_contracts
        }
        != {
            str(row["qos"]) for row in [*servers, *clients]
        }
        or any(
            int(qos_contract_by_name[str(row["qos"])][
                "required_wall_seconds"
            ])
            != MIN_SCIENTIFIC_WALL_SECONDS
            for row in servers
        )
        or any(
            int(qos_contract_by_name[str(row["qos"])][
                "required_wall_seconds"
            ])
            < CLIENT_WALL_SECONDS
            for row in clients
        )
        or (
            scheduler_max_jobs is not None
            and scheduler_max_jobs < expected_running_jobs
        )
        or values["active_gpus"] != effective_gpus
        or values["warm_headroom_gpus"] != warm_gpus
        or server_totals["base_active_gpus"] != base_gpus
        or server_totals["reserved_additive_gpus"] != additive_gpus
        or server_totals["effective_active_gpus"] != effective_gpus
        or server_totals["retained_warm_turnover_gpus"] != warm_gpus
        or server_totals["attested_total_gpus"]
        != marker.get("attested_total_gpus")
        or any(values[field] != client_totals[field] for field in client_totals)
        or any(
            not isinstance(marker.get(field), str)
            or _SHA256_RE.fullmatch(str(marker[field])) is None
            for field in source_identities
        )
        or marker.get("scientific_server_preempt_mode") != "OFF"
        or marker.get("scientific_client_preempt_mode") != "OFF"
        or marker.get("preempt_type")
        not in {"preempt/partition_prio", "preempt/qos"}
        or marker.get("squeue_complete") is not True
        or marker.get("sacct_complete") is not True
        or marker.get("fleet_contract_sha256")
        != marker.get("effective_fleet_contract_sha256")
        or marker.get("active_fleet_topology_sha256")
        != marker.get("effective_active_topology_sha256")
        or marker.get("base_fleet_contract_sha256")
        != certificate.base_fleet_contract_sha256
        or marker.get("effective_fleet_contract_sha256")
        != certificate.effective_fleet_contract_sha256
        or marker.get("static_feasibility_wave_passed")
        is not certificate.wave_passed
        or marker.get("static_feasibility_selected_cell_count")
        != certificate.selected_cell_count
        or marker.get("static_feasibility_target_cell_count")
        != certificate.target_cell_count
        or marker.get("static_feasibility_shortfall_cells")
        != certificate.shortfall_cells
        or marker.get("static_feasibility_configured_client_ceiling")
        != certificate.target_cell_count
        or marker.get("static_feasibility_certified_saturation_target")
        != certificate.selected_cell_count
        or marker.get("cell_ceiling") != certificate.target_cell_count
        or marker.get("additive_overlay_contract_sha256")
        != certificate.additive_overlay_contract_sha256
        or base_count != BASE_LOGICAL_REPLICAS
        or base_gpus != BASE_ACTIVE_GPUS
        or effective_count != base_count + additive_count
        or effective_gpus != base_gpus + additive_gpus
        or additive_count != additive_tp1 + additive_tp2
        or additive_gpus != additive_tp1 + 2 * additive_tp2
        or additive_tp1 != certificate.additive_tp1_logical_replicas
        or additive_tp2 != certificate.additive_tp2_logical_replicas
        or additive_gpus != certificate.additive_allocated_gpus
        or set(base_by_id).intersection(additive_by_id)
        or effective_by_id != {**base_by_id, **additive_by_id}
        or warm_count != EXPECTED_WARM_TURNOVER_JOB_ELEMENTS
        or warm_gpus != RETAINED_WARM_TURNOVER_GPUS
        or marker.get("retained_warm_turnover_tp1_allocations") != 2
        or marker.get("retained_warm_turnover_tp2_allocations") != 1
        or sorted(
            int(row["gpus"])
            for row in parsed_topologies[
                "retained_warm_turnover_topology"
            ]
        )
        != [1, 1, 2]
        or marker.get("attested_total_gpus", 0)
        < effective_gpus + warm_gpus
        or expected_held <= 0
        or marker.get("job_element_accounting") != expected_accounting
        or any(
            marker.get(f"{field}_sha256")
            != hashlib.sha256(canonical_bytes(marker.get(field))).hexdigest()
            for field in (
                "base_active_topology",
                "additive_reserved_topology",
                "effective_active_topology",
                "retained_warm_turnover_topology",
            )
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker does not satisfy placement/capacity/"
            "scheduler invariants"
        )
    return dict(marker)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes) -> None:
    """Atomically link a fully written read-only inode into place, without replace."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".publishing",
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError as exc:
            raise ProtectedCapacityError(
                f"completion marker appeared concurrently: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(path.parent)


def verify_marker(
    recovery_root: Path,
    *,
    expected_release_git_commit: str | None = None,
    expected_release_tag_object: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the canonical read-only marker without consulting live evidence."""

    root = _path_without_symlinks(
        recovery_root,
        description="recovery root",
        kind="directory",
    )
    marker_path, marker = read_sealed_json(
        root / MARKER_FILENAME,
        description="protected-capacity completion marker",
    )
    if marker_path.parent != root:
        raise ProtectedCapacityError(
            "protected-capacity marker is outside the canonical recovery root"
        )
    return validate_marker_payload(
        marker,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
    )


def _require_evidence_unchanged(
    path: Path,
    expected: Mapping[str, Any],
    *,
    expected_sha256: str,
    description: str,
) -> None:
    reread_path, reread, reread_sha256 = _read_sealed_json_with_sha256(
        path,
        description=description,
    )
    if (
        reread_path != path
        or reread != dict(expected)
        or reread_sha256 != expected_sha256
    ):
        raise ProtectedCapacityError(
            f"{description} drifted during protected-capacity attestation"
        )


def _require_live_scheduler_sources_unchanged(
    evidence: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> None:
    """Re-execute every normalized capture immediately before publication."""

    source_fields = (
        "scheduler_configuration",
        "partition_configuration",
        "qos_configuration",
        "association_configuration",
        "base_fleet_contract",
        "effective_fleet_contract",
        "additive_overlay_contract",
        "static_feasibility_certificate_source",
        "builder_source",
        "publisher_source",
        "squeue",
        "sacct",
    )
    for field in source_fields:
        source = evidence.get(field)
        if not isinstance(source, Mapping):
            raise ProtectedCapacityError(
                f"live protected-capacity source {field} is missing"
            )
        argv = source.get("argv")
        if not isinstance(argv, list) or not argv:
            raise ProtectedCapacityError(
                f"live protected-capacity source {field} has no argv"
            )
        try:
            result = (
                subprocess.run(
                    list(argv),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=300.0,
                )
                if runner is None
                else runner(list(argv), timeout=300.0)
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProtectedCapacityError(
                f"live protected-capacity recapture failed for {field}: {exc}"
            ) from exc
        if (
            not isinstance(result, subprocess.CompletedProcess)
            or result.returncode != 0
            or result.stdout != source.get("raw_output")
            or not isinstance(result.stderr, str)
        ):
            raise ProtectedCapacityError(
                f"live protected-capacity recapture drifted for {field}"
            )


def attest(
    *,
    recovery_root: Path,
    scheduler_evidence_path: Path,
    canary_evidence_path: Path,
    expected_release_git_commit: str,
    expected_release_tag_object: str,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
    apply: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Validate sealed evidence and optionally publish the marker exactly once."""

    root = _path_without_symlinks(
        recovery_root,
        description="recovery root",
        kind="directory",
    )
    scheduler_path, scheduler_evidence, scheduler_evidence_sha256 = (
        _read_sealed_json_with_sha256(
            scheduler_evidence_path,
            description="protected-capacity scheduler evidence",
        )
    )
    canary_path, canary_evidence, canary_evidence_sha256 = (
        _read_sealed_json_with_sha256(
            canary_evidence_path,
            description="protected-capacity canary evidence",
        )
    )
    if (
        scheduler_path == canary_path
        or scheduler_path == root / MARKER_FILENAME
        or canary_path == root / MARKER_FILENAME
    ):
        raise ProtectedCapacityError(
            "scheduler, canary, and completion marker paths must be distinct"
        )
    scheduler_stat = scheduler_path.stat(follow_symlinks=False)
    canary_stat = canary_path.stat(follow_symlinks=False)
    if (scheduler_stat.st_dev, scheduler_stat.st_ino) == (
        canary_stat.st_dev,
        canary_stat.st_ino,
    ):
        raise ProtectedCapacityError(
            "scheduler and canary evidence cannot share an inode"
        )
    proposed = build_marker(
        scheduler_evidence,
        canary_evidence,
        scheduler_evidence_sha256=scheduler_evidence_sha256,
        canary_evidence_sha256=canary_evidence_sha256,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
    )
    # Reopen both trust anchors after all semantic checks.  This closes the
    # validation/publication window even if another process can chmod a sealed file.
    _require_evidence_unchanged(
        scheduler_path,
        scheduler_evidence,
        expected_sha256=scheduler_evidence_sha256,
        description="protected-capacity scheduler evidence",
    )
    _require_evidence_unchanged(
        canary_path,
        canary_evidence,
        expected_sha256=canary_evidence_sha256,
        description="protected-capacity canary evidence",
    )
    marker_path = root / MARKER_FILENAME
    payload = canonical_bytes(proposed)
    if marker_path.exists() or marker_path.is_symlink():
        existing = verify_marker(
            root,
            expected_release_git_commit=expected_release_git_commit,
            expected_release_tag_object=expected_release_tag_object,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_dispatcher_source_sha256=(
                expected_dispatcher_source_sha256
            ),
            expected_qualification_runner_source_sha256=(
                expected_qualification_runner_source_sha256
            ),
        )
        if canonical_bytes(existing) != payload:
            raise ProtectedCapacityError(
                "existing protected-capacity marker conflicts with sealed evidence"
            )
        return {
            "status": "already_complete",
            "apply": apply,
            "marker_path": str(marker_path),
            "marker": existing,
        }
    if not apply:
        return {
            "status": "dry_run",
            "apply": False,
            "marker_path": str(marker_path),
            "marker": proposed,
        }
    _require_live_scheduler_sources_unchanged(
        scheduler_evidence,
        runner=runner,
    )
    _require_evidence_unchanged(
        scheduler_path,
        scheduler_evidence,
        expected_sha256=scheduler_evidence_sha256,
        description="protected-capacity scheduler evidence",
    )
    _publish_once(marker_path, payload)
    published = verify_marker(
        root,
        expected_release_git_commit=expected_release_git_commit,
        expected_release_tag_object=expected_release_tag_object,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_dispatcher_source_sha256=(
            expected_dispatcher_source_sha256
        ),
        expected_qualification_runner_source_sha256=(
            expected_qualification_runner_source_sha256
        ),
    )
    if canonical_bytes(published) != payload:
        raise ProtectedCapacityError(
            "published protected-capacity marker differs from validated evidence"
        )
    return {
        "status": "complete",
        "apply": True,
        "marker_path": str(marker_path),
        "marker": published,
    }


# Stable, descriptive API aliases for callers that do not use the CLI.
publish_protected_capacity = attest
verify_protected_capacity_marker = verify_marker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    attest_parser = subparsers.add_parser(
        "attest",
        help="validate sealed evidence and optionally publish marker-last",
    )
    attest_parser.add_argument("--recovery-root", required=True, type=Path)
    attest_parser.add_argument(
        "--scheduler-evidence",
        required=True,
        type=Path,
    )
    attest_parser.add_argument("--canary-evidence", required=True, type=Path)
    attest_parser.add_argument(
        "--release-git-commit",
        required=True,
        help="exact 40-hex commit peeled from the release annotated tag",
    )
    attest_parser.add_argument(
        "--release-tag-object",
        required=True,
        help="exact 40-hex annotated-tag object ID",
    )
    attest_parser.add_argument(
        "--source-tree-sha256",
        required=True,
        help="exact schema5_control.sha256_tree digest of the release worktree",
    )
    attest_parser.add_argument(
        "--dispatcher-source-sha256",
        required=True,
        help="exact digest of tagged slurm/dispatch_sweeps.py",
    )
    attest_parser.add_argument(
        "--qualification-runner-source-sha256",
        required=True,
        help=(
            "exact digest of tagged "
            "scripts/run_schema5_throughput_qualification.py"
        ),
    )
    attest_parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically publish the read-only marker; default is dry-run",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="verify an already-published marker without scheduler access",
    )
    verify_parser.add_argument("--recovery-root", required=True, type=Path)
    verify_parser.add_argument("--release-git-commit", required=True)
    verify_parser.add_argument("--release-tag-object", required=True)
    verify_parser.add_argument("--source-tree-sha256", required=True)
    verify_parser.add_argument("--dispatcher-source-sha256", required=True)
    verify_parser.add_argument(
        "--qualification-runner-source-sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "attest":
            report = attest(
                recovery_root=args.recovery_root,
                scheduler_evidence_path=args.scheduler_evidence,
                canary_evidence_path=args.canary_evidence,
                expected_release_git_commit=args.release_git_commit,
                expected_release_tag_object=args.release_tag_object,
                expected_source_tree_sha256=args.source_tree_sha256,
                expected_dispatcher_source_sha256=(
                    args.dispatcher_source_sha256
                ),
                expected_qualification_runner_source_sha256=(
                    args.qualification_runner_source_sha256
                ),
                apply=args.apply,
            )
        else:
            marker = verify_marker(
                args.recovery_root,
                expected_release_git_commit=args.release_git_commit,
                expected_release_tag_object=args.release_tag_object,
                expected_source_tree_sha256=args.source_tree_sha256,
                expected_dispatcher_source_sha256=(
                    args.dispatcher_source_sha256
                ),
                expected_qualification_runner_source_sha256=(
                    args.qualification_runner_source_sha256
                ),
            )
            report = {
                "status": "verified",
                "marker_path": str(
                    _lexical_absolute(args.recovery_root) / MARKER_FILENAME
                ),
                "marker": marker,
            }
    except ProtectedCapacityError as exc:
        print(f"protected-capacity attestation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
