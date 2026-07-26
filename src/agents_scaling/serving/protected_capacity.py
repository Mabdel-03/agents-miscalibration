"""Immutable protected-placement authority for schema-5 scientific jobs.

The production fleet and cell dispatcher are not allowed to infer safety from a
partition name, from ``--no-requeue``, or from an old readiness report.  This module
loads the marker-last ``PROTECTED_CAPACITY_COMPLETE.json`` contract, binds it to the
exact release, and rechecks every partition and QOS immediately before an ``sbatch``
boundary.

Controllers are deliberately outside this contract: they perform no scientific
requests and have their own successor/watchdog recovery policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
from typing import Any

from agents_scaling.serving import scheduler_safety


SCHEMA_VERSION = 2
PROTOCOL = "schema5-v1.2-r2-protected-capacity-v2"
LIVE_CLIENT_CAPACITY_PROTOCOL = (
    "schema5-v1.2-r2-live-protected-client-capacity-v1"
)
CAPACITY_SOURCE = (
    "sealed_protected_canary+partition_inventory+association"
)
RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r2"
CHAIN_NAMESPACE = "schema5-v1.2-r2"
MARKER_FILENAME = "PROTECTED_CAPACITY_COMPLETE.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PLACEMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MARKER_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "release_id",
    "release_tag",
    "release_git_commit",
    "release_tag_object",
    "chain_namespace",
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
    "scheduler_max_submit_jobs",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "fleet_contract_sha256",
    "active_fleet_topology_sha256",
    "scientific_server_preempt_mode",
    "scientific_client_preempt_mode",
    "squeue_complete",
    "sacct_complete",
    "scheduler_evidence_id",
    "scheduler_evidence_sha256",
    "canary_id",
    "canary_evidence_sha256",
    "scientific_server_placements",
    "scientific_client_placements",
    "marker_id",
}
_SERVER_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "active_serving_gpus",
    "warm_headroom_gpus",
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


class ProtectedCapacityError(RuntimeError):
    """Protected placement is absent, drifted, or no longer live."""


@dataclass(frozen=True)
class ProtectedPlacement:
    partition: str
    qos: str
    capacity: Mapping[str, int]


@dataclass(frozen=True)
class ProtectedCapacityContract:
    path: Path
    sha256: str
    marker_id: str
    release_git_commit: str
    release_tag_object: str
    scheduler_evidence_id: str
    scheduler_evidence_sha256: str
    canary_id: str
    canary_evidence_sha256: str
    fleet_contract_sha256: str
    active_fleet_topology_sha256: str
    preempt_type: str
    capacity_source: str
    scheduler_cluster: str
    scheduler_account: str
    scheduler_user: str
    scheduler_max_submit_jobs: int
    partition_cpus: int
    partition_memory_mib: int
    partition_gpus: int
    server_placements: tuple[ProtectedPlacement, ...]
    client_placements: tuple[ProtectedPlacement, ...]

    def placement(self, *, role: str, partition: str, qos: str) -> ProtectedPlacement:
        rows = (
            self.server_placements
            if role == "server"
            else self.client_placements
            if role == "client"
            else ()
        )
        matches = [
            row
            for row in rows
            if row.partition == partition and row.qos == qos
        ]
        if len(matches) != 1:
            raise ProtectedCapacityError(
                f"scientific {role} placement {partition}/{qos} is not explicitly "
                f"authorized by protected-capacity marker {self.marker_id}"
            )
        return matches[0]


def canonical_bytes(value: Any) -> bytes:
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


def _identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(value))).hexdigest()


def canonical_marker_path(results_root: str | Path) -> Path:
    """Return the one protected-capacity marker location for a result hierarchy."""

    return (
        Path(results_root).expanduser().resolve()
        / "recovery"
        / "schema5-v1"
        / MARKER_FILENAME
    )


def _read_sealed_json(path: Path) -> tuple[Path, bytes, dict[str, Any]]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtectedCapacityError(
            f"protected-capacity marker is missing or unsafe: {lexical}: {exc}"
        ) from exc
    if resolved != lexical or lexical.is_symlink() or not lexical.is_file():
        raise ProtectedCapacityError(
            f"protected-capacity marker must be a non-symlink regular file: {lexical}"
        )
    descriptor_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        descriptor_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, descriptor_flags)
    except OSError as exc:
        raise ProtectedCapacityError(
            f"cannot open protected-capacity marker {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) & 0o222
        or before.st_nlink != 1
        or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker must be a stable, read-only, single-link file"
        )
    raw = b"".join(chunks)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtectedCapacityError(
                    f"protected-capacity marker duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ProtectedCapacityError(
            f"protected-capacity marker contains {token!r}"
        )

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except ProtectedCapacityError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"invalid protected-capacity marker JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtectedCapacityError(
            "protected-capacity marker must contain one JSON object"
        )
    current = lexical.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise ProtectedCapacityError(
            "protected-capacity marker was replaced while being read"
        )
    return lexical, raw, payload


def _positive_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ProtectedCapacityError(
            f"protected-capacity {field} must be an integer >= {minimum}"
        )
    return value


def _placements(
    value: Any,
    *,
    role: str,
    preempt_type: str,
) -> tuple[ProtectedPlacement, ...]:
    expected_fields = _SERVER_FIELDS if role == "server" else _CLIENT_FIELDS
    capacity_fields = (
        ("active_serving_gpus", "warm_headroom_gpus")
        if role == "server"
        else ("slots", "cpus", "memory_mib", "reserve_jobs", "submit_headroom")
    )
    if not isinstance(value, list) or not value:
        raise ProtectedCapacityError(
            f"protected-capacity marker has no scientific {role} placements"
        )
    parsed: list[ProtectedPlacement] = []
    seen: set[tuple[str, str]] = set()
    previous_sort_key: tuple[str, str, bytes] | None = None
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ProtectedCapacityError(
                f"scientific {role} placement {index} fields drifted"
            )
        partition, qos = row.get("partition"), row.get("qos")
        if (
            not isinstance(partition, str)
            or _PLACEMENT_RE.fullmatch(partition) is None
            or not isinstance(qos, str)
            or _PLACEMENT_RE.fullmatch(qos) is None
            or row.get("partition_preempt_mode") != "OFF"
            or (
                row.get("qos_preempt_mode") != "OFF"
                and not (
                    preempt_type == "preempt/partition_prio"
                    and row.get("qos_preempt_mode") == "cluster"
                )
            )
        ):
            raise ProtectedCapacityError(
                f"scientific {role} placement {index} is not explicitly "
                "effectively nonpreemptible under the bound PreemptType"
            )
        identity = (partition, qos)
        if identity in seen:
            raise ProtectedCapacityError(
                f"duplicate scientific {role} placement {partition}/{qos}"
            )
        seen.add(identity)
        sort_key = (partition, qos, canonical_bytes(row))
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ProtectedCapacityError(
                f"scientific {role} placements are not in canonical order"
            )
        previous_sort_key = sort_key
        parsed.append(
            ProtectedPlacement(
                partition=partition,
                qos=qos,
                capacity={
                    field: _positive_integer(
                        row.get(field),
                        field=f"{role} placement {partition}/{qos} {field}",
                    )
                    for field in capacity_fields
                },
            )
        )
    return tuple(parsed)


def load_contract(
    path: str | Path,
    *,
    expected_release_git_commit: str,
    expected_release_tag_object: str | None = None,
    expected_marker_id: str | None = None,
    expected_sha256: str | None = None,
) -> ProtectedCapacityContract:
    """Load and verify the exact sealed protected-capacity authority."""

    marker_path, raw, payload = _read_sealed_json(Path(path))
    if set(payload) != _MARKER_FIELDS:
        raise ProtectedCapacityError(
            "protected-capacity marker fields differ from the closed schema"
        )
    identity = dict(payload)
    marker_id = identity.pop("marker_id", None)
    digest = hashlib.sha256(raw).hexdigest()
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("passed") is not True
        or payload.get("release_id") != RELEASE_ID
        or payload.get("release_tag") != RELEASE_TAG
        or payload.get("chain_namespace") != CHAIN_NAMESPACE
        or not isinstance(expected_release_git_commit, str)
        or _GIT_OBJECT_RE.fullmatch(expected_release_git_commit) is None
        or payload.get("release_git_commit") != expected_release_git_commit
        or (
            expected_release_tag_object is not None
            and payload.get("release_tag_object") != expected_release_tag_object
        )
        or _GIT_OBJECT_RE.fullmatch(str(payload.get("release_tag_object", "")))
        is None
        or not isinstance(marker_id, str)
        or _SHA256_RE.fullmatch(marker_id) is None
        or marker_id != _identity_sha256(identity)
        or (
            expected_marker_id is not None and marker_id != expected_marker_id
        )
        or (expected_sha256 is not None and digest != expected_sha256)
        or payload.get("scientific_server_preempt_mode") != "OFF"
        or payload.get("scientific_client_preempt_mode") != "OFF"
        or payload.get("capacity_source") != CAPACITY_SOURCE
        or any(
            not isinstance(payload.get(field), str)
            or _PLACEMENT_RE.fullmatch(str(payload.get(field))) is None
            for field in (
                "scheduler_cluster",
                "scheduler_account",
                "scheduler_user",
            )
        )
        or any(
            not isinstance(payload.get(field), int)
            or isinstance(payload.get(field), bool)
            or int(payload[field]) < minimum
            for field, minimum in (
                ("scheduler_max_submit_jobs", 448),
                ("partition_cpus", 384),
                ("partition_memory_mib", 384 * 4096),
                ("partition_gpus", 0),
            )
        )
        or payload.get("squeue_complete") is not True
        or payload.get("sacct_complete") is not True
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker identity/release/seal is invalid"
        )
    for field in (
        "scheduler_evidence_id",
        "scheduler_evidence_sha256",
        "canary_id",
        "canary_evidence_sha256",
        "fleet_contract_sha256",
        "active_fleet_topology_sha256",
    ):
        if _SHA256_RE.fullmatch(str(payload.get(field, ""))) is None:
            raise ProtectedCapacityError(
                f"protected-capacity marker {field} is not SHA-256"
            )
    preempt_type = payload.get("preempt_type")
    if preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
        raise ProtectedCapacityError(
            "protected-capacity marker has unsupported PreemptType"
        )
    server_placements = _placements(
        payload["scientific_server_placements"],
        role="server",
        preempt_type=str(preempt_type),
    )
    client_placements = _placements(
        payload["scientific_client_placements"],
        role="client",
        preempt_type=str(preempt_type),
    )
    if (
        sum(row.capacity["active_serving_gpus"] for row in server_placements)
        != _positive_integer(payload.get("active_gpus"), field="active_gpus", minimum=24)
        or sum(row.capacity["warm_headroom_gpus"] for row in server_placements)
        != _positive_integer(
            payload.get("warm_headroom_gpus"),
            field="warm_headroom_gpus",
            minimum=4,
        )
        or sum(row.capacity["slots"] for row in client_placements)
        != _positive_integer(payload.get("cell_ceiling"), field="cell_ceiling", minimum=384)
        or sum(row.capacity["cpus"] for row in client_placements)
        != _positive_integer(payload.get("cpu"), field="cpu", minimum=384)
        or sum(row.capacity["memory_mib"] for row in client_placements)
        != _positive_integer(
            payload.get("memory_mib"), field="memory_mib", minimum=384 * 4096
        )
        or sum(row.capacity["reserve_jobs"] for row in client_placements)
        != _positive_integer(payload.get("reserve_jobs"), field="reserve_jobs", minimum=64)
        or sum(row.capacity["submit_headroom"] for row in client_placements)
        != _positive_integer(
            payload.get("submit_headroom"), field="submit_headroom", minimum=448
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity placement totals differ from the marker envelope"
        )
    return ProtectedCapacityContract(
        path=marker_path,
        sha256=digest,
        marker_id=marker_id,
        release_git_commit=expected_release_git_commit,
        release_tag_object=str(payload["release_tag_object"]),
        scheduler_evidence_id=str(payload["scheduler_evidence_id"]),
        scheduler_evidence_sha256=str(payload["scheduler_evidence_sha256"]),
        canary_id=str(payload["canary_id"]),
        canary_evidence_sha256=str(payload["canary_evidence_sha256"]),
        fleet_contract_sha256=str(payload["fleet_contract_sha256"]),
        active_fleet_topology_sha256=str(
            payload["active_fleet_topology_sha256"]
        ),
        preempt_type=str(preempt_type),
        capacity_source=str(payload["capacity_source"]),
        scheduler_cluster=str(payload["scheduler_cluster"]),
        scheduler_account=str(payload["scheduler_account"]),
        scheduler_user=str(payload["scheduler_user"]),
        scheduler_max_submit_jobs=int(
            payload["scheduler_max_submit_jobs"]
        ),
        partition_cpus=int(payload["partition_cpus"]),
        partition_memory_mib=int(payload["partition_memory_mib"]),
        partition_gpus=int(payload["partition_gpus"]),
        server_placements=server_placements,
        client_placements=client_placements,
    )


def authorize_fleet(
    fleet: Any,
    contract: ProtectedCapacityContract,
) -> None:
    """Require explicit QOS and marker coverage for every scientific replica."""

    replicas = getattr(fleet, "replicas", None)
    if not isinstance(replicas, tuple) or not replicas:
        raise ProtectedCapacityError("frozen serving fleet is missing replicas")
    fleet_sha256 = getattr(fleet, "sha256", None)
    if len(replicas) == 22 and fleet_sha256 != contract.fleet_contract_sha256:
        raise ProtectedCapacityError(
            "protected-capacity marker is not bound to the exact frozen fleet"
        )
    active_by_placement: dict[tuple[str, str], int] = {}
    for replica in replicas:
        partition = getattr(replica, "partition", None)
        qos = getattr(replica, "qos", None)
        if not isinstance(partition, str) or not isinstance(qos, str) or not qos:
            raise ProtectedCapacityError(
                f"scientific server {getattr(replica, 'replica_id', '<unknown>')} "
                "has no explicit partition/QOS placement"
            )
        contract.placement(role="server", partition=partition, qos=qos)
        active_by_placement[(partition, qos)] = (
            active_by_placement.get((partition, qos), 0)
            + int(getattr(replica, "gpus_per_replica", 0))
        )
    for identity, required_gpus in active_by_placement.items():
        placement = contract.placement(
            role="server", partition=identity[0], qos=identity[1]
        )
        authorized_gpus = (
            placement.capacity["active_serving_gpus"]
            + placement.capacity["warm_headroom_gpus"]
        )
        if authorized_gpus < required_gpus:
            raise ProtectedCapacityError(
                f"protected placement {identity[0]}/{identity[1]} authorizes "
                f"{authorized_gpus} active-or-warm GPUs, "
                f"but the fleet requires {required_gpus}"
            )


def authorize_client(
    contract: ProtectedCapacityContract,
    *,
    partition: str,
    qos: str,
    required_slots: int,
    required_reserve_jobs: int,
) -> ProtectedPlacement:
    placement = contract.placement(role="client", partition=partition, qos=qos)
    required_slots = _positive_integer(
        required_slots, field="required client slots", minimum=1
    )
    required_reserve_jobs = _positive_integer(
        required_reserve_jobs, field="required client reserve", minimum=0
    )
    capacity = placement.capacity
    if (
        capacity["slots"] < required_slots
        or capacity["cpus"] < required_slots
        or capacity["memory_mib"] < required_slots * 4096
        or capacity["reserve_jobs"] < required_reserve_jobs
        or capacity["submit_headroom"] < required_slots + required_reserve_jobs
    ):
        raise ProtectedCapacityError(
            f"protected client placement {partition}/{qos} cannot authorize "
            f"{required_slots} cells plus {required_reserve_jobs} reserved jobs"
        )
    return placement


def _invoke(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    argv: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    try:
        result = (
            subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if runner is None
            else runner(list(argv), timeout=30.0)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtectedCapacityError(
            f"protected placement live query failed for {list(argv)!r}: {exc}"
        ) from exc
    if (
        not isinstance(result, subprocess.CompletedProcess)
        or result.returncode != 0
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
    ):
        raise ProtectedCapacityError(
            f"protected placement live query failed for {list(argv)!r}: "
            f"rc={getattr(result, 'returncode', 'invalid')} "
            f"{str(getattr(result, 'stderr', ''))[:500]}"
        )
    return result


def _validate_live_qos(
    qos: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    preempt_type: str,
) -> None:
    argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        "format=Name,PreemptMode",
    ]
    raw = _invoke(runner, argv).stdout
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedCapacityError(
            f"live QOS query for {qos!r} did not return exactly one row"
        )
    fields = lines[0].split("|")
    qos_mode = fields[1] if len(fields) == 2 else ""
    effectively_safe = qos_mode.upper() == "OFF" or (
        preempt_type == "preempt/partition_prio"
        and qos_mode.lower() == "cluster"
    )
    if len(fields) != 2 or fields[0] != qos or not effectively_safe:
        raise ProtectedCapacityError(
            f"scientific QOS {qos!r} drifted from PreemptMode=OFF"
        )


def verify_live_placements(
    contract: ProtectedCapacityContract,
    *,
    role: str,
    placements: Sequence[tuple[str, str]],
    required_time_limits_seconds: Mapping[str, int],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Requery and reject partition/QOS drift immediately before ``sbatch``."""

    unique = sorted(set(placements))
    if not unique or len(unique) != len(placements):
        raise ProtectedCapacityError(
            f"scientific {role} live placement list is empty or ambiguous"
        )
    for partition, qos in unique:
        contract.placement(role=role, partition=partition, qos=qos)
    partitions = sorted({partition for partition, _qos in unique})
    if set(required_time_limits_seconds) != set(partitions):
        raise ProtectedCapacityError(
            "live protected-placement time requirements differ from partitions"
        )
    try:
        evidence = scheduler_safety.capture_scheduler_safety_evidence(
            partitions,
            runner=runner,
        )
        policy = scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=partitions,
            required_time_limits_seconds=required_time_limits_seconds,
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"scientific {role} partition drifted from protected contract: {exc}"
        ) from exc
    if policy["preemptible_partitions"]:
        raise ProtectedCapacityError(
            f"scientific {role} partition became preemptible"
        )
    if policy["preempt_type"] != contract.preempt_type:
        raise ProtectedCapacityError(
            "live scheduler PreemptType drifted from the sealed "
            "protected-capacity contract"
        )
    for qos in sorted({qos for _partition, qos in unique}):
        _validate_live_qos(
            qos,
            runner=runner,
            preempt_type=str(policy["preempt_type"]),
        )
    return {
        "marker_id": contract.marker_id,
        "marker_sha256": contract.sha256,
        "role": role,
        "placements": [
            {"partition": partition, "qos": qos} for partition, qos in unique
        ],
        "scheduler_evidence_id": policy["scheduler_evidence_id"],
        "scheduler_policy_id": policy["policy_id"],
    }


def _single_pipe_row(
    raw_output: str,
    *,
    width: int,
    description: str,
) -> list[str]:
    if not isinstance(raw_output, str) or "\x00" in raw_output:
        raise ProtectedCapacityError(f"{description} is not safe text")
    rows = [
        line.strip().split("|")
        for line in raw_output.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != width:
        raise ProtectedCapacityError(
            f"{description} must contain exactly one {width}-field row"
        )
    return rows[0]


def _optional_limit(value: str, *, description: str) -> int | None:
    if value == "":
        return None
    if not value.isdigit():
        raise ProtectedCapacityError(f"{description} is malformed")
    return int(value)


def _memory_mib(value: str, *, description: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]?)", value, re.IGNORECASE)
    if match is None:
        raise ProtectedCapacityError(f"{description} is malformed")
    amount = int(match.group(1))
    factor = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024**2,
        "P": 1024**3,
    }[match.group(2).upper()]
    converted = amount * factor
    if converted != int(converted) or converted < 0:
        raise ProtectedCapacityError(f"{description} is not integral MiB")
    return int(converted)


def _key_value_fields(raw_output: str, *, description: str) -> dict[str, str]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedCapacityError(
            f"{description} must contain one scheduler record"
        )
    try:
        tokens = shlex.split(lines[0], posix=True)
    except (TypeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"cannot parse {description}: {exc}"
        ) from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ProtectedCapacityError(
                f"{description} duplicates scheduler field {key!r}"
            )
        fields[key] = value.strip('"')
    return fields


def _partition_inventory(
    scheduler_evidence: Mapping[str, Any],
    *,
    partition: str,
    qos: str,
) -> dict[str, Any]:
    rows = scheduler_evidence.get("partitions")
    if not isinstance(rows, list):
        raise ProtectedCapacityError(
            "live protected scheduler evidence lacks partition rows"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("partition") == partition
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("raw_output"), str):
        raise ProtectedCapacityError(
            "live protected scheduler evidence has ambiguous partition inventory"
        )
    fields = _key_value_fields(
        str(matches[0]["raw_output"]),
        description=f"partition {partition} inventory",
    )
    tres: dict[str, str] = {}
    for item in fields.get("TRES", "").split(","):
        key, separator, value = item.rpartition("=")
        if not separator or not key or not value or key in tres:
            raise ProtectedCapacityError(
                f"partition {partition} TRES inventory is malformed"
            )
        tres[key] = value
    try:
        cpus = int(tres["cpu"])
        gpus = int(tres.get("gres/gpu", tres.get("gpu", "0")))
        memory_mib = _memory_mib(
            tres["mem"],
            description=f"partition {partition} memory",
        )
    except (KeyError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"partition {partition} inventory is incomplete"
        ) from exc
    allowed_qoses = set(
        filter(None, fields.get("AllowQos", "").split(","))
    )
    if "ALL" not in allowed_qoses and qos not in allowed_qoses:
        raise ProtectedCapacityError(
            f"partition {partition} no longer allows QOS {qos}"
        )
    return {
        "cpus": cpus,
        "memory_mib": memory_mib,
        "gpus": gpus,
        "allow_qos": sorted(allowed_qoses),
    }


def _source_record(
    *,
    argv: Sequence[str],
    raw_output: str,
) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(
            raw_output.encode("utf-8")
        ).hexdigest(),
    }


def _validate_source_record(
    value: Any,
    *,
    argv: Sequence[str],
    description: str,
) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"argv", "raw_output", "raw_output_sha256"}
        or value.get("argv") != list(argv)
        or not isinstance(value.get("raw_output"), str)
        or value.get("raw_output_sha256")
        != hashlib.sha256(
            str(value.get("raw_output", "")).encode("utf-8")
        ).hexdigest()
    ):
        raise ProtectedCapacityError(
            f"{description} source record is invalid"
        )
    return str(value["raw_output"])


def validate_live_client_capacity_evidence(
    contract: ProtectedCapacityContract,
    evidence: Mapping[str, Any],
    *,
    partition: str,
    qos: str,
    required_time_limit_seconds: int,
) -> dict[str, Any]:
    """Reparse one live recapture against the sealed simultaneous canary.

    Blank inherited QOS limits are never treated as infinity.  CPU and memory
    authority comes only from the sealed simultaneous client canary; the live
    partition inventory must remain byte-semantically equal to its attested
    configured totals, and the exact user/account association must retain the
    attested submit limit.
    """

    placement = contract.placement(
        role="client",
        partition=partition,
        qos=qos,
    )
    fields = {
        "schema_version",
        "protocol",
        "capacity_source",
        "marker_id",
        "marker_sha256",
        "scheduler_evidence_id",
        "canary_id",
        "partition",
        "qos",
        "scheduler_cluster",
        "scheduler_account",
        "scheduler_user",
        "scheduler_max_submit_jobs",
        "authorized_capacity",
        "captured_timestamp",
        "scheduler_policy_evidence",
        "qos_source",
        "association_source",
        "evidence_id",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != fields
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != LIVE_CLIENT_CAPACITY_PROTOCOL
        or evidence.get("capacity_source") != CAPACITY_SOURCE
        or evidence.get("marker_id") != contract.marker_id
        or evidence.get("marker_sha256") != contract.sha256
        or evidence.get("scheduler_evidence_id")
        != contract.scheduler_evidence_id
        or evidence.get("canary_id") != contract.canary_id
        or evidence.get("partition") != partition
        or evidence.get("qos") != qos
        or evidence.get("scheduler_cluster") != contract.scheduler_cluster
        or evidence.get("scheduler_account") != contract.scheduler_account
        or evidence.get("scheduler_user") != contract.scheduler_user
        or evidence.get("scheduler_max_submit_jobs")
        != contract.scheduler_max_submit_jobs
        or evidence.get("authorized_capacity")
        != dict(placement.capacity)
        or not isinstance(evidence.get("captured_timestamp"), (int, float))
        or isinstance(evidence.get("captured_timestamp"), bool)
        or not math.isfinite(float(evidence["captured_timestamp"]))
        or float(evidence["captured_timestamp"]) <= 0
        or evidence.get("evidence_id")
        != _identity_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_id"
            }
        )
    ):
        raise ProtectedCapacityError(
            "live protected client-capacity evidence envelope drifted"
        )
    scheduler_evidence = evidence.get("scheduler_policy_evidence")
    if not isinstance(scheduler_evidence, Mapping):
        raise ProtectedCapacityError(
            "live protected client-capacity scheduler evidence is absent"
        )
    if scheduler_evidence.get("captured_timestamp") != evidence["captured_timestamp"]:
        raise ProtectedCapacityError(
            "live protected client-capacity timestamps disagree"
        )
    try:
        policy = scheduler_safety.validate_scheduler_safety_evidence(
            scheduler_evidence,
            expected_partitions=[partition],
            required_time_limits_seconds={
                partition: required_time_limit_seconds
            },
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"live protected client partition policy drifted: {exc}"
        ) from exc
    if (
        policy["preemptible_partitions"]
        or policy["preempt_type"] != contract.preempt_type
    ):
        raise ProtectedCapacityError(
            "live protected client partition became preemptible"
        )
    inventory = _partition_inventory(
        scheduler_evidence,
        partition=partition,
        qos=qos,
    )
    if (
        inventory["cpus"] != contract.partition_cpus
        or inventory["memory_mib"] != contract.partition_memory_mib
        or inventory["gpus"] != contract.partition_gpus
        or inventory["cpus"] < placement.capacity["cpus"]
        or inventory["memory_mib"] < placement.capacity["memory_mib"]
    ):
        raise ProtectedCapacityError(
            "live partition inventory differs from the sealed protected canary "
            "authority"
        )

    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        "format=Name,PreemptMode,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxTRESPerUser",
    ]
    qos_raw = _validate_source_record(
        evidence.get("qos_source"),
        argv=qos_argv,
        description="live protected QOS",
    )
    qos_row = _single_pipe_row(
        qos_raw,
        width=5,
        description="live protected QOS",
    )
    qos_mode = qos_row[1]
    qos_safe = qos_mode.upper() == "OFF" or (
        contract.preempt_type == "preempt/partition_prio"
        and qos_mode.lower() == "cluster"
    )
    if qos_row[0] != qos or not qos_safe:
        raise ProtectedCapacityError(
            "live protected QOS identity/preemption mode drifted"
        )
    qos_max_jobs = _optional_limit(
        qos_row[2],
        description=f"{qos} MaxJobsPerUser",
    )
    qos_max_submit = _optional_limit(
        qos_row[3],
        description=f"{qos} MaxSubmitJobsPerUser",
    )
    if (
        qos_max_jobs is not None
        and qos_max_jobs < placement.capacity["slots"]
    ) or (
        qos_max_submit is not None
        and qos_max_submit < placement.capacity["submit_headroom"]
    ):
        raise ProtectedCapacityError(
            "live protected QOS explicit limits fell below the sealed canary"
        )
    if qos_row[4]:
        qos_tres: dict[str, str] = {}
        for item in qos_row[4].split(","):
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in qos_tres:
                raise ProtectedCapacityError(
                    "live protected QOS MaxTRESPerUser is malformed"
                )
            qos_tres[key] = value
        try:
            qos_cpu_limit = (
                int(qos_tres["cpu"]) if "cpu" in qos_tres else None
            )
        except ValueError as exc:
            raise ProtectedCapacityError(
                f"{qos} MaxTRESPerUser cpu is malformed"
            ) from exc
        if (
            qos_cpu_limit is not None
            and qos_cpu_limit < placement.capacity["cpus"]
        ) or (
            "mem" in qos_tres
            and _memory_mib(
                qos_tres["mem"],
                description=f"{qos} MaxTRESPerUser memory",
            )
            < placement.capacity["memory_mib"]
        ):
            raise ProtectedCapacityError(
                "live protected QOS MaxTRESPerUser fell below the sealed canary"
            )

    association_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={contract.scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    association_raw = _validate_source_record(
        evidence.get("association_source"),
        argv=association_argv,
        description="live protected association",
    )
    association_row = _single_pipe_row(
        association_raw,
        width=6,
        description="live protected association",
    )
    association_qoses = set(filter(None, association_row[3].split(",")))
    association_max_jobs = _optional_limit(
        association_row[4],
        description="association MaxJobs",
    )
    association_max_submit = _optional_limit(
        association_row[5],
        description="association MaxSubmitJobs",
    )
    if (
        association_row[:3]
        != [
            contract.scheduler_cluster,
            contract.scheduler_account,
            contract.scheduler_user,
        ]
        or qos not in association_qoses
        or association_max_submit != contract.scheduler_max_submit_jobs
        or association_max_submit < placement.capacity["submit_headroom"]
        or (
            association_max_jobs is not None
            and association_max_jobs < placement.capacity["slots"]
        )
    ):
        raise ProtectedCapacityError(
            "live protected user/account/QOS association drifted"
        )
    return {
        "evidence_id": evidence["evidence_id"],
        "capacity_source": CAPACITY_SOURCE,
        "marker_id": contract.marker_id,
        "partition": partition,
        "qos": qos,
        "scheduler_cluster": contract.scheduler_cluster,
        "scheduler_account": contract.scheduler_account,
        "scheduler_user": contract.scheduler_user,
        "scheduler_max_submit_jobs": contract.scheduler_max_submit_jobs,
        "cpu_limit": placement.capacity["cpus"],
        "memory_limit_mib": placement.capacity["memory_mib"],
        "max_submit_jobs": placement.capacity["submit_headroom"],
        "partition_cpus": inventory["cpus"],
        "partition_memory_mib": inventory["memory_mib"],
        "partition_gpus": inventory["gpus"],
        "scheduler_policy_id": policy["policy_id"],
        "scheduler_policy_contract_id": policy["policy_contract_id"],
    }


def capture_live_client_capacity(
    contract: ProtectedCapacityContract,
    *,
    partition: str,
    qos: str,
    required_time_limit_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Capture and validate live inherited-QOS truth against the sealed canary."""

    placement = contract.placement(
        role="client",
        partition=partition,
        qos=qos,
    )
    try:
        scheduler_evidence = (
            scheduler_safety.capture_scheduler_safety_evidence(
                [partition],
                runner=runner,
                captured_timestamp=captured_timestamp,
            )
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"cannot capture protected client partition policy: {exc}"
        ) from exc
    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        "format=Name,PreemptMode,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxTRESPerUser",
    ]
    association_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={contract.scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    qos_raw = _invoke(runner, qos_argv).stdout
    association_raw = _invoke(runner, association_argv).stdout
    timestamp = (
        time.time()
        if captured_timestamp is None
        else float(captured_timestamp)
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": LIVE_CLIENT_CAPACITY_PROTOCOL,
        "capacity_source": CAPACITY_SOURCE,
        "marker_id": contract.marker_id,
        "marker_sha256": contract.sha256,
        "scheduler_evidence_id": contract.scheduler_evidence_id,
        "canary_id": contract.canary_id,
        "partition": partition,
        "qos": qos,
        "scheduler_cluster": contract.scheduler_cluster,
        "scheduler_account": contract.scheduler_account,
        "scheduler_user": contract.scheduler_user,
        "scheduler_max_submit_jobs": (
            contract.scheduler_max_submit_jobs
        ),
        "authorized_capacity": dict(placement.capacity),
        "captured_timestamp": timestamp,
        "scheduler_policy_evidence": scheduler_evidence,
        "qos_source": _source_record(argv=qos_argv, raw_output=qos_raw),
        "association_source": _source_record(
            argv=association_argv,
            raw_output=association_raw,
        ),
    }
    evidence["evidence_id"] = _identity_sha256(evidence)
    validate_live_client_capacity_evidence(
        contract,
        evidence,
        partition=partition,
        qos=qos,
        required_time_limit_seconds=required_time_limit_seconds,
    )
    return evidence
