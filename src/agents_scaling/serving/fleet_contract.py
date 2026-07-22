"""Frozen schema-5 serving-fleet identity and placement contract.

The serving supervisor, launcher, registry repair path, and readiness audit all consume
this one checksummed document.  The contract names every logical replica and freezes its
Slurm placement/resources; production cannot synthesize an equivalent-looking fleet
from an ad hoc replica-count string.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from agents_scaling.serving.model_contracts import FrozenModelContracts
from agents_scaling.serving.profiles import SERVING_PROFILES


FLEET_ID = "schema5-v1"
RELEASE_ID = "sweep-recovery-schema5-v1"
EXPECTED_COUNTS = {
    "0.6B": 2,
    "1.7B": 2,
    "4B": 2,
    "8B": 3,
    "14B": 2,
    "32B": 4,
    "0.6B-long": 1,
    "1.7B-long": 1,
    "4B-long": 1,
    "8B-long": 1,
    "14B-long": 1,
    "32B-long": 2,
}
_PROFILE_SLUGS = {
    "0.6B": "0p6b",
    "1.7B": "1p7b",
    "4B": "4b",
    "8B": "8b",
    "14B": "14b",
    "32B": "32b",
    "0.6B-long": "0p6b",
    "1.7B-long": "1p7b",
    "4B-long": "4b",
    "8B-long": "8b",
    "14B-long": "14b",
    "32B-long": "32b",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FleetContractError(RuntimeError):
    """The canonical fleet contract is absent, drifted, or internally inconsistent."""


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class FleetReplica:
    serving_profile: str
    model_size: str
    pool_id: str
    replica_id: str
    replica_index: int
    scheduler_job_name: str
    partition: str
    gpu_type: str
    time_limit: str
    cpus_per_task: int
    memory: str
    gpus_per_replica: int


def expected_replica_id(profile_name: str, replica_index: int) -> str:
    """Return the one canonical logical identity for a frozen fleet replica."""

    if (
        profile_name not in _PROFILE_SLUGS
        or type(replica_index) is not int
        or replica_index < 0
    ):
        raise FleetContractError(
            f"cannot construct replica identity for {profile_name!r} r{replica_index!r}"
        )
    layout = "long" if profile_name.endswith("-long") else "standard"
    return (
        f"{FLEET_ID}--{_PROFILE_SLUGS[profile_name]}--{layout}--"
        f"r{replica_index:02d}"
    )


def expected_scheduler_job_name(profile_name: str, replica_index: int) -> str:
    """Return the exact pool/profile/replica-scoped Slurm job name."""

    if (
        profile_name not in _PROFILE_SLUGS
        or type(replica_index) is not int
        or replica_index < 0
    ):
        raise FleetContractError(
            f"cannot construct scheduler identity for {profile_name!r} r{replica_index!r}"
        )
    layout = "l" if profile_name.endswith("-long") else "s"
    return f"asys-s5-serve-{_PROFILE_SLUGS[profile_name]}-{layout}-r{replica_index:02d}"


@dataclass(frozen=True)
class FrozenFleetContract:
    path: Path
    sha256: str
    release_id: str
    fleet_id: str
    root_suffix: str
    replicas: tuple[FleetReplica, ...]
    by_profile: Mapping[str, tuple[FleetReplica, ...]]

    def for_replica(self, profile_name: str, replica_index: int) -> FleetReplica:
        try:
            replicas = self.by_profile[profile_name]
        except KeyError as exc:
            raise FleetContractError(
                f"profile {profile_name!r} is outside fleet {self.sha256}"
            ) from exc
        matches = [item for item in replicas if item.replica_index == replica_index]
        if len(matches) != 1:
            raise FleetContractError(
                f"fleet does not uniquely define {profile_name} replica {replica_index}"
            )
        return matches[0]

    def verify_pool_root(self, run_root: str | Path) -> Path:
        root = Path(run_root).expanduser().resolve()
        suffix = Path(self.root_suffix)
        if (
            len(root.parts) < len(suffix.parts)
            or root.parts[-len(suffix.parts) :] != suffix.parts
        ):
            raise FleetContractError(
                f"canonical fleet root must end in {self.root_suffix!r}, found {root}"
            )
        return root


def default_fleet_contract_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "schema5_fleet.v1.json"


def _checksum(path: Path) -> str:
    checksum_path = path.with_suffix(".sha256")
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise FleetContractError(f"missing fleet checksum: {checksum_path}")
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        len(fields) != 2
        or _SHA256_RE.fullmatch(fields[0]) is None
        or fields[1] != path.name
        or fields[0] != digest
    ):
        raise FleetContractError(f"fleet checksum does not bind {path}")
    return digest


def load_fleet_contract(
    path: str | Path | None,
    *,
    model_contracts: FrozenModelContracts,
    expected_sha256: str | None = None,
) -> FrozenFleetContract:
    fleet_path = (
        default_fleet_contract_path() if path is None else Path(path).expanduser().resolve()
    )
    if fleet_path.is_symlink() or not fleet_path.is_file():
        raise FleetContractError(f"missing regular fleet contract: {fleet_path}")
    digest = _checksum(fleet_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FleetContractError(
            f"fleet contract hash drift: expected {expected_sha256}, observed {digest}"
        )
    try:
        payload = json.loads(
            fleet_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FleetContractError(f"invalid fleet JSON: {exc}") from exc
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
    if not isinstance(payload, dict) or set(payload) != required_root:
        raise FleetContractError("fleet contract has the wrong root fields")
    if (
        payload["schema_version"] != 1
        or payload["fleet_id"] != FLEET_ID
        or payload["release_id"] != RELEASE_ID
        or payload["model_contract_sha256"] != model_contracts.sha256
        or payload["logical_replica_count"] != 22
        or payload["allocated_gpu_count"] != 24
    ):
        raise FleetContractError("fleet release/model/cardinality identity drifted")
    if payload["offline_environment"] != {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }:
        raise FleetContractError("fleet must require fully offline model access")
    pool = payload["server_pool"]
    expected_pool = {
        "pool_id": FLEET_ID,
        "root_suffix": "server_pools/schema5-v1",
        "scheduler_job_name_prefix": "asys-s5-serve-",
        "registration_scope": "exact_pool_root",
        "run_root_argument_required": True,
        "adoption_requires_spooled_run_root_match": True,
        "no_requeue": True,
    }
    if pool != expected_pool:
        raise FleetContractError("fleet pool isolation policy drifted")

    profiles = payload["profiles"]
    if not isinstance(profiles, list):
        raise FleetContractError("fleet profiles must be an array")
    by_profile: dict[str, tuple[FleetReplica, ...]] = {}
    all_ids: set[str] = set()
    all_names: set[str] = set()
    all_replicas: list[FleetReplica] = []
    for raw_profile in profiles:
        if not isinstance(raw_profile, dict):
            raise FleetContractError("fleet profile must be an object")
        profile_name = raw_profile.get("serving_profile")
        if profile_name not in SERVING_PROFILES or profile_name in by_profile:
            raise FleetContractError(f"invalid or duplicate fleet profile {profile_name!r}")
        profile = SERVING_PROFILES[profile_name]
        identity = model_contracts.for_size(profile.model_size)
        exact_profile = {
            "serving_profile": profile.name,
            "model_size": profile.model_size,
            "hf_id": profile.hf_id,
            "model_revision": identity.model_revision,
            "tokenizer_id": identity.tokenizer_id,
            "tokenizer_revision": identity.tokenizer_revision,
            "served_model_name": profile.served_model_name,
            "tensor_parallel_size": profile.tp_size,
            "effective_context_limit": profile.max_model_len,
            "gpus_per_replica": profile.tp_size,
        }
        if any(raw_profile.get(field) != value for field, value in exact_profile.items()):
            raise FleetContractError(f"fleet profile identity drifted for {profile.name}")
        if set(raw_profile) != set(exact_profile) | {"replicas"}:
            raise FleetContractError(f"fleet profile fields drifted for {profile.name}")
        raw_replicas = raw_profile["replicas"]
        if not isinstance(raw_replicas, list) or len(raw_replicas) != EXPECTED_COUNTS[profile.name]:
            raise FleetContractError(f"wrong replica count for {profile.name}")
        parsed: list[FleetReplica] = []
        required_replica = {
            "pool_id",
            "replica_id",
            "replica_index",
            "scheduler_job_name",
            "partition",
            "gpu_type",
            "time_limit",
            "cpus_per_task",
            "memory",
        }
        for raw in raw_replicas:
            if not isinstance(raw, dict) or set(raw) != required_replica:
                raise FleetContractError(f"replica fields drifted for {profile.name}")
            index = raw["replica_index"]
            replica = FleetReplica(
                serving_profile=profile.name,
                model_size=profile.model_size,
                gpus_per_replica=profile.tp_size,
                **raw,
            )
            expected_partition = (
                "ou_bcs_normal"
                if profile.name == "14B-long"
                or (profile.name == "32B-long" and index == 0)
                else "ou_bcs_low"
            )
            if (
                raw["pool_id"] != FLEET_ID
                or type(index) is not int
                or index < 0
                or raw["replica_id"] != expected_replica_id(profile.name, index)
                or raw["scheduler_job_name"]
                != expected_scheduler_job_name(profile.name, index)
                or raw["partition"] != expected_partition
                or raw["gpu_type"] != "a100"
                or raw["time_limit"] != "1-00:00:00"
                or raw["cpus_per_task"] != profile.tp_size * 8
                or raw["memory"] != f"{profile.tp_size * 120}G"
                or raw["replica_id"] in all_ids
                or raw["scheduler_job_name"] in all_names
            ):
                raise FleetContractError(f"invalid replica placement for {profile.name}")
            all_ids.add(raw["replica_id"])
            all_names.add(raw["scheduler_job_name"])
            parsed.append(replica)
        if {item.replica_index for item in parsed} != set(range(len(parsed))):
            raise FleetContractError(f"replica indices are not contiguous for {profile.name}")
        by_profile[profile.name] = tuple(parsed)
        all_replicas.extend(parsed)
    if set(by_profile) != set(EXPECTED_COUNTS):
        raise FleetContractError("fleet profile set is incomplete")
    if len(all_replicas) != 22 or sum(item.gpus_per_replica for item in all_replicas) != 24:
        raise FleetContractError("fleet aggregate cardinality drifted")
    return FrozenFleetContract(
        path=fleet_path,
        sha256=digest,
        release_id=RELEASE_ID,
        fleet_id=FLEET_ID,
        root_suffix=pool["root_suffix"],
        replicas=tuple(all_replicas),
        by_profile=MappingProxyType(by_profile),
    )
