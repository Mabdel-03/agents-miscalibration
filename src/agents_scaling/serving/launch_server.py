"""Launch (and register) a vLLM server for one model size on SLURM.

Two roles, selected by flag:

* default — render ``slurm/serve_qwen.sbatch.tmpl`` for a model size and ``sbatch`` it.
  Picks a deterministic port from the model size so re-launches are stable.
* ``--register`` — run *inside* the serving job after vLLM starts: healthcheck the local
  server, then write its ``node:port`` to the registry so cell jobs can discover it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling import runtime_integrity
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving.fleet_contract import (
    FleetContractError,
    load_fleet_contract,
)
from agents_scaling.serving.model_contracts import load_model_contracts
from agents_scaling.serving.profiles import ServingProfile, get_serving_profile

REPO = Path(__file__).resolve().parents[3]  # .../agents_scaling
TEMPLATE = REPO / "slurm" / "serve_qwen.sbatch.tmpl"

# A100-80GB nodes; pi_tpoggio is the dedicated 7-day partition.
_PARTITION_DEFAULT = "pi_tpoggio"
_GPU_TYPE_DEFAULT = "a100"


@dataclass(frozen=True)
class _EnvironmentRuntimePins:
    prefix: Path
    python: Path
    python_version: str
    transformers_version: str
    tokenizers_version: str
    torch_version: str
    vllm_version: str
    cuda_version: str


def _environment_runtime_pins(
    *,
    role: str,
    prefix: str,
    manifest_path: str,
    expected_manifest_sha256: str,
    release_id: str,
) -> _EnvironmentRuntimePins:
    """Load exact runtime facts from a frozen, checksummed environment manifest."""

    resolved_prefix = Path(prefix).expanduser().resolve()
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    if not resolved_prefix.is_dir():
        raise ValueError(f"pinned {role} environment prefix is missing: {resolved_prefix}")
    if stat.S_IMODE(resolved_prefix.stat().st_mode) & 0o222:
        raise ValueError(f"pinned {role} environment prefix is not read-only")
    if resolved_manifest.is_symlink() or not resolved_manifest.is_file():
        raise ValueError(f"pinned {role} environment manifest is missing")
    raw = resolved_manifest.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"{role} environment manifest hash drift: expected "
            f"{expected_manifest_sha256}, observed {observed_sha256}"
        )
    try:
        payload: Any = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} environment manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} environment manifest must be an object")
    if (
        payload.get("role") != role
        or payload.get("release_id") != release_id
        or Path(str(payload.get("prefix", ""))).expanduser().resolve()
        != resolved_prefix
        or payload.get("sealed_read_only") is not True
        or payload.get("offline_environment")
        != {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    ):
        raise ValueError(f"{role} environment manifest identity is not production-safe")
    runtime = payload.get("runtime")
    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or not isinstance(packages, dict):
        raise ValueError(f"{role} environment manifest lacks runtime package pins")
    required_versions = {
        "python_version": runtime.get("python_version"),
        "transformers": packages.get("transformers"),
        "tokenizers": packages.get("tokenizers"),
    }
    if any(not isinstance(value, str) or not value for value in required_versions.values()):
        raise ValueError(f"{role} environment manifest has incomplete runtime pins")
    if role == "serving":
        for field, value in {
            "torch": packages.get("torch"),
            "vllm": packages.get("vllm"),
            "cuda_version": runtime.get("cuda_version"),
        }.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"serving environment manifest lacks {field}")
        if packages.get("vllm") != "0.21.0":
            raise ValueError("schema-5 serving requires vLLM 0.21.0")

    python = resolved_prefix / "bin" / "python"
    if not python.is_file():
        raise ValueError(f"pinned {role} Python is missing: {python}")
    if stat.S_IMODE(python.stat().st_mode) & 0o222:
        raise ValueError(f"pinned {role} Python remains writable")
    if role == "serving" and not (resolved_prefix / "bin" / "vllm").is_file():
        raise ValueError(f"pinned vLLM executable is missing under {resolved_prefix}")
    if role == "serving" and stat.S_IMODE(
        (resolved_prefix / "bin" / "vllm").stat().st_mode
    ) & 0o222:
        raise ValueError("pinned vLLM executable remains writable")
    return _EnvironmentRuntimePins(
        prefix=resolved_prefix,
        python=python,
        python_version=str(runtime["python_version"]),
        transformers_version=str(packages["transformers"]),
        tokenizers_version=str(packages["tokenizers"]),
        torch_version=str(packages.get("torch") or ""),
        vllm_version=str(packages.get("vllm") or ""),
        cuda_version=str(runtime.get("cuda_version") or ""),
    )


def _legacy_runtime_pins(role: str, prefix: str) -> _EnvironmentRuntimePins:
    """Render an explicit legacy prefix without claiming frozen package versions."""

    resolved_prefix = Path(prefix).expanduser().resolve()
    return _EnvironmentRuntimePins(
        prefix=resolved_prefix,
        python=resolved_prefix / "bin" / "python",
        python_version="",
        transformers_version="",
        tokenizers_version="",
        torch_version="",
        vllm_version="0.21.0" if role == "serving" else "",
        cuda_version="",
    )


def _port_for(profile_name: str, replica: int = 0) -> int:
    """Deterministic port from the serving profile + replica index.

    Base port in [8000, 8999] from the size; ``+ replica`` offsets each replica so two
    replicas of the SAME size landing on the SAME node don't bind the same port. The
    registry keys by ``host_port`` so distinct ports give distinct registry entries even
    co-located. (Replicas of different sizes already differ via crc32.)
    """
    return 8000 + (zlib.crc32(profile_name.encode()) % 1000) + replica


def _resolve_profile(model_size: str, profile_name: str | None) -> ServingProfile:
    """Resolve legacy model-only calls and explicit named-profile calls.

    Passing ``model_size="32B-long"`` remains useful to keepalive-style callers; it is
    interpreted as a profile key and resolves back to scientific model identity ``32B``.
    The explicit CLI form is ``--model-size 32B --profile 32B-long``.
    """
    profile = get_serving_profile(profile_name or model_size)
    if profile_name is not None and profile.model_size != model_size:
        raise ValueError(
            f"serving profile {profile.name!r} belongs to model "
            f"{profile.model_size!r}, not {model_size!r}"
        )
    return profile


def _production_release_resources(
    *,
    release_worktree: str | None,
    model_contract_path: str | None,
    fleet_contract_path: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve all repository resources from one explicit immutable worktree.

    The production harness is installed into an environment prefix and runs with
    ``python -I``.  Package ``__file__`` therefore points below ``site-packages`` and
    cannot be used to find repository-level configs or Slurm templates.  Require every
    path to agree with the canonical layout below the control-pinned release root.
    """

    if not release_worktree or not model_contract_path or not fleet_contract_path:
        raise ValueError(
            "schema-5 server rendering requires release worktree, model contract, "
            "and fleet contract paths"
        )
    raw_release = Path(release_worktree).expanduser()
    raw_model = Path(model_contract_path).expanduser()
    raw_fleet = Path(fleet_contract_path).expanduser()
    if not all(path.is_absolute() for path in (raw_release, raw_model, raw_fleet)):
        raise ValueError("schema-5 release resource paths must be absolute")
    release = raw_release.resolve()
    expected_model = release / "configs" / "model_contracts.v1.json"
    expected_fleet = release / "configs" / "schema5_fleet.v1.json"
    expected_template = release / "slurm" / "serve_qwen.sbatch.tmpl"
    if raw_release.is_symlink() or not release.is_dir():
        raise ValueError("schema-5 release worktree is missing or symlinked")
    for label, raw_path, expected_path in (
        ("model contract", raw_model, expected_model),
        ("fleet contract", raw_fleet, expected_fleet),
    ):
        if raw_path.resolve() != expected_path:
            raise ValueError(
                f"schema-5 {label} is outside the immutable release layout"
            )
        if raw_path.is_symlink() or not raw_path.is_file():
            raise ValueError(f"schema-5 {label} is missing or symlinked")
    if expected_template.is_symlink() or not expected_template.is_file():
        raise ValueError("schema-5 serving template is missing or symlinked")
    return release, expected_model, expected_fleet, expected_template


def render_sbatch(
    model_size: str,
    run_root: str,
    partition: str,
    gpu_type: str,
    time_limit: str,
    log_dir: str,
    replica: int = 0,
    serving_profile: str | None = None,
    release_id: str | None = None,
    environment_hash: str | None = None,
    model_contract_sha256: str | None = None,
    release_worktree: str | None = None,
    model_contract_path: str | None = None,
    harness_environment_prefix: str | None = None,
    serving_environment_prefix: str | None = None,
    harness_environment_manifest_path: str | None = None,
    serving_environment_manifest_path: str | None = None,
    harness_environment_hash: str | None = None,
    fleet_contract_path: str | None = None,
    fleet_contract_sha256: str | None = None,
    release_fleet_contract_sha256: str | None = None,
    capacity_generation: int | None = None,
    hf_home: str | None = None,
    runtime_attestation: str | None = None,
    runtime_attestation_sha256: str | None = None,
    runtime_integrity_lease: str | None = None,
    immutable_pins_sha256: str | None = None,
    rollout_generation: int | None = None,
    standby: bool = False,
    qos: str | None = None,
) -> str:
    profile = _resolve_profile(model_size, serving_profile)
    canonical_run_root = str(Path(run_root).expanduser().resolve())
    release_id = release_id or os.environ.get("ASYS_RELEASE_ID")
    environment_hash = environment_hash or os.environ.get(
        "ASYS_SERVING_ENVIRONMENT_SHA256"
    )
    if bool(release_id) != bool(environment_hash):
        raise ValueError(
            "serving release ID and environment hash must be supplied together"
        )
    if environment_hash is not None and (
        len(environment_hash) != 64
        or any(character not in "0123456789abcdef" for character in environment_hash)
    ):
        raise ValueError("serving environment hash must be lowercase SHA-256")
    model_contract_sha256 = model_contract_sha256 or os.environ.get(
        "ASYS_MODEL_CONTRACT_SHA256"
    )
    release_worktree = release_worktree or os.environ.get("ASYS_RELEASE_WORKTREE")
    model_contract_path = model_contract_path or os.environ.get(
        "ASYS_MODEL_CONTRACT"
    )
    harness_environment_prefix = harness_environment_prefix or os.environ.get(
        "ASYS_HARNESS_ENVIRONMENT_PREFIX"
    )
    serving_environment_prefix = serving_environment_prefix or os.environ.get(
        "ASYS_SERVING_ENVIRONMENT_PREFIX"
    )
    harness_environment_manifest_path = (
        harness_environment_manifest_path
        or os.environ.get("ASYS_HARNESS_ENVIRONMENT_MANIFEST")
    )
    serving_environment_manifest_path = (
        serving_environment_manifest_path
        or os.environ.get("ASYS_SERVING_ENVIRONMENT_MANIFEST")
    )
    harness_environment_hash = harness_environment_hash or os.environ.get(
        "ASYS_HARNESS_ENVIRONMENT_SHA256"
    )
    fleet_contract_path = fleet_contract_path or os.environ.get(
        "ASYS_FLEET_CONTRACT"
    )
    fleet_contract_sha256 = fleet_contract_sha256 or os.environ.get(
        "ASYS_FLEET_CONTRACT_SHA256"
    )
    release_fleet_contract_sha256 = (
        release_fleet_contract_sha256
        or os.environ.get("ASYS_RELEASE_FLEET_CONTRACT_SHA256")
    )
    if capacity_generation is None:
        raw_capacity_generation = os.environ.get("ASYS_CAPACITY_GENERATION")
        capacity_generation = (
            None
            if raw_capacity_generation is None
            else int(raw_capacity_generation)
        )
    hf_home = hf_home or os.environ.get("HF_HOME") or (
        "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"
    )
    runtime_attestation = runtime_attestation or os.environ.get(
        "ASYS_RUNTIME_ATTESTATION"
    )
    runtime_attestation_sha256 = runtime_attestation_sha256 or os.environ.get(
        "ASYS_RUNTIME_ATTESTATION_SHA256"
    )
    runtime_integrity_lease = runtime_integrity_lease or os.environ.get(
        "ASYS_RUNTIME_INTEGRITY_LEASE"
    )
    immutable_pins_sha256 = immutable_pins_sha256 or os.environ.get(
        "ASYS_IMMUTABLE_PINS_SHA256"
    )
    if rollout_generation is None:
        raw_generation = os.environ.get("ASYS_ROLLOUT_GENERATION")
        rollout_generation = None if raw_generation is None else int(raw_generation)
    production_fields = {
        "release_worktree": release_worktree,
        "model_contract_path": model_contract_path,
        "harness_environment_prefix": harness_environment_prefix,
        "serving_environment_prefix": serving_environment_prefix,
        "harness_environment_manifest_path": harness_environment_manifest_path,
        "serving_environment_manifest_path": serving_environment_manifest_path,
        "harness_environment_hash": harness_environment_hash,
        "fleet_contract_path": fleet_contract_path,
        "fleet_contract_sha256": fleet_contract_sha256,
        "release_fleet_contract_sha256": release_fleet_contract_sha256,
        "capacity_generation": capacity_generation,
        "runtime_attestation": runtime_attestation,
        "runtime_attestation_sha256": runtime_attestation_sha256,
        "runtime_integrity_lease": runtime_integrity_lease,
        "immutable_pins_sha256": immutable_pins_sha256,
        "rollout_generation": rollout_generation,
    }
    if release_id:
        missing = [name for name, value in production_fields.items() if not value]
        if missing:
            raise ValueError(
                "schema-5 server rendering lacks immutable environment pins: "
                + ", ".join(missing)
            )
        (
            resolved_release_worktree,
            resolved_model_contract,
            resolved_fleet_contract,
            template_path,
        ) = _production_release_resources(
            release_worktree=release_worktree,
            model_contract_path=model_contract_path,
            fleet_contract_path=fleet_contract_path,
        )
        harness_runtime = _environment_runtime_pins(
            role="harness",
            prefix=str(harness_environment_prefix),
            manifest_path=str(harness_environment_manifest_path),
            expected_manifest_sha256=str(harness_environment_hash),
            release_id=release_id,
        )
        serving_runtime = _environment_runtime_pins(
            role="serving",
            prefix=str(serving_environment_prefix),
            manifest_path=str(serving_environment_manifest_path),
            expected_manifest_sha256=str(environment_hash),
            release_id=release_id,
        )
        if (
            not isinstance(rollout_generation, int)
            or isinstance(rollout_generation, bool)
            or rollout_generation < 1
        ):
            raise ValueError("schema-5 server rendering requires a positive rollout generation")
        if (
            not isinstance(capacity_generation, int)
            or isinstance(capacity_generation, bool)
            or capacity_generation < 1
            or not isinstance(release_fleet_contract_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", release_fleet_contract_sha256) is None
        ):
            raise ValueError(
                "schema-5 server rendering requires exact release-fleet and "
                "capacity-generation lineage"
            )
        try:
            runtime_integrity.verify_generation_lease(
                lease_path=Path(str(runtime_integrity_lease)),
                attestation_path=Path(str(runtime_attestation)),
                attestation_sha256=str(runtime_attestation_sha256),
                generation=rollout_generation,
                release_id=release_id,
                immutable_pins_sha256=str(immutable_pins_sha256),
                expected_environment_hashes={
                    "harness": str(harness_environment_hash),
                    "serving": str(environment_hash),
                },
                expected_prefixes={
                    "harness": str(harness_environment_prefix),
                    "serving": str(serving_environment_prefix),
                },
            )
        except runtime_integrity.RuntimeIntegrityError as exc:
            raise ValueError(f"schema-5 runtime integrity failed: {exc}") from exc
    else:
        resolved_release_worktree = None
        resolved_model_contract = (
            None
            if model_contract_path is None
            else Path(model_contract_path).expanduser().resolve()
        )
        resolved_fleet_contract = (
            None
            if fleet_contract_path is None
            else Path(fleet_contract_path).expanduser().resolve()
        )
        template_path = TEMPLATE
        harness_runtime = _legacy_runtime_pins(
            "harness",
            harness_environment_prefix
            or os.environ.get("ASYS_HARNESS_ENV")
            or sys.prefix,
        )
        serving_runtime = _legacy_runtime_pins(
            "serving",
            serving_environment_prefix
            or os.environ.get("ASYS_SERVE_ENV")
            or "/home/mabdel03/conda_envs/serve_env",
        )
    for label, value in {
        "harness environment prefix": str(harness_runtime.prefix),
        "serving environment prefix": str(serving_runtime.prefix),
        "HF_HOME": hf_home,
        "release worktree": (
            "" if resolved_release_worktree is None else str(resolved_release_worktree)
        ),
        "model contract": (
            "" if resolved_model_contract is None else str(resolved_model_contract)
        ),
        "fleet contract": (
            "" if resolved_fleet_contract is None else str(resolved_fleet_contract)
        ),
        "runtime attestation": runtime_attestation or "",
        "runtime integrity lease": runtime_integrity_lease or "",
        "release ID": release_id or "",
        "run root": canonical_run_root,
        "log directory": log_dir,
    }.items():
        if any(character in value for character in ('"', "\n", "\r", "`", "$")):
            raise ValueError(f"{label} contains an unsafe shell character")
    contracts = load_model_contracts(
        resolved_model_contract, expected_sha256=model_contract_sha256
    )
    identity = contracts.for_size(profile.model_size)
    if identity.hf_id != profile.hf_id:
        raise ValueError(
            f"serving profile {profile.name!r} HF id does not match frozen model contract"
        )
    if release_id:
        fleet = load_fleet_contract(
            str(resolved_fleet_contract),
            model_contracts=contracts,
            expected_sha256=str(fleet_contract_sha256),
            allow_capacity_layout=True,
        )
        fleet.verify_pool_root(canonical_run_root)
        replica_contract = fleet.for_replica(profile.name, replica)
        effective_qos = replica_contract.qos if qos is None else qos
        observed_placement = (partition, effective_qos, gpu_type, time_limit)
        expected_placement = (
            replica_contract.partition,
            replica_contract.qos,
            replica_contract.gpu_type,
            replica_contract.time_limit,
        )
        if replica_contract.qos is None:
            raise FleetContractError(
                f"production replica {replica_contract.replica_id} has no explicit "
                "protected QOS placement"
            )
        if observed_placement != expected_placement:
            raise FleetContractError(
                f"placement drift for {profile.name} replica {replica}: expected "
                f"{expected_placement!r}, observed {observed_placement!r}"
            )
        job_name = replica_contract.scheduler_job_name
        pool_id = replica_contract.pool_id
        replica_id: str | int = replica_contract.replica_id
        cpus = replica_contract.cpus_per_task
        mem = replica_contract.memory
        qos = effective_qos
    else:
        job_name = registry.serving_job_name(canonical_run_root, profile.name)
        pool_id = registry.server_pool_id(canonical_run_root)
        replica_id = replica
        cpus = max(8, profile.tp_size * 8)
        mem = f"{profile.tp_size * 120}G"
    port = _port_for(profile.name, replica)
    text = template_path.read_text(encoding="utf-8")
    repl = {
        "JOB_NAME": job_name,
        "SERVER_POOL_ID": pool_id,
        "REPLICA_ID": str(replica_id),
        "REPLICA_INDEX": str(replica),
        "FLEET_CONTRACT_SHA256": fleet_contract_sha256 or "",
        "RELEASE_FLEET_CONTRACT_SHA256": (
            release_fleet_contract_sha256 or ""
        ),
        "CAPACITY_GENERATION": str(capacity_generation or 0),
        "RELEASE_WORKTREE": (
            "" if resolved_release_worktree is None else str(resolved_release_worktree)
        ),
        "MODEL_CONTRACT_PATH": (
            "" if resolved_model_contract is None else str(resolved_model_contract)
        ),
        "FLEET_CONTRACT_PATH": (
            "" if resolved_fleet_contract is None else str(resolved_fleet_contract)
        ),
        "PROFILE_NAME": profile.name,
        "MODEL_SIZE": profile.model_size,
        "SERVED_MODEL_NAME": profile.served_model_name,
        "HF_ID": profile.hf_id,
        "MODEL_REVISION": identity.model_revision,
        "TOKENIZER_ID": identity.tokenizer_id,
        "TOKENIZER_REVISION": identity.tokenizer_revision,
        "MODEL_CONTRACT_SHA256": contracts.sha256,
        "RELEASE_ID": release_id or "",
        "ENVIRONMENT_HASH": environment_hash or "",
        "HARNESS_ENVIRONMENT_HASH": harness_environment_hash or "",
        "PRODUCTION_RUNTIME": "1" if release_id else "0",
        "RUNTIME_ATTESTATION": runtime_attestation or "",
        "RUNTIME_ATTESTATION_SHA256": runtime_attestation_sha256 or "",
        "RUNTIME_INTEGRITY_LEASE": runtime_integrity_lease or "",
        "IMMUTABLE_PINS_SHA256": immutable_pins_sha256 or "",
        "ROLLOUT_GENERATION": str(rollout_generation or 0),
        "HARNESS_PREFIX": str(harness_runtime.prefix),
        "HARNESS_PYTHON": str(harness_runtime.python),
        "HARNESS_PYTHON_VERSION": harness_runtime.python_version,
        "HARNESS_TRANSFORMERS_VERSION": harness_runtime.transformers_version,
        "HARNESS_TOKENIZERS_VERSION": harness_runtime.tokenizers_version,
        "SERVING_PREFIX": str(serving_runtime.prefix),
        "SERVING_PYTHON": str(serving_runtime.python),
        "SERVING_PYTHON_VERSION": serving_runtime.python_version,
        "SERVING_TRANSFORMERS_VERSION": serving_runtime.transformers_version,
        "SERVING_TOKENIZERS_VERSION": serving_runtime.tokenizers_version,
        "SERVING_TORCH_VERSION": serving_runtime.torch_version,
        "SERVING_VLLM_VERSION": serving_runtime.vllm_version,
        "SERVING_CUDA_VERSION": serving_runtime.cuda_version,
        "VLLM_EXECUTABLE": str(serving_runtime.prefix / "bin" / "vllm"),
        "HF_HOME": hf_home,
        "TP_SIZE": str(profile.tp_size),
        "MAX_MODEL_LEN": str(profile.max_model_len),
        "PARTITION": partition,
        "QOS_LINE": "" if qos is None else f"#SBATCH --qos={qos}\n",
        "GPU_TYPE": gpu_type,
        "CPUS": str(cpus),
        "MEM": mem,
        "TIME": time_limit,
        "PORT": str(port),
        "RUN_ROOT": canonical_run_root,
        "LOG_DIR": log_dir,
        "STANDBY_LINE": "  --standby \\\n" if standby else "",
    }
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    return text


def submit(
    model_size: str,
    run_root: str,
    partition: str,
    gpu_type: str,
    time_limit: str,
    replica: int = 0,
    serving_profile: str | None = None,
    release_id: str | None = None,
    environment_hash: str | None = None,
    model_contract_sha256: str | None = None,
    release_worktree: str | None = None,
    model_contract_path: str | None = None,
    harness_environment_prefix: str | None = None,
    serving_environment_prefix: str | None = None,
    harness_environment_manifest_path: str | None = None,
    serving_environment_manifest_path: str | None = None,
    harness_environment_hash: str | None = None,
    fleet_contract_path: str | None = None,
    fleet_contract_sha256: str | None = None,
    hf_home: str | None = None,
    runtime_attestation: str | None = None,
    runtime_attestation_sha256: str | None = None,
    runtime_integrity_lease: str | None = None,
    immutable_pins_sha256: str | None = None,
    rollout_generation: int | None = None,
) -> str:
    profile = _resolve_profile(model_size, serving_profile)
    log_dir = Path(run_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    effective_generation = rollout_generation
    if effective_generation is None:
        raw_generation = os.environ.get("ASYS_ROLLOUT_GENERATION")
        effective_generation = None if raw_generation is None else int(raw_generation)
    sbatch_text = render_sbatch(
        model_size, run_root, partition, gpu_type, time_limit, str(log_dir), replica=replica,
        serving_profile=serving_profile,
        release_id=release_id,
        environment_hash=environment_hash,
        model_contract_sha256=model_contract_sha256,
        release_worktree=release_worktree,
        model_contract_path=model_contract_path,
        harness_environment_prefix=harness_environment_prefix,
        serving_environment_prefix=serving_environment_prefix,
        harness_environment_manifest_path=harness_environment_manifest_path,
        serving_environment_manifest_path=serving_environment_manifest_path,
        harness_environment_hash=harness_environment_hash,
        fleet_contract_path=fleet_contract_path,
        fleet_contract_sha256=fleet_contract_sha256,
        hf_home=hf_home,
        runtime_attestation=runtime_attestation,
        runtime_attestation_sha256=runtime_attestation_sha256,
        runtime_integrity_lease=runtime_integrity_lease,
        immutable_pins_sha256=immutable_pins_sha256,
        rollout_generation=effective_generation,
    )
    # Production launch records are immutable and intent-addressed.  A controller crash
    # after sbatch cannot cause a later attempt to overwrite the script Slurm actually
    # spooled, and the command path remains enough to reconstruct the replica index.
    generation = int(effective_generation or 0)
    intent = secrets.token_hex(8)
    production_launch = bool(release_id or os.environ.get("ASYS_RELEASE_ID"))
    if production_launch and generation < 1:
        raise ValueError(
            "schema-5 serving launch requires a positive ASYS_ROLLOUT_GENERATION"
        )
    suffix = f"_r{replica}" if production_launch or replica else ""
    if production_launch:
        sbatch_path = (
            Path(run_root)
            / "servers"
            / "sbatch"
            / f"serve_{profile.name}{suffix}.g{generation:06d}.{intent}.sbatch"
        )
    else:
        sbatch_path = Path(run_root) / "servers" / f"serve_{profile.name}{suffix}.sbatch"
    sbatch_path.parent.mkdir(parents=True, exist_ok=True)
    if sbatch_path.exists():
        raise RuntimeError(f"refusing to overwrite serving intent {sbatch_path}")
    io.atomic_write_text(sbatch_path, sbatch_text)
    sbatch_path.chmod(stat.S_IMODE(sbatch_path.stat().st_mode) & ~0o222)
    out = subprocess.run(
        ["sbatch", str(sbatch_path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    # "Submitted batch job 12345"
    job_id = out.split()[-1]
    print(
        f"[launch] {profile.name} (model={profile.model_size}) r{replica} "
        f"{partition} -> job {job_id} (sbatch: {sbatch_path})"
    )
    return job_id


def _register_role(
    run_root: str,
    model_size: str,
    hf_id: str,
    port: int,
    serving_profile: str | None = None,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    tp_size: int | None = None,
    release_id: str | None = None,
    environment_hash: str | None = None,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    model_contract_sha256: str | None = None,
    release_worktree: str | None = None,
    model_contract_path: str | None = None,
    fleet_contract_path: str | None = None,
    fleet_contract_sha256: str | None = None,
    release_fleet_contract_sha256: str | None = None,
    capacity_generation: int | None = None,
    rollout_generation: int | None = None,
    expected_server_pool_id: str | None = None,
    replica_id: str | None = None,
    replica_index: int | None = None,
    standby: bool = False,
) -> None:
    """Run inside the serving job: wait for vLLM, then publish literal launch facts.

    The batch script is an immutable launch record, while the imported profile registry
    is shared code that may change while a job is pending.  Consequently the register
    command must pass the values rendered into that same batch script.  Recording values
    looked up from today's profile would let an older process masquerade as a newer
    context/TP layout after a profile migration.
    """
    profile = _resolve_profile(model_size, serving_profile)
    if served_model_name is None or max_model_len is None or tp_size is None:
        raise ValueError(
            "registration requires literal --served-model-name, --max-model-len, "
            "and --tp-size launch values"
        )
    release_fields = {
        "release_id": release_id,
        "environment_hash": environment_hash,
    }
    identity_fields = {
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "model_contract_sha256": model_contract_sha256,
    }
    release_provided = [name for name, value in release_fields.items() if value]
    release_missing = [name for name, value in release_fields.items() if not value]
    if release_provided and release_missing:
        raise ValueError(
            "serving release/environment provenance is all-or-none; missing "
            + ", ".join(release_missing)
        )
    identity_provided = [name for name, value in identity_fields.items() if value]
    identity_missing = [name for name, value in identity_fields.items() if not value]
    if identity_provided and identity_missing:
        raise ValueError(
            "serving model provenance is all-or-none; missing "
            + ", ".join(identity_missing)
        )
    contracts = None
    resolved_fleet_contract: Path | None = None
    if identity_provided:
        resolved_model_contract: Path | None = None
        if release_provided:
            (
                _resolved_release,
                resolved_model_contract,
                resolved_fleet_contract,
                _template_path,
            ) = _production_release_resources(
                release_worktree=release_worktree,
                model_contract_path=model_contract_path,
                fleet_contract_path=fleet_contract_path,
            )
        elif model_contract_path is not None:
            resolved_model_contract = Path(model_contract_path).expanduser().resolve()
        contracts = load_model_contracts(
            resolved_model_contract,
            expected_sha256=str(model_contract_sha256),
        )
        contracts.verify_identity(
            size=profile.model_size,
            hf_id=hf_id,
            model_revision=str(model_revision),
            tokenizer_id=str(tokenizer_id),
            tokenizer_revision=str(tokenizer_revision),
        )
    observed_pool_id = registry.server_pool_id(run_root)
    is_schema5_registration = bool(release_provided)
    if is_schema5_registration and not identity_provided:
        raise ValueError("schema-5 serving registration requires frozen model provenance")
    if is_schema5_registration and (
        len(str(environment_hash)) != 64
        or any(character not in "0123456789abcdef" for character in str(environment_hash))
    ):
        raise ValueError("schema-5 serving environment hash must be lowercase SHA-256")
    if is_schema5_registration and (
        not isinstance(fleet_contract_sha256, str)
        or len(fleet_contract_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in fleet_contract_sha256
        )
    ):
        raise ValueError("schema-5 serving registration requires a fleet-contract SHA-256")
    if is_schema5_registration and (
        not isinstance(release_fleet_contract_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", release_fleet_contract_sha256) is None
        or not isinstance(capacity_generation, int)
        or isinstance(capacity_generation, bool)
        or capacity_generation < 1
        or not isinstance(rollout_generation, int)
        or isinstance(rollout_generation, bool)
        or rollout_generation < 1
    ):
        raise ValueError(
            "schema-5 serving registration requires exact release-fleet, "
            "capacity, and rollout generations"
        )
    if (
        expected_server_pool_id is not None
        and expected_server_pool_id != observed_pool_id
    ):
        raise ValueError(
            "serving job run-root/pool identity mismatch: "
            f"expected {expected_server_pool_id!r}, observed {observed_pool_id!r}"
        )
    if is_schema5_registration and expected_server_pool_id is None:
        raise ValueError("schema-5 serving registration requires a server pool identity")
    if replica_index is not None and replica_index < 0:
        raise ValueError("serving registration requires a non-negative replica index")
    if is_schema5_registration and (
        replica_index is None
        or not isinstance(replica_id, str)
        or not replica_id.startswith(f"{observed_pool_id}--")
    ):
        raise ValueError("schema-5 serving registration requires frozen replica identity")
    resolved_replica_index = 0 if replica_index is None else replica_index
    resolved_replica_id: str | int = (
        resolved_replica_index if replica_id is None else replica_id
    )
    if is_schema5_registration:
        if not fleet_contract_path or contracts is None:
            raise ValueError("schema-5 serving registration requires a fleet contract")
        fleet = load_fleet_contract(
            resolved_fleet_contract,
            model_contracts=contracts,
            expected_sha256=fleet_contract_sha256,
            allow_capacity_layout=True,
        )
        fleet.verify_pool_root(run_root)
        replica_contract = fleet.for_replica(profile.name, resolved_replica_index)
        if (
            resolved_replica_id != replica_contract.replica_id
            or observed_pool_id != replica_contract.pool_id
        ):
            raise ValueError("schema-5 serving replica identity differs from fleet contract")
    expected_port = _port_for(profile.name, resolved_replica_index)
    if expected_server_pool_id is not None and port != expected_port:
        raise ValueError(
            "serving replica/port mismatch: "
            f"profile {profile.name!r} replica {resolved_replica_index} requires "
            f"port {expected_port}, observed {port}"
        )
    # Validate immutable identity before spending up to forty minutes waiting for a
    # server that this job would ultimately be forbidden to publish.
    healthcheck.wait_until_ready("localhost", port, timeout_s=2400.0)
    entry = registry.register_server(
        run_root,
        model_size,
        hf_id,
        port,
        serving_profile=profile.name,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        tp_size=tp_size,
        release_id=release_id or None,
        environment_hash=environment_hash or None,
        model_revision=model_revision or None,
        tokenizer_id=tokenizer_id or None,
        tokenizer_revision=tokenizer_revision or None,
        model_contract_sha256=model_contract_sha256 or None,
        fleet_contract_sha256=fleet_contract_sha256 or None,
        expected_server_pool_id=observed_pool_id,
        replica_id=resolved_replica_id,
        replica_index=resolved_replica_index,
        release_fleet_contract_sha256=release_fleet_contract_sha256,
        capacity_generation=capacity_generation,
        rollout_generation=rollout_generation,
        standby=standby,
    )
    print(f"[register] {profile.name} ready at {entry.base_url}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch or register a vLLM server.")
    ap.add_argument("--register", action="store_true", help="run inside the serving job")
    ap.add_argument(
        "--standby",
        action="store_true",
        help="register below the transactional standby namespace until promoted",
    )
    ap.add_argument("--model-size", required=True)
    ap.add_argument(
        "--profile",
        dest="serving_profile",
        help="named runtime profile (for example 32B-long); model identity stays --model-size",
    )
    ap.add_argument("--server-pool-id", dest="expected_server_pool_id")
    ap.add_argument("--replica-id")
    ap.add_argument("--replica-index", type=int)
    ap.add_argument("--run-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--hf-id", help="(register role) the HF id being served")
    ap.add_argument("--model-revision", help="(register role) exact model commit")
    ap.add_argument("--tokenizer-id", help="(register role) exact tokenizer repository")
    ap.add_argument("--tokenizer-revision", help="(register role) exact tokenizer commit")
    ap.add_argument(
        "--model-contract-sha256",
        default=os.environ.get("ASYS_MODEL_CONTRACT_SHA256"),
        help="exact frozen model-contract byte hash",
    )
    ap.add_argument(
        "--release-worktree",
        default=os.environ.get("ASYS_RELEASE_WORKTREE"),
        help="exact immutable source worktree supplying server runtime resources",
    )
    ap.add_argument(
        "--model-contract",
        default=os.environ.get("ASYS_MODEL_CONTRACT"),
        help="exact model contract below --release-worktree",
    )
    ap.add_argument(
        "--release-id",
        default=os.environ.get("ASYS_RELEASE_ID"),
        help="immutable serving release id",
    )
    ap.add_argument(
        "--environment-hash",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_SHA256"),
        help="immutable serving environment hash",
    )
    ap.add_argument(
        "--harness-environment-prefix",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_PREFIX"),
    )
    ap.add_argument(
        "--serving-environment-prefix",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_PREFIX"),
    )
    ap.add_argument(
        "--harness-environment-manifest",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_MANIFEST"),
    )
    ap.add_argument(
        "--serving-environment-manifest",
        default=os.environ.get("ASYS_SERVING_ENVIRONMENT_MANIFEST"),
    )
    ap.add_argument(
        "--harness-environment-hash",
        default=os.environ.get("ASYS_HARNESS_ENVIRONMENT_SHA256"),
    )
    ap.add_argument(
        "--fleet-contract",
        default=os.environ.get("ASYS_FLEET_CONTRACT"),
    )
    ap.add_argument(
        "--fleet-contract-sha256",
        default=os.environ.get("ASYS_FLEET_CONTRACT_SHA256"),
    )
    ap.add_argument(
        "--release-fleet-contract-sha256",
        default=os.environ.get("ASYS_RELEASE_FLEET_CONTRACT_SHA256"),
    )
    ap.add_argument(
        "--capacity-generation",
        type=int,
        default=(
            None
            if os.environ.get("ASYS_CAPACITY_GENERATION") is None
            else int(os.environ["ASYS_CAPACITY_GENERATION"])
        ),
    )
    ap.add_argument(
        "--rollout-generation",
        type=int,
        default=(
            None
            if os.environ.get("ASYS_ROLLOUT_GENERATION") is None
            else int(os.environ["ASYS_ROLLOUT_GENERATION"])
        ),
    )
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument(
        "--served-model-name",
        help="(register role) literal --served-model-name passed to vLLM",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        help="(register role) literal --max-model-len passed to vLLM",
    )
    ap.add_argument(
        "--tp-size",
        type=int,
        help="(register role) literal --tensor-parallel-size passed to vLLM",
    )
    ap.add_argument("--port", type=int, help="(register role) local vLLM port")
    ap.add_argument("--partition", default=_PARTITION_DEFAULT)
    ap.add_argument("--gpu-type", default=_GPU_TYPE_DEFAULT)
    ap.add_argument("--time", dest="time_limit", default="2-00:00:00")
    ap.add_argument("--replica", type=int, default=0, help="replica index (port offset)")
    args = ap.parse_args()

    if args.register:
        if (
            not args.hf_id
            or args.port is None
            or not args.served_model_name
            or args.max_model_len is None
            or args.tp_size is None
        ):
            ap.error(
                "--register requires --hf-id, --served-model-name, "
                "--max-model-len, --tp-size, and --port"
            )
        _register_role(
            args.run_root,
            args.model_size,
            args.hf_id,
            args.port,
            serving_profile=args.serving_profile,
            served_model_name=args.served_model_name,
            max_model_len=args.max_model_len,
            tp_size=args.tp_size,
            release_id=args.release_id,
            environment_hash=args.environment_hash,
            model_revision=args.model_revision,
            tokenizer_id=args.tokenizer_id,
            tokenizer_revision=args.tokenizer_revision,
            model_contract_sha256=args.model_contract_sha256,
            release_worktree=args.release_worktree,
            model_contract_path=args.model_contract,
            fleet_contract_path=args.fleet_contract,
            fleet_contract_sha256=args.fleet_contract_sha256,
            release_fleet_contract_sha256=args.release_fleet_contract_sha256,
            capacity_generation=args.capacity_generation,
            rollout_generation=args.rollout_generation,
            expected_server_pool_id=args.expected_server_pool_id,
            replica_id=args.replica_id,
            replica_index=args.replica_index,
            standby=args.standby,
        )
    else:
        submit(
            args.model_size, args.run_root, args.partition, args.gpu_type,
            args.time_limit, replica=args.replica, serving_profile=args.serving_profile,
            release_id=args.release_id,
            environment_hash=args.environment_hash,
            model_contract_sha256=args.model_contract_sha256,
            release_worktree=args.release_worktree,
            model_contract_path=args.model_contract,
            harness_environment_prefix=args.harness_environment_prefix,
            serving_environment_prefix=args.serving_environment_prefix,
            harness_environment_manifest_path=args.harness_environment_manifest,
            serving_environment_manifest_path=args.serving_environment_manifest,
            harness_environment_hash=args.harness_environment_hash,
            fleet_contract_path=args.fleet_contract,
            fleet_contract_sha256=args.fleet_contract_sha256,
            hf_home=args.hf_home,
        )


if __name__ == "__main__":
    main()
