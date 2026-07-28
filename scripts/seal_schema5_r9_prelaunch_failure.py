#!/usr/bin/env python3
"""Seal r9's deterministic offline-probe diagnostic-contract failure for r12."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import types
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = (
    "schema5-v1.2-r9-prelaunch-offline-diagnostic-contract-failure-seal-v3"
)
CLASSIFICATION = (
    "deterministic_prelaunch_offline_diagnostic_cardinality_failure"
)
R9_TAG = "sweep-recovery-schema5-v1.2-r9"
R9_NAMESPACE = "schema5-v1.2-r9"
R9_COMMIT = "f134bba109ce77beca60584a5f589361856f2301"
R9_TAG_OBJECT = "17b0386800c70fb9e72d090c7b8e1c3ddf9c7b52"
R9_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R9_COMPLETE.json"
R9_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r9"
R9_TOOLCHAIN_RELATIVE = Path("toolchains/r9/conda")
R3_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r3"
R9_PROBE_ROOT = Path("/tmp/schema5-r3-prelaunch-mabdel03-r9")
BROKEN_ENVELOPE_NAME = (
    "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
)
OFFLINE_ENVELOPE_NAME = (
    "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json"
)
EVIDENCE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r9-offline-diagnostic"
)
MARKER_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_INTENT.json"
ARCHIVE_NAME = "original_probe_tree"
REPRODUCTION_NAME = "equivalent_offline_reproduction"
STDOUT_NAME = "conda.stdout"
STDERR_NAME = "conda.stderr"
RECORDER_TRANSCRIPT_NAME = "r9_recorder_reproduction.json"
R10_FAILURE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r10-r9-sealer"
)
R10_FAILURE_MARKER_NAME = "PRELAUNCH_R9_SEALER_FAILURE_SEALED.json"
R11_FAILURE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r11-r9-sealer"
)
R11_FAILURE_MARKER_NAME = (
    "PRELAUNCH_EQUIVALENT_DIAGNOSTIC_FAILURE_SEALED.json"
)
EXPECTED_R9_REJECTION = (
    "r3 offline-cache probe lacks the canonical Conda OfflineError "
    "remote-fetch context"
)
EXPECTED_R9_CLI_STDERR = (
    f"[schema5-evidence] ERROR: {EXPECTED_R9_REJECTION}\n"
)
PRIOR_VOLATILE_AUDIT = {
    "status": "observed_before_external_tmp_cleanup",
    "authoritative_seal_evidence": False,
    "entry_count": 18,
    "inventory_sha256": (
        "3d59befdad33086a897b2a28b07d9181db9fc6891a099fb9641c1b006c957c0f"
    ),
    "accepted_broken_envelope": {
        "failure_id": (
            "41bd8e668d7c5696dee0ac03829c999d523d30ad5ad92027524409c359609ea2"
        ),
        "sha256": (
            "1eb9f483b6ea39a8959a6ca3f7391b53146f6372bd719d1c5ab29a06667304f8"
        ),
        "size": 4066,
    },
    "partial_artifact_count": 8,
}
_CHUNK_SIZE = 8 * 1024 * 1024


class R9FailureSealError(RuntimeError):
    """The r9 prelaunch failure cannot be sealed or verified exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r8_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r8_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R9FailureSealError("cannot load marker-last evidence helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    _helpers.R8FailureSealError,
    *_helpers.MarkerLastEvidenceError,
)
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory
_run_git = _helpers._helpers._run_git
_require_recursively_read_only = (
    _helpers._helpers._require_recursively_read_only
)


def _git(checkout: Path, *arguments: str) -> str:
    return _run_git(checkout, *arguments, optional_locks=False)


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(recovery / R9_CHECKOUT, description="r9 checkout")
    tag_ref = f"refs/tags/{R9_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R9_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R9_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R9_COMMIT
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R9FailureSealError("r9 checkout identity drifted")
    durable_path = recovery / R9_DURABLE_MARKER
    durable, durable_raw = _read_json(
        durable_path, description="r9 durable release marker"
    )
    bundle_path = Path(str(durable.get("bundle_path", "")))
    if (
        durable.get("passed") is not True
        or durable.get("release_tag") != R9_TAG
        or durable.get("chain_namespace") != R9_NAMESPACE
        or durable.get("release_git_commit") != R9_COMMIT
        or durable.get("release_tag_object") != R9_TAG_OBJECT
        or durable.get("remote_commit") != R9_COMMIT
        or durable.get("remote_peeled_commit") != R9_COMMIT
        or durable.get("remote_tag_object") != R9_TAG_OBJECT
        or not bundle_path.is_file()
        or bundle_path.is_symlink()
        or _file_ref(bundle_path, description="r9 durable Git bundle")["sha256"]
        != durable.get("bundle_sha256")
    ):
        raise R9FailureSealError("r9 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R9_TAG,
        "release_git_commit": R9_COMMIT,
        "release_tag_object": R9_TAG_OBJECT,
        "recorder": _file_ref(
            checkout / "scripts/seal_recovery_evidence.py",
            description="r9 recovery-evidence recorder",
        ),
        "durable_marker": {
            "path": str(durable_path),
            "sha256": hashlib.sha256(durable_raw).hexdigest(),
            "size": len(durable_raw),
            "marker_id": durable.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle_path, description="r9 durable Git bundle"
        ),
    }


def _toolchain_binding(recovery: Path) -> dict[str, Any]:
    checkout = recovery / R9_CHECKOUT
    runtime_identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    toolchain = recovery / R9_TOOLCHAIN_RELATIVE
    names = (
        "scripts",
        "scripts.schema5_conda_runtime_identity",
        "scripts.provision_schema5_conda_toolchain",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("scripts")
        package.__path__ = []
        sys.modules["scripts"] = package
        runtime_spec = importlib.util.spec_from_file_location(
            "scripts.schema5_conda_runtime_identity", runtime_identity
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise R9FailureSealError("cannot load r9 runtime-identity verifier")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        provisioner_spec = importlib.util.spec_from_file_location(
            "scripts.provision_schema5_conda_toolchain", provisioner
        )
        if provisioner_spec is None or provisioner_spec.loader is None:
            raise R9FailureSealError("cannot load r9 toolchain verifier")
        provisioner_module = importlib.util.module_from_spec(provisioner_spec)
        sys.modules[provisioner_spec.name] = provisioner_module
        provisioner_spec.loader.exec_module(provisioner_module)
        binding = provisioner_module.verified_conda_toolchain_binding(
            toolchain, exercise=False
        )
    except Exception as exc:
        if isinstance(exc, R9FailureSealError):
            raise
        raise R9FailureSealError(
            f"r9 sealed toolchain verification failed: {exc}"
        ) from exc
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    if (
        not isinstance(binding, dict)
        or binding.get("protocol")
        != "schema5-v1.2-r9-offline-conda-toolchain-v1"
        or binding.get("release_tag") != R9_TAG
        or binding.get("chain_namespace") != R9_NAMESPACE
        or binding.get("toolchain_root") != str(toolchain)
        or binding.get("portable_shebang", {}).get("interpreter")
        != str(toolchain / "base/bin/python")
    ):
        raise R9FailureSealError("r9 sealed toolchain binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _load_r9_evidence_module(recovery: Path):
    path = recovery / R9_CHECKOUT / "scripts/seal_recovery_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "_schema5_r9_recovery_evidence", path
    )
    if spec is None or spec.loader is None:
        raise R9FailureSealError("cannot load immutable r9 evidence validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise R9FailureSealError("r9 evidence-validator source drifted")
    return module


def _r10_failure_binding(recovery: Path, scheduler_user: str) -> dict[str, Any]:
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r10_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r10_prelaunch_failure_binding", path
    )
    if spec is None or spec.loader is None:
        raise R9FailureSealError("cannot load r10 failure-seal verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = recovery / R10_FAILURE_RELATIVE_ROOT
    try:
        binding = module.verify_failure_seal(
            root,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        )
    except Exception as exc:
        raise R9FailureSealError(
            f"r10 wrapper-failure seal is invalid: {exc}"
        ) from exc
    if (
        not isinstance(binding, dict)
        or binding.get("passed") is not True
        or binding.get("root") != str(root)
        or binding.get("protocol") != module.PROTOCOL
        or binding.get("release_tag") != module.R10_TAG
        or binding.get("release_git_commit") != module.R10_COMMIT
        or binding.get("chain_namespace") != module.R10_NAMESPACE
        or binding.get("marker")
        != str(root / R10_FAILURE_MARKER_NAME)
        or binding.get("classification") != module.CLASSIFICATION
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("known_scheduler_job_ids") != []
    ):
        raise R9FailureSealError("r10 wrapper-failure binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _r11_failure_binding(recovery: Path, scheduler_user: str) -> dict[str, Any]:
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r11_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r11_prelaunch_failure_binding", path
    )
    if spec is None or spec.loader is None:
        raise R9FailureSealError("cannot load r11 failure-seal verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = recovery / R11_FAILURE_RELATIVE_ROOT
    try:
        binding = module.verify_failure_seal(
            root,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        )
    except Exception as exc:
        raise R9FailureSealError(
            f"r11 delimiter-failure seal is invalid: {exc}"
        ) from exc
    if (
        not isinstance(binding, dict)
        or binding.get("passed") is not True
        or binding.get("root") != str(root)
        or binding.get("protocol") != module.PROTOCOL
        or binding.get("release_tag") != module.R11_TAG
        or binding.get("release_git_commit") != module.R11_COMMIT
        or binding.get("chain_namespace") != module.R11_NAMESPACE
        or binding.get("marker")
        != str(root / R11_FAILURE_MARKER_NAME)
        or binding.get("classification") != module.CLASSIFICATION
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("known_scheduler_job_ids") != []
    ):
        raise R9FailureSealError("r11 delimiter-failure binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_inventory(root: Path) -> dict[str, Any]:
    root = _safe_directory(root, description="r9 probe evidence tree")
    rows: list[dict[str, Any]] = []
    for path in [root, *sorted(root.rglob("*"))]:
        metadata = path.lstat()
        relative = "." if path == root else str(path.relative_to(root))
        row: dict[str, Any] = {"path": relative}
        if stat.S_ISDIR(metadata.st_mode):
            row["type"] = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            row.update(
                {
                    "type": "file",
                    "size": metadata.st_size,
                    "sha256": _sha256_file(path),
                }
            )
        elif stat.S_ISLNK(metadata.st_mode):
            row.update({"type": "symlink", "target": os.readlink(path)})
        else:
            raise R9FailureSealError(f"unsupported r9 probe entry: {path}")
        rows.append(row)
    return {
        "entry_count": len(rows),
        "rows": rows,
        "inventory_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _validate_original_probe_tree(recovery: Path) -> dict[str, Any]:
    root = _safe_directory(R9_PROBE_ROOT, description="failed r9 probe root")
    expected_top = {
        BROKEN_ENVELOPE_NAME,
        f"{BROKEN_ENVELOPE_NAME}.sha256",
        "empty-conda-pkgs",
        "offline-clone-destination",
    }
    if {path.name for path in root.iterdir()} != expected_top:
        raise R9FailureSealError("failed r9 probe tree topology drifted")
    offline = root / OFFLINE_ENVELOPE_NAME
    if (
        offline.exists()
        or offline.is_symlink()
        or offline.with_suffix(".json.sha256").exists()
        or offline.with_suffix(".json.sha256").is_symlink()
    ):
        raise R9FailureSealError("r9 unexpectedly published an offline envelope")
    r9_module = _load_r9_evidence_module(recovery)
    try:
        payload, _raw, source_record = (
            r9_module._validate_r3_prelaunch_failure_envelope(
                root / BROKEN_ENVELOPE_NAME
            )
        )
    except Exception as exc:
        raise R9FailureSealError(
            f"accepted r9 broken-link envelope is invalid: {exc}"
        ) from exc
    if (
        payload.get("classification")
        != "unsafe_recorded_broken_internal_symlink"
        or payload.get("scheduler_job_ids") != []
        or payload.get("pre_scheduler_submission") is not True
    ):
        raise R9FailureSealError("accepted r9 broken-link envelope drifted")
    cache = _safe_directory(
        root / "empty-conda-pkgs", description="r9 failed offline cache"
    )
    cache_names = {path.name for path in cache.iterdir()}
    partials = sorted(name for name in cache_names if name.endswith(".partial"))
    if (
        not partials
        or not cache_names.issubset({*partials, "urls", "urls.txt"})
        or any(
            not re.fullmatch(
                r"[A-Za-z0-9_.+%-]+(?:[.]conda|[.]tar[.]bz2)[.]partial",
                name,
            )
            for name in partials
        )
        or any(
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != 0
            for path in cache.iterdir()
        )
    ):
        raise R9FailureSealError("r9 failed offline-cache residue drifted")
    destination = _safe_directory(
        root / "offline-clone-destination",
        description="r9 failed offline destination",
    )
    if {path.name for path in destination.iterdir()} != {
        ".condarc",
        "_conda",
        "micromamba",
    }:
        raise R9FailureSealError("r9 failed offline destination drifted")
    conda_link = destination / "_conda"
    expected_target = str(recovery / R9_TOOLCHAIN_RELATIVE / "base/micromamba")
    if (
        not conda_link.is_symlink()
        or os.readlink(conda_link) != expected_target
        or not (destination / ".condarc").is_file()
        or (destination / ".condarc").is_symlink()
        or not (destination / "micromamba").is_file()
        or (destination / "micromamba").is_symlink()
    ):
        raise R9FailureSealError("r9 failed offline destination identity drifted")
    return {
        "path": str(root),
        "inventory": _portable_inventory(root),
        "accepted_broken_envelope": source_record
        | {"failure_id": payload.get("failure_id")},
        "offline_envelope_published": False,
        "partial_artifact_count": len(partials),
        "partial_artifacts": partials,
        "destination_entries": [".condarc", "_conda", "micromamba"],
    }


def _validate_volatile_probe_absence() -> dict[str, Any]:
    if R9_PROBE_ROOT.exists() or R9_PROBE_ROOT.is_symlink():
        raise R9FailureSealError(
            "volatile r9 probe root reappeared before controlled reproduction"
        )
    return {
        "path": str(R9_PROBE_ROOT),
        "present_at_seal": False,
        "loss_classification": "external_volatile_tmp_cleanup_before_durable_seal",
        "prior_live_audit": json.loads(
            json.dumps(PRIOR_VOLATILE_AUDIT, sort_keys=True)
        ),
        "replacement_evidence_policy": (
            "reproduce_both_exact_r9_recorder_commands_and_archive_marker_first"
        ),
    }


def _scheduler_quiescence(user: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/bin/squeue", "-h", "-r", "-u", user, "-o", "%i|%j|%k|%T"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise R9FailureSealError(f"squeue failed: {completed.stderr.strip()}")
    matching = [
        row
        for row in completed.stdout.splitlines()
        if "s5v12r9" in row
        or "schema5-v1.2-r9" in row
        or "sweep-recovery-schema5-v1.2-r9" in row
    ]
    if matching:
        raise R9FailureSealError(f"r9 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r9_jobs": []}


def _scientific_state(recovery: Path) -> dict[str, Any]:
    results = recovery.parent.parent
    required_absent = [
        str(path)
        for path in (
            results / ".dispatcher-schema5-v1",
            results / "server_pools/schema5-v1",
            results / "full_sweep_schema5_v1",
            results / "full_sweep_agent_counts_schema5_v1",
            results / "full_sweep_agent_count_7_schema5_v1",
            recovery / "slurm_canaries/schema5-v1.2-r9",
            recovery / "materialization_pilots/schema5-v1.2-r9",
            recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R9.json",
            recovery / "prelaunch_failures/schema5-v1.2-r3",
        )
    ]
    present = [
        path
        for path in required_absent
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R9FailureSealError(
            f"r9 failure is not prelaunch zero-result evidence: {present}"
        )
    return {
        "captured_before_r12_production_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": required_absent,
    }


def _permanently_absent_paths(recovery: Path) -> list[str]:
    return [
        str(recovery / "slurm_canaries/schema5-v1.2-r9"),
        str(recovery / "materialization_pilots/schema5-v1.2-r9"),
        str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R9.json"),
    ]


def _validate_permanent_absence(recovery: Path, expected: Any) -> None:
    canonical = _permanently_absent_paths(recovery)
    if expected != canonical:
        raise R9FailureSealError("r9 permanent-absence contract drifted")
    present = [
        path
        for path in canonical
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R9FailureSealError(f"r9 execution namespace appeared later: {present}")


def _reproduction_contract(recovery: Path, evidence: Path) -> dict[str, Any]:
    toolchain = recovery / R9_TOOLCHAIN_RELATIVE
    root = evidence / REPRODUCTION_NAME
    cache = root / "empty-conda-pkgs"
    destination = root / "offline-clone-destination"
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "HOME": str(root / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
        "XDG_STATE_HOME": str(root / "xdg-state"),
        "CONDA_ENVS_PATH": str(root / "conda-envs"),
        "CONDA_NO_PLUGINS": "true",
        "CONDA_OFFLINE": "true",
        "CONDA_PIP_INTEROP_ENABLED": "false",
        "CONDA_PKGS_DIRS": str(cache),
    }
    return {
        "argv": [
            str(toolchain / "base/bin/conda"),
            "create",
            "--yes",
            "--offline",
            "--clone",
            str(toolchain / "base"),
            "--prefix",
            str(destination),
        ],
        "cwd": str(recovery / R3_CHECKOUT),
        "environment": environment,
        "root": str(root),
        "cache": str(cache),
        "destination": str(destination),
        "expected_returncode": 1,
        "r9_expected_rejection": EXPECTED_R9_REJECTION,
    }


def _r9_probe_environment() -> list[tuple[str, str]]:
    root = R9_PROBE_ROOT
    return [
        ("HOME", str(root / "home")),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("PYTHONNOUSERSITE", "1"),
        ("TEMP", str(root / "tmp")),
        ("TMP", str(root / "tmp")),
        ("TMPDIR", str(root / "tmp")),
        ("XDG_CACHE_HOME", str(root / "xdg-cache")),
        ("XDG_CONFIG_HOME", str(root / "xdg-config")),
        ("XDG_DATA_HOME", str(root / "xdg-data")),
        ("XDG_STATE_HOME", str(root / "xdg-state")),
    ]


def _r9_recorder_argv(
    recovery: Path,
    *,
    classification: str,
    apply: bool,
) -> list[str]:
    checkout = recovery / R9_CHECKOUT
    r3_checkout = recovery / R3_CHECKOUT
    toolchain = recovery / R9_TOOLCHAIN_RELATIVE
    output = (
        R9_PROBE_ROOT / BROKEN_ENVELOPE_NAME
        if classification == "unsafe_recorded_broken_internal_symlink"
        else R9_PROBE_ROOT / OFFLINE_ENVELOPE_NAME
    )
    command = [
        sys.executable,
        "-I",
        "-B",
        str(checkout / "scripts/seal_recovery_evidence.py"),
        "record-prelaunch-attempt",
        "--output",
        str(output),
        "--classification",
        classification,
        "--cwd",
        str(r3_checkout),
    ]
    environment = _r9_probe_environment()
    if classification == "offline_clone_unseeded_release_local_cache":
        environment.extend(
            [
                ("CONDA_ENVS_PATH", str(R9_PROBE_ROOT / "conda-envs")),
                ("CONDA_NO_PLUGINS", "true"),
                ("CONDA_OFFLINE", "true"),
                ("CONDA_PIP_INTEROP_ENABLED", "false"),
                ("CONDA_PKGS_DIRS", str(R9_PROBE_ROOT / "empty-conda-pkgs")),
            ]
        )
    for key, value in environment:
        command.extend(["--environment", key, value])
    inputs = [
        ("tagged-r3-release-checkout", r3_checkout),
        ("tagged-r9-release-checkout", checkout),
        ("sealed-r9-conda-toolchain", toolchain),
    ]
    if classification == "unsafe_recorded_broken_internal_symlink":
        inputs.append(
            (
                "recorded-shared-conda-base",
                Path("/orcd/data/lhtsai/001/om2/mabdel03/miniforge3"),
            )
        )
    for name, path in inputs:
        command.extend(["--input-root", name, str(path)])
    command.extend(["--write-root", str(R9_PROBE_ROOT)])
    if apply:
        command.append("--apply")
    command.append("--command")
    if classification == "unsafe_recorded_broken_internal_symlink":
        command.extend(
            [
                str(toolchain / "base/bin/python"),
                "-I",
                "-B",
                str(r3_checkout / "scripts/schema5_conda_runtime_identity.py"),
                "--conda-executable",
                "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda",
            ]
        )
    else:
        command.extend(
            [
                str(toolchain / "base/bin/conda"),
                "create",
                "--yes",
                "--offline",
                "--clone",
                str(toolchain / "base"),
                "--prefix",
                str(R9_PROBE_ROOT / "offline-clone-destination"),
            ]
        )
    return command


def _run_r9_recorder(
    recovery: Path,
    *,
    classification: str,
    apply: bool,
) -> dict[str, Any]:
    argv = _r9_recorder_argv(
        recovery, classification=classification, apply=apply
    )
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        timeout=1_200,
        check=False,
    )
    return {
        "argv": argv,
        "apply": apply,
        "classification": classification,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _reproduce_volatile_probe_tree(
    recovery: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_volatile_probe_absence()
    R9_PROBE_ROOT.mkdir(mode=0o700)
    (R9_PROBE_ROOT / "empty-conda-pkgs").mkdir(mode=0o700)
    records = [
        _run_r9_recorder(
            recovery,
            classification="unsafe_recorded_broken_internal_symlink",
            apply=False,
        ),
        _run_r9_recorder(
            recovery,
            classification="unsafe_recorded_broken_internal_symlink",
            apply=True,
        ),
        _run_r9_recorder(
            recovery,
            classification="unsafe_recorded_broken_internal_symlink",
            apply=True,
        ),
        _run_r9_recorder(
            recovery,
            classification="offline_clone_unseeded_release_local_cache",
            apply=False,
        ),
        _run_r9_recorder(
            recovery,
            classification="offline_clone_unseeded_release_local_cache",
            apply=True,
        ),
    ]
    for index, record in enumerate(records[:4]):
        if record["returncode"] != 0:
            raise R9FailureSealError(
                f"exact r9 recorder reproduction step {index} failed: "
                f"{record['stderr'].strip()}"
            )
    rejected = records[-1]
    if (
        rejected["returncode"] != 2
        or rejected["stdout"] != ""
        or rejected["stderr"] != EXPECTED_R9_CLI_STDERR
    ):
        raise R9FailureSealError(
            "exact r9 offline recorder rejection reproduction drifted"
        )
    state = _validate_original_probe_tree(recovery)
    transcript: dict[str, Any] = {
        "schema_version": 1,
        "protocol": f"{PROTOCOL}-exact-r9-recorder-reproduction",
        "source_release": R9_TAG,
        "volatile_root_recreated_from_empty": True,
        "records": records,
        "resulting_probe_state": state,
    }
    transcript["transcript_id"] = _identity(transcript, "transcript_id")
    return state, transcript


def _intent_payload(
    recovery: Path, evidence: Path, user: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "sealed_r9_toolchain": _toolchain_binding(recovery),
        "sealed_r10_wrapper_failure": _r10_failure_binding(recovery, user),
        "sealed_r11_delimiter_failure": _r11_failure_binding(recovery, user),
        "original_probe_state": source,
        "reproduction": _reproduction_contract(recovery, evidence),
        "scientific_state": _scientific_state(recovery),
        "permanently_absent_r9_execution_paths": _permanently_absent_paths(
            recovery
        ),
        "scheduler": _scheduler_quiescence(user),
        "evidence_root": str(evidence),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any], recovery: Path, evidence: Path, user: str
) -> None:
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != _release_binding(recovery)
        or intent.get("sealed_r9_toolchain") != _toolchain_binding(recovery)
        or intent.get("sealed_r10_wrapper_failure")
        != _r10_failure_binding(recovery, user)
        or intent.get("sealed_r11_delimiter_failure")
        != _r11_failure_binding(recovery, user)
        or intent.get("reproduction") != _reproduction_contract(recovery, evidence)
        or intent.get("evidence_root") != str(evidence)
        or intent.get("scheduler") != _scheduler_quiescence(user)
    ):
        raise R9FailureSealError("r9 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r12_production_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
    ):
        raise R9FailureSealError("r9 prelaunch scientific-state evidence drifted")
    source = intent.get("original_probe_state")
    if (
        not isinstance(source, dict)
        or source.get("path") != str(R9_PROBE_ROOT)
        or source.get("present_at_seal") is not False
        or source.get("loss_classification")
        != "external_volatile_tmp_cleanup_before_durable_seal"
        or source.get("prior_live_audit") != PRIOR_VOLATILE_AUDIT
        or source.get("replacement_evidence_policy")
        != "reproduce_both_exact_r9_recorder_commands_and_archive_marker_first"
    ):
        raise R9FailureSealError("r9 volatile probe-loss binding drifted")
    _validate_permanent_absence(
        recovery, intent.get("permanently_absent_r9_execution_paths")
    )


def _validate_structural_diagnostic(
    *, stdout: str, stderr: str, contract: Mapping[str, Any]
) -> list[str]:
    base = Path(contract["argv"][5])
    destination = Path(contract["destination"])
    prefix = (
        f"Source:      {base}\n"
        f"Destination: {destination}\n"
        "Packages: 89\n"
        "Files: 3\n\n"
        "Downloading and Extracting Packages: ...working..."
    )
    suffix = " done\n"
    body = (
        stdout[len(prefix) : -len(suffix)]
        if stdout.startswith(prefix) and stdout.endswith(suffix)
        else ""
    )
    if (
        len(stdout.encode("utf-8")) > 262_144
        or not body
        or re.fullmatch(r"[\x1b\r\n A-Za-z0-9_.+%|()[\];-]+", body) is None
        or any(token in stdout for token in ("http://", "https://", "OfflineError", "Traceback"))
    ):
        raise R9FailureSealError("equivalent r9 Conda progress output drifted")
    block = re.compile(
        r"OfflineError: EnforceUnusedAdapter called with url "
        r"(?P<url>https://conda[.]anaconda[.]org/conda-forge/"
        r"(?:linux-64|noarch)/"
        r"[A-Za-z0-9_.+%-]+(?:[.]conda|[.]tar[.]bz2))[.]\n"
        r"This command is using a remote connection in offline mode[.]\n"
    )
    if stderr.startswith("\n\n") or stderr.endswith("\n\n\n"):
        raise R9FailureSealError("equivalent r9 OfflineError stream drifted")
    text = stderr[1:] if stderr.startswith("\n") else stderr
    if text.endswith("\n\n"):
        text = text[:-1]
    urls: list[str] = []
    offset = 0
    while offset < len(text):
        match = block.match(text, offset)
        if match is None:
            raise R9FailureSealError("equivalent r9 OfflineError stream drifted")
        urls.append(match.group("url"))
        offset = match.end()
    if (
        not urls
        or len(urls) > 512
        or offset != len(text)
        or len(stderr.encode("utf-8")) > 524_288
    ):
        raise R9FailureSealError("equivalent r9 OfflineError stream drifted")
    return urls


def _r9_rejection(
    recovery: Path, *, stdout: str, stderr: str
) -> dict[str, Any]:
    module = _load_r9_evidence_module(recovery)
    try:
        module._validate_r3_probe_failure_signature(
            classification="offline_clone_unseeded_release_local_cache",
            returncode=1,
            stdout=stdout,
            stderr=stderr,
            contract={},
        )
    except module.EvidenceError as exc:
        if str(exc) != EXPECTED_R9_REJECTION:
            raise R9FailureSealError(
                f"r9 validator rejection drifted: {exc}"
            ) from exc
        return {
            "rejected": True,
            "error_class": "EvidenceError",
            "error": str(exc),
        }
    raise R9FailureSealError("r9 validator unexpectedly accepted the diagnostic")


def _remove_write_bits(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            continue
        os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o222)


def _execute_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    reproduced_state, recorder_transcript = _reproduce_volatile_probe_tree(
        recovery
    )
    transcript_path = evidence / RECORDER_TRANSCRIPT_NAME
    _publish(transcript_path, _canonical_bytes(recorder_transcript))
    archive = evidence / ARCHIVE_NAME
    shutil.copytree(R9_PROBE_ROOT, archive, symlinks=True, copy_function=shutil.copy2)
    if _portable_inventory(archive) != reproduced_state["inventory"]:
        raise R9FailureSealError("archived reproduced r9 probe tree drifted")
    contract = intent["reproduction"]
    reproduction = Path(contract["root"])
    reproduction.mkdir(mode=0o700)
    Path(contract["cache"]).mkdir(mode=0o700)
    completed = subprocess.run(
        list(contract["argv"]),
        cwd=contract["cwd"],
        env=dict(contract["environment"]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
        check=False,
    )
    if completed.returncode != 1:
        raise R9FailureSealError(
            f"equivalent r9 offline probe returned {completed.returncode}"
        )
    stdout = completed.stdout.decode("utf-8", errors="strict")
    stderr = completed.stderr.decode("utf-8", errors="strict")
    urls = _validate_structural_diagnostic(
        stdout=stdout, stderr=stderr, contract=contract
    )
    rejection = _r9_rejection(recovery, stdout=stdout, stderr=stderr)
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    _publish(stdout_path, completed.stdout)
    _publish(stderr_path, completed.stderr)
    _remove_write_bits(R9_PROBE_ROOT)
    _remove_write_bits(archive)
    _remove_write_bits(reproduction)
    _require_recursively_read_only(R9_PROBE_ROOT)
    _require_recursively_read_only(archive)
    _require_recursively_read_only(reproduction)
    return {
        "returncode": completed.returncode,
        "stdout": _file_ref(stdout_path, description="r9 Conda stdout"),
        "stderr": _file_ref(stderr_path, description="r9 Conda stderr"),
        "remote_fetch_count": len(urls),
        "remote_fetch_urls": urls,
        "structural_diagnostic_valid": True,
        "immutable_r9_validator": rejection,
        "historical_recorder_transcript": _file_ref(
            transcript_path,
            description="exact r9 recorder reproduction transcript",
        )
        | {"transcript_id": recorder_transcript["transcript_id"]},
        "reproduced_probe_state": reproduced_state,
        "original_probe_archive": {
            "path": str(archive),
            "inventory": _portable_inventory(archive),
            "recursively_read_only": True,
        },
        "equivalent_reproduction": {
            "path": str(reproduction),
            "inventory": _portable_inventory(reproduction),
            "recursively_read_only": True,
        },
        "recreated_volatile_source_sealed_read_only": True,
        "offline_envelope_published": False,
        "known_scheduler_job_ids": [],
        "result_mutation_count": 0,
    }


def _verify_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    expected_keys = {
        "returncode",
        "stdout",
        "stderr",
        "remote_fetch_count",
        "remote_fetch_urls",
        "structural_diagnostic_valid",
        "immutable_r9_validator",
        "historical_recorder_transcript",
        "reproduced_probe_state",
        "original_probe_archive",
        "equivalent_reproduction",
        "recreated_volatile_source_sealed_read_only",
        "offline_envelope_published",
        "known_scheduler_job_ids",
        "result_mutation_count",
    }
    if (
        set(proof) != expected_keys
        or proof.get("returncode") != 1
        or proof.get("structural_diagnostic_valid") is not True
        or proof.get("immutable_r9_validator")
        != {
            "rejected": True,
            "error_class": "EvidenceError",
            "error": EXPECTED_R9_REJECTION,
        }
        or proof.get("recreated_volatile_source_sealed_read_only") is not True
        or proof.get("offline_envelope_published") is not False
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
    ):
        raise R9FailureSealError("r9 failure proof drifted")
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    stdout_ref = _file_ref(stdout_path, description="sealed r9 Conda stdout")
    stderr_ref = _file_ref(stderr_path, description="sealed r9 Conda stderr")
    if proof.get("stdout") != stdout_ref or proof.get("stderr") != stderr_ref:
        raise R9FailureSealError("r9 diagnostic file binding drifted")
    transcript_path = evidence / RECORDER_TRANSCRIPT_NAME
    transcript, transcript_raw = _read_json(
        transcript_path,
        description="exact r9 recorder reproduction transcript",
    )
    transcript_id = transcript.get("transcript_id")
    if (
        transcript.get("schema_version") != 1
        or transcript.get("protocol")
        != f"{PROTOCOL}-exact-r9-recorder-reproduction"
        or transcript.get("source_release") != R9_TAG
        or transcript.get("volatile_root_recreated_from_empty") is not True
        or transcript_id != _identity(transcript, "transcript_id")
        or proof.get("historical_recorder_transcript")
        != {
            "path": str(transcript_path),
            "sha256": hashlib.sha256(transcript_raw).hexdigest(),
            "size": len(transcript_raw),
            "transcript_id": transcript_id,
        }
        or proof.get("reproduced_probe_state")
        != transcript.get("resulting_probe_state")
    ):
        raise R9FailureSealError("exact r9 recorder reproduction binding drifted")
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    urls = _validate_structural_diagnostic(
        stdout=stdout,
        stderr=stderr,
        contract=intent["reproduction"],
    )
    if (
        proof.get("remote_fetch_count") != len(urls)
        or proof.get("remote_fetch_urls") != urls
    ):
        raise R9FailureSealError("r9 remote-fetch evidence drifted")
    archive = evidence / ARCHIVE_NAME
    reproduction = evidence / REPRODUCTION_NAME
    if proof.get("original_probe_archive") != {
        "path": str(archive),
        "inventory": _portable_inventory(archive),
        "recursively_read_only": True,
    }:
        raise R9FailureSealError("r9 original probe archive drifted")
    if (
        proof["original_probe_archive"]["inventory"]
        != proof["reproduced_probe_state"]["inventory"]
    ):
        raise R9FailureSealError("r9 reproduced probe archive/source mismatch")
    if proof.get("equivalent_reproduction") != {
        "path": str(reproduction),
        "inventory": _portable_inventory(reproduction),
        "recursively_read_only": True,
    }:
        raise R9FailureSealError("r9 equivalent reproduction drifted")
    _require_recursively_read_only(archive)
    _require_recursively_read_only(reproduction)


def _binding(
    evidence: Path, marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R9_TAG,
        "release_git_commit": R9_COMMIT,
        "chain_namespace": R9_NAMESPACE,
        "marker": str(evidence / MARKER_NAME),
        "marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "marker_size": len(marker_raw),
        "marker_id": marker["marker_id"],
        "classification": CLASSIFICATION,
        "retry_in_place": False,
        "requires_superseding_release": True,
        "pre_scheduler_submission": True,
        "known_scheduler_job_ids": [],
    }


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(evidence_root, description="r9 failure evidence")
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r9 prelaunch failure marker"
    )
    if (
        set(marker)
        != {
            "schema_version",
            "protocol",
            "classification",
            "intent",
            "proof",
            "marker_id",
        }
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("classification") != CLASSIFICATION
        or marker.get("marker_id") != _identity(marker, "marker_id")
    ):
        raise R9FailureSealError("r9 prelaunch failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r9 prelaunch failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R9FailureSealError("r9 failure scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R9FailureSealError("r9 failure marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R9FailureSealError("r9 failure marker proof is malformed")
    _verify_proof(evidence, recovery, intent, proof)
    _require_recursively_read_only(evidence)
    return _binding(evidence, marker, marker_raw)


def seal_failure(
    *,
    evidence_root: str | Path,
    recovery_root: str | Path,
    scheduler_user: str,
    apply: bool,
) -> dict[str, Any]:
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    expected_root = recovery / EVIDENCE_RELATIVE_ROOT
    evidence = Path(evidence_root).expanduser().absolute()
    if evidence != expected_root:
        raise R9FailureSealError(f"r9 failure evidence root must be {expected_root}")
    marker_path = evidence / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        return {
            "action": "already_sealed",
            **verify_failure_seal(
                evidence,
                recovery_root=recovery,
                scheduler_user=scheduler_user,
            ),
        }
    source = _validate_volatile_probe_absence()
    intent = _intent_payload(recovery, evidence, scheduler_user, source)
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "original_probe_state": source,
            "reproduction": intent["reproduction"],
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R9FailureSealError(
            "incomplete r9 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _execute_proof(evidence, recovery, intent)
    marker: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "classification": CLASSIFICATION,
        "intent": {
            "path": str(intent_path),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
            "size": len(intent_raw),
            "intent_id": intent["intent_id"],
        },
        "proof": proof,
    }
    marker["marker_id"] = _identity(marker, "marker_id")
    _publish(marker_path, _canonical_bytes(marker))
    os.chmod(evidence, 0o555)
    _fsync_directory(evidence.parent)
    return {
        "action": "sealed",
        **verify_failure_seal(
            evidence,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subcommand", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--recovery-root", type=Path, required=True)
    seal.add_argument("--scheduler-user", required=True)
    seal.add_argument("--apply", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--recovery-root", type=Path, required=True)
    verify.add_argument("--scheduler-user")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.subcommand == "seal":
            report = seal_failure(
                evidence_root=args.evidence_root,
                recovery_root=args.recovery_root,
                scheduler_user=args.scheduler_user,
                apply=args.apply,
            )
        else:
            report = verify_failure_seal(
                args.evidence_root,
                recovery_root=args.recovery_root,
                scheduler_user=args.scheduler_user,
            )
    except (
        R9FailureSealError,
        *MarkerLastEvidenceError,
        OSError,
        UnicodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
