#!/usr/bin/env python3
"""Freeze and verify the immutable schema-5 production release identity.

This tool deliberately does not create a Git worktree, install an environment, or
submit scheduler jobs.  It seals identities that an operator has already materialized:

* a clean worktree at the exact operational retry tag
  ``sweep-recovery-schema5-v1.2-r2``;
* harness and serving Conda prefixes, including reproducible Conda/pip locks;
* the frozen Qwen model/tokenizer contract; and
* the canonical 22-replica/24-GPU fleet contract.

The environment SHA values consumed by :mod:`slurm.schema5_control` are the SHA-256
digests of the exact environment-manifest *files*.  Each such manifest independently
contains a full, sorted directory inventory and its digest.  The release source digest
is computed by calling ``schema5_control.sha256_tree`` itself, so the freezer and the
live control plane cannot silently acquire different tree-hash semantics.
Conda locks are reconstructed only from the sealed ``conda-meta/*.json`` records;
the freezer never locates or invokes a Conda executable.

``create`` is a dry run unless ``--apply`` is supplied.  Publication writes all
artifacts atomically, verifies them from disk, and creates ``RELEASE_COMPLETE.json``
last.  Once that marker exists, all invocations are verification-only.  Optional
read-only sealing is explicit because it mutates the supplied worktree/environment
permissions and may require a new physical copy to undo.
"""

from __future__ import annotations

import argparse
import csv
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.serving.model_contracts import (  # noqa: E402
    FrozenModelContracts,
    load_model_contracts,
    offline_environment,
)
from agents_scaling.serving.fleet_contract import (  # noqa: E402
    expected_replica_id,
    expected_scheduler_job_name,
)
from agents_scaling.serving import scheduler_safety  # noqa: E402
from agents_scaling.runtime_integrity import (  # noqa: E402
    RuntimeIntegrityError,
    directory_inventory as _runtime_directory_inventory,
)
from slurm.schema5_control import sha256_tree  # noqa: E402


RELEASE_SCHEMA_VERSION = 4
ENVIRONMENT_SCHEMA_VERSION = 3
FLEET_SCHEMA_VERSION = 1
RELEASE_ID = "sweep-recovery-schema5-v1.2"
REQUIRED_GIT_TAG = "sweep-recovery-schema5-v1.2-r2"
FLEET_ID = "schema5-v1"
HARNESS_MANIFEST_FILENAME = "harness_environment.schema5-v1.json"
SERVING_MANIFEST_FILENAME = "serving_environment.schema5-v1.json"
RELEASE_IDENTITY_FILENAME = "release_identity.schema5-v1.json"
COMPLETE_MARKER_FILENAME = "RELEASE_COMPLETE.json"
MATERIALIZATION_COMPLETE_FILENAME = "MATERIALIZATION_COMPLETE.json"
MATERIALIZATION_STAGE_FILENAMES = {
    "worktree": "WORKTREE_MATERIALIZED.json",
    "harness_clone": "HARNESS_CLONE_COMPLETE.json",
    "serving_clone": "SERVING_CLONE_COMPLETE.json",
    "package_cache": "CONDA_PACKAGE_CACHE_COMPLETE.json",
    "harness_package": "HARNESS_PACKAGE_COMPLETE.json",
}
CHECKSUM_SUFFIX = ".sha256"
REQUIRED_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
EXPECTED_PROFILE_LAYOUT = {
    "0.6B": ("0.6B", 2, 1, 32_768),
    "1.7B": ("1.7B", 2, 1, 32_768),
    "4B": ("4B", 2, 1, 32_768),
    "8B": ("8B", 3, 1, 32_768),
    "14B": ("14B", 2, 1, 32_768),
    "32B": ("32B", 4, 1, 16_384),
    "0.6B-long": ("0.6B", 1, 1, 40_960),
    "1.7B-long": ("1.7B", 1, 1, 40_960),
    "4B-long": ("4B", 1, 1, 40_960),
    "8B-long": ("8B", 1, 1, 40_960),
    "14B-long": ("14B", 1, 1, 40_960),
    "32B-long": ("32B", 2, 2, 40_960),
}
EXPECTED_LOGICAL_REPLICAS = 22
EXPECTED_GPUS = 24
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PINNED_REQUIREMENT_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?==[^\s=]+\Z"
)
_DIRECT_REQUIREMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)\s+@\s+(?P<url>\S+)\Z"
)
_RELEASE_PACKAGE_NAME = "agents-scaling"
_CHUNK_SIZE = 8 * 1024 * 1024


class ReleaseFreezeError(RuntimeError):
    """A release identity cannot be proven immutable and internally consistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Some shared filesystems do not implement directory fsync.  The file
            # itself is still fsynced before rename.
            pass
    finally:
        os.close(descriptor)


def _atomic_write_exact(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Publish exact bytes once; never overwrite a conflicting preimage."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ReleaseFreezeError(f"conflicting immutable release artifact: {path}")
        os.chmod(path, mode)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".freezing", dir=path.parent
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


def _checksum_bytes(filename: str, payload: bytes) -> bytes:
    return f"{_sha256_bytes(payload)}  {filename}\n".encode("utf-8")


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseFreezeError(f"missing regular {description}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseFreezeError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseFreezeError(f"{description} must contain one JSON object: {path}")
    return value


def _run(argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            env=None if env is None else dict(env),
        )
    except OSError as exc:
        raise ReleaseFreezeError(f"cannot execute {argv[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise ReleaseFreezeError(
            f"command failed ({completed.returncode}): {argv!r}: {detail}"
        )
    return completed.stdout


def _python_probe_environment() -> dict[str, str]:
    blocked = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
    environment = {
        key: value for key, value in os.environ.items() if key not in blocked
    }
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _resolve_directory(path: str | Path, *, description: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ReleaseFreezeError(f"missing or symlinked {description}: {candidate}")
    resolved = candidate.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ReleaseFreezeError(f"refusing unsafe broad {description}: {resolved}")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_nonoverlap(output_root: Path, observed_roots: Iterable[Path]) -> None:
    # An observed root may live below a release bundle root, but the generated
    # artifacts must never live *inside* a hashed worktree/environment.
    for observed in observed_roots:
        if output_root == observed or _is_relative_to(output_root, observed):
            raise ReleaseFreezeError(
                f"release artifact root {output_root} is inside inventoried tree {observed}"
            )


def _git(worktree: Path, *args: str) -> str:
    return _run(("git", "-C", str(worktree), *args)).strip()


def verify_clean_exact_tag(worktree: Path) -> dict[str, str]:
    """Return a Git identity only for the clean exact production tag."""

    top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
    if top_level != worktree:
        raise ReleaseFreezeError(
            f"release worktree must be its Git top-level: {worktree} != {top_level}"
        )
    commit = _git(worktree, "rev-parse", "HEAD")
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ReleaseFreezeError(f"Git HEAD is not an exact commit: {commit!r}")
    status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ReleaseFreezeError("release worktree is not clean")
    ignored = _git(
        worktree, "ls-files", "--others", "--ignored", "--exclude-standard"
    ).splitlines()
    ignored_cache_names = {".pytest_cache", "__pycache__"}
    influential_ignored = [
        value
        for value in ignored
        if value
        and not any(part in ignored_cache_names for part in Path(value).parts)
        and Path(value).suffix != ".pyc"
    ]
    if influential_ignored:
        raise ReleaseFreezeError(
            "release worktree contains ignored files included by the source hash: "
            + ", ".join(influential_ignored[:5])
        )
    try:
        tag_commit = _git(
            worktree,
            "rev-parse",
            "--verify",
            f"refs/tags/{REQUIRED_GIT_TAG}^{{commit}}",
        )
    except ReleaseFreezeError as exc:
        raise ReleaseFreezeError(
            f"required exact Git tag is absent: {REQUIRED_GIT_TAG}"
        ) from exc
    if tag_commit != commit:
        raise ReleaseFreezeError(
            f"tag {REQUIRED_GIT_TAG} resolves to {tag_commit}, but HEAD is {commit}"
        )
    return {
        "git_commit": commit,
        "git_tag": REQUIRED_GIT_TAG,
        # Do not reproduce this algorithm here.  The control plane owns it.
        "source_tree_sha256": sha256_tree(worktree),
    }


def _iter_paths(root: Path) -> list[Path]:
    paths: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ReleaseFreezeError(f"cannot inventory directory {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            paths.append(path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ReleaseFreezeError(
                        f"special file is forbidden in immutable environment: {path}"
                    )
            except OSError as exc:
                raise ReleaseFreezeError(f"cannot inspect environment entry {path}: {exc}") from exc

    visit(root)
    return paths


def _tree_signatures(root: Path) -> tuple[tuple[Any, ...], ...]:
    signatures: list[tuple[Any, ...]] = []
    for path in _iter_paths(root):
        try:
            info = path.lstat()
        except OSError as exc:
            raise ReleaseFreezeError(f"cannot stat environment entry {path}: {exc}") from exc
        relative = path.relative_to(root).as_posix()
        file_type = stat.S_IFMT(info.st_mode)
        target = os.readlink(path) if stat.S_ISLNK(info.st_mode) else None
        signatures.append(
            (
                relative,
                file_type,
                stat.S_IMODE(info.st_mode),
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                target,
            )
        )
    return tuple(signatures)


def directory_inventory(root: Path) -> dict[str, Any]:
    """Build the canonical inventory also consumed by runtime attestation."""

    try:
        return _runtime_directory_inventory(root)
    except RuntimeIntegrityError as exc:
        raise ReleaseFreezeError(str(exc)) from exc


_RUNTIME_PROBE = r"""
# SCHEMA5_RUNTIME_PROBE
import importlib.metadata
import json
import platform

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

cuda = None
try:
    import torch
    cuda = torch.version.cuda
except Exception:
    pass

print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "cuda_version": cuda,
    "packages": {
        "torch": version("torch"),
        "vllm": version("vllm"),
        "transformers": version("transformers"),
        "tokenizers": version("tokenizers"),
    },
}, sort_keys=True))
""".strip()


_RELEASE_PACKAGE_IMPORT_PROBE = r"""
# SCHEMA5_RELEASE_IMPORT_PROBE
import importlib.metadata
import json
from pathlib import Path
import sys

prefix = Path(sys.prefix).resolve()
module = __import__("agents_scaling")
distribution = importlib.metadata.distribution("agents_scaling")
print(json.dumps({
    "prefix": str(prefix),
    "module_path": str(Path(module.__file__).resolve()),
    "version": distribution.version,
    "direct_url": json.loads(distribution.read_text("direct_url.json")),
}, sort_keys=True))
""".strip()


def _collect_runtime(prefix: Path, *, role: str) -> dict[str, Any]:
    python = prefix / "bin" / "python"
    if python.is_symlink() and not python.exists():
        raise ReleaseFreezeError(f"broken environment Python symlink: {python}")
    if not python.is_file():
        raise ReleaseFreezeError(f"environment Python is missing: {python}")
    raw = _run(
        (str(python), "-I", "-c", _RUNTIME_PROBE),
        env=_python_probe_environment(),
    )
    try:
        runtime = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReleaseFreezeError(f"invalid runtime probe from {python}: {exc}") from exc
    if not isinstance(runtime, dict) or set(runtime) != {
        "python_version",
        "python_implementation",
        "cuda_version",
        "packages",
    }:
        raise ReleaseFreezeError(f"runtime probe returned the wrong fields for {prefix}")
    packages = runtime.get("packages")
    if not isinstance(packages, dict) or set(packages) != {
        "torch",
        "vllm",
        "transformers",
        "tokenizers",
    }:
        raise ReleaseFreezeError(f"runtime package probe is incomplete for {prefix}")
    if not runtime["python_version"] or not runtime["python_implementation"]:
        raise ReleaseFreezeError(f"Python identity is missing for {prefix}")
    for package in ("transformers", "tokenizers"):
        if not packages[package]:
            raise ReleaseFreezeError(f"{role} environment lacks required {package}")
    if role == "serving":
        if packages["vllm"] != "0.21.0":
            raise ReleaseFreezeError(
                f"serving environment requires vLLM 0.21.0, found {packages['vllm']!r}"
            )
        if not runtime["cuda_version"]:
            raise ReleaseFreezeError("serving environment has no recorded CUDA runtime")
    return runtime


def _normalized_project_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _safe_prefix_entry(prefix: Path, relative: str, *, description: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ReleaseFreezeError(f"unsafe {description} path {relative!r}")
    resolved = (prefix / candidate).resolve()
    if not _is_relative_to(resolved, prefix):
        raise ReleaseFreezeError(f"{description} escapes environment prefix: {relative!r}")
    return resolved


def _conda_records(prefix: Path) -> dict[str, dict[str, Any]]:
    """Return exact Conda package records indexed by normalized project name."""

    records: dict[str, dict[str, Any]] = {}
    metadata_root = prefix / "conda-meta"
    for path in sorted(metadata_root.glob("*.json")):
        payload = _read_json_object(path, description="Conda package record")
        name = payload.get("name")
        version = payload.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ReleaseFreezeError(f"Conda package record lacks name/version: {path}")
        normalized = _normalized_project_name(name)
        if normalized in records:
            raise ReleaseFreezeError(
                f"duplicate normalized Conda package record {normalized!r} in {prefix}"
            )
        records[normalized] = payload
    return records


def _metadata_identity(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseFreezeError(f"distribution metadata is not a regular file: {path}")
    try:
        metadata = BytesParser().parsebytes(path.read_bytes(), headersonly=True)
    except (OSError, UnicodeError) as exc:
        raise ReleaseFreezeError(f"cannot parse distribution metadata {path}: {exc}") from exc
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ReleaseFreezeError(f"distribution metadata lacks Name/Version: {path}")
    return name, version


def _validate_conda_owned_direct_requirement(
    prefix: Path,
    *,
    requirement_name: str,
    requirement_url: str,
    record: Mapping[str, Any],
) -> str:
    """Canonicalize one local Conda-build URL after proving exact ownership.

    Recent Conda packages can contain ``direct_url.json`` files left by their build
    frontend, causing ``pip freeze`` to print an unavailable builder ``file://`` URL.
    That URL is not the reproduction source: the exact remote Conda artifact and its
    SHA-256 are.  We accept such a row only when the Conda record owns and hashes the
    matching dist-info metadata, then normalize it to ``name==version``.  The separate
    explicit Conda lock continues to bind the true artifact URL and digest.
    """

    record_url = record.get("url")
    record_sha256 = record.get("sha256")
    parsed_record_url = urlsplit(str(record_url))
    if (
        parsed_record_url.scheme not in {"https", "http"}
        or not parsed_record_url.netloc
        or _SHA256_RE.fullmatch(str(record_sha256)) is None
    ):
        raise ReleaseFreezeError(
            f"Conda provenance for direct requirement {requirement_name!r} is not exact"
        )
    paths = record.get("paths_data", {}).get("paths")
    if not isinstance(paths, list):
        raise ReleaseFreezeError(
            f"Conda provenance lacks paths_data for {requirement_name!r}"
        )
    normalized = _normalized_project_name(requirement_name)
    matches: list[tuple[Path, Path, str, str]] = []
    for item in paths:
        if not isinstance(item, dict) or not isinstance(item.get("_path"), str):
            continue
        relative = item["_path"]
        if not relative.endswith(".dist-info/direct_url.json"):
            continue
        direct_path = _safe_prefix_entry(
            prefix, relative, description="Conda-owned direct_url.json"
        )
        metadata_path = direct_path.with_name("METADATA")
        name, version = _metadata_identity(metadata_path)
        if _normalized_project_name(name) == normalized:
            matches.append((direct_path, metadata_path, name, version))
    if len(matches) != 1:
        raise ReleaseFreezeError(
            f"Conda record does not own exactly one direct URL for {requirement_name!r}"
        )
    direct_path, metadata_path, metadata_name, metadata_version = matches[0]
    if metadata_version != record.get("version"):
        raise ReleaseFreezeError(
            f"Conda/distribution version mismatch for {requirement_name!r}"
        )
    try:
        direct_payload = json.loads(
            direct_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseFreezeError(f"invalid Conda-owned direct URL {direct_path}: {exc}") from exc
    if not isinstance(direct_payload, dict) or direct_payload.get("url") != requirement_url:
        raise ReleaseFreezeError(
            f"pip/Conda direct URL mismatch for {requirement_name!r}"
        )
    # Validate every hashed regular file owned by this Conda package, not merely its
    # dist-info.  Otherwise stale conda-meta could bless a pip replacement that kept
    # the original metadata but changed importable package code.
    validated = 0
    validated_paths: set[str] = set()
    for item in paths:
        if not isinstance(item, dict):
            continue
        relative = item.get("_path")
        expected = item.get("sha256_in_prefix") or item.get("sha256")
        if (
            not isinstance(relative, str)
            or _SHA256_RE.fullmatch(str(expected)) is None
        ):
            continue
        candidate = _safe_prefix_entry(prefix, relative, description="Conda-owned file")
        lexical_candidate = prefix / relative
        if lexical_candidate.is_symlink():
            # Conda softlinks do not carry stable file-content hashes in the records
            # observed here.  If one ever does, resolving it must still remain inside
            # the prefix (already enforced above), but it is not counted as regular
            # ownership evidence.
            continue
        if not candidate.is_file():
            raise ReleaseFreezeError(f"Conda-owned file is absent or non-regular: {candidate}")
        if _sha256_file(candidate) != expected:
            raise ReleaseFreezeError(f"Conda-owned file drifted from package record: {candidate}")
        validated += 1
        validated_paths.add(relative)
    required_evidence = {
        metadata_path.relative_to(prefix).as_posix(),
        direct_path.relative_to(prefix).as_posix(),
    }
    if validated < 2 or not required_evidence.issubset(validated_paths):
        raise ReleaseFreezeError(
            f"Conda ownership evidence is incomplete for {requirement_name!r}"
        )
    return f"{metadata_name}=={metadata_version}"


def _local_file_url_path(value: str) -> Path:
    parsed = urlsplit(value)
    if parsed.scheme != "file" or parsed.query or parsed.fragment:
        raise ReleaseFreezeError(f"local release requirement has invalid URL: {value!r}")
    if parsed.netloc not in {"", "localhost"}:
        raise ReleaseFreezeError(f"local release URL has a remote authority: {value!r}")
    return Path(unquote(parsed.path)).resolve()


def _release_package_import_binding(
    prefix: Path, *, release_worktree: Path
) -> dict[str, str]:
    python = prefix / "bin" / "python"
    raw = _run(
        (str(python), "-I", "-c", _RELEASE_PACKAGE_IMPORT_PROBE),
        env=_python_probe_environment(),
    )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReleaseFreezeError(f"invalid isolated agents_scaling import probe: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "prefix",
        "module_path",
        "version",
        "direct_url",
    }:
        raise ReleaseFreezeError("isolated agents_scaling import probe returned wrong fields")
    observed_prefix = Path(str(payload["prefix"])).resolve()
    module_path = Path(str(payload["module_path"])).resolve()
    if observed_prefix != prefix or not _is_relative_to(module_path, prefix):
        raise ReleaseFreezeError(
            f"agents_scaling imports outside the frozen harness prefix: {module_path}"
        )
    if _is_relative_to(module_path, release_worktree):
        raise ReleaseFreezeError("agents_scaling import still resolves to the release source tree")
    direct = payload.get("direct_url")
    if (
        not isinstance(direct, dict)
        or set(direct) != {"url", "dir_info"}
        or not isinstance(direct.get("dir_info"), dict)
        or direct["dir_info"].get("editable", False) is not False
        or _local_file_url_path(str(direct.get("url", ""))) != release_worktree
    ):
        raise ReleaseFreezeError(
            "isolated agents_scaling import is editable or has the wrong source"
        )
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ReleaseFreezeError("isolated agents_scaling import lacks a version")
    return {
        "prefix": str(observed_prefix),
        "module_path": str(module_path),
        "version": payload["version"],
    }


def _installed_distribution_inventory(dist_info: Path, prefix: Path) -> tuple[str, int]:
    record_path = dist_info / "RECORD"
    if record_path.is_symlink() or not record_path.is_file():
        raise ReleaseFreezeError(f"installed release package lacks RECORD: {record_path}")
    entries: list[dict[str, Any]] = []
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReleaseFreezeError(f"cannot read installed package RECORD {record_path}: {exc}") from exc
    site_packages = dist_info.parent
    seen: set[str] = set()
    for row in rows:
        if not row or not row[0] or row[0] in seen:
            raise ReleaseFreezeError(f"invalid or duplicate installed RECORD row: {row!r}")
        seen.add(row[0])
        candidate = _safe_prefix_entry(
            prefix,
            (site_packages.relative_to(prefix) / row[0]).as_posix(),
            description="installed release package RECORD",
        )
        if candidate.is_symlink():
            entries.append(
                {
                    "path": candidate.relative_to(prefix).as_posix(),
                    "type": "symlink",
                    "target": os.readlink(candidate),
                }
            )
        elif candidate.is_file():
            entries.append(
                {
                    "path": candidate.relative_to(prefix).as_posix(),
                    "type": "file",
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
        else:
            raise ReleaseFreezeError(f"installed release package file is missing: {candidate}")
    entries.sort(key=lambda entry: entry["path"])
    return _sha256_bytes(_canonical_bytes(entries)), len(entries)


def _release_package_binding(
    prefix: Path,
    *,
    requirement_name: str,
    requirement_url: str,
    release_worktree: Path,
    git_identity: Mapping[str, str],
) -> dict[str, Any]:
    if _normalized_project_name(requirement_name) != _RELEASE_PACKAGE_NAME:
        raise ReleaseFreezeError(f"unexpected local release package {requirement_name!r}")
    if _local_file_url_path(requirement_url) != release_worktree:
        raise ReleaseFreezeError(
            "agents_scaling must be installed from the exact tagged release worktree"
        )
    pyproject_path = release_worktree / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise ReleaseFreezeError(f"cannot read release package identity: {exc}") from exc
    if (
        _normalized_project_name(str(project.get("name", ""))) != _RELEASE_PACKAGE_NAME
        or not isinstance(project.get("version"), str)
        or not project["version"]
    ):
        raise ReleaseFreezeError("tagged pyproject has the wrong agents_scaling identity")
    # Conda commonly creates compatibility aliases such as
    # ``lib/python3.1 -> python3.11``.  ``Path.glob("lib/python*/...")`` follows
    # those directory symlinks and would otherwise count one physical dist-info
    # directory twice.  Canonical paths deduplicate only aliases of the same
    # installation; a second physical dist-info directory remains a hard error.
    canonical_prefix = prefix.resolve(strict=True)
    matches_by_metadata: dict[Path, tuple[Path, str, str]] = {}
    for metadata_path in sorted(prefix.glob("lib/python*/site-packages/*.dist-info/METADATA")):
        try:
            canonical_metadata = metadata_path.resolve(strict=True)
            canonical_metadata.relative_to(canonical_prefix)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ReleaseFreezeError(
                f"installed distribution metadata escapes the environment: {metadata_path}"
            ) from exc
        name, version = _metadata_identity(canonical_metadata)
        if _normalized_project_name(name) == _RELEASE_PACKAGE_NAME:
            matches_by_metadata.setdefault(
                canonical_metadata,
                (canonical_metadata.parent, name, version),
            )
    matches = [matches_by_metadata[path] for path in sorted(matches_by_metadata)]
    if len(matches) != 1:
        raise ReleaseFreezeError(
            "harness must contain exactly one installed agents_scaling distribution"
        )
    dist_info, metadata_name, metadata_version = matches[0]
    if metadata_version != project["version"]:
        raise ReleaseFreezeError("installed agents_scaling version differs from tagged pyproject")
    direct_path = dist_info / "direct_url.json"
    direct = _read_json_object(direct_path, description="agents_scaling direct URL")
    if (
        direct.get("url") != requirement_url
        or set(direct) != {"url", "dir_info"}
        or not isinstance(direct.get("dir_info"), dict)
        or direct["dir_info"].get("editable", False) is not False
    ):
        raise ReleaseFreezeError(
            "agents_scaling install is editable or not bound to the release worktree"
        )
    inventory_sha256, installed_file_count = _installed_distribution_inventory(
        dist_info, prefix
    )
    import_binding = _release_package_import_binding(
        prefix, release_worktree=release_worktree
    )
    if import_binding["version"] != metadata_version:
        raise ReleaseFreezeError(
            "isolated agents_scaling import version differs from installed metadata"
        )
    return {
        "name": metadata_name,
        "version": metadata_version,
        "direct_url": requirement_url,
        "source_path": str(release_worktree),
        "source_git_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "installed_distribution_sha256": inventory_sha256,
        "installed_file_count": installed_file_count,
        "isolated_import_prefix": import_binding["prefix"],
        "isolated_import_path": import_binding["module_path"],
        "editable": False,
    }


def _pip_lock_material(
    prefix: Path,
    *,
    release_worktree: Path | None = None,
    git_identity: Mapping[str, str] | None = None,
    require_release_package: bool = False,
) -> tuple[list[str], dict[str, Any] | None]:
    python = prefix / "bin" / "python"
    output = _run(
        (str(python), "-I", "-m", "pip", "freeze", "--all", "--local"),
        env=_python_probe_environment(),
    )
    raw_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not raw_lines:
        raise ReleaseFreezeError(f"pip lock is empty for {prefix}")
    conda_records = _conda_records(prefix)
    normalized_lines: list[str] = []
    seen_projects: set[str] = set()
    release_binding: dict[str, Any] | None = None
    invalid: list[str] = []
    for line in raw_lines:
        if _PINNED_REQUIREMENT_RE.fullmatch(line) is not None:
            lhs = line.split("==", 1)[0].split("[", 1)[0]
            project = _normalized_project_name(lhs)
            if project == _RELEASE_PACKAGE_NAME:
                invalid.append(line)
                continue
            normalized = line
        else:
            direct_match = _DIRECT_REQUIREMENT_RE.fullmatch(line)
            if direct_match is None:
                invalid.append(line)
                continue
            name = direct_match.group("name")
            url = direct_match.group("url")
            project = _normalized_project_name(name)
            if urlsplit(url).scheme != "file":
                invalid.append(line)
                continue
            if (
                project == _RELEASE_PACKAGE_NAME
                and release_worktree is not None
                and git_identity is not None
                and release_binding is None
            ):
                release_binding = _release_package_binding(
                    prefix,
                    requirement_name=name,
                    requirement_url=url,
                    release_worktree=release_worktree,
                    git_identity=git_identity,
                )
                normalized = f"{release_binding['name']} @ {url}"
            elif project in conda_records:
                normalized = _validate_conda_owned_direct_requirement(
                    prefix,
                    requirement_name=name,
                    requirement_url=url,
                    record=conda_records[project],
                )
            else:
                invalid.append(line)
                continue
        if project in seen_projects:
            raise ReleaseFreezeError(
                f"pip lock contains duplicate project {project!r} for {prefix}"
            )
        seen_projects.add(project)
        normalized_lines.append(normalized)
    if invalid:
        raise ReleaseFreezeError(
            "pip lock contains editable, VCS, unowned direct, or non-exact requirements: "
            + ", ".join(repr(line) for line in invalid[:5])
        )
    if require_release_package and release_binding is None:
        raise ReleaseFreezeError(
            "harness pip lock lacks non-editable agents_scaling from the exact release"
        )
    if not require_release_package and release_binding is not None:
        raise ReleaseFreezeError("release package is forbidden in this environment role")
    normalized_lines.sort(key=str.casefold)
    return normalized_lines, release_binding


def _pip_lock(
    prefix: Path,
    *,
    release_worktree: Path | None = None,
    git_identity: Mapping[str, str] | None = None,
    require_release_package: bool = False,
) -> list[str]:
    """Return the canonical combined pip/Conda Python-distribution view."""

    lock, _binding = _pip_lock_material(
        prefix,
        release_worktree=release_worktree,
        git_identity=git_identity,
        require_release_package=require_release_package,
    )
    return lock


def _conda_lock_from_records(prefix: Path) -> list[str]:
    """Reconstruct an exact explicit lock from the prefix's sealed records.

    Schema-5 v1.2 never invokes Conda to derive a lock.  Every installed artifact's
    ``conda-meta/*.json`` record contains its exact remote URL and SHA-256, so those
    inventoried records are the sole lock authority.  The materializer separately
    records the clone-time executable as provenance without granting it a role in
    release freezing or sealed verification.
    """

    metadata_root = prefix / "conda-meta"
    if metadata_root.is_symlink() or not metadata_root.is_dir():
        raise ReleaseFreezeError(
            f"environment lacks a regular conda-meta directory: {prefix}"
        )
    packages: list[str] = []
    seen_artifacts: set[str] = set()
    record_paths = sorted(metadata_root.glob("*.json"))
    if not record_paths:
        raise ReleaseFreezeError(f"Conda package records are empty for {prefix}")
    for record_path in record_paths:
        record = _read_json_object(record_path, description="Conda package record")
        name = record.get("name")
        version = record.get("version")
        url = record.get("url")
        package_sha256 = record.get("sha256")
        parsed = urlsplit(str(url))
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not isinstance(url, str)
            or parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
            or _SHA256_RE.fullmatch(str(package_sha256)) is None
        ):
            raise ReleaseFreezeError(
                f"Conda package record lacks exact remote provenance: {record_path}"
            )
        artifact = f"{url}#{package_sha256}"
        if artifact in seen_artifacts:
            raise ReleaseFreezeError(
                f"duplicate Conda artifact provenance in {prefix}: {artifact!r}"
            )
        seen_artifacts.add(artifact)
        packages.append(artifact)
    return ["@EXPLICIT", *sorted(packages)]


def _environment_manifest(
    prefix: Path,
    *,
    role: str,
    sealed_read_only: bool,
    release_worktree: Path,
    git_identity: Mapping[str, str],
    materialization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not (prefix / "conda-meta").is_dir():
        raise ReleaseFreezeError(f"environment is not a materialized Conda prefix: {prefix}")
    runtime = _collect_runtime(prefix, role=role)
    pip_lock, release_package = _pip_lock_material(
        prefix,
        release_worktree=release_worktree,
        git_identity=git_identity,
        require_release_package=role == "harness",
    )
    record_lock = _conda_lock_from_records(prefix)
    inventory = directory_inventory(prefix)
    capture_binding = materialization_binding.get("environment_capture")
    conda_creation_tool = materialization_binding.get("conda_creation_tool")
    if (
        not isinstance(capture_binding, Mapping)
        or not isinstance(conda_creation_tool, Mapping)
        or set(conda_creation_tool) != {"path", "sha256"}
        or _SHA256_RE.fullmatch(str(conda_creation_tool.get("sha256", ""))) is None
    ):
        raise ReleaseFreezeError(
            "materialization lacks sealed environment/tool provenance"
        )
    capture_stages = capture_binding.get("stage_records")
    seed_prefixes = capture_binding.get("seed_prefixes")
    if (
        not isinstance(capture_stages, Mapping)
        or not isinstance(seed_prefixes, Mapping)
        or role not in capture_stages
        or role not in seed_prefixes
    ):
        raise ReleaseFreezeError(f"materialization lacks {role} seed provenance")
    seed_stage = capture_stages[role]
    if not isinstance(seed_stage, Mapping):
        raise ReleaseFreezeError(f"invalid {role} seed provenance")
    installed_files = {
        "inventory_sha256": inventory["inventory_sha256"],
        "entry_count": inventory["entry_count"],
        "file_count": inventory["file_count"],
        "total_file_bytes": inventory["total_file_bytes"],
    }
    payload: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "role": role,
        "prefix": str(prefix),
        "sealed_read_only": sealed_read_only,
        "offline_environment": dict(REQUIRED_OFFLINE_ENVIRONMENT),
        "conda_creation_tool": dict(conda_creation_tool),
        "environment_seed": {
            "capture_id": capture_binding["capture_id"],
            "capture_marker_sha256": capture_binding["capture_marker_sha256"],
            "prefix": seed_prefixes[role],
            "normalized_content_inventory_sha256": seed_stage[
                "normalized_content_inventory_sha256"
            ],
        },
        "ownership_policy": {
            "path": capture_binding["ownership_policy_path"],
            "sha256": capture_binding["ownership_policy_sha256"],
        },
        "integrity_normalization_policy": {
            "path": capture_binding[
                "integrity_normalization_policy_path"
            ],
            "sha256": capture_binding[
                "integrity_normalization_policy_sha256"
            ],
        },
        "normalization_receipt": {
            "id": seed_stage["normalization_receipt_id"],
        },
        "conda_package_cache_sha256": materialization_binding[
            "conda_package_cache_sha256"
        ],
        "runtime": runtime,
        "locks": {
            "conda_explicit": record_lock,
            "pip_freeze_all": pip_lock,
        },
        "release_package": release_package,
        "installed_files": installed_files,
        "directory_inventory": inventory,
    }
    payload["environment_content_sha256"] = _sha256_bytes(
        _canonical_bytes(
            {
                "runtime": runtime,
                "locks": payload["locks"],
                "release_package": release_package,
                "environment_seed": payload["environment_seed"],
                "ownership_policy": payload["ownership_policy"],
                "integrity_normalization_policy": payload[
                    "integrity_normalization_policy"
                ],
                "normalization_receipt": payload["normalization_receipt"],
                "conda_creation_tool": payload["conda_creation_tool"],
                "conda_package_cache_sha256": payload[
                    "conda_package_cache_sha256"
                ],
                "inventory_sha256": inventory["inventory_sha256"],
            }
        )
    )
    return payload


def _verified_checksum(path: Path, *, expected_filename: str) -> str:
    # Checked-in contracts follow the repository convention
    # ``contract.v1.json`` + ``contract.v1.sha256``.
    checksum_path = path.with_suffix(CHECKSUM_SUFFIX)
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ReleaseFreezeError(f"missing checksum for {path}")
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    observed = _sha256_file(path)
    if (
        len(fields) != 2
        or _SHA256_RE.fullmatch(fields[0]) is None
        or fields[1] != expected_filename
        or fields[0] != observed
    ):
        raise ReleaseFreezeError(f"checksum does not bind exact bytes of {path}")
    return observed


def load_and_validate_fleet_contract(
    path: str | Path,
    *,
    model_contracts: FrozenModelContracts,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the canonical pool and return its payload and exact-byte digest."""

    fleet_path = Path(path).expanduser().resolve()
    payload = _read_json_object(fleet_path, description="fleet contract")
    observed_sha256 = _verified_checksum(
        fleet_path, expected_filename=fleet_path.name
    )
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ReleaseFreezeError(
            f"fleet-contract drift: expected {expected_sha256}, got {observed_sha256}"
        )
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
    if set(payload) != required_root:
        raise ReleaseFreezeError("fleet contract has the wrong root fields")
    if payload["schema_version"] != FLEET_SCHEMA_VERSION:
        raise ReleaseFreezeError("unsupported fleet-contract schema")
    if payload["fleet_id"] != FLEET_ID or payload["release_id"] != RELEASE_ID:
        raise ReleaseFreezeError("fleet/release identity mismatch")
    if payload["model_contract_sha256"] != model_contracts.sha256:
        raise ReleaseFreezeError("fleet contract uses a different model contract")
    if payload["offline_environment"] != REQUIRED_OFFLINE_ENVIRONMENT:
        raise ReleaseFreezeError("fleet contract does not fail closed in offline mode")
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
        raise ReleaseFreezeError("fleet server-pool isolation contract is not exact")
    profiles = payload["profiles"]
    if not isinstance(profiles, list):
        raise ReleaseFreezeError("fleet profiles must be an array")
    by_name: dict[str, dict[str, Any]] = {}
    all_replica_ids: set[str] = set()
    all_job_names: set[str] = set()
    logical_count = 0
    gpu_count = 0
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ReleaseFreezeError("fleet profile must be an object")
        required_profile = {
            "serving_profile",
            "model_size",
            "hf_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "served_model_name",
            "tensor_parallel_size",
            "effective_context_limit",
            "gpus_per_replica",
            "replicas",
        }
        if set(profile) != required_profile:
            raise ReleaseFreezeError("fleet profile has the wrong fields")
        name = profile["serving_profile"]
        if not isinstance(name, str) or name in by_name:
            raise ReleaseFreezeError(f"duplicate or invalid serving profile {name!r}")
        by_name[name] = profile
        if name not in EXPECTED_PROFILE_LAYOUT:
            raise ReleaseFreezeError(f"unexpected serving profile {name!r}")
        model_size, expected_count, expected_tp, expected_context = EXPECTED_PROFILE_LAYOUT[name]
        identity = model_contracts.for_size(model_size)
        wanted_identity = {
            "model_size": model_size,
            "hf_id": identity.hf_id,
            "model_revision": identity.model_revision,
            "tokenizer_id": identity.tokenizer_id,
            "tokenizer_revision": identity.tokenizer_revision,
            "served_model_name": model_size,
            "tensor_parallel_size": expected_tp,
            "effective_context_limit": expected_context,
            "gpus_per_replica": expected_tp,
        }
        for field, expected in wanted_identity.items():
            if profile[field] != expected:
                raise ReleaseFreezeError(
                    f"fleet profile {name} drifted at {field}: "
                    f"expected {expected!r}, got {profile[field]!r}"
                )
        replicas = profile["replicas"]
        if not isinstance(replicas, list) or len(replicas) != expected_count:
            raise ReleaseFreezeError(
                f"fleet profile {name} requires {expected_count} logical replicas"
            )
        observed_indices: set[int] = set()
        for replica in replicas:
            if not isinstance(replica, dict) or set(replica) != {
                "replica_id",
                "replica_index",
                "scheduler_job_name",
                "pool_id",
                "partition",
                "qos",
                "gpu_type",
                "time_limit",
                "cpus_per_task",
                "memory",
            }:
                raise ReleaseFreezeError(f"fleet profile {name} has invalid replica identity")
            replica_id = replica["replica_id"]
            job_name = replica["scheduler_job_name"]
            index = replica["replica_index"]
            if replica["pool_id"] != FLEET_ID:
                raise ReleaseFreezeError("replica identity is not scoped to schema5-v1")
            if type(index) is not int or index < 0 or index in observed_indices:
                raise ReleaseFreezeError(f"invalid replica index for profile {name}")
            if replica_id != expected_replica_id(name, index):
                raise ReleaseFreezeError(
                    f"replica ID drifted for {name} r{index}: {replica_id!r}"
                )
            if job_name != expected_scheduler_job_name(name, index):
                raise ReleaseFreezeError(
                    f"scheduler job name drifted for {name} r{index}: {job_name!r}"
                )
            if replica_id in all_replica_ids or job_name in all_job_names:
                raise ReleaseFreezeError("fleet replica/job identities must be globally unique")
            expected_resources = {
                "gpu_type": "a100",
                "time_limit": "1-00:00:00",
                "cpus_per_task": expected_tp * 8,
                "memory": f"{expected_tp * 120}G",
            }
            if (
                any(
                    replica[field] != value
                    for field, value in expected_resources.items()
                )
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                    str(replica.get("partition", "")),
                )
                is None
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                    str(replica.get("qos", "")),
                )
                is None
            ):
                raise ReleaseFreezeError(
                    f"fleet replica placement drifted for {name} r{index}: "
                    "explicit partition/QOS and frozen resources are required"
                )
            observed_indices.add(index)
            all_replica_ids.add(replica_id)
            all_job_names.add(job_name)
        if observed_indices != set(range(expected_count)):
            raise ReleaseFreezeError(f"profile {name} replica indices are not contiguous")
        logical_count += len(replicas)
        gpu_count += len(replicas) * expected_tp
    if set(by_name) != set(EXPECTED_PROFILE_LAYOUT):
        raise ReleaseFreezeError("fleet does not contain the exact required profile set")
    if (
        logical_count != EXPECTED_LOGICAL_REPLICAS
        or payload["logical_replica_count"] != EXPECTED_LOGICAL_REPLICAS
    ):
        raise ReleaseFreezeError("fleet must contain exactly 22 logical replicas")
    if gpu_count != EXPECTED_GPUS or payload["allocated_gpu_count"] != EXPECTED_GPUS:
        raise ReleaseFreezeError("fleet must allocate exactly 24 GPUs")
    return payload, observed_sha256


def _seal_tree_read_only(root: Path) -> None:
    """Remove write bits without following symlinks or changing executable bits."""

    paths = _iter_paths(root)
    # Files first and directories deepest-first keep traversal possible throughout.
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            continue
        # ``follow_symlinks=False`` is unavailable for chmod on some Linux/Python
        # builds.  We have just rejected symlinks via lstat, so the ordinary call is
        # safe for this exact resolved entry.
        os.chmod(path, stat.S_IMODE(info.st_mode) & ~0o222)
    info = root.stat(follow_symlinks=False)
    os.chmod(root, stat.S_IMODE(info.st_mode) & ~0o222)


def _assert_read_only(root: Path) -> None:
    for path in [root, *_iter_paths(root)]:
        info = path.lstat()
        if not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o222:
            raise ReleaseFreezeError(f"sealed tree contains writable entry: {path}")


def _seal_output_root_read_only(root: Path) -> None:
    """Seal the publication directory after its marker has become durable."""

    os.chmod(root, stat.S_IMODE(root.stat().st_mode) & ~0o222)
    _fsync_directory(root)
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise ReleaseFreezeError(f"failed to seal release artifact root: {root}")


def _verified_materialization_binding(
    *,
    output_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    serving_prefix: Path,
) -> dict[str, Any]:
    """Verify and bind the exact materialization proof adjacent to identity/.

    The release identity directory is a direct child of the materialization root.  A
    release cannot be frozen from arbitrary prefixes: the sibling marker and every
    stage/live check owned by the materializer must validate first.
    """

    materialization_root = output_root.parent
    marker_path = materialization_root / MATERIALIZATION_COMPLETE_FILENAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ReleaseFreezeError(
            f"missing regular sibling materialization marker: {marker_path}"
        )
    try:
        # Local import avoids the intentional materializer -> freezer import at module
        # initialization while retaining one shared authoritative verifier.
        from scripts import materialize_schema5_release as materialize
    except ImportError as exc:
        raise ReleaseFreezeError(f"cannot load materialization verifier: {exc}") from exc
    try:
        report = materialize.verify_materialization(materialization_root)
    except materialize.MaterializationError as exc:
        raise ReleaseFreezeError(f"materialization proof is invalid: {exc}") from exc
    expected_paths = {
        "release_worktree": str(release_worktree),
        "harness_prefix": str(harness_prefix),
        "serving_prefix": str(serving_prefix),
    }
    if any(report.get("paths", {}).get(key) != value for key, value in expected_paths.items()):
        raise ReleaseFreezeError(
            "materialization proof belongs to different release inputs"
        )
    marker = _read_json_object(marker_path, description="materialization completion marker")
    stage_records = marker.get("stage_records")
    if (
        report.get("release_id") != RELEASE_ID
        or marker.get("release_id") != RELEASE_ID
        or marker.get("materialization_id") != report.get("materialization_id")
        or not isinstance(stage_records, dict)
    ):
        raise ReleaseFreezeError("materialization identity is inconsistent")
    return {
        "schema_version": marker.get("schema_version"),
        "release_id": RELEASE_ID,
        "root": str(materialization_root),
        "marker_path": str(marker_path),
        "marker_sha256": _sha256_file(marker_path),
        "materialization_id": report["materialization_id"],
        "tag_commit": report["tag_commit"],
        "source_tree_sha256": report["source_tree_sha256"],
        "paths": report["paths"],
        "stage_records": stage_records,
        "environment_capture": report["environment_capture"],
        "conda_creation_tool": report["conda_creation_tool"],
        "conda_package_cache_sha256": report["conda_package_cache_sha256"],
    }


def _verify_bound_materialization_evidence(
    binding: Mapping[str, Any],
    *,
    output_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    serving_prefix: Path,
    git_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Verify sealed materialization evidence without consulting mutable sources."""

    # Re-run the authoritative materialization verifier.  It reads only the
    # immutable capture seeds, release-local cache, tagged worktree, production
    # prefixes, and archived stage records; it never invokes Conda or consults the
    # retired developer prefixes/source checkout.  This prevents a collection of
    # individually self-hashed but semantically substituted stage records from
    # becoming a valid release provenance chain.
    materialization_root = output_root.parent
    marker_path = materialization_root / MATERIALIZATION_COMPLETE_FILENAME
    try:
        from scripts import materialize_schema5_release as materialize
    except ImportError as exc:
        raise ReleaseFreezeError(
            f"cannot load materialization verifier: {exc}"
        ) from exc
    try:
        report = materialize.verify_materialization(materialization_root)
    except materialize.MaterializationError as exc:
        raise ReleaseFreezeError(
            f"materialization proof is invalid: {exc}"
        ) from exc
    expected_paths = {
        "release_worktree": str(release_worktree),
        "harness_prefix": str(harness_prefix),
        "serving_prefix": str(serving_prefix),
    }
    if any(
        report.get("paths", {}).get(key) != value
        for key, value in expected_paths.items()
    ):
        raise ReleaseFreezeError(
            "materialization proof belongs to different release inputs"
        )
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ReleaseFreezeError(
            f"missing regular sibling materialization marker: {marker_path}"
        )
    marker = _read_json_object(
        marker_path, description="materialization completion marker"
    )
    stage_records = marker.get("stage_records")
    if (
        report.get("release_id") != RELEASE_ID
        or marker.get("release_id") != RELEASE_ID
        or marker.get("materialization_id")
        != report.get("materialization_id")
        or not isinstance(stage_records, dict)
    ):
        raise ReleaseFreezeError("materialization identity is inconsistent")
    live_binding = {
        "schema_version": marker.get("schema_version"),
        "release_id": RELEASE_ID,
        "root": str(materialization_root),
        "marker_path": str(marker_path),
        "marker_sha256": _sha256_file(marker_path),
        "materialization_id": report["materialization_id"],
        "tag_commit": report["tag_commit"],
        "source_tree_sha256": report["source_tree_sha256"],
        "paths": report["paths"],
        "stage_records": stage_records,
        "environment_capture": report["environment_capture"],
        "conda_creation_tool": report["conda_creation_tool"],
        "conda_package_cache_sha256": report[
            "conda_package_cache_sha256"
        ],
    }
    if (
        stat.S_IMODE(marker_path.stat().st_mode) & 0o222
        or _sha256_file(marker_path) != binding.get("marker_sha256")
        or not isinstance(stage_records, dict)
        or set(stage_records) != set(MATERIALIZATION_STAGE_FILENAMES)
    ):
        raise ReleaseFreezeError(
            "sealed materialization marker/stage inventory drifted"
        )
    for stage, filename in MATERIALIZATION_STAGE_FILENAMES.items():
        record = stage_records[stage]
        stage_path = materialization_root / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"filename", "sha256", "record_sha256"}
            or record.get("filename") != filename
            or stage_path.is_symlink()
            or not stage_path.is_file()
            or stat.S_IMODE(stage_path.stat().st_mode) & 0o222
            or _sha256_file(stage_path) != record.get("sha256")
        ):
            raise ReleaseFreezeError(
                f"materialization stage artifact drifted: {stage}"
            )
    if dict(binding) != live_binding:
        raise ReleaseFreezeError(
            "sealed materialization binding differs from authoritative "
            "upstream evidence"
        )
    if (
        live_binding.get("tag_commit") != git_identity["git_commit"]
        or live_binding.get("source_tree_sha256")
        != git_identity["source_tree_sha256"]
    ):
        raise ReleaseFreezeError(
            "sealed materialization source-tree binding drifted"
        )
    return live_binding


def _build_release_material(
    *,
    output_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    serving_prefix: Path,
    model_contract_path: Path,
    fleet_contract_path: Path,
    worktree_sealed: bool,
    environments_sealed: bool,
    materialization_binding: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    git_identity = verify_clean_exact_tag(release_worktree)
    transport_binding = (
        scheduler_safety.expected_transport_uncertainty_binding()
    )
    transport_binding_sha256 = (
        scheduler_safety.transport_uncertainty_binding_sha256(
            transport_binding
        )
    )
    model_contracts = load_model_contracts(model_contract_path)
    if offline_environment() != REQUIRED_OFFLINE_ENVIRONMENT:
        raise ReleaseFreezeError("code-level offline environment contract drifted")
    fleet_payload, fleet_sha256 = load_and_validate_fleet_contract(
        fleet_contract_path, model_contracts=model_contracts
    )
    harness_manifest = _environment_manifest(
        harness_prefix,
        role="harness",
        sealed_read_only=environments_sealed,
        release_worktree=release_worktree,
        git_identity=git_identity,
        materialization_binding=materialization_binding,
    )
    serving_manifest = _environment_manifest(
        serving_prefix,
        role="serving",
        sealed_read_only=environments_sealed,
        release_worktree=release_worktree,
        git_identity=git_identity,
        materialization_binding=materialization_binding,
    )
    harness_bytes = _json_bytes(harness_manifest)
    serving_bytes = _json_bytes(serving_manifest)
    harness_path = output_root / HARNESS_MANIFEST_FILENAME
    serving_path = output_root / SERVING_MANIFEST_FILENAME
    identity: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "git": git_identity,
        "release_worktree": str(release_worktree),
        "worktree_sealed_read_only": worktree_sealed,
        "materialization": dict(materialization_binding),
        "offline_environment": dict(REQUIRED_OFFLINE_ENVIRONMENT),
        "transport_uncertainty": {
            "binding": transport_binding,
            "binding_sha256": transport_binding_sha256,
            "source_tree_sha256": git_identity["source_tree_sha256"],
        },
        "model_contract": {
            "path": str(model_contracts.path),
            "sha256": model_contracts.sha256,
        },
        "fleet_contract": {
            "path": str(fleet_contract_path),
            "sha256": fleet_sha256,
            "fleet_id": fleet_payload["fleet_id"],
            "logical_replica_count": fleet_payload["logical_replica_count"],
            "allocated_gpu_count": fleet_payload["allocated_gpu_count"],
        },
        "environments": {
            "harness": {
                "prefix": str(harness_prefix),
                "manifest_path": str(harness_path),
                "manifest_sha256": _sha256_bytes(harness_bytes),
                "directory_inventory_sha256": harness_manifest[
                    "directory_inventory"
                ]["inventory_sha256"],
            },
            "serving": {
                "prefix": str(serving_prefix),
                "manifest_path": str(serving_path),
                "manifest_sha256": _sha256_bytes(serving_bytes),
                "directory_inventory_sha256": serving_manifest[
                    "directory_inventory"
                ]["inventory_sha256"],
            },
        },
        "publication": {
            "protocol": "fsync_verify_marker_last",
            "complete_marker": COMPLETE_MARKER_FILENAME,
        },
    }
    identity["control_pin_fragment"] = {
        "release_id": RELEASE_ID,
        "release_worktree": str(release_worktree),
        "git_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "transport_uncertainty_binding": transport_binding,
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
        "model_contract_path": str(model_contracts.path),
        "model_contract_sha256": model_contracts.sha256,
        "fleet_contract_path": str(fleet_contract_path),
        "fleet_contract_sha256": fleet_sha256,
        "harness_environment_prefix": str(harness_prefix),
        "harness_environment_manifest_path": str(harness_path),
        "harness_environment_sha256": _sha256_bytes(harness_bytes),
        "serving_environment_prefix": str(serving_prefix),
        "serving_environment_manifest_path": str(serving_path),
        "serving_environment_sha256": _sha256_bytes(serving_bytes),
    }
    identity_bytes = _json_bytes(identity)
    artifacts = {
        HARNESS_MANIFEST_FILENAME: harness_bytes,
        SERVING_MANIFEST_FILENAME: serving_bytes,
        RELEASE_IDENTITY_FILENAME: identity_bytes,
    }
    for filename, payload in tuple(artifacts.items()):
        artifacts[filename + CHECKSUM_SUFFIX] = _checksum_bytes(filename, payload)
    report = {
        "release_id": RELEASE_ID,
        "git_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "transport_uncertainty_binding": transport_binding,
        "transport_uncertainty_binding_sha256": (
            transport_binding_sha256
        ),
        "materialization_id": materialization_binding["materialization_id"],
        "materialization_marker_sha256": materialization_binding["marker_sha256"],
        "harness_environment_manifest_sha256": _sha256_bytes(harness_bytes),
        "serving_environment_manifest_sha256": _sha256_bytes(serving_bytes),
        "model_contract_sha256": model_contracts.sha256,
        "fleet_contract_sha256": fleet_sha256,
        "logical_replica_count": EXPECTED_LOGICAL_REPLICAS,
        "allocated_gpu_count": EXPECTED_GPUS,
        "artifact_sha256": {
            filename: _sha256_bytes(payload)
            for filename, payload in sorted(artifacts.items())
        },
    }
    return artifacts, report


def _marker_payload(
    artifacts: Mapping[str, bytes], report: Mapping[str, Any]
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "complete": True,
        "publication_protocol": "fsync_verify_marker_last",
        "artifacts": {
            filename: {
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
            for filename, payload in sorted(artifacts.items())
        },
        "git_commit": report["git_commit"],
        "source_tree_sha256": report["source_tree_sha256"],
        "transport_uncertainty_binding": report[
            "transport_uncertainty_binding"
        ],
        "transport_uncertainty_binding_sha256": report[
            "transport_uncertainty_binding_sha256"
        ],
    }
    marker["release_bundle_id"] = _sha256_bytes(_canonical_bytes(marker))
    return marker


def create_release_bundle(
    output_root: str | Path,
    *,
    release_worktree: str | Path,
    harness_prefix: str | Path,
    serving_prefix: str | Path,
    model_contract_path: str | Path,
    fleet_contract_path: str | Path,
    apply: bool = False,
    seal_worktree: bool = False,
    seal_environments: bool = False,
    seal_output_root: bool = False,
) -> dict[str, Any]:
    """Dry-run or publish one immutable release bundle."""

    root = Path(output_root).expanduser().resolve()
    worktree = _resolve_directory(release_worktree, description="release worktree")
    harness = _resolve_directory(harness_prefix, description="harness environment")
    serving = _resolve_directory(serving_prefix, description="serving environment")
    if harness == serving:
        raise ReleaseFreezeError("harness and serving environments must be distinct")
    _verify_nonoverlap(root, (worktree, harness, serving))
    model_path = Path(model_contract_path).expanduser().resolve()
    fleet_path = Path(fleet_contract_path).expanduser().resolve()
    for contract_path, label in (
        (model_path, "model contract"),
        (fleet_path, "fleet contract"),
    ):
        if not _is_relative_to(contract_path, worktree):
            raise ReleaseFreezeError(
                f"{label} must live inside the exact tagged release worktree: {contract_path}"
            )
    marker = root / COMPLETE_MARKER_FILENAME
    if marker.exists() or marker.is_symlink():
        report = verify_release_bundle(root)
        identity = _read_json_object(
            root / RELEASE_IDENTITY_FILENAME, description="release identity"
        )
        expected_paths = {
            "release_worktree": str(worktree),
            "harness": str(harness),
            "serving": str(serving),
            "model_contract": str(model_path),
            "fleet_contract": str(fleet_path),
        }
        observed_paths = {
            "release_worktree": identity["release_worktree"],
            "harness": identity["environments"]["harness"]["prefix"],
            "serving": identity["environments"]["serving"]["prefix"],
            "model_contract": identity["model_contract"]["path"],
            "fleet_contract": identity["fleet_contract"]["path"],
        }
        if observed_paths != expected_paths:
            raise ReleaseFreezeError(
                "completed release bundle was created for different inputs"
            )
        harness_manifest = _read_json_object(
            root / HARNESS_MANIFEST_FILENAME,
            description="harness environment manifest",
        )
        serving_manifest = _read_json_object(
            root / SERVING_MANIFEST_FILENAME,
            description="serving environment manifest",
        )
        if seal_worktree and identity.get("worktree_sealed_read_only") is not True:
            raise ReleaseFreezeError(
                "completed release has an incompatible unsealed worktree"
            )
        if seal_environments and (
            harness_manifest.get("sealed_read_only") is not True
            or serving_manifest.get("sealed_read_only") is not True
        ):
            raise ReleaseFreezeError(
                "completed release has incompatible unsealed environments"
            )
        output_sealed = stat.S_IMODE(root.stat().st_mode) & 0o222 == 0
        if apply and seal_output_root and not output_sealed:
            # ``RELEASE_COMPLETE.json`` is intentionally published before this chmod.
            # A retry of the exact command must therefore finish the requested seal
            # instead of treating the durable marker as the end of the transaction.
            _seal_output_root_read_only(root)
            output_sealed = True
            report = verify_release_bundle(root)
        return {
            **report,
            "status": "already_complete",
            "output_root_sealed_read_only": output_sealed,
            "would_seal_output_root": bool(
                seal_output_root and not apply and not output_sealed
            ),
        }

    # Optional permission mutation occurs only under --apply and before inventories are
    # captured, so stored modes describe the actual immutable prefixes.  First perform
    # a complete, read-only preflight; an invalid tag, lock, runtime, or contract must
    # never leave an otherwise usable prefix unexpectedly sealed.
    materialization_binding = _verified_materialization_binding(
        output_root=root,
        release_worktree=worktree,
        harness_prefix=harness,
        serving_prefix=serving,
    )
    if apply and (seal_worktree or seal_environments):
        _build_release_material(
            output_root=root,
            release_worktree=worktree,
            harness_prefix=harness,
            serving_prefix=serving,
            model_contract_path=model_path,
            fleet_contract_path=fleet_path,
            worktree_sealed=False,
            environments_sealed=False,
            materialization_binding=materialization_binding,
        )
    if apply and seal_worktree:
        _seal_tree_read_only(worktree)
    if apply and seal_environments:
        _seal_tree_read_only(harness)
        _seal_tree_read_only(serving)
    artifacts, report = _build_release_material(
        output_root=root,
        release_worktree=worktree,
        harness_prefix=harness,
        serving_prefix=serving,
        model_contract_path=model_path,
        fleet_contract_path=fleet_path,
        worktree_sealed=bool(apply and seal_worktree),
        environments_sealed=bool(apply and seal_environments),
        materialization_binding=materialization_binding,
    )
    if not apply:
        return {
            **report,
            "status": "dry_run",
            "would_seal_worktree": seal_worktree,
            "would_seal_environments": seal_environments,
            "would_seal_output_root": seal_output_root,
        }

    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in artifacts.items():
        _atomic_write_exact(root / filename, payload)
    # Verify every pre-marker byte before publication.
    for filename, payload in artifacts.items():
        path = root / filename
        if _sha256_file(path) != _sha256_bytes(payload):
            raise ReleaseFreezeError(
                f"release artifact failed post-write verification: {path}"
            )
    # Re-read every pinned live input after artifact publication and before publishing
    # success.  If an unsealed worktree or prefix changed during the write window, the
    # immutable artifacts remain as useful evidence but no completion marker appears.
    final_artifacts, final_report = _build_release_material(
        output_root=root,
        release_worktree=worktree,
        harness_prefix=harness,
        serving_prefix=serving,
        model_contract_path=model_path,
        fleet_contract_path=fleet_path,
        worktree_sealed=bool(apply and seal_worktree),
        environments_sealed=bool(apply and seal_environments),
        materialization_binding=materialization_binding,
    )
    if final_artifacts != artifacts or final_report != report:
        raise ReleaseFreezeError(
            "release inputs changed during publication; completion marker withheld"
        )
    final_materialization_binding = _verified_materialization_binding(
        output_root=root,
        release_worktree=worktree,
        harness_prefix=harness,
        serving_prefix=serving,
    )
    if final_materialization_binding != materialization_binding:
        raise ReleaseFreezeError(
            "materialization proof changed during release publication"
        )
    marker_payload = _marker_payload(artifacts, report)
    _atomic_write_exact(root / COMPLETE_MARKER_FILENAME, _json_bytes(marker_payload))
    verified = verify_release_bundle(root)
    if seal_output_root:
        _seal_output_root_read_only(root)
        verified = verify_release_bundle(root)
    return {
        **verified,
        "status": "created",
        "output_root_sealed_read_only": bool(seal_output_root),
    }


def _verify_manifest_environment(
    payload: Mapping[str, Any],
    *,
    path: Path,
    release_worktree: Path,
    git_identity: Mapping[str, str],
    materialization_binding: Mapping[str, Any],
) -> None:
    required_fields = {
        "schema_version",
        "release_id",
        "role",
        "prefix",
        "sealed_read_only",
        "offline_environment",
        "conda_creation_tool",
        "environment_seed",
        "ownership_policy",
        "integrity_normalization_policy",
        "normalization_receipt",
        "conda_package_cache_sha256",
        "runtime",
        "locks",
        "release_package",
        "installed_files",
        "directory_inventory",
        "environment_content_sha256",
    }
    if set(payload) != required_fields:
        raise ReleaseFreezeError(f"environment manifest has the wrong fields: {path}")
    if payload.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ReleaseFreezeError(f"unsupported environment manifest schema: {path}")
    if payload.get("release_id") != RELEASE_ID:
        raise ReleaseFreezeError(f"environment release ID drifted: {path}")
    role = payload.get("role")
    if role not in {"harness", "serving"}:
        raise ReleaseFreezeError(f"invalid environment role in {path}")
    if payload.get("offline_environment") != REQUIRED_OFFLINE_ENVIRONMENT:
        raise ReleaseFreezeError(f"offline contract drifted in {path}")
    if type(payload.get("sealed_read_only")) is not bool:
        raise ReleaseFreezeError(f"environment seal state is not Boolean in {path}")
    prefix = _resolve_directory(
        payload.get("prefix", ""), description=f"{role} environment"
    )
    locks = payload.get("locks")
    runtime = payload.get("runtime")
    conda_record = payload.get("conda_creation_tool")
    environment_seed = payload.get("environment_seed")
    ownership_policy = payload.get("ownership_policy")
    integrity_normalization_policy = payload.get(
        "integrity_normalization_policy"
    )
    normalization_receipt = payload.get("normalization_receipt")
    installed_files = payload.get("installed_files")
    if (
        not isinstance(locks, dict)
        or set(locks) != {"conda_explicit", "pip_freeze_all"}
        or not isinstance(runtime, dict)
        or not isinstance(conda_record, dict)
        or set(conda_record) != {"path", "sha256"}
        or not isinstance(environment_seed, dict)
        or set(environment_seed)
        != {
            "capture_id",
            "capture_marker_sha256",
            "prefix",
            "normalized_content_inventory_sha256",
        }
        or not isinstance(ownership_policy, dict)
        or set(ownership_policy) != {"path", "sha256"}
        or not isinstance(integrity_normalization_policy, dict)
        or set(integrity_normalization_policy) != {"path", "sha256"}
        or not isinstance(normalization_receipt, dict)
        or set(normalization_receipt) != {"id"}
        or not isinstance(installed_files, dict)
        or set(installed_files)
        != {"inventory_sha256", "entry_count", "file_count", "total_file_bytes"}
    ):
        raise ReleaseFreezeError(f"environment locks/runtime absent in {path}")
    if (
        not isinstance(conda_record.get("path"), str)
        or not conda_record["path"]
        or not Path(conda_record["path"]).is_absolute()
        or _SHA256_RE.fullmatch(str(conda_record.get("sha256", ""))) is None
    ):
        raise ReleaseFreezeError(f"Conda creation-tool provenance is malformed: {path}")
    for value, description in (
        (environment_seed.get("capture_id"), "environment capture ID"),
        (
            environment_seed.get("capture_marker_sha256"),
            "environment capture marker SHA",
        ),
        (
            environment_seed.get("normalized_content_inventory_sha256"),
            "normalized seed inventory SHA",
        ),
        (ownership_policy.get("sha256"), "ownership-policy SHA"),
        (
            integrity_normalization_policy.get("sha256"),
            "integrity-normalization-policy SHA",
        ),
        (normalization_receipt.get("id"), "normalization receipt ID"),
        (
            payload.get("conda_package_cache_sha256"),
            "Conda package-cache SHA",
        ),
    ):
        if _SHA256_RE.fullmatch(str(value)) is None:
            raise ReleaseFreezeError(f"invalid {description} in {path}")
    if not Path(str(environment_seed.get("prefix", ""))).is_absolute():
        raise ReleaseFreezeError(f"environment seed path is malformed: {path}")
    capture_binding = materialization_binding.get("environment_capture")
    materialization_paths = materialization_binding.get("paths")
    if (
        not isinstance(capture_binding, Mapping)
        or not isinstance(materialization_paths, Mapping)
        or not isinstance(capture_binding.get("seed_prefixes"), Mapping)
        or not isinstance(capture_binding.get("stage_records"), Mapping)
        or role not in capture_binding["seed_prefixes"]
        or role not in capture_binding["stage_records"]
        or not isinstance(capture_binding["stage_records"][role], Mapping)
    ):
        raise ReleaseFreezeError(
            f"materialization lacks exact {role} upstream provenance"
        )
    seed_stage = capture_binding["stage_records"][role]
    expected_seed = {
        "capture_id": capture_binding["capture_id"],
        "capture_marker_sha256": capture_binding[
            "capture_marker_sha256"
        ],
        "prefix": capture_binding["seed_prefixes"][role],
        "normalized_content_inventory_sha256": seed_stage[
            "normalized_content_inventory_sha256"
        ],
    }
    expected_policy = {
        "path": capture_binding["ownership_policy_path"],
        "sha256": capture_binding["ownership_policy_sha256"],
    }
    expected_integrity_policy = {
        "path": capture_binding["integrity_normalization_policy_path"],
        "sha256": capture_binding[
            "integrity_normalization_policy_sha256"
        ],
    }
    expected_receipt = {
        "id": seed_stage["normalization_receipt_id"]
    }
    expected_prefix = materialization_paths.get(f"{role}_prefix")
    if (
        environment_seed != expected_seed
        or ownership_policy != expected_policy
        or integrity_normalization_policy != expected_integrity_policy
        or normalization_receipt != expected_receipt
        or conda_record != materialization_binding.get(
            "conda_creation_tool"
        )
        or payload.get("conda_package_cache_sha256")
        != materialization_binding.get("conda_package_cache_sha256")
        or payload.get("prefix") != expected_prefix
    ):
        raise ReleaseFreezeError(
            f"{role} environment provenance is not bound to the sealed "
            "materialization evidence"
        )
    if _conda_lock_from_records(prefix) != locks.get("conda_explicit"):
        raise ReleaseFreezeError(f"Conda explicit lock drifted: {prefix}")
    pip_lock, release_package = _pip_lock_material(
        prefix,
        release_worktree=release_worktree,
        git_identity=git_identity,
        require_release_package=role == "harness",
    )
    if pip_lock != locks.get("pip_freeze_all"):
        raise ReleaseFreezeError(f"pip explicit lock drifted: {prefix}")
    if release_package != payload.get("release_package"):
        raise ReleaseFreezeError(f"release package provenance drifted: {prefix}")
    if _collect_runtime(prefix, role=role) != runtime:
        raise ReleaseFreezeError(f"runtime package/version identity drifted: {prefix}")
    # Runtime/lock probes precede the inventory check so any accidental cache mutation
    # caused by those probes is visible rather than deferred until the next audit.
    inventory = directory_inventory(prefix)
    if inventory != payload.get("directory_inventory"):
        raise ReleaseFreezeError(f"environment directory inventory drifted: {prefix}")
    if installed_files != {
        "inventory_sha256": inventory["inventory_sha256"],
        "entry_count": inventory["entry_count"],
        "file_count": inventory["file_count"],
        "total_file_bytes": inventory["total_file_bytes"],
    }:
        raise ReleaseFreezeError(f"installed-file provenance drifted: {prefix}")
    expected_content_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "runtime": runtime,
                "locks": locks,
                "release_package": release_package,
                "environment_seed": environment_seed,
                "ownership_policy": ownership_policy,
                "integrity_normalization_policy": (
                    integrity_normalization_policy
                ),
                "normalization_receipt": normalization_receipt,
                "conda_creation_tool": conda_record,
                "conda_package_cache_sha256": payload[
                    "conda_package_cache_sha256"
                ],
                "inventory_sha256": inventory["inventory_sha256"],
            }
        )
    )
    if payload.get("environment_content_sha256") != expected_content_sha:
        raise ReleaseFreezeError(f"environment content digest is invalid: {path}")
    if payload.get("sealed_read_only") is True:
        _assert_read_only(prefix)


def verify_release_bundle(output_root: str | Path) -> dict[str, Any]:
    """Verify a published bundle and every live path it pins."""

    root = _resolve_directory(output_root, description="release artifact root")
    marker_path = root / COMPLETE_MARKER_FILENAME
    marker = _read_json_object(marker_path, description="release completion marker")
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o222:
        raise ReleaseFreezeError("release completion marker is writable")
    if (
        marker.get("schema_version") != RELEASE_SCHEMA_VERSION
        or marker.get("release_id") != RELEASE_ID
        or marker.get("complete") is not True
        or marker.get("publication_protocol") != "fsync_verify_marker_last"
    ):
        raise ReleaseFreezeError("release completion marker is invalid")
    if set(marker) != {
        "schema_version",
        "release_id",
        "complete",
        "publication_protocol",
        "artifacts",
        "git_commit",
        "source_tree_sha256",
        "transport_uncertainty_binding",
        "transport_uncertainty_binding_sha256",
        "release_bundle_id",
    }:
        raise ReleaseFreezeError("release completion marker has the wrong fields")
    if (
        _GIT_COMMIT_RE.fullmatch(str(marker.get("git_commit", ""))) is None
        or _SHA256_RE.fullmatch(str(marker.get("source_tree_sha256", ""))) is None
        or _SHA256_RE.fullmatch(
            str(marker.get("transport_uncertainty_binding_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(str(marker.get("release_bundle_id", ""))) is None
    ):
        raise ReleaseFreezeError("release completion marker has invalid digests")
    artifacts = marker.get("artifacts")
    expected_names = {
        HARNESS_MANIFEST_FILENAME,
        SERVING_MANIFEST_FILENAME,
        RELEASE_IDENTITY_FILENAME,
        HARNESS_MANIFEST_FILENAME + CHECKSUM_SUFFIX,
        SERVING_MANIFEST_FILENAME + CHECKSUM_SUFFIX,
        RELEASE_IDENTITY_FILENAME + CHECKSUM_SUFFIX,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ReleaseFreezeError("release marker has the wrong artifact inventory")
    for filename, record in artifacts.items():
        path = root / filename
        if path.is_symlink() or not path.is_file() or not isinstance(record, dict):
            raise ReleaseFreezeError(f"missing immutable release artifact: {path}")
        if set(record) != {"sha256", "size"}:
            raise ReleaseFreezeError(f"invalid release artifact record: {filename}")
        if (
            _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            or type(record.get("size")) is not int
            or record["size"] < 0
        ):
            raise ReleaseFreezeError(f"invalid release artifact identity: {filename}")
        if (
            path.stat().st_size != record["size"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ReleaseFreezeError(f"release artifact drifted: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ReleaseFreezeError(f"release artifact is writable: {path}")
    for filename in (
        HARNESS_MANIFEST_FILENAME,
        SERVING_MANIFEST_FILENAME,
        RELEASE_IDENTITY_FILENAME,
    ):
        payload = (root / filename).read_bytes()
        checksum = (root / (filename + CHECKSUM_SUFFIX)).read_text(
            encoding="utf-8"
        )
        if checksum != _checksum_bytes(filename, payload).decode("utf-8"):
            raise ReleaseFreezeError(f"release checksum sidecar is invalid: {filename}")
    candidate_marker = dict(marker)
    bundle_id = candidate_marker.pop("release_bundle_id", None)
    if bundle_id != _sha256_bytes(_canonical_bytes(candidate_marker)):
        raise ReleaseFreezeError("release bundle ID is invalid")
    identity_path = root / RELEASE_IDENTITY_FILENAME
    identity = _read_json_object(identity_path, description="release identity")
    required_identity_fields = {
        "schema_version",
        "release_id",
        "git",
        "release_worktree",
        "worktree_sealed_read_only",
        "materialization",
        "offline_environment",
        "transport_uncertainty",
        "model_contract",
        "fleet_contract",
        "environments",
        "publication",
        "control_pin_fragment",
    }
    if set(identity) != required_identity_fields:
        raise ReleaseFreezeError("release identity has the wrong fields")
    if identity.get("schema_version") != RELEASE_SCHEMA_VERSION or identity.get(
        "release_id"
    ) != RELEASE_ID:
        raise ReleaseFreezeError("release identity is invalid")
    if type(identity.get("worktree_sealed_read_only")) is not bool:
        raise ReleaseFreezeError("release worktree seal state is not Boolean")
    worktree = _resolve_directory(
        identity.get("release_worktree", ""), description="release worktree"
    )
    stored_git_identity = identity.get("git")
    if (
        not isinstance(stored_git_identity, dict)
        or set(stored_git_identity)
        != {"git_commit", "git_tag", "source_tree_sha256"}
        or stored_git_identity.get("git_tag") != REQUIRED_GIT_TAG
        or _GIT_COMMIT_RE.fullmatch(str(stored_git_identity.get("git_commit", "")))
        is None
        or _SHA256_RE.fullmatch(
            str(stored_git_identity.get("source_tree_sha256", ""))
        )
        is None
    ):
        raise ReleaseFreezeError("release source identity is malformed")
    if identity.get("worktree_sealed_read_only") is True:
        _assert_read_only(worktree)
        if sha256_tree(worktree) != stored_git_identity["source_tree_sha256"]:
            raise ReleaseFreezeError("sealed release source tree drifted")
        git_identity = dict(stored_git_identity)
    else:
        git_identity = verify_clean_exact_tag(worktree)
        if stored_git_identity != git_identity:
            raise ReleaseFreezeError("release source identity drifted")
    if marker.get("git_commit") != git_identity["git_commit"] or marker.get(
        "source_tree_sha256"
    ) != git_identity["source_tree_sha256"]:
        raise ReleaseFreezeError("completion marker source identity drifted")
    if identity.get("offline_environment") != REQUIRED_OFFLINE_ENVIRONMENT:
        raise ReleaseFreezeError("release offline requirement drifted")
    transport_record = identity.get("transport_uncertainty")
    expected_transport_binding = (
        scheduler_safety.expected_transport_uncertainty_binding()
    )
    expected_transport_sha256 = (
        scheduler_safety.transport_uncertainty_binding_sha256(
            expected_transport_binding
        )
    )
    if (
        not isinstance(transport_record, dict)
        or set(transport_record)
        != {"binding", "binding_sha256", "source_tree_sha256"}
        or transport_record.get("binding") != expected_transport_binding
        or transport_record.get("binding_sha256")
        != expected_transport_sha256
        or transport_record.get("source_tree_sha256")
        != git_identity["source_tree_sha256"]
        or marker.get("transport_uncertainty_binding")
        != expected_transport_binding
        or marker.get("transport_uncertainty_binding_sha256")
        != expected_transport_sha256
    ):
        raise ReleaseFreezeError(
            "release transport/checkpoint/self-consistency identity drifted"
        )
    if identity.get("publication") != {
        "protocol": "fsync_verify_marker_last",
        "complete_marker": COMPLETE_MARKER_FILENAME,
    }:
        raise ReleaseFreezeError("release publication contract drifted")
    model_record = identity.get("model_contract")
    fleet_record = identity.get("fleet_contract")
    if not isinstance(model_record, dict) or not isinstance(fleet_record, dict):
        raise ReleaseFreezeError("release model/fleet contract identity is absent")
    if set(model_record) != {"path", "sha256"} or set(fleet_record) != {
        "path",
        "sha256",
        "fleet_id",
        "logical_replica_count",
        "allocated_gpu_count",
    }:
        raise ReleaseFreezeError("release model/fleet identity has the wrong fields")
    model_contracts = load_model_contracts(
        model_record.get("path"), expected_sha256=model_record.get("sha256")
    )
    fleet_payload, fleet_sha = load_and_validate_fleet_contract(
        fleet_record.get("path"),
        model_contracts=model_contracts,
        expected_sha256=fleet_record.get("sha256"),
    )
    if (
        fleet_record.get("fleet_id") != fleet_payload["fleet_id"]
        or fleet_record.get("logical_replica_count") != EXPECTED_LOGICAL_REPLICAS
        or fleet_record.get("allocated_gpu_count") != EXPECTED_GPUS
        or fleet_sha != fleet_record.get("sha256")
    ):
        raise ReleaseFreezeError("release fleet summary drifted")
    environment_records = identity.get("environments")
    if not isinstance(environment_records, dict) or set(environment_records) != {
        "harness",
        "serving",
    }:
        raise ReleaseFreezeError("release environment identity is incomplete")
    for role in ("harness", "serving"):
        record = environment_records[role]
        if not isinstance(record, dict) or set(record) != {
            "prefix",
            "manifest_path",
            "manifest_sha256",
            "directory_inventory_sha256",
        }:
            raise ReleaseFreezeError(f"invalid {role} environment identity")
    materialization_binding = _verify_bound_materialization_evidence(
        identity.get("materialization", {}),
        output_root=root,
        release_worktree=worktree,
        harness_prefix=_resolve_directory(
            environment_records["harness"]["prefix"],
            description="harness environment",
        ),
        serving_prefix=_resolve_directory(
            environment_records["serving"]["prefix"],
            description="serving environment",
        ),
        git_identity=git_identity,
    )
    for role, filename in (
        ("harness", HARNESS_MANIFEST_FILENAME),
        ("serving", SERVING_MANIFEST_FILENAME),
    ):
        record = environment_records[role]
        manifest_path = Path(record.get("manifest_path", "")).resolve()
        if manifest_path != root / filename:
            raise ReleaseFreezeError(f"{role} environment manifest path drifted")
        manifest_sha = _sha256_file(manifest_path)
        if manifest_sha != record.get("manifest_sha256"):
            raise ReleaseFreezeError(f"{role} environment manifest SHA drifted")
        payload = _read_json_object(
            manifest_path, description=f"{role} environment manifest"
        )
        if payload.get("prefix") != record.get("prefix"):
            raise ReleaseFreezeError(f"{role} environment prefix drifted")
        if payload.get("directory_inventory", {}).get("inventory_sha256") != record.get(
            "directory_inventory_sha256"
        ):
            raise ReleaseFreezeError(f"{role} directory inventory digest drifted")
        _verify_manifest_environment(
            payload,
            path=manifest_path,
            release_worktree=worktree,
            git_identity=git_identity,
            materialization_binding=materialization_binding,
        )
    expected_control_fragment = {
        "release_id": RELEASE_ID,
        "release_worktree": str(worktree),
        "git_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "transport_uncertainty_binding": expected_transport_binding,
        "transport_uncertainty_binding_sha256": expected_transport_sha256,
        "model_contract_path": str(model_contracts.path),
        "model_contract_sha256": model_contracts.sha256,
        "fleet_contract_path": str(Path(fleet_record["path"]).resolve()),
        "fleet_contract_sha256": fleet_sha,
        "harness_environment_prefix": environment_records["harness"]["prefix"],
        "harness_environment_manifest_path": environment_records["harness"][
            "manifest_path"
        ],
        "harness_environment_sha256": environment_records["harness"][
            "manifest_sha256"
        ],
        "serving_environment_prefix": environment_records["serving"]["prefix"],
        "serving_environment_manifest_path": environment_records["serving"][
            "manifest_path"
        ],
        "serving_environment_sha256": environment_records["serving"][
            "manifest_sha256"
        ],
    }
    if identity.get("control_pin_fragment") != expected_control_fragment:
        raise ReleaseFreezeError("schema5_control pin fragment drifted")
    return {
        "status": "verified",
        "release_id": RELEASE_ID,
        "release_bundle_id": bundle_id,
        "git_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "transport_uncertainty_binding_sha256": expected_transport_sha256,
        "materialization_id": materialization_binding["materialization_id"],
        "materialization_marker_sha256": materialization_binding["marker_sha256"],
        "harness_environment_manifest_sha256": environment_records["harness"][
            "manifest_sha256"
        ],
        "serving_environment_manifest_sha256": environment_records["serving"][
            "manifest_sha256"
        ],
        "model_contract_sha256": model_contracts.sha256,
        "fleet_contract_sha256": fleet_sha,
        "logical_replica_count": EXPECTED_LOGICAL_REPLICAS,
        "allocated_gpu_count": EXPECTED_GPUS,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="dry-run or publish a release bundle")
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--release-worktree", type=Path, required=True)
    create.add_argument("--harness-prefix", type=Path, required=True)
    create.add_argument("--serving-prefix", type=Path, required=True)
    create.add_argument(
        "--model-contract",
        type=Path,
        default=REPO / "configs" / "model_contracts.v1.json",
    )
    create.add_argument(
        "--fleet-contract",
        type=Path,
        default=REPO / "configs" / "schema5_fleet.v1.json",
    )
    create.add_argument("--apply", action="store_true")
    create.add_argument("--seal-worktree", action="store_true")
    create.add_argument("--seal-environments", action="store_true")
    create.add_argument("--seal-output-root", action="store_true")
    verify = subparsers.add_parser("verify", help="verify an already published release")
    verify.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_release_bundle(args.output_root)
        else:
            report = create_release_bundle(
                args.output_root,
                release_worktree=args.release_worktree,
                harness_prefix=args.harness_prefix,
                serving_prefix=args.serving_prefix,
                model_contract_path=args.model_contract,
                fleet_contract_path=args.fleet_contract,
                apply=args.apply,
                seal_worktree=args.seal_worktree,
                seal_environments=args.seal_environments,
                seal_output_root=args.seal_output_root,
            )
    except (OSError, UnicodeError, ValueError, ReleaseFreezeError) as exc:
        print(f"[schema5-release] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
