#!/usr/bin/env python3
"""Materialize the immutable schema-5 source and runtime prefixes.

This is the mutation step immediately before :mod:`freeze_schema5_release`.  It is
dry-run by default.  ``--apply`` creates a detached worktree at the exact production
tag, independently clones the harness and serving Conda prefixes with copy semantics,
and replaces the cloned harness's development checkout with one non-editable install
from that exact worktree.

The command is deliberately fail-closed and resumable at explicit stage boundaries.
It never deletes or overwrites a destination.  A process interrupted inside a Conda
clone leaves an untrusted prefix without a stage record; that prefix must be moved to
quarantine before retrying.  Completed stages and the marker-last final record are
content bound and can be verified idempotently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import freeze_schema5_release as freeze  # noqa: E402


# Version 2 adds a transitive symlink-independence contract to every clone stage.
# Version-1 stage records intentionally fail closed rather than being adopted without
# evidence that their links are independent of the mutable source prefixes.
SCHEMA_VERSION = 2
RELEASE_ID = freeze.RELEASE_ID
REQUIRED_TAG = freeze.REQUIRED_GIT_TAG
COMPLETE_MARKER = "MATERIALIZATION_COMPLETE.json"
BUILD_EVIDENCE_COMPLETE_MARKER = "HARNESS_BUILD_EVIDENCE_COMPLETE.json"
STAGE_FILENAMES = {
    "worktree": "WORKTREE_MATERIALIZED.json",
    "harness_clone": "HARNESS_CLONE_COMPLETE.json",
    "serving_clone": "SERVING_CLONE_COMPLETE.json",
    "harness_package": "HARNESS_PACKAGE_COMPLETE.json",
}
_SHA256_RE = freeze._SHA256_RE
_GIT_COMMIT_RE = freeze._GIT_COMMIT_RE


class MaterializationError(RuntimeError):
    """The requested release cannot be materialized or proven exact."""


def _run(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise MaterializationError(f"cannot execute {argv[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise MaterializationError(
            f"command failed ({completed.returncode}): {list(argv)!r}: {detail}"
        )
    return completed.stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=freeze._reject_duplicate_keys,
            parse_constant=freeze._reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MaterializationError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{description} must contain one JSON object: {path}")
    return value


def _safe_existing_directory(value: str | Path, *, description: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise MaterializationError(f"missing or symlinked {description}: {path}")
    resolved = path.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise MaterializationError(f"refusing unsafe broad {description}: {resolved}")
    return resolved


def _safe_destination(value: str | Path, *, description: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise MaterializationError(f"symlinked {description} is forbidden: {lexical}")
    path = lexical.resolve()
    if path in {Path(path.anchor), Path.home().resolve()}:
        raise MaterializationError(f"refusing unsafe broad {description}: {path}")
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_nonoverlap(paths: Mapping[str, Path]) -> None:
    output_root = paths.get("output_root")
    items = [(name, path) for name, path in paths.items() if name != "output_root"]
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or _is_relative_to(left, right) or _is_relative_to(right, left):
                raise MaterializationError(
                    f"materialization paths overlap: {left_name}={left}, "
                    f"{right_name}={right}"
                )
    if output_root is not None:
        for name, path in items:
            # The materialization root may own the three new destinations, but its
            # stage records must not be written inside a source/worktree/environment.
            if output_root == path or _is_relative_to(output_root, path):
                raise MaterializationError(
                    f"output root {output_root} is inside observed path {name}={path}"
                )


def _git(repository: Path, *args: str) -> str:
    return _run(("git", "-C", str(repository), *args)).strip()


def _tag_commit(repository: Path) -> str:
    top = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top != repository:
        raise MaterializationError(f"source repository is not its Git top-level: {repository}")
    try:
        commit = _git(
            repository,
            "rev-parse",
            "--verify",
            f"refs/tags/{REQUIRED_TAG}^{{commit}}",
        )
    except MaterializationError as exc:
        raise MaterializationError(f"required release tag is absent: {REQUIRED_TAG}") from exc
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise MaterializationError(f"release tag does not resolve to an exact commit: {commit!r}")
    return commit


def _conda_executable(value: str | Path) -> Path:
    candidate = Path(value).expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise MaterializationError(f"Conda executable is absent or not executable: {candidate}")
    return candidate


def _command_environment() -> dict[str, str]:
    blocked = {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
    env = {key: value for key, value in os.environ.items() if key not in blocked}
    env.update(
        {
            "CONDA_ALWAYS_COPY": "true",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return env


def _regular_inode_set(root: Path) -> tuple[set[tuple[int, int]], int]:
    identities: set[tuple[int, int]] = set()
    count = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                identities.add((info.st_dev, info.st_ino))
                count += 1
    return identities, count


_SYMLINK_AUDIT_COUNT_FIELDS = (
    "destination_symlink_count",
    "destination_internal_symlink_count",
    "destination_external_symlink_count",
    "source_prefix_target_symlink_count",
    "unresolvable_symlink_count",
)

_MAX_SYMLINK_HOPS = 64


def _normalized_symlink_audit_sources(
    source_prefixes: Sequence[Path], *, allow_empty: bool = False
) -> tuple[Path, ...]:
    if not source_prefixes and not allow_empty:
        raise MaterializationError(
            "clone symlink audit requires at least one mutable source prefix"
        )
    resolved: set[Path] = set()
    for source in source_prefixes:
        candidate = Path(source)
        if candidate.is_symlink() or not candidate.is_dir():
            raise MaterializationError(
                f"clone symlink audit source is absent or symlinked: {candidate}"
            )
        try:
            resolved.add(candidate.resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise MaterializationError(
                f"cannot resolve clone symlink audit source {candidate}: {exc}"
            ) from exc
    return tuple(sorted(resolved, key=lambda value: str(value)))


def verify_clone_symlinks(
    destination: Path,
    *,
    source_prefixes: Sequence[Path],
    _allow_empty_source_prefixes: bool = False,
) -> dict[str, Any]:
    """Fail closed if a cloned prefix contains an unsafe symlink.

    Conda's ``--copy`` contract applies to regular files but does not guarantee that
    absolute links embedded in a source environment are rewritten.  A copied link
    back into either mutable source prefix would therefore make the supposedly
    immutable production environment depend on mutable state.  Resolve every link
    transitively and reject source-prefix targets as well as broken links and loops,
    whose ultimate target cannot be proven safe.  Every accepted link must resolve
    inside the cloned prefix; unpinned external/system targets are forbidden.
    """

    candidate = Path(destination)
    if candidate.is_symlink() or not candidate.is_dir():
        raise MaterializationError(
            f"clone symlink audit destination is absent or symlinked: {candidate}"
        )
    try:
        destination_root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(
            f"cannot resolve clone symlink audit destination {candidate}: {exc}"
        ) from exc
    audit_sources = _normalized_symlink_audit_sources(
        source_prefixes, allow_empty=_allow_empty_source_prefixes
    )
    total = 0

    def inside(path: Path, root: Path) -> bool:
        return path == root or _is_relative_to(path, root)

    def resolve_dependency(path: Path) -> Path:
        """Resolve one link while observing every filesystem dependency hop."""

        try:
            pending = list(path.relative_to(destination_root).parts)
        except ValueError as exc:
            raise MaterializationError(
                f"clone symlink candidate escapes its destination: {path}"
            ) from exc
        # Start at the canonical clone root.  Beginning at the filesystem root would
        # make the ordinary parent components of an absolute destination look like
        # external dependencies and, more importantly, would make it impossible to
        # distinguish those parents from a symlink target that really leaves the
        # clone and later re-enters it.
        current = destination_root
        visited_links: set[Path] = set()
        hop_count = 0
        while pending:
            component = pending.pop(0)
            if component in {"", "."}:
                continue
            if component == "..":
                parent = current.parent
                if not inside(parent, destination_root):
                    raise MaterializationError(
                        "clone contains an unpinned external symlink dependency: "
                        f"{path} leaves {destination_root} via {parent}"
                    )
                current = parent
                continue
            candidate_path = current / component
            if not inside(candidate_path, destination_root):
                raise MaterializationError(
                    "clone contains an unpinned external symlink dependency: "
                    f"{path} leaves {destination_root} via {candidate_path}"
                )
            # Reject entering a mutable source even when a later source-owned symlink
            # resolves back outside it.  Checking only Path.resolve()'s final endpoint
            # misses precisely that dependency chain.
            source_dependency = next(
                (source for source in audit_sources if inside(candidate_path, source)),
                None,
            )
            if source_dependency is not None:
                raise MaterializationError(
                    "clone symlink dependency enters a mutable source prefix: "
                    f"{path} via {candidate_path} (source {source_dependency})"
                )
            try:
                info = candidate_path.lstat()
            except OSError as exc:
                raise MaterializationError(
                    "clone contains an unresolvable symlink (broken link, loop, or "
                    f"inaccessible target): {path}: {exc}"
                ) from exc
            if not stat.S_ISLNK(info.st_mode):
                current = candidate_path
                continue
            hop_count += 1
            if candidate_path in visited_links or hop_count > _MAX_SYMLINK_HOPS:
                raise MaterializationError(
                    f"clone contains an unresolvable symlink cycle: {path}"
                )
            visited_links.add(candidate_path)
            try:
                raw_target = os.readlink(candidate_path)
            except OSError as exc:
                raise MaterializationError(
                    f"cannot read clone symlink dependency {candidate_path}: {exc}"
                ) from exc
            target_path = Path(raw_target)
            if target_path.is_absolute():
                source_target = next(
                    (source for source in audit_sources if inside(target_path, source)),
                    None,
                )
                if source_target is not None:
                    raise MaterializationError(
                        "clone symlink dependency enters a mutable source prefix: "
                        f"{path} via {target_path} (source {source_target})"
                    )
                # Absolute links are acceptable only when their lexical path begins
                # inside this exact clone.  Do not use resolve()/normpath here: either
                # would hide an external symlink hop or a ``..`` escape followed by
                # re-entry.
                try:
                    target_parts = list(target_path.relative_to(destination_root).parts)
                except ValueError as exc:
                    raise MaterializationError(
                        "clone contains an unpinned external symlink dependency: "
                        f"{path} -> {target_path}"
                    ) from exc
                current = destination_root
            else:
                target_parts = list(target_path.parts)
                current = candidate_path.parent
            # Preserve ``..`` components: the kernel applies them *after* resolving
            # any preceding symlink component, so lexical normpath would hide hops.
            pending = [*target_parts, *pending]
        return current

    def fail_walk(exc: OSError) -> None:
        raise MaterializationError(
            f"cannot traverse cloned prefix during symlink audit: {exc}"
        ) from exc

    for directory, directory_names, file_names in os.walk(
        destination_root, followlinks=False, onerror=fail_walk
    ):
        directory_names[:] = sorted(directory_names)
        for name in sorted((*directory_names, *file_names)):
            path = Path(directory) / name
            try:
                info = path.lstat()
            except OSError as exc:
                raise MaterializationError(
                    f"cannot inspect clone symlink candidate {path}: {exc}"
                ) from exc
            if not stat.S_ISLNK(info.st_mode):
                continue
            total += 1
            target = resolve_dependency(path)
            escaped_source = next(
                (source for source in audit_sources if inside(target, source)), None
            )
            if escaped_source is not None:
                raise MaterializationError(
                    "clone symlink resolves into a mutable source prefix: "
                    f"{path} -> {target} (source {escaped_source})"
                )
            if not inside(target, destination_root):
                raise MaterializationError(
                    "clone contains an unpinned external symlink: "
                    f"{path} -> {target}"
                )
    return {
        "symlink_audit_source_prefixes": [str(value) for value in audit_sources],
        "destination_symlink_count": total,
        "destination_internal_symlink_count": total,
        "destination_external_symlink_count": 0,
        "source_prefix_target_symlink_count": 0,
        "unresolvable_symlink_count": 0,
    }


def verify_sealed_environment_symlinks(destination: Path) -> dict[str, Any]:
    """Prove a sealed prefix has only resolvable, prefix-internal symlinks."""

    return verify_clone_symlinks(
        destination,
        source_prefixes=(),
        _allow_empty_source_prefixes=True,
    )


def _validate_recorded_symlink_audit(
    payload: Mapping[str, Any], *, source_prefixes: Sequence[Path]
) -> None:
    expected_sources = [
        str(value) for value in _normalized_symlink_audit_sources(source_prefixes)
    ]
    if payload.get("symlink_audit_source_prefixes") != expected_sources:
        raise MaterializationError("clone symlink audit source prefixes drifted")
    counts = {field: payload.get(field) for field in _SYMLINK_AUDIT_COUNT_FIELDS}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise MaterializationError("clone symlink audit has invalid count fields")
    if (
        counts["destination_symlink_count"]
        != counts["destination_internal_symlink_count"]
        + counts["destination_external_symlink_count"]
        or counts["source_prefix_target_symlink_count"] != 0
        or counts["unresolvable_symlink_count"] != 0
    ):
        raise MaterializationError("clone symlink audit contract drifted")


def _assert_live_symlink_audit(
    payload: Mapping[str, Any], *, destination: Path, source_prefixes: Sequence[Path]
) -> dict[str, Any]:
    _validate_recorded_symlink_audit(payload, source_prefixes=source_prefixes)
    live = verify_clone_symlinks(destination, source_prefixes=source_prefixes)
    expected = {
        "symlink_audit_source_prefixes": payload.get("symlink_audit_source_prefixes"),
        **{field: payload.get(field) for field in _SYMLINK_AUDIT_COUNT_FIELDS},
    }
    if live != expected:
        raise MaterializationError(
            f"clone symlink audit drifted after its stage was recorded: {destination}"
        )
    return live


def _content_inventory_identity(root: Path) -> dict[str, int | str]:
    """Hash path/type/content identity while deliberately excluding permission modes.

    The release freezer removes write bits after materialization.  Mode-independent
    content identity lets that authorized sealing operation remain verifiable while
    still detecting every added, removed, retargeted, or byte-modified entry.
    """

    try:
        inventory = freeze.directory_inventory(root)
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(
            f"cannot content-inventory materialized prefix {root}: {exc}"
        ) from exc
    normalized = [
        {key: value for key, value in entry.items() if key != "mode"}
        for entry in inventory["entries"]
    ]
    return {
        "content_inventory_sha256": _sha256_bytes(_json_bytes(normalized)),
        "content_inventory_entry_count": len(normalized),
        "content_inventory_file_count": sum(
            entry.get("type") == "file" for entry in normalized
        ),
        "content_inventory_symlink_count": sum(
            entry.get("type") == "symlink" for entry in normalized
        ),
    }


def verify_independent_copy(source: Path, destination: Path) -> dict[str, int]:
    """Prove that no destination regular file shares an inode with its source."""

    source_inodes, source_files = _regular_inode_set(source)
    destination_inodes, destination_files = _regular_inode_set(destination)
    shared = source_inodes & destination_inodes
    if shared:
        raise MaterializationError(
            f"Conda clone shares {len(shared)} regular-file inode(s) with {source}; "
            "copy semantics were not honored"
        )
    return {
        "source_regular_file_count": source_files,
        "destination_regular_file_count": destination_files,
        "shared_regular_inode_count": 0,
    }


def _raw_pip_freeze(prefix: Path) -> list[str]:
    output = _run(
        (
            str(prefix / "bin" / "python"),
            "-I",
            "-m",
            "pip",
            "freeze",
            "--all",
            "--local",
        ),
        env=_command_environment(),
    )
    return sorted((line.strip() for line in output.splitlines() if line.strip()), key=str.casefold)


def _clone_identity(
    *,
    source: Path,
    destination: Path,
    conda_executable: Path,
    source_prefixes: Sequence[Path] | None = None,
) -> dict[str, Any]:
    copy_report = verify_independent_copy(source, destination)
    symlink_report = verify_clone_symlinks(
        destination,
        source_prefixes=(source,) if source_prefixes is None else source_prefixes,
    )
    try:
        source_conda = freeze._conda_lock(
            source, conda_executable=conda_executable
        )
        destination_conda = freeze._conda_lock(
            destination, conda_executable=conda_executable
        )
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    if destination_conda != source_conda:
        raise MaterializationError(f"Conda explicit lock changed while cloning {source}")
    source_pip = _raw_pip_freeze(source)
    destination_pip = _raw_pip_freeze(destination)
    if destination_pip != source_pip:
        raise MaterializationError(f"pip distribution view changed while cloning {source}")
    source_inventory = _content_inventory_identity(source)
    destination_inventory = _content_inventory_identity(destination)
    return {
        "source_prefix": str(source),
        "destination_prefix": str(destination),
        "clone_mode": "conda_create_clone_copy",
        "conda_always_copy": True,
        "conda_explicit_sha256": _sha256_bytes(_json_bytes(source_conda)),
        "pip_freeze_sha256": _sha256_bytes(_json_bytes(source_pip)),
        **{f"source_{key}": value for key, value in source_inventory.items()},
        **{
            f"destination_{key}": value
            for key, value in destination_inventory.items()
        },
        **copy_report,
        **symlink_report,
    }


def _clone_identity_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"schema_version", "release_id", "stage", "record_sha256"}
    }


def _assert_live_clone_identity(
    payload: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
    conda_executable: Path,
    source_prefixes: Sequence[Path],
) -> dict[str, Any]:
    live = _clone_identity(
        source=source,
        destination=destination,
        conda_executable=conda_executable,
        source_prefixes=source_prefixes,
    )
    if _clone_identity_fields(payload) != live:
        raise MaterializationError(
            f"clone identity drifted after its stage was recorded: {destination}"
        )
    return live


def _post_install_identity(
    *,
    harness_prefix: Path,
    release_worktree: Path,
    git_identity: Mapping[str, str],
    conda_executable: Path,
    source_prefixes: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        conda_lock = freeze._conda_lock(
            harness_prefix, conda_executable=conda_executable
        )
        pip_lock, binding = freeze._pip_lock_material(
            harness_prefix,
            release_worktree=release_worktree,
            git_identity=git_identity,
            require_release_package=True,
        )
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    if binding is None:
        raise MaterializationError("harness release-package binding is absent")
    return (
        {
            "conda_explicit_sha256": _sha256_bytes(_json_bytes(conda_lock)),
            "pip_freeze_sha256": _sha256_bytes(_json_bytes(pip_lock)),
            **_content_inventory_identity(harness_prefix),
            **verify_clone_symlinks(
                harness_prefix, source_prefixes=source_prefixes
            ),
        },
        binding,
    )


_IMPORT_PROBE = r"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from agents_scaling.prompts.system_prompts import N_LEVELS, get_prompt
from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving.launch_server import _production_release_resources
from agents_scaling.serving.model_contracts import load_model_contracts

prefix = Path(sys.prefix).resolve()
release = Path(sys.argv[1]).resolve()
module = __import__("agents_scaling")
module_path = Path(module.__file__).resolve()
module_path.relative_to(prefix)
distribution = importlib.metadata.distribution("agents_scaling")
direct = json.loads(distribution.read_text("direct_url.json"))
model_path = release / "configs" / "model_contracts.v1.json"
fleet_path = release / "configs" / "schema5_fleet.v1.json"
resolved_release, resolved_model, resolved_fleet, serving_template = (
    _production_release_resources(
        release_worktree=str(release),
        model_contract_path=str(model_path),
        fleet_contract_path=str(fleet_path),
    )
)
models = load_model_contracts(resolved_model)
fleet = load_fleet_contract(resolved_fleet, model_contracts=models)
prompt_root = release / "configs" / "prompts"
prompt_sha256 = [
    hashlib.sha256(get_prompt(level, prompt_root=prompt_root).encode("utf-8")).hexdigest()
    for level in range(N_LEVELS)
]
print(json.dumps({
    "prefix": str(prefix),
    "module_path": str(module_path),
    "version": distribution.version,
    "direct_url": direct,
    "runtime_resources": {
        "release_worktree": str(resolved_release),
        "model_contract_path": str(resolved_model),
        "model_contract_sha256": models.sha256,
        "fleet_contract_path": str(resolved_fleet),
        "fleet_contract_sha256": fleet.sha256,
        "serving_template": str(serving_template),
        "prompt_sha256": prompt_sha256,
        "fleet_replica_count": len(fleet.replicas),
    },
}, sort_keys=True))
""".strip()


def verify_harness_import(harness_prefix: Path, release_worktree: Path) -> dict[str, Any]:
    raw = _run(
        (
            str(harness_prefix / "bin" / "python"),
            "-I",
            "-c",
            _IMPORT_PROBE,
            str(release_worktree),
        ),
        env=_command_environment(),
        cwd=Path("/"),
    )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=freeze._reject_duplicate_keys,
            parse_constant=freeze._reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise MaterializationError(f"invalid isolated harness import probe: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "prefix",
        "module_path",
        "version",
        "direct_url",
        "runtime_resources",
    }:
        raise MaterializationError("isolated harness import probe returned wrong fields")
    if Path(payload["prefix"]).resolve() != harness_prefix:
        raise MaterializationError("agents_scaling probe did not use the harness prefix")
    module_path = Path(payload["module_path"]).resolve()
    if not _is_relative_to(module_path, harness_prefix) or _is_relative_to(
        module_path, release_worktree
    ):
        raise MaterializationError(
            f"agents_scaling imported outside the independent harness prefix: {module_path}"
        )
    direct = payload.get("direct_url")
    if (
        not isinstance(direct, dict)
        or set(direct) != {"url", "dir_info"}
        or not isinstance(direct.get("dir_info"), dict)
        or direct["dir_info"].get("editable", False) is not False
        or freeze._local_file_url_path(str(direct.get("url", ""))) != release_worktree
    ):
        raise MaterializationError(
            "agents_scaling import is editable or not sourced from the tagged worktree"
        )
    resources = payload.get("runtime_resources")
    expected_paths = {
        "release_worktree": release_worktree,
        "model_contract_path": release_worktree / "configs" / "model_contracts.v1.json",
        "fleet_contract_path": release_worktree / "configs" / "schema5_fleet.v1.json",
        "serving_template": release_worktree / "slurm" / "serve_qwen.sbatch.tmpl",
    }
    if not isinstance(resources, dict) or any(
        Path(str(resources.get(field, ""))).resolve() != expected.resolve()
        for field, expected in expected_paths.items()
    ):
        raise MaterializationError(
            "isolated harness cannot resolve immutable runtime resources"
        )
    if resources.get("fleet_replica_count") != 22:
        raise MaterializationError("isolated harness loaded the wrong serving fleet")
    prompt_hashes = resources.get("prompt_sha256")
    expected_prompt_hashes = [
        hashlib.sha256(
            (release_worktree / "configs" / "prompts" / f"level{level}.txt")
            .read_text(encoding="utf-8")
            .strip()
            .encode("utf-8")
        ).hexdigest()
        for level in range(4)
    ]
    if prompt_hashes != expected_prompt_hashes:
        raise MaterializationError("isolated harness prompt resources drifted")
    for field in ("model_contract_sha256", "fleet_contract_sha256"):
        value = resources.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise MaterializationError(
                f"isolated harness returned an invalid {field}"
            )
    return payload


def _stage_payload(stage: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "stage": stage,
        **dict(fields),
    }
    payload["record_sha256"] = _sha256_bytes(_json_bytes(payload))
    return payload


def _write_stage(output_root: Path, stage: str, payload: Mapping[str, Any]) -> Path:
    path = output_root / STAGE_FILENAMES[stage]
    freeze._atomic_write_exact(path, _json_bytes(payload))
    return path


def _verify_stage(path: Path, *, stage: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationError(
            f"missing, symlinked, or writable {stage} stage record: {path}"
        )
    payload = _read_json(path, description=f"{stage} stage record")
    record_sha = payload.pop("record_sha256", None)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("release_id") != RELEASE_ID
        or payload.get("stage") != stage
        or record_sha != _sha256_bytes(_json_bytes(payload))
    ):
        raise MaterializationError(f"invalid or drifted {stage} stage record: {path}")
    payload["record_sha256"] = record_sha
    return payload


def _materialize_worktree(
    *,
    source_repository: Path,
    worktree: Path,
    output_root: Path,
    tag_commit: str,
) -> dict[str, Any]:
    stage_path = output_root / STAGE_FILENAMES["worktree"]
    if worktree.exists():
        if stage_path.is_file():
            recorded = _verify_stage(stage_path, stage="worktree")
            if recorded.get("source_repository") != str(source_repository):
                raise MaterializationError("worktree stage was created from another repository")
    else:
        _run(
            (
                "git",
                "-C",
                str(source_repository),
                "worktree",
                "add",
                "--detach",
                str(worktree),
                f"refs/tags/{REQUIRED_TAG}",
            )
        )
    try:
        identity = freeze.verify_clean_exact_tag(worktree)
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    if identity["git_commit"] != tag_commit:
        raise MaterializationError("materialized worktree is not the preflight tag commit")
    payload = _stage_payload(
        "worktree",
        {
            "source_repository": str(source_repository),
            "release_worktree": str(worktree),
            **identity,
        },
    )
    _write_stage(output_root, "worktree", payload)
    return payload


def _materialize_clone(
    *,
    role: str,
    source: Path,
    destination: Path,
    output_root: Path,
    conda_executable: Path,
    source_prefixes: Sequence[Path] | None = None,
) -> dict[str, Any]:
    stage = f"{role}_clone"
    stage_path = output_root / STAGE_FILENAMES[stage]
    audit_sources = (source,) if source_prefixes is None else tuple(source_prefixes)
    if destination.exists():
        if not stage_path.is_file():
            raise MaterializationError(
                f"untrusted partial {role} prefix without stage record: {destination}; "
                "move it to quarantine before retrying"
            )
        recorded = _verify_stage(stage_path, stage=stage)
        if (
            recorded.get("source_prefix") != str(source)
            or recorded.get("destination_prefix") != str(destination)
        ):
            raise MaterializationError(f"{role} clone stage paths drifted")
        _assert_live_symlink_audit(
            recorded, destination=destination, source_prefixes=audit_sources
        )
        package_stage_exists = (
            role == "harness"
            and (output_root / STAGE_FILENAMES["harness_package"]).is_file()
        )
        if not package_stage_exists:
            _assert_live_clone_identity(
                recorded,
                source=source,
                destination=destination,
                conda_executable=conda_executable,
                source_prefixes=audit_sources,
            )
        return recorded
    _run(
        (
            str(conda_executable),
            "create",
            "--yes",
            "--copy",
            "--prefix",
            str(destination),
            "--clone",
            str(source),
        ),
        env=_command_environment(),
    )
    if not (destination / "conda-meta").is_dir():
        raise MaterializationError(f"Conda clone did not materialize {destination}")
    identity = _clone_identity(
        source=source,
        destination=destination,
        conda_executable=conda_executable,
        source_prefixes=audit_sources,
    )
    _validate_recorded_symlink_audit(identity, source_prefixes=audit_sources)
    payload = _stage_payload(stage, identity)
    _write_stage(output_root, stage, payload)
    return payload


def _materialize_harness_package(
    *,
    harness_prefix: Path,
    release_worktree: Path,
    output_root: Path,
    git_identity: Mapping[str, str],
    conda_executable: Path,
    source_prefixes: Sequence[Path],
) -> dict[str, Any]:
    stage = "harness_package"
    stage_path = output_root / STAGE_FILENAMES[stage]
    build_marker_path = output_root / BUILD_EVIDENCE_COMPLETE_MARKER
    build_evidence: dict[str, Any]
    build_evidence_marker: dict[str, Any]
    recorded_stage: dict[str, Any] | None = None
    if not stage_path.is_file():
        if build_marker_path.exists() or build_marker_path.is_symlink():
            # The package install and evidence archival are already complete.  This is
            # the exact crash-recovery path after evidence publication but before the
            # encompassing harness-package stage record.  Never reinstall in this
            # state: doing so would redraw build evidence and could change installed
            # bytes that the durable marker already identifies.
            build_evidence, build_evidence_marker = (
                _load_completed_build_evidence(
                    release_worktree=release_worktree,
                    output_root=output_root,
                )
            )
        else:
            python = harness_prefix / "bin" / "python"
            _run(
                (str(python), "-m", "pip", "uninstall", "--yes", "agents_scaling"),
                env=_command_environment(),
            )
            _run(
                (
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    "--no-compile",
                    "--force-reinstall",
                    str(release_worktree),
                ),
                env=_command_environment(),
                cwd=Path("/"),
            )
            archived = _archive_release_build_evidence(
                release_worktree=release_worktree,
                output_root=output_root,
            )
            build_evidence, build_evidence_marker = (
                _load_completed_build_evidence(
                    release_worktree=release_worktree,
                    output_root=output_root,
                )
            )
            if build_evidence != archived:
                raise MaterializationError(
                    "published package-build evidence differs from its archive result"
                )
    else:
        recorded_stage = _verify_stage(stage_path, stage=stage)
        recorded_evidence = recorded_stage.get("build_evidence")
        if not isinstance(recorded_evidence, dict):
            raise MaterializationError("harness package stage lacks build evidence")
        build_evidence, build_evidence_marker = _load_completed_build_evidence(
            release_worktree=release_worktree,
            output_root=output_root,
        )
        if build_evidence != recorded_evidence:
            raise MaterializationError(
                "harness package stage build evidence drifted"
            )
    _verify_build_evidence(build_evidence, output_root=output_root)
    try:
        verified_source = freeze.verify_clean_exact_tag(release_worktree)
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(
            f"package build dirtied the exact release worktree: {exc}"
        ) from exc
    if dict(verified_source) != dict(git_identity):
        raise MaterializationError("release source identity changed during package install")
    try:
        post_install_identity, binding = _post_install_identity(
            harness_prefix=harness_prefix,
            release_worktree=release_worktree,
            git_identity=git_identity,
            conda_executable=conda_executable,
            source_prefixes=source_prefixes,
        )
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    if binding is None:
        raise MaterializationError("harness release-package binding is absent")
    import_probe = verify_harness_import(harness_prefix, release_worktree)
    payload = _stage_payload(
        stage,
        {
            "harness_prefix": str(harness_prefix),
            "release_worktree": str(release_worktree),
            "package_binding": binding,
            "isolated_import": import_probe,
            "post_install_identity": post_install_identity,
            "build_evidence": build_evidence,
            "build_evidence_marker": build_evidence_marker,
            "install_contract": {
                "editable": False,
                "dependencies_installed": False,
                "build_isolation": False,
                "bytecode_compiled": False,
                "index_access": False,
            },
        },
    )
    if recorded_stage is not None:
        if recorded_stage != payload:
            raise MaterializationError(
                "installed harness identity drifted after its stage was recorded"
            )
    else:
        _write_stage(output_root, stage, payload)
    return payload


def _archive_release_build_evidence(
    *, release_worktree: Path, output_root: Path
) -> dict[str, Any]:
    """Move the one permitted setuptools byproduct out of the clean worktree.

    Pip performs local PEP-517 builds in place and setuptools writes
    ``src/agents_scaling.egg-info`` and can also write ``build/`` or ``dist/``.  The
    final worktree must remain the exact tag, so those generated directories are
    retained as evidence under the materialization root and removed from the source
    tree with same-filesystem renames.  No other ignored or untracked path is accepted.
    """

    marker_path = output_root / BUILD_EVIDENCE_COMPLETE_MARKER
    if marker_path.exists() or marker_path.is_symlink():
        evidence, _marker = _load_completed_build_evidence(
            release_worktree=release_worktree,
            output_root=output_root,
        )
        return evidence

    ordinary = _git(
        release_worktree, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if ordinary:
        raise MaterializationError(
            "package build changed tracked or non-ignored release files: "
            + ordinary.splitlines()[0]
        )
    ignored = [
        value
        for value in _run(
            (
                "git",
                "-C",
                str(release_worktree),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            )
        ).split("\0")
        if value
    ]
    if not ignored:
        evidence = {
            "generated_paths": [],
            "archive_path": None,
            "archive_inventory_sha256": None,
        }
        _publish_build_evidence_marker(
            release_worktree=release_worktree,
            output_root=output_root,
            evidence=evidence,
        )
        return evidence
    allowed_roots = (
        Path("src") / "agents_scaling.egg-info",
        Path("build"),
        Path("dist"),
    )
    if any(
        not any(
            Path(relative) == allowed
            or _lexically_relative_to(Path(relative), allowed)
            for allowed in allowed_roots
        )
        for relative in ignored
    ):
        raise MaterializationError(
            "package build produced an unexpected ignored path: " + ignored[0]
        )
    evidence_root = output_root / "build_evidence"
    if evidence_root.exists() or evidence_root.is_symlink():
        raise MaterializationError(
            f"refusing to overwrite prior package-build evidence: {evidence_root}"
        )
    selected_roots = [
        allowed
        for allowed in allowed_roots
        if any(
            Path(relative) == allowed
            or _lexically_relative_to(Path(relative), allowed)
            for relative in ignored
        )
    ]
    for allowed_root in selected_roots:
        tracked = _git(release_worktree, "ls-files", "--", allowed_root.as_posix())
        if tracked:
            raise MaterializationError(
                f"refusing to archive a tracked release path: {allowed_root}"
            )
        generated_root = release_worktree / allowed_root
        if generated_root.is_symlink() or not generated_root.is_dir():
            raise MaterializationError(
                "expected setuptools build evidence is absent or symlinked: "
                f"{generated_root}"
            )
    evidence_root.mkdir(parents=True)
    for allowed_root in selected_roots:
        generated_root = release_worktree / allowed_root
        archive = evidence_root / allowed_root
        archive.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(generated_root, archive)
        except OSError as exc:
            raise MaterializationError(
                "cannot atomically retain package-build evidence; keep the release root "
                f"and worktree on one filesystem: {exc}"
            ) from exc
    inventory = freeze.directory_inventory(evidence_root)
    evidence = {
        "generated_paths": sorted(ignored),
        "archive_path": str(evidence_root),
        "archive_inventory_sha256": inventory["inventory_sha256"],
    }
    _publish_build_evidence_marker(
        release_worktree=release_worktree,
        output_root=output_root,
        evidence=evidence,
    )
    return evidence


def _lexically_relative_to(path: Path, parent: Path) -> bool:
    """Lexical relative-path containment helper for Git-reported paths."""

    if path.is_absolute() or parent.is_absolute():
        return False
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_build_evidence(
    payload: Mapping[str, Any], *, output_root: Path
) -> None:
    if set(payload) != {
        "generated_paths",
        "archive_path",
        "archive_inventory_sha256",
    } or not isinstance(payload.get("generated_paths"), list):
        raise MaterializationError("package-build evidence has the wrong fields")
    archive_path = payload.get("archive_path")
    inventory_sha = payload.get("archive_inventory_sha256")
    if archive_path is None:
        if payload["generated_paths"] or inventory_sha is not None:
            raise MaterializationError("empty package-build evidence is inconsistent")
        return
    archive = Path(str(archive_path)).resolve()
    evidence_root = (output_root / "build_evidence").resolve()
    if (
        not _is_relative_to(archive, evidence_root)
        or archive.is_symlink()
        or not archive.is_dir()
        or _SHA256_RE.fullmatch(str(inventory_sha)) is None
    ):
        raise MaterializationError("package-build evidence archive is invalid")
    if freeze.directory_inventory(archive)["inventory_sha256"] != inventory_sha:
        raise MaterializationError("package-build evidence archive drifted")


def _build_evidence_marker_payload(
    *, release_worktree: Path, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "harness_package_build_evidence",
        "release_worktree": str(release_worktree),
        "build_evidence": dict(evidence),
    }
    payload["record_sha256"] = _sha256_bytes(_json_bytes(payload))
    return payload


def _publish_build_evidence_marker(
    *,
    release_worktree: Path,
    output_root: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the package-build transaction boundary before its outer stage.

    The marker is intentionally separate from ``HARNESS_PACKAGE_COMPLETE.json``.  Its
    sole purpose is to make the narrow post-archive/pre-stage crash window resumable
    without rerunning pip or overwriting evidence.
    """

    _verify_build_evidence(evidence, output_root=output_root)
    path = output_root / BUILD_EVIDENCE_COMPLETE_MARKER
    payload = _build_evidence_marker_payload(
        release_worktree=release_worktree,
        evidence=evidence,
    )
    try:
        freeze._atomic_write_exact(path, _json_bytes(payload))
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    return {
        "filename": BUILD_EVIDENCE_COMPLETE_MARKER,
        "sha256": freeze._sha256_file(path),
        "record_sha256": payload["record_sha256"],
    }


def _load_completed_build_evidence(
    *, release_worktree: Path, output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and adopt only an exact completed build-evidence transaction."""

    path = output_root / BUILD_EVIDENCE_COMPLETE_MARKER
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationError(
            f"missing, symlinked, or writable package-build evidence marker: {path}"
        )
    payload = _read_json(path, description="package-build evidence marker")
    record_sha = payload.pop("record_sha256", None)
    if (
        set(payload)
        != {
            "schema_version",
            "release_id",
            "kind",
            "release_worktree",
            "build_evidence",
        }
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("release_id") != RELEASE_ID
        or payload.get("kind") != "harness_package_build_evidence"
        or payload.get("release_worktree") != str(release_worktree)
        or _SHA256_RE.fullmatch(str(record_sha)) is None
        or record_sha != _sha256_bytes(_json_bytes(payload))
    ):
        raise MaterializationError("package-build evidence marker is invalid")
    evidence = payload.get("build_evidence")
    if not isinstance(evidence, dict):
        raise MaterializationError("package-build evidence marker lacks evidence")
    _verify_build_evidence(evidence, output_root=output_root)

    # Archival is complete only when the exact tagged worktree has no tracked,
    # untracked, or ignored build byproduct left behind.  This check prevents a forged
    # marker from causing a partial archive to be adopted on retry.
    ordinary = _git(
        release_worktree, "status", "--porcelain=v1", "--untracked-files=all"
    )
    ignored = [
        value
        for value in _run(
            (
                "git",
                "-C",
                str(release_worktree),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            )
        ).split("\0")
        if value
    ]
    if ordinary or ignored:
        detail = ordinary.splitlines()[0] if ordinary else ignored[0]
        raise MaterializationError(
            "completed package-build evidence left release-worktree drift: " + detail
        )
    identity = {
        "filename": BUILD_EVIDENCE_COMPLETE_MARKER,
        "sha256": freeze._sha256_file(path),
        "record_sha256": record_sha,
    }
    return dict(evidence), identity


def _materialization_paths(
    *,
    output_root: str | Path,
    source_repository: str | Path,
    release_worktree: str | Path,
    source_harness_prefix: str | Path,
    source_serving_prefix: str | Path,
    harness_prefix: str | Path,
    serving_prefix: str | Path,
) -> dict[str, Path]:
    paths = {
        "output_root": _safe_destination(output_root, description="output root"),
        "source_repository": _safe_existing_directory(
            source_repository, description="source repository"
        ),
        "release_worktree": _safe_destination(
            release_worktree, description="release worktree"
        ),
        "source_harness_prefix": _safe_existing_directory(
            source_harness_prefix, description="source harness prefix"
        ),
        "source_serving_prefix": _safe_existing_directory(
            source_serving_prefix, description="source serving prefix"
        ),
        "harness_prefix": _safe_destination(
            harness_prefix, description="production harness prefix"
        ),
        "serving_prefix": _safe_destination(
            serving_prefix, description="production serving prefix"
        ),
    }
    _validate_nonoverlap(paths)
    for role in ("source_harness_prefix", "source_serving_prefix"):
        if not (paths[role] / "conda-meta").is_dir():
            raise MaterializationError(f"{role} is not a Conda prefix: {paths[role]}")
    return paths


def materialize_release(
    *,
    output_root: str | Path,
    source_repository: str | Path,
    release_worktree: str | Path,
    source_harness_prefix: str | Path,
    source_serving_prefix: str | Path,
    harness_prefix: str | Path,
    serving_prefix: str | Path,
    conda_executable: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    paths = _materialization_paths(
        output_root=output_root,
        source_repository=source_repository,
        release_worktree=release_worktree,
        source_harness_prefix=source_harness_prefix,
        source_serving_prefix=source_serving_prefix,
        harness_prefix=harness_prefix,
        serving_prefix=serving_prefix,
    )
    conda = _conda_executable(conda_executable)
    commit = _tag_commit(paths["source_repository"])
    marker_path = paths["output_root"] / COMPLETE_MARKER
    if marker_path.is_file():
        report = verify_materialization(paths["output_root"])
        expected = {
            key: str(paths[key])
            for key in (
                "source_repository",
                "release_worktree",
                "source_harness_prefix",
                "source_serving_prefix",
                "harness_prefix",
                "serving_prefix",
            )
        }
        if report["paths"] != expected or report["tag_commit"] != commit:
            raise MaterializationError("completed materialization belongs to different inputs")
        return {**report, "status": "already_complete"}
    try:
        source_identity = freeze.verify_clean_exact_tag(paths["source_repository"])
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(
            "source repository must itself be the clean exact production tag before "
            f"materialization: {exc}"
        ) from exc
    if source_identity["git_commit"] != commit:
        raise MaterializationError("clean source identity differs from release tag")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "git_tag": REQUIRED_TAG,
        "tag_commit": commit,
        "source_tree_sha256": source_identity["source_tree_sha256"],
        "paths": {
            key: str(value)
            for key, value in paths.items()
            if key != "output_root"
        },
        "output_root": str(paths["output_root"]),
        "conda_executable": str(conda),
        "clone_contract": {
            "command": "conda create --yes --copy --prefix DEST --clone SOURCE",
            "CONDA_ALWAYS_COPY": "true",
            "shared_regular_inode_count": 0,
            "source_prefix_target_symlink_count": 0,
            "unresolvable_symlink_count": 0,
        },
        "harness_install_contract": {
            "source": str(paths["release_worktree"]),
            "editable": False,
            "dependencies_installed": False,
            "build_isolation": False,
            "bytecode_compiled": False,
            "index_access": False,
            "isolated_import_required": True,
        },
    }
    if not apply:
        return {**plan, "status": "dry_run"}
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    mutable_source_prefixes = (
        paths["source_harness_prefix"],
        paths["source_serving_prefix"],
    )
    worktree_stage = _materialize_worktree(
        source_repository=paths["source_repository"],
        worktree=paths["release_worktree"],
        output_root=paths["output_root"],
        tag_commit=commit,
    )
    harness_clone = _materialize_clone(
        role="harness",
        source=paths["source_harness_prefix"],
        destination=paths["harness_prefix"],
        output_root=paths["output_root"],
        conda_executable=conda,
        source_prefixes=mutable_source_prefixes,
    )
    serving_clone = _materialize_clone(
        role="serving",
        source=paths["source_serving_prefix"],
        destination=paths["serving_prefix"],
        output_root=paths["output_root"],
        conda_executable=conda,
        source_prefixes=mutable_source_prefixes,
    )
    git_identity = {
        key: worktree_stage[key]
        for key in ("git_commit", "git_tag", "source_tree_sha256")
    }
    harness_package = _materialize_harness_package(
        harness_prefix=paths["harness_prefix"],
        release_worktree=paths["release_worktree"],
        output_root=paths["output_root"],
        git_identity=git_identity,
        conda_executable=conda,
        source_prefixes=mutable_source_prefixes,
    )
    if harness_package["post_install_identity"]["conda_explicit_sha256"] != (
        harness_clone["conda_explicit_sha256"]
    ):
        raise MaterializationError(
            "harness package installation changed the cloned Conda lock identity"
        )
    try:
        freeze._pip_lock_material(
            paths["serving_prefix"],
            release_worktree=paths["release_worktree"],
            git_identity=git_identity,
            require_release_package=False,
        )
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    stages = {
        "worktree": worktree_stage,
        "harness_clone": harness_clone,
        "serving_clone": serving_clone,
        "harness_package": harness_package,
    }
    marker: dict[str, Any] = {
        **plan,
        "complete": True,
        "publication_protocol": "stage_records_fsync_marker_last",
        "stage_records": {
            name: {
                "filename": STAGE_FILENAMES[name],
                "sha256": freeze._sha256_file(
                    paths["output_root"] / STAGE_FILENAMES[name]
                ),
                "record_sha256": payload["record_sha256"],
            }
            for name, payload in stages.items()
        },
    }
    marker["materialization_id"] = _sha256_bytes(_json_bytes(marker))
    freeze._atomic_write_exact(marker_path, _json_bytes(marker))
    report = verify_materialization(paths["output_root"])
    return {**report, "status": "created"}


def verify_materialization(output_root: str | Path) -> dict[str, Any]:
    root = _safe_existing_directory(output_root, description="materialization root")
    marker_path = root / COMPLETE_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise MaterializationError(
            f"missing regular materialization completion marker: {marker_path}"
        )
    marker = _read_json(marker_path, description="materialization completion marker")
    materialization_id = marker.pop("materialization_id", None)
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("release_id") != RELEASE_ID
        or marker.get("git_tag") != REQUIRED_TAG
        or marker.get("complete") is not True
        or marker.get("publication_protocol") != "stage_records_fsync_marker_last"
        or materialization_id != _sha256_bytes(_json_bytes(marker))
    ):
        raise MaterializationError("materialization completion marker is invalid")
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o222:
        raise MaterializationError("materialization completion marker is writable")
    raw_paths = marker.get("paths")
    if not isinstance(raw_paths, dict):
        raise MaterializationError("materialization marker lacks exact paths")
    expected_path_names = {
        "source_repository",
        "release_worktree",
        "source_harness_prefix",
        "source_serving_prefix",
        "harness_prefix",
        "serving_prefix",
    }
    if set(raw_paths) != expected_path_names:
        raise MaterializationError("materialization marker has the wrong path fields")
    paths = {
        key: _safe_existing_directory(value, description=key)
        for key, value in raw_paths.items()
    }
    conda_executable = _conda_executable(str(marker.get("conda_executable", "")))
    commit = _tag_commit(paths["source_repository"])
    if commit != marker.get("tag_commit"):
        raise MaterializationError("release tag moved after materialization")
    try:
        git_identity = freeze.verify_clean_exact_tag(paths["release_worktree"])
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    if git_identity["git_commit"] != commit:
        raise MaterializationError("materialized worktree commit drifted")
    records = marker.get("stage_records")
    if not isinstance(records, dict) or set(records) != set(STAGE_FILENAMES):
        raise MaterializationError("materialization marker has the wrong stage inventory")
    stages: dict[str, dict[str, Any]] = {}
    for stage, filename in STAGE_FILENAMES.items():
        record = records[stage]
        path = root / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"filename", "sha256", "record_sha256"}
            or record["filename"] != filename
            or path.is_symlink()
            or not path.is_file()
            or _SHA256_RE.fullmatch(str(record["sha256"])) is None
            or freeze._sha256_file(path) != record["sha256"]
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise MaterializationError(f"stage artifact drifted: {stage}")
        stages[stage] = _verify_stage(path, stage=stage)
        if stages[stage]["record_sha256"] != record["record_sha256"]:
            raise MaterializationError(f"stage record identity drifted: {stage}")
    mutable_source_prefixes = (
        paths["source_harness_prefix"],
        paths["source_serving_prefix"],
    )
    live_symlink_audits: dict[str, dict[str, Any]] = {}
    for role in ("harness", "serving"):
        stage = stages[f"{role}_clone"]
        if (
            stage.get("source_prefix") != str(paths[f"source_{role}_prefix"])
            or stage.get("destination_prefix") != str(paths[f"{role}_prefix"])
            or stage.get("clone_mode") != "conda_create_clone_copy"
            or stage.get("conda_always_copy") is not True
            or stage.get("shared_regular_inode_count") != 0
        ):
            raise MaterializationError(f"{role} clone contract drifted")
        live_symlink_audits[role] = _assert_live_symlink_audit(
            stage,
            destination=paths[f"{role}_prefix"],
            source_prefixes=mutable_source_prefixes,
        )
        if role == "serving":
            _assert_live_clone_identity(
                stage,
                source=paths["source_serving_prefix"],
                destination=paths["serving_prefix"],
                conda_executable=conda_executable,
                source_prefixes=mutable_source_prefixes,
            )
        else:
            # The one authorized harness mutation is captured by its package stage;
            # regular files must nevertheless remain independent from the source.
            verify_independent_copy(
                paths["source_harness_prefix"], paths["harness_prefix"]
            )
    try:
        _lock, binding = freeze._pip_lock_material(
            paths["harness_prefix"],
            release_worktree=paths["release_worktree"],
            git_identity=git_identity,
            require_release_package=True,
        )
        freeze._pip_lock_material(
            paths["serving_prefix"],
            release_worktree=paths["release_worktree"],
            git_identity=git_identity,
            require_release_package=False,
        )
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationError(str(exc)) from exc
    import_probe = verify_harness_import(
        paths["harness_prefix"], paths["release_worktree"]
    )
    package_stage = stages["harness_package"]
    build_evidence = package_stage.get("build_evidence")
    if not isinstance(build_evidence, dict):
        raise MaterializationError("harness package stage lacks build evidence")
    _verify_build_evidence(build_evidence, output_root=root)
    live_build_evidence, live_build_marker = _load_completed_build_evidence(
        release_worktree=paths["release_worktree"],
        output_root=root,
    )
    if (
        live_build_evidence != build_evidence
        or package_stage.get("build_evidence_marker") != live_build_marker
    ):
        raise MaterializationError("installed harness build evidence drifted")
    post_install_identity, post_install_binding = _post_install_identity(
        harness_prefix=paths["harness_prefix"],
        release_worktree=paths["release_worktree"],
        git_identity=git_identity,
        conda_executable=conda_executable,
        source_prefixes=mutable_source_prefixes,
    )
    if (
        binding != package_stage.get("package_binding")
        or post_install_binding != binding
        or import_probe != package_stage.get("isolated_import")
        or post_install_identity != package_stage.get("post_install_identity")
    ):
        raise MaterializationError("installed harness package drifted")
    if post_install_identity["conda_explicit_sha256"] != stages[
        "harness_clone"
    ].get("conda_explicit_sha256"):
        raise MaterializationError(
            "installed harness Conda lock differs from its clone stage"
        )
    return {
        "status": "verified",
        "release_id": RELEASE_ID,
        "materialization_id": materialization_id,
        "tag_commit": commit,
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": {key: str(value) for key, value in paths.items()},
        "harness_package": binding,
        "copy_contract": "conda_create_clone_copy_no_shared_regular_inodes",
        "clone_symlink_audits": live_symlink_audits,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser(
        "materialize", help="dry-run or apply exact worktree/environment materialization"
    )
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--source-repository", type=Path, required=True)
    materialize.add_argument("--release-worktree", type=Path, required=True)
    materialize.add_argument("--source-harness-prefix", type=Path, required=True)
    materialize.add_argument("--source-serving-prefix", type=Path, required=True)
    materialize.add_argument("--harness-prefix", type=Path, required=True)
    materialize.add_argument("--serving-prefix", type=Path, required=True)
    materialize.add_argument("--conda-executable", type=Path, required=True)
    materialize.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a completed materialization")
    verify.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_materialization(args.output_root)
        else:
            report = materialize_release(
                output_root=args.output_root,
                source_repository=args.source_repository,
                release_worktree=args.release_worktree,
                source_harness_prefix=args.source_harness_prefix,
                source_serving_prefix=args.source_serving_prefix,
                harness_prefix=args.harness_prefix,
                serving_prefix=args.serving_prefix,
                conda_executable=args.conda_executable,
                apply=args.apply,
            )
    except (OSError, UnicodeError, ValueError, MaterializationError) as exc:
        print(f"[schema5-materialize] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
