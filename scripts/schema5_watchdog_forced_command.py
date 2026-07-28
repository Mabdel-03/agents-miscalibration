#!/usr/bin/env python3
"""Restricted SSH forced command for the external schema-5 watchdog."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ALLOWED = {
    "schema5-watchdog status": ("watchdog-status",),
    "schema5-watchdog observe": ("watchdog-observe",),
    "schema5-watchdog repair-chain": ("watchdog-repair-chain",),
    "schema5-watchdog finalizer-reconcile": (
        "watchdog-finalizer-reconcile",
    ),
}
BOOTSTRAP_ALLOWED = {
    "schema5-bootstrap-watchdog status": (
        "bootstrap-status",
        "canonical",
    ),
    "schema5-bootstrap-watchdog repair": (
        "bootstrap-repair",
        "canonical",
    ),
    "schema5-bootstrap-watchdog drill-status": (
        "bootstrap-status",
        "isolated",
    ),
    "schema5-bootstrap-watchdog drill-repair": (
        "bootstrap-repair",
        "isolated",
    ),
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BOOTSTRAP_TAGGED_SOURCE_ROOTS = (Path("scripts"), Path("src"))


class ForcedCommandError(RuntimeError):
    """The forced command request is outside the watchdog authority."""


def _child_process_environment() -> dict[str, str]:
    """Return the complete environment delegated to the frozen control command."""

    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }


def _canonical_path(path: Path, *, description: str, kind: str) -> Path:
    """Resolve one fixed deployment path without accepting symlink traversal."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise ForcedCommandError(f"{description} path contains unsafe characters")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ForcedCommandError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical:
        raise ForcedCommandError(f"{description} traverses a symlink")
    try:
        metadata = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise ForcedCommandError(f"{description} cannot be inspected: {exc}") from exc
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise ForcedCommandError(f"{description} is not a directory")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise ForcedCommandError(f"{description} is not a regular file")
    return lexical


def _stable_source_bytes(path: Path, *, description: str) -> bytes:
    """Read one immutable tagged source without following links or races."""

    canonical = _canonical_path(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    current = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        or stat.S_IMODE(before.st_mode) & 0o222
        or before.st_nlink != 1
    ):
        raise ForcedCommandError(
            f"{description} is mutable, linked, or changed while read"
        )
    return b"".join(chunks)


def _bootstrap_tagged_source_records(
    release_root: Path,
) -> list[dict[str, Any]]:
    """Recompute the conservative transitive tagged execution closure."""

    records: list[dict[str, Any]] = []
    for relative_root in BOOTSTRAP_TAGGED_SOURCE_ROOTS:
        source_root = _canonical_path(
            release_root / relative_root,
            description=f"bootstrap tagged source root {relative_root}",
            kind="directory",
        )
        for current, directory_names, file_names in os.walk(
            source_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            current_info = current_path.stat(follow_symlinks=False)
            if (
                current_path.is_symlink()
                or not stat.S_ISDIR(current_info.st_mode)
                or stat.S_IMODE(current_info.st_mode) & 0o222
            ):
                raise ForcedCommandError(
                    f"bootstrap tagged source directory is mutable or unsafe: "
                    f"{current_path}"
                )
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = current_path / name
                child_info = child.stat(follow_symlinks=False)
                if child.is_symlink() or not stat.S_ISDIR(child_info.st_mode):
                    raise ForcedCommandError(
                        "bootstrap tagged source tree contains an unsafe "
                        f"directory entry: {child}"
                    )
            for name in file_names:
                source = current_path / name
                relative = source.relative_to(release_root)
                raw = _stable_source_bytes(
                    source,
                    description=f"bootstrap tagged source {relative}",
                )
                records.append(
                    {
                        "path": relative.as_posix(),
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    records.sort(key=lambda item: str(item["path"]))
    required = {
        "scripts/schema5_bootstrap_watchdog.py",
        "scripts/schema5_watchdog_forced_command.py",
        "scripts/render_schema5_recovery_chain_v12.py",
        "scripts/build_schema5_watchdog_deployment.py",
        "src/agents_scaling/serving/external_watchdog.py",
    }
    missing = sorted(required - {str(item["path"]) for item in records})
    if missing:
        raise ForcedCommandError(
            "bootstrap tagged-source execution closure is incomplete: "
            + ", ".join(missing)
        )
    return records


def _load_deployment_runtime(release_root: Path) -> Any:
    """Load the one sealed implementation of the harness-inventory contract."""

    source = _canonical_path(
        release_root / "scripts" / "build_schema5_watchdog_deployment.py",
        description="frozen watchdog deployment verifier",
        kind="file",
    )
    watchdog_runtime = _canonical_path(
        release_root
        / "src"
        / "agents_scaling"
        / "serving"
        / "external_watchdog.py",
        description="frozen external-watchdog decision runtime",
        kind="file",
    )
    if any(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222
        for path in (
            source,
            source.parent,
            watchdog_runtime,
            watchdog_runtime.parent,
        )
    ):
        raise ForcedCommandError(
            "watchdog deployment verifier/runtime is not read-only"
        )
    spec = importlib.util.spec_from_file_location(
        "_schema5_watchdog_forced_deployment_runtime", source
    )
    if spec is None or spec.loader is None:
        raise ForcedCommandError(
            "cannot load the watchdog deployment verifier"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ForcedCommandError(
            f"cannot initialize the watchdog deployment verifier: {exc}"
        ) from exc
    return module


def command_for(
    original_command: str,
    *,
    state_dir: Path,
    release_root: Path,
    harness_python: Path,
    resolved_harness_python: Path,
    harness_environment_manifest: Path,
    harness_environment_sha256: str,
    control_sha256: str,
) -> list[str]:
    selector = ALLOWED.get(original_command)
    if selector is None:
        raise ForcedCommandError("requested SSH command is not watchdog-authorized")
    release_root = _canonical_path(
        release_root, description="release root", kind="directory"
    )
    state_dir = _canonical_path(
        state_dir, description="control state", kind="directory"
    )
    control_script = release_root / "slurm" / "schema5_control.py"
    control_script = _canonical_path(
        control_script,
        description="frozen schema5_control.py",
        kind="file",
    )
    deployment = _load_deployment_runtime(release_root)
    try:
        pinned_python, binding = (
            deployment._resolve_inventory_pinned_harness_python(
                control_state_dir=state_dir,
                harness_python=harness_python,
                control_sha256=control_sha256,
                require_pristine_paused=False,
            )
        )
    except Exception as exc:
        raise ForcedCommandError(
            f"sealed harness Python verification failed: {exc}"
        ) from exc
    expected_resolved = _canonical_path(
        resolved_harness_python,
        description="resolved harness Python",
        kind="file",
    )
    expected_manifest = _canonical_path(
        harness_environment_manifest,
        description="harness environment manifest",
        kind="file",
    )
    if (
        pinned_python != expected_resolved
        or binding.get("resolved_path") != str(expected_resolved)
        or binding.get("lexical_path")
        != str(Path(os.path.abspath(os.fspath(harness_python.expanduser()))))
        or binding.get("manifest_path") != str(expected_manifest)
        or binding.get("manifest_sha256") != harness_environment_sha256
    ):
        raise ForcedCommandError(
            "forced command harness runtime differs from its sealed binding"
        )
    if not os.access(pinned_python, os.X_OK):
        raise ForcedCommandError("frozen harness Python is not executable")
    if (
        stat.S_IMODE(pinned_python.stat(follow_symlinks=False).st_mode) & 0o222
        or stat.S_IMODE(control_script.stat(follow_symlinks=False).st_mode) & 0o222
        or stat.S_IMODE(release_root.stat(follow_symlinks=False).st_mode) & 0o222
        or stat.S_IMODE(
            control_script.parent.stat(follow_symlinks=False).st_mode
        )
        & 0o222
    ):
        raise ForcedCommandError(
            "forced command requires read-only harness and control executables"
        )
    return [
        str(pinned_python),
        "-I",
        "-u",
        str(control_script),
        "--state-dir",
        str(state_dir),
        *selector,
    ]


def bootstrap_command_for(
    original_command: str,
    *,
    release_root: Path,
    harness_python: Path,
    resolved_harness_python: Path,
    harness_environment_manifest: Path,
    harness_environment_sha256: str,
    materialization_pilot_marker: Path,
    materialization_pilot_sha256: str,
    materialization_pilot_id: str,
    renderer_sha256: str,
    deployment_verifier_sha256: str,
    forced_command_sha256: str,
    bootstrap_runtime_sha256: str,
    tagged_source_inventory_sha256: str,
    chain_manifest: Path,
    chain_manifest_sha256: str,
    submission_receipt: Path,
    submission_receipt_sha256: str,
    isolated_drill_chain_manifest: Path,
    isolated_drill_chain_manifest_sha256: str,
    isolated_drill_submission_receipt: Path,
    isolated_drill_submission_receipt_sha256: str,
) -> list[str]:
    """Resolve the separate pre-control forced-command authority."""

    selection = BOOTSTRAP_ALLOWED.get(original_command)
    if selection is None:
        raise ForcedCommandError(
            "requested SSH command is not bootstrap-watchdog-authorized"
        )
    selector, namespace = selection
    release_root = _canonical_path(
        release_root, description="bootstrap release root", kind="directory"
    )
    renderer = _canonical_path(
        release_root / "scripts" / "render_schema5_recovery_chain_v12.py",
        description="frozen recovery renderer",
        kind="file",
    )
    deployment_verifier = _canonical_path(
        release_root / "scripts" / "build_schema5_watchdog_deployment.py",
        description="frozen watchdog deployment verifier",
        kind="file",
    )
    forced_source = _canonical_path(
        release_root / "scripts" / "schema5_watchdog_forced_command.py",
        description="frozen watchdog forced command",
        kind="file",
    )
    bootstrap_runtime = _canonical_path(
        release_root / "scripts" / "schema5_bootstrap_watchdog.py",
        description="frozen bootstrap watchdog runtime",
        kind="file",
    )
    source_records = _bootstrap_tagged_source_records(release_root)
    by_path = {str(item["path"]): item for item in source_records}
    expected_sources = (
        (
            "scripts/schema5_bootstrap_watchdog.py",
            bootstrap_runtime_sha256,
        ),
        (
            "scripts/schema5_watchdog_forced_command.py",
            forced_command_sha256,
        ),
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            renderer_sha256,
        ),
        (
            "scripts/build_schema5_watchdog_deployment.py",
            deployment_verifier_sha256,
        ),
    )
    for relative, expected in expected_sources:
        record = by_path.get(relative)
        if (
            not isinstance(record, dict)
            or SHA256.fullmatch(expected) is None
            or record.get("sha256") != expected
        ):
            raise ForcedCommandError(
                f"bootstrap tagged source binding drifted: {relative}"
            )
    if (
        SHA256.fullmatch(tagged_source_inventory_sha256) is None
        or hashlib.sha256(
            (
                json.dumps(
                    source_records,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        != tagged_source_inventory_sha256
    ):
        raise ForcedCommandError(
            "bootstrap tagged source inventory binding drifted"
        )
    python = _canonical_path(
        resolved_harness_python,
        description="bootstrap resolved harness Python",
        kind="file",
    )
    environment_manifest = _canonical_path(
        harness_environment_manifest,
        description="bootstrap harness environment manifest",
        kind="file",
    )
    pilot_marker = _canonical_path(
        materialization_pilot_marker,
        description="bootstrap materialization pilot completion",
        kind="file",
    )
    manifest = _canonical_path(
        chain_manifest, description="recovery-chain manifest", kind="file"
    )
    receipt = _canonical_path(
        submission_receipt,
        description="recovery-chain submission receipt",
        kind="file",
    )
    isolated_manifest = _canonical_path(
        isolated_drill_chain_manifest,
        description="isolated drill recovery-chain manifest",
        kind="file",
    )
    isolated_receipt = _canonical_path(
        isolated_drill_submission_receipt,
        description="isolated drill recovery-chain submission receipt",
        kind="file",
    )
    expected_isolated_root = manifest.parent / "isolated_cancellation_drill"
    if (
        isolated_manifest
        != expected_isolated_root / manifest.name
        or isolated_receipt
        != expected_isolated_root / receipt.name
    ):
        raise ForcedCommandError(
            "bootstrap isolated drill binding is outside its exact sibling "
            "namespace"
        )
    deployment = _load_deployment_runtime(release_root)
    try:
        pilot_raw = _stable_source_bytes(
            pilot_marker,
            description="bootstrap materialization pilot completion",
        )
        pilot = json.loads(pilot_raw)
        candidate = dict(pilot)
        observed_pilot_id = candidate.pop("pilot_id", None)
        layout = pilot.get("layout")
        if (
            stat.S_IMODE(
                pilot_marker.stat(follow_symlinks=False).st_mode
            )
            & 0o222
            or pilot_marker.stat(follow_symlinks=False).st_nlink != 1
            or hashlib.sha256(pilot_raw).hexdigest()
            != materialization_pilot_sha256
            or observed_pilot_id != materialization_pilot_id
            or observed_pilot_id
            != hashlib.sha256(
                json.dumps(
                    candidate,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            or pilot.get("kind")
            != "schema5-materialization-pilot-completion"
            or pilot.get("complete") is not True
            or not isinstance(layout, dict)
        ):
            raise ForcedCommandError(
                "bootstrap materialization pilot binding drifted"
            )
        pinned_python, binding = (
            deployment._resolve_inventory_pinned_harness_python(
                control_state_dir=None,
                harness_python=harness_python,
                control_sha256=None,
                require_pristine_paused=False,
                environment_prefix=Path(str(layout["harness_prefix"])),
                environment_manifest_path=environment_manifest,
                environment_manifest_sha256=(
                    harness_environment_sha256
                ),
            )
        )
    except Exception as exc:
        raise ForcedCommandError(
            f"sealed bootstrap harness verification failed: {exc}"
        ) from exc
    if (
        pinned_python != python
        or binding.get("manifest_path") != str(environment_manifest)
        or binding.get("manifest_sha256") != harness_environment_sha256
    ):
        raise ForcedCommandError(
            "bootstrap Python differs from its sealed pilot harness"
        )
    for path, expected, description in (
        (manifest, chain_manifest_sha256, "chain manifest"),
        (receipt, submission_receipt_sha256, "submission receipt"),
        (
            isolated_manifest,
            isolated_drill_chain_manifest_sha256,
            "isolated drill chain manifest",
        ),
        (
            isolated_receipt,
            isolated_drill_submission_receipt_sha256,
            "isolated drill submission receipt",
        ),
    ):
        if (
            stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222
            or path.stat(follow_symlinks=False).st_nlink != 1
            or hashlib.sha256(
                _stable_source_bytes(path, description=description)
            ).hexdigest()
            != expected
        ):
            raise ForcedCommandError(
                f"bootstrap {description} binding drifted"
            )
    if (
        stat.S_IMODE(renderer.stat(follow_symlinks=False).st_mode) & 0o222
        or stat.S_IMODE(
            renderer.parent.stat(follow_symlinks=False).st_mode
        )
        & 0o222
        or stat.S_IMODE(python.stat(follow_symlinks=False).st_mode) & 0o222
        or not os.access(python, os.X_OK)
    ):
        raise ForcedCommandError(
            "bootstrap forced command requires immutable executables"
        )
    selected_manifest = (
        manifest if namespace == "canonical" else isolated_manifest
    )
    command = [
        str(python),
        "-I",
        "-u",
        str(renderer),
        selector,
        "--chain-manifest",
        str(selected_manifest),
    ]
    if selector == "bootstrap-repair":
        command.append("--apply")
        if namespace == "isolated":
            command.extend(
                [
                    "--result-output",
                    str(
                        isolated_manifest.parent
                        / "BOOTSTRAP_REPAIR_RESULT.json"
                    ),
                ]
            )
    return command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", choices=("control-dispatch", "bootstrap-dispatch"),
        default="control-dispatch",
    )
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--harness-python", type=Path, required=True)
    parser.add_argument("--resolved-harness-python", type=Path)
    parser.add_argument(
        "--harness-environment-manifest", type=Path
    )
    parser.add_argument("--harness-environment-sha256")
    parser.add_argument("--control-sha256")
    parser.add_argument("--chain-manifest", type=Path)
    parser.add_argument("--chain-manifest-sha256")
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--submission-receipt-sha256")
    parser.add_argument("--isolated-drill-chain-manifest", type=Path)
    parser.add_argument("--isolated-drill-chain-manifest-sha256")
    parser.add_argument("--isolated-drill-submission-receipt", type=Path)
    parser.add_argument("--isolated-drill-submission-receipt-sha256")
    parser.add_argument("--materialization-pilot-marker", type=Path)
    parser.add_argument("--materialization-pilot-sha256")
    parser.add_argument("--materialization-pilot-id")
    parser.add_argument("--renderer-sha256")
    parser.add_argument("--deployment-verifier-sha256")
    parser.add_argument("--forced-command-sha256")
    parser.add_argument("--bootstrap-runtime-sha256")
    parser.add_argument("--tagged-source-inventory-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    try:
        invoked_python = Path(sys.executable).resolve(strict=True)
        expected_source = args.resolved_harness_python
        if expected_source is None:
            raise ForcedCommandError(
                "forced command lacks its expected Python binding"
            )
        expected_python = expected_source.expanduser().resolve(strict=True)
        if invoked_python != expected_python:
            raise ForcedCommandError(
                "forced command was not invoked by its exact resolved harness Python"
            )
        if args.mode == "bootstrap-dispatch":
            if (
                args.chain_manifest is None
                or args.chain_manifest_sha256 is None
                or args.submission_receipt is None
                or args.submission_receipt_sha256 is None
                or args.isolated_drill_chain_manifest is None
                or args.isolated_drill_chain_manifest_sha256 is None
                or args.isolated_drill_submission_receipt is None
                or args.isolated_drill_submission_receipt_sha256 is None
                or args.resolved_harness_python is None
                or args.harness_environment_manifest is None
                or args.harness_environment_sha256 is None
                or args.materialization_pilot_marker is None
                or args.materialization_pilot_sha256 is None
                or args.materialization_pilot_id is None
                or args.renderer_sha256 is None
                or args.deployment_verifier_sha256 is None
                or args.forced_command_sha256 is None
                or args.bootstrap_runtime_sha256 is None
                or args.tagged_source_inventory_sha256 is None
            ):
                raise ForcedCommandError(
                    "bootstrap forced command binding is incomplete"
                )
            command = bootstrap_command_for(
                original,
                release_root=args.release_root.expanduser().absolute(),
                harness_python=args.harness_python.expanduser().absolute(),
                resolved_harness_python=(
                    args.resolved_harness_python.expanduser().absolute()
                ),
                harness_environment_manifest=(
                    args.harness_environment_manifest.expanduser().absolute()
                ),
                harness_environment_sha256=args.harness_environment_sha256,
                materialization_pilot_marker=(
                    args.materialization_pilot_marker.expanduser().absolute()
                ),
                materialization_pilot_sha256=(
                    args.materialization_pilot_sha256
                ),
                materialization_pilot_id=args.materialization_pilot_id,
                renderer_sha256=args.renderer_sha256,
                deployment_verifier_sha256=(
                    args.deployment_verifier_sha256
                ),
                forced_command_sha256=args.forced_command_sha256,
                bootstrap_runtime_sha256=args.bootstrap_runtime_sha256,
                tagged_source_inventory_sha256=(
                    args.tagged_source_inventory_sha256
                ),
                chain_manifest=args.chain_manifest.expanduser().absolute(),
                chain_manifest_sha256=args.chain_manifest_sha256,
                submission_receipt=(
                    args.submission_receipt.expanduser().absolute()
                ),
                submission_receipt_sha256=args.submission_receipt_sha256,
                isolated_drill_chain_manifest=(
                    args.isolated_drill_chain_manifest.expanduser().absolute()
                ),
                isolated_drill_chain_manifest_sha256=(
                    args.isolated_drill_chain_manifest_sha256
                ),
                isolated_drill_submission_receipt=(
                    args.isolated_drill_submission_receipt.expanduser().absolute()
                ),
                isolated_drill_submission_receipt_sha256=(
                    args.isolated_drill_submission_receipt_sha256
                ),
            )
        else:
            if (
                args.state_dir is None
                or args.resolved_harness_python is None
                or args.harness_environment_manifest is None
                or args.harness_environment_sha256 is None
                or args.control_sha256 is None
            ):
                raise ForcedCommandError(
                    "production forced command binding is incomplete"
                )
            command = command_for(
                original,
                state_dir=args.state_dir.expanduser().absolute(),
                release_root=args.release_root.expanduser().absolute(),
                harness_python=args.harness_python.expanduser().absolute(),
                resolved_harness_python=(
                    args.resolved_harness_python.expanduser().absolute()
                ),
                harness_environment_manifest=(
                    args.harness_environment_manifest.expanduser().absolute()
                ),
                harness_environment_sha256=args.harness_environment_sha256,
                control_sha256=args.control_sha256,
            )
        proc = subprocess.run(
            command,
            check=False,
            env=_child_process_environment(),
        )
        return int(proc.returncode)
    except (ForcedCommandError, OSError) as exc:
        print(
            json.dumps({"authorized": False, "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
