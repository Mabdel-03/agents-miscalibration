#!/usr/bin/env python3
"""Restricted SSH forced command for the external schema-5 watchdog."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ALLOWED = {
    "schema5-watchdog status": ("status", "--live"),
    "schema5-watchdog repair-chain": ("repair-chain",),
    "schema5-watchdog finalizer-reconcile": ("finalizer-reconcile",),
}


class ForcedCommandError(RuntimeError):
    """The forced command request is outside the watchdog authority."""


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--harness-python", type=Path, required=True)
    parser.add_argument("--resolved-harness-python", type=Path, required=True)
    parser.add_argument(
        "--harness-environment-manifest", type=Path, required=True
    )
    parser.add_argument("--harness-environment-sha256", required=True)
    parser.add_argument("--control-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    try:
        invoked_python = Path(sys.executable).resolve(strict=True)
        expected_python = args.resolved_harness_python.expanduser().resolve(
            strict=True
        )
        if invoked_python != expected_python:
            raise ForcedCommandError(
                "forced command was not invoked by its exact resolved harness Python"
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
        proc = subprocess.run(command, check=False)
        return int(proc.returncode)
    except (ForcedCommandError, OSError) as exc:
        print(
            json.dumps({"authorized": False, "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
