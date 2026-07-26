#!/usr/bin/env python3
"""Compute a deterministic identity for the Conda runtime without invoking Conda."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL = "schema5-conda-runtime-toolchain-identity-v1"
EXCLUDED_TOP_LEVEL = (".conda", "conda-bld", "envs", "pkgs")
_CHUNK_SIZE = 8 * 1024 * 1024


class CondaRuntimeIdentityError(RuntimeError):
    """The Conda entrypoint or its complete base runtime is unsafe or unstable."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CondaRuntimeIdentityError(f"not a regular runtime file: {path}")
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise CondaRuntimeIdentityError(f"runtime file changed while hashing: {path}")
    return digest.hexdigest()


def _regular_canonical(path: str | Path, *, description: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        metadata = os.lstat(lexical)
    except OSError as exc:
        raise CondaRuntimeIdentityError(f"missing {description}: {lexical}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CondaRuntimeIdentityError(
            f"{description} must be one regular non-symlink file: {lexical}"
        )
    if lexical.resolve(strict=True) != lexical:
        raise CondaRuntimeIdentityError(f"{description} path is not canonical: {lexical}")
    return lexical


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _entrypoint_contract(conda_executable: str | Path) -> tuple[Path, Path, Path]:
    executable = _regular_canonical(
        conda_executable, description="Conda executable"
    )
    if not os.access(executable, os.X_OK):
        raise CondaRuntimeIdentityError(
            f"Conda executable lacks execute permission: {executable}"
        )
    try:
        first_line = executable.open("rb").readline(16 * 1024)
    except OSError as exc:
        raise CondaRuntimeIdentityError(
            f"cannot read Conda entrypoint shebang: {exc}"
        ) from exc
    if (
        not first_line.startswith(b"#!")
        or len(first_line) >= 16 * 1024
        or b"\x00" in first_line
    ):
        raise CondaRuntimeIdentityError(
            "Conda entrypoint lacks one bounded absolute interpreter shebang"
        )
    try:
        shebang = first_line[2:].decode("utf-8").strip().split()
    except UnicodeDecodeError as exc:
        raise CondaRuntimeIdentityError("Conda shebang is not UTF-8") from exc
    if len(shebang) != 1 or not Path(shebang[0]).is_absolute():
        raise CondaRuntimeIdentityError(
            "Conda entrypoint must name one absolute base-prefix interpreter"
        )
    interpreter_lexical = Path(shebang[0]).absolute()
    if interpreter_lexical.is_symlink():
        try:
            interpreter = interpreter_lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CondaRuntimeIdentityError(
                f"Conda interpreter symlink is unsafe: {exc}"
            ) from exc
    else:
        interpreter = _regular_canonical(
            interpreter_lexical, description="Conda interpreter"
        )
    if not interpreter.is_file() or interpreter.is_symlink():
        raise CondaRuntimeIdentityError(
            f"resolved Conda interpreter is not regular: {interpreter}"
        )
    base_prefix = executable.parent.parent
    if executable.parent.name != "bin":
        raise CondaRuntimeIdentityError(
            "Conda executable must be the canonical base-prefix bin entrypoint"
        )
    if (
        base_prefix.resolve(strict=True) != base_prefix
        or not _is_relative_to(interpreter, base_prefix)
        or not _is_relative_to(interpreter_lexical, base_prefix)
    ):
        raise CondaRuntimeIdentityError(
            "Conda entrypoint interpreter escapes its canonical base prefix"
        )
    return executable, interpreter_lexical, interpreter


def _inventory_once(base_prefix: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    file_count = 0
    directory_count = 0
    symlink_count = 0
    total_bytes = 0
    stack = [base_prefix]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise CondaRuntimeIdentityError(
                f"cannot enumerate Conda runtime directory {directory}: {exc}"
            ) from exc
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(base_prefix)
            if len(relative.parts) == 1 and relative.name in EXCLUDED_TOP_LEVEL:
                continue
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise CondaRuntimeIdentityError(
                    f"cannot stat Conda runtime entry {path}: {exc}"
                ) from exc
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                records.append(
                    {"path": relative.as_posix(), "type": "directory", "mode": mode}
                )
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                digest = _sha256_file(path)
                file_count += 1
                total_bytes += metadata.st_size
                records.append(
                    {
                        "path": relative.as_posix(),
                        "type": "file",
                        "mode": mode,
                        "size": metadata.st_size,
                        "sha256": digest,
                    }
                )
            elif stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(path)
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise CondaRuntimeIdentityError(
                        f"unsafe Conda runtime symlink {path}: {exc}"
                    ) from exc
                if not _is_relative_to(resolved, base_prefix):
                    raise CondaRuntimeIdentityError(
                        f"Conda runtime symlink escapes base prefix: {path} -> {target}"
                    )
                symlink_count += 1
                records.append(
                    {
                        "path": relative.as_posix(),
                        "type": "symlink",
                        "mode": mode,
                        "target": target,
                        "resolved": resolved.relative_to(base_prefix).as_posix(),
                    }
                )
            else:
                raise CondaRuntimeIdentityError(
                    f"unsupported special file in Conda runtime: {path}"
                )
    records.sort(key=lambda row: (row["path"], row["type"]))
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_bytes": total_bytes,
        "inventory_sha256": hashlib.sha256(_canonical(records)).hexdigest(),
    }


def conda_runtime_identity(conda_executable: str | Path) -> dict[str, Any]:
    """Hash the complete base runtime, excluding only caches and child env roots."""

    executable, interpreter_lexical, interpreter = _entrypoint_contract(
        conda_executable
    )
    base_prefix = executable.parent.parent
    first = _inventory_once(base_prefix)
    second = _inventory_once(base_prefix)
    if first != second:
        raise CondaRuntimeIdentityError(
            "Conda base runtime changed across repeated inventory passes"
        )
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "base_prefix": str(base_prefix),
        "excluded_top_level": list(EXCLUDED_TOP_LEVEL),
        "conda_executable": {
            "path": str(executable),
            "sha256": _sha256_file(executable),
            "size": executable.stat().st_size,
        },
        "shebang_interpreter": {
            "path": str(interpreter_lexical),
            "resolved_path": str(interpreter),
            "sha256": _sha256_file(interpreter),
            "size": interpreter.stat().st_size,
        },
        "runtime_inventory": first,
    }
    identity["identity_sha256"] = hashlib.sha256(_canonical(identity)).hexdigest()
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-executable", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = conda_runtime_identity(args.conda_executable)
    except (OSError, CondaRuntimeIdentityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
