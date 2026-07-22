#!/usr/bin/env python3
"""Run one recovery command and atomically retain its exact invocation and output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.label):
        parser.error("--label must be one safe path component")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    payload = {
        "schema_version": 1,
        "label": args.label,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "cwd": str(Path.cwd().resolve()),
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr": completed.stderr,
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    _atomic_json(args.output_dir.resolve() / f"{args.label}.json", payload)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=__import__("sys").stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
