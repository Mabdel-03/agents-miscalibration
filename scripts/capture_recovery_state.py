#!/usr/bin/env python3
"""Atomically capture scheduler and source state for a recovery operation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
import platform


def _run(*argv: str) -> dict[str, object]:
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {
            "argv": list(argv),
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "argv": list(argv),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--user", required=True)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--harness-prefix", type=Path)
    parser.add_argument("--serving-prefix", type=Path)
    args = parser.parse_args()

    captures = {
        "squeue": _run(
            "squeue", "-u", args.user, "-o",
            "%i|%T|%j|%M|%l|%R|%P|%b",
        ),
        "scontrol_jobs": _run("scontrol", "show", "job", "-dd"),
        "git_status": _run("git", "-C", str(args.repo), "status", "--porcelain=v2"),
        "git_head": _run("git", "-C", str(args.repo), "rev-parse", "HEAD"),
        "git_diff": _run("git", "-C", str(args.repo), "diff", "--binary", "HEAD"),
        "git_diff_cached": _run(
            "git", "-C", str(args.repo), "diff", "--binary", "--cached", "HEAD"
        ),
        "slurm_config": _run("scontrol", "show", "config"),
        "slurm_partitions": _run("sinfo", "-a", "-o", "%P|%a|%l|%D|%G"),
        "gpu_inventory": _run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,cuda_version",
            "--format=csv,noheader",
        ),
    }
    for label, prefix in (
        ("harness", args.harness_prefix),
        ("serving", args.serving_prefix),
    ):
        if prefix is None:
            continue
        python = prefix / "bin" / "python"
        captures[f"{label}_python"] = _run(str(python), "--version")
        captures[f"{label}_pip_freeze"] = _run(
            str(python), "-m", "pip", "freeze", "--all"
        )
        captures[f"{label}_packages_json"] = _run(
            str(python),
            "-c",
            (
                "import importlib.metadata,json;"
                "print(json.dumps(sorted((d.metadata['Name'],d.version) "
                "for d in importlib.metadata.distributions()),separators=(',',':')))"
            ),
        )
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(args.repo.resolve()),
        "host": platform.node(),
        "platform": platform.platform(),
        "harness_prefix": (
            None if args.harness_prefix is None else str(args.harness_prefix.resolve())
        ),
        "serving_prefix": (
            None if args.serving_prefix is None else str(args.serving_prefix.resolve())
        ),
        "captures": captures,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    payload["capture_failures"] = sorted(
        name for name, item in captures.items() if item["returncode"] != 0
    )
    _atomic_json(args.output.resolve(), payload)
    required = {
        "squeue",
        "scontrol_jobs",
        "git_status",
        "git_head",
        "git_diff",
        "git_diff_cached",
    }
    return 0 if all(captures[name]["returncode"] == 0 for name in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
