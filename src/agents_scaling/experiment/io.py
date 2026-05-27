"""Append-only JSONL logging + run/cell path helpers (consortium logger pattern)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT


def results_root() -> Path:
    return Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))


def run_dir(run_id: str) -> Path:
    d = results_root() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_dir(run_id: str, cell_id: str) -> Path:
    d = run_dir(run_id) / "cells" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    """Append one JSON record + newline. Append mode is crash-safe for resumable runs."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_json(path: str | os.PathLike, record: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(record, indent=2))


def git_commit() -> str | None:
    try:
        repo = Path(__file__).resolve().parents[3]
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None
