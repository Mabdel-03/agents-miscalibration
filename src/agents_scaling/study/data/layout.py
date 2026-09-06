"""On-disk layout of ``<run_root>/data`` (WP1).  Path constants only — no data access.

Kept separate from :mod:`.protected` so that the exporter (the writer of the protected
files) does not import the protected *reader*, which only ``study.evaluation`` may import
(§10.6; corrections P1-4).
"""

from __future__ import annotations

import os
from pathlib import Path

PUBLIC_DIR = "public"
TASKS_FILE = "tasks.jsonl"
PROTECTED_DIR = "protected"
HLE_LABELS_FILE = "hle_labels.jsonl"
BCB_TESTS_FILE = "bcb_tests.jsonl"
DIR_MODE = 0o700
FILE_MODE = 0o600


def data_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / "data"


def public_tasks_path(run_root: str | os.PathLike) -> Path:
    return data_dir(run_root) / PUBLIC_DIR / TASKS_FILE


def protected_dir(run_root: str | os.PathLike) -> Path:
    return data_dir(run_root) / PROTECTED_DIR


__all__ = [name for name in globals() if not name.startswith("_")]
