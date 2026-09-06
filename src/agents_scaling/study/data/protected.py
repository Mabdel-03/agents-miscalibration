"""Protected (evaluator-only) export readers (WP1).

RESTRICTION — §3.2, §3.3, §10.6 and amendment P2: ONLY ``agents_scaling.study.evaluation``
may import this module.  Generate, selection, packet, policy, prompt and resource code
must never import it or open a path under ``data/protected/``; WP5's import-graph test
enforces this statically (corrections P1-4).  Files are written 0600 inside a 0700
directory by ``data/export.py``; readers refuse group/world-readable files so a copied
export cannot silently widen access.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from agents_scaling.study.data.bcb import ProtectedBcb
from agents_scaling.study.data.hle import ProtectedLabel
from agents_scaling.study.data.layout import (  # noqa: F401 (re-exported for evaluation code)
    BCB_TESTS_FILE,
    DIR_MODE,
    FILE_MODE,
    HLE_LABELS_FILE,
    PROTECTED_DIR,
    protected_dir,
)
from agents_scaling.study.types import ProtocolError


def _check_mode(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ProtocolError(f"{path} is group/world accessible (mode {oct(mode)}); protected files must be 0600")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"protected export missing: {path}")
    _check_mode(path)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{lineno}: row is not an object")
            rows.append(row)
    return rows


def load_protected_hle(run_root: str | os.PathLike) -> dict[str, ProtectedLabel]:
    """``{source_id: ProtectedLabel}`` from ``data/protected/hle_labels.jsonl``."""
    out: dict[str, ProtectedLabel] = {}
    for row in _read_jsonl(protected_dir(run_root) / HLE_LABELS_FILE):
        label = ProtectedLabel.from_dict(row)
        if label.source_id in out:
            raise ProtocolError(f"duplicate protected label {label.source_id}")
        out[label.source_id] = label
    return out


def load_protected_bcb(run_root: str | os.PathLike) -> dict[str, ProtectedBcb]:
    """``{source_id: ProtectedBcb}`` from ``data/protected/bcb_tests.jsonl``."""
    out: dict[str, ProtectedBcb] = {}
    for row in _read_jsonl(protected_dir(run_root) / BCB_TESTS_FILE):
        record = ProtectedBcb.from_dict(row)
        if record.source_id in out:
            raise ProtocolError(f"duplicate protected record {record.source_id}")
        out[record.source_id] = record
    return out


__all__ = [name for name in globals() if not name.startswith("_")]
