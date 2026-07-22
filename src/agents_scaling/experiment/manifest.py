"""Immutable sweep-manifest helpers.

The persisted ``cells.json`` is the authoritative index for an active run.  Config files
may evolve, but an in-flight array must never observe a regenerated ordering or silently
admit directories left over from an older grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agents_scaling.config import ExperimentCell


@dataclass(frozen=True)
class ManifestSnapshot:
    path: Path
    cells: tuple[ExperimentCell, ...]
    sha256: str

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.cells)


def load_manifest(run_root: str | Path, *, verify_frozen: bool = True) -> ManifestSnapshot:
    """Load the persisted manifest and optionally verify its frozen checksum."""
    root = Path(run_root)
    path = root / "cells.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    cells = tuple(ExperimentCell.from_dict(row) for row in rows)
    ids = [cell.cell_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate cell ids in manifest: {path}")

    frozen = root / "cells.sha256"
    if verify_frozen and frozen.exists():
        expected = frozen.read_text().strip().split()[0]
        if expected != digest:
            raise ValueError(
                f"manifest checksum mismatch for {path}: expected {expected}, got {digest}"
            )
    return ManifestSnapshot(path=path, cells=cells, sha256=digest)


def freeze_manifest(run_root: str | Path, *, overwrite: bool = False) -> Path:
    """Write ``cells.sha256`` atomically without modifying ``cells.json``."""
    from agents_scaling.experiment.io import atomic_write_text

    root = Path(run_root)
    snapshot = load_manifest(root, verify_frozen=False)
    target = root / "cells.sha256"
    if target.exists() and not overwrite:
        recorded = target.read_text().strip().split()[0]
        if recorded != snapshot.sha256:
            raise ValueError(
                f"refusing to replace mismatched frozen checksum in {target}; "
                "pass overwrite=True only for an audited migration"
            )
        return target
    atomic_write_text(target, f"{snapshot.sha256}  cells.json\n")
    return target


def manifested_cell_dirs(
    run_root: str | Path, *, include_unmanifested: bool = False
) -> tuple[list[Path], set[str], set[str]]:
    """Return selected directories plus ``(stale_ids, missing_ids)`` audit sets."""
    root = Path(run_root)
    snapshot = load_manifest(root)
    manifest_ids = set(snapshot.ids)
    cells_root = root / "cells"
    present = {
        path.name: path for path in cells_root.iterdir() if path.is_dir()
    } if cells_root.exists() else {}
    stale = set(present) - manifest_ids
    missing = manifest_ids - set(present)
    selected = manifest_ids | stale if include_unmanifested else manifest_ids
    return [present[cid] for cid in sorted(selected) if cid in present], stale, missing
