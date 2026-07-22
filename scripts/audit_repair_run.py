#!/usr/bin/env python
"""Audit or safely repair manifested sweep-cell artifacts.

Dry-run is the default.  ``--apply`` acquires the same per-cell lock as the runner,
atomically rewrites ``results.jsonl`` to the first valid row per expected QID, and moves
an invalid ``meta.json`` into a recoverable quarantine directory.  A resumed worker then
fills only missing QIDs and publishes fresh metadata through the normal completion path.
Unmanifested/pre-trim directories are reported but never mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

# Permit the documented ``python scripts/...`` invocation from a clean release
# worktree without relying on an editable installation or ambient PYTHONPATH.
REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.experiment import io  # noqa: E402
from agents_scaling.experiment.completion import (
    FAILURE_FILENAME,
    META_FILENAME,
    RESULTS_FILENAME,
    CellLockUnavailable,
    CompletionState,
    canonicalize_results,
    cell_lock,
    get_completion_status,
    quarantine_failure,
    quarantine_metadata,
)  # noqa: E402
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest  # noqa: E402


REPAIR_INCIDENT_FILENAME = "canonical_repair_incident_v1.json"
REPAIR_PREIMAGE_DIRECTORY = Path("recovery") / "canonical_repair_preimages"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_jsonl(records: tuple[dict[str, Any], ...]) -> bytes:
    return "".join(
        json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def _artifact_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"refusing unsafe artifact: {path}")
    return _sha256(path.read_bytes())


def _archive_results_preimage(
    run_root: Path,
    *,
    cell_id: str,
    source: Path,
    source_sha256: str,
    target_sha256: str,
    manifest_sha256: str,
) -> str:
    """Durably archive the exact results bytes and incident link before rewriting."""

    payload = source.read_bytes()
    if _sha256(payload) != source_sha256:
        raise RuntimeError(f"results changed before preimage archival: {source}")
    archive = run_root / REPAIR_PREIMAGE_DIRECTORY / source_sha256[:2] / (
        f"{source_sha256}.jsonl"
    )
    if archive.exists():
        if archive.is_symlink() or not archive.is_file() or archive.read_bytes() != payload:
            raise RuntimeError(f"preimage archive collision: {archive}")
    else:
        io.atomic_write_text(archive, payload.decode("utf-8"))
        archive.chmod(stat.S_IMODE(archive.stat().st_mode) & ~0o222)
    relative_archive = archive.relative_to(run_root).as_posix()

    incident_path = run_root / REPAIR_INCIDENT_FILENAME
    if incident_path.exists():
        if incident_path.is_symlink() or not incident_path.is_file():
            raise RuntimeError(f"refusing unsafe repair incident: {incident_path}")
        incident = json.loads(incident_path.read_text(encoding="utf-8"))
    else:
        incident = {
            "schema_version": 1,
            "incident_type": "canonical_results_repair",
            "run_id": run_root.name,
            "manifest_sha256": manifest_sha256,
            "rewrites": {},
        }
    if (
        incident.get("schema_version") != 1
        or incident.get("incident_type") != "canonical_results_repair"
        or incident.get("run_id") != run_root.name
        or incident.get("manifest_sha256") != manifest_sha256
        or not isinstance(incident.get("rewrites"), dict)
    ):
        raise RuntimeError(f"incompatible repair incident: {incident_path}")
    evidence_id = _sha256(
        f"{cell_id}\0{source_sha256}\0{target_sha256}".encode("utf-8")
    )
    expected = {
        "evidence_id": evidence_id,
        "cell_id": cell_id,
        "source_results_sha256": source_sha256,
        "canonical_results_sha256": target_sha256,
        "preimage_archive": relative_archive,
    }
    existing = incident["rewrites"].get(evidence_id)
    if existing is not None and existing != expected:
        raise RuntimeError(f"repair evidence collision: {evidence_id}")
    incident["rewrites"][evidence_id] = expected
    io.write_json(incident_path, incident)
    verified = json.loads(incident_path.read_text(encoding="utf-8"))
    if verified != incident or archive.read_bytes() != payload:
        raise RuntimeError(f"repair preimage verification failed for {cell_id}")
    return relative_archive


def _planned_quarantines(status: Any, cdir: Path) -> tuple[bool, bool]:
    lowered = [str(error).lower() for error in getattr(status, "errors", ())]
    metadata = (
        (cdir / META_FILENAME).exists()
        and status.status is not CompletionState.COMPLETE
        and any("meta" in error for error in lowered)
    )
    failure = (
        (cdir / FAILURE_FILENAME).exists()
        and status.status is CompletionState.CORRUPT
        and any("failure" in error for error in lowered)
    )
    return metadata, failure


def audit_run(run_root: Path, *, apply: bool, excluded_cell_ids: set[str] | None = None) -> dict:
    snapshot = load_manifest(run_root)
    question_catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    excluded = set(excluded_cell_ids or ())
    if apply:
        freeze_manifest(run_root)
    cells_root = run_root / "cells"
    present_dirs = {
        path.name for path in cells_root.iterdir() if path.is_dir()
    } if cells_root.exists() else set()
    stale = present_dirs - set(snapshot.ids)
    counts: Counter[str] = Counter()
    rewrites = meta_quarantined = failure_quarantined = locked = 0
    results_would_rewrite = metadata_would_quarantine = failure_would_quarantine = 0
    change_plan: list[dict[str, Any]] = []
    no_repairable_artifacts = 0
    explicitly_skipped = 0

    for cell in snapshot.cells:
        cdir = cells_root / cell.cell_id
        questions = question_catalog.questions_for(cell)
        qids = tuple(question.qid for question in questions)
        before = get_completion_status(
            cell,
            cdir,
            expected_qids=qids,
            expected_questions=questions,
            verified_benchmark_contracts=question_catalog.frozen,
            verified_manifest=question_catalog.snapshot,
        )
        counts[before.status.value] += 1
        if cell.cell_id in excluded:
            explicitly_skipped += 1
            continue
        if before.status in {CompletionState.COMPLETE, CompletionState.ACTIVE}:
            continue
        # ``cell_lock`` creates its directory and lock file.  A repair audit must not
        # materialize every wholly missing manifest cell merely to discover that there
        # is nothing to repair.  This pre-lock check is deliberately limited to the
        # artifacts this tool can mutate.  If a worker creates one just after the check,
        # the audit safely defers it to a later pass; if one already exists, the shared
        # lock below arbitrates with the worker before any mutation.
        if not any(
            (cdir / filename).exists()
            for filename in (RESULTS_FILENAME, META_FILENAME, FAILURE_FILENAME)
        ):
            no_repairable_artifacts += 1
            continue
        parsed = canonicalize_results(
            cell,
            cdir,
            expected_qids=qids,
            expected_questions=questions,
            verified_benchmark_contracts=question_catalog.frozen,
            verified_manifest=question_catalog.snapshot,
            write=False,
        )
        results_path = cdir / RESULTS_FILENAME
        source_results_sha = _artifact_digest(results_path)
        canonical_payload = _canonical_jsonl(parsed.records)
        canonical_results_sha = _sha256(canonical_payload)
        planned_results = bool(results_path.exists() and parsed.needs_rewrite)
        planned_meta, planned_failure = _planned_quarantines(before, cdir)
        if planned_results:
            results_would_rewrite += 1
        if planned_meta:
            metadata_would_quarantine += 1
        if planned_failure:
            failure_would_quarantine += 1
        if planned_results or planned_meta or planned_failure:
            change_plan.append(
                {
                    "cell_id": cell.cell_id,
                    "results": {
                        "would_rewrite": planned_results,
                        "before_sha256": source_results_sha,
                        "after_sha256": canonical_results_sha if planned_results else source_results_sha,
                        "raw_rows": parsed.raw_rows,
                        "canonical_rows": len(parsed.records),
                        "malformed_lines": parsed.malformed_lines,
                        "invalid_rows": parsed.invalid_rows,
                        "duplicate_qids": list(parsed.duplicate_qids),
                        "unexpected_qids": list(parsed.unexpected_qids),
                        "out_of_order": parsed.out_of_order,
                    },
                    "metadata": {
                        "would_quarantine": planned_meta,
                        "before_sha256": _artifact_digest(cdir / META_FILENAME),
                        "after_sha256": None if planned_meta else _artifact_digest(cdir / META_FILENAME),
                    },
                    "failure": {
                        "would_quarantine": planned_failure,
                        "before_sha256": _artifact_digest(cdir / FAILURE_FILENAME),
                        "after_sha256": None if planned_failure else _artifact_digest(cdir / FAILURE_FILENAME),
                    },
                }
            )
        if not apply or not (planned_results or planned_meta or planned_failure):
            continue
        try:
            with cell_lock(cdir):
                # Recompute every decision inside the authoritative lock.  The dry-run
                # plan is evidence for the operator, never permission to mutate stale
                # bytes observed before acquiring ownership.
                locked_before = get_completion_status(
                    cell,
                    cdir,
                    expected_qids=qids,
                    expected_questions=questions,
                    verified_benchmark_contracts=question_catalog.frozen,
                    verified_manifest=question_catalog.snapshot,
                    check_active=False,
                )
                locked_parsed = canonicalize_results(
                    cell,
                    cdir,
                    expected_qids=qids,
                    expected_questions=questions,
                    verified_benchmark_contracts=question_catalog.frozen,
                    verified_manifest=question_catalog.snapshot,
                    write=False,
                )
                if (cdir / RESULTS_FILENAME).exists() and locked_parsed.needs_rewrite:
                    locked_source_sha = _artifact_digest(cdir / RESULTS_FILENAME)
                    assert locked_source_sha is not None
                    locked_target_sha = _sha256(_canonical_jsonl(locked_parsed.records))
                    _archive_results_preimage(
                        run_root,
                        cell_id=cell.cell_id,
                        source=cdir / RESULTS_FILENAME,
                        source_sha256=locked_source_sha,
                        target_sha256=locked_target_sha,
                        manifest_sha256=snapshot.sha256,
                    )
                    canonicalize_results(
                        cell,
                        cdir,
                        expected_qids=qids,
                        expected_questions=questions,
                        verified_benchmark_contracts=question_catalog.frozen,
                        verified_manifest=question_catalog.snapshot,
                        write=True,
                    )
                    rewrites += 1
                # Completion is reassessed after canonicalization.  If metadata still
                # claims completion without satisfying the semantic contract, preserve it
                # for audit and let the resumable runner publish a clean replacement.
                after = get_completion_status(
                    cell,
                    cdir,
                    expected_qids=qids,
                    expected_questions=questions,
                    verified_benchmark_contracts=question_catalog.frozen,
                    verified_manifest=question_catalog.snapshot,
                    check_active=False,
                )
                locked_meta, locked_failure = _planned_quarantines(after, cdir)
                if locked_meta:
                    if quarantine_metadata(cdir) is not None:
                        meta_quarantined += 1
                if locked_failure:
                    if quarantine_failure(cdir) is not None:
                        failure_quarantined += 1
        except CellLockUnavailable:
            locked += 1

    return {
        "run_root": str(run_root),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": question_catalog.sidecar_sha256,
        "manifest_cells": len(snapshot.cells),
        "states_before": dict(sorted(counts.items())),
        "stale_unmanifested_dirs": len(stale),
        "results_rewritten": rewrites,
        "meta_quarantined": meta_quarantined,
        "failure_quarantined": failure_quarantined,
        "results_would_rewrite": results_would_rewrite,
        "metadata_would_quarantine": metadata_would_quarantine,
        "failure_would_quarantine": failure_would_quarantine,
        "would_change_cells": sorted(change_plan, key=lambda item: item["cell_id"]),
        "active_or_locked_skipped": locked,
        "no_repairable_artifacts_skipped": no_repairable_artifacts,
        "explicitly_excluded": explicitly_skipped,
        "applied": apply,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    parser.add_argument("--apply", action="store_true", help="perform lock-protected repairs")
    parser.add_argument(
        "--exclude-cell-id",
        action="append",
        default=[],
        help="cell id known to be active in a legacy worker without advisory locking",
    )
    parser.add_argument(
        "--exclude-cell-file",
        type=Path,
        help="newline-delimited legacy-active cell ids to exclude from mutation",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    excluded = set(args.exclude_cell_id)
    if args.exclude_cell_file is not None:
        excluded.update(
            line.strip()
            for line in args.exclude_cell_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    reports = [
        audit_run(root / run_id, apply=args.apply, excluded_cell_ids=excluded)
        for run_id in args.run_id
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
