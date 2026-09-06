"""BigCodeBench evaluator: one ``apptainer exec`` per call around the in-container driver (WP5).

Spec §3.3 (official tests in a fresh environment), §10.6 (network-less, bounded, no host
credentials).  06_bcb_container.md (the VERIFIED exec line, the SIF path, the cache
environment the driver sets).  Architecture §1.12 (unique code strings evaluated once per
item); corrections P1-5 (``program = strip_one_fence(final_answer)`` — the same frozen
rule as ``selection.normalize.code_key`` — is applied *here*, before the job is written).

``BcbEvaluator(image_sif, timeout_s=120, mem_bytes=8 GiB).evaluate_many(jobs)`` writes
``jobs.jsonl`` + the driver into a fresh work directory, runs exactly one container exec,
and reads ``results.jsonl`` back.  ``container="none"`` runs the driver in-process with the
current interpreter (tests, and the local smoke); ``apptainer_argv`` is the exact argv so
tests can pin it.  The evaluator never pulls images (cells never pull).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.data.protected import load_protected_bcb  # evaluator identity only
from agents_scaling.study.parse.candidate import strip_one_fence
from agents_scaling.study.types import CandidateRecord, InfraFailure, ProtocolError

DEFAULT_SIF = Path("/orcd/data/tpoggio/001/mabdel03/containers/bcb-evaluate-v0.2.4.sif")
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MEM_BYTES = 8 * 1024 ** 3
DRIVER_NAME = "bcb_container_driver.py"
DRIVER_SOURCE = Path(__file__).with_name(DRIVER_NAME)
STATUSES: tuple[str, ...] = ("pass", "fail", "timeout", "error")
EVAL_BCB_SUBDIR = ("eval", "bcb")
CONTAINER_MODES: tuple[str, ...] = ("apptainer", "none")


def apptainer_argv(sif: str | os.PathLike, workdir: str | os.PathLike, apptainer_bin: str = "apptainer") -> list[str]:
    """The verified exec line of 06_bcb_container.md, byte-for-byte in argv form."""
    return [
        apptainer_bin,
        "exec",
        "--containall",
        "--cleanenv",
        "--no-home",
        "--net",
        "--network",
        "none",
        "--bind",
        f"{Path(workdir)}:/work",
        "--pwd",
        "/work",
        str(sif),
        "python3",
        f"/work/{DRIVER_NAME}",
        "/work/jobs.jsonl",
        "/work/results.jsonl",
    ]


def program_of(final_answer: str) -> str:
    """P1-5 / E3': the single frozen code-string rule (identical to ``code_key``)."""
    return strip_one_fence(final_answer)


def dedupe_jobs(jobs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Collapse identical ``(code, test)`` pairs → ``(unique jobs, {cid: representative cid})``."""
    reps: dict[str, str] = {}
    unique: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for job in jobs:
        cid, code, test = str(job["cid"]), str(job["code"]), str(job["test_src"] if "test_src" in job else job["test"])
        key = identity.sha256_hex(identity.jcs([code, test]))
        if key in seen:
            reps[cid] = seen[key]
            continue
        seen[key] = cid
        reps[cid] = cid
        unique.append({"cid": cid, "code": code, "test": test})
    return unique, reps


class BcbEvaluator:
    """See the module docstring.  ``work_root`` (default: a fresh temp dir under the results
    tree passed by the runner) receives one throw-away work directory per call."""

    def __init__(
        self,
        image_sif: str | os.PathLike = DEFAULT_SIF,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mem_bytes: int = DEFAULT_MEM_BYTES,
        *,
        container: str = "apptainer",
        apptainer_bin: str = "apptainer",
        work_root: str | os.PathLike | None = None,
        exec_timeout_s: float | None = None,
        keep_work: bool = False,
    ) -> None:
        if container not in CONTAINER_MODES:
            raise ValueError(f"container must be one of {CONTAINER_MODES}")
        if timeout_s <= 0 or mem_bytes <= 0:
            raise ValueError("timeout_s and mem_bytes must be positive")
        self.image_sif = Path(image_sif)
        self.timeout_s = float(timeout_s)
        self.mem_bytes = int(mem_bytes)
        self.container = container
        self.apptainer_bin = apptainer_bin
        self.work_root = Path(work_root) if work_root is not None else None
        self.exec_timeout_s = exec_timeout_s
        self.keep_work = keep_work
        if container == "apptainer" and not self.image_sif.exists():
            raise InfraFailure(f"evaluator image {self.image_sif} is missing (cells never pull; see 06_bcb_container.md)")

    # ---- one container exec -----------------------------------------------------------------
    def argv(self, workdir: Path) -> list[str]:
        if self.container == "apptainer":
            return apptainer_argv(self.image_sif, workdir, self.apptainer_bin)
        return [
            sys.executable,
            str(workdir / DRIVER_NAME),
            str(workdir / "jobs.jsonl"),
            str(workdir / "results.jsonl"),
            "--timeout",
            str(self.timeout_s),
            "--mem",
            str(self.mem_bytes),
            "--python",
            sys.executable,
        ]

    def _workdir(self) -> Path:
        if self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="bcb_", dir=self.work_root))
        return Path(tempfile.mkdtemp(prefix="bcb_"))

    def evaluate_many(self, jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """``{cid: {status, stdout_tail, stderr_tail, elapsed, returncode}}`` for every job.

        Jobs: ``{"cid", "code", "test_src"}`` (``test`` accepted as an alias).  Identical
        ``(code, test)`` pairs are evaluated once and the verdict copied.  Every input cid is
        present in the output; a job the driver did not report is ``error`` (never silently
        dropped).  Exceptions of the container layer itself raise :class:`InfraFailure`.
        """
        unique, reps = dedupe_jobs(jobs)
        if not unique:
            return {}
        work = self._workdir()
        try:
            shutil.copyfile(DRIVER_SOURCE, work / DRIVER_NAME)
            with open(work / "jobs.jsonl", "w", encoding="utf-8") as handle:
                for job in unique:
                    handle.write(json.dumps(job, ensure_ascii=False) + "\n")
            argv = self.argv(work)
            # Inside the container the driver reads the per-job limits from its CLI defaults
            # unless overridden; pass them explicitly for both modes.
            if self.container == "apptainer":
                argv = argv + ["--timeout", str(self.timeout_s), "--mem", str(self.mem_bytes)]
            budget = self.exec_timeout_s or (self.timeout_s * len(unique) + 600.0)
            started = time.time()
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=budget, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise InfraFailure(f"evaluator exec failed: {type(exc).__name__}: {exc}") from exc
            results: dict[str, dict[str, Any]] = {}
            results_path = work / "results.jsonl"
            if results_path.exists():
                for line in results_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("status") not in STATUSES:
                            raise ProtocolError(f"driver reported unknown status {row.get('status')!r}")
                        results[row["cid"]] = row
            if proc.returncode != 0 and len(results) < len(unique):
                raise InfraFailure(
                    f"evaluator exec exited {proc.returncode} after {time.time() - started:.0f}s with "
                    f"{len(results)}/{len(unique)} verdicts: {proc.stderr[-2000:]}"
                )
            out: dict[str, dict[str, Any]] = {}
            for job in unique:
                row = results.get(job["cid"])
                if row is None:
                    row = {"cid": job["cid"], "status": "error", "returncode": None, "stdout_tail": "", "stderr_tail": "driver reported no verdict", "elapsed": 0.0}
                out[job["cid"]] = row
            for cid, rep in reps.items():
                if cid != rep:
                    out[cid] = {**out[rep], "cid": cid, "deduplicated_from": rep}
            return out
        finally:
            if not self.keep_work:
                shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- cell-level helpers


def bcb_eval_path(run_root: str | os.PathLike, cell_id: str) -> Path:
    return Path(run_root).joinpath(*EVAL_BCB_SUBDIR) / f"{cell_id}.jsonl"


def jobs_for_item(source_id: str, test_src: str, candidates: Sequence[CandidateRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    """Jobs for the valid candidates of one item (fence-stripped programs) and the invalid ids."""
    jobs: list[dict[str, Any]] = []
    invalid: list[str] = []
    for cand in candidates:
        if not cand.valid or cand.candidate is None:
            invalid.append(cand.candidate_id)
            continue
        jobs.append({"cid": cand.candidate_id, "code": program_of(cand.candidate.final_answer), "test_src": test_src})
    return jobs, invalid


def load_tests(run_root: str | os.PathLike) -> dict[str, str]:
    """``{source_id: hidden unittest source}`` (protected; evaluator identity only)."""
    return {sid: rec.test for sid, rec in load_protected_bcb(run_root).items()}


def iter_bcb_eval_rows(run_root: str | os.PathLike, cell_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Every verdict row of ``eval/bcb/*.jsonl`` (all cells or the given ones), validated."""
    folder = Path(run_root).joinpath(*EVAL_BCB_SUBDIR)
    rows: list[dict[str, Any]] = []
    if not folder.exists():
        return rows
    paths = [folder / f"{cid}.jsonl" for cid in cell_ids] if cell_ids is not None else sorted(folder.glob("*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in STATUSES:
                raise ProtocolError(f"{path}: unknown status {row.get('status')!r}")
            rows.append(row)
    return rows


def load_bcb_eval(run_root: str | os.PathLike, cell_ids: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """``{candidate_id: row}`` (one verdict per candidate; a conflicting pair is a ``ProtocolError``)."""
    out: dict[str, dict[str, Any]] = {}
    for row in iter_bcb_eval_rows(run_root, cell_ids):
        prev = out.get(row["candidate_id"])
        if prev is not None and prev["status"] != row["status"]:
            raise ProtocolError(f"{row['candidate_id'][:12]}: conflicting BCB verdicts {prev['status']} vs {row['status']}")
        out[row["candidate_id"]] = row
    return out


__all__ = [
    "CONTAINER_MODES",
    "DEFAULT_MEM_BYTES",
    "DEFAULT_SIF",
    "DEFAULT_TIMEOUT_S",
    "DRIVER_NAME",
    "DRIVER_SOURCE",
    "EVAL_BCB_SUBDIR",
    "STATUSES",
    "BcbEvaluator",
    "apptainer_argv",
    "bcb_eval_path",
    "dedupe_jobs",
    "iter_bcb_eval_rows",
    "jobs_for_item",
    "load_bcb_eval",
    "load_tests",
    "program_of",
]
