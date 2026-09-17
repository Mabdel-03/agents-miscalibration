"""``allocation grade --run-dir DIR --resume [--follow] [--workers N] [--config PATH]`` (agent 05).

Scans ``parsed/``; runs public checks for every code candidate first, then private grades (code via
``grade_code``; text via ``agent_allocation.grading.text.grade.grade_text``, imported lazily). Writes
``public_checks/{shard}/{candidate_id}.json`` (PublicCheckResult) and ``grades/{shard}/{candidate_id}.json``
(GradeResult, 0700 tree) through ``dump_record``. CPU work runs in a spawn process pool of
``min(--workers or grading.offline_workers_max, grading.offline_workers_max, max(1, affinity - 2))`` processes;
the parent is the only writer.

Cache: code verdicts are cached by sha256 of (source revision, item, code sha256, test kind, evaluator version,
environment digest, public-inputs hash / private hash). Only real verdicts are cached (OK checks, GRADED code
grades); outages are never cached and are redone on ``--resume``. Identical code shares a verdict, never an API
draw: every candidate keeps its own record.

Resume: an existing record with a terminal status is kept; INFRASTRUCTURE_ERROR / EVALUATOR_ERROR / NOT_EVALUATED
records are redone. Without ``--resume`` the command refuses to run over existing records.

Config keys consumed: ``grading.*`` via ``grading.code.profile.profile_from_config`` (the run's config lock when
present, else ``--config``, else ``configs/balanced150.json``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent_allocation.contracts import (BENCHMARKS, Candidate, CheckStatus, GradeResult, GradeStatus, OutputKind,
                                        PrivateEvalSpec, PublicCheckResult, PublicTask, canonical_hash, dump_record,
                                        load_record)
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.paths import PRIVATE_DIR_MODE, RunPaths

EXIT_BLOCKED = 2
TERMINAL_CHECK = {CheckStatus.OK, CheckStatus.INVALID_CANDIDATE}
TERMINAL_GRADE = {GradeStatus.GRADED, GradeStatus.INVALID_OUTPUT}


# ---------------------------------------------------------------------------------------------------------------
# Paths (shim: requested as RunPaths.grade_cache_dir in docs/requests/05.md)
# ---------------------------------------------------------------------------------------------------------------

def grade_cache_dir(paths: RunPaths) -> Path:
    return paths.run_dir / "grade_cache"


def _atomic_write(path: Path, payload: Mapping[str, Any], *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(path.parent, PRIVATE_DIR_MODE)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_record(path: Path, cls: type) -> Any | None:
    try:
        return load_record(cls, json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, KeyError):
        return None


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------

def _load_jsonl(path: Path, cls: type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = load_record(cls, json.loads(line))
                out[record.item_id] = record
    return out


def load_tasks(paths: RunPaths) -> tuple[dict[str, PublicTask], dict[str, PrivateEvalSpec]]:
    public: dict[str, PublicTask] = {}
    private: dict[str, PrivateEvalSpec] = {}
    for benchmark in BENCHMARKS:
        public.update(_load_jsonl(paths.public_manifest(benchmark), PublicTask))
        private.update(_load_jsonl(paths.private_manifest(benchmark), PrivateEvalSpec))
    return public, private


def scan_candidates(paths: RunPaths) -> list[Candidate]:
    parsed_root = paths.run_dir / "parsed"
    out: list[Candidate] = []
    if not parsed_root.is_dir():
        return out
    for path in sorted(parsed_root.glob("*/*.json")):
        payload = json.loads(path.read_text())
        candidate = load_record(Candidate, payload["candidate"])
        if paths.parsed(candidate.node_id) != path or candidate.candidate_id != candidate.node_id:
            raise ValueError(f"{path}: candidate identity does not match its parsed path")
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------------------------------------------

def public_cache_key(task: PublicTask, code_sha: str, env: str) -> str:
    from agent_allocation.grading.code.public import CHECKER_VERSION, public_inputs_sha256
    return canonical_hash({"revision": task.source_revision, "item_id": task.item_id, "code_sha256": code_sha,
                           "test_kind": "public_examples", "evaluator_version": CHECKER_VERSION,
                           "environment_digest": env, "public_inputs_sha256": public_inputs_sha256(task)})


def private_cache_key(private: PrivateEvalSpec, code_sha: str, env: str) -> str:
    from agent_allocation.grading.code.private import GRADER_VERSION
    return canonical_hash({"revision": private.source_revision, "item_id": private.item_id, "code_sha256": code_sha,
                           "test_kind": private.eval_kind.value, "evaluator_version": GRADER_VERSION,
                           "environment_digest": env, "private_hash": private.private_hash})


def _cache_path(paths: RunPaths, kind: str, key: str) -> Path:
    return grade_cache_dir(paths) / kind / key[:2] / f"{key}.json"


# ---------------------------------------------------------------------------------------------------------------
# Pool jobs (run in spawned worker processes)
# ---------------------------------------------------------------------------------------------------------------

def _init_worker(profile: profile_mod.GradingProfile) -> None:
    profile_mod.configure(profile)


def _job_public(candidate_data: dict, task_data: dict) -> dict:
    from agent_allocation.grading.code.public import run_public_checks
    candidate = load_record(Candidate, candidate_data)
    task = load_record(PublicTask, task_data)
    return dump_record(run_public_checks(candidate, task))


def _job_grade(candidate_data: dict, private_data: dict, run_dir: str, locks_dir: str, local: str) -> dict:
    candidate = load_record(Candidate, candidate_data)
    private = load_record(PrivateEvalSpec, private_data)
    paths = RunPaths(Path(run_dir), Path(locks_dir), Path(local))
    if private.benchmark == "mbpp_plus":
        from agent_allocation.grading.code.private import grade_code
        return dump_record(grade_code(candidate, private, paths))
    from agent_allocation.grading.text.grade import grade_text   # agent 04; lazy on purpose
    return dump_record(grade_text(candidate, private))


# ---------------------------------------------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------------------------------------------

def _resolve_profile(paths: RunPaths, config: str | None) -> profile_mod.GradingProfile:
    if config:
        source = Path(config)
    elif paths.config_lock.exists():
        source = paths.config_lock
    else:
        source = profile_mod.DEFAULT_CONFIG
    cfg = json.loads(source.read_text())
    cfg = cfg.get("config", cfg) if "grading" not in cfg else cfg
    return profile_mod.profile_from_config(cfg)


def worker_count(profile: profile_mod.GradingProfile, requested: int | None) -> int:
    affinity = len(os.sched_getaffinity(0))
    cap = min(profile.workers_max, max(1, affinity - 2))
    return max(1, min(cap, requested)) if requested else cap


def grade_pass(paths: RunPaths, profile: profile_mod.GradingProfile, *, resume: bool, workers: int,
               log=print) -> dict[str, int]:
    """One scan: public checks for code candidates first, then private grades. Returns counters."""
    from agent_allocation.grading.code.public import code_sha256

    profile_mod.configure(profile)
    env = profile_mod.environment_digest(profile)
    public_tasks, private_specs = load_tasks(paths)
    candidates = scan_candidates(paths)
    counts = {"candidates": len(candidates), "public_written": 0, "public_cached": 0, "public_skipped": 0,
              "grades_written": 0, "grades_cached": 0, "grades_skipped": 0, "blocked": 0, "unknown_item": 0}
    if not resume:
        existing = [c for c in candidates if paths.public_check(c.candidate_id).exists()
                    or paths.grade(c.candidate_id).exists()]
        if existing:
            raise SystemExit(f"{len(existing)} candidates already have checks/grades; rerun with --resume")
    paths.grades_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(paths.grades_dir, PRIVATE_DIR_MODE)
    grade_cache_dir(paths).mkdir(parents=True, exist_ok=True)
    os.chmod(grade_cache_dir(paths), PRIVATE_DIR_MODE)

    public_jobs: list[tuple[Candidate, PublicTask, str | None]] = []
    grade_jobs: list[tuple[Candidate, PrivateEvalSpec, str | None]] = []
    for candidate in candidates:
        task = public_tasks.get(candidate.item_id)
        private = private_specs.get(candidate.item_id)
        if task is None or private is None:
            counts["unknown_item"] += 1
            continue
        is_code = task.output_kind is OutputKind.PYTHON
        program = candidate.final_answer
        sha = code_sha256(program) if (is_code and program) else None
        if is_code:
            existing = _read_record(paths.public_check(candidate.candidate_id), PublicCheckResult)
            if existing is not None and existing.status in TERMINAL_CHECK:
                counts["public_skipped"] += 1
            else:
                public_jobs.append((candidate, task, sha))
        existing_grade = _read_record(paths.grade(candidate.candidate_id), GradeResult)
        if existing_grade is not None and existing_grade.status in TERMINAL_GRADE:
            counts["grades_skipped"] += 1
        else:
            grade_jobs.append((candidate, private, sha))

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(profile,)) as pool:
        # ---- public checks first --------------------------------------------------------------------------
        pending = []
        for candidate, task, sha in public_jobs:
            key = public_cache_key(task, sha, env) if (sha and profile.cache_by_code_hash) else None
            cached = _read_cached(paths, "public", key, PublicCheckResult)
            if cached is not None:
                record = dataclasses.replace(cached, candidate_id=candidate.candidate_id)
                _atomic_write(paths.public_check(candidate.candidate_id), dump_record(record), private=False)
                counts["public_cached"] += 1
                continue
            pending.append((candidate, key, pool.submit(_job_public, dump_record(candidate), dump_record(task))))
        for candidate, key, future in pending:
            record = load_record(PublicCheckResult, future.result())
            _atomic_write(paths.public_check(candidate.candidate_id), dump_record(record), private=False)
            counts["public_written"] += 1
            if key and record.status is CheckStatus.OK:
                _atomic_write(_cache_path(paths, "public", key), dump_record(record), private=True)
        # ---- private grades ---------------------------------------------------------------------------------
        pending = []
        text_grader_ok = True
        for candidate, private, sha in grade_jobs:
            if private.benchmark != "mbpp_plus" and text_grader_ok:
                try:
                    import agent_allocation.grading.text.grade  # noqa: F401
                except ImportError:
                    text_grader_ok = False
            if private.benchmark != "mbpp_plus" and not text_grader_ok:
                counts["blocked"] += 1
                continue
            key = private_cache_key(private, sha, env) if (sha and private.benchmark == "mbpp_plus"
                                                           and profile.cache_by_code_hash) else None
            cached = _read_cached(paths, "private", key, GradeResult)
            if cached is not None:
                record = dataclasses.replace(cached, candidate_id=candidate.candidate_id)
                _atomic_write(paths.grade(candidate.candidate_id), dump_record(record), private=True)
                counts["grades_cached"] += 1
                continue
            pending.append((candidate, key, pool.submit(_job_grade, dump_record(candidate), dump_record(private),
                                                        str(paths.run_dir), str(paths.locks_dir),
                                                        str(paths.local_state_dir))))
        for candidate, key, future in pending:
            record = load_record(GradeResult, future.result())
            _atomic_write(paths.grade(candidate.candidate_id), dump_record(record), private=True)
            counts["grades_written"] += 1
            if key and record.status is GradeStatus.GRADED:
                _atomic_write(_cache_path(paths, "private", key), dump_record(record), private=True)
    if counts["blocked"]:
        log(f"blocked: {counts['blocked']} text candidates need agent_allocation.grading.text.grade")
    return counts


def _read_cached(paths: RunPaths, kind: str, key: str | None, cls: type) -> Any | None:
    if not key:
        return None
    return _read_record(_cache_path(paths, kind, key), cls)


def cmd_grade(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation grade", description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--follow", action="store_true", help="keep scanning parsed/ until control/STOP exists")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    paths = RunPaths.for_run(args.run_dir)
    profile = _resolve_profile(paths, args.config)
    workers = worker_count(profile, args.workers)
    resume = args.resume
    while True:
        counts = grade_pass(paths, profile, resume=resume, workers=workers)
        print(json.dumps({"grade_pass": counts, "workers": workers}, sort_keys=True), flush=True)
        if not args.follow or paths.stop_sentinel.exists():
            break
        resume = True
        time.sleep(args.poll_seconds)
    return EXIT_BLOCKED if counts["blocked"] or counts["unknown_item"] else 0


if __name__ == "__main__":
    raise SystemExit(cmd_grade(sys.argv[1:]))
