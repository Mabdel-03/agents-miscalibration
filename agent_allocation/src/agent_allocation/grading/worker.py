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

Resume: an existing record with a terminal status is kept (for code: only if produced by the current checker/grader
version); INFRASTRUCTURE_ERROR / EVALUATOR_ERROR / NOT_EVALUATED records are redone. A verdict from a different
version is moved to ``paths.superseded_public_check`` / ``paths.superseded_grade`` before the new one is written (D9). Without ``--resume`` the command refuses (exit 2) to run over existing records. Exit 2 also
when text grading is blocked (agent 04's module absent), an item is unknown, or an outage record remains.

Config keys consumed: ``grading.*`` via ``grading.code.profile.profile_from_config`` (the run's config lock when
present, else ``--config``, else ``configs/balanced150.json``).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.contracts import (BENCHMARKS, Candidate, CheckStatus, GradeResult, GradeStatus, OutputKind,
                                        PrivateEvalSpec, PublicCheckResult, PublicTask, canonical_hash, dump_record,
                                        load_record)
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.paths import PRIVATE_DIR_MODE, RunPaths

EXIT_BLOCKED = 2
TERMINAL_CHECK = {CheckStatus.OK, CheckStatus.INVALID_CANDIDATE}
TERMINAL_GRADE = {GradeStatus.GRADED, GradeStatus.INVALID_OUTPUT}


# ---------------------------------------------------------------------------------------------------------------
# Writing (single writer; D9: a verdict from a different checker/grader version is moved aside, never overwritten)
# ---------------------------------------------------------------------------------------------------------------

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


def write_public_check(paths: RunPaths, record: PublicCheckResult) -> None:
    current = paths.public_check(record.candidate_id)
    previous = _read_record(current, PublicCheckResult)
    if previous is not None and previous.checker_version != record.checker_version:
        _move_aside(current, paths.superseded_public_check(record.candidate_id, previous.checker_version), private=False)
    _atomic_write(current, dump_record(record), private=False)


def write_grade(paths: RunPaths, record: GradeResult) -> None:
    current = paths.grade(record.candidate_id)
    previous = _read_record(current, GradeResult)
    if previous is not None and previous.grader_version != record.grader_version:
        _move_aside(current, paths.superseded_grade(record.candidate_id, previous.grader_version), private=True)
    _atomic_write(current, dump_record(record), private=True)


def _move_aside(current: Path, target: Path, *, private: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if private:
        for directory in (target.parent, target.parent.parent, target.parent.parent.parent):
            os.chmod(directory, PRIVATE_DIR_MODE)
    if target.exists():
        raise FileExistsError(f"{target} already holds a superseded verdict; refusing to overwrite it")
    os.replace(current, target)


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


def scan_candidates(paths: RunPaths, skip: set[str] | frozenset[str] = frozenset()) -> list[Candidate]:
    """Candidates from ``parsed/``; ``skip`` holds candidate ids already finished in an earlier pass (follow mode)."""
    parsed_root = paths.run_dir / "parsed"
    out: list[Candidate] = []
    if not parsed_root.is_dir():
        return out
    for path in sorted(parsed_root.glob("*/*.json")):
        if path.stem in skip:
            continue
        payload = json.loads(path.read_text())
        candidate = load_record(Candidate, payload["candidate"])
        if paths.parsed(candidate.node_id) != path or candidate.candidate_id != candidate.node_id:
            raise ValueError(f"{path}: candidate identity does not match its parsed path")
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------------------------------------------

def public_cache_material(task: PublicTask, code_sha: str, env: str) -> dict[str, str]:
    from agent_allocation.grading.code.public import CHECKER_VERSION, public_inputs_sha256
    return {"revision": task.source_revision, "item_id": task.item_id, "code_sha256": code_sha,
            "test_kind": "public_examples", "evaluator_version": CHECKER_VERSION, "environment_digest": env,
            "public_inputs_sha256": public_inputs_sha256(task)}


def private_cache_material(private: PrivateEvalSpec, code_sha: str, env: str) -> dict[str, str]:
    from agent_allocation.grading.code.private import GRADER_VERSION
    return {"revision": private.source_revision, "item_id": private.item_id, "code_sha256": code_sha,
            "test_kind": private.eval_kind.value, "evaluator_version": GRADER_VERSION, "environment_digest": env,
            "private_hash": private.private_hash}


def public_cache_key(task: PublicTask, code_sha: str, env: str) -> str:
    return canonical_hash(public_cache_material(task, code_sha, env))


def private_cache_key(private: PrivateEvalSpec, code_sha: str, env: str) -> str:
    return canonical_hash(private_cache_material(private, code_sha, env))


def _cache_path(paths: RunPaths, kind: str, key: str) -> Path:
    return paths.grade_cache_dir / kind / key[:2] / f"{key}.json"


def _write_cache(paths: RunPaths, kind: str, key: str, material: Mapping[str, str], record: Any) -> None:
    """Cache schema: ``grade_cache/{public|private}/{key[:2]}/{key}.json`` =
    ``{"key": sha256(canonical material), "material": {...}, "record": dump_record(verdict)}`` (0700 tree)."""
    _atomic_write(_cache_path(paths, kind, key), {"key": key, "material": dict(material), "record": dump_record(record)},
                  private=True)


# ---------------------------------------------------------------------------------------------------------------
# Pool jobs (run in spawned worker processes)
# ---------------------------------------------------------------------------------------------------------------

TEXT_GRADER = "agent_allocation.grading.text.grade:grade_text"      # agent 04; imported lazily on purpose


def _resolve_callable(spec: str):
    module_name, attribute = spec.split(":")
    return getattr(importlib.import_module(module_name), attribute)


def _init_worker(profile: profile_mod.GradingProfile) -> None:
    profile_mod.configure(profile)


def _job_public(candidate_data: dict, task_data: dict) -> dict:
    """An unexpected checker exception is an infrastructure error (selector pauses), never a failed case."""
    from agent_allocation.grading.code import public as public_mod
    candidate = load_record(Candidate, candidate_data)
    task = load_record(PublicTask, task_data)
    try:
        return dump_record(public_mod.run_public_checks(candidate, task))
    except Exception:
        from agent_allocation.grading.code.profile import active_profile, environment_digest
        n = len(task.public_examples)
        return dump_record(PublicCheckResult(
            candidate_id=candidate.candidate_id, item_id=task.item_id,
            code_sha256=public_mod.code_sha256(candidate.final_answer) if candidate.final_answer else None,
            status=CheckStatus.INFRASTRUCTURE_ERROR, cases_total=n, cases_passed=0, case_outcomes=("error",) * n,
            checker_version=public_mod.CHECKER_VERSION, public_inputs_sha256=public_mod.public_inputs_sha256(task),
            environment_digest=environment_digest(active_profile()), cpu_seconds=0.0))


def _job_code_grade(candidate_data: dict, private_data: dict, run_dir: str, locks_dir: str, local: str) -> dict:
    """An unexpected grader exception is an evaluator error (unknown), never a wrong answer."""
    from agent_allocation.grading.code import private as private_mod
    candidate = load_record(Candidate, candidate_data)
    private = load_record(PrivateEvalSpec, private_data)
    paths = RunPaths(Path(run_dir), Path(locks_dir), Path(local))
    try:
        return dump_record(private_mod.grade_code(candidate, private, paths))
    except Exception as exc:
        from agent_allocation.grading.code.profile import active_profile, environment_digest
        return dump_record(GradeResult(
            candidate_id=candidate.candidate_id, status=GradeStatus.EVALUATOR_ERROR, semantic_correct=None,
            operational_success=None, grader_version=private_mod.GRADER_VERSION,
            private_eval_hash=private.private_hash, detail_code=f"grader_exception:{type(exc).__name__}",
            environment_digest=environment_digest(active_profile()), cpu_seconds=None))


def _job_text_grade(candidate_data: dict, private_data: dict, grader_spec: str) -> dict:
    """Agent 04's grader owns text semantics; an exception escaping it is recorded as unknown, never as wrong."""
    candidate = load_record(Candidate, candidate_data)
    private = load_record(PrivateEvalSpec, private_data)
    try:
        return dump_record(_resolve_callable(grader_spec)(candidate, private))
    except Exception as exc:
        return dump_record(GradeResult(
            candidate_id=candidate.candidate_id, status=GradeStatus.EVALUATOR_ERROR, semantic_correct=None,
            operational_success=None, grader_version=f"{grader_spec}#uncaught-exception",
            private_eval_hash=private.private_hash, detail_code=f"grader_exception:{type(exc).__name__}",
            environment_digest=None, cpu_seconds=None))


# ---------------------------------------------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------------------------------------------

class GradeRefused(RuntimeError):
    pass


def _resolve_profile(paths: RunPaths, config: str | None) -> profile_mod.GradingProfile:
    if config:
        source = Path(config)
    elif paths.config_lock.exists():
        source = paths.config_lock
    else:
        source = profile_mod.DEFAULT_CONFIG
    cfg = json.loads(source.read_text())
    if "grading" not in cfg and isinstance(cfg.get("config"), Mapping):
        cfg = cfg["config"]
    return profile_mod.profile_from_config(cfg)


def worker_count(profile: profile_mod.GradingProfile, requested: int | None) -> int:
    affinity = len(os.sched_getaffinity(0))
    cap = min(profile.workers_max, max(1, affinity - 2))
    return max(1, min(cap, requested)) if requested else cap


def _read_cached(paths: RunPaths, kind: str, key: str | None, cls: type) -> Any | None:
    if not key:
        return None
    try:
        payload = json.loads(_cache_path(paths, kind, key).read_text())
        if payload.get("key") != key or canonical_hash(payload["material"]) != key:
            return None
        return load_record(cls, payload["record"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def grade_pass(paths: RunPaths, profile: profile_mod.GradingProfile, *, resume: bool, workers: int,
               text_grader: str = TEXT_GRADER, log=print, finished: set[str] | None = None) -> dict[str, int]:
    """One scan: public checks for code candidates first, then private grades. Returns counters.

    Identical code for the same item shares one sandbox run within a pass and across passes (cache); every
    candidate still gets its own record. The parent process is the only writer.
    """
    from agent_allocation.grading.code.private import GRADER_VERSION
    from agent_allocation.grading.code.public import CHECKER_VERSION, code_sha256

    profile_mod.configure(profile)
    env = profile_mod.environment_digest(profile)
    public_tasks, private_specs = load_tasks(paths)
    candidates = scan_candidates(paths, finished if finished is not None else frozenset())
    counts = {"candidates": len(candidates), "public_written": 0, "public_cached": 0, "public_skipped": 0,
              "grades_written": 0, "grades_cached": 0, "grades_skipped": 0, "blocked_text": 0, "unknown_item": 0,
              "infrastructure_errors": 0, "evaluator_errors": 0, "sandbox_denied": 0, "pool_failures": 0}
    if not resume:
        existing = [c for c in candidates if paths.public_check(c.candidate_id).exists()
                    or paths.grade(c.candidate_id).exists()]
        if existing:
            raise GradeRefused(f"{len(existing)} candidates already have checks/grades; rerun with --resume")
    for directory in (paths.grades_dir, paths.grade_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, PRIVATE_DIR_MODE)

    public_groups: dict[str, list[tuple[Candidate, PublicTask]]] = {}
    code_groups: dict[str, list[tuple[Candidate, PrivateEvalSpec]]] = {}
    text_jobs: list[tuple[Candidate, PrivateEvalSpec]] = []
    materials: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        task = public_tasks.get(candidate.item_id)
        private = private_specs.get(candidate.item_id)
        if task is None or private is None:
            counts["unknown_item"] += 1
            continue
        is_code = task.output_kind is OutputKind.PYTHON
        sha = code_sha256(candidate.final_answer) if (is_code and candidate.final_answer) else None
        if is_code:
            existing = _read_record(paths.public_check(candidate.candidate_id), PublicCheckResult)
            if existing is not None and existing.status in TERMINAL_CHECK and existing.checker_version == CHECKER_VERSION:
                counts["public_skipped"] += 1
            else:
                key = public_cache_key(task, sha, env) if sha else "nocache:" + candidate.candidate_id
                public_groups.setdefault(key, []).append((candidate, task))
                if sha:
                    materials[key] = public_cache_material(task, sha, env)
        existing_grade = _read_record(paths.grade(candidate.candidate_id), GradeResult)
        if existing_grade is not None and existing_grade.status in TERMINAL_GRADE \
                and (not is_code or existing_grade.grader_version == GRADER_VERSION):
            counts["grades_skipped"] += 1
        elif is_code:
            key = private_cache_key(private, sha, env) if sha else "nocache:" + candidate.candidate_id
            code_groups.setdefault(key, []).append((candidate, private))
            if sha:
                materials[key] = private_cache_material(private, sha, env)
        else:
            text_jobs.append((candidate, private))

    text_ok = True
    if text_jobs:
        try:
            _resolve_callable(text_grader)
        except (ImportError, AttributeError):
            text_ok = False
            counts["blocked_text"] = len(text_jobs)
            log(f"blocked: {len(text_jobs)} text candidates need {text_grader}")

    def cacheable(key: str) -> bool:
        return profile.cache_by_code_hash and not key.startswith("nocache:")

    pool = None
    try:
        # ---- public checks first ------------------------------------------------------------------------
        futures = {}
        for key, members in public_groups.items():
            cached = _read_cached(paths, "public", key, PublicCheckResult) if cacheable(key) else None
            if cached is not None:
                for candidate, _ in members:
                    write_public_check(paths, dataclasses.replace(cached, candidate_id=candidate.candidate_id))
                    counts["public_cached"] += 1
                continue
            pool = pool or _pool(workers, profile)
            candidate, task = members[0]
            futures[pool.submit(_job_public, dump_record(candidate), dump_record(task))] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = load_record(PublicCheckResult, future.result())
            except BrokenProcessPool:
                counts["pool_failures"] += 1            # no record written: the candidate is redone on resume
                continue
            for candidate, _ in public_groups[key]:
                record = dataclasses.replace(result, candidate_id=candidate.candidate_id)
                write_public_check(paths, record)
                counts["public_written"] += 1
                counts["infrastructure_errors"] += record.status is CheckStatus.INFRASTRUCTURE_ERROR
            if cacheable(key) and result.status is CheckStatus.OK:
                _write_cache(paths, "public", key, materials[key], result)
        # ---- private grades ---------------------------------------------------------------------------------
        if counts["pool_failures"] and pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        futures = {}
        # Submit item by item: worker processes keep a small per-process oracle cache, and some oracles are large.
        for key, members in sorted(code_groups.items(), key=lambda kv: (kv[1][0][0].item_id, kv[0])):
            cached = _read_cached(paths, "private", key, GradeResult) if cacheable(key) else None
            if cached is not None:
                for candidate, _ in members:
                    write_grade(paths, dataclasses.replace(cached, candidate_id=candidate.candidate_id))
                    counts["grades_cached"] += 1
                continue
            pool = pool or _pool(workers, profile)
            candidate, private = members[0]
            futures[pool.submit(_job_code_grade, dump_record(candidate), dump_record(private), str(paths.run_dir),
                                str(paths.locks_dir), str(paths.local_state_dir))] = ("code", key)
        if text_ok:
            for candidate, private in text_jobs:
                pool = pool or _pool(workers, profile)
                futures[pool.submit(_job_text_grade, dump_record(candidate), dump_record(private),
                                    text_grader)] = ("text", candidate)
        for future in as_completed(futures):
            kind, ref = futures[future]
            try:
                result = load_record(GradeResult, future.result())
            except BrokenProcessPool:
                counts["pool_failures"] += 1
                continue
            members = [c for c, _ in code_groups[ref]] if kind == "code" else [ref]
            for candidate in members:
                record = dataclasses.replace(result, candidate_id=candidate.candidate_id)
                write_grade(paths, record)
                counts["grades_written"] += 1
                counts["evaluator_errors"] += record.status is GradeStatus.EVALUATOR_ERROR
                counts["sandbox_denied"] += "sandbox_denied=" in (record.detail_code or "")
            if kind == "code" and cacheable(ref) and result.status is GradeStatus.GRADED:
                _write_cache(paths, "private", ref, materials[ref], result)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if finished is not None:
        for candidate in candidates:
            if _is_finished(paths, candidate, public_tasks, CHECKER_VERSION, GRADER_VERSION):
                finished.add(candidate.candidate_id)
    return counts


def _is_finished(paths: RunPaths, candidate: Candidate, public_tasks: Mapping[str, PublicTask], checker_version: str,
                 grader_version: str) -> bool:
    task = public_tasks.get(candidate.item_id)
    if task is None:
        return False
    grade = _read_record(paths.grade(candidate.candidate_id), GradeResult)
    if grade is None or grade.status not in TERMINAL_GRADE:
        return False
    if task.output_kind is not OutputKind.PYTHON:
        return True
    check = _read_record(paths.public_check(candidate.candidate_id), PublicCheckResult)
    return (grade.grader_version == grader_version and check is not None and check.status in TERMINAL_CHECK
            and check.checker_version == checker_version)


class _InlineExecutor:
    """``workers=0``: run jobs in this process (debugging and tests); same submit/shutdown surface."""

    def submit(self, fn, *args):
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:            # surfaced by future.result(), as with a process pool
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


def _pool(workers: int, profile: profile_mod.GradingProfile):
    if workers == 0:
        return _InlineExecutor()
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker,
                               initargs=(profile,))


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
    finished: set[str] | None = set() if args.follow else None
    while True:
        try:
            counts = grade_pass(paths, profile, resume=resume, workers=workers, finished=finished)
        except GradeRefused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return EXIT_BLOCKED
        print(json.dumps({"grade_pass": counts, "workers": workers}, sort_keys=True), flush=True)
        if not args.follow or paths.stop_sentinel.exists():
            break
        resume = True
        time.sleep(args.poll_seconds)
    incomplete = counts["blocked_text"] or counts["unknown_item"] or counts["infrastructure_errors"] \
        or counts["evaluator_errors"] or counts["pool_failures"]
    return EXIT_BLOCKED if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(cmd_grade(sys.argv[1:]))
