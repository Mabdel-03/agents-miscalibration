"""MBPP+ reference certification at the frozen grading limits, and EvalPlus comparison parity (agent 05).

CPU-heavy: run as a Slurm batch job, never on the shared interactive node.

    python -m agent_allocation.grading.code.certify references --run-dir DIR --out REPORT.json [--workers N]
    python -m agent_allocation.grading.code.certify parity --out REPORT.json
    python -m agent_allocation.grading.code.certify probe --run-dir DIR --out REPORT.json [--items Mbpp/255]
        [--memory-mib 512,1024]

``references``: installs the MBPP+ private records into ``DIR`` (``load_inventory``), then for every task runs its
canonical reference (``prompt + canonical_solution``) through the production public checker (its own prompt
assert) and the production private grader (``grade_code``), both at the config limits. A reference that is not
``GRADED``/correct is re-run through the scorer at the config limits and at the previously certified 4 GiB profile
so the report can tell a limit conflict from any other cause. Nothing is changed silently: the report lists every
failure; the lead decides amendments, and prompt-assert failures become documented pre-generation exclusions.

``parity``: the 20 artificial fixtures through official upstream ``untrusted_check`` and through the vendored
wrapper at the config limits.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from agent_allocation._vendor.learnagents_io import write_json
from agent_allocation.contracts import (Candidate, CheckStatus, GradeStatus, OutputKind, ParseStatus, PrivateEvalSpec,
                                        PublicTask, dump_record, load_record)
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.paths import RunPaths

CERTIFIED_MEMORY_BYTES = 4 * 1024 ** 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init(profile: profile_mod.GradingProfile) -> None:
    profile_mod.configure(profile)


def _diagnose(program: str, record: dict[str, Any], profile: profile_mod.GradingProfile) -> dict[str, Any]:
    from agent_allocation.sandbox.evalplus_scorer import EvaluatorOutage, grade_evalplus
    started = time.monotonic()
    try:
        grade = grade_evalplus(program, record["restricted"], record["oracle"], python_prefix=profile.python_prefix,
                               memory_bytes=profile.memory_bytes, output_bytes=profile.stdout_bytes,
                               stack_bytes=profile.stack_bytes, suite_wall_seconds=profile.suite_wall_seconds,
                               row=record["row"], scratch_bytes=profile.scratch_bytes, file_bytes=profile.file_bytes)
    except EvaluatorOutage as exc:
        return {"outage": f"{exc.kind}/{exc.phase}/{exc.returncode}: {str(exc)[:300]}",
                "wall_seconds": round(time.monotonic() - started, 3)}
    return {"success": grade.success, "base": grade.details["base"], "plus": grade.details["plus"],
            "candidate_crash": grade.details["candidate_crash"],
            "cpu_seconds": round(grade.details["_resource_usage"]["sandbox_cpu_seconds"], 3),
            "wall_seconds": round(time.monotonic() - started, 3)}


def certify_one(public_data: dict, private_data: dict, run_dir: str, locks_dir: str, local: str) -> dict[str, Any]:
    from agent_allocation.grading.code.private import grade_code, load_private_record
    from agent_allocation.grading.code.public import run_public_checks

    task = load_record(PublicTask, public_data)
    private = load_record(PrivateEvalSpec, private_data)
    paths = RunPaths(Path(run_dir), Path(locks_dir), Path(local))
    profile = profile_mod.active_profile()
    row: dict[str, Any] = {"item_id": task.item_id}
    try:
        record = load_private_record(private, paths)
        native = record["row"]
        row["task_id"] = native["task_id"]
        program = native["prompt"] + native["canonical_solution"]
        node = hashlib.sha256(("reference|" + task.item_id).encode()).hexdigest()
        candidate = Candidate(candidate_id=node, node_id=node, item_id=task.item_id, visible_content=program,
                              final_answer=program, rationale=None, parse_status=ParseStatus.VALID,
                              finish_reason="stop", output_kind=OutputKind.PYTHON)
        started = time.monotonic()
        check = run_public_checks(candidate, task)
        row["public"] = {"status": check.status.value, "outcomes": list(check.case_outcomes),
                         "cpu_seconds": round(check.cpu_seconds, 3), "wall_seconds": round(time.monotonic() - started, 3)}
        started = time.monotonic()
        grade = grade_code(candidate, private, paths)
        row["private"] = {"status": grade.status.value, "semantic_correct": grade.semantic_correct,
                          "detail_code": grade.detail_code, "cpu_seconds": grade.cpu_seconds,
                          "wall_seconds": round(time.monotonic() - started, 3)}
        row["public_pass"] = check.status is CheckStatus.OK and all(o == "pass" for o in check.case_outcomes)
        row["private_pass"] = grade.status is GradeStatus.GRADED and grade.semantic_correct is True
        if not row["private_pass"]:
            row["diagnostic_config_limits"] = _diagnose(program, record, profile)
            row["diagnostic_certified_4gib"] = _diagnose(program, record, dataclasses.replace(
                profile, memory_bytes=CERTIFIED_MEMORY_BYTES, stack_bytes=CERTIFIED_MEMORY_BYTES))
        if not row["public_pass"]:
            relaxed = dataclasses.replace(profile, memory_bytes=CERTIFIED_MEMORY_BYTES,
                                          stack_bytes=CERTIFIED_MEMORY_BYTES)
            profile_mod.configure(relaxed)
            try:
                again = run_public_checks(candidate, task)
            finally:
                profile_mod.configure(profile)
            row["public_diagnostic_certified_4gib"] = {"status": again.status.value,
                                                       "outcomes": list(again.case_outcomes)}
    except Exception as exc:                                   # reported, never swallowed into a pass
        row["driver_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["traceback"] = traceback.format_exc(limit=5)[-2000:]
        row.setdefault("public_pass", False)
        row.setdefault("private_pass", False)
    return row


def run_references(args: argparse.Namespace) -> int:
    from agent_allocation.benchmarks.mbpp_plus import load_inventory

    cfg = json.loads(Path(args.config).read_text())
    profile = profile_mod.profile_from_config(cfg)
    profile_mod.configure(profile)
    paths = RunPaths.for_run(args.run_dir, local_state_dir=Path(args.run_dir) / "local_state")
    items = load_inventory(cfg, paths)
    affinity = len(os.sched_getaffinity(0))
    workers = args.workers or min(profile.workers_max, max(1, affinity - 2))
    header = {"started_at": _now(), "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "git_commit": os.environ.get("WP05_GIT_COMMIT"), "affinity_cpus": affinity, "workers": workers,
              "profile": dataclasses.asdict(profile), "environment_digest": profile_mod.environment_digest(profile),
              "config": str(args.config), "items": len(items), "eligible": sum(i.eligible for i in items)}
    out = Path(args.out)
    rows: list[dict[str, Any]] = []

    def report(status: str) -> dict[str, Any]:
        failures = [r for r in rows if not (r.get("public_pass") and r.get("private_pass"))]
        return {**header, "status": status, "completed": len(rows),
                "public_pass": sum(bool(r.get("public_pass")) for r in rows),
                "private_pass": sum(bool(r.get("private_pass")) for r in rows),
                "failures": sorted(failures, key=lambda r: r["item_id"]),
                "max_private_wall_seconds": max((r.get("private", {}).get("wall_seconds", 0) for r in rows), default=0),
                "results": sorted(rows, key=lambda r: r["item_id"])}

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(profile,)) as pool:
        futures = [pool.submit(certify_one, dump_record(item.public), dump_record(item.private), str(paths.run_dir),
                               str(paths.locks_dir), str(paths.local_state_dir)) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                write_json(out, report("running"))
                print(json.dumps({"completed": index, "total": len(futures),
                                  "public_pass": sum(bool(r.get("public_pass")) for r in rows),
                                  "private_pass": sum(bool(r.get("private_pass")) for r in rows)}), flush=True)
    final = report("passed" if all(r.get("public_pass") and r.get("private_pass") for r in rows)
                   else "failures_reported")
    final["finished_at"] = _now()
    write_json(out, final)
    print(json.dumps({k: final[k] for k in ("status", "completed", "public_pass", "private_pass")}), flush=True)
    for failure in final["failures"]:
        print(json.dumps({k: failure.get(k) for k in ("item_id", "public", "private", "diagnostic_config_limits",
                                                       "diagnostic_certified_4gib", "driver_error")}), flush=True)
    return 0


def run_parity(args: argparse.Namespace) -> int:
    from agent_allocation.sandbox.evalplus_validation import validate

    cfg = json.loads(Path(args.config).read_text())
    profile = profile_mod.profile_from_config(cfg)
    report = validate(python_prefix=profile.python_prefix, memory_bytes=profile.memory_bytes,
                      output_bytes=profile.stdout_bytes, stack_bytes=profile.stack_bytes,
                      suite_wall_seconds=profile.suite_wall_seconds)
    report.update({"generated_at": _now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "git_commit": os.environ.get("WP05_GIT_COMMIT"),
                   "environment_digest": profile_mod.environment_digest(profile)})
    write_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("status", "count")}), flush=True)
    for row in report["results"]:
        print(json.dumps(row), flush=True)
    return 0 if report["status"] == "passed" else 1


def probe_one(public_data: dict, private_data: dict, run_dir: str, locks_dir: str, local: str,
              memories_mib: list[int]) -> dict[str, Any]:
    """Reference through the scorer at a ladder of memory limits (other limits from the config profile)."""
    from agent_allocation.grading.code.private import load_private_record

    private = load_record(PrivateEvalSpec, private_data)
    paths = RunPaths(Path(run_dir), Path(locks_dir), Path(local))
    profile = profile_mod.active_profile()
    record = load_private_record(private, paths)
    program = record["row"]["prompt"] + record["row"]["canonical_solution"]
    out: dict[str, Any] = {"item_id": private.item_id, "task_id": record["row"]["task_id"], "by_memory_mib": {}}
    for mib in memories_mib:
        limits = dataclasses.replace(profile, memory_bytes=mib * 1024 ** 2, stack_bytes=mib * 1024 ** 2)
        out["by_memory_mib"][str(mib)] = _diagnose(program, record, limits)
    return out


def run_probe(args: argparse.Namespace) -> int:
    from agent_allocation.benchmarks.mbpp_plus import load_inventory

    cfg = json.loads(Path(args.config).read_text())
    profile = profile_mod.profile_from_config(cfg)
    profile_mod.configure(profile)
    paths = RunPaths.for_run(args.run_dir, local_state_dir=Path(args.run_dir) / "local_state")
    items = load_inventory(cfg, paths)
    wanted = set(args.items.split(",")) if args.items else None
    chosen = [i for i in items if wanted is None or i.public.item_id.split("/", 1)[1].replace("_", "/") in wanted]
    ladder = [int(x) for x in args.memory_mib.split(",")]
    affinity = len(os.sched_getaffinity(0))
    workers = args.workers or min(profile.workers_max, max(1, affinity - 2))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(profile,)) as pool:
        futures = [pool.submit(probe_one, dump_record(i.public), dump_record(i.private), str(paths.run_dir),
                               str(paths.locks_dir), str(paths.local_state_dir), ladder) for i in chosen]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: r["item_id"])
    headroom = []
    for row in rows:
        for mib, diag in row["by_memory_mib"].items():
            for split in ("base", "plus"):
                batch = diag.get(split) or {}
                if batch.get("batch_ceiling_seconds"):
                    headroom.append({"item_id": row["item_id"], "memory_mib": int(mib), "split": split,
                                     "wall_seconds": batch["wall_seconds"],
                                     "batch_ceiling_seconds": batch["batch_ceiling_seconds"],
                                     "fraction": round(batch["wall_seconds"] / batch["batch_ceiling_seconds"], 4)})
    headroom.sort(key=lambda h: -h["fraction"])
    report = {"generated_at": _now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "git_commit": os.environ.get("WP05_GIT_COMMIT"), "workers": workers, "affinity_cpus": affinity,
              "memory_ladder_mib": ladder, "profile": dataclasses.asdict(profile), "items": len(rows),
              "failures": [{"item_id": r["item_id"], "memory_mib": int(m), **d} for r in rows
                           for m, d in r["by_memory_mib"].items() if not d.get("success")],
              "tightest_batches": headroom[:25], "results": rows}
    write_json(Path(args.out), report)
    print(json.dumps({"items": len(rows), "failures": len(report["failures"])}), flush=True)
    for failure in report["failures"]:
        print(json.dumps(failure), flush=True)
    for row in headroom[:10]:
        print(json.dumps(row), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    refs = sub.add_parser("references")
    refs.add_argument("--run-dir", required=True)
    refs.add_argument("--out", required=True)
    refs.add_argument("--workers", type=int, default=None)
    refs.add_argument("--config", default=str(profile_mod.DEFAULT_CONFIG))
    parity = sub.add_parser("parity")
    parity.add_argument("--out", required=True)
    parity.add_argument("--config", default=str(profile_mod.DEFAULT_CONFIG))
    probe = sub.add_parser("probe")
    probe.add_argument("--run-dir", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--items", default="", help="comma-separated task ids (Mbpp/255); empty = all")
    probe.add_argument("--memory-mib", default="512")
    probe.add_argument("--workers", type=int, default=None)
    probe.add_argument("--config", default=str(profile_mod.DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    return {"references": run_references, "parity": run_parity, "probe": run_probe}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
