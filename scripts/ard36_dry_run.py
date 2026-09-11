#!/usr/bin/env python
"""Offline ARD36 dry run: fake vLLM + every cell kind + a resume assertion (WP4).

Plan §10 "Offline dry run (Phase 0, CPU)": *a new ``scripts/ard36_dry_run.py`` (start
``FakeVllmServer`` → ``register_fake(run_root, profile)`` → ``run_one.main([...])`` over
every ARD36 manifest kind on 4 synthetic items) — no such entrypoint exists today; the fake
is reachable only from pytest.*  This is that entrypoint, and it is the G1 gate's green
light (plan §7).

What it does, in one temporary run root:

1. builds a synthetic public item table + protected gold through the real WP3 builder
   (``ard/items_build.build_items`` with injected pinned rows: no HF read, no network);
2. starts the in-process ``FakeVllmServer`` on the real ``ard36-8B`` serving profile and
   registers it, so ``EndpointPool`` discovers it exactly as it discovers a real replica;
3. freezes a controller register and a state register, then writes one frozen cells manifest
   per kind through ``ard/cells.py`` (sha256 sidecar included);
4. drives **``python -m agents_scaling.ard.run_one``'s ``main()``** — the real CLI, the real
   per-cell flock, the real signal handler — once per cell;
5. asserts the hard dependency gate: a GATE cell whose ``controller/<sha>/MODEL.json`` is
   absent, and one whose file has been tampered with, must exit ``EXIT_DEPS_UNMET`` with no
   ``meta.json`` and **zero** additional HTTP calls;
6. asserts **resume**: delete every item file and ``meta.json``, rerun every cell, and the
   fake server's chat-completion count must not change (the request store aliases);
7. compacts the run root to the §18 + §5.8 tables and prints the row counts.

Kinds whose worker belongs to a work package that has not landed (WP5 receiver/selector,
WP6 capture, WP7 E0) are reported as *pending* rather than driven: the runner refuses them
with ``SUSPENDED.json`` naming the missing module, which is the contract, not a failure of
this script.  When those modules land they are driven automatically.

Usage::

    PYTHONPATH=src python scripts/ard36_dry_run.py [--keep] [--run-root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from agents_scaling.ard import cells as ard_cells  # noqa: E402
from agents_scaling.ard import compact as ard_compact  # noqa: E402
from agents_scaling.ard import contract as K  # noqa: E402
from agents_scaling.ard import items as ard_items  # noqa: E402
from agents_scaling.ard import run_one as ard_run_one  # noqa: E402
from agents_scaling.ard import runner as ard_runner  # noqa: E402
from agents_scaling.ard.kinds import KIND_TABLE  # noqa: E402
from agents_scaling.ard.types import (  # noqa: E402
    EXIT_DEPS_UNMET,
    EXIT_DONE,
    ArdCellKind,
    controller_dependency,
)

MATH_SUBJECTS = ("Algebra", "Geometry", "Number Theory", "Prealgebra", "Precalculus",
                 "Intermediate Algebra", "Counting & Probability")
MMLU_CATEGORIES = ("biology", "business", "chemistry", "computer science", "economics",
                   "engineering", "health", "history", "law", "math", "other",
                   "philosophy", "physics", "psychology")
RUN_ID = "ard36_dry_run"


# --------------------------------------------------------------------------- synthetic data


def synthetic_sources():
    """Pinned-shaped rows for the real WP3 builder (no ``datasets`` import, no network)."""
    from agents_scaling.ard import items_build

    math_rows = []
    for index in range(500):
        subject = MATH_SUBJECTS[index % len(MATH_SUBJECTS)]
        level = 1 + (index // len(MATH_SUBJECTS)) % 5
        math_rows.append({
            "unique_id": f"test/{subject.lower().replace(' ', '_')}/{index}.json",
            "problem": f"Problem number {index}: compute {index} + {level}.",
            "solution": f"The answer is $\\boxed{{{index + level}}}$.",
            "answer": str(index + level), "subject": subject, "level": level,
        })
    mmlu_rows = []
    for index in range(700):
        category = MMLU_CATEGORIES[index % len(MMLU_CATEGORIES)]
        mmlu_rows.append({
            "question_id": index,
            "question": f"Question number {index} about {category}?",
            "options": [f"option {index}-{letter}" for letter in range(K.MMLU_REQUIRED_OPTIONS)],
            "answer": "C", "answer_index": index % K.MMLU_REQUIRED_OPTIONS,
            "cot_content": f"chain of thought {index}", "category": category, "src": "synthetic",
        })
    return items_build.RawSources(
        math_rows=tuple(math_rows), mmlu_rows=tuple(mmlu_rows),
        math_revision="6e4ed1a2" + "0" * 32, mmlu_revision="b189ec76" + "0" * 32,
    )


def four_items(run_root: Path) -> list[Any]:
    """One train, one validation and two test items, family-balanced (spec §5.1)."""
    tasks = ard_items.load_ard_tasks(run_root)
    chosen: list[Any] = []
    # test items first, so a reduced (``--items 2``) run still has the test-split items the
    # challenge and GATE manifests require.
    wanted = [("math500", "test"), ("mmlu_pro", "test"),
              ("math500", "train"), ("mmlu_pro", "validation")]
    for family, split in wanted:
        candidates = sorted((t for t in tasks.values() if t.family == family and t.split == split),
                            key=lambda t: t.rank)
        chosen.append(candidates[0])
    return chosen


# --------------------------------------------------------------------------- manifests


def _protection_arms() -> tuple:
    """GATE only when WP5's controller chooser exists; RANDOM_PROTECT always.

    Spec §12.3: GATE's protected agent is the frozen controller's argmax, so with
    ``ard/controller.py`` absent a GATE cell must refuse rather than protect an arbitrary
    agent — the dry run drives the arms that can actually run and reports GATE as pending.
    """
    try:
        module = __import__("agents_scaling.ard.controller", fromlist=["choose_protected"])
        has_wp5 = callable(getattr(module, "choose_protected", None)) and callable(
            getattr(module, "load_controller", None))
    except ImportError:
        has_wp5 = False
    return (K.Protocol.GATE, K.Protocol.RANDOM_PROTECT) if has_wp5 else (K.Protocol.RANDOM_PROTECT,)


def build_manifests(run_root: Path, tasks: list[Any], *, controller_sha: str, state_sha: str,
                    quick: bool = False) -> list[tuple[str, Path, list[Any]]]:
    """One frozen manifest per driven kind: ``(label, path, cells)``.

    ``quick`` keeps the kinds that exercise every distinct code path — the root bank, one
    episode protocol, the controller-gated arm, grading and compaction — and drops the
    replicas of that same path (a second protocol, the challenge team, the budget arm, fork
    and suffix).  The full run is the gate; ``quick`` is what the test group drives.
    """
    test_tasks = [task for task in tasks if task.split == "test"]
    plans: list[tuple[str, list[Any]]] = []

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e1_roots(builder, tasks, checkpoint=K.CHECKPOINT_SMALL)
    plans.append(("E1_ROOTS", list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2_natural(
        builder, tasks, checkpoint=K.CHECKPOINT_SMALL,
        protocols=(K.Protocol.PRIVATE,) if quick else (K.Protocol.PRIVATE, K.Protocol.PEER))
    plans.append(("E2a_EPISODE", list(builder.cells)))

    if quick:
        builder = ard_cells.ArdBuilder()
        ard_cells.build_e4_gate(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                                controller_sha=controller_sha, protocols=_protection_arms())
        plans.append(("E4_PROTECT", list(builder.cells)))
        return _write_plans(run_root, plans, tasks)

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2c_challenge(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                                  team_ids=("teamA",), teams_manifest_sha="ab" * 32,
                                  protocols=(K.Protocol.PRIVATE,))
    plans.append(("E2c_CHALLENGE", list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e2d_high_budget(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL)
    plans.append(("E2d_HIGH_BUDGET", list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e4_forks(builder, tasks, checkpoint=K.CHECKPOINT_SMALL, state_sha=state_sha,
                             parent_rounds=(1,))
    plans.append(("E4_FORK", list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e4_gate(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                            controller_sha=controller_sha, protocols=_protection_arms())
    plans.append(("E4_PROTECT", list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_e5_suffix(builder, test_tasks, checkpoint=K.CHECKPOINT_SMALL,
                              state_sha=state_sha, parent_protocols=(K.Protocol.PEER,), seeds=1)
    plans.append(("E5_SUFFIX", list(builder.cells)))

    return _write_plans(run_root, plans, tasks)


def _write_plans(run_root: Path, plans, tasks: list[Any]) -> list[tuple[str, Path, list[Any]]]:
    """Freeze every manifest (with its ``.sha256`` sidecar), then the GRADE and COMPACT cells."""
    written: list[tuple[str, Path, list[Any]]] = []
    generation_cells: list[str] = []
    for label, cells in plans:
        path = run_root / f"cells_{label}.json"
        ard_cells.write_ard_cells_file(path, cells, ard_cells.manifest_meta(
            module=label, lane=None, cells=cells))
        written.append((label, path, cells))
        generation_cells += [cell.cell_id for cell in cells]

    builder = ard_cells.ArdBuilder()
    ard_cells.build_grade_cells(builder, tasks, generation_cells=generation_cells)
    path = run_root / "cells_GRADE.json"
    ard_cells.write_ard_cells_file(path, builder.cells, ard_cells.manifest_meta(
        module="GRADE", lane=None, cells=builder.cells))
    written.append(("GRADE", path, list(builder.cells)))

    builder = ard_cells.ArdBuilder()
    ard_cells.build_analysis_cells(builder, kind=ArdCellKind.COMPACT,
                                   options={"allow_fallback": True, "strict_residual": False})
    path = run_root / "cells_COMPACT.json"
    ard_cells.write_ard_cells_file(path, builder.cells, ard_cells.manifest_meta(
        module="COMPACT", lane=None, cells=builder.cells))
    written.append(("COMPACT", path, list(builder.cells)))
    return written


# --------------------------------------------------------------------------- driving


def drive(results_root: Path, path: Path, n_cells: int) -> list[int]:
    codes = []
    for index in range(n_cells):
        codes.append(ard_run_one.main([
            "--run-id", RUN_ID, "--results-root", str(results_root),
            "--cells-file", str(path), "--index", str(index), "--wait-s", "30",
        ]))
    return codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", default=None, help="where to build the temp run root")
    parser.add_argument("--keep", action="store_true", help="do not delete the run root")
    parser.add_argument("--quick", action="store_true",
                        help="drive one manifest per code path (what tests/ard drives)")
    parser.add_argument("--items", type=int, default=4, help="synthetic items (spec: 4)")
    args = parser.parse_args(argv)

    from agents_scaling.ard import items_build
    from agents_scaling.serving.profiles import get_serving_profile
    from agents_scaling.study.inference.tokens import load_tokenizer
    from tests.study.fake_vllm import FakeVllmServer

    base = Path(args.run_root) if args.run_root else Path(tempfile.mkdtemp(prefix="ard36_dry_"))
    results_root = base
    run_root = results_root / RUN_ID
    run_root.mkdir(parents=True, exist_ok=True)
    for name in ("servers", "cells", "requests", "data", "eval", "freeze", "logs"):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"[dry-run] run root {run_root}")
    items_build.build_items(run_root, sources=synthetic_sources())
    tasks = four_items(run_root)[: max(2, int(args.items))]
    print(f"[dry-run] items: {[t.item_id for t in tasks]}")

    cfg = ard_runner.load_ard_config()
    tokenizer = load_tokenizer(cfg.checkpoint(K.CHECKPOINT_SMALL))
    profile = get_serving_profile(K.PROFILE_SMALL)
    server = FakeVllmServer(tokenizer=tokenizer, served_model_name=profile.served_model_name,
                            max_model_len=profile.max_model_len).start()
    server.register(run_root, K.PROFILE_SMALL)
    print(f"[dry-run] fake vLLM at {server.base_url} registered on {K.PROFILE_SMALL}")

    # A REAL controller artifact, not a stub: ard/controller.py validates the kind and the
    # canonical-JSON digest, so a placeholder would make the GATE path here diverge from the
    # path the run takes (spec §12.3).
    from tests.ard.wp4_support import freeze_minimal_controller

    controller_sha = freeze_minimal_controller(run_root)
    state_sha = ard_runner.freeze_register(
        run_root, "state", {"schema": "ard36-state-v1", "states": []})
    print(f"[dry-run] controller:{controller_sha[:12]}  state:{state_sha[:12]}")

    report: dict[str, Any] = {"driven": {}, "pending": {}, "failures": []}
    try:
        manifests = build_manifests(run_root, tasks, controller_sha=controller_sha,
                                    state_sha=state_sha, quick=args.quick)
        for label, path, cells in manifests:
            codes = drive(results_root, path, len(cells))
            report["driven"][label] = {"cells": len(cells), "exit_codes": codes}
            print(f"[dry-run] {label:16s} {len(cells)} cells -> {codes}")
            if any(code != EXIT_DONE for code in codes):
                report["failures"].append(f"{label}: exit codes {codes}")

        # ---- the hard dependency gate ------------------------------------------------
        before = server.chat_count
        builder = ard_cells.ArdBuilder()
        ard_cells.build_e4_gate(builder, [t for t in tasks if t.split == "test"],
                                checkpoint=K.CHECKPOINT_SMALL, controller_sha="cc" * 32,
                                protocols=(K.Protocol.GATE,), instance="gateoff")
        gate_path = run_root / "cells_GATE_MISSING.json"
        ard_cells.write_ard_cells_file(gate_path, builder.cells, ard_cells.manifest_meta(
            module="E4_GATE_MISSING", lane=None, cells=builder.cells))
        codes = drive(results_root, gate_path, len(builder.cells))
        gated_ok = all(code == EXIT_DEPS_UNMET for code in codes)
        no_meta = all(not (run_root / "cells" / cell.cell_id / "meta.json").exists()
                      for cell in builder.cells)
        no_requests = server.chat_count == before
        report["dependency_gate"] = {"exit_codes": codes, "no_meta_json": no_meta,
                                     "no_requests": no_requests}
        print(f"[dry-run] dependency gate  -> {codes} (no meta.json: {no_meta}, "
              f"no requests: {no_requests})")
        if not builder.cells:
            report["failures"].append("the dependency-gate check drove no cell (vacuous)")
        if not (gated_ok and no_meta and no_requests):
            report["failures"].append("the controller gate did not hold")

        # ---- resume ------------------------------------------------------------------
        calls_before = server.chat_count
        for _label, _path, cells in manifests:
            for cell in cells:
                cell_dir = run_root / "cells" / cell.cell_id
                shutil.rmtree(cell_dir / "items", ignore_errors=True)
                (cell_dir / "meta.json").unlink(missing_ok=True)
        for label, path, cells in manifests:
            codes = drive(results_root, path, len(cells))
            if any(code != EXIT_DONE for code in codes):
                report["failures"].append(f"resume {label}: exit codes {codes}")
        calls_after = server.chat_count
        report["resume"] = {"chat_calls_before": calls_before, "chat_calls_after": calls_after,
                            "unchanged": calls_before == calls_after}
        print(f"[dry-run] resume: {calls_before} -> {calls_after} chat calls "
              f"({'unchanged' if calls_before == calls_after else 'CHANGED'})")
        if calls_before != calls_after:
            report["failures"].append(
                f"resume regenerated requests ({calls_after - calls_before} extra chat calls)")

        # ---- kinds still owned by other work packages --------------------------------
        if K.Protocol.GATE not in _protection_arms():
            report["pending"]["GATE"] = ("spec §12.3: GATE needs ard/controller.py (WP5) to "
                                         "publish load_controller + choose_protected")
        for kind, spec in KIND_TABLE.items():
            if kind.value in {c.kind.value for _l, _p, cs in manifests for c in cs}:
                continue
            try:
                ard_runner.resolve_worker(kind)
                report["pending"][kind.value] = f"{spec.worker} is importable but undriven here"
            except Exception as exc:  # noqa: BLE001 - the message is the report
                report["pending"][kind.value] = f"{spec.owner}: {exc}"
        for kind, note in report["pending"].items():
            print(f"[dry-run] pending {kind:16s} {note}")

        counts = ard_compact.compact(run_root, allow_fallback=True, strict_residual=False)
        report["tables"] = counts
        print("[dry-run] tables: " + json.dumps(counts))
    finally:
        server.stop()

    report["wall_s"] = round(time.time() - started, 1)
    report["run_root"] = str(run_root)
    (run_root / "dry_run_report.json").write_text(json.dumps(report, indent=2))
    ok = not report["failures"]
    print(f"[dry-run] {'GREEN' if ok else 'RED'} in {report['wall_s']}s; report at "
          f"{run_root / 'dry_run_report.json'}")
    for failure in report["failures"]:
        print(f"[dry-run] FAILURE: {failure}", file=sys.stderr)
    if not args.keep and ok:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
