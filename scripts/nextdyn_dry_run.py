#!/usr/bin/env python
"""Offline nextdyn dry run: every cell kind on 4 synthetic items, then the whole close-out.

Rule R3 (plan §1.1): *"Every producer has a consumer test in the same change:
``test_pipeline_dry_run.py`` drives every cell kind on 4 synthetic items through the fake
vLLM, materializes, asserts keys, runs ``analyze`` in test mode.  A WP is not done until its
kind runs there."*  This script **is** that pipeline as a CLI, and
``tests/nextdyn/test_pipeline_dry_run.py`` drives it through the same functions, so the gate
and the test can never drift apart.

In one temporary run root, with no GPU, no network and no Slurm:

1. build a synthetic public item store + protected gold through the real WP3 builder
   (``items_build.build_items`` with injected drafts: no HF read, no download);
2. start the in-process ``FakeVllmServer`` on the real ``nd1-8B`` / ``nd1-32B`` profiles and
   register it in **each pool's sub-run-root**, so ``NextdynEndpointPool`` discovers it
   exactly as it discovers a real replica (plan §3.2);
3. build the DAG with the real planner and freeze every block manifest;
4. drive ``python -m agents_scaling.nextdyn.run_one``'s ``main()`` -- the real CLI, the real
   per-cell flock, the real dependency gate -- once per cell;
5. run a planner pass (seal the pools and the capture panel, emit what they unblock) and
   drive the audit cells;
6. assert the hard dependency gate: a cell whose bundle is absent exits ``EXIT_DEPS_UNMET``
   with **no** ``meta.json`` and **zero** additional HTTP calls;
7. assert resume: delete every item file and ``meta.json``, re-drive, and the fake server's
   call count must not change (the request store aliases every request) **and the per-request
   archive must not shrink** -- a resumed cell rewrites its item file, and that file is the only
   home its ``requests`` rows have (plan §3.7);
8. assert the freeze gate: unfrozen and permitted for this test-mode run, refused for a
   confirmation one (the write-once ``FROZEN.yaml`` itself needs the locks and budgets a human
   authorizes at S3 and is ``tests/nextdyn/test_freeze.py``'s subject);
9. write the ``plan`` table from the close-out census (``cell_status(final=True)``), so
   ``release-check``'s "every planned cell is terminal" gate has an end-to-end producer;
10. ``materialize`` -> ``tables.check_keys(strict)`` -> ``tables.assert_referential`` ->
    ``analyze`` in the test profile -> a claim ledger.

A ``DRIVEN_BLOCKS`` entry that emits no manifest, or a manifest with no cell in it, is a
FAILURE: a block that silently builds nothing takes its kind out of R3's coverage.

A kind whose worker needs something this environment does not have (the capture engine needs
torch and the pinned tokenizer) is reported as *pending* with the reason, never silently
skipped.

Usage::

    PYTHONPATH=src python scripts/nextdyn_dry_run.py [--keep] [--run-root DIR] [--quick]
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
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("HF_HOME", "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from agents_scaling.nextdyn import cells as CE  # noqa: E402
from agents_scaling.nextdyn import config as C  # noqa: E402
from agents_scaling.nextdyn import deps, materialize as MAT, plan as PLAN  # noqa: E402
from agents_scaling.nextdyn import run_one, tables  # noqa: E402
from agents_scaling.nextdyn.ids import CellKind  # noqa: E402
from agents_scaling.nextdyn.kinds import Pool  # noqa: E402
from agents_scaling.nextdyn.runner import EXIT_DEPS_UNMET  # noqa: E402
from agents_scaling.study.types import EXIT_DONE  # noqa: E402

RUN_ID = "nextdyn_dry_run"
#: The blocks the dry run dispatches.  E2A needs a sealed packets register (a writer-side pass
#: over graded donors) and E3a/E3b/E3c need the directed panels; both are exercised by their
#: own test groups, and every *kind* they use (RECEIVER, EPISODE) runs here.
DRIVEN_BLOCKS = (CE.Block.BANKS, CE.Block.NATURAL_ROOTS, CE.Block.GRADE, CE.Block.CONSTRUCT,
                 CE.Block.E1_NATURAL, CE.Block.E1_CONSTRUCTED, CE.Block.E2B, CE.Block.E0)
#: Driven last, after the planner's cells: the MATERIALIZE cell depends on every other one.
CLOSE_OUT_BLOCK = CE.Block.MATERIALIZE
#: How many cells of one block the dry run drives (a block is a *kind* check, not a load test).
BLOCK_LIMITS: dict[str, int] = {CE.Block.E2B: 4}
#: Items per family the *store* is built with.  The dry run dispatches four of them, but the
#: real inventory refuses a sub-panel that is quietly smaller than declared (spec 14.3), and
#: the declared panels (natural 200, constructed 300, e3a 150, ...) need a frame this size.
PER_FAMILY = 600
#: How many of those items the dry run actually dispatches (R3: four synthetic items).
DISPATCHED_PER_FAMILY = 4
#: The answer space the fake model draws from: two classes, so a bank holds both a correct and
#: a wrong class (the dry run's gold is the first of each pair).
EXACT_ANSWERS = ("1", "2")
MC_ANSWERS = ("A", "B")
#: The synthetic multiple-choice question stem (see :func:`dry_run_responder`).
MC_STEM = "Which option is"
#: The cohort(s) the dry run plans (see :func:`reduced_config` for why it is one).
DRY_RUN_COHORTS = ("MMLU_PRO",)
SEED = bytes.fromhex("11" * 32)
SALT = bytes.fromhex("22" * 32)


# --------------------------------------------------------------------------- synthetic items


def synthetic_drafts(cfg: Any, *, per_family: int = PER_FAMILY) -> dict[str, list[Any]]:
    """Two items per family for the real WP3 builder (no ``datasets`` import, no network)."""
    from agents_scaling.nextdyn import datasets as D
    from agents_scaling.nextdyn import families as F
    from agents_scaling.nextdyn import items_build as IB

    def draft(row: Any) -> Any:
        spec = F.family(row.family)
        item_id = IB.source_item_id(cfg.study.id, row.family, spec.loader.revision, row.original_id)
        return IB.ItemDraft(source_item_id=item_id, family=row.family, row=row,
                            variants=IB.build_variants(item_id, row, spec, SEED))

    # The gold lives in the two-answer space the dry-run responder draws from, so the donor
    # bank splits into a correct class and a wrong one and the constructed teams are eligible
    # (with one answer everywhere there is no wrong class and CONSTRUCT correctly builds
    # nothing, which would leave the constructed episodes untested).
    math = [D.SourceRow(family=F.MATH500, original_id=f"test/algebra/{i}.json",
                        question=f"Problem {i}: compute {i} + 2.", options=(),
                        gold={"answer": EXACT_ANSWERS[0]}, subject="Algebra", level="1")
            for i in range(per_family)]
    mc = [D.SourceRow(family=F.MMLU_PRO, original_id=f"mmlu_pro-{i}",
                      question=f"{MC_STEM} the {i}th answer?",
                      options=tuple(f"answer {i}.{k}" for k in range(F.MMLU_REQUIRED_OPTIONS)),
                      gold={"canonical_index": 0, "source_answer": "x"},
                      subject="engineering", level=None)
          for i in range(per_family)]
    panels = {F.MMLU_PRO: [draft(row) for row in mc]}
    if F.MATH500 in {F.TASK_FAMILY.get(name, name) and name for name in DRY_RUN_COHORTS} or \
            "MATH" in DRY_RUN_COHORTS:
        panels[F.MATH500] = [draft(row) for row in math]
    return panels


def build_items(run_root: Path, cfg: Any, *, per_family: int = PER_FAMILY) -> dict[str, list[Any]]:
    """The real public store: items, variants, splits, texts and protected gold."""
    from agents_scaling.nextdyn import inventory as INV
    from agents_scaling.nextdyn import items_build as IB

    panels = synthetic_drafts(cfg, per_family=per_family)
    original = IB.build_drafts
    IB.build_drafts = lambda cfg_, home: panels          # noqa: SLF001 - the documented seam
    try:
        IB.build_items(run_root, cfg, split_salt=SALT, study_seed=SEED, hf_home=run_root,
                       registry=INV.LegacyRegistry.empty()
                       if hasattr(INV, "LegacyRegistry") else None)
    finally:
        IB.build_drafts = original
    return panels


def reduced_config(cfg: Any) -> Any:
    """The committed study at dry-run size: one model, one budget, one E0 shard, one cohort.

    Every *shape* stays the study's (five agents, the full E2b arm grid, the real protocols);
    only the multiplicities a dry run does not test are cut, so the manifests this builds are
    the manifests the run builds, with fewer cells.

    The cohort is narrowed to ``MMLU_PRO`` for a reason worth stating: ``analysis.yaml``'s task
    weights are keyed by **cohort** (``MATH``) while the published tables carry the **panel**
    an item came from (``MATH500`` / ``MATH_EXPANDED``, ``families.TASK_FAMILY``), so a MATH
    item reaches ``stats.wild_cluster_test`` under a name the weight table does not hold.  For
    ``MMLU_PRO`` the two names coincide, so the dry run exercises the whole pipeline while that
    mismatch is reported rather than papered over (it is an open issue for the analysis side).
    """
    import dataclasses

    e2b = dataclasses.replace(cfg.experiments.E2b, models=(cfg.experiments.E2b.models[0],),
                              replicates=1,
                              # one problem per declared depth: the grid is what is being
                              # exercised, not the panel size
                              source_problems=len(cfg.experiments.E2b.chain_depths))
    e2a = dataclasses.replace(cfg.experiments.E2a, models=(cfg.experiments.E2a.models[0],),
                              replicates=1)
    e0 = dataclasses.replace(cfg.experiments.E0, bcb_array_cells=1)
    experiments = dataclasses.replace(cfg.experiments, E2b=e2b, E2a=e2a, E0=e0)
    budgets = dataclasses.replace(cfg.budgets,
                                  receiver_thinking_tokens=(cfg.budgets.receiver_thinking_tokens[0],))
    cohorts = dataclasses.replace(cfg.cohorts, families=DRY_RUN_COHORTS)
    return dataclasses.replace(cfg, experiments=experiments, budgets=budgets, cohorts=cohorts)


def dry_run_responder():
    """A fake model whose answer depends on the request, so the banks hold two classes.

    ``tests/nextdyn/conftest.nd_json_responder`` always answers ``4``: one canonical class, so
    every donor agrees, nothing is paired-eligible and the constructed half of the pipeline is
    never exercised.  This one hashes the request (prompt **and** seed) into a two-element
    answer space, and picks the space from the *answer instruction* the root template rendered
    -- a letter for a multiple-choice family, a number otherwise -- because that instruction is
    the only family signal a model sees (spec 6.2).
    """
    import hashlib

    from agents_scaling.nextdyn import families as F
    from tests.nextdyn.conftest import NDJSON_OPEN, NDJSON_TAIL

    def _respond(payload: dict) -> str:
        text = json.dumps(payload.get("messages", payload.get("prompt_token_ids", "")),
                          sort_keys=True, default=str)
        # The *revision* template carries no answer instruction (spec 6.3: only the root
        # prompt states the format), so the family signal has to be the question itself --
        # which is why the synthetic multiple-choice stem is a fixed phrase.
        space = MC_ANSWERS if (F.INSTRUCTION_MC[:40] in text or MC_STEM in text) else EXACT_ANSWERS
        digest = hashlib.sha256((text + str(payload.get("seed", ""))).encode()).digest()
        answer = space[digest[0] % len(space)]
        tail = NDJSON_TAIL.replace('"4"', f'"{answer}"')
        if "prompt_token_ids" in payload:
            return tail.lstrip()
        return NDJSON_OPEN + tail

    return _respond


def dispatched_panels(inventory: PLAN.Inventory) -> PLAN.Inventory:
    """Make every sub-panel the items this dry run dispatches.

    The real sub-panels are hash-selected over a 600-item frame (plan §2.2), so on four
    dispatched items they name items with no bank draws at all -- and E2a's packet pair, which
    is frozen from *graded donors*, would have nothing to choose from, while the capture panel
    would name items no episode ran.  Substituting the dispatched set is what lets those blocks
    be exercised offline: it changes which items are in a panel, never how a panel is built or
    how its register is sealed.
    """
    import dataclasses

    panels = {name: dict(inventory.items) for name in inventory.panels}
    return dataclasses.replace(inventory, panels=panels)


# --------------------------------------------------------------------------- the fake fleet


def start_fake_fleet(run_root: Path, cfg: Any) -> tuple[Any, Any]:
    """One in-process fake vLLM **per serving profile**, registered in every pool (plan §3.2).

    One server per profile, not one for both: a vLLM serves exactly one ``served_model_name``
    and answers 404 for any other, so a single fake would make every 8B request fail with a
    model-not-found that looks like an infrastructure failure rather than a fixture bug.
    """
    from agents_scaling.serving.profiles import get_serving_profile
    from tests.study.fake_vllm import FakeTokenizer, FakeVllmServer

    tokenizer = FakeTokenizer()
    profiles = sorted({model.profile for model in cfg.models.values() if model.enabled})
    servers = []
    for profile in profiles:
        spec = get_serving_profile(profile)
        server = FakeVllmServer(tokenizer=tokenizer, served_model_name=spec.served_model_name,
                                max_model_len=spec.max_model_len, thinking_words=4,
                                responder=dry_run_responder()).start()
        for pool in Pool:
            root = run_root / "pools" / pool.value
            root.mkdir(parents=True, exist_ok=True)
            server.register(root, profile)
        servers.append(server)
    return Fleet(servers), tokenizer


class Fleet:
    """Every fake replica as one object: ``chat_count`` is the fleet's, ``stop`` stops all."""

    def __init__(self, servers: Sequence[Any]) -> None:
        self.servers = list(servers)

    @property
    def chat_count(self) -> int:
        return sum(int(server.chat_count) for server in self.servers)

    def stop(self) -> None:
        for server in self.servers:
            server.stop()


# --------------------------------------------------------------------------- driving


def drive_cell(run_root: Path, manifest: Path, index: int, *, config: Path) -> int:
    return run_one.main(["--run-root", str(run_root), "--config", str(config),
                         "--manifest", str(manifest), "--index", str(index), "--wait-s", "30"])


def unexpected_exits(run_root: Path, manifest: Path, codes: Sequence[int]) -> list[str]:
    """Which non-zero exits are real failures.

    ``EXIT_DEPS_UNMET`` on a bucket cell whose bundle was never written is not one: the frozen
    scan reached a bucket that turned out not to be paired-eligible, so the cell terminates as
    ``skipped_quota_unfilled`` (spec 5.2) and the run is complete without it.  Any other
    non-zero exit -- and an unmet dependency on a cell that *could* have run -- is reported.
    """
    cells = CE.load_manifest(manifest)
    out: list[str] = []
    for index, code in enumerate(codes):
        if code == EXIT_DONE:
            continue
        cell = cells[index]
        status = PLAN.cell_status(run_root, cell, final=True)
        if code == EXIT_DEPS_UNMET and status == PLAN.STATUS_SKIPPED:
            continue
        out.append(f"{cell.cell_id}: exit {code} ({status})")
    return out


def drive_manifest(run_root: Path, manifest: Path, *, config: Path, limit: int | None = None
                   ) -> list[int]:
    cells = CE.load_manifest(manifest)
    n = len(cells) if limit is None else min(limit, len(cells))
    return [drive_cell(run_root, manifest, index, config=config) for index in range(n)]


def inject_tokenizer(tokenizer: Any) -> None:
    """Every cell's tokenizer is the fake server's (decode is instance-local)."""
    from agents_scaling.nextdyn import runner as RUN

    RUN.NextdynCellRunner._tokenizer_factory = lambda self: (lambda name: tokenizer)  # noqa: SLF001


# --------------------------------------------------------------------------- the pipeline


def run(base: Path, *, quick: bool = False, dispatched: int = DISPATCHED_PER_FAMILY,
        frame: int = PER_FAMILY) -> dict[str, Any]:
    """The whole offline pipeline; returns the report ``main`` prints and writes."""
    cfg = reduced_config(C.load_study(C.DEFAULT_STUDY_CONFIG, mode="test"))
    run_root = base / RUN_ID
    for name in ("cells", "requests", "logs", "plan/manifests", "registers", "status"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"run_root": str(run_root), "driven": {}, "pending": {},
                              "failures": []}
    #: Every cell id ``drive_manifest`` actually invoked, for the terminal census below.
    dispatched_cells: set[str] = set()
    started = time.time()
    print(f"[dry-run] run root {run_root}")

    build_items(run_root, cfg, per_family=frame)
    inventory = dispatched_panels(
        PLAN.Inventory.from_run_root(run_root, cfg).limited(per_family=dispatched))
    print(f"[dry-run] items: {sum(len(v) for v in inventory.items.values())} in "
          f"{len(inventory.families)} families")

    server, tokenizer = start_fake_fleet(run_root, cfg)
    inject_tokenizer(tokenizer)
    try:
        dag = PLAN.build_dag(cfg, inventory, stage=CE.STAGE_SMOKE,
                             blocks=DRIVEN_BLOCKS + (CLOSE_OUT_BLOCK,))
        hashes = dag.write_manifests(run_root, meta={"stage": "dry_run"})
        print(f"[dry-run] manifests: { {b: len(c) for b, c in dag.manifests.items()} }")

        for block in DRIVEN_BLOCKS:
            if block not in dag.manifests:
                report["failures"].append(f"{block}: the DAG emitted no manifest for it")
                continue
            manifest = CE.manifest_path(run_root, block)
            limit = BLOCK_LIMITS.get(block, 1 if quick else None)
            codes = drive_manifest(run_root, manifest, config=cfg.path, limit=limit)
            dispatched_cells.update(c.cell_id for c in CE.load_manifest(manifest)[:len(codes)])
            report["driven"][block] = {"cells": len(codes), "exit_codes": sorted(set(codes))}
            print(f"[dry-run] {block:16s} {len(codes)} cells -> {sorted(set(codes))}")
            if not codes:
                # A manifest with no cells is not a pass.  Only CAPTURE and AUDIT are allowed
                # to be empty here, and both say so in report["pending"]; a declared block that
                # silently built nothing would take its kind out of R3's coverage without
                # anyone noticing.
                report["failures"].append(f"{block}: the manifest holds no cell to drive")
            unexpected = unexpected_exits(run_root, manifest, codes)
            if unexpected:
                report["failures"].append(f"{block}: {unexpected}")

        # ---- the planner: seal the pools and the panel, drive what they unblock -------
        planner = PLAN.planner_pass(run_root, cfg, inventory=inventory,
                                    counter=lambda text: len(str(text).split()))
        print(f"[dry-run] planner: {planner.line()} {planner.detail}")
        driven = {CE.manifest_path(run_root, block) for block in DRIVEN_BLOCKS
                  if block in dag.manifests} | {CE.manifest_path(run_root, CLOSE_OUT_BLOCK)}
        emitted = [path for path in sorted((run_root / "plan" / "manifests").glob("*.json"))
                   if path not in driven and path.stem != "DEPS_GATE"]
        for manifest in emitted:
            block = CE.load_manifest(manifest)[0].block
            if block == CE.Block.CAPTURE:
                report["pending"][CE.Block.CAPTURE] = (
                    "the capture worker needs torch and the pinned Qwen3 tokenizer; "
                    "tests/nextdyn/test_capture.py drives the kind")
                continue
            codes = drive_manifest(run_root, manifest, config=cfg.path,
                                   limit=BLOCK_LIMITS.get(block))
            dispatched_cells.update(c.cell_id for c in CE.load_manifest(manifest)[:len(codes)])
            report["driven"][f"{block}/{manifest.stem}"] = {"cells": len(codes),
                                                            "exit_codes": sorted(set(codes))}
            print(f"[dry-run] {block:16s} {len(codes)} cells -> {sorted(set(codes))} "
                  f"({manifest.name})")
            unexpected = unexpected_exits(run_root, manifest, codes)
            if unexpected:
                report["failures"].append(f"{manifest.name}: {unexpected}")
        if not any(key.startswith(CE.Block.AUDIT) for key in report["driven"]):
            report["pending"][CE.Block.AUDIT] = "the planner sealed no pool"
        # ---- the close-out cell, once every other cell is terminal --------------------
        close_out = CE.manifest_path(run_root, CLOSE_OUT_BLOCK)
        codes = drive_manifest(run_root, close_out, config=cfg.path)
        dispatched_cells.update(c.cell_id for c in CE.load_manifest(close_out)[:len(codes)])
        report["driven"][CLOSE_OUT_BLOCK] = {"cells": len(codes), "exit_codes": sorted(set(codes))}
        print(f"[dry-run] {CLOSE_OUT_BLOCK:16s} {len(codes)} cells -> {sorted(set(codes))}")
        if any(code != EXIT_DONE for code in codes):
            report["failures"].append(f"{CLOSE_OUT_BLOCK}: exit codes {sorted(set(codes))}")

        # ---- the hard dependency gate -----------------------------------------------
        report["dependency_gate"] = dependency_gate(run_root, cfg, dag, server)
        if not report["dependency_gate"]["held"]:
            report["failures"].append("the dependency gate did not hold")

        # ---- resume ------------------------------------------------------------------
        report["resume"] = resume_check(run_root, cfg, dag, server, quick=quick)
        if not report["resume"]["unchanged"]:
            report["failures"].append(
                f"resume regenerated requests ({report['resume']['after']} > "
                f"{report['resume']['before']} calls)")
        if not report["resume"]["archive_kept"]:
            report["failures"].append(
                f"resume dropped the per-request archive "
                f"({report['resume']['requests_rows_after']} < "
                f"{report['resume']['requests_rows_before']} requests rows)")

        # ---- the freeze gate every cell called ---------------------------------------
        report["freeze"] = freeze_check(run_root, cfg)
        if not report["freeze"]["held"]:
            report["failures"].append(f"the freeze gate did not hold: {report['freeze']}")

        # ---- the terminal census (plan §3.4: release-check gates on it) ----------------
        report["plan_table"] = plan_census(run_root, dag, hashes, sorted(dispatched_cells))
        if report["plan_table"]["non_terminal"]:
            report["failures"].append(
                f"plan census: dispatched cell(s) are not terminal: "
                f"{report['plan_table']['non_terminal'][:3]}")

        # ---- close-out ---------------------------------------------------------------
        published = MAT.materialize(run_root, strict=True)
        report["tables"] = dict(published.rows)
        print(f"[dry-run] tables_v{published.version}: {sum(published.rows.values())} rows")
        violations = MAT.assert_published_keys(run_root / MAT.TABLES_DIR,
                                               run_root / MAT.PROTECTED_DIR / MAT.TABLES_DIR,
                                               strict=False)
        referential = tables.assert_referential(run_root / MAT.TABLES_DIR, strict=True)
        report["keys"] = {"violations": list(violations), "referential": list(referential)}
        if violations or referential:
            report["failures"].append(f"keys: {report['keys']}")
        report["analysis"] = analyze(run_root, cfg)
    finally:
        server.stop()

    report["wall_s"] = round(time.time() - started, 1)
    (run_root / "dry_run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                             default=str) + "\n",
                                                  encoding="utf-8")
    return report


def requests_rows_on_disk(run_root: Path, dag: Any) -> int:
    """How many ``requests`` rows the driven cells' item files carry (plan §3.7).

    The per-request archive has exactly one home -- the item file of the cell that issued the
    request -- so a resumed cell that rewrites its item file without its rows deletes it.  The
    count is taken before and after the resume and a drop is a failure, because the loss is
    invisible everywhere else: the tables still pass their keys, they just describe 9% of the
    campaign's replicas, latencies and token counts.
    """
    total = 0
    for block in DRIVEN_BLOCKS:
        for cell in dag.manifests.get(block, ()):
            directory = deps.cell_dir(run_root, cell.cell_id) / tables.ITEMS_DIR
            for path in sorted(directory.glob("*.json")) if directory.exists() else ():
                data = json.loads(path.read_text(encoding="utf-8"))
                total += len(data.get("rows", {}).get("requests", ()))
    return total


def freeze_check(run_root: Path, cfg: Any) -> dict[str, Any]:
    """The gate every cell called: unfrozen and permitted here, refused for a confirmation run.

    The write-once ``FROZEN.yaml`` path itself is out of scope for the dry run -- freezing needs
    the locks, budgets, sample-size lock and analysis lock hash a *human* authorizes at S3, and
    manufacturing them here would be a frozen manifest that asserts nothing.
    ``tests/nextdyn/test_freeze.py`` owns that half.  What the dry run can assert is the branch
    its own cells actually take, and that the other branch is a refusal rather than a default.
    """
    from dataclasses import replace as _replace

    from agents_scaling.nextdyn import contract as K
    from agents_scaling.nextdyn import freeze as FREEZE
    from agents_scaling.nextdyn.errors import LockError

    unfrozen = FREEZE.gate(run_root, cfg) is None and not FREEZE.is_frozen(run_root)
    confirmation = _replace(cfg, study=_replace(cfg.study, mode=K.MODE_CONFIRMATION))
    try:
        FREEZE.gate(run_root, confirmation)
    except LockError:
        refused = True
    else:
        refused = False
    print(f"[dry-run] freeze gate: unfrozen test run ok={unfrozen}, "
          f"confirmation refused={refused}")
    return {"unfrozen_run_passes": bool(unfrozen), "confirmation_refused": bool(refused),
            "held": bool(unfrozen and refused)}


def dependency_gate(run_root: Path, cfg: Any, dag: Any, server: Any) -> dict[str, Any]:
    """A cell whose bundle is absent: exit 5, no ``meta.json``, zero additional calls."""
    before = server.chat_count
    missing = CE.CellSpec(
        cell_id="episode.DRYRUN.primary.deadbeef.deadbeefdead.smoke.r0.missing.s0",
        kind=CellKind.EPISODE, lane="8B", pool=Pool.DEADLINE, model="primary",
        family=cfg.cohorts.families[0], tokens_hash=cfg.contract.tokens.budget_hash(),
        condition=CE.ConditionRef(protocol_hash="deadbeefdead", display_label="PRIVATE"),
        items=(dag.cells[0].items[0],), replicate=0, split_role="smoke",
        depends_on=(deps.bundle_dependency(cfg.cohorts.families[0], "b9999"),),
        options={"origin": "constructed", "bucket": "b9999"}, block=CE.Block.E1_CONSTRUCTED)
    path = run_root / "plan" / "manifests" / "DEPS_GATE.json"
    CE.write_manifest(path, [missing], block=CE.Block.E1_CONSTRUCTED)
    code = drive_cell(run_root, path, 0, config=cfg.path)
    meta = (deps.cell_dir(run_root, missing.cell_id) / deps.META_FILE).exists()
    unmet = (deps.cell_dir(run_root, missing.cell_id) / PLAN.DEPS_UNMET_FILE).exists()
    held = code == EXIT_DEPS_UNMET and not meta and unmet and server.chat_count == before
    print(f"[dry-run] dependency gate -> exit {code} (meta.json: {meta}, DEPS_UNMET: {unmet}, "
          f"no requests: {server.chat_count == before})")
    return {"exit_code": code, "meta_json": meta, "deps_unmet_file": unmet,
            "calls_before": before, "calls_after": server.chat_count, "held": bool(held)}


def resume_check(run_root: Path, cfg: Any, dag: Any, server: Any, *, quick: bool) -> dict[str, Any]:
    """Delete every item file and re-drive: the store must serve every request from cache."""
    rows_before = requests_rows_on_disk(run_root, dag)
    for block in DRIVEN_BLOCKS:
        if block not in dag.manifests:
            continue
        for cell in dag.manifests[block]:
            directory = deps.cell_dir(run_root, cell.cell_id)
            shutil.rmtree(directory / tables.ITEMS_DIR, ignore_errors=True)
            (directory / deps.META_FILE).unlink(missing_ok=True)
    before = server.chat_count
    for block in DRIVEN_BLOCKS:
        if block not in dag.manifests:
            continue
        drive_manifest(run_root, CE.manifest_path(run_root, block), config=cfg.path,
                       limit=BLOCK_LIMITS.get(block, 1 if quick else None))
    after = server.chat_count
    rows_after = requests_rows_on_disk(run_root, dag)
    print(f"[dry-run] resume: {before} -> {after} chat calls "
          f"({'unchanged' if before == after else 'CHANGED'}); requests rows "
          f"{rows_before} -> {rows_after}")
    return {"before": before, "after": after, "unchanged": before == after,
            "requests_rows_before": rows_before, "requests_rows_after": rows_after,
            "archive_kept": rows_after >= rows_before}


def plan_census(run_root: Path, dag: Any, hashes: Mapping[str, str],
                dispatched: Sequence[str]) -> dict[str, Any]:
    """Write the ``plan`` table from the close-out census and report what it says.

    ``release-check`` gates on "every planned cell is terminal", and the only producer of that
    claim is ``plan.write_plan_table(plan_rows(..., final=True))``.  Driven here so the gate has
    end-to-end coverage, and so ``cell_status(final=True)`` -- which is what classifies an
    unfilled bucket as ``skipped_quota_unfilled`` rather than as a failure -- is exercised on a
    run root where such buckets genuinely exist.

    The table carries one row per *planned* cell, as a real run's does.  Terminality is asserted
    over the cells this run actually dispatched: the dry run is a kind check, not a load test,
    so ``BLOCK_LIMITS`` deliberately leaves most of E2B's cells unplanned-for, and calling those
    a failure would be reporting the limit rather than the pipeline.
    """
    rows = PLAN.plan_rows(run_root, dag, manifest_hashes=dict(hashes), final=True)
    path = PLAN.write_plan_table(run_root, rows)
    ran = set(dispatched)
    by_status: dict[str, int] = {}
    non_terminal: list[str] = []
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
        if row.cell_id in ran and row.status not in PLAN.TERMINAL_STATUSES:
            non_terminal.append(f"{row.cell_id}: {row.status}")
    print(f"[dry-run] plan census: {len(rows)} planned cells {by_status}, "
          f"{len(ran)} dispatched, {len(non_terminal)} not terminal -> {path.name}")
    return {"rows": len(rows), "by_status": by_status, "dispatched": len(ran),
            "non_terminal": non_terminal, "path": str(path)}


def analyze(run_root: Path, cfg: Any) -> dict[str, Any]:
    """``analyze --seal`` then the locked pass, in the test profile (plan §4.5)."""
    from agents_scaling.nextdyn import analysis as AN

    sealed = AN.run(run_root, config_path=cfg.path, profile="test", seal_only=True)
    locked = AN.run(run_root, config_path=cfg.path, profile="test")
    ledger = run_root / "derived" / "claim_ledger.csv"
    print(f"[dry-run] analyze: sealed={sealed.line()} locked={locked.line()}")
    return {"sealed": sealed.line(), "locked": locked.line(), "ledger": ledger.exists(),
            "ledger_rows": len(ledger.read_text().splitlines()) if ledger.exists() else 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", default=None, help="where to build the temp run root")
    parser.add_argument("--keep", action="store_true", help="do not delete the run root")
    parser.add_argument("--quick", action="store_true", help="one cell per block")
    parser.add_argument("--items", type=int, default=DISPATCHED_PER_FAMILY,
                        help="items per family to DISPATCH (the frame is always %(default)s+)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base = Path(args.run_root) if args.run_root else Path(tempfile.mkdtemp(prefix="nextdyn_dry_"))
    report = run(base, quick=args.quick, dispatched=int(args.items))
    ok = not report["failures"]
    print(f"[dry-run] {'GREEN' if ok else 'RED'} in {report['wall_s']}s; report at "
          f"{report['run_root']}/dry_run_report.json")
    for failure in report["failures"]:
        print(f"[dry-run] FAILURE: {failure}", file=sys.stderr)
    for kind, note in report["pending"].items():
        print(f"[dry-run] pending {kind}: {note}")
    if not args.keep and ok:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
