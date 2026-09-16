"""The per-cell runner: the dependency gate, the directory refusal, the concurrency site.

Plan §3.3 / §3.4, spec 15.3, contract §2.5.  Everything here is about what happens *before*
any request is issued -- which is the half ARD36 got wrong: its dependency check logged and
proceeded, so a cell could burn GPU time against an arm that was not the registered arm.
"""

from __future__ import annotations

import json
import threading

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import deps, kinds, run_one, runner as RUN, tables
from agents_scaling.nextdyn.cells import Block, CellSpec, ConditionRef
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool
from agents_scaling.study.types import EXIT_DONE, EXIT_SUSPENDED

ITEMS = ("nditem0000000001", "nditem0000000002")


def bank_cell(cell_id: str = "difficulty_bank.MATH.primary.deadbeef.x.pilot.r0.s0", *,
              items=ITEMS, depends_on=()) -> CellSpec:
    return CellSpec(cell_id=cell_id, kind=CellKind.DIFFICULTY_BANK, lane=Lane.SMALL,
                    pool=Pool.LOW, model="primary", family="MATH", tokens_hash="deadbeef",
                    condition=None, items=items, replicate=0, split_role="pilot",
                    depends_on=depends_on, options={"extension_stage": 0}, block=Block.BANKS)


def episode_cell(*, depends_on) -> CellSpec:
    return CellSpec(cell_id="episode.MATH.primary.deadbeef.aaaaaaaaaaaa.pilot.r0.constructed.b0000.m0.s0",
                    kind=CellKind.EPISODE, lane=Lane.SMALL, pool=Pool.DEADLINE, model="primary",
                    family="MATH", tokens_hash="deadbeef",
                    condition=ConditionRef(protocol_hash="a" * 12, display_label="PRIVATE",
                                           composition="C1_COHERENT"),
                    items=ITEMS[:1], replicate=0, split_role="pilot", depends_on=depends_on,
                    options={"origin": "constructed", "bucket": "b0000"},
                    block=Block.E1_CONSTRUCTED)


def make_runner(cell, run_root, study_cfg, compute_cfg, **kwargs) -> RUN.NextdynCellRunner:
    return RUN.NextdynCellRunner(cell, run_root, run_root, 0, threading.Event(), cfg=study_cfg,
                                 compute=compute_cfg, items_loader=lambda root: {}, **kwargs)


# --------------------------------------------------------------------------- knobs


def test_worker_concurrency_is_the_one_site_that_reads_the_kind_knobs(study_cfg, compute_cfg):
    """D-03: ARD36 declared ``max_inflight`` and never read it, so rounds went out serially."""
    contract = study_cfg.contract
    episode = RUN.worker_concurrency(CellKind.EPISODE, compute=compute_cfg, contract=contract,
                                     model=contract.model("primary"))
    assert episode.max_inflight == contract.team.n_agents      # 'n_agents' resolved, not a literal
    block = compute_cfg.cells.kinds["episode"]
    assert episode.parallel_items == block.parallel_items
    bank = RUN.worker_concurrency(CellKind.DIFFICULTY_BANK, compute=compute_cfg, contract=contract,
                                  model=contract.model("primary"))
    assert bank.max_inflight == compute_cfg.cells.kinds["difficulty_bank"].max_inflight


def test_the_runner_carries_the_declared_width_into_the_study_projection(study_cfg, compute_cfg,
                                                                        tmp_run_root):
    runner = make_runner(bank_cell(), tmp_run_root, study_cfg, compute_cfg)
    assert runner.study_cell.max_inflight == runner.concurrency.max_inflight
    assert runner.study_cell.parallel_items == runner.concurrency.parallel_items
    # the workers read the NEXTDYN cell (its kind, its options), not the projection
    assert runner.cell is runner.nd_cell
    assert runner.cell.kind is CellKind.DIFFICULTY_BANK


def test_resolve_worker_names_the_work_package_when_a_worker_is_missing():
    for kind in CellKind:
        assert callable(RUN.resolve_worker(kind)), kind
    assert kinds.spec_for(CellKind.EPISODE).worker.endswith(":episode_worker")


# --------------------------------------------------------------------------- the gate


def test_deps_unmet_writes_no_meta_and_issues_no_request(tmp_run_root, study_cfg, compute_cfg):
    """plan §3.4 / D12: a hard dependency is refused, never assumed.

    The cell exits ``EXIT_DEPS_UNMET`` with ``DEPS_UNMET.json`` naming what it waited for, no
    ``meta.json`` (so nothing downstream reads it as complete) and -- the part that matters --
    no endpoint was opened and no request was generated.
    """
    cell = episode_cell(depends_on=(deps.bundle_dependency("MATH", "b0000"),))
    runner = make_runner(cell, tmp_run_root, study_cfg, compute_cfg)
    opened = []
    runner._connect = lambda ckpt: opened.append(ckpt)         # would be the first GPU contact
    assert runner.run() == RUN.EXIT_DEPS_UNMET
    assert not opened
    directory = deps.cell_dir(tmp_run_root, cell.cell_id)
    assert not (directory / deps.META_FILE).exists()
    unmet = json.loads((directory / RUN.DEPS_UNMET_FILE).read_text())
    assert unmet["exit_code"] == RUN.EXIT_DEPS_UNMET
    assert [u["dep"] for u in unmet["unmet"]] == [deps.bundle_dependency("MATH", "b0000")]
    assert not list((directory / tables.ITEMS_DIR).glob("*.json"))
    assert not (tmp_run_root / "requests").exists() or \
        not list((tmp_run_root / "requests").rglob("*.json"))


def test_a_tampered_register_is_unmet_not_met(tmp_run_root, study_cfg, compute_cfg):
    sha = deps.write_register(tmp_run_root, deps.REGISTER_POOL, {"entries": [{"a": "b"}]})
    path = deps.register_path(tmp_run_root, deps.REGISTER_POOL, sha)
    path.write_text(path.read_text().replace('"b"', '"c"'))
    cell = CellSpec(cell_id="selection_audit.x.scale.deadbeef.x.pilot.r0.p0.s0",
                    kind=CellKind.SELECTION_AUDIT, lane=Lane.AUDIT, pool=Pool.AUDIT,
                    model="scale", family="", tokens_hash="deadbeef", condition=None,
                    items=ITEMS[:1], replicate=0, split_role="pilot",
                    depends_on=(deps.register_dependency(deps.REGISTER_POOL, sha),),
                    options={"pool": sha}, block=Block.AUDIT)
    runner = make_runner(cell, tmp_run_root, study_cfg, compute_cfg)
    assert runner.run() == RUN.EXIT_DEPS_UNMET
    statuses = json.loads((deps.cell_dir(tmp_run_root, cell.cell_id) / RUN.DEPS_UNMET_FILE)
                          .read_text())["unmet"]
    assert statuses[0]["status"] == deps.STATUS_TAMPERED


# --------------------------------------------------------------------------- ownership


def test_a_cell_refuses_a_directory_that_holds_another_manifests_items(tmp_run_root, study_cfg,
                                                                      compute_cfg):
    """D-04: two manifests given one cell id would merge two cohorts in one directory.

    The requests are content-addressed, so re-dispatching under a correct id is free; silently
    merging two cohorts is not, and nothing downstream could tell -- every item file is
    individually well formed.
    """
    cell = bank_cell()
    items = deps.cell_dir(tmp_run_root, cell.cell_id) / tables.ITEMS_DIR
    items.mkdir(parents=True)
    (items / "foreign_item_00.json").write_text("{}")
    runner = make_runner(cell, tmp_run_root, study_cfg, compute_cfg)
    assert runner.run() == EXIT_SUSPENDED
    suspended = json.loads((deps.cell_dir(tmp_run_root, cell.cell_id)
                            / deps.SUSPENDED_FILE).read_text())
    assert "does not declare" in suspended["error"]
    assert not (deps.cell_dir(tmp_run_root, cell.cell_id) / deps.META_FILE).exists()


def test_an_item_file_that_claims_another_cell_is_refused(tmp_run_root, study_cfg, compute_cfg):
    cell = bank_cell()
    path = tables.item_path(tmp_run_root, cell.cell_id, ITEMS[0])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"cell_id": "another.cell", "source_item_id": ITEMS[0],
                                "status": "complete"}))
    runner = make_runner(cell, tmp_run_root, study_cfg, compute_cfg)
    assert runner.run() == EXIT_SUSPENDED


# --------------------------------------------------------------------------- the CLI


def test_a_second_runner_on_a_locked_cell_exits_zero(tmp_run_root, study_cfg, compute_cfg,
                                                     monkeypatch):
    """Two array tasks on one cell: the second exits 0 immediately, never waits.

    A waiting array task is a Slurm slot that could be running another cell, and the first
    runner is already doing the work.
    """
    import fcntl

    cell = bank_cell()
    manifest = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(manifest, [cell], block=Block.BANKS)
    directory = deps.cell_dir(tmp_run_root, cell.cell_id)
    directory.mkdir(parents=True, exist_ok=True)
    holder = open(directory / run_one.LOCK_FILE, "a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        code = run_one.main(["--run-root", str(tmp_run_root), "--manifest", str(manifest),
                             "--index", "0", "--config", str(study_cfg.path)])
    finally:
        holder.close()
    assert code == EXIT_DONE
    assert not (directory / deps.META_FILE).exists()


def test_run_one_refuses_a_manifest_whose_sidecar_does_not_match(tmp_run_root, study_cfg):
    manifest = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(manifest, [bank_cell()], block=Block.BANKS)
    manifest.with_name(manifest.name + CE.SHA_SUFFIX).write_text("00" * 32 + "  x\n")
    assert run_one.main(["--run-root", str(tmp_run_root), "--manifest", str(manifest),
                         "--index", "0", "--config", str(study_cfg.path)]) == EXIT_SUSPENDED


def test_run_one_dry_run_resolves_the_cell_without_touching_it(tmp_run_root, study_cfg, capsys):
    manifest = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(manifest, [bank_cell()], block=Block.BANKS)
    assert run_one.main(["--run-root", str(tmp_run_root), "--manifest", str(manifest),
                         "--index", "0", "--config", str(study_cfg.path), "--dry-run"]) == EXIT_DONE
    printed = json.loads(capsys.readouterr().out)
    assert printed["cell"]["cell_id"] == bank_cell().cell_id
    assert not deps.cell_dir(tmp_run_root, bank_cell().cell_id).exists()


def test_a_reused_cell_is_never_dispatched(tmp_run_root, study_cfg):
    cell = bank_cell()
    reused = CE.CellSpec(**{**cell.to_dict(), "kind": cell.kind, "reuse_of": "some.other.cell"})
    manifest = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(manifest, [reused], block=Block.BANKS)
    assert run_one.main(["--run-root", str(tmp_run_root), "--manifest", str(manifest),
                         "--index", "0", "--config", str(study_cfg.path)]) == EXIT_DONE
    assert not (deps.cell_dir(tmp_run_root, cell.cell_id) / deps.META_FILE).exists()


def test_the_pool_selects_the_sub_run_root(tmp_run_root):
    assert RUN.pool_run_root(tmp_run_root, Pool.DEADLINE) == tmp_run_root / "pools" / "deadline"
    assert RUN.pool_run_root(tmp_run_root, "low") == tmp_run_root / "pools" / "low"


def test_the_counting_tokenizer_is_both_a_tokenizer_and_a_counter():
    """``banks`` passes it to ``parse_public_output`` as a counter; ``episode`` wraps it."""
    class Stub:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    wrapped = RUN.CountingTokenizer.of(Stub())
    assert wrapped("a b c") == 3
    assert wrapped.decode([1, 2]) == "1 2"
    assert RUN.CountingTokenizer.of(wrapped) is wrapped
    from agents_scaling.study.packets import make_counter

    assert make_counter(wrapped)("a b") == 2
