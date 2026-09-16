"""The DAG, the pilot subset, the mechanical reuse census and budget admission.

Plan §3.1 (workload), §3.4 (frozen manifests, no waves), §1.3 (the mechanical census),
spec 19.1 (the pilot rows), 16.4 (the authorized ceiling).

Four of the names here are spec-18 names: ``test_budget_admission_reserves_selection_cost`` and
``test_stop_is_available_in_every_action_policy`` (18.5), and
``test_n3_ring_equals_all_to_all_is_not_duplicate_treatment`` (18.2) -- the planner half of the
claim whose graph half lives in ``test_graphs.py``.
"""

from __future__ import annotations

import dataclasses

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import deps, plan as PLAN, tables, workload as WL
from agents_scaling.nextdyn.cells import Block, Bucket
from agents_scaling.nextdyn.ids import CellKind

PER_FAMILY = 4


@pytest.fixture
def inventory(study_cfg) -> PLAN.Inventory:
    families = tuple(study_cfg.cohorts.families)
    items = {f: tuple(f"{f[:3].lower()}item{i:04d}" for i in range(PER_FAMILY)) for f in families}
    return PLAN.Inventory(
        families=families, items=items, natural_items=items,
        panels={p: {f: items[f] for f in families} for p in CE.PANELS},
        buckets={f: tuple(Bucket(family=f, index=i, source_item_id=item, pair="primary")
                          for i, item in enumerate(items[f])) for f in families},
        problems=tuple(f"mod97{i:03d}" for i in range(2)))


@pytest.fixture
def dag(study_cfg, compute_cfg, inventory):
    return PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)


# --------------------------------------------------------------------------- the DAG


def test_the_dag_emits_every_enabled_block_once_with_path_addressed_dependencies(dag):
    """plan §3.4: no waves, no INDEX ledger -- a bucket cell waits for a path it can compute."""
    assert Block.BANKS in dag.manifests and Block.E1_CONSTRUCTED in dag.manifests
    for block, cells in dag.manifests.items():
        assert len({c.cell_id for c in cells}) == len(cells), f"{block} repeats a cell id"
    constructed = dag.manifests[Block.E1_CONSTRUCTED]
    assert all(dep.startswith(deps.BUNDLE_PREFIX) for c in constructed for dep in c.depends_on)
    # every dependency is resolvable by the reader the runner uses (one vocabulary, D16)
    for cell in dag.cells:
        for dep in cell.depends_on:
            assert deps.parse_dependency(dep).kind in ("cell", "bundle", "register")


def test_the_close_out_cell_is_not_blocked_by_a_bucket_the_scan_never_filled(dag):
    """A blanket ``cell:`` dependency on everything would let one unfillable bucket block the
    close-out forever; the census in the ``plan`` table is what says the run is done."""
    materialize = dag.manifests[Block.MATERIALIZE]
    assert len(materialize) == 1
    assert materialize[0].depends_on == ()


def test_replanning_reproduces_the_same_manifest_bytes(study_cfg, compute_cfg, inventory,
                                                       tmp_run_root):
    """``release-check: replan_reproduces_manifests`` -- the DAG is a pure function."""
    first = PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    second = PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    assert first.write_manifests(tmp_run_root) == second.write_manifests(tmp_run_root)


# --------------------------------------------------------------------------- reuse


def test_n3_ring_equals_all_to_all_is_not_duplicate_treatment(study_cfg, compute_cfg, inventory):
    """spec 18.2: a condition that resolves to an existing one is *recorded*, not dispatched.

    ``IN1_OUT1`` resolves to the complete graph, so its episodes are byte-for-byte PEER's --
    the same claim ``test_graphs.py`` makes about an N=3 ring.  The planner's half is that the
    duplicate is emitted as a ``reuse_of`` cell (one treatment, recorded twice, dispatched
    once) rather than as a second arm whose numbers would double-count.
    """
    dag = PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    registry = PLAN._load_registry(study_cfg)
    equivalent = {name: spec.equivalent_to for name, spec in registry.episodes.items()
                  if spec.equivalent_to}
    assert equivalent, "protocols.yaml declares no equivalence; the census would be vacuous"
    reused = [c for c in dag.cells if c.reuse_of]
    assert reused, "a declared equivalence produced no reuse_of cell"
    labels = {c.display_label for c in reused}
    assert labels <= set(equivalent)
    for cell in reused:
        twin = dag.cell(cell.reuse_of)
        assert twin.cell_id != cell.cell_id
        assert twin.items == cell.items and twin.family == cell.family
        assert twin.display_label == equivalent[cell.display_label]
    # and the census charges them nothing: a reused cell issues no new request
    load = WL.calls(dag.cells, contract=study_cfg.contract, reuse_fraction=dag.reuse)
    assert load.unique_calls < load.raw_calls


def test_the_reuse_census_is_mechanical_and_names_the_arms_that_alias(study_cfg):
    """plan §1.3: render one synthetic team's prompts per condition, hash them, no model call."""
    census = PLAN.reuse_census(study_cfg, PLAN._load_registry(study_cfg))
    assert census.total_prompts > 0 and census.total_unique <= census.total_prompts
    # PRIVATE and PEER are the primary arms and neither reuses the other
    assert census.fractions["PRIVATE"] == 0.0
    assert census.fractions["PEER"] == 0.0
    # IN1_OUT1 resolves to the complete graph: every one of its prompts is PEER's
    assert census.fractions["IN1_OUT1"] == pytest.approx(1.0)
    # the isolated focal slot of IN0_OUT1 is that slot's PRIVATE trajectory (plan §1.3)
    assert 0.0 < census.fractions["IN0_OUT1"] < 1.0
    # a filled arm shows an inert filler where the missing packet would be: different bytes
    assert census.fractions["IN0_OUT1_FILLED"] < census.fractions["IN0_OUT1"]


# --------------------------------------------------------------------------- pilot


def test_pilot_subsets_match_spec_19_1_exactly(study_cfg):
    """spec 19.1: every pilot row size is read from ``study.yaml``, never re-typed in code."""
    subset = PLAN.pilot_subsets(study_cfg)
    pilot = study_cfg.pilot
    assert subset.natural_items_per_family == pilot.natural_items_per_family
    assert subset.constructed_paired_items_per_family == pilot.constructed_paired_items_per_family
    assert subset.directional_items_per_family == pilot.directional_items_per_family
    assert subset.natural_receiver_items == pilot.natural_receiver_items
    assert subset.natural_receiver_tasks == tuple(pilot.natural_receiver_tasks)
    assert subset.synthetic_receiver_problems == pilot.synthetic_receiver_problems
    assert subset.adapter_check_problems == pilot.adapter_check_problems
    assert subset.capture_items_per_family == pilot.capture_items_per_family
    # and the pilot inventory is exactly those prefixes
    families = tuple(study_cfg.cohorts.families)
    items = {f: tuple(f"{f}-{i}" for i in range(500)) for f in families}
    inventory = PLAN.Inventory(
        families=families, items=items, natural_items=items,
        panels={p: {f: items[f] for f in families} for p in CE.PANELS},
        buckets={f: tuple(Bucket(family=f, index=i, source_item_id=x, pair="primary")
                          for i, x in enumerate(items[f])) for f in families})
    reduced = PLAN.pilot_inventory(study_cfg, inventory)
    for family in families:
        assert len(reduced.natural_items[family]) == subset.natural_items_per_family
        assert len(reduced.items[family]) == subset.constructed_paired_items_per_family
        assert len(reduced.panel(CE.PANEL_E3A, family)) == subset.directional_items_per_family
        assert len(reduced.panel(CE.PANEL_E2A, family)) == subset.natural_receiver_items
        assert len(reduced.panel(CE.PANEL_CAPTURE, family)) == subset.capture_items_per_family


# --------------------------------------------------------------------------- admission


def test_budget_admission_reserves_selection_cost(study_cfg, compute_cfg, inventory):
    """spec 18.5 / plan §3.4: the audit, the capture and the grading are charged FIRST.

    ARD36 admitted episodes against the whole ceiling and discovered at the end that the
    selection audit had no budget -- at which point the only move left was to drop the
    measurement the spec asks for.  Reserving first means the run that cannot afford its audit
    is refused at ``plan --dry-run``, before a GPU-hour is spent.
    """
    dag = PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    pools = tuple((cell.items[0], "ab" * 8) for cell in dag.manifests[Block.E1_NATURAL])
    state = dataclasses.replace(dag.state, pool_registers=pools)
    audit = CE.build_block(Block.AUDIT, state)
    assert audit, "the audit block built nothing; the reservation would be vacuous"
    cells = list(dag.cells) + audit
    load = WL.calls(cells, contract=study_cfg.contract, reuse_fraction=dag.reuse)
    reserved = sum(b.gpu_hours for b in load.blocks if b.block in WL.RESERVED_BLOCKS)
    assert reserved > 0

    ceiling = dataclasses.replace(study_cfg.budgets, campaign_gpu_hours=load.gpu_hours * 2,
                                  storage_limit_bytes=int(load.storage_bytes * 2))
    admission = WL.admit(load, budgets=ceiling)
    assert admission.admitted
    assert admission.reserved_gpu_hours == pytest.approx(reserved)
    assert admission.episode_gpu_hours == pytest.approx(load.gpu_hours - reserved)

    # a ceiling that covers the episodes but not the reserved blocks is REFUSED, and says so
    tight = dataclasses.replace(study_cfg.budgets, campaign_gpu_hours=reserved / 2,
                                storage_limit_bytes=int(load.storage_bytes * 2))
    refused = WL.admit(load, budgets=tight)
    assert not refused.admitted
    assert "reserved" in refused.detail


def test_stop_is_available_in_every_action_policy(study_cfg, compute_cfg, inventory):
    """spec 18.5: dispatching nothing is always feasible, and is reported as such.

    A planner that can only answer "run everything" has no way to say "this does not fit", so
    the refusal has to be a *decision* it can take: ``Admission.stop_feasible`` is true under
    every ceiling, including a null one (no authorization yet) and a zero one.
    """
    dag = PLAN.build_dag(study_cfg, inventory, stage=CE.STAGE_CONFIRMATION, compute=compute_cfg)
    load = WL.calls(dag.cells, contract=study_cfg.contract, reuse_fraction=dag.reuse)
    for hours in (None, 0.0, load.gpu_hours * 10):
        budgets = dataclasses.replace(study_cfg.budgets, campaign_gpu_hours=hours)
        admission = WL.admit(load, budgets=budgets)
        assert admission.stop_feasible
        if hours is None:
            assert not admission.admitted and "null" in admission.detail
    # STOP costs nothing: the empty plan is admissible under any authorized ceiling
    empty = WL.calls([], contract=study_cfg.contract)
    assert empty.gpu_hours == 0.0
    assert WL.admit(empty, budgets=dataclasses.replace(
        study_cfg.budgets, campaign_gpu_hours=1.0, campaign_cpu_hours=1.0,
        storage_limit_bytes=1)).admitted


def test_the_workload_counts_calls_from_the_contract_not_from_a_guess(study_cfg, dag):
    contract = study_cfg.contract
    bank = next(c for c in dag.cells if c.kind is CellKind.DIFFICULTY_BANK)
    assert WL.calls_per_item(bank, contract=contract) == contract.banks.difficulty_draws
    episode = next(c for c in dag.cells if c.kind is CellKind.EPISODE)
    assert WL.calls_per_item(episode, contract=contract) == (contract.team.n_agents
                                                             * contract.team.revision_rounds)
    receiver = next(c for c in dag.cells if c.kind is CellKind.RECEIVER)
    assert WL.calls_per_item(receiver, contract=contract) == study_cfg.experiments.E2b.generation_phases


# --------------------------------------------------------------------------- the plan table


def test_the_plan_table_status_comes_from_the_terminal_census(tmp_run_root, dag):
    """plan §1.4: "the directory exists" is not a status."""
    cell = dag.manifests[Block.BANKS][0]
    directory = deps.cell_dir(tmp_run_root, cell.cell_id)
    (directory / "items").mkdir(parents=True)
    assert PLAN.cell_status(tmp_run_root, cell) == PLAN.STATUS_PLANNED     # a bare directory
    (directory / "items" / f"{cell.items[0]}.json").write_text("{}")
    assert PLAN.cell_status(tmp_run_root, cell) == PLAN.STATUS_RUNNING
    (directory / deps.SUSPENDED_FILE).write_text("{}")
    assert PLAN.cell_status(tmp_run_root, cell) == PLAN.STATUS_FAILED

    rows = PLAN.plan_rows(tmp_run_root, dag)
    assert len(rows) == len(dag.cells)
    assert {r.status for r in rows} <= set(tables.PLAN_STATUSES)
    path = PLAN.write_plan_table(tmp_run_root, rows)
    assert path.exists()
    assert len(tables.read_table(path.parent, PLAN.PLAN_TABLE)) == len(rows)


def test_a_reused_cell_is_terminal_without_ever_running(tmp_run_root, dag):
    reused = next(c for c in dag.cells if c.reuse_of)
    assert PLAN.cell_status(tmp_run_root, reused) == PLAN.STATUS_REUSED
    assert PLAN.STATUS_REUSED in PLAN.TERMINAL_STATUSES


def test_repair_registers_rewrites_only_what_is_missing(tmp_run_root, study_cfg):
    payload = {"entries": [{"source_item_id": "item0"}], "scope": "terminal_only"}
    sha = PLAN.seal_register(tmp_run_root, deps.REGISTER_POOL, payload)
    assert deps.verify_register(tmp_run_root, deps.REGISTER_POOL, sha).met
    report = PLAN.repair_registers(tmp_run_root)
    assert report.completed == 0 and report.skipped == 1        # nothing to repair

    deps.register_path(tmp_run_root, deps.REGISTER_POOL, sha).unlink()
    repaired = PLAN.repair_registers(tmp_run_root)
    assert repaired.completed == 1 and repaired.failed == 0
    assert deps.verify_register(tmp_run_root, deps.REGISTER_POOL, sha).met

    # a *tampered* register is reported, never silently rewritten: the edit is the evidence
    path = deps.register_path(tmp_run_root, deps.REGISTER_POOL, sha)
    path.write_text(path.read_text().replace("item0", "item9"))
    tampered = PLAN.repair_registers(tmp_run_root)
    assert tampered.failed == 1 and "tampered" in tampered.detail


def test_a_second_planner_pass_on_a_locked_run_exits_zero(tmp_run_root, study_cfg,
                                                          inventory):
    """Plan §3.4: "a second instance exits 0".

    ``slurm/nextdyn_planner.sbatch.tmpl`` tells the operator the pass holds a flock on
    ``$RUN/plan/.planner.lock``; a hand-run ``nextdyn plan --planner-pass`` on the login node
    while the Slurm planner is mid-pass would otherwise be two sealers racing over the same
    registers, payloads and content-addressed manifests.
    """
    import fcntl

    manifests = tmp_run_root / "plan" / "manifests"
    lock_path = PLAN.planner_lock_path(tmp_run_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = PLAN.planner_pass(tmp_run_root, study_cfg, inventory=inventory)
        assert report.skipped == 1 and "lock" in report.detail
        assert report.completed == 0
        assert not list(manifests.glob("*.json")) if manifests.exists() else True
        assert not (tmp_run_root / deps.REGISTERS_DIR).exists() or \
            not list((tmp_run_root / deps.REGISTERS_DIR).rglob("*.json"))

    # the lock is released with the file: the next pass does the work
    after = PLAN.planner_pass(tmp_run_root, study_cfg, inventory=inventory)
    assert after.skipped != 1 or "lock" not in after.detail
