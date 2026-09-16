"""Kinds: the table covers every cell kind, pools follow (kind, model, compute), deadline-bearing
kinds never get a preemptable pool, and every cell number comes from compute.yaml."""

from __future__ import annotations

import dataclasses

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps, kinds
from agents_scaling.nextdyn.errors import ConfigError, ManifestError
from agents_scaling.nextdyn.ids import BANK_KINDS, GENERATION_KINDS, CellKind


def test_kind_table_covers_every_cell_kind_exactly_once():
    assert set(kinds.KIND_TABLE) == set(CellKind)
    assert set(kinds.PRIORITY) == set(CellKind) and len(kinds.PRIORITY) == len(CellKind)
    for kind, spec in kinds.KIND_TABLE.items():
        assert spec.kind is kind
        assert spec.worker.startswith("agents_scaling.nextdyn.") and ":" in spec.worker
        assert spec.doc
        assert spec.needs_endpoint == (kind in GENERATION_KINDS)
        assert set(spec.dep_prefixes) <= set(deps.PREFIXES)
        assert (spec.lane is None) == (kind in GENERATION_KINDS - {CellKind.SELECTION_AUDIT})


def test_priority_puts_banks_first_and_materialize_last():
    assert kinds.priority_rank(CellKind.DIFFICULTY_BANK) < kinds.priority_rank(CellKind.EPISODE)
    assert kinds.priority_rank(CellKind.MATERIALIZE) == len(CellKind) - 1
    assert all(kinds.priority_rank(k) < kinds.priority_rank(CellKind.EPISODE) for k in BANK_KINDS)


def test_cell_kind_compat_resolves_every_spec_name():
    assert set(kinds.CELL_KIND_COMPAT) == {"difficulty_bank", "donor_bank", "natural_root", "revision",
                                           "receiver_plan", "receiver_decision", "selection", "activation_replay"}
    assert kinds.CELL_KIND_COMPAT["revision"] == (CellKind.EPISODE, K.PHASE_REVISION)
    assert kinds.CELL_KIND_COMPAT["receiver_plan"] == (CellKind.RECEIVER, K.PHASE_RECEIVER_PLAN)
    assert kinds.CELL_KIND_COMPAT["receiver_decision"] == (CellKind.RECEIVER, K.PHASE_RECEIVER_DECISION)
    assert kinds.CELL_KIND_COMPAT["selection"] == (CellKind.SELECTION_AUDIT, K.PHASE_AUDIT)
    assert kinds.CELL_KIND_COMPAT["activation_replay"] == (CellKind.CAPTURE, None)
    for name, (kind, phase) in kinds.CELL_KIND_COMPAT.items():
        assert phase is None or phase in K.PHASES
        assert kind in kinds.KIND_TABLE


def test_lane_follows_the_model_for_generation_kinds(study_cfg):
    """The lane is the model's declared scale class, not its served name: the Llama entry
    (served as ``llama8B``) compiles onto the 8B lane so ``plan --block E3C_FAMILY --dry-run``
    can build its cells and block only at request time (plan D11)."""
    contract = study_cfg.contract
    assert kinds.lane_for(CellKind.EPISODE, contract.model("primary")) is kinds.Lane.SMALL
    assert kinds.lane_for(CellKind.RECEIVER, contract.model("scale")) is kinds.Lane.LARGE
    assert kinds.lane_for(CellKind.SELECTION_AUDIT, contract.model("scale")) is kinds.Lane.AUDIT
    assert kinds.lane_for(CellKind.GRADE, None) is kinds.Lane.CPU
    assert kinds.lane_for(CellKind.CAPTURE, None) is kinds.Lane.CAPTURE
    assert kinds.lane_for(CellKind.EPISODE, contract.model("family_transfer")) is kinds.Lane.SMALL
    assert kinds.lane_for(CellKind.EPISODE, contract.model("qwen_nonthinking")) is kinds.Lane.SMALL
    with pytest.raises(ManifestError):
        kinds.lane_for(CellKind.EPISODE, None)
    for bad in ("70B", "audit", "cpu"):
        with pytest.raises(ManifestError, match="not a generation lane"):
            kinds.lane_for(CellKind.EPISODE, dataclasses.replace(contract.model("primary"), lane=bad))


def test_pool_is_a_function_of_kind_model_and_compute(study_cfg, compute_cfg):
    contract = study_cfg.contract
    assert kinds.pool_for(CellKind.EPISODE, contract.model("primary"), compute_cfg) is kinds.Pool.DEADLINE
    assert kinds.pool_for(CellKind.DONOR_BANK, contract.model("primary"), compute_cfg) is kinds.Pool.LOW
    assert kinds.pool_for(CellKind.RECEIVER, contract.model("scale"), compute_cfg) is kinds.Pool.LOW
    assert kinds.pool_for(CellKind.RECEIVER, contract.model("primary"), compute_cfg) is kinds.Pool.LOW
    assert kinds.pool_for(CellKind.SELECTION_AUDIT, contract.model("scale"), compute_cfg) is kinds.Pool.AUDIT
    assert kinds.pool_for(CellKind.GRADE, None, compute_cfg) is None
    assert kinds.pool_for(CellKind.CAPTURE, None, compute_cfg) is None


def test_deadline_bearing_kinds_never_get_a_low_pool(study_cfg, compute_cfg):
    assert kinds.DEADLINE_BEARING == {CellKind.EPISODE, CellKind.CAPTURE}
    assert kinds.DEADLINE_BEARING == {k for k, s in kinds.KIND_TABLE.items() if s.deadline_bearing}
    for kind in kinds.DEADLINE_BEARING:
        for model in study_cfg.contract.models.values():
            pool = kinds.pool_for(kind, model, compute_cfg)
            if pool is not None:
                assert not getattr(compute_cfg.pools, pool.value).preemptable_allowed
    bad = dataclasses.replace(compute_cfg, pools=dataclasses.replace(
        compute_cfg.pools, assignment={**compute_cfg.pools.assignment, "episode": "low"}))
    with pytest.raises(ConfigError, match="deadline-bearing"):
        kinds.pool_for(CellKind.EPISODE, study_cfg.contract.model("primary"), bad)


def test_items_per_cell_scales_with_team_shape_and_thinking_cap(compute_cfg):
    ref = compute_cfg.cells.reference
    assert kinds.items_per_cell(12, reference=ref, n=5, revisions=4, thinking_cap=2048) == 12
    assert kinds.items_per_cell(12, reference=ref, n=5, revisions=8, thinking_cap=2048) == 6
    assert kinds.items_per_cell(12, reference=ref, n=9, revisions=4, thinking_cap=2048) == 6
    assert kinds.items_per_cell(12, reference=ref, n=3, revisions=4, thinking_cap=2048) == 12   # clipped at base
    assert kinds.items_per_cell(12, reference=ref, n=5, revisions=4, thinking_cap=4096) == 6
    assert kinds.items_per_cell(12, reference=ref, n=9, revisions=8, thinking_cap=4096) == 1    # clipped at 1
    with pytest.raises(ManifestError):
        kinds.items_per_cell(0, reference=ref, n=5, revisions=4, thinking_cap=2048)


def test_cell_shape_reads_every_number_from_compute(study_cfg, compute_cfg):
    contract = study_cfg.contract
    team = K.TeamShape(n_agents=5, revision_rounds=4, score_rounds=(0, 1, 2, 3, 4))
    episode = kinds.cell_shape(CellKind.EPISODE, compute=compute_cfg, model=contract.model("primary"),
                               thinking_cap=contract.tokens.thinking_cap, team=team)
    assert episode == kinds.CellShape(items_per_cell=12, parallel_items=2, max_inflight=5, cpus=1, mem="3G",
                                      time="1-00:00:00", partition="mit_normal")
    r8 = kinds.cell_shape(CellKind.EPISODE, compute=compute_cfg, thinking_cap=2048,
                          team=K.TeamShape(n_agents=5, revision_rounds=8, score_rounds=(0, 4, 8)))
    assert r8.items_per_cell == 6
    bank = kinds.cell_shape(CellKind.DONOR_BANK, compute=compute_cfg)
    assert bank.items_per_cell == 16 and bank.max_inflight == 16 and bank.partition == "mit_preemptable"
    grade = kinds.cell_shape(CellKind.GRADE, compute=compute_cfg)
    assert (grade.cpus, grade.mem, grade.time, grade.items_per_cell) == (2, "4G", "4:00:00", 64)
    bcb = kinds.cell_shape(CellKind.GRADE, compute=compute_cfg,
                           grade_resources={"cpus": 4, "mem": "16G", "time": "8:00:00", "items_per_cell": 8})
    assert (bcb.cpus, bcb.mem, bcb.time, bcb.items_per_cell) == (4, "16G", "8:00:00", 8)
    capture = kinds.cell_shape(CellKind.CAPTURE, compute=compute_cfg)
    assert capture.partition == compute_cfg.capture.partition == "ou_bcs_low" and capture.cpus == 8
    assert compute_cfg.cells.kinds["capture"].partition is None      # one declaration (R1)
    with pytest.raises(ManifestError):
        kinds.cell_shape(CellKind.EPISODE, compute=compute_cfg)


def test_every_kind_has_a_cell_block_in_compute(compute_cfg):
    assert set(compute_cfg.cells.kinds) == {k.value for k in CellKind}
    for kind in CellKind:
        spec = kinds.spec_for(kind)
        block = compute_cfg.cells.kinds[kind.value]
        if spec.needs_endpoint:
            assert block.cpus is None and block.partition is None
        else:
            assert block.cpus is not None
            assert (block.partition is None) == (kind is CellKind.CAPTURE)
    assert set(compute_cfg.dispatch.throttle_per_endpoint) == {k.value for k in GENERATION_KINDS}
    assert set(compute_cfg.dispatch.lane_submit_caps) == {lane.value for lane in kinds.Lane}


def test_kindspec_refuses_a_foreign_worker_or_prefix():
    with pytest.raises(ManifestError):
        kinds.KindSpec(kind=CellKind.GRADE, worker="agents_scaling.ard.runner:grade_worker", needs_endpoint=False,
                       lane=kinds.Lane.CPU, deadline_bearing=False, dep_prefixes=(), doc="x")
    with pytest.raises(ManifestError):
        kinds.KindSpec(kind=CellKind.GRADE, worker="agents_scaling.nextdyn.grading:grade_worker", needs_endpoint=False,
                       lane=kinds.Lane.CPU, deadline_bearing=False, dep_prefixes=("state:",), doc="x")
    assert set(kinds.KIND_FIELD_CONSUMERS) == {f.name for f in dataclasses.fields(kinds.KindSpec)}
    assert kinds.is_bank(CellKind.NATURAL_ROOT) and not kinds.is_bank(CellKind.EPISODE)
