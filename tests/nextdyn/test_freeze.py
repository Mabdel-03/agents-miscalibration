"""The frozen manifest: the same class writes it and verifies it, and it is written once.

Plan §3.5 S3, spec 11.2 / 17.4.  ARD36's defect D-05 was a frozen key written where the gate
did not read it -- every cell passed a gate that checked nothing.  The three tests named in the
brief are the structural answer: ``test_freeze_round_trip_gate_passes_on_the_file_it_just_wrote``,
``test_freeze_refuses_null_budget_or_null_lock`` and ``test_frozen_is_write_once``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import freeze as FREEZE
from agents_scaling.nextdyn.errors import LockError, ManifestError

LOCKS = {name: f"{index}" * 64 for index, name in enumerate(FREEZE.REQUIRED_LOCKS)}
BUDGETS = {"campaign_gpu_hours": 670.0, "campaign_cpu_hours": 120.0,
           "storage_limit_bytes": 200_000_000_000}


def authorized(cfg):
    """The config a human authorizes at S3: locks filled, budgets set, analysis locked."""
    locks = dataclasses.replace(cfg.locks, **LOCKS)
    budgets = dataclasses.replace(cfg.budgets, **BUDGETS)
    analysis = dataclasses.replace(cfg.analysis, confirmatory_sample_size_lock={"H1": 300},
                                   analysis_lock_hash="a" * 64)
    return dataclasses.replace(cfg, locks=locks, budgets=budgets, analysis=analysis)


def manifest(cfg, **kwargs) -> FREEZE.FrozenManifest:
    payload = dict(
        schema_version=FREEZE.SCHEMA_VERSION, study_id=cfg.study.id, stage="confirmation",
        contract_sha256=cfg.contract.sha256(), config_sha256=cfg.raw_sha256, locks=dict(LOCKS),
        budgets=dict(BUDGETS), manifests={"BANKS": "ab" * 32},
        sample_size_lock={"H1": 300}, analysis_lock_hash="a" * 64, amendments_sha256="0" * 64,
        cost_profile={}, projection={}, code_version="deadbeef", host="node", frozen_at=1.0)
    payload.update(kwargs)
    return FREEZE.FrozenManifest(**payload)


# --------------------------------------------------------------------------- round trip


def test_freeze_round_trip_gate_passes_on_the_file_it_just_wrote(tmp_run_root, study_cfg):
    """The writer and the reader are one class, so this cannot become a happy-path test."""
    cfg = authorized(study_cfg)
    frozen = manifest(cfg)
    path = frozen.write(tmp_run_root, cfg=cfg)
    assert path.exists() and (tmp_run_root / FREEZE.FROZEN_YAML).exists()
    loaded = FREEZE.load_frozen(tmp_run_root)
    assert loaded == frozen
    assert FREEZE.gate(tmp_run_root, cfg) == frozen
    assert loaded.sha256 == frozen.sha256


def test_the_two_file_names_carry_identical_bytes(tmp_run_root, study_cfg):
    cfg = authorized(study_cfg)
    manifest(cfg).write(tmp_run_root, cfg=cfg)
    assert (tmp_run_root / FREEZE.FROZEN_JSON).read_bytes() == \
        (tmp_run_root / FREEZE.FROZEN_YAML).read_bytes()
    (tmp_run_root / FREEZE.FROZEN_YAML).write_text("{}")
    with pytest.raises(LockError, match="disagree"):
        FREEZE.load_frozen(tmp_run_root)


def test_frozen_is_write_once(tmp_run_root, study_cfg):
    """spec 11.2: a changed design is a new run root, never an edit."""
    cfg = authorized(study_cfg)
    frozen = manifest(cfg)
    frozen.write(tmp_run_root, cfg=cfg)
    frozen.write(tmp_run_root, cfg=cfg)                       # identical bytes: idempotent
    with pytest.raises(LockError, match="write-once"):
        manifest(cfg, stage="pilot").write(tmp_run_root, cfg=cfg)


# --------------------------------------------------------------------------- refusals


def test_freeze_refuses_null_budget_or_null_lock(tmp_run_root, study_cfg):
    """spec 16.4 / 17.4: a null here is a human who has not authorized the run, not a default."""
    cfg = authorized(study_cfg)
    for name in FREEZE.REQUIRED_LOCKS:
        locks = dict(LOCKS)
        locks.pop(name)
        with pytest.raises(LockError, match=name):
            manifest(cfg, locks=locks).write(tmp_run_root, cfg=cfg)
    for name in FREEZE.REQUIRED_BUDGETS:
        budgets = dict(BUDGETS)
        budgets.pop(name)
        with pytest.raises(LockError, match=name):
            manifest(cfg, budgets=budgets).write(tmp_run_root, cfg=cfg)
    with pytest.raises(LockError, match="sample_size_lock"):
        manifest(cfg, sample_size_lock=None).write(tmp_run_root, cfg=cfg)
    with pytest.raises(LockError, match="analysis_lock_hash"):
        manifest(cfg, analysis_lock_hash=None).write(tmp_run_root, cfg=cfg)
    with pytest.raises(LockError, match="block manifest"):
        manifest(cfg, manifests={}).write(tmp_run_root, cfg=cfg)
    assert not (tmp_run_root / FREEZE.FROZEN_JSON).exists()    # and nothing was written


def test_the_gate_notices_a_config_that_changed_since_the_freeze(tmp_run_root, study_cfg):
    cfg = authorized(study_cfg)
    manifest(cfg).write(tmp_run_root, cfg=cfg)
    drifted = dataclasses.replace(cfg, raw_sha256="f" * 64)
    with pytest.raises(LockError, match="has changed since freeze"):
        FREEZE.gate(tmp_run_root, drifted)


def test_a_confirmation_cell_never_runs_unfrozen(tmp_run_root, study_cfg):
    """A pilot may run unfrozen; a confirmation cell may not (spec 11.2)."""
    assert FREEZE.gate(tmp_run_root, study_cfg) is None        # pilot_only: nothing to check
    confirmation = dataclasses.replace(
        study_cfg, study=dataclasses.replace(study_cfg.study, mode=K.MODE_CONFIRMATION))
    with pytest.raises(LockError, match="never runs unfrozen"):
        FREEZE.gate(tmp_run_root, confirmation)


def test_an_unknown_key_in_the_frozen_file_is_refused(tmp_run_root, study_cfg):
    cfg = authorized(study_cfg)
    manifest(cfg).write(tmp_run_root, cfg=cfg)
    path = tmp_run_root / FREEZE.FROZEN_JSON
    payload = json.loads(path.read_text())
    payload["a_key_nobody_declared"] = 1
    path.write_text(json.dumps(payload))
    (tmp_run_root / FREEZE.FROZEN_YAML).write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="unknown key"):
        FREEZE.load_frozen(tmp_run_root)


# --------------------------------------------------------------------------- amendments


def test_the_amendment_ledger_is_snapshotted_at_freeze(tmp_run_root):
    """Freezing without recording which amendments were in force makes the ledger
    unfalsifiable afterwards (plan §3.5 S3)."""
    (tmp_run_root / FREEZE.AMENDMENTS_FILE).write_text('{"id": "A1"}\n{"id": "A3"}\n')
    path, sha = FREEZE.snapshot_amendments(tmp_run_root)
    assert path.read_text().count("\n") == 2 and len(sha) == 64
    again, same = FREEZE.snapshot_amendments(tmp_run_root)
    assert same == sha                                        # idempotent
    (tmp_run_root / FREEZE.AMENDMENTS_FILE).write_text('{"id": "A1"}\n')
    with pytest.raises(LockError, match="taken once"):
        FREEZE.snapshot_amendments(tmp_run_root)


def test_an_absent_ledger_is_an_empty_snapshot_not_a_missing_one(tmp_run_root):
    path, sha = FREEZE.snapshot_amendments(tmp_run_root)
    assert path.exists() and path.read_bytes() == b""
    assert len(sha) == 64


# --------------------------------------------------------------------------- the verb


def test_freeze_refuses_to_emit_anything_when_the_projection_exceeds_the_ceiling(
        tmp_run_root, study_cfg, monkeypatch):
    """plan §3.5 S3: the refusal happens before any manifest is written."""
    from agents_scaling.nextdyn import cells as CE
    from agents_scaling.nextdyn import plan as PLAN
    from agents_scaling.nextdyn.errors import ContractViolation

    families = tuple(study_cfg.cohorts.families)
    items = {f: (f"{f}-0",) for f in families}
    inventory = PLAN.Inventory(
        families=families, items=items, natural_items=items,
        panels={p: {f: items[f] for f in families} for p in CE.PANELS},
        buckets={f: (CE.Bucket(family=f, index=0, source_item_id=items[f][0], pair="primary"),)
                 for f in families})
    cfg = authorized(study_cfg)
    tiny = dataclasses.replace(cfg, budgets=dataclasses.replace(
        cfg.budgets, campaign_gpu_hours=0.0))
    with pytest.raises(ContractViolation, match="refusing to freeze"):
        FREEZE.freeze(tmp_run_root, tiny, inventory=inventory)
    assert not (tmp_run_root / FREEZE.FROZEN_JSON).exists()
    assert not list((tmp_run_root / "plan" / "manifests").glob("*.json"))
