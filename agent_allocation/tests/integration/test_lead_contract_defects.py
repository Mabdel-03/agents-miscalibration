"""Defects found in lead-owned files (contracts.py, paths.py, config.py). Each is a strict xfail: the suite stays
green, the defect stays visible, and the xfail turns into a failure (XPASS) the moment the lead fixes it, so the
marker gets removed deliberately. Details and proposed fixes: ``docs/requests/08.md``."""
from __future__ import annotations

import copy
from decimal import Decimal

import pytest

from agent_allocation.apikey import redact, redact_headers
from agent_allocation.config import grids_from_config, validate_config
from agent_allocation.contracts import (
    CheckStatus,
    CheckpointView,
    LedgerEventKind,
    PolicyCheckpoint,
    PublicCheckResult,
    SelectionResult,
    dump_record,
    load_record,
)
from agent_allocation.paths import RunPaths

from tests.integration import adversarial


@pytest.mark.xfail(strict=True, reason="D2: contracts.py:165-171 no ledger event separates unresolved from in-flight "
                                       "exposure, so a journal replay cannot rebuild LedgerSnapshot.unresolved")
def test_d2_ledger_journal_can_mark_an_attempt_unresolved():
    names = {k.value for k in LedgerEventKind}
    assert names & {"unresolved", "mark_unresolved", "ambiguous", "timeout"}


@pytest.mark.xfail(strict=True, reason="D4: contracts.py:461-463 pass_fraction is 0.0 when no public case was parsed; "
                                       "unknown must stay unknown (R9) or the code selector silently degrades to random")
def test_d4_pass_fraction_with_zero_public_cases_is_not_zero():
    result = PublicCheckResult("c", "mbpp_plus/1", None, CheckStatus.OK, 0, 0, (), "v1", "h", "env", 0.0)
    assert result.pass_fraction is None


@pytest.mark.xfail(strict=True, reason="D5: paths.py:31-36,52-53 node-local SQLite is keyed by the run-dir basename "
                                       "only; distinct run dirs named alike share one state.sqlite")
def test_d5_distinct_run_dirs_never_share_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOCATION_LOCAL_STATE", raising=False)
    monkeypatch.setattr("agent_allocation.paths.default_local_state_dir", lambda name: tmp_path / "local" / name)
    a = RunPaths.for_run(tmp_path / "agent1" / "run")
    b = RunPaths.for_run(tmp_path / "agent2" / "run")
    assert a.state_db_local != b.state_db_local


@pytest.mark.xfail(strict=True, reason="D6: paths.py:51 sibling run dirs share one write-once locks/ directory, so a "
                                       "mocked or dry run under artifacts/ can freeze the real run's locks")
def test_d6_sibling_run_dirs_do_not_share_write_once_locks(tmp_path):
    real = RunPaths.for_run(tmp_path / "artifacts" / "openrouter_agent_allocation_v1", local_state_dir=tmp_path / "l1")
    mock = RunPaths.for_run(tmp_path / "artifacts" / "mock_e2e", local_state_dir=tmp_path / "l2")
    assert real.endpoint_lock != mock.endpoint_lock


@pytest.mark.xfail(strict=True, reason="D7: config.py:91-122 never reads protocols.cp_team_size / cp_peer_rounds / "
                                       "cp_private_rounds / pooled_synthesis_team_size (R9: a declared knob must be read)")
def test_d7_changing_a_declared_cp_knob_changes_the_grid_or_fails_validation(cfg):
    changed = copy.deepcopy(cfg)
    changed["protocols"]["cp_private_rounds"] = 2
    assert validate_config(changed) or grids_from_config(changed) != grids_from_config(cfg)


@pytest.mark.xfail(strict=True, reason="D7b: config.py:24-56 accepts deep budgets above reasoning.requested_reasoning_max "
                                       "(spec 5.5: validate, never silently clip)")
def test_d7b_deep_budget_above_requested_max_fails_validation(cfg):
    changed = copy.deepcopy(cfg)
    changed["protocols"]["deep_high_multipliers"] = [2, 4, 8]          # 4096 * 8 = 32768 > 24576
    assert validate_config(changed)


@pytest.mark.xfail(strict=True, reason="D7c: config.py:24-56 accepts reservation_multiplier < 1 and an unresolved-"
                                       "exposure pause far above the spec's $2")
def test_d7c_budget_safety_knobs_are_validated(cfg):
    under = copy.deepcopy(cfg)
    under["budget"]["reservation_multiplier"] = 0.5
    loose = copy.deepcopy(cfg)
    loose["budget"]["max_unresolved_spend_usd"] = 50
    assert validate_config(under) and validate_config(loose)


@pytest.mark.xfail(strict=True, reason="D10: contracts.py:475 SelectionResult.status is a bare str; the codec loads "
                                       "any value, so a misspelled status bypasses every status check")
def test_d10_selection_status_is_validated_by_the_codec():
    data = dump_record(SelectionResult("p", None, "plurality", "v1", "h", None, 0.0, "selected"))
    with pytest.raises(ValueError):
        load_record(SelectionResult, {**data, "status": "Selected "})


@pytest.mark.xfail(strict=True, reason="D11: contracts.py:593-605 load_record fills missing fields from dataclass "
                                       "defaults, so an ARCHIVE checkpoint written without 'view' reloads as CURRENT")
def test_d11_load_record_refuses_missing_fields():
    cp = PolicyCheckpoint("main/i/peer/2/1/archive", "i", "peer", 2, 4, 1, ("a",), ("a",), "plurality", "v1", "main",
                          "peer/2/1", CheckpointView.ARCHIVE)
    data = dump_record(cp)
    del data["view"]
    with pytest.raises(ValueError):
        load_record(PolicyCheckpoint, data)


def test_redaction_removes_the_fake_key_everywhere():
    key = adversarial.fake_api_key()
    text = f"key={key} Authorization: Bearer {key} body {{\"k\": \"{key}\"}}"
    assert key not in (redact(text) or "")
    headers = redact_headers({"Authorization": f"Bearer {key}", "X-Echo": key, "X-Title": "ok"})
    assert all(key not in v for v in headers.values()) and headers["x-title"] == "ok"
    assert key not in (redact(key.encode()) or "")


def test_presets_share_protocol_grids_and_differ_in_ceiling(cfg, lean_cfg):
    assert grids_from_config(cfg) == grids_from_config(lean_cfg)
    assert Decimal(str(cfg["budget"]["credit_ceiling_usd"])) == 135 and Decimal(str(lean_cfg["budget"]["credit_ceiling_usd"])) == 90
    raised = copy.deepcopy(cfg)
    raised["budget"]["credit_ceiling_usd"] = 200
    assert validate_config(raised), "agents may not amend the ceiling"
