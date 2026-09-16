"""Measured gates: silence is a failure, the call types cover every phase, the differential.

Plan §3.5, spec 18.6, 19.1.  The three names in the brief are here:
``test_gate_treats_silence_as_failure``, ``test_differential_infra_failure_is_computed_between_
primary_arms`` and ``test_smoke_call_types_cover_every_phase``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import gates as G
from agents_scaling.nextdyn.cells import Block, CellSpec, ConditionRef
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool


class Row:
    """The columns the gate reads off a ``decisions`` row (``telemetry.decision_row`` writes)."""

    def __init__(self, *, phase: str, label: str | None = None, parse_status: str = "OK",
                 infra: str | None = None, continuation: bool = False, boundary: bool = False):
        self.phase = phase
        self.display_label = label
        self.parse_status = parse_status
        self.infrastructure_status = infra
        self.continuation_used = continuation
        self.boundary_injected = boundary
        self.generated_tokens = 100
        self.prompt_tokens = 200
        self.latency_s = 1.0


def measurement(phase: str, **kwargs) -> G.PhaseMeasurement:
    base = dict(phase=phase, n_calls=20, parse_valid=1.0, boundary_rate=0.28,
                continuation_rate=0.28, generated_tokens=1700.0, prompt_tokens=2300.0,
                latency_s=12.0, http_error_rate=0.0)
    base.update(kwargs)
    return G.PhaseMeasurement(**base)


def full_measurement(**kwargs) -> dict[str, G.PhaseMeasurement]:
    return {phase: measurement(phase, **kwargs) for phase in G.GENERATING_PHASES}


# --------------------------------------------------------------------------- the vocabulary


def test_smoke_call_types_cover_every_phase():
    """The ARD36 defect: the gate's table was keyed on a vocabulary the engine never emitted.

    ``episode.py`` wrote ``update``/``leader`` while the table looked up ``revise``/``hub``, so
    every revision fell through to ROOT: PEER_REVISE had no measured row and revisions were
    silently mixed into *root* parse validity.  Here the keys are ``contract.PHASE_*`` -- the
    strings ``telemetry.decision_row`` actually writes -- and this test holds the two together.
    """
    declared = {name: value for name, value in vars(K).items()
                if name.startswith("PHASE_") and isinstance(value, str)}
    assert set(G.PHASES) == set(declared.values()), sorted(set(declared.values()) ^ set(G.PHASES))
    # a root *reference* costs nothing: it is the bank draw, seen from an episode
    assert K.PHASE_ROOT_REF in G.PHASES and K.PHASE_ROOT_REF not in G.GENERATING_PHASES
    assert K.PHASE_BANK in G.GENERATING_PHASES and K.PHASE_REVISION in G.GENERATING_PHASES


def test_the_measurement_is_per_phase_and_counts_what_it_saw():
    rows = [Row(phase=K.PHASE_REVISION, parse_status="NO_JSON"),
            Row(phase=K.PHASE_REVISION, boundary=True, continuation=True),
            Row(phase=K.PHASE_BANK),
            Row(phase=K.PHASE_AUDIT, infra="api_error")]
    measured = G.smoke_measurements("/nonexistent", rows=rows)
    assert set(measured) == {K.PHASE_REVISION, K.PHASE_BANK, K.PHASE_AUDIT}
    assert measured[K.PHASE_REVISION].n_calls == 2
    assert measured[K.PHASE_REVISION].parse_valid == pytest.approx(0.5)
    assert measured[K.PHASE_REVISION].continuation_rate == pytest.approx(0.5)
    assert measured[K.PHASE_AUDIT].http_error_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------- silence


def test_gate_treats_silence_as_failure(study_cfg):
    """"No rows for this phase" is the ARD36 shape, not a pass."""
    empty = G.launch_gate("/nonexistent", {}, cfg=study_cfg)
    assert not empty.ok
    assert all(check.status == G.UNMEASURED for check in empty.checks)
    assert all("nothing measured" in c.observed or "no call" in c.observed
               for c in empty.checks), [c.observed for c in empty.checks]

    partial = {K.PHASE_BANK: measurement(K.PHASE_BANK)}
    report = G.launch_gate("/nonexistent", partial, cfg=study_cfg)
    unmeasured = [c.name for c in report.checks if c.status == G.UNMEASURED]
    assert f"parse_validity[{K.PHASE_REVISION}]" in unmeasured
    assert not report.ok

    full = G.launch_gate("/nonexistent", full_measurement(), cfg=study_cfg)
    assert full.ok, [c.line() for c in full.failed]


def test_the_launch_gate_measures_the_fleet_floor_rather_than_assuming_it(study_cfg):
    below = G.launch_gate("/nonexistent", full_measurement(), cfg=study_cfg,
                          required_replicas=12, live_endpoints=3)
    assert not below.ok and any(c.name == "live_replicas" for c in below.failed)
    at = G.launch_gate("/nonexistent", full_measurement(), cfg=study_cfg,
                       required_replicas=12, live_endpoints=12)
    assert at.ok


def test_a_format_failure_rate_over_the_declared_target_fails(study_cfg):
    bad = full_measurement(parse_valid=0.5)
    report = G.launch_gate("/nonexistent", bad, cfg=study_cfg)
    assert not report.ok
    failed = {c.name for c in report.failed}
    assert f"parse_validity[{K.PHASE_REVISION}]" in failed
    assert all(str(study_cfg.gates.max_format_failure_rate) in c.required
               for c in report.checks if c.name.startswith("parse_validity"))


# --------------------------------------------------------------------------- the pilot gate


def test_differential_infra_failure_is_computed_between_primary_arms(study_cfg):
    """spec 18.6: < 1 pp between the two primary arms, on infra failure AND continuation.

    A *differential* is the thing that can bias a contrast: an arm that fails more often, or
    continues more often, is an arm whose rows are drawn from a different process.
    """
    private, peer = tuple(study_cfg.protocol.protocols)[:G.PRIMARY_ARM_COUNT]
    rows = ([Row(phase=K.PHASE_REVISION, label=private) for _ in range(100)]
            + [Row(phase=K.PHASE_REVISION, label=peer) for _ in range(98)]
            + [Row(phase=K.PHASE_REVISION, label=peer, infra="api_error") for _ in range(2)])
    rates = G.arm_rates(rows)
    assert rates[private].infra_failure == 0.0
    assert rates[peer].infra_failure == pytest.approx(0.02)
    assert G.differential(rates, (private, peer), "infra_failure") == pytest.approx(2.0)

    report = G.pilot_gate("/nonexistent", cfg=study_cfg, rows=rows)
    failed = {c.name for c in report.failed}
    assert "differential_infra_failure_pp" in failed
    assert not report.ok

    # an arm with no rows at all is UNMEASURED, never "no difference"
    one_arm = [Row(phase=K.PHASE_REVISION, label=private)]
    assert G.differential(G.arm_rates(one_arm), (private, peer), "infra_failure") is None
    lonely = G.pilot_gate("/nonexistent", cfg=study_cfg, rows=one_arm)
    assert any(c.status == G.UNMEASURED and c.name == "differential_continuation_pp"
               for c in lonely.checks)


def test_the_pilot_gate_reports_eligibility_per_family_and_the_parities(study_cfg):
    rows = [Row(phase=K.PHASE_REVISION, label=p)
            for p in tuple(study_cfg.protocol.protocols)[:G.PRIMARY_ARM_COUNT] for _ in range(50)]
    minimum = study_cfg.cohorts.pilot_min_paired_items
    eligibility = {family: minimum for family in study_cfg.cohorts.families}
    report = G.pilot_gate("/nonexistent", cfg=study_cfg, rows=rows, eligibility=eligibility,
                          storage_bytes_per_request=50_000.0, adapter_parity=True,
                          capture_parity=True)
    assert report.ok, [c.line() for c in report.failed]

    short = dict(eligibility)
    short[study_cfg.cohorts.families[0]] = minimum - 1
    under = G.pilot_gate("/nonexistent", cfg=study_cfg, rows=rows, eligibility=short,
                         storage_bytes_per_request=50_000.0, adapter_parity=True,
                         capture_parity=True)
    assert not under.ok
    assert f"paired_eligibility[{study_cfg.cohorts.families[0]}]" in {c.name for c in under.failed}
    # an unmeasured parity is a failure, not an omission
    silent = G.pilot_gate("/nonexistent", cfg=study_cfg, rows=rows, eligibility=eligibility)
    assert {"adapter_parity", "capture_parity", "storage_per_request"} <= \
        {c.name for c in silent.failed}


# --------------------------------------------------------------------------- coverage


def test_coverage_gate_fails_the_block_that_is_short(tmp_run_root, study_cfg):
    from agents_scaling.nextdyn import deps

    cells = [CellSpec(cell_id=f"difficulty_bank.MATH.primary.a.x.pilot.r0.s{i}",
                      kind=CellKind.DIFFICULTY_BANK, lane=Lane.SMALL, pool=Pool.LOW,
                      model="primary", family="MATH", tokens_hash="a", condition=None,
                      items=(f"item{i}",), replicate=0, split_role="pilot", block=Block.BANKS)
             for i in range(3)]
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, cells, block=Block.BANKS)
    report = G.coverage_gate(tmp_run_root)
    assert not report.ok
    assert report.checks[0].name == f"coverage[{Block.BANKS}]"
    assert "0 terminal" in report.checks[0].observed

    for cell in cells:
        directory = deps.cell_dir(tmp_run_root, cell.cell_id)
        (directory / "items").mkdir(parents=True)
        (directory / "items" / f"{cell.items[0]}.json").write_text(
            json.dumps({"cell_id": cell.cell_id, "source_item_id": cell.items[0],
                        "status": "complete"}))
        (directory / deps.META_FILE).write_text(json.dumps({"cell_id": cell.cell_id,
                                                            "n_items": 1}))
    assert G.coverage_gate(tmp_run_root).ok


def test_a_run_root_with_no_manifest_is_unmeasured_not_covered(tmp_run_root):
    report = G.coverage_gate(tmp_run_root)
    assert not report.ok and report.checks[0].status == G.UNMEASURED


# --------------------------------------------------------------------------- reports


def test_a_gate_report_says_what_it_required_and_what_it_saw(study_cfg, tmp_run_root):
    report = G.launch_gate(tmp_run_root, full_measurement(), cfg=study_cfg)
    path = G.write_report(tmp_run_root, report)
    payload = json.loads(path.read_text())
    assert payload["gate"] == "launch" and payload["ok"] is True
    assert all({"name", "status", "required", "observed"} <= set(check) for check in payload["checks"])
    line = report.as_report().line()
    assert line.startswith("completed=") and "failed=0" in line


def test_the_smoke_measurement_is_written_where_the_launch_gate_reads_it(tmp_run_root, study_cfg):
    measured = full_measurement()
    path = G.write_measured(tmp_run_root, measured)
    assert path == tmp_run_root / G.SMOKE_DIR / G.MEASURED_FILE
    assert G.load_measured(path) == measured
    assert G.launch_gate(tmp_run_root, G.load_measured(path), cfg=study_cfg).ok


def test_a_draw_that_ran_past_its_thinking_budget_fails_the_launch_gate(study_cfg):
    """The boundary has to be exact, or the continuation silently never fires.

    ``boundary_injected`` is *defined* as ``thinking_tokens == budget``
    (``adapters.split_native``), so a draw that ran past the budget is recorded as an ordinary
    completion: no continuation, tail discarded, and nothing in the rates to show it.  ARD36
    measured the boundary at 1024 and 4096; this study runs at 2048, so the gate checks it.
    """
    clean = full_measurement(max_thinking_tokens=2048, over_budget_draws=0)
    report = G.launch_gate("/tmp/x", clean, cfg=study_cfg)
    check = next(c for c in report.checks if c.name == "thinking_boundary_exact")
    assert check.status == G.PASS and check.observed == "none"
    assert "bank=2048" in check.detail

    overshot = dict(clean)
    overshot["revision"] = measurement("revision", max_thinking_tokens=2100, over_budget_draws=3)
    report = G.launch_gate("/tmp/x", overshot, cfg=study_cfg)
    check = next(c for c in report.checks if c.name == "thinking_boundary_exact")
    assert check.status == G.FAIL and "revision: 3 draws" in check.observed


def test_the_measurement_counts_an_overshoot_against_the_phases_own_budget(tmp_run_root):
    """Per phase, because the receiver budgets are larger than the main ones.

    A single ceiling over all phases would be the largest receiver budget, and a 2048-budget
    revision draw that reached 4096 would pass it -- which is the case the check exists for.
    """
    rows = [Row(phase=K.PHASE_REVISION), Row(phase=K.PHASE_REVISION)]
    rows[0].thinking_tokens, rows[1].thinking_tokens = 2048, 3000
    measured = G.smoke_measurements(tmp_run_root, rows=rows, budget_of=lambda r: 2048)
    assert measured[K.PHASE_REVISION].max_thinking_tokens == 3000
    assert measured[K.PHASE_REVISION].over_budget_draws == 1
    # with no budget supplied nothing is claimed either way -- an unmeasured thing is not a pass
    measured = G.smoke_measurements(tmp_run_root, rows=rows)
    assert measured[K.PHASE_REVISION].over_budget_draws == 0
