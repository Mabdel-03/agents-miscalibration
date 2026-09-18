"""The independent recompute hook against a fully known synthetic run, adversarial tampering, and agent 07's audit."""
from __future__ import annotations

import gzip
import json
from decimal import Decimal

import pytest

from agent_allocation.contracts import GradeResult, GradeStatus, dump_record, load_record

from tests.integration import independent_recompute as ir
from tests.integration import synthetic_run as syn
from tests.integration.conftest import optional_module


@pytest.fixture()
def synthetic(run_paths, cfg):
    summary = syn.write_synthetic_run(run_paths, cfg)
    return run_paths, summary


def _checks(payload):
    return {c["name"]: c for c in payload["checks"]}


def test_contrasts_match_hand_computation_and_all_checks_pass(synthetic):
    paths, _ = synthetic
    payload = ir.independent_recompute(paths.run_dir)
    assert payload["contrasts"][ir.PRIMARY_I] == pytest.approx(syn.EXPECTED_PRIMARY_I, abs=1e-12)
    assert payload["contrasts"][ir.PRIMARY_II] == pytest.approx(syn.EXPECTED_PRIMARY_II, abs=1e-12)
    details = payload["details"]["contrasts"]
    assert details[ir.PRIMARY_I]["exact"] == "5/12" and details[ir.PRIMARY_II]["exact"] == "5/6"
    assert details[ir.PRIMARY_I]["per_benchmark"]["hmmt"]["estimate"] == pytest.approx(-1 / 3)   # items weigh equally
    failing = {n: c["detail"] for n, c in _checks(payload).items() if c["passed"] is not True}
    assert not failing, failing


def test_v1_flat_selection_layout_is_read_too(run_paths, cfg):
    syn.write_synthetic_run(run_paths, cfg, selection_layout="v1")
    assert ir.independent_recompute(run_paths.run_dir)["contrasts"][ir.PRIMARY_I] == pytest.approx(5 / 12)


def test_recompute_item_traces_one_item_per_benchmark(synthetic):
    paths, summary = synthetic
    for item in ("aime/adv/1", "gpqa_diamond/adv/1", "hmmt/adv/1", "omni_math_hard/adv/1"):
        trace = ir.recompute_item(paths.run_dir, item)
        main = trace["partitions"]["main"]
        assert main["nodes"] == main["parsed"] == main["raw_verified"] == main["graded"] == 138
        assert main["raw_problems"] == []
        assert main["kind_counts"] == {"root": 25, "private": 40, "peer": 56, "deep": 4, "communicate_then_private": 12,
                                       "pooled": 1}
        cps = main["checkpoints"]
        assert cps["peer/5/4/current"]["nominal_generations"] == 25
        assert cps["cp/3/current"]["nominal_generations"] == 20 and cps["pooled/4/current"]["nominal_generations"] == 5
        assert all(c["required_matches_closure"] for c in cps.values())
        heads = summary["items"][item]["heads"]
        for name, bit in heads.items():
            assert cps[f"{name}/current"]["delivered_correct"] == bit, (item, name)
        assert cps["sample/25/current"]["correct_bounds"] == [24, 24]                   # root 24 is invalid, not wrong
        assert Decimal(cps["peer/5/4/current"]["cost_usd"]) > Decimal(cps["peer/2/1/current"]["cost_usd"])


def test_tampered_selection_bytes_break_the_seal_and_block_the_join(synthetic):
    paths, summary = synthetic
    path = summary["selection_file"]
    text = path.read_text()
    path.write_text(text.replace('"synthetic_first"', '"synthetic_firsT"', 1))
    payload = ir.independent_recompute(paths.run_dir)
    assert _checks(payload)["selection_seals"]["passed"] is False
    assert payload["contrasts"][ir.PRIMARY_I] is None, "unsealed selections must never be joined to grades"


def test_evaluator_outage_makes_the_contrast_unknown_with_bounds(synthetic):
    paths, _ = synthetic
    run = ir.load(paths.run_dir)
    cp = next(c for c in run.policies["main"] if c.item_id == "omni_math_hard/adv/1" and c.name == "peer/4/4"
              and c.view.value == "current")
    head = cp.candidate_ids[0]
    grade = load_record(GradeResult, json.loads(paths.grade(head).read_text()))
    outage = GradeResult(head, GradeStatus.EVALUATOR_ERROR, None, None, grade.grader_version, grade.private_eval_hash,
                         detail_code="timeout")
    paths.grade(head).write_text(json.dumps(dump_record(outage)))
    details = ir.recompute_primary_contrasts(paths.run_dir)[ir.PRIMARY_I]
    assert details["estimate"] is None and details["n_unknown_items"] == 1
    # omni's D_i = Y(peer) - Y(private_team) = 1 - 0 becomes [0 - 0, 1 - 0] = [0, 1]: macro bounds (5/12 - 1/4, 5/12)
    assert details["lower"] == pytest.approx(5 / 12 - 1 / 4) and details["upper"] == pytest.approx(5 / 12)


def test_parsed_content_that_differs_from_raw_is_flagged(synthetic):
    paths, _ = synthetic
    run = ir.load(paths.run_dir)
    victim = next(iter(run.parsed))
    data = json.loads(paths.parsed(victim).read_text())
    data["candidate"]["visible_content"] = "<final>edited after the fact</final>"
    paths.parsed(victim).write_text(json.dumps(data))
    assert _checks(ir.independent_recompute(paths.run_dir))["raw_matches_parsed"]["passed"] is False


def test_missing_selector_rows_and_leaked_keys_are_flagged(synthetic):
    from tests.integration import adversarial
    paths, summary = synthetic
    path = summary["selection_file"]
    lines = path.read_bytes().splitlines(keepends=True)
    body = b"".join(lines[1:])
    path.write_bytes(body)
    seal_path = path.with_name(path.name.replace(".jsonl", ".seal.json"))
    seal = json.loads(seal_path.read_text())
    from hashlib import sha256
    seal.update(sha256=sha256(body).hexdigest(), count=len(lines) - 1)
    seal_path.write_text(json.dumps(seal))
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    (paths.logs_dir / "dispatcher.log").write_text(f"Authorization: Bearer {adversarial.fake_api_key()}\n")
    checks = _checks(ir.independent_recompute(paths.run_dir))
    assert checks["selector_rows_present"]["passed"] is False
    assert checks["no_credentials_in_artifacts"]["passed"] is False


def test_ledger_that_never_settles_a_success_is_flagged(synthetic):
    paths, _ = synthetic
    lines = paths.ledger_journal.read_text().splitlines()
    kept = [line for line in lines if not (json.loads(line)["kind"] == "settle" and lines.index(line) < 30)]
    paths.ledger_journal.write_text("\n".join(kept) + "\n")
    assert _checks(ir.independent_recompute(paths.run_dir))["ledger_replay"]["passed"] is False


def test_invalid_output_graded_as_wrong_is_flagged(synthetic):
    paths, _ = synthetic
    run = ir.load(paths.run_dir)
    invalid = next(nid for nid, (_, cand) in run.parsed.items() if cand.parse_status.value != "valid")
    grade = load_record(GradeResult, json.loads(paths.grade(invalid).read_text()))
    paths.grade(invalid).write_text(json.dumps(dump_record(GradeResult(invalid, GradeStatus.GRADED, False, True,
                                                                       grade.grader_version, grade.private_eval_hash))))
    assert _checks(ir.independent_recompute(paths.run_dir))["invalid_not_graded"]["passed"] is False


def test_agent07_audit_agrees_with_the_independent_recompute(synthetic, cfg):
    """Spec 08 brief: recompute two headline contrasts from raw per-item records and compare with the analysis module."""
    audit = optional_module("agent_allocation.reporting.audit")
    paths, _ = synthetic
    results, _report = audit.audit_run(paths.run_dir)
    by_name = {r.name: r for r in results}
    hook = by_name.get("independent_recompute_contrasts")
    assert hook is not None, sorted(by_name)
    assert hook.status == "pass", (hook.status, getattr(hook, "detail", None), getattr(hook, "evidence", None))
