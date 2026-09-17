"""``allocation grade`` worker: public checks then private grades, contract paths, cache and idempotent resume."""
from __future__ import annotations

import json

import pytest

from agent_allocation.contracts import (CheckStatus, GenerationResult, GradeResult, GradeStatus, PublicCheckResult,
                                        TransportStatus, dump_record, load_record)
from agent_allocation.grading import worker
from agent_allocation.grading.code import profile as profile_mod

from .conftest import make_candidate, reference_program

pytestmark = [pytest.mark.sandbox, pytest.mark.needs_sources]


def _write_run(run_paths, chosen, candidates):
    run_paths.ensure()
    public = run_paths.public_manifest("mbpp_plus")
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text("".join(json.dumps(dump_record(item.public)) + "\n" for item in chosen.values()))
    run_paths.private_manifest("mbpp_plus").write_text(
        "".join(json.dumps(dump_record(item.private)) + "\n" for item in chosen.values()))
    for candidate in candidates:
        result = GenerationResult(node_id=candidate.node_id, attempt_id=candidate.node_id + ":1",
                                  transport_status=TransportStatus.OK, completion_status=None, generation_id=None,
                                  observed_model=None, observed_provider=None, finish_reason="stop",
                                  native_finish_reason=None, visible_content=candidate.visible_content,
                                  reasoning_returned=False, usage=None, cache_status=None, raw_response_path=None,
                                  raw_response_sha256=None, latency_seconds=None, http_status=200)
        path = run_paths.parsed(candidate.node_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"result": dump_record(result), "candidate": dump_record(candidate)}))


def test_grade_pass_writes_checks_and_grades_and_resumes(small_inventory, run_paths):
    chosen, plan = small_inventory
    item = chosen["Mbpp/3"]
    good = reference_program(plan, item.public.item_id)
    candidates = [make_candidate(item.public.item_id, good, node="a1" * 32),
                  make_candidate(item.public.item_id, good, node="b2" * 32),       # identical code: cache hit
                  make_candidate(item.public.item_id, "def is_not_prime(n):\n    return False\n", node="c3" * 32)]
    _write_run(run_paths, chosen, candidates)
    profile = profile_mod.active_profile()
    counts = worker.grade_pass(run_paths, profile, resume=False, workers=1)
    assert counts["public_written"] + counts["public_cached"] == 3
    assert counts["grades_written"] + counts["grades_cached"] == 3
    checks = {c.candidate_id: load_record(PublicCheckResult, json.loads(run_paths.public_check(c.candidate_id)
                                                                        .read_text())) for c in candidates}
    grades = {c.candidate_id: load_record(GradeResult, json.loads(run_paths.grade(c.candidate_id).read_text()))
              for c in candidates}
    assert checks["a1" * 32].status is CheckStatus.OK and checks["a1" * 32].cases_passed == 1
    assert checks["b2" * 32].candidate_id == "b2" * 32 and checks["b2" * 32].cases_passed == 1
    assert grades["a1" * 32].semantic_correct is True and grades["b2" * 32].semantic_correct is True
    assert grades["c3" * 32].status is GradeStatus.GRADED and grades["c3" * 32].semantic_correct is False
    assert oct(run_paths.grades_dir.stat().st_mode & 0o777) == "0o700"
    again = worker.grade_pass(run_paths, profile, resume=True, workers=1)
    assert again["public_skipped"] == 3 and again["grades_skipped"] == 3
    assert again["public_written"] == again["grades_written"] == 0


def test_cmd_grade_refuses_existing_records_without_resume(small_inventory, run_paths):
    chosen, plan = small_inventory
    item = chosen["Mbpp/6"]
    candidate = make_candidate(item.public.item_id, reference_program(plan, item.public.item_id), node="d4" * 32)
    _write_run(run_paths, chosen, [candidate])
    assert worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--workers", "1"]) == 0
    with pytest.raises(SystemExit):
        worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--workers", "1"])
    assert worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--workers", "1", "--resume"]) == 0
