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
    assert counts["public_written"] == 3 and counts["grades_written"] == 3
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
    assert worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--workers", "1"]) == worker.EXIT_BLOCKED
    assert worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--workers", "1", "--resume"]) == 0


# --- v2 behaviours: dedupe/cache, outage redo, supersede, text grader, worker count ------------------------------

from agent_allocation.contracts import (EvalKind, GradeResult as _GradeResult, OutputKind, ParseStatus,  # noqa: E402
                                        PrivateEvalSpec, PublicTask)
from agent_allocation.contracts import Candidate  # noqa: E402

from .conftest import GOOD_PROGRAM, install_synthetic_task  # noqa: E402


def _synthetic_run(run_paths, programs):
    public, private = install_synthetic_task(run_paths)

    class _Item:
        pass

    item = _Item()
    item.public, item.private = public, private
    candidates = [make_candidate(public.item_id, program, node=f"{index:02x}" * 32)
                  for index, program in enumerate(programs, start=1)]
    _write_run(run_paths, {"synthetic": item}, candidates)
    return public, private, candidates


def test_identical_code_runs_once_per_pass_and_is_cached_across_passes(run_paths, monkeypatch):
    from agent_allocation.grading.code import private as private_mod, public as public_mod
    calls = {"public": 0, "private": 0}
    real_public, real_private = public_mod.run_public_checks, private_mod.grade_code

    def counting_public(*args):
        calls["public"] += 1
        return real_public(*args)

    def counting_private(*args):
        calls["private"] += 1
        return real_private(*args)

    monkeypatch.setattr(public_mod, "run_public_checks", counting_public)
    monkeypatch.setattr(private_mod, "grade_code", counting_private)
    _, _, candidates = _synthetic_run(run_paths, [GOOD_PROGRAM, GOOD_PROGRAM, "def f(x):\n    return x\n"])
    profile = profile_mod.active_profile()
    first = worker.grade_pass(run_paths, profile, resume=False, workers=0)
    assert calls == {"public": 2, "private": 2}
    assert first["public_written"] == 3 and first["grades_written"] == 3
    # A new candidate with already-seen code: served from the cache, still its own record.
    extra = make_candidate(candidates[0].item_id, GOOD_PROGRAM, node="ee" * 32)
    _write_parsed(run_paths, extra)
    second = worker.grade_pass(run_paths, profile, resume=True, workers=0)
    assert calls == {"public": 2, "private": 2}
    assert second["public_cached"] == 1 and second["grades_cached"] == 1
    record = load_record(GradeResult, json.loads(run_paths.grade(extra.candidate_id).read_text()))
    assert record.candidate_id == extra.candidate_id and record.semantic_correct is True


def _write_parsed(run_paths, candidate):
    result = GenerationResult(node_id=candidate.node_id, attempt_id=candidate.node_id + ":1",
                              transport_status=TransportStatus.OK, completion_status=None, generation_id=None,
                              observed_model=None, observed_provider=None, finish_reason="stop",
                              native_finish_reason=None, visible_content=candidate.visible_content,
                              reasoning_returned=False, usage=None, cache_status=None, raw_response_path=None,
                              raw_response_sha256=None, latency_seconds=None, http_status=200)
    path = run_paths.parsed(candidate.node_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"result": dump_record(result), "candidate": dump_record(candidate)}))


def test_outage_records_are_never_cached_and_are_redone_on_resume(run_paths, monkeypatch):
    from agent_allocation.grading.code import public as public_mod
    from agent_allocation.sandbox import evalplus_scorer
    from agent_allocation.sandbox.sandbox import STARTUP_FAILED, Sandbox, SandboxError

    class Broken(Sandbox):
        def __init__(self, *args, **kwargs):
            raise SandboxError("injected outage", kind=STARTUP_FAILED, phase="startup")

    _, _, candidates = _synthetic_run(run_paths, [GOOD_PROGRAM])
    profile = profile_mod.active_profile()
    with monkeypatch.context() as patch:
        patch.setattr(public_mod, "Sandbox", Broken)
        patch.setattr(evalplus_scorer, "Sandbox", Broken)
        counts = worker.grade_pass(run_paths, profile, resume=False, workers=0)
    assert counts["infrastructure_errors"] == 1 and counts["evaluator_errors"] == 1
    cid = candidates[0].candidate_id
    assert load_record(PublicCheckResult, json.loads(run_paths.public_check(cid).read_text())).status \
        is CheckStatus.INFRASTRUCTURE_ERROR
    assert not any(run_paths.grade_cache_dir.rglob("*.json"))
    again = worker.grade_pass(run_paths, profile, resume=True, workers=0)
    assert again["public_written"] == 1 and again["grades_written"] == 1
    assert load_record(GradeResult, json.loads(run_paths.grade(cid).read_text())).semantic_correct is True
    assert not (run_paths.run_dir / "grades_superseded").exists()          # same version: redo, not supersede


def test_verdicts_from_another_version_are_superseded_not_overwritten(run_paths):
    from agent_allocation.grading.code.private import GRADER_VERSION
    from agent_allocation.grading.code.public import CHECKER_VERSION
    _, private, candidates = _synthetic_run(run_paths, [GOOD_PROGRAM])
    cid = candidates[0].candidate_id
    old_grade = GradeResult(candidate_id=cid, status=GradeStatus.GRADED, semantic_correct=False,
                            operational_success=True, grader_version="evalplus-native-old", private_eval_hash="x",
                            detail_code="base_wrong_output", environment_digest="old", cpu_seconds=0.0)
    old_check = PublicCheckResult(candidate_id=cid, item_id=private.item_id, code_sha256="old",
                                  status=CheckStatus.OK, cases_total=1, cases_passed=0, case_outcomes=("fail",),
                                  checker_version="public-old", public_inputs_sha256="old", environment_digest="old",
                                  cpu_seconds=0.0)
    worker.write_grade(run_paths, old_grade)
    worker.write_public_check(run_paths, old_check)
    counts = worker.grade_pass(run_paths, profile_mod.active_profile(), resume=True, workers=0)
    assert counts["grades_written"] == 1 and counts["public_written"] == 1
    superseded = load_record(GradeResult, json.loads(run_paths.superseded_grade(cid, "evalplus-native-old").read_text()))
    assert superseded == old_grade
    assert load_record(PublicCheckResult, json.loads(
        run_paths.superseded_public_check(cid, "public-old").read_text())) == old_check
    current = load_record(GradeResult, json.loads(run_paths.grade(cid).read_text()))
    assert current.grader_version == GRADER_VERSION and current.semantic_correct is True
    assert load_record(PublicCheckResult, json.loads(run_paths.public_check(cid).read_text())).checker_version \
        == CHECKER_VERSION


def fake_grade_text(candidate: Candidate, private: PrivateEvalSpec) -> _GradeResult:
    correct = candidate.final_answer == private.gold_answer
    return _GradeResult(candidate_id=candidate.candidate_id, status=GradeStatus.GRADED, semantic_correct=correct,
                        operational_success=True, grader_version="fake-text-v1", private_eval_hash=private.private_hash,
                        detail_code=None, environment_digest=None, cpu_seconds=0.0)


def _text_item(run_paths):
    public = PublicTask(item_id="math500/test_1", benchmark="math500", source_revision="r", prompt="1+1?",
                        output_kind=OutputKind.MATH)
    private = PrivateEvalSpec(item_id="math500/test_1", benchmark="math500", source_revision="r",
                              eval_kind=EvalKind.MATH_VERIFY, gold_answer="2", private_record_ref=None,
                              private_hash="h")
    run_paths.public_manifest("math500").parent.mkdir(parents=True, exist_ok=True)
    run_paths.public_manifest("math500").write_text(json.dumps(dump_record(public)) + "\n")
    run_paths.private_manifest("math500").parent.mkdir(parents=True, exist_ok=True)
    run_paths.private_manifest("math500").write_text(json.dumps(dump_record(private)) + "\n")
    candidate = Candidate(candidate_id="77" * 32, node_id="77" * 32, item_id=public.item_id, visible_content="2",
                          final_answer="2", rationale=None, parse_status=ParseStatus.VALID, finish_reason="stop",
                          output_kind=OutputKind.MATH)
    _write_parsed(run_paths, candidate)
    return candidate


def test_text_candidates_use_the_text_grader_and_block_loudly_without_it(run_paths):
    _synthetic_run(run_paths, [GOOD_PROGRAM])
    text_candidate = _text_item(run_paths)
    profile = profile_mod.active_profile()
    blocked = worker.grade_pass(run_paths, profile, resume=False, workers=0, log=lambda *_: None,
                                text_grader="agent_allocation.grading.text.no_such_module:grade_text")
    assert blocked["blocked_text"] == 1 and blocked["grades_written"] == 1          # code still graded
    assert not run_paths.grade(text_candidate.candidate_id).exists()
    graded = worker.grade_pass(run_paths, profile, resume=True, workers=0,
                               text_grader="tests.unit.code.test_worker:fake_grade_text")
    assert graded["grades_written"] == 1 and graded["blocked_text"] == 0
    record = load_record(GradeResult, json.loads(run_paths.grade(text_candidate.candidate_id).read_text()))
    assert record.semantic_correct is True and record.grader_version == "fake-text-v1"
    assert not run_paths.public_check(text_candidate.candidate_id).exists()        # public checks are code-only


def test_worker_count_is_bounded_by_config_and_affinity(monkeypatch):
    profile = profile_mod.active_profile()
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _pid: set(range(16)))
    assert worker.worker_count(profile, None) == min(profile.workers_max, 14)
    assert worker.worker_count(profile, 3) == 3 and worker.worker_count(profile, 99) == profile.workers_max
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _pid: set(range(4)))
    assert worker.worker_count(profile, None) == 2
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _pid: {0})
    assert worker.worker_count(profile, None) == 1


def test_parsed_record_identity_is_verified(run_paths):
    _, _, candidates = _synthetic_run(run_paths, [GOOD_PROGRAM])
    wrong = run_paths.parsed("ff" * 32)
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_text(run_paths.parsed(candidates[0].node_id).read_text())
    with pytest.raises(ValueError):
        worker.scan_candidates(run_paths)


def test_follow_mode_skips_finished_candidates_on_later_passes(run_paths, monkeypatch):
    _, _, candidates = _synthetic_run(run_paths, [GOOD_PROGRAM])
    finished: set[str] = set()
    profile = profile_mod.active_profile()
    worker.grade_pass(run_paths, profile, resume=False, workers=0, finished=finished)
    assert finished == {candidates[0].candidate_id}
    loads = []
    real_loads = worker.json.loads
    monkeypatch.setattr(worker.json, "loads", lambda text: loads.append(1) or real_loads(text))
    again = worker.grade_pass(run_paths, profile, resume=True, workers=0, finished=finished)
    assert again["candidates"] == 0
