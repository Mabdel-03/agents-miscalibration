"""Public checker semantics: EvalPlus-like module globals, pre-imports, isolation between candidates."""
from __future__ import annotations

import json

import pytest

from agent_allocation.contracts import CheckStatus
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.grading.code.public import public_inputs_sha256, run_public_checks
from agent_allocation.grading import worker

from .conftest import install_synthetic_task, make_candidate

pytestmark = pytest.mark.sandbox


def _check(run_paths, program, **task):
    public, _ = install_synthetic_task(run_paths, **task)
    return run_public_checks(make_candidate(public.item_id, program), public)


def test_math_and_sys_are_preimported_for_the_public_assert(run_paths):
    prompt = '"""\nReturn half.\nassert math.isclose(f(3), 1.5, rel_tol=0.001) and sys.maxsize > 0\n"""\n'
    result = _check(run_paths, "def f(x):\n    return x / 2\n", prompt=prompt, base=([[3]], [1.5]),
                    plus=([[5]], [2.5]))
    assert result.status is CheckStatus.OK and result.case_outcomes == ("pass",)


def test_main_block_does_not_run_and_stdin_is_unreadable(run_paths):
    program = ("def f(x):\n    return x + 1\n"
               "if __name__ == '__main__':\n    raise SystemExit('main block must not run')\n")
    assert _check(run_paths, program).case_outcomes == ("pass",)
    reading = "def f(x):\n    return int(input()) + x\n"
    assert _check(run_paths, reading).case_outcomes == ("error",)


def test_module_level_hang_is_a_timeout(run_paths):
    assert _check(run_paths, "import time\ntime.sleep(30)\ndef f(x):\n    return x + 1\n").case_outcomes \
        == ("timeout",)


def test_each_candidate_gets_a_fresh_sandbox(run_paths):
    writer = "open('/tmp/shared.txt', 'w').write('leak')\ndef f(x):\n    return x + 1\n"
    reader = "def f(x):\n    return x + 1 if open('/tmp/shared.txt').read() == 'leak' else -1\n"
    assert _check(run_paths, writer).case_outcomes == ("pass",)
    assert _check(run_paths, reader).case_outcomes == ("error",)          # FileNotFoundError: nothing carried over


def test_public_inputs_hash_is_stable_and_example_bound(run_paths):
    public, _ = install_synthetic_task(run_paths)
    other, _ = install_synthetic_task(run_paths, task_id="Mbpp/901",
                                      prompt='"""\nWrite f.\nassert f(2) == 3\n"""\n')
    assert public_inputs_sha256(public) == public_inputs_sha256(public)
    assert public_inputs_sha256(public) != public_inputs_sha256(other)


def test_environment_digest_tracks_limits():
    import dataclasses
    profile = profile_mod.active_profile()
    assert profile_mod.environment_digest(profile) == profile_mod.environment_digest(profile)
    bigger = dataclasses.replace(profile, memory_bytes=profile.memory_bytes * 2, stack_bytes=0)
    assert bigger.stack_bytes == bigger.memory_bytes
    assert profile_mod.environment_digest(bigger) != profile_mod.environment_digest(profile)
    assert profile.memory_bytes == 512 * 1024 ** 2 and profile.stack_bytes == profile.memory_bytes


def test_profile_reads_every_grading_knob_and_refuses_unsafe_values():
    cfg = json.loads(profile_mod.DEFAULT_CONFIG.read_text())
    profile = profile_mod.profile_from_config(cfg)
    grading = cfg["grading"]
    assert profile.stdout_bytes == grading["stdout_limit_bytes"]
    assert profile.public_case_wall_seconds == grading["public_case_wall_seconds"]
    assert profile.suite_wall_seconds == grading["full_suite_wall_seconds"]
    assert profile.workers_max == grading["offline_workers_max"] and profile.pids_limit == grading["pids_limit"]
    for key, bad in (("untrusted_code_network", True), ("expose_private_tests_to_generation", True),
                     ("official_evalplus_timeout_rules", False)):
        with pytest.raises(ValueError):
            profile_mod.profile_from_config({**cfg, "grading": {**grading, key: bad}})


def test_cmd_grade_follow_stops_on_sentinel_and_reads_the_run_config_lock(run_paths):
    install_synthetic_task(run_paths)
    run_paths.ensure()
    cfg = json.loads(profile_mod.DEFAULT_CONFIG.read_text())
    run_paths.config_lock.parent.mkdir(parents=True, exist_ok=True)
    run_paths.config_lock.write_text(json.dumps({"config": cfg}))
    assert worker._resolve_profile(run_paths, None) == profile_mod.profile_from_config(cfg)
    run_paths.stop_sentinel.parent.mkdir(parents=True, exist_ok=True)
    run_paths.stop_sentinel.write_text("stop")
    assert worker.cmd_grade(["--run-dir", str(run_paths.run_dir), "--follow", "--workers", "1",
                             "--poll-seconds", "0.01"]) == 0
