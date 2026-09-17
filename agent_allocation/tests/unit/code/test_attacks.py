"""Required negative tests: containment of hostile/buggy programs, and that every limit hit is a verdict, not an outage.

Each attack goes through the production entry points (``run_public_checks`` / ``grade_code``) where a verdict is at
stake, and through ``Sandbox.execute`` where the observable is the environment itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_allocation.contracts import CheckStatus, GradeStatus
from agent_allocation.grading.code.private import grade_code
from agent_allocation.grading.code.public import run_public_checks
from agent_allocation.sandbox import sandbox as sandbox_mod
from agent_allocation.sandbox.evalplus_scorer import SIGXCPU_EXIT
from agent_allocation.sandbox.sandbox import PROCESS_EXITED, Sandbox, SandboxLimits

from .conftest import EXPECTED_CANARY, install_synthetic_task, make_candidate, reference_program

pytestmark = pytest.mark.sandbox

LIMITS = SandboxLimits(cpu_seconds=10, wall_seconds=10, memory_bytes=512 * 1024 ** 2, stack_bytes=512 * 1024 ** 2)
GOOD = "def f(x):\n    return x + 1\n"


def _probe(code: str) -> dict:
    with Sandbox({}, limits=LIMITS) as sandbox:
        result = sandbox.execute(code)
    assert result["ok"], result
    return json.loads(result["stdout"].strip().splitlines()[-1])


def _check_and_grade(run_paths, program: str, **task):
    public, private = install_synthetic_task(run_paths, **task)
    candidate = make_candidate(public.item_id, program)
    return run_public_checks(candidate, public), grade_code(candidate, private, run_paths)


# --- network --------------------------------------------------------------------------------------------------

def test_network_is_unreachable():
    observed = _probe(
        "import json, socket, urllib.request\nout = {}\n"
        "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=2); out['tcp'] = 'connected'\n"
        "except OSError as exc:\n    out['tcp'] = exc.errno\n"
        "try:\n    urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=2); out['http'] = 'ok'\n"
        "except Exception as exc:\n    out['http'] = type(exc).__name__\n"
        "print(json.dumps(out))\n")
    assert observed["tcp"] == 1                       # EPERM from the seccomp socket denial
    assert observed["http"] != "ok"


def test_network_dependent_program_is_a_wrong_answer_not_an_outage(run_paths):
    program = "import socket\ndef f(x):\n    socket.create_connection(('1.1.1.1', 53), timeout=1)\n    return x + 1\n"
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert "sandbox_denied=eperm" in grade.detail_code


# --- secrets, home, repository, oracles -------------------------------------------------------------------------

def test_home_key_file_repository_and_oracles_are_invisible(run_paths, small_inventory, tmp_path, monkeypatch):
    secret = tmp_path / "home" / ".config" / "agents_scaling" / "openrouter_allocation_v1.key"
    secret.parent.mkdir(parents=True)
    secret.write_text("sk-or-v1-CANARYSECRET0000")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-CANARYENV00000")
    chosen, _ = small_inventory
    oracle = run_paths.run_dir / chosen["Mbpp/2"].private.private_record_ref / "oracle.json"
    assert oracle.exists()
    paths = {"secret": str(secret), "home": str(Path.home()), "orcd": "/orcd", "oracle": str(oracle),
             "repo": str(Path(__file__).resolve().parents[3]), "real_key": str(Path.home() / ".config"),
             "results": str(run_paths.run_dir)}
    observed = _probe(
        "import json, os\npaths = " + json.dumps(paths) + "\nout = {}\n"
        "for name, path in paths.items():\n"
        "    try:\n        open(path).read(); out[name] = 'READ'\n"
        "    except IsADirectoryError:\n        out[name] = 'DIR_VISIBLE'\n"
        "    except OSError as exc:\n        out[name] = type(exc).__name__\n"
        "    out[name + '_exists'] = os.path.exists(path)\n"
        "out['env'] = sorted(os.environ)\nout['key_env'] = os.environ.get('OPENROUTER_API_KEY')\n"
        "out['root'] = sorted(os.listdir('/'))\nprint(json.dumps(out))\n")
    for name in paths:
        assert observed[name] == "FileNotFoundError", (name, observed[name])
        assert observed[name + "_exists"] is False, name
    assert observed["key_env"] is None
    assert set(observed["env"]) <= {"PATH", "LANG", "LD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                    "PYTHONDONTWRITEBYTECODE", "LC_CTYPE", "PWD"}
    assert "orcd" not in observed["root"] and "home" not in observed["root"]


def test_program_reading_the_oracle_cannot_pass(run_paths, small_inventory):
    chosen, _ = small_inventory
    item = chosen["Mbpp/3"]
    oracle = run_paths.run_dir / item.private.private_record_ref / "oracle.json"
    program = ("import json\ndef is_not_prime(n):\n    data = json.load(open(" + repr(str(oracle)) + "))\n"
               "    return data['batches']\n")
    grade = grade_code(make_candidate(item.public.item_id, program), item.private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_expected_outputs_never_enter_the_sandbox(run_paths, monkeypatch):
    public, private = install_synthetic_task(run_paths, base=([["a"]], [EXPECTED_CANARY + "-base"]),
                                             plus=([["b"]], [EXPECTED_CANARY + "-plus"]),
                                             prompt='"""\nReturn it.\nassert f("a") is not None\n"""\n')
    sent: list[str] = []
    original = Sandbox._send

    def recording_send(self, obj, phase):
        sent.append(json.dumps(obj))
        return original(self, obj, phase)

    monkeypatch.setattr(Sandbox, "_send", recording_send)
    program = "def f(x):\n    return 'guess'\n"
    run_public_checks(make_candidate(public.item_id, program), public)
    grade = grade_code(make_candidate(public.item_id, program), private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert sent and not any(EXPECTED_CANARY in payload for payload in sent)
    assert not any("private_only" in payload or "oracle" in payload for payload in sent)


# --- processes, threads, CPU, memory, output, recursion ---------------------------------------------------------

def test_fork_bomb_and_subprocess_are_denied(run_paths):
    observed = _probe(
        "import json, os, subprocess\nout = {}\n"
        "try:\n    os.fork(); out['fork'] = 'forked'\nexcept OSError as exc:\n    out['fork'] = exc.errno\n"
        "try:\n    subprocess.run(['/usr/bin/true']); out['subprocess'] = 'ran'\n"
        "except OSError as exc:\n    out['subprocess'] = exc.errno\nprint(json.dumps(out))\n")
    assert observed == {"fork": 1, "subprocess": 1}
    check, grade = _check_and_grade(run_paths, "import os\nwhile True:\n    os.fork()\n" + GOOD)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_thread_creation_is_denied_and_counted_as_a_documented_limitation(run_paths):
    program = ("import threading\ndef f(x):\n    out = []\n"
               "    t = threading.Thread(target=lambda: out.append(x + 1))\n    t.start()\n    t.join()\n"
               "    return out[0]\n")
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert grade.detail_code.endswith(";sandbox_denied=thread")


def test_infinite_loop_is_a_timeout_verdict_not_an_outage(run_paths):
    check, grade = _check_and_grade(run_paths, "def f(x):\n    while True:\n        pass\n")
    assert check.status is CheckStatus.OK and check.case_outcomes == ("timeout",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert grade.detail_code == "base_input_timeout"


def test_cpu_limit_kill_is_classified_as_a_limit():
    with pytest.raises(sandbox_mod.SandboxError) as info:
        with Sandbox({}, limits=SandboxLimits(cpu_seconds=1, wall_seconds=20)) as sandbox:
            sandbox.execute("sum(range(10**15))")                     # C loop: SIGALRM cannot interrupt it
    assert info.value.kind == PROCESS_EXITED and info.value.returncode == SIGXCPU_EXIT
    from agent_allocation.sandbox.evalplus_scorer import classify_sandbox_error
    assert classify_sandbox_error(info.value) == ("cpu_limit", False)


def test_memory_bomb_is_contained_and_graded_wrong(run_paths):
    program = "def f(x):\n    hog = bytearray(8 * 1024 ** 3)\n    return x + 1\n"
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_output_flood_is_truncated_and_does_not_change_the_verdict(run_paths):
    with Sandbox({}, limits=LIMITS) as sandbox:
        result = sandbox.execute("import sys\nprint('x' * 10_000_000)\nsys.stderr.write('y' * 10_000_000)\n")
    assert result["ok"] and result["stdout_truncated"] is True
    assert len(result["stdout"].encode()) <= LIMITS.output_bytes
    program = "print('z' * 5_000_000)\ndef f(x):\n    print('w' * 5_000_000)\n    return x + 1\n"
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("pass",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is True


def test_deep_recursion_is_contained(run_paths):
    check, grade = _check_and_grade(run_paths, "def f(x):\n    return f(x) + 1\n")        # RecursionError
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_explicit_stack_limit_is_enforced():
    deep_c_recursion = ("import sys, json\nsys.setrecursionlimit(10**8)\nnest = []\n"
                        "for _ in range(2_000_000):\n    nest = [nest]\njson.dumps(nest)\n")
    small_stack = SandboxLimits(cpu_seconds=20, wall_seconds=20, memory_bytes=512 * 1024 ** 2,
                                stack_bytes=8 * 1024 ** 2)
    with pytest.raises(sandbox_mod.SandboxError) as info:
        with Sandbox({}, limits=small_stack) as sandbox:
            sandbox.execute(deep_c_recursion)
    assert info.value.kind == PROCESS_EXITED and info.value.returncode == 128 + 11      # SIGSEGV at the limit


def test_reproduced_crash_is_a_verdict_after_canary_and_retry(run_paths, monkeypatch):
    from agent_allocation.grading.code import private as private_mod
    from agent_allocation.grading.code import public as public_mod
    canaries = []
    real_private_canary, real_public_canary = private_mod._canary_ok, public_mod.canary_ok
    monkeypatch.setattr(private_mod, "_canary_ok", lambda p: canaries.append("private") or real_private_canary(p))
    monkeypatch.setattr(public_mod, "canary_ok", lambda p: canaries.append("public") or real_public_canary(p))
    program = "import ctypes\ndef f(x):\n    ctypes.string_at(0)\n    return x + 1\n"     # SIGSEGV every time
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert grade.detail_code == "base_candidate_crash"
    assert canaries == ["public", "private"]


# --- selection-relevant mutants -----------------------------------------------------------------------------------

def test_hardcoded_public_case_passes_selection_but_fails_private(small_inventory, run_paths):
    chosen, _ = small_inventory
    item = chosen["Mbpp/2"]          # assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
    candidate = make_candidate(item.public.item_id, "def similar_elements(a, b):\n    return (4, 5)\n")
    check = run_public_checks(candidate, item.public)
    grade = grade_code(candidate, item.private, run_paths)
    assert check.status is CheckStatus.OK and check.pass_fraction == 1.0
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_constant_output_mutant_fails_private(small_inventory, run_paths, ):
    chosen, plan = small_inventory
    item = chosen["Mbpp/6"]
    grade = grade_code(make_candidate(item.public.item_id, "def differ_At_One_Bit_Pos(a, b):\n    return True\n"),
                       item.private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    reference = grade_code(make_candidate(item.public.item_id, reference_program(plan, item.public.item_id)),
                           item.private, run_paths)
    assert reference.semantic_correct is True


def test_unsupported_dependency_is_a_failure_not_silently_dropped(run_paths):
    program = "import definitely_not_installed_module_xyz\n" + GOOD
    check, grade = _check_and_grade(run_paths, program)
    assert check.status is CheckStatus.OK and check.case_outcomes == ("error",)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False
    assert grade.detail_code == "base_compile_or_initialization_failure"
