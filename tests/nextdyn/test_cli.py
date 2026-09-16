"""The dispatcher: every runbook verb parses, a missing module is blocked (exit 2) not a
traceback, the summary line is exact, and ``status`` never fails the process."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from agents_scaling.nextdyn import cli
from agents_scaling.nextdyn.errors import NextdynError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "nextdyn"
SUMMARY = re.compile(r"completed=\d+ failed=\d+ skipped=\d+ blocked=\d+ reused=\d+")
#: ``CONFIG`` is replaced by a temporary copy of study.yaml so no verb touches the committed
#: lock files; ``HF`` by an empty cache so lock-inputs fails fast instead of reading the real one.
RUNBOOK = [
    ["inventory", "--config", "CONFIG"],
    ["audit-legacy", "--config", "CONFIG"],
    ["lock-inputs", "--config", "CONFIG", "--hf-home", "HF"],
    ["validate", "--config", "CONFIG", "--smoke"],
    ["pilot", "--config", "CONFIG"],
    ["power", "--pilot-root", "/tmp/x/pilot"],
    ["plan", "--config", "CONFIG", "--dry-run"],
    ["freeze", "--config", "CONFIG"],
    # --dry-run: parsing every verb must not submit anything.  The real submit path is
    # covered by tests/nextdyn/test_dispatch.py with an injected runner.
    ["run", "--manifest", "/tmp/x/frozen_manifest.json", "--dry-run"],
    ["materialize", "--run-root", "/tmp/x"],
    ["capture", "--run-root", "/tmp/x"],
    ["analyze", "--run-root", "/tmp/x", "--locked"],
    ["release-check", "--run-root", "/tmp/x"],
    ["fleet", "up", "--pool", "deadline", "--spec", "nd1-8B:1:ou_bcs_normal:12:00:00"],
    ["gates", "launch", "--measured", "/tmp/x/smoke/measured.json", "--required-replicas", "smoke"],
    ["status", "--run-root", "/tmp/x", "--lane", "cpu"],
]


def test_verbs_are_the_thirteen_spec_verbs_plus_the_operational_three():
    assert cli.SPEC_VERBS == ("inventory", "audit-legacy", "lock-inputs", "validate", "pilot", "power", "plan",
                              "freeze", "run", "materialize", "capture", "analyze", "release-check")
    assert cli.OPERATIONAL_VERBS == ("fleet", "gates", "status") and cli.INTERNAL_VERBS == ("drive",)
    assert set(cli.VERBS) == set(cli.SPEC_VERBS) | set(cli.OPERATIONAL_VERBS) | set(cli.INTERNAL_VERBS)
    for target in cli.VERBS.values():
        module, _, attr = target.partition(":")
        assert module.startswith("agents_scaling.nextdyn.") and attr


@pytest.fixture
def tmp_config(tmp_path):
    for name in ("study.yaml", "compute.yaml"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
    (tmp_path / "hf").mkdir()
    return tmp_path


@pytest.mark.parametrize("argv", RUNBOOK, ids=[a[0] for a in RUNBOOK])
def test_every_runbook_verb_parses(argv, capsys, monkeypatch, tmp_config):
    """A pending verb is blocked (exit 2), a landed verb runs; neither tracebacks."""
    monkeypatch.setenv(cli.RESULTS_ROOT_ENV, str(tmp_config / "results"))
    argv = [str(tmp_config / "study.yaml") if a == "CONFIG" else str(tmp_config / "hf") if a == "HF" else a for a in argv]
    code = cli.main(argv)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert SUMMARY.fullmatch(out), out
    verb = argv[0]
    try:
        cli.resolve_verb(verb)
    except NextdynError:
        assert code == (cli.EXIT_OK if verb in cli.MONITOR_VERBS else cli.EXIT_BLOCKED)
        assert " blocked=1 " in out or out.endswith("blocked=1 reused=0")
    else:
        assert code in (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_BLOCKED)


def test_missing_module_is_blocked_not_a_traceback(capsys, monkeypatch):
    """A verb whose work package has not landed reports ``blocked``, never a traceback.

    Written against a verb that is *made* to point at an absent module rather than against one
    that happened to be absent when this was written: every verb's module has landed now, and a
    test that depended on that not being true would only have kept passing by accident.
    """
    monkeypatch.setitem(cli.VERBS, "plan", "agents_scaling.nextdyn.not_yet_written:main")
    report = cli.run_verb("plan", ["--dry-run"])
    assert report == cli.Report(blocked=1, detail=report.detail)
    assert "agents_scaling.nextdyn.not_yet_written" in report.detail
    assert "work package has not landed" in report.detail
    assert cli.main(["plan", "--dry-run"]) == cli.EXIT_BLOCKED
    assert capsys.readouterr().out.strip() == "completed=0 failed=0 skipped=0 blocked=1 reused=0"
    with pytest.raises(NextdynError):
        cli.resolve_verb("teleport")


def test_summary_line_and_exit_codes():
    assert cli.Report().line() == "completed=0 failed=0 skipped=0 blocked=0 reused=0"
    assert cli.Report(completed=3, reused=2).line() == "completed=3 failed=0 skipped=0 blocked=0 reused=2"
    assert cli.Report(completed=3).exit_code("plan") == cli.EXIT_OK
    assert cli.Report(failed=1, blocked=1).exit_code("plan") == cli.EXIT_FAILED
    assert cli.Report(blocked=1).exit_code("plan") == cli.EXIT_BLOCKED
    assert cli.Report(failed=5, blocked=5).exit_code("status") == cli.EXIT_OK
    total = cli.Report(completed=1, detail="a") + cli.Report(blocked=2, detail="b")
    assert total == cli.Report(completed=1, blocked=2, detail="a; b")


def test_unknown_verb_and_help(capsys):
    assert cli.main(["teleport"]) == cli.EXIT_BLOCKED
    assert cli.main([]) == cli.EXIT_BLOCKED
    assert cli.main(["--help"]) == cli.EXIT_OK
    assert "inventory" in capsys.readouterr().out


def test_run_root_defaults_to_results_root_slash_study_id(monkeypatch, tmp_path):
    monkeypatch.setenv(cli.RESULTS_ROOT_ENV, str(tmp_path))
    args = cli.common_parser("x").parse_args(["--config", str(CONFIG_DIR / "study.yaml")])
    assert cli.resolve_run_root(args) == tmp_path / "task_conditioned_dynamics_v1"
    args = cli.common_parser("x").parse_args(["--run-root", "/tmp/elsewhere"])
    assert cli.resolve_run_root(args) == Path("/tmp/elsewhere")
    monkeypatch.delenv(cli.RESULTS_ROOT_ENV)
    assert str(cli.results_root()).startswith("/orcd/data/") and "/orcd/scratch" not in str(cli.results_root())
    assert args.manifest is None


def test_module_entry_point_imports():
    import agents_scaling.nextdyn.__main__ as entry

    assert entry.main is cli.main
