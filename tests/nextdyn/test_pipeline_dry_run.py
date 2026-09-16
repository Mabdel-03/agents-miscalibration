"""R3, the keystone: every cell kind on synthetic items, then the whole close-out.

Plan §1.1 R3: *"Every producer has a consumer test in the same change:
``test_pipeline_dry_run.py`` drives every cell kind on 4 synthetic items through the fake
vLLM, materializes, asserts keys, runs ``analyze`` in test mode.  A WP is not done until its
kind runs there."*

The pipeline itself is ``scripts/nextdyn_dry_run.py`` -- the same code the CLI gate runs,
loaded here by path so the gate and the test can never drift apart.  It runs once per session
(it is the only slow test in the group) and the assertions below read its report.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

from agents_scaling.nextdyn import cli, config as C, deps, kinds, tables
from agents_scaling.nextdyn.ids import CellKind

REPO = Path(__file__).resolve().parents[2]
#: The whole offline pipeline should stay well inside the group's five-minute budget.
SOFT_BUDGET_S = 180.0


@pytest.fixture(scope="session")
def dry_run_module():
    spec = importlib.util.spec_from_file_location("nextdyn_dry_run",
                                                  REPO / "scripts" / "nextdyn_dry_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pipeline(dry_run_module, tmp_path_factory):
    """One offline run of the whole pipeline; the report is what every test below reads."""
    base = tmp_path_factory.mktemp("nextdyn_dry")
    started = time.time()
    report = dry_run_module.run(base, quick=False, dispatched=2)
    report["_wall_s"] = time.time() - started
    report["_base"] = base
    return report


# --------------------------------------------------------------------------- R3


def test_the_dry_run_is_green(pipeline):
    assert pipeline["failures"] == [], pipeline["failures"]


def test_every_cell_kind_is_driven_or_explicitly_pending(pipeline, dry_run_module):
    """A kind nobody drives here is a work package that is not done (R3).

    A kind this environment genuinely cannot run -- capture needs torch and the pinned
    tokenizer -- is *named* with its reason, never quietly absent.
    """
    run_root = Path(pipeline["run_root"])
    driven: set[str] = set()
    for directory in (run_root / deps.CELLS_DIR).iterdir():
        if (directory / deps.META_FILE).exists():
            driven.add(directory.name.split(".")[0])
    pending = {CellKind.CAPTURE.value}
    assert pending <= {key.split("/")[0].lower() for key in pipeline["pending"]} | pending
    assert {kind.value for kind in CellKind} - driven == pending, sorted(driven)
    for kind in sorted(pending):
        assert any(kind.upper() in key.upper() for key in pipeline["pending"]), kind
        assert all(note for note in pipeline["pending"].values())


def test_the_published_tables_pass_strict_keys_and_the_referential_rules(pipeline):
    """``tables.check_keys`` and ``tables.assert_referential`` are FATAL at materialize."""
    assert pipeline["keys"] == {"violations": [], "referential": []}
    run_root = Path(pipeline["run_root"])
    published = run_root / "tables"
    assert (published / "MANIFEST.json").exists()
    assert tables.assert_referential(published, strict=True) == []
    decisions = tables.read_table(published, "decisions")
    assert decisions
    # every generation kind that ran wrote decisions, and every row names its model (spec 17.2)
    assert all(row["receiver_model_hash"] for row in decisions)
    assert {row["cell_kind"] for row in decisions} >= {CellKind.EPISODE.value,
                                                       CellKind.DIFFICULTY_BANK.value,
                                                       CellKind.RECEIVER.value,
                                                       CellKind.SELECTION_AUDIT.value}


def test_round_zero_is_a_reference_to_the_bank_draw_not_a_new_sample(pipeline):
    """spec 8.2 / D14: an episode never generates its roots."""
    published = Path(pipeline["run_root"]) / "tables"
    rows = tables.read_table(published, "decisions")
    references = [r for r in rows if r["is_reference"]]
    assert references
    from agents_scaling.nextdyn.ids import BANK_KINDS

    bank_kinds = {kind.value for kind in BANK_KINDS}
    by_uid = {r["generation_uid"]: r for r in rows if r["cell_kind"] in bank_kinds}
    for reference in references:
        assert reference["reuse_of"] is not None
        assert reference["generation_uid"] in by_uid          # the draw's own generation


def test_the_analysis_runs_in_the_test_profile_and_writes_a_claim_ledger(pipeline):
    assert pipeline["analysis"]["ledger"]
    assert pipeline["analysis"]["ledger_rows"] > 1
    ledger = Path(pipeline["run_root"]) / "derived" / "claim_ledger.csv"
    header = ledger.read_text().splitlines()[0]
    assert "claim_id" in header or "id" in header


def test_resume_regenerates_nothing(pipeline):
    """Delete every item file, re-drive: the store serves every request from cache."""
    assert pipeline["resume"]["unchanged"], pipeline["resume"]
    assert pipeline["resume"]["before"] > 0


def test_a_resume_keeps_the_whole_per_request_archive(pipeline):
    """The archive a resumed cell rewrites its item file over (plan §3.7).

    ``runner.generate`` returned early whenever the request store served the sample from cache,
    on the premise that "another cell already recorded its requests row".  For a *resumed* cell
    that premise is false -- the cell that recorded the row is this one, and its item file did
    not survive -- so the resume rewrote the file with no rows at all and the published
    ``requests`` table came out at 9% of the store.  Nothing else notices: the tables still
    pass their keys, they just describe a ninth of the campaign's replicas and latencies.
    """
    resume = pipeline["resume"]
    assert resume["requests_rows_before"] > 0
    assert resume["requests_rows_after"] >= resume["requests_rows_before"], resume
    assert resume["archive_kept"]


def test_every_generated_decision_has_a_request_row_under_its_generation_uid(pipeline):
    """spec 16.1 / plan §3.7: ``resources.parquet`` is built by joining these two tables, and a
    decision whose ``generation_uid`` has no ``requests`` row contributes a silent NULL."""
    published = Path(pipeline["run_root"]) / "tables"
    decisions = tables.read_table(published, "decisions")
    requests = tables.read_table(published, "requests")
    assert requests
    have = {row["generation_uid"] for row in requests}
    orphans = sorted({row["generation_uid"] for row in decisions
                      if not row["is_reference"]} - have)
    assert not orphans, f"{len(orphans)} generated decision(s) with no requests row: {orphans[:3]}"


def test_the_plan_table_carries_one_row_per_planned_cell(pipeline):
    """``release-check``'s "every planned cell is terminal" gate has a producer here.

    The census is written by ``plan.write_plan_table(plan_rows(..., final=True))`` before
    materialize, so ``cell_status(final=True)`` -- the function that classifies an unfilled
    bucket as ``skipped_quota_unfilled`` rather than as a failure -- runs end to end on a run
    root where such buckets genuinely exist.
    """
    census = pipeline["plan_table"]
    assert census["rows"] == pipeline["tables"]["plan"] > 0
    assert census["non_terminal"] == []
    assert set(census["by_status"]) <= set(tables.PLAN_STATUSES)
    assert census["dispatched"] > 0
    rows = tables.read_table(Path(pipeline["run_root"]) / "tables", "plan")
    assert len({row["cell_id"] for row in rows}) == len(rows) == census["rows"]


def test_the_freeze_gate_refuses_a_confirmation_run_that_is_not_frozen(pipeline):
    """``freeze.gate`` is what every cell calls before its first request (contract §2.5)."""
    gate = pipeline["freeze"]
    assert gate["unfrozen_run_passes"] and gate["confirmation_refused"]


def test_the_dependency_gate_holds_with_no_meta_and_no_request(pipeline):
    gate = pipeline["dependency_gate"]
    assert gate["held"]
    assert gate["exit_code"] == 5 and not gate["meta_json"] and gate["deps_unmet_file"]
    assert gate["calls_before"] == gate["calls_after"]


def test_the_dry_run_stays_inside_the_test_groups_budget(pipeline):
    assert pipeline["_wall_s"] < SOFT_BUDGET_S, f"{pipeline['_wall_s']:.0f}s"


# --------------------------------------------------------------------------- the CLI


#: The minimal valid invocation of each verb: a verb with a required choice gets one.
VERB_ARGS = {"fleet": ["status"], "gates": ["coverage"], "drive": ["--lane", "cpu"],
             "analyze": ["--seal"],
             # an empty HF home, so the verb reports "blocked" in milliseconds instead of
             # hashing sixteen gigabytes of safetensors shards; the locking itself is
             # ``tests/nextdyn/test_locks.py``'s subject
             "lock-inputs": ["--hf-home", "{HF}"]}


def test_every_verb_reports_the_five_counters(monkeypatch, tmp_path, capsys):
    """spec 17.5: one summary line, the same five counters, from every verb.

    Including the verbs whose work this environment cannot do: a verb that cannot run reports
    ``blocked``, which is a counter, rather than a traceback -- that is what makes the runbook
    exercisable end to end while a stage is pending.

    Every verb is pointed at a *copy* of the config directory: ``lock-inputs`` writes the lock
    files beside the config it is given, and a test that ran it against the committed ones
    would rewrite the repository as a side effect.
    """
    import shutil

    configs = tmp_path / "configs"
    shutil.copytree(Path(C.DEFAULT_STUDY_CONFIG).parent, configs)
    monkeypatch.setenv("ASYS_RESULTS_ROOT", str(tmp_path))

    for verb in cli.VERBS:
        capsys.readouterr()
        extra = [arg.replace("{HF}", str(tmp_path / "hf")) for arg in VERB_ARGS.get(verb, [])]
        code = cli.main([verb, "--run-root", str(tmp_path / "empty"),
                         "--config", str(configs / "study.yaml"), *extra])
        printed = capsys.readouterr().out.strip().splitlines()[-1]
        assert printed.startswith("completed="), (verb, printed)
        for counter in ("completed", "failed", "skipped", "blocked", "reused"):
            assert f"{counter}=" in printed, (verb, counter)
        assert code in (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_BLOCKED), (verb, code)
        if verb in cli.MONITOR_VERBS:
            assert code == cli.EXIT_OK                        # a monitor reports, never gates


def test_every_runbook_verb_resolves_to_a_module_that_exists():
    """A verb whose module is pending is *reported* as blocked; none of ours is pending."""
    for verb in cli.SPEC_VERBS + cli.OPERATIONAL_VERBS + cli.INTERNAL_VERBS:
        assert callable(cli.resolve_verb(verb)), verb


def test_the_kind_table_and_the_dry_run_agree_on_the_workers(pipeline, dry_run_module):
    """Every kind's declared worker is importable and is the one the dry run drove.

    The second half used to be ``set(json.dumps(report["driven"]))``, which is the set of
    *characters* of the serialized report and is therefore non-empty for ``{}`` -- so the claim
    in the docstring was never checked.
    """
    for kind, spec in kinds.KIND_TABLE.items():
        module, _, attr = spec.worker.partition(":")
        assert attr and module.startswith("agents_scaling.nextdyn")
    driven = set(pipeline["driven"])
    assert driven >= set(dry_run_module.DRIVEN_BLOCKS), sorted(driven)
    assert all(entry["cells"] > 0 for entry in pipeline["driven"].values())
