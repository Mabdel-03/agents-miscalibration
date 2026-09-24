"""WP-D: verb registration, refusal paths, the dry-run plan, and the run flow with a stubbed runner (contract §8)."""
from __future__ import annotations

import json
import os
import sys
import types
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation import cli as root_cli
from agent_allocation.config import load_config
from agent_allocation.dynorch import cli as dcli
from agent_allocation.paths import RunPaths

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"
VERBS = ("dynorch-plan", "dynorch-run", "dynorch-tables", "dynorch-gates", "dynorch-reconcile", "dynorch-doctor")


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


@pytest.fixture(autouse=True)
def _isolated_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOCATION_LOCAL_STATE", str(tmp_path / "local"))
    monkeypatch.delenv(dcli.REHEARSAL_ENV, raising=False)


def base_mounted(cfg):
    return (Path(cfg["base_run_root"]) / "manifests" / "panels.json").exists()


# ------------------------------------------------------------------------------------------------------------------
# Registration and the stage table
# ------------------------------------------------------------------------------------------------------------------

def test_verbs_registered_and_resolvable():
    for verb in VERBS:
        target = root_cli.VERBS[verb]
        module, func = target.split(":")
        assert module == "agent_allocation.dynorch.cli" and callable(getattr(dcli, func))
    assert root_cli.main(["no-such-verb"]) == root_cli.EXIT_BLOCKED


def test_stage_specs_match_the_plan(cfg):
    canary = dcli.stage_spec(cfg, "canary")
    assert canary.allowance_usd == Decimal("2") and canary.requires == ()
    pilot = dcli.stage_spec(cfg, "pilot")
    assert pilot.allowance_usd == Decimal("7") and pilot.requires == ("gate1_canary",)
    main = dcli.stage_spec(cfg, "main")
    assert main.allowance_usd is None and main.requires == ("gate2_pilot",)
    assert main.blocks[0].arms == ("H1", "H3", "H0") and main.blocks[0].rungs == (1, 2, 4, 8, 16, 25)
    e1 = dcli.stage_spec(cfg, "ext_e1")
    assert e1.blocks[0].repeat == 2 and e1.blocks[0].rungs == (16, 4, 25, 8, 2, 1) and e1.requires == ("gate4_main",)
    assert dcli.stage_spec(cfg, "ext_e0").blocks[0].arms == ("H1c", "H3c")
    assert dcli.stage_spec(cfg, "ext_e2").blocks[0].arms == ("H1n", "H3n")
    assert dcli.stage_spec(cfg, "ext_e3").blocks[0].arms == ("H2",)
    with pytest.raises(dcli.RunRefused):
        dcli.stage_spec(cfg, "nope")


def test_enumerate_cells_counts(cfg):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    counts = {stage: len(dcli.enumerate_cells(cfg, stage)) for stage in dcli.STAGES}
    assert counts == {"canary": 10, "pilot": 144, "main": 1728, "ext_e0": 384, "ext_e1": 1152, "ext_e2": 384, "ext_e3": 192}
    canary = dcli.enumerate_cells(cfg, "canary")
    assert {c["benchmark"] for c in canary} == {"aime", "gpqa_diamond"}
    assert dcli.cell_summary(canary) == {"H0/4": 2, "H1/16": 2, "H1/4": 2, "H3/16": 2, "H3/4": 2}
    main = dcli.enumerate_cells(cfg, "main", arms=("H1",), rungs=(16,))
    assert len(main) == 96 and {c["arm"] for c in main} == {"H1"} and all(c["repeat"] == 1 for c in main)
    assert dcli.enumerate_cells(cfg, "main", arms=("Hx",)) == []
    e1 = dcli.enumerate_cells(cfg, "ext_e1", rungs=(16,))
    assert len(e1) == 192 and dcli.cell_summary(e1) == {"H1/16/r2": 96, "H3/16/r2": 96}
    pilot = dcli.enumerate_cells(cfg, "pilot")
    assert dcli.cell_summary(pilot)["H0/16"] == 16 and dcli.cell_summary(pilot)["H1/2"] == 32


def test_dry_run_plan_prints_cells(cfg, capsys):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    assert root_cli.main(["dynorch-plan", "--config", str(CFG_PATH), "--stage", "pilot"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["cells"] == 144 and out["allowance_usd"] == "7"
    assert dcli.cmd_dynorch_plan(["--config", str(CFG_PATH), "--stage", "main", "--arms", "H3", "--rungs", "4,16"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["by_cell"] == {"H3/16": 96, "H3/4": 96}
    assert dcli.cmd_dynorch_plan(["--config", str(CFG_PATH), "--stage", "main", "--write"]) == dcli.EXIT_STOP   # needs --run-dir


# ------------------------------------------------------------------------------------------------------------------
# Refusals
# ------------------------------------------------------------------------------------------------------------------

def test_checked_max_usd_ceiling_and_allowances(cfg):
    assert dcli.preset_ceiling(cfg) == Decimal("95")
    value, override = dcli.checked_max_usd(cfg, "90", stage="main", approval=None)
    assert value == Decimal("90") and override is None
    with pytest.raises(dcli.RunRefused):
        dcli.checked_max_usd(cfg, "95.01", stage="main", approval=None)
    value, override = dcli.checked_max_usd(cfg, "96", stage="main", approval="owner: raise to 96")
    assert override["kind"] == "max_usd_override" and override["approval"] == "owner: raise to 96"
    with pytest.raises(dcli.RunRefused):
        dcli.checked_max_usd(cfg, "2.5", stage="canary", approval=None)
    assert dcli.checked_max_usd(cfg, "2", stage="canary", approval=None)[1] is None
    assert dcli.checked_max_usd(cfg, "7", stage="pilot", approval=None)[1] is None
    _, override = dcli.checked_max_usd(cfg, "8", stage="pilot", approval="owner ok")
    assert override["kind"] == "stage_allowance_override"
    with pytest.raises(dcli.RunRefused):
        dcli.checked_max_usd(cfg, "0", stage="main", approval=None)


def run_argv(tmp_path, stage, *extra, max_usd="1"):
    return ["--config", str(CFG_PATH), "--run-dir", str(tmp_path / "run"), "--stage", stage, "--max-usd", max_usd, *extra]


def test_run_refuses_paid_stage_without_allow_paid(cfg, tmp_path, capsys):
    for stage in dcli.STAGES:
        assert dcli.cmd_dynorch_run(run_argv(tmp_path, stage)) == dcli.EXIT_STOP
        assert "--allow-paid" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_run_refuses_ceiling_without_approval(cfg, tmp_path, capsys):
    assert dcli.cmd_dynorch_run(run_argv(tmp_path, "main", "--allow-paid", max_usd="100")) == dcli.EXIT_STOP
    assert "preset ceiling" in capsys.readouterr().err
    assert dcli.cmd_dynorch_run(run_argv(tmp_path, "canary", "--allow-paid", max_usd="3")) == dcli.EXIT_STOP
    assert "allowance" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_run_refuses_the_base_run_root_and_missing_prerequisites(cfg, tmp_path, capsys):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    argv = ["--config", str(CFG_PATH), "--run-dir", cfg["base_run_root"], "--stage", "canary", "--max-usd", "1", "--allow-paid"]
    assert dcli.cmd_dynorch_run(argv) == dcli.EXIT_STOP
    assert "read-only" in capsys.readouterr().err
    assert dcli.cmd_dynorch_run(run_argv(tmp_path, "pilot", "--allow-paid", max_usd="5")) == dcli.EXIT_STOP
    err = capsys.readouterr().err
    assert "gate1_canary" in err and "openrouter_endpoints.json" in err
    assert dcli.cmd_dynorch_run(run_argv(tmp_path, "main", "--allow-paid", max_usd="90")) == dcli.EXIT_STOP
    assert "owner go/no-go" in capsys.readouterr().err


def test_stage_prerequisites(cfg, tmp_path):
    paths = RunPaths.for_run(tmp_path / "run")
    assert dcli.stage_prerequisites(cfg, paths, "canary") == []
    problems = dcli.stage_prerequisites(cfg, paths, "main")
    assert len(problems) == 3
    (paths.run_dir / "gates").mkdir(parents=True)
    (paths.run_dir / "gates" / "gate2_pilot.json").write_text(json.dumps({"decision": "advance"}))
    paths.locks_dir.mkdir(parents=True, exist_ok=True)
    paths.endpoint_lock.write_text("{}")
    assert dcli.stage_prerequisites(cfg, paths, "main") == ["owner go/no-go not recorded (dynorch-gates --gate 2 --owner-go \"<text>\")"]
    from agent_allocation.live import record_decision
    record_decision(paths, kind="owner_go", text="go")
    assert dcli.stage_prerequisites(cfg, paths, "main") == []
    (paths.run_dir / "gates" / "gate2_pilot.json").write_text(json.dumps({"decision": "stop_notify"}))
    assert dcli.stage_prerequisites(cfg, paths, "main") == ["gate gate2_pilot has not recorded advance"]
    assert dcli.stage_prerequisites(cfg, paths, "ext_e0") == ["gate gate4_main has not recorded advance"]


# ------------------------------------------------------------------------------------------------------------------
# The run flow with a stubbed runner (rehearsal transport: no network, no key)
# ------------------------------------------------------------------------------------------------------------------

def stub_module(monkeypatch, name: str, module):
    """Shadow ``agent_allocation.dynorch.<name>`` both in ``sys.modules`` and as the package attribute (the real
    module may already be imported by an earlier test; ``from pkg import name`` prefers the attribute)."""
    import agent_allocation.dynorch as pkg
    monkeypatch.setitem(sys.modules, f"agent_allocation.dynorch.{name}", module)
    monkeypatch.setattr(pkg, name, module, raising=False)


def stub_runner(monkeypatch, *, port=45678, summary=None, fail=False):
    calls = {}

    async def launch_stage(cfg, paths, **kw):
        calls.update(kw)
        if fail:
            raise RuntimeError("runner exploded")
        kw["on_bound"](kw["proxy_host"], port)
        return summary or {"runs": len(kw["cells"]), "upstream_402": False}

    stub_module(monkeypatch, "runner", types.SimpleNamespace(launch_stage=launch_stage))
    return calls


def test_run_canary_flow_writes_proxy_status_locks_and_reconciliation(cfg, tmp_path, monkeypatch, capsys):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    calls = stub_runner(monkeypatch)
    rc = dcli.cmd_dynorch_run(run_argv(tmp_path, "canary", "--allow-paid", "--rehearsal", "--account-auto-reload", "--concurrency", "3", max_usd="1"))
    assert rc == 0, capsys.readouterr().err
    paths = RunPaths.for_run(tmp_path / "run")
    proxy = json.loads((paths.state_dir / "proxy.json").read_text())
    assert proxy == {**proxy, "host": "127.0.0.1", "port": 45678, "stage": "canary"}
    assert proxy["base_url_template"] == "http://127.0.0.1:45678/r/<run_key>/v1"
    status = json.loads(paths.status_json.read_text())
    assert status["state"] == "finished" and status["cells"] == 10 and status["endpoint_rule"] == "keep"
    assert status["reconciliation"]["decision"] == "clean" and status["runner"] == {"runs": 10, "upstream_402": False}
    assert calls["stage"] == "canary" and calls["concurrency"] == 3 and calls["account_auto_reload"] is True
    assert calls["max_usd"] == Decimal("1") and calls["proxy_port"] == 0 and calls["resume"] is False
    assert Decimal(calls["active_ceiling_usd"]) <= Decimal("1")
    assert paths.endpoint_lock.exists() and (paths.locks_dir / "provider_map.json").exists()
    assert (paths.state_dir / "reconcile_canary_before.json").exists() and (paths.state_dir / "reconciliation_canary.json").exists()
    assert (paths.state_dir / "stage_canary_started").exists()
    decisions = [json.loads(l) for l in paths.run_decisions.read_text().splitlines()]
    assert [d["kind"] for d in decisions] == ["dynorch_run_start"] and decisions[0]["rehearsal"] is True
    # nothing under the run tree carries a key-shaped string
    assert dcli.key_hits_in_run_tree(paths.run_dir) == []
    # a second canary invocation reuses the frozen lock (no re-evaluation) and records the ceiling override
    rc = dcli.cmd_dynorch_run(run_argv(tmp_path, "canary", "--allow-paid", "--rehearsal", "--approval", "owner: canary to 3", max_usd="3"))
    assert rc == 0
    decisions = [json.loads(l) for l in paths.run_decisions.read_text().splitlines()]
    assert [d["kind"] for d in decisions] == ["dynorch_run_start", "stage_allowance_override", "dynorch_run_start"]
    assert decisions[1]["approval"] == "owner: canary to 3"


def test_run_blocked_without_runner(cfg, tmp_path, monkeypatch, capsys):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    stub_module(monkeypatch, "runner", types.SimpleNamespace())
    rc = dcli.cmd_dynorch_run(run_argv(tmp_path, "canary", "--allow-paid", "--rehearsal", max_usd="1"))
    assert rc == dcli.EXIT_BLOCKED and "launch_stage" in capsys.readouterr().err
    status = json.loads(RunPaths.for_run(tmp_path / "run").status_json.read_text())
    assert status["state"] == "running"            # the status left behind says the stage never finished


def test_run_stop_notify_when_the_endpoint_rule_stops(cfg, tmp_path, monkeypatch, capsys):
    if not base_mounted(cfg):
        pytest.skip("base run root not mounted")
    from agent_allocation.api import fake as fake_api
    from agent_allocation.api.client import OpenRouterClient
    listing = fake_api.fake_forced_endpoints_listing("qwen/qwen3.8-27b")
    for e in listing["data"]["endpoints"]:
        e["status"] = -2
    script = fake_api.FakeScript(endpoints={"qwen/qwen3.8-27b": listing})
    monkeypatch.setattr(dcli, "make_client", lambda cfg: OpenRouterClient(api_key="rehearsal-no-key", transport=fake_api.fake_openrouter_transport(script)))
    stub_runner(monkeypatch)
    rc = dcli.cmd_dynorch_run(run_argv(tmp_path, "canary", "--allow-paid", "--rehearsal", max_usd="1"))
    assert rc == dcli.EXIT_STOP_NOTIFY
    paths = RunPaths.for_run(tmp_path / "run")
    assert not paths.endpoint_lock.exists() and (paths.locks_dir / "provider_decision.json").exists()
    assert json.loads(paths.status_json.read_text())["state"] == "stop_notify"
    assert not (paths.state_dir / "proxy.json").exists()


def test_proxy_state_and_status_writers(tmp_path):
    paths = RunPaths.for_run(tmp_path / "run")
    target = dcli.write_proxy_state(paths, host="127.0.0.1", port=5, stage="pilot", pid=42)
    doc = json.loads(target.read_text())
    assert doc["pid"] == 42 and doc["port"] == 5 and target == paths.state_dir / "proxy.json"
    leaked = "sk-or-v1-" + "b" * 40
    dcli.write_status(paths, {"note": leaked})
    assert leaked not in paths.status_json.read_text() and "[REDACTED_KEY]" in paths.status_json.read_text()


# ------------------------------------------------------------------------------------------------------------------
# Gates and tables verbs
# ------------------------------------------------------------------------------------------------------------------

def test_tables_and_gates_are_blocked_without_their_inputs(cfg, tmp_path, monkeypatch, capsys):
    stub_module(monkeypatch, "tables", types.SimpleNamespace())
    assert dcli.cmd_dynorch_tables(["--config", str(CFG_PATH), "--run-dir", str(tmp_path / "run")]) == dcli.EXIT_BLOCKED
    assert "build_tables" in capsys.readouterr().err
    assert dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(tmp_path / "run"), "--gate", "1"]) == dcli.EXIT_BLOCKED
    assert "dynorch-tables" in capsys.readouterr().err
    assert dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(tmp_path / "run"), "--gate", "4"]) == dcli.EXIT_BLOCKED


def write_tables(paths: RunPaths, tables):
    runs, calls = tables.frames()
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    runs.to_parquet(paths.tables_dir / "dynorch_runs.parquet", index=False)
    calls.to_parquet(paths.tables_dir / "dynorch_calls.parquet", index=False)


def write_reconciliation(paths: RunPaths, stage, doc):
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    doc = {**doc, "stage": stage}
    (paths.state_dir / f"reconciliation_{stage}.json").write_text(json.dumps(doc, default=str))


def test_gate1_verb_writes_result_and_harness_prompts(cfg, tmp_path, capsys):
    from tests.unit.dynorch.test_endpoint_rule import snapshot, ASSIGNMENT
    from tests.unit.dynorch.test_gates import clean_canary, recon, sample_row
    from agent_allocation.dynorch import endpoint_rule
    paths = RunPaths.for_run(tmp_path / "run")
    write_tables(paths, clean_canary(cfg))
    write_reconciliation(paths, "canary", recon(rows=[sample_row()]))
    endpoint_rule.freeze(paths.run_dir, snapshot(), json.dumps(snapshot()).encode(), cfg, assignment=ASSIGNMENT, fetched_at="t")
    rc = dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(paths.run_dir), "--gate", "1", "--no-home-scan"])
    out = capsys.readouterr().out
    assert rc == dcli.EXIT_STOP_NOTIFY                       # only the skipped home scan fails
    result = json.loads((paths.run_dir / "gates" / "gate1_canary.json").read_text())
    failing = [c["name"] for c in result["criteria"] if not c["passed"]]
    assert failing == ["home_scan_completed"], failing       # "not scanned", not "files were written" (minor 21)
    prompts = json.loads((paths.locks_dir / "harness_prompts.json").read_text())
    assert prompts["S_H1"]["mean"] == 2000.0 and prompts["S_H3"]["n"] == 4
    assert '"gate": "gate1"' in out


def test_gate2_verb_writes_pilot_report_and_records_owner_go(cfg, tmp_path, monkeypatch, capsys):
    from tests.unit.dynorch.test_gates import clean_pilot, recon, sample_row
    paths = RunPaths.for_run(tmp_path / "run")
    write_tables(paths, clean_pilot(cfg))
    write_reconciliation(paths, "pilot", recon(rows=[sample_row()]))
    stub_module(monkeypatch, "forecast",
                types.SimpleNamespace(gate2_refit=lambda cfg, runs, spent_usd: {"core_forecast_usd": "60", "spent_usd": str(spent_usd), "cuts_applied": []}))
    rc = dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(paths.run_dir), "--gate", "2", "--owner-go", "owner: go"])
    assert rc == 0, capsys.readouterr().out
    report = (paths.reports_dir / "PILOT_REPORT.md").read_text()
    assert "Accuracy pooled per benchmark" in report and "| H1 | 16 | macro |" in report
    decisions = [json.loads(l) for l in paths.run_decisions.read_text().splitlines()]
    assert decisions[-1]["kind"] == "owner_go" and decisions[-1]["gate_decision"] == "advance"
    assert dcli.gate_advanced(paths, "gate2_pilot") and dcli.owner_go_recorded(paths)
    # without the forecast module the gate notifies instead of advancing
    stub_module(monkeypatch, "forecast", types.SimpleNamespace())
    rc = dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(paths.run_dir), "--gate", "2"])
    assert rc == dcli.EXIT_STOP_NOTIFY


def test_gate0_verb_with_tests_skipped(cfg, tmp_path, capsys):
    paths = RunPaths.for_run(tmp_path / "run")
    rc = dcli.cmd_dynorch_gates(["--config", str(CFG_PATH), "--run-dir", str(paths.run_dir), "--gate", "0", "--skip-tests"])
    assert rc in (dcli.EXIT_STOP, dcli.EXIT_STOP_NOTIFY)          # no smoke report yet; the unit suite was skipped
    result = json.loads((paths.run_dir / "gates" / "gate0_offline.json").read_text())
    names = {c["name"] for c in result["criteria"]}
    assert names == {"doctor_preconditions", "unit_suite_green", "real_cli_smoke"}


def test_new_files_under_home_scan(tmp_path):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / "old.txt").write_text("x")
    os.utime(home / "old.txt", (1_000_000, 1_000_000))
    since = 2_000_000
    (home / ".claude" / "session.jsonl").write_text("ignored")
    assert dcli.new_files_under_home(home, since) == []
    (home / ".npm").mkdir()
    (home / ".npm" / "cache").write_text("new")
    assert dcli.new_files_under_home(home, since) == [str(home / ".npm" / "cache")]
    assert dcli.new_files_under_home(home, since, budget_seconds=-1) is None


def test_key_hits_in_run_tree_reads_gzip(tmp_path):
    import gzip
    (tmp_path / "raw").mkdir()
    with gzip.open(tmp_path / "raw" / "call_0001.json.gz", "wb") as h:
        h.write(("{\"h\": \"" + "sk-or-v1-" + "c" * 40 + "\"}").encode())
    (tmp_path / "ok.json").write_text("{}")
    assert dcli.key_hits_in_run_tree(tmp_path) == ["raw/call_0001.json.gz"]


# ------------------------------------------------------------------------------------------------------------------
# Gate 4's grader re-run (review major 3)
# ------------------------------------------------------------------------------------------------------------------

def test_regrade_mismatches_regrades_the_rows_the_runner_actually_writes(cfg, tmp_path):
    """Contract §10.9: the runner writes ``parse_status = "valid"`` (the codec VALUE). Comparing against the enum
    NAME ("VALID") skipped every row, so ``mismatches`` was always 0 and Gate 4's ``grader_rerun_identical``
    could never fail."""
    if not base_mounted(cfg):
        pytest.skip("base run root not readable here")
    from agent_allocation.grading.worker import load_tasks

    base = Path(cfg["base_run_root"])
    paths = RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")
    public, _private = load_tasks(RunPaths.for_run(base, local_state_dir=paths.local_state_dir))
    item = sorted(i for i, t in public.items() if t.benchmark == "aime")[0]
    row = {"run_key": "rk0", "item_id": item, "parse_status": dcli.__dict__.get("PARSE_VALID", "valid"),
           "final_message": "<rationale>r</rationale>\n<final>424242</final>", "correct": True}
    assert dcli.regrade_mismatches(paths, pd.DataFrame([row]), base) == 1      # the table says correct; it is not
    assert dcli.regrade_mismatches(paths, pd.DataFrame([{**row, "correct": False}]), base) == 0
    # a row the runner never writes (enum NAME) is not silently counted as agreeing
    assert dcli.regrade_mismatches(paths, pd.DataFrame([{**row, "parse_status": "VALID"}]), base) == 0


# ------------------------------------------------------------------------------------------------------------------
# Extension admission (review minor 23)
# ------------------------------------------------------------------------------------------------------------------

def _advance(paths: RunPaths, name: str) -> None:
    (paths.run_dir / "gates").mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "gates" / f"{name}.json").write_text(json.dumps({"decision": "advance"}))


def test_extensions_need_the_previous_one_sealed_and_the_125x_headroom(cfg, tmp_path):
    """Plan §1/§8: each extension runs only after the core AND the previous extension are sealed, and only while
    ``remaining >= 1.25 x forecast``. Only ``gate4_main`` was checked, so E2 could start before E1 and an
    extension that cannot finish within the ceiling could be launched and die at the ledger mid-way."""
    paths = RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")
    paths.ensure()
    paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
    paths.endpoint_lock.write_text("{}")
    _advance(paths, "gate4_main")
    problems = dcli.stage_prerequisites(cfg, paths, "ext_e2")
    assert any("ext_e0 must be sealed" in p for p in problems) and any("ext_e1 must be sealed" in p for p in problems)
    _advance(paths, "gate4_ext_e0")
    _advance(paths, "gate4_ext_e1")
    assert not [p for p in dcli.stage_prerequisites(cfg, paths, "ext_e2") if "must be sealed" in p]
    # the headroom rule: with the ceiling already spent, the extension is refused
    ok, detail = dcli.extension_headroom(cfg, paths, "ext_e1")
    assert ok and Decimal(detail["forecast_usd"]) > 0
    (paths.state_dir / "ledger_journal.jsonl").parent.mkdir(parents=True, exist_ok=True)
    import agent_allocation.dynorch.reconcile as _rec

    real_totals = _rec.ledger_totals
    try:
        _rec.ledger_totals = lambda p: {"settled_usd": str(cfg["budget"]["credit_ceiling_usd"])}
        ok, detail = dcli.extension_headroom(cfg, paths, "ext_e1")
        assert not ok and Decimal(detail["remaining_usd"]) == 0
        assert any("extension admission" in p for p in dcli.stage_prerequisites(cfg, paths, "ext_e1"))
    finally:
        _rec.ledger_totals = real_totals
