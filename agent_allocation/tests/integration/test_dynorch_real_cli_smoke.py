"""Gate 0 real-CLI smoke ($0): OpenClaude 0.13.0 and Codex 0.156.1 driven by ``dynorch-run --rehearsal`` through the
REAL proxy stack over the fake upstream (plan §9 Gate 0; contract §10.2, §10.3, §10.18).

Each case runs the canary's four cells of one harness arm (2 dev items × rungs {4, 16}) exactly as a paid stage
would, minus the network: the endpoint rule is frozen from the fake listing, the proxy is bound, the real CLI is
launched by the real adapter with the scrubbed environment, the fake upstream answers every call with a natural
``<final>`` reply, tables are built and the stage reconciled. The checks are the Gate 0 items that need no
scripted upstream (tool surface, ``messages[0]`` / ``instructions``, classification, no ``budget_usd`` text,
nothing written under ``$HOME``, no key material in the run tree). Scripted-spawn, 402 and long-delay items of the
Gate 0 list are covered by the fake-harness e2e; a scripted spawn through the real CLIs needs the fake upstream's
script to be reachable from the verb, which ``--rehearsal`` does not offer (open issue).

Skipped when a binary is absent. Marked ``slow`` (each case takes ~10-20 s plus the version probes).
Set ``DYNORCH_GATE0_SMOKE_OUT=<run_root>/gates/gate0_smoke.json`` to have the report written there
(``{passed, failures, cases}``; ``dynorch-gates --gate 0`` reads it). The scripted-spawn, 402, cut-stream,
timeout, rung-25 and long-delay items of the Gate 0 list are covered by ``test_dynorch_openclaude_smoke.py`` and
``test_dynorch_codex_smoke.py`` (WP-E), which merge their cases into the same report.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.config import load_config
from agent_allocation.dynorch import cli as dcli
from agent_allocation.dynorch.journal import RunStore
from agent_allocation.paths import RunPaths

PROJECT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT / "configs" / "dynorch_v1.json"
CFG = load_config(CFG_PATH)
OPENCLAUDE_PREFIX = str(CFG["dynorch"]["openclaude_system_prefix"])
SYSTEM_TEXT = (PROJECT / CFG["dynorch"]["prompt_files"]["system"]).read_text("utf-8")
H1_TOOLS = ("Agent",)
H3_TOOLS = ("update_plan", "spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent")
HOME_EXCLUDES = (".claude", ".cursor-server", ".vscode-server", ".cache/uv")
_REPORT: dict[str, Any] = {"passed": True, "failures": [], "cases": {}}


def _binary(arm: str) -> str | None:
    if arm == "H1":
        return shutil.which("openclaude")
    codex = Path(CFG["dynorch"]["codex_prefix"]) / "bin" / "codex"
    return str(codex) if codex.is_file() else None


def _new_under_home(since: float, *, max_depth: int = 3, budget_seconds: float = 120.0) -> list[str] | None:
    """Files under ``$HOME`` (depth <= ``max_depth``, the interactive tools' own state pruned) modified after
    ``since``; ``None`` when the walk exceeds its budget (reported as "not scanned", never as clean). The full
    ``/home`` tree holds ~860k files, so the walk is bounded like ``cli.new_files_under_home``."""
    home = Path.home()
    started = time.monotonic()
    hits: list[str] = []
    for dirpath, dirs, files in os.walk(home):
        rel_dir = os.path.relpath(dirpath, home)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        kept = []
        for d in dirs:
            rel = d if rel_dir == "." else f"{rel_dir}/{d}"
            if any(rel == e or e.startswith(rel + "/") and rel_dir == "." and "/" in e and e.split("/")[0] == d
                   for e in HOME_EXCLUDES):
                continue
            kept.append(d)
        dirs[:] = [] if depth >= max_depth else kept
        for name in files:
            path = Path(dirpath) / name
            try:
                if path.stat().st_mtime > since:
                    hits.append(str(path.relative_to(home)))
            except OSError:
                continue
        if time.monotonic() - started > budget_seconds:
            return None
    return hits


def _run_arm(arm: str, tmp_path: Path, monkeypatch) -> tuple[RunPaths, dict[str, Any]]:
    monkeypatch.setenv("ALLOCATION_LOCAL_STATE", str(tmp_path / "local"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.delenv(dcli.REHEARSAL_ENV, raising=False)
    doc = json.loads(CFG_PATH.read_text())
    doc["dynorch"]["scratch_root"] = str(tmp_path / "scratch")
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(doc, indent=1))
    rc = dcli.cmd_dynorch_run(["--config", str(config_path), "--run-dir", str(tmp_path / "run"), "--stage", "canary",
                               "--max-usd", "1", "--allow-paid", "--rehearsal", "--arms", arm, "--concurrency", "2"])
    assert rc == 0, f"dynorch-run exited {rc}"
    paths = RunPaths.for_run(tmp_path / "run")
    return paths, json.loads(paths.status_json.read_text())


def _journaled_requests(paths: RunPaths) -> list[dict[str, Any]]:
    out = []
    for raw in sorted((paths.run_dir / "raw").rglob("call_*.json.gz")):
        doc = json.loads(gzip.open(raw).read())
        out.append(doc)
    return out


def _record(arm: str, checks: dict[str, bool]) -> None:
    """Merge this case into ``$DYNORCH_GATE0_SMOKE_OUT`` through the shared writer (the OpenClaude / Codex smoke
    modules write the same file; ``passed`` / ``failures`` are re-derived over every case recorded so far)."""
    from tests.integration.dynorch_smoke_support import record_case

    _REPORT["cases"][arm] = checks
    for name, ok in checks.items():
        if not ok:
            _REPORT["passed"] = False
            _REPORT["failures"].append(f"{arm}: {name}")
    record_case(f"canary_rehearsal/{arm}", checks)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("arm", ["H1", "H3"])
def test_real_cli_canary_rehearsal_through_the_real_proxy(arm: str, tmp_path: Path, monkeypatch):
    if _binary(arm) is None:
        pytest.skip(f"{arm}'s CLI is not installed here")
    marker = time.time() - 1.0
    paths, status = _run_arm(arm, tmp_path, monkeypatch)
    records = RunStore(paths.run_dir).scan()
    docs = _journaled_requests(paths)
    requests = [d["harness_request"] for d in docs]
    tools = {frozenset(t["name"] for t in r["tools"]) for r in requests}       # order is the CLI's own
    system_texts = [r["messages"][0]["content"] for r in requests]
    checks: dict[str, bool] = {
        "four_runs_done": status["runner"]["done"] == 4 and status["runner"]["transport_failed"] == 0,
        "reconciliation_clean": status["reconciliation"]["decision"] == "clean",
        "one_root_call_per_run_no_other": all(r.calls == 1 and r.calls_by_class.get("other", 0) == 0
                                              and r.calls_by_class.get("root", 0) == 1 for r in records.values()),
        "delivered_answer_valid_and_cross_checked": all(r.parse_status == "valid" and r.cli_result_mismatch is False
                                                        and r.final_source == "proxy_root" for r in records.values()),
        "harness_exit_0": all(r.harness_exit_code == 0 for r in records.values()),
        "no_budget_usd_text": not any("budget_usd" in json.dumps(r) for r in requests)
                              and not any("budget_usd" in p.read_text("utf-8")
                                          for p in (tmp_path / "scratch" / "work").rglob("launch.json")),
        "no_key_material_in_run_tree": dcli.key_hits_in_run_tree(paths.run_dir) == [],
        "nothing_written_under_home": _new_under_home(marker) == [],       # None (not scanned) fails too
        "wire_and_tools": (tools == {frozenset(H1_TOOLS)} and all(r["wire"] == "chat" for r in requests)) if arm == "H1"
                          else (tools == {frozenset(H3_TOOLS)} and all(r["wire"] == "responses" for r in requests)
                                and all(t["namespace"] == "multi_agent_v1" for r in requests for t in r["tools"]
                                        if t["name"] != "update_plan")),
        "system_text_as_verified": (all(s == OPENCLAUDE_PREFIX + SYSTEM_TEXT.rstrip("\n") for s in system_texts)
                                    if arm == "H1"
                                    else all(s.startswith(SYSTEM_TEXT.rstrip("\n")) for s in system_texts)),
        "budget_sentence_delivered": all(any("Input tokens are not counted." in m["content"] for m in r["messages"]
                                             if m["role"] == "user") for r in requests),
        "harness_version_stamped": all(r.spec.harness_version not in ("absent", "unrecorded", "unknown")
                                       for r in records.values()),
    }
    _record(arm, checks)
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, (arm, failed, status.get("reconciliation"), _new_under_home(marker)[:5])
