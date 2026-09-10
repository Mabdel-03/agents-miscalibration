"""The offline dry run itself (WP4; plan §10 "Offline dry run (Phase 0, CPU)").

Drives ``scripts/ard36_dry_run.py`` in ``--quick`` mode: the real ``run_one`` CLI over a
frozen manifest per code path (root bank, one episode protocol, the controller-gated arm,
grading, compaction) against the in-process fake vLLM server, plus the two assertions the
plan names — the dependency gate (absent controller ⇒ ``EXIT_DEPS_UNMET``, no ``meta.json``,
zero requests) and resume (delete the item files and ``meta.json``, rerun, HTTP call count
unchanged).

The full four-item run over every manifest is the G1 gate command:

    PYTHONPATH=src python scripts/ard36_dry_run.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dry_run_module():
    path = REPO / "scripts" / "ard36_dry_run.py"
    spec = importlib.util.spec_from_file_location("_ard36_dry_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(dry_run_module, tmp_path_factory):
    root = tmp_path_factory.mktemp("dry")
    code = dry_run_module.main(["--quick", "--items", "2", "--keep", "--run-root", str(root)])
    payload = json.loads((root / dry_run_module.RUN_ID / "dry_run_report.json").read_text())
    return code, payload


def test_the_dry_run_is_green(report):
    code, payload = report
    assert payload["failures"] == [] and code == 0


def test_every_driven_manifest_completed(report):
    _code, payload = report
    assert set(payload["driven"]) >= {"E1_ROOTS", "E2a_EPISODE", "E4_PROTECT", "GRADE",
                                      "COMPACT"}
    for label, block in payload["driven"].items():
        assert block["cells"] > 0, label
        assert set(block["exit_codes"]) == {0}, label


def test_the_dependency_gate_held_without_generating_a_request(report):
    _code, payload = report
    gate = payload["dependency_gate"]
    assert gate["exit_codes"] and set(gate["exit_codes"]) == {5}
    assert gate["no_meta_json"] is True and gate["no_requests"] is True


def test_resume_regenerated_nothing(report):
    _code, payload = report
    assert payload["resume"]["unchanged"] is True
    assert payload["resume"]["chat_calls_before"] > 0


def test_the_dry_run_compacts_every_table(report):
    _code, payload = report
    assert set(payload["tables"]) >= set(K.ALL_TABLES)
    for name in ("episodes", "decisions", "labels", "round_summary", "transitions",
                 "cost_summary", "bank_summary"):
        assert payload["tables"][name] > 0, name


def test_kinds_owned_by_later_work_packages_are_reported_as_pending(report):
    _code, payload = report
    pending = payload["pending"]
    for kind in ("RECEIVER", "SELECTOR_AUDIT", "E0", "CAPTURE"):
        assert kind in pending
