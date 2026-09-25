"""Gate P sandbox seam: ``passk-analyze --stage e5-fit|seal --sandbox`` really re-execs inside bwrap (skipped
where user namespaces are blocked), reads analysis_inputs/ + splits.json there, records every attempt's argv, and a
verified sandboxed fit is a no-op on re-run (resumable pre-seal chain)."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent_allocation.passk import analyze

from . import sealkit
from .test_e5 import synth
from .test_run_root_entrypoints import _write_root


def _bwrap_ok() -> bool:
    if shutil.which("bwrap") is None:
        return False
    return subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"], capture_output=True).returncode == 0


def test_verified_sandboxed_fit_is_a_noop(tmp_path, monkeypatch):
    (tmp_path / "analysis_inputs").mkdir()
    sealkit.write_prereg(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bwrap")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    assert analyze.cmd_passk_analyze(["--run-root", str(tmp_path), "--stage", "e5-fit", "--sandbox"]) == 0
    assert calls == []                                            # nothing re-ran
    (tmp_path / "sealed" / "t200" / "features.parquet").write_bytes(b"tampered")
    assert analyze.cmd_passk_analyze(["--run-root", str(tmp_path), "--stage", "e5-fit", "--sandbox"]) == 1


@pytest.mark.skipif(not _bwrap_ok(), reason="bwrap / user namespaces not available here")
def test_real_sandboxed_fit_runs_inside_bwrap(tmp_path, capfd):
    labelled, panels = synth(n_dev=4, n_t200=2, t200_labels=True)
    pre = labelled[labelled.partition != "t200_bank"].astype({"correct": object, "grade_state": object})
    pre.loc[pre.partition == "t200_diag", ["correct", "grade_state"]] = None
    root = tmp_path / "run"
    _write_root(root, pre, panels)
    fenced = root / "artifacts" / "shards" / "mancer" / "grades"
    fenced.mkdir(parents=True)
    rc = analyze.cmd_passk_analyze(["--run-root", str(root), "--stage", "e5-fit", "--sandbox"])
    # no classes.parquet in this synthetic root: the production fit refuses (exact-key fallback) -- from inside
    # the sandbox, after reading analysis_inputs/ and splits.json there
    assert rc != 0
    assert "exact vote keys" in capfd.readouterr().err       # the model ran to its final check in the sandbox
    assert not analyze.run_record_path(root, "e5-fit").exists()
    attempts = sorted((root / "logs" / "sandbox").glob("e5-fit.*.json"))
    assert len(attempts) == 1
    rec = json.loads(attempts[0].read_text())
    assert rec["argv"][0] == "bwrap" and analyze.check_sandbox_argv(rec["argv"], root) == []
    assert analyze.argv_sha256(rec["argv"]) == rec["argv_sha256"] and rec["rc"] == rc
    assert str(root.resolve() / "artifacts" / "shards") not in rec["argv"]
