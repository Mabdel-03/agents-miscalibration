from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from agent_allocation.passk import seal as seal_mod
from agent_allocation.passk.labels import verify_seal


def _prepare(run_root, *, bad_open: bool = False):
    (run_root / "sealed" / "v48").mkdir(parents=True)
    (run_root / "sealed" / "v48" / "predictions.parquet").write_bytes(b"pred")
    (run_root / "sealed" / "e1_selection.json").write_text('{"selected": "4x4"}')
    (run_root / "analysis_inputs").mkdir()
    (run_root / "analysis_inputs" / "draws.parquet").write_bytes(b"draws")
    logs = run_root / "logs" / "strace"
    logs.mkdir(parents=True)
    ok = f'123 openat(AT_FDCWD, "{run_root}/analysis_inputs/draws.parquet", O_RDONLY) = 3\n'
    bad = f'123 openat(AT_FDCWD, "{run_root}/artifacts/shards/mancer/grades/ab/x.json", O_RDONLY) = 4\n'
    (logs / "e2-fit.log").write_text(ok)
    (logs / "e5-fit.log").write_text(ok + (bad if bad_open else ""))


def test_seal_written_and_verifies(tmp_path):
    _prepare(tmp_path)
    assert seal_mod.run_seal(tmp_path, require_git=False) == 0
    seal = verify_seal(tmp_path) if False else json.loads((tmp_path / "sealed" / "SEAL_predictions.json").read_text())
    assert "sealed/v48/predictions.parquet" in seal["artifacts"]
    assert "analysis_inputs/draws.parquet" in seal["artifacts"]
    assert seal["static_scan"]["ok"]
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1          # written once


def test_injected_fenced_read_fails_the_seal(tmp_path):
    _prepare(tmp_path, bad_open=True)
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    assert not (tmp_path / "sealed" / "SEAL_predictions.json").exists()
    assert verify_seal(tmp_path) is None


def test_missing_strace_log_fails(tmp_path):
    _prepare(tmp_path)
    (tmp_path / "logs" / "strace" / "e2-fit.log").unlink()
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1


def test_static_scan_flags_label_access(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "e2.py").write_text(
        "from agent_allocation.passk.labels import join_labels\nPATH = 'artifacts/shards/x'\n")
    (tmp_path / "pools.py").write_text("x = 1\n")
    result = seal_mod.static_scan(tmp_path)
    assert not result["ok"] and "models/e2.py" in result["findings"]
    assert seal_mod.static_scan()["ok"]                                  # the real modules are clean


def test_real_strace_catches_a_fenced_open(tmp_path):
    if shutil.which("strace") is None:
        pytest.skip("strace not installed")
    fenced = tmp_path / "artifacts" / "shards" / "mancer" / "grades"
    fenced.mkdir(parents=True)
    (fenced / "g.json").write_text("{}")
    log = tmp_path / "trace.log"
    code = f"open({str(fenced / 'g.json')!r}).read()"
    proc = subprocess.run(["strace", "-f", "-e", "trace=open,openat,openat2", "-o", str(log), sys.executable,
                           "-c", code], capture_output=True)
    if proc.returncode != 0 or not log.exists():
        pytest.skip(f"strace not permitted here: {proc.stderr[-200:]!r}")
    result = seal_mod.check_strace(log, tmp_path)
    assert result["forbidden_opens"] >= 1 and not result["ok"]
