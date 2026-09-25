from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from agent_allocation.passk import analyze
from agent_allocation.passk import seal as seal_mod
from agent_allocation.passk.labels import verify_seal
from . import sealkit


def _prepare(run_root, **kw):
    (run_root / "analysis_inputs").mkdir(parents=True, exist_ok=True)
    (run_root / "analysis_inputs" / "draws.parquet").write_bytes(b"draws")
    (run_root / "analysis_inputs" / "manifest.json").write_text('{"n_rows": 1}')
    sealkit.write_prereg(run_root, **kw)


def test_seal_written_and_verifies(tmp_path):
    _prepare(tmp_path)
    assert seal_mod.run_seal(tmp_path, require_git=False) == 0
    seal = json.loads((tmp_path / "sealed" / "SEAL_predictions.json").read_text())
    assert "sealed/v48/predictions.parquet" in seal["artifacts"]
    assert "sealed/runs/e2-fit.json" in seal["artifacts"] and set(seal["sandbox_runs"]) == {"e2-fit", "e5-fit"}
    assert seal["splits"] == ["v48", "t200"]
    assert "analysis_inputs/draws.parquet" in seal["inputs_at_seal"]
    assert "sealed/inputs_manifest_at_seal.json" in seal["artifacts"]
    (tmp_path / "analysis_inputs" / "draws.parquet").write_bytes(b"re-extracted after release")
    (tmp_path / "analysis_inputs" / "manifest.json").write_text('{"n_rows": 2, "releases": ["v48"]}')
    assert verify_seal(tmp_path) is not None                            # a release re-extract keeps the seal
    assert seal["static_scan"]["ok"]
    before = (tmp_path / "sealed" / "SEAL_predictions.json").read_bytes()
    assert seal_mod.run_seal(tmp_path, require_git=False) == 0          # idempotent: re-run is a verified no-op
    assert (tmp_path / "sealed" / "SEAL_predictions.json").read_bytes() == before
    assert verify_seal(tmp_path) is not None                            # ... and never un-verifies


@pytest.mark.parametrize("missing", ["sealed/v48/crossover.parquet", "sealed/t200/predictions.parquet",
                                     "sealed/t200/seeds.json", "sealed/v48/models/ridge.json"])
def test_partial_prereg_contents_refused(tmp_path, missing):
    _prepare(tmp_path, skip=(missing,))
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    assert verify_seal(tmp_path) is None


def test_missing_predictor_set_refused(tmp_path):
    _prepare(tmp_path, predictors=("P0", "P1", "P2"))
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1


def test_unsandboxed_or_foreign_runs_refused(tmp_path):
    _prepare(tmp_path)
    analyze.run_record_path(tmp_path, "e5-fit").unlink()                # fit ran without --sandbox
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    outs = [p for p in ("sealed/t200/predictions.parquet", "sealed/t200/features.parquet",
                        "sealed/t200/manifest.json", "sealed/t200/seeds.json", "sealed/t200/models/P1.json")]
    wide = analyze.sandbox_argv(tmp_path, ["--stage", "e5-fit"])
    i = wide.index("--chdir")
    wide = wide[:i] + ["--ro-bind", str(tmp_path / "artifacts"), str(tmp_path / "artifacts")] + wide[i:]
    sealkit.record_sandboxed_run(tmp_path, "e5-fit", outs, argv=wide)  # shards visible inside -> not the fence
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    sealkit.record_sandboxed_run(tmp_path, "e5-fit", outs, rc=1)
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    sealkit.record_sandboxed_run(tmp_path, "e5-fit", outs)
    (tmp_path / "sealed" / "t200" / "features.parquet").write_bytes(b"edited after the run")
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1
    assert not (tmp_path / "sealed" / "SEAL_predictions.json").exists()


def test_audit_hook_and_strace_logs_are_not_accepted(tmp_path):
    _prepare(tmp_path)
    for stage in ("e2-fit", "e5-fit"):
        analyze.run_record_path(tmp_path, stage).unlink()
    logs = tmp_path / "logs" / "strace"
    logs.mkdir(parents=True)
    for stage in ("e2-fit", "e5-fit"):
        (logs / f"{stage}.log").write_text("# mode=audithook\n1 openat(AT_FDCWD, \"x\", O_RDONLY) = 3\n")
    assert seal_mod.run_seal(tmp_path, require_git=False) == 1


def test_sandbox_argv_mounts_only_the_allow_list(tmp_path):
    (tmp_path / "analysis_inputs").mkdir()
    (tmp_path / "sealed").mkdir()
    (tmp_path / "gates").mkdir()
    (tmp_path / "artifacts" / "shards").mkdir(parents=True)
    argv = analyze.sandbox_argv(tmp_path, ["--stage", "e2-fit"])
    assert analyze.check_sandbox_argv(argv, tmp_path) == []
    assert "--unshare-all" in argv and "--die-with-parent" in argv
    srcs = [argv[i + 1] for i, t in enumerate(argv) if t in ("--bind", "--ro-bind")]
    assert str(tmp_path.resolve() / "artifacts" / "shards") not in srcs
    assert [argv[i + 1] for i, t in enumerate(argv) if t == "--bind"] == [str(tmp_path.resolve() / "sealed")]
    assert not any(s.startswith("/orcd/data") and not (s.startswith(str(analyze.CODE_ROOT))
                                                       or s.startswith(str(tmp_path.resolve())))
                   for s in srcs)


def test_real_bwrap_hides_fenced_data(tmp_path):
    if shutil.which("bwrap") is None:
        pytest.skip("bwrap not installed")
    probe = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/", "/", "true"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f"user namespaces not permitted here: {probe.stderr[-200:]!r}")
    run_root = tmp_path / "root"
    fenced = run_root / "artifacts" / "shards" / "mancer" / "grades"
    fenced.mkdir(parents=True)
    (fenced / "g.json").write_text("{}")
    (run_root / "analysis_inputs").mkdir()
    (run_root / "analysis_inputs" / "ok.txt").write_text("ok")
    (run_root / "sealed").mkdir()
    argv = analyze.sandbox_argv(run_root, ["--stage", "e2-fit"])
    prog = argv.index(sys.executable)
    code = (f"import os; print(os.path.exists({str(fenced / 'g.json')!r}), "
            f"open({str(run_root / 'analysis_inputs' / 'ok.txt')!r}).read())")
    proc = subprocess.run(argv[:prog] + [sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0 and "Operation not permitted" in proc.stderr:
        pytest.skip(f"bwrap blocked here: {proc.stderr[-200:]!r}")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "ok"]


def test_static_scan_flags_label_access(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "e2.py").write_text(
        "from agent_allocation.passk.labels import join_labels\nPATH = 'artifacts/shards/x'\n")
    (tmp_path / "models" / "leak.py").write_text(
        "import importlib\nfrom pathlib import Path\n"
        "def f(r):\n    return list((Path(r) / 'artifacts' / 'shards' / 'm' / 'grades').rglob('*'))\n"
        "def g(r):\n    import pyarrow.parquet as pq\n    return pq.read_table(Path(r) / 'catalog' / 'x.parquet')\n")
    (tmp_path / "models" / "grader.py").write_text("from agent_allocation.grading import worker\n")
    (tmp_path / "pools.py").write_text('"""Never reads shards/ or private/."""\nx = 1\n')
    result = seal_mod.static_scan(tmp_path)
    assert not result["ok"] and "models/e2.py" in result["findings"] and "pools.py" not in result["findings"]
    assert "models/leak.py" in result["findings"] and "models/grader.py" in result["findings"]
    leak = " ".join(result["findings"]["models/leak.py"])
    assert "importlib" in leak and "'shards'" in leak and "read_table()" in leak
    assert seal_mod.static_scan()["ok"]                                  # the real modules are clean
