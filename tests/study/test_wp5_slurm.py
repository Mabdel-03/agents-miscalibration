"""Slurm forks: the study array template renders with no leftover placeholder and keeps the
frozen flags (``--signal=B:USR1@1200``, ``--no-requeue``, common.sh, run_one exec); the loop
driver template renders per-lane commands; both pass ``bash -n``.  Nothing is submitted."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "slurm" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _bash_n(text: str, tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text(text)
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_array_template_renders(tmp_path: Path):
    lc = _load("study_launch_chunked")
    text = lc._render("study_v4", "study_v4", "32B", tmp_path / "cells_1_32B.json", 100, 199, 48, "mit_preemptable", "1-00:00:00", str(tmp_path), mem="4G", cpus=1)
    assert "{" not in text.replace("${", "").replace('"$', "") or all(f"{{{k}}}" not in text for k in ("RUN_ID", "LANE", "MEM", "TIME", "CPUS", "LO", "HI", "THROTTLE", "PARTITION", "LOG_DIR", "REPO", "CELLS_FILE", "SERVER_RUN_ID"))
    assert "#SBATCH --job-name=asys-study-study_v4-32B" in text
    assert "#SBATCH --array=100-199%48" in text and "#SBATCH --mem=4G" in text and "#SBATCH --time=1-00:00:00" in text and "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --signal=B:USR1@1200" in text and "#SBATCH --no-requeue" in text
    assert 'source "' in text and "slurm/common.sh" in text and 'mamba activate "$ASYS_HARNESS_ENV"' in text
    assert "exec python -m agents_scaling.study.run_one" in text and '--server-run-id "study_v4"' in text and '--index "$SLURM_ARRAY_TASK_ID"' in text
    _bash_n(text, tmp_path, "array.sbatch")
    assert lc.job_name("study_v4", "eval") == "asys-study-study_v4-eval"


def test_chunk_complete_reads_cell_id(tmp_path: Path):
    lc = _load("study_launch_chunked")
    cells = [{"cell_id": "F.BANK.32B.N1.B0.F00.e0.s000"}, {"cell_id": "A.DEC.32B.N5.B4.Fnat.e0.s000"}]
    (tmp_path / "cells_x.json").write_text(json.dumps({"meta": {}, "cells": cells}))
    assert lc.load_cells(tmp_path / "cells_x.json") == cells
    assert lc._chunk_complete(tmp_path, 0, 1, cells) is False
    for c in cells:
        d = tmp_path / "cells" / c["cell_id"]
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
    assert lc._chunk_complete(tmp_path, 0, 1, cells) is True
    (tmp_path / "cells_x.json.sha256").write_text("00 cells_x.json\n")
    with pytest.raises(RuntimeError):
        lc.load_cells(tmp_path / "cells_x.json")


def test_loop_driver_renders(tmp_path: Path):
    loops = _load("study_loops")
    lane = loops.parse_lane("eval=cells_1-eval_eval.json:20")
    assert lane == {"lane": "eval", "cells_file": "cells_1-eval_eval.json", "throttle": 20, "cpus": 2, "mem": "8G"}
    assert loops.parse_lane("32B=cells_1_32B.json:48:1:4G")["mem"] == "4G"
    with pytest.raises(ValueError):
        loops.parse_lane("9B=x.json:1")

    class Args:
        run_id = "study_v4"
        server_run_id = None
        chunk_size = 100
        submit_cap = 380
        qos_limit = 460
        cell_partition = "mit_preemptable"
        cell_time = "1-00:00:00"
        poll_s = 120.0

    commands = "\n".join(loops.lane_command(Args, loops.parse_lane(s)) for s in ("32B=cells_1_32B.json:48", "eval=cells_1-eval_eval.json:20"))
    path, text = loops.render("study_loop_driver.sbatch.tmpl", {"REPO": str(REPO), "RUN_ID": "study_v4", "PARTITION": "mit_preemptable", "TIME": "12:00:00",
                                                                 "LOG_DIR": str(tmp_path), "LANE_COMMANDS": commands})
    assert path.name == "study_loop_driver.sbatch.study_v4"
    assert text.count("python -u slurm/study_launch_chunked.py") == 2 and "--lane 32B" in text and "--lane eval" in text and "--cpus 2" in text
    assert "afterany:$SLURM_JOB_ID" in text and "#SBATCH --requeue" not in text and "sbatch --requeue" not in text and "ASYS_ALLOW_LEGACY_CONTROL" in text and text.rstrip().endswith("wait")
    assert "{LANE_COMMANDS}" not in text and "{RUN_ID}" not in text
    _bash_n(text, tmp_path, "driver.sbatch")
    keep = (REPO / "slurm" / "loop_keepalive.sbatch.tmpl").read_text()
    assert "keepalive.py" in keep  # rendered unchanged by study_loops
