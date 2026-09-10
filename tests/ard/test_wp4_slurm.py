"""WP4 Slurm assets: the ARD lane tuple, the EXIT_DEPS_UNMET path, the operational fixes.

Plan §6 (one independent 15-minute loop per lane, sparse-array resubmission of only the
unfinished cells, manifest-scoped in-flight checks, chunk-log merge, ``--submit-cap 380
--qos-limit 420`` on mit_preemptable with 230 for ou_bcs_*/pi_*), §10 (the dependency gate
threaded through the array template and the chunked launcher so the driver re-queues instead
of counting the cell failed).

Nothing here submits anything: the templates are rendered and read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.types import EXIT_DEPS_UNMET

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "slurm"


def _load(name: str):
    path = SLURM / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ard36_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chunked():
    return _load("ard36_launch_chunked")


@pytest.fixture(scope="module")
def topup():
    return _load("ard36_topup_lane")


@pytest.fixture(scope="module")
def loops():
    return _load("ard36_loops")


# --------------------------------------------------------------------------- lanes


def test_the_drivers_use_the_ard_lane_tuple_and_leave_the_study_ones_alone(chunked, loops):
    assert tuple(chunked.LANES) == K.LANES
    assert set(loops.LANE_DEFAULTS) == set(K.LANES)
    study_chunked = (SLURM / "study_launch_chunked.py").read_text()
    study_loops = (SLURM / "study_loops.py").read_text()
    for text in (study_chunked, study_loops):
        assert 'LANES = ("32B", "14B", "8B", "4B", "eval")' in text
    from agents_scaling.study.cells import LANES as STUDY_LANES

    assert STUDY_LANES == ("32B", "14B", "8B", "4B", "eval")


def test_job_names_are_ard_specific_so_the_per_lane_counts_never_mix_studies(chunked):
    assert chunked.job_name("ard36", "8B") == "asys-ard36-ard36-8B"
    assert "study" not in chunked.job_name("ard36", "8B")


def test_qos_limits_follow_the_partition(chunked):
    assert chunked.qos_limit_for("mit_preemptable") == 420
    assert chunked.qos_limit_for("ou_bcs_low") == 230
    assert chunked.qos_limit_for("pi_manoli") == 230
    assert chunked.qos_limit_for("pi_tpoggio,ou_bcs_normal") == 230
    assert chunked.qos_limit_for("mit_preemptable", 99) == 99
    assert chunked.DEFAULT_SUBMIT_CAP == 380


# --------------------------------------------------------------------------- array template


def test_the_array_template_treats_exit_deps_unmet_as_a_wait_not_a_failure():
    text = (SLURM / "ard36_cell_array.sbatch.tmpl").read_text()
    assert "agents_scaling.ard.run_one" in text
    assert f'"$rc" -eq {EXIT_DEPS_UNMET}' in text
    assert "--signal=B:USR1@1200" in text and "--no-requeue" in text
    assert "--exclude=node3804" in text


def test_the_chunk_render_fills_every_placeholder(chunked, tmp_path):
    text = chunked._render("ard36", "ard36", "8B", tmp_path / "cells.json", 0, 9, 24,
                           "mit_preemptable", "1-00:00:00", str(tmp_path), mem="4G", cpus=1)
    assert "#SBATCH --array=0-9%24" in text
    assert "asys-ard36-ard36-8B" in text and "{" not in text.split("set -euo")[0]


# --------------------------------------------------------------------------- admission gates


def _manifest(tmp_path, cell_ids):
    payload = {"meta": {}, "cells": [{"cell_id": cid} for cid in cell_ids]}
    path = tmp_path / "cells_x.json"
    path.write_text(json.dumps(payload))
    return path, payload["cells"]


def test_a_deps_unmet_cell_is_not_complete_but_a_suspended_one_is_terminal(chunked, tmp_path):
    run_root = tmp_path / "run"
    cells = [{"cell_id": "a"}, {"cell_id": "b"}, {"cell_id": "c"}]
    for cell in cells:
        (run_root / "cells" / cell["cell_id"]).mkdir(parents=True)
    (run_root / "cells" / "a" / "meta.json").write_text("{}")
    (run_root / "cells" / "b" / "SUSPENDED.json").write_text("{}")
    assert chunked._chunk_complete(run_root, 0, 1, cells) is True     # meta + suspended
    (run_root / "cells" / "c" / chunked.DEPS_UNMET_FILE).write_text("{}")
    assert chunked._chunk_complete(run_root, 0, 2, cells) is False    # waiting, must be re-queued


def test_the_manifest_scoped_in_flight_check_ignores_another_manifests_chunk(chunked):
    log = [{"lo": 0, "hi": 9, "job_id": "1", "cells_file": "cells_other.json"}]
    assert chunked._chunk_job_in_flight(log, 0, 9, cells_file="cells_mine.json") is None


def test_a_frozen_manifest_with_a_wrong_sidecar_is_refused(chunked, tmp_path):
    path, _cells = _manifest(tmp_path, ["a"])
    path.with_name(path.name + ".sha256").write_text("00" * 32 + "  cells_x.json\n")
    with pytest.raises(RuntimeError, match="sha256"):
        chunked.load_cells(path)


# --------------------------------------------------------------------------- top-up


def test_topup_dispatches_only_unfinished_cells_that_no_live_task_owns(topup, tmp_path):
    run_root = tmp_path / "run"
    cells = [{"cell_id": name} for name in ("a", "b", "c", "d")]
    for cell in cells:
        (run_root / "cells" / cell["cell_id"]).mkdir(parents=True)
    (run_root / "cells" / "a" / "meta.json").write_text("{}")
    (run_root / "cells" / "b" / "SUSPENDED.json").write_text("{}")
    (run_root / "cells" / "c" / topup.DEPS_UNMET_FILE).write_text("{}")
    todo = topup.unfinished_indices(run_root, cells)
    assert todo == [2, 3]                              # a done, b terminal
    assert topup.waiting_indices(run_root, cells) == [2]
    assert topup.plan_topup(todo, live=[3]) == [2]
    assert topup.plan_topup(todo, live=[], max_tasks=1) == [2]


def test_topup_renders_a_sparse_array_over_absolute_indices(topup, tmp_path):
    template = tmp_path / "chunk.sbatch"
    template.write_text("#!/bin/bash\n#SBATCH --array=0-9%24\necho hi\n")
    out = topup.render_topup(template, [3, 7], 4, tmp_path / "chunk.topup.sbatch")
    assert "#SBATCH --array=3,7%4" in out.read_text()


# --------------------------------------------------------------------------- loops


def test_each_lane_gets_its_own_independent_fifteen_minute_loop(loops, tmp_path):
    assert loops.LOOP_INTERVAL_S == 900
    text = (SLURM / "ard36_loop_driver.sbatch.tmpl").read_text()
    assert "lane_loop()" in text and "sleep {LOOP_INTERVAL_S}" in text
    assert "afterany" in text
    # the only "--requeue" in the file is the comment forbidding it
    assert [line for line in text.splitlines() if "--requeue" in line] == \
        ["# job, never a login-node nohup, and self-resubmits with afterany so a walltime expiry is a",
         "# baton pass.  NEVER add --requeue (chain duplication; see launch_loops.py)."][1:]
    args = type("A", (), {"run_id": "ard36", "server_run_id": None, "chunk_size": 60,
                          "submit_cap": 380, "qos_limit": 420, "cell_partition": "mit_preemptable",
                          "cell_time": "1-00:00:00", "poll_s": 120.0, "topup": True})()
    command = loops.lane_command(args, loops.parse_lane("8B=cells_E2a_8B.json:24"))
    assert command.startswith("lane_loop 8B '") and command.endswith("&")
    assert "ard36_launch_chunked.py" in command and "ard36_topup_lane.py" in command
    assert "--submit-cap 380" in command and "--qos-limit 420" in command


def test_lane_specs_default_to_the_plan_cpu_shapes(loops):
    assert loops.parse_lane("grade=cells_GRADE.json")["cpus"] == 2
    assert loops.parse_lane("analysis=cells_COMPACT.json")["mem"] == "32G"
    assert loops.parse_lane("8B=cells.json:12:1:8G") == {
        "lane": "8B", "cells_file": "cells.json", "throttle": 12, "cpus": 1, "mem": "8G"}
    with pytest.raises(ValueError):
        loops.parse_lane("14B=cells.json:12")


def test_the_gpu_sampler_helper_writes_the_frozen_query():
    from agents_scaling.ard import resources as RES

    text = (SLURM / "ard36_gpu_sampler.sh").read_text()
    assert "--query-gpu=" + ",".join(RES.NVIDIA_SMI_FIELDS) in text
    assert "ard36_start_gpu_sampler" in text and "systime()" in text
