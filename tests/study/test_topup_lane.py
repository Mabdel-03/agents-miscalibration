"""study_topup_lane: never re-dispatch a cell that a live array task already owns."""
import importlib.util, json, sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("study_topup_lane", Path(__file__).resolve().parents[2] / "slurm" / "study_topup_lane.py")
T = importlib.util.module_from_spec(spec); sys.modules["study_topup_lane"] = T; spec.loader.exec_module(T)


def test_plan_topup_excludes_live_and_sorts_and_caps():
    assert T.plan_topup([5, 1, 3, 9], [3]) == [1, 5, 9]
    assert T.plan_topup([5, 1, 3, 9], [3], max_tasks=2) == [1, 5]
    assert T.plan_topup([1, 2], [1, 2]) == []
    assert T.plan_topup([], []) == []
    assert T.plan_topup([2, 2, 2], []) == [2]


def test_unfinished_indices_reads_meta(tmp_path):
    cells = [{"cell_id": f"c{i}"} for i in range(4)]
    for i in (0, 2):
        (tmp_path / "cells" / f"c{i}").mkdir(parents=True)
        (tmp_path / "cells" / f"c{i}" / "meta.json").write_text("{}")
    assert T.unfinished_indices(tmp_path, cells) == [1, 3]


def test_live_indices_only_counts_matching_manifest(tmp_path, monkeypatch):
    (tmp_path / "chunk_jobs_32B.json").write_text(json.dumps([
        {"job_id": "111", "cells_file": "cells_1_32B.json"},
        {"job_id": "222", "cells_file": "other.json"},
        {"job_id": None, "cells_file": "cells_1_32B.json"},
    ]))
    seen = []
    monkeypatch.setattr(T, "_squeue_array_indices", lambda jid: (seen.append(jid), {7} if jid == "111" else {99})[1])
    assert T.live_indices(tmp_path, "32B", "cells_1_32B.json") == {7}
    assert seen == ["111"]


def test_live_indices_without_log(tmp_path):
    assert T.live_indices(tmp_path, "32B", "cells_1_32B.json") == set()


def test_render_topup_replaces_only_the_array_line(tmp_path):
    tpl = tmp_path / "chunk.sbatch"
    tpl.write_text("#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --array=0-99%40\n#SBATCH --output=/l/%A_%a.out\nrun --index \"$SLURM_ARRAY_TASK_ID\"\n")
    out = T.render_topup(tpl, [3, 8, 12], 16, tmp_path / "chunk.topup.sbatch")
    text = out.read_text()
    assert "#SBATCH --array=3,8,12%16" in text and "0-99%40" not in text
    assert text.count("#SBATCH --array=") == 1 and "--index \"$SLURM_ARRAY_TASK_ID\"" in text
    assert "#SBATCH --job-name=x" in text and "%A_%a.out" in text


def test_render_topup_refuses_a_template_without_an_array_line(tmp_path):
    tpl = tmp_path / "no_array.sbatch"; tpl.write_text("#!/bin/bash\necho hi\n")
    with pytest.raises(SystemExit):
        T.render_topup(tpl, [1], 4, tmp_path / "out.sbatch")
