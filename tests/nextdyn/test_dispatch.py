"""The lane driver: the two admission gates, sparse resume, top-up and the live throttle.

Plan §3.3 (every number from ``compute.yaml``), §3.4 (only dependency-met cells are
submitted), §3.5 S4.  ``squeue`` and ``sbatch`` are injected everywhere, so the whole pass is
exercised with no scheduler.
"""

from __future__ import annotations

import json
import re

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import deps, dispatch as D, tables
from agents_scaling.nextdyn.cells import Block, CellSpec, ConditionRef
from agents_scaling.nextdyn.errors import ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool


def cells(n: int, *, depends_on=()) -> list[CellSpec]:
    return [CellSpec(cell_id=f"difficulty_bank.MATH.primary.deadbeef.x.pilot.r0.s{i}",
                     kind=CellKind.DIFFICULTY_BANK, lane=Lane.SMALL, pool=Pool.LOW,
                     model="primary", family="MATH", tokens_hash="deadbeef", condition=None,
                     items=(f"item{i:04d}",), replicate=0, split_role="pilot",
                     depends_on=depends_on, options={"extension_stage": 0}, block=Block.BANKS)
            for i in range(n)]


def episodes(n: int, *, depends_on=()) -> list[CellSpec]:
    """Episode cells, which are the kind whose dependency is a bucket bundle."""
    return [CellSpec(cell_id=f"episode.MATH.primary.deadbeef.aaaaaaaaaaaa.pilot.r0.c.b000{i}.s{i}",
                     kind=CellKind.EPISODE, lane=Lane.SMALL, pool=Pool.DEADLINE, model="primary",
                     family="MATH", tokens_hash="deadbeef",
                     condition=ConditionRef(protocol_hash="a" * 12, display_label="PRIVATE"),
                     items=(f"item{i:04d}",), replicate=0, split_role="pilot",
                     depends_on=depends_on, options={"origin": "constructed"},
                     block=Block.E1_CONSTRUCTED) for i in range(n)]


def complete(run_root, cell: CellSpec) -> None:
    directory = deps.cell_dir(run_root, cell.cell_id)
    (directory / tables.ITEMS_DIR).mkdir(parents=True, exist_ok=True)
    for item in cell.items:
        (directory / tables.ITEMS_DIR / f"{item}.json").write_text(
            json.dumps({"cell_id": cell.cell_id, "source_item_id": item, "status": "complete"}))
    (directory / deps.META_FILE).write_text(json.dumps(
        {"cell_id": cell.cell_id, "n_items": len(cell.items)}))


@pytest.fixture
def caps(compute_cfg) -> D.Caps:
    return D.Caps.of(compute_cfg, Lane.SMALL)


# --------------------------------------------------------------------------- caps


def test_every_cap_and_throttle_comes_from_compute_yaml(compute_cfg, caps):
    assert caps.lane_submit_cap == compute_cfg.dispatch.lane_submit_caps["8B"]
    assert caps.global_submit_cap == compute_cfg.dispatch.global_submit_cap
    assert caps.association_cap == compute_cfg.association.max_submit_jobs
    assert caps.bank_submit_cap == compute_cfg.dispatch.bank_submit_cap
    assert caps.lane_cap_for(banks_only=True) <= caps.lane_cap_for(banks_only=False)
    with pytest.raises(ManifestError, match="lane_submit_caps"):
        D.Caps.of(compute_cfg, "not-a-lane")


def test_throttle_tracks_live_endpoints(compute_cfg, caps):
    """plan §3.3: recomputed every pass, so replicas that died take the throttle with them."""
    per_endpoint = compute_cfg.dispatch.throttle_per_endpoint[CellKind.EPISODE.value]
    assert D.throttle_for(Lane.SMALL, 4, caps=caps) == per_endpoint * 4
    assert D.throttle_for(Lane.SMALL, 12, caps=caps) == per_endpoint * 12
    # a fleet that is down does not stop dispatch: one cell runs and exits 2 (no endpoint),
    # which is how the driver learns the difference between "no room" and "no server"
    assert D.throttle_for(Lane.SMALL, 0, caps=caps) == 1
    audit = compute_cfg.dispatch.throttle_per_endpoint[CellKind.SELECTION_AUDIT.value]
    assert D.throttle_for(Lane.AUDIT, 2, caps=caps) == audit * 2
    assert D.throttle_for(Lane.SMALL, 3, caps=caps, kind=CellKind.DIFFICULTY_BANK) == \
        compute_cfg.dispatch.throttle_per_endpoint[CellKind.DIFFICULTY_BANK.value] * 3


def test_admission_respects_global_and_lane_caps(caps):
    """Two gates: this lane's own tasks (exact job name) and every task the association counts."""
    state = D.chunk_state("/nonexistent", cells(300))
    assert len(state.todo) == 300

    admitted, reason = D.admit_chunks(state, caps=caps, lane_count=0, total_count=0,
                                      chunk_size=caps.chunk_size)
    assert sum(len(c) for c in admitted) == caps.lane_submit_cap
    assert "admitted" in reason

    # the lane is full: nothing is submitted, and the pass says why rather than submitting
    none, reason = D.admit_chunks(state, caps=caps, lane_count=caps.lane_submit_cap,
                                  total_count=0, chunk_size=caps.chunk_size)
    assert none == [] and "no headroom" in reason

    # the lane has room but the association does not (what QOSMaxSubmitJobPerUserLimit counts)
    none, reason = D.admit_chunks(state, caps=caps, lane_count=0,
                                  total_count=caps.global_submit_cap, chunk_size=caps.chunk_size)
    assert none == [] and "global" in reason

    # a partial headroom admits exactly that many
    partial, _ = D.admit_chunks(state, caps=caps, lane_count=caps.lane_submit_cap - 7,
                                total_count=0, chunk_size=caps.chunk_size)
    assert sum(len(c) for c in partial) == 7


def test_the_bank_sub_cap_keeps_the_banks_from_crowding_out_the_episodes(caps):
    state = D.chunk_state("/nonexistent", cells(500))
    admitted, _ = D.admit_chunks(state, caps=caps, lane_count=0, total_count=0,
                                 chunk_size=caps.chunk_size, banks_only=True)
    assert sum(len(c) for c in admitted) == min(caps.lane_submit_cap, caps.bank_submit_cap)


# --------------------------------------------------------------------------- resume


def test_sparse_resume_targets_only_cells_without_meta(tmp_run_root):
    """A chunk whose array ended with stragglers is re-submitted over exactly those indices."""
    manifest = cells(6)
    for index in (0, 1, 4):
        complete(tmp_run_root, manifest[index])
    deps.cell_dir(tmp_run_root, manifest[5].cell_id).mkdir(parents=True)
    (deps.cell_dir(tmp_run_root, manifest[5].cell_id) / deps.SUSPENDED_FILE).write_text("{}")
    state = D.chunk_state(tmp_run_root, manifest)
    assert state.done == (0, 1, 4)
    assert state.suspended == (5,)                 # terminal until a human removes the file
    assert state.todo == (2, 3)

    text = "#!/bin/bash\n#SBATCH --array=0-5%24\necho hi\n"
    resumed = D.sparse_resume(text, state.todo)
    assert "#SBATCH --array=2,3%24" in resumed
    assert D.sparse_resume(text, [3], throttle=4).count("--array=3%4") == 1
    with pytest.raises(ManifestError):
        D.sparse_resume(text, [])


def test_a_waiting_cell_is_redispatched_not_counted_failed(tmp_run_root):
    manifest = episodes(3, depends_on=(deps.bundle_dependency("MATH", "b0000"),))
    directory = deps.cell_dir(tmp_run_root, manifest[0].cell_id)
    directory.mkdir(parents=True)
    (directory / D.DEPS_UNMET_FILE).write_text("{}")
    state = D.chunk_state(tmp_run_root, manifest)
    assert state.waiting == (0,)
    assert state.suspended == ()
    assert set(state.blocked) == {0, 1, 2}         # the bundle is not there yet
    assert state.todo == ()                        # so no array slot is burned on them


def test_only_dependency_met_cells_are_admitted(tmp_run_root):
    ready = cells(2)
    waiting = episodes(2, depends_on=(deps.bundle_dependency("MATH", "b0001"),))
    state = D.chunk_state(tmp_run_root, ready + waiting)
    assert state.todo == (0, 1)
    assert set(state.blocked) == {2, 3}


# --------------------------------------------------------------------------- one pass


class FakeSqueue:
    """A ``squeue``/``scontrol`` stand-in that records what the driver asked and answered."""

    def __init__(self, *, lane_tasks: int = 0, total_tasks: int = 0, live: tuple[int, ...] = ()):
        self.lane_tasks, self.total_tasks, self.live = lane_tasks, total_tasks, live
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if cmd[:2] == ["scontrol", "update"]:
            return ""
        if "-o" in cmd and cmd[cmd.index("-o") + 1] == "%j":
            return "\n".join([D.job_name("run", Lane.SMALL)] * self.lane_tasks)
        if "-o" in cmd and cmd[cmd.index("-o") + 1] == "%K":
            return " ".join(str(i) for i in self.live)
        if "-j" in cmd and cmd[-1] == "%i":
            return cmd[cmd.index("-j") + 1]            # the job is live
        return "\n".join(str(i) for i in range(self.total_tasks))


class FakeSubmitter(D.Submitter):
    def submit(self, path):
        self.submitted.append({"path": str(path), "text": path.read_text()})
        return f"job{len(self.submitted)}"


def test_one_pass_renders_a_real_array_and_records_the_job(tmp_run_root, study_cfg, compute_cfg,
                                                           caps):
    manifest_cells = cells(3)
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, manifest_cells, block=Block.BANKS)
    squeue = FakeSqueue()
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.SMALL.value)
    report = D.lane_pass(tmp_run_root, Lane.SMALL, manifests=[path], caps=caps,
                         live_endpoints=2, cfg=study_cfg, compute=compute_cfg, run_id="run",
                         squeue=squeue, submitter=submitter)
    assert report.lane == Lane.SMALL.value
    assert len(submitter.submitted) == 1
    text = submitter.submitted[0]["text"]
    assert f"#SBATCH --job-name={D.job_name('run', Lane.SMALL)}" in text
    assert "#SBATCH --array=0,1,2%" in text
    assert "agents_scaling.nextdyn.run_one" in text
    assert "{" not in text.split("set -euo")[0]              # every placeholder rendered
    logged = json.loads((tmp_run_root / f"{D.CHUNK_LOG}_8B.json").read_text())
    assert logged[0]["indices"] == [0, 1, 2] and logged[0]["job_id"] == "job1"
    assert report.as_report().line().startswith("completed=")


def test_a_pass_counts_the_five_counters_and_submits_nothing_when_complete(
        tmp_run_root, study_cfg, compute_cfg, caps):
    manifest_cells = cells(2)
    for cell in manifest_cells:
        complete(tmp_run_root, cell)
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, manifest_cells, block=Block.BANKS)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.SMALL.value)
    report = D.lane_pass(tmp_run_root, Lane.SMALL, manifests=[path], caps=caps, live_endpoints=1,
                         cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=FakeSqueue(),
                         submitter=submitter)
    assert report.completed == 2 and report.failed == 0
    assert submitter.submitted == []


def test_a_cell_a_live_array_task_owns_is_never_dispatched_twice(tmp_run_root, study_cfg,
                                                                 compute_cfg, caps):
    manifest_cells = cells(3)
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, manifest_cells, block=Block.BANKS)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.SMALL.value)
    submitter.record({"manifest": str(path), "lane": "8B", "indices": [0, 1, 2],
                      "job_id": "999", "at": 0.0, "throttle": 8})
    squeue = FakeSqueue(live=(0, 1))
    report = D.lane_pass(tmp_run_root, Lane.SMALL, manifests=[path], caps=caps, live_endpoints=1,
                         cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=squeue,
                         submitter=submitter)
    assert report.skipped == 2                       # 0 and 1 are owned by the live array
    assert "#SBATCH --array=2%" in submitter.submitted[-1]["text"]


def test_the_throttle_is_pushed_to_every_live_array(tmp_run_root, study_cfg, compute_cfg, caps):
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, cells(1), block=Block.BANKS)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.SMALL.value)
    squeue = FakeSqueue(live=(0,))
    D.lane_pass(tmp_run_root, Lane.SMALL, manifests=[path], caps=caps, live_endpoints=3,
                cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=squeue,
                submitter=submitter)
    updates = [c for c in squeue.calls if c[:2] == ["scontrol", "update"]]
    assert updates, "the pass never applied a throttle"
    assert any(f"ArrayTaskThrottle={D.throttle_for(Lane.SMALL, 3, caps=caps)}" in c[-1]
               for c in updates)


def test_driver_refuses_a_manifest_whose_sha_is_not_in_the_plan(tmp_run_root, study_cfg,
                                                                compute_cfg, caps):
    """The frozen manifest is the authority: a rewritten one is refused, not driven.

    ARD36 kept an ``INDEX.json`` ledger and compared the manifest against it; here the
    manifest carries its own ``.sha256`` sidecar, which is the same guarantee without a second
    file that can disagree.
    """
    path = CE.manifest_path(tmp_run_root, Block.BANKS)
    CE.write_manifest(path, cells(2), block=Block.BANKS)
    path.write_text(path.read_text().replace("s0", "s9"))
    with pytest.raises(ManifestError, match="sha256"):
        D.lane_pass(tmp_run_root, Lane.SMALL, manifests=[path], caps=caps, live_endpoints=1,
                    cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=FakeSqueue(),
                    submitter=FakeSubmitter(run_root=tmp_run_root, lane="8B"))


# --------------------------------------------------------------------------- control jobs


def test_job_names_are_study_specific_so_two_runs_never_count_each_other():
    assert D.job_name("nd1", Lane.SMALL) == "asys-nextdyn-nd1-8B"
    assert D.driver_job_name("nd1", Lane.CPU) == "asys-nextdyn-nd1-driver-cpu"
    assert D.planner_job_name("nd1") == "asys-nextdyn-nd1-planner"
    assert D.status_job_name("nd1") == "asys-nextdyn-nd1-status"
    assert "ard36" not in D.job_name("nd1", Lane.SMALL)


def test_run_resubmit_does_not_duplicate_queued_arrays(tmp_run_root, study_cfg, compute_cfg,
                                                       monkeypatch):
    """``run`` re-submitted after a crash re-chains nothing that is already queued."""
    queued = {D.planner_job_name(tmp_run_root.name)}
    monkeypatch.setattr(D, "_queued_job_names", lambda runner=None: queued)
    submitted: list[str] = []
    monkeypatch.setattr(D, "_sh", lambda cmd: submitted.append(cmd) or "123")
    report = D.run_main(["--run-root", str(tmp_run_root), "--config", str(study_cfg.path),
                         "--lanes", "8B,cpu", "--dry-run"])
    assert report.skipped == 1                       # the planner was already queued
    assert report.completed == 3                     # status + two lane drivers
    assert D.planner_job_name(tmp_run_root.name) not in report.detail.split("already queued")[0]


# --------------------------------------------------------------------------- the capture lane


def capture_cells(n: int) -> list[CellSpec]:
    return [CellSpec(cell_id=f"capture.MATH.primary.deadbeef.aaaaaaaaaaaa.pilot.r0.panel.s{i}",
                     kind=CellKind.CAPTURE, lane=Lane.CAPTURE, pool=None, model="primary",
                     family="MATH", tokens_hash="deadbeef",
                     condition=ConditionRef(protocol_hash="a" * 12, display_label="PRIVATE"),
                     items=(f"item{i:04d}",), replicate=0, split_role="pilot",
                     options={}, block=Block.CAPTURE) for i in range(n)]


def test_a_capture_cell_is_rendered_from_the_template_that_allocates_a_gpu(
        tmp_run_root, study_cfg, compute_cfg):
    """A CAPTURE cell rendered from the cell array would land on a GPU partition with no GPU.

    ``nextdyn_capture.sbatch.tmpl`` is the only template carrying ``--gres``, and it is the one
    the replay worker needs; the cell array carries none.
    """
    assert D._template_for(capture_cells(1)[0]).name == D.CAPTURE_TMPL
    path = CE.manifest_path(tmp_run_root, Block.CAPTURE)
    CE.write_manifest(path, capture_cells(3), block=Block.CAPTURE)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.CAPTURE.value)
    report = D.lane_pass(tmp_run_root, Lane.CAPTURE, manifests=[path],
                         caps=D.Caps.of(compute_cfg, Lane.CAPTURE), live_endpoints=1,
                         cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=FakeSqueue(),
                         submitter=submitter)
    assert report.lane == Lane.CAPTURE.value
    assert len(submitter.submitted) == 1              # 0,1,2 is one contiguous run
    text = submitter.submitted[0]["text"]
    partition, gpu_type, gpus = D.capture_target(compute_cfg, runner=FakeSqueue())
    assert f"#SBATCH --gres=gpu:{gpu_type}:{gpus}" in text
    assert f"#SBATCH --partition={partition}" in text
    assert partition == compute_cfg.capture.partition
    assert "agents_scaling.nextdyn.run_one" in text
    assert not re.findall(r"\{[A-Z_]+\}", text)
    logged = json.loads((tmp_run_root / f"{D.CHUNK_LOG}_capture.json").read_text())
    assert logged[0]["indices"] == [0, 1, 2]


def test_a_sparse_capture_resume_becomes_one_job_per_contiguous_run(tmp_run_root, study_cfg,
                                                                    compute_cfg):
    """The capture job walks an inclusive index *range*, so a gap must not be walked over."""
    manifest_cells = capture_cells(4)
    complete(tmp_run_root, manifest_cells[1])
    path = CE.manifest_path(tmp_run_root, Block.CAPTURE)
    CE.write_manifest(path, manifest_cells, block=Block.CAPTURE)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.CAPTURE.value)
    D.lane_pass(tmp_run_root, Lane.CAPTURE, manifests=[path],
                caps=D.Caps.of(compute_cfg, Lane.CAPTURE), live_endpoints=1, cfg=study_cfg,
                compute=compute_cfg, run_id="run", squeue=FakeSqueue(), submitter=submitter)
    ranges = [(t["text"].split('FIRST="')[1].split('"')[0],
               t["text"].split('LAST="')[1].split('"')[0]) for t in submitter.submitted]
    assert ranges == [("0", "0"), ("2", "3")]


def test_the_capture_lane_falls_back_when_its_own_partition_has_no_idle_gpu(compute_cfg):
    """plan §3.2's fallback rule, which ``capture_fallback`` declared and nothing called."""
    primary = (compute_cfg.capture.partition, compute_cfg.capture.gpu_type)
    fallback_partition, fallback_types = D.capture_fallback(compute_cfg)

    def sinfo(cmd):
        partition = cmd[cmd.index("-p") + 1]
        if partition == primary[0]:
            return f"node1 gpu:{primary[1]}:4 gpu:{primary[1]}:4 idle\n"      # fully used
        return f"node2 gpu:{fallback_types[0]}:4 gpu:{fallback_types[0]}:0 idle\n"

    assert D.capture_target(compute_cfg, runner=sinfo) == (
        fallback_partition, fallback_types[0], compute_cfg.capture.gpus_per_job)
    # nothing idle anywhere: queue on the primary lane rather than inventing a partition
    assert D.capture_target(compute_cfg, runner=lambda cmd: "") == (
        *primary, compute_cfg.capture.gpus_per_job)


def test_a_live_capture_job_owns_the_indices_the_chunk_log_recorded(tmp_run_root, study_cfg,
                                                                    compute_cfg):
    """A non-array job reports no ``%K`` task ids, so without the chunk-log fallback every
    pass would re-submit cells a running capture job already holds."""
    path = CE.manifest_path(tmp_run_root, Block.CAPTURE)
    CE.write_manifest(path, capture_cells(3), block=Block.CAPTURE)
    submitter = FakeSubmitter(run_root=tmp_run_root, lane=Lane.CAPTURE.value)
    submitter.record({"manifest": str(path), "lane": "capture", "indices": [0, 1],
                      "job_id": "777", "at": 0.0, "throttle": 1})
    report = D.lane_pass(tmp_run_root, Lane.CAPTURE, manifests=[path],
                         caps=D.Caps.of(compute_cfg, Lane.CAPTURE), live_endpoints=1,
                         cfg=study_cfg, compute=compute_cfg, run_id="run", squeue=FakeSqueue(),
                         submitter=submitter)
    assert report.skipped == 2
    assert len(submitter.submitted) == 1
    assert submitter.submitted[0]["text"].split('FIRST="')[1].startswith("2")
