"""The seven nextdyn Slurm assets: the invariants ``tests/ard/test_wp4_slurm.py:55-78`` pins.

Re-asserted here for the nextdyn templates because they are forks: the job-name prefix is the
study's (so two studies' lanes never count each other), ``--no-requeue`` is on every one of
them (a requeued array task races the cell's flock and duplicates work), the signal line and
the rc-5 rewrite make preemption and a waiting dependency ordinary outcomes rather than
failures, the exclude line is rendered, and ``--requeue`` appears nowhere except in the
comment that forbids it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import dispatch as D, fleet as FL
from agents_scaling.nextdyn.cells import Block, CellSpec, ConditionRef
from agents_scaling.nextdyn.errors import ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool
from agents_scaling.nextdyn.runner import EXIT_DEPS_UNMET

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "slurm"

#: Every template WP4 owns, and whether it is an array of cells or a control job.
ARRAY_TEMPLATES = (D.CELL_ARRAY_TMPL, D.CPU_ARRAY_TMPL)
CONTROL_TEMPLATES = (D.LANE_DRIVER_TMPL, D.PLANNER_TMPL, D.STATUS_TMPL, FL.KEEPALIVE_TMPL)
#: The capture job is a cell runner too, but it is neither an array nor a harness job: one job
#: holds one GPU and walks an inclusive index range, and it deliberately does not source
#: ``common.sh`` (that activates the harness env and the serve env's libstdc++, neither of which
#: a teacher-forced replay wants -- ``tests/nextdyn/test_capture.py`` pins that).  So it is
#: parametrized into the invariants it shares and tested on its own for the rest.
CELL_TEMPLATES = ARRAY_TEMPLATES + (D.CAPTURE_TMPL,)
ALL_TEMPLATES = CELL_TEMPLATES + CONTROL_TEMPLATES
HARNESS_TEMPLATES = ARRAY_TEMPLATES + CONTROL_TEMPLATES


def text_of(name: str) -> str:
    return (SLURM / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_every_template_exists_and_is_a_bash_job(name):
    assert text_of(name).startswith("#!/bin/bash\n")


@pytest.mark.parametrize("name", HARNESS_TEMPLATES)
def test_every_harness_job_is_named_and_sources_the_common_environment(name):
    text = text_of(name)
    assert "#SBATCH --job-name={JOB_NAME}" in text
    assert "source \"{REPO}/slurm/common.sh\"" in text


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_no_template_ever_asks_for_a_requeue(name):
    """A requeued array task races the cell's flock and duplicates work (study_v4's chain bug)."""
    text = text_of(name)
    assert "#SBATCH --no-requeue" in text
    offenders = [line for line in text.splitlines()
                 if "--requeue" in line and "--no-requeue" not in line
                 and not line.lstrip().startswith("#") or
                 ("--requeue" in line and line.lstrip().startswith("#SBATCH")
                  and "--no-requeue" not in line)]
    assert not offenders, offenders


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_the_job_name_prefix_is_this_studys(name):
    """A predecessor may be *cited* in a comment; it may never be part of a name or a path."""
    code = [line for line in text_of(name).splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in code if "ard36" in line or "study_v4" in line]
    # the name is rendered, and every renderer builds it from the nextdyn prefix
    assert D.JOB_PREFIX == FL.JOB_PREFIX == "asys-nextdyn"


@pytest.mark.parametrize("name", ARRAY_TEMPLATES)
def test_the_array_template_treats_exit_deps_unmet_as_a_wait_not_a_failure(name):
    """Exit 5 means "no meta.json, no request, the dependency is not here yet"."""
    text = text_of(name)
    assert "agents_scaling.nextdyn.run_one" in text
    assert '"$rc" -eq {EXIT_DEPS_UNMET}' in text
    assert "exit 0" in text.split('"$rc" -eq')[1]
    assert "#SBATCH --signal=B:USR1@{SIGNAL_SECONDS}" in text
    assert "{EXCLUDE_LINE}" in text
    assert "#SBATCH --array={ARRAY}%{THROTTLE}" in text


@pytest.mark.parametrize("name", CONTROL_TEMPLATES)
def test_every_control_job_self_resubmits_with_a_guard(name):
    """``afterany`` so a walltime expiry is a baton pass; ``NO_RESUBMIT`` so a test is not one."""
    text = text_of(name)
    assert "--dependency=afterany:$SLURM_JOB_ID" in text
    assert 'if [ -z "${NO_RESUBMIT:-}" ]; then' in text


def test_the_keepalive_checks_fleet_stop_before_every_launch():
    text = text_of(FL.KEEPALIVE_TMPL)
    assert f'{{RUN_ROOT}}/{FL.FLEET_STOP_FILE}' in text
    assert text.index(FL.FLEET_STOP_FILE) < text.index("fleet up")
    assert "export ASYS_PORT_OFFSET={PORT_OFFSET}" in text


def test_the_capture_template_is_the_capture_lanes(compute_cfg):
    """WP6 owns the capture job; WP4 owns the lane it runs on (``compute.capture``)."""
    text = text_of(D.CAPTURE_TMPL)
    assert "--no-requeue" in text
    assert compute_cfg.capture.gpu_type in text or "{GPU_TYPE}" in text
    # it is the ONLY template that allocates a GPU, and it is the one CAPTURE cells render from
    assert "#SBATCH --gres=gpu:{GPU_TYPE}:{GPUS_PER_JOB}" in text
    for other in ARRAY_TEMPLATES:
        assert "--gres" not in text_of(other)
    assert "#SBATCH --job-name=asys-nextdyn-{RUN_ID}-capture" in text


def test_the_capture_render_fills_every_placeholder(tmp_run_root, compute_cfg):
    """The lane's own renderer: ``--gres``, the capture partition and an inclusive range."""
    from agents_scaling.nextdyn import kinds as K

    shape = K.cell_shape(CellKind.CAPTURE, compute=compute_cfg)
    manifest = CE.manifest_path(tmp_run_root, Block.CAPTURE)
    partition, gpu_type, gpus = D.capture_target(compute_cfg, runner=lambda cmd: "")
    text = D.render_capture(SLURM / D.CAPTURE_TMPL, run_root=tmp_run_root, manifest=manifest,
                            run_id="nd1", first=0, last=3, partition=partition,
                            gpu_type=gpu_type, gpus_per_job=gpus, cpus=shape.cpus,
                            mem=shape.mem, time_limit=shape.time,
                            log_dir=tmp_run_root / "logs")
    assert f"#SBATCH --gres=gpu:{gpu_type}:{gpus}" in text
    assert f"#SBATCH --partition={partition}" in text
    assert "#SBATCH --job-name=asys-nextdyn-nd1-capture" in text
    assert 'FIRST="0"' in text and 'LAST="3"' in text
    assert not re.findall(r"\{[A-Z_]+\}", text)


def test_an_unrendered_capture_placeholder_is_refused(tmp_run_root, tmp_path):
    template = tmp_path / "broken_capture.sbatch.tmpl"
    template.write_text("#!/bin/bash\n#SBATCH --partition={PARTITION}\necho {NOT_A_KNOWN_KEY}\n")
    with pytest.raises(ManifestError, match="unrendered"):
        D.render_capture(template, run_root=tmp_run_root, manifest=tmp_run_root / "m.json",
                         run_id="nd1", first=0, last=0, partition="p", gpu_type="a100",
                         gpus_per_job=1, cpus=8, mem="64G", time_limit="4:00:00",
                         log_dir=tmp_run_root)


# --------------------------------------------------------------------------- rendering


def cell() -> CellSpec:
    return CellSpec(cell_id="episode.MATH.primary.deadbeef.aaaaaaaaaaaa.pilot.r0.c.s0",
                    kind=CellKind.EPISODE, lane=Lane.SMALL, pool=Pool.DEADLINE, model="primary",
                    family="MATH", tokens_hash="deadbeef",
                    condition=ConditionRef(protocol_hash="a" * 12, display_label="PRIVATE"),
                    items=("item0",), replicate=0, split_role="pilot",
                    options={"origin": "constructed"}, block=Block.E1_CONSTRUCTED)


def test_the_chunk_render_fills_every_placeholder(tmp_run_root, study_cfg, compute_cfg):
    manifest = CE.manifest_path(tmp_run_root, Block.E1_CONSTRUCTED)
    CE.write_manifest(manifest, [cell()], block=Block.E1_CONSTRUCTED)
    text = D.render_array(
        SLURM / D.CELL_ARRAY_TMPL, run_root=tmp_run_root, manifest=manifest, lane=Lane.SMALL,
        run_id="nd1", indices=[0, 1, 5], throttle=24, partition="mit_normal", cpus=1, mem="3G",
        time_limit="1-00:00:00", log_dir=tmp_run_root / "logs",
        exclude=compute_cfg.serving.exclude_nodes,
        signal_s=compute_cfg.dispatch.signal_seconds_before_walltime, config=study_cfg.path)
    assert "#SBATCH --array=0,1,5%24" in text
    assert "#SBATCH --job-name=asys-nextdyn-nd1-8B" in text
    assert f"#SBATCH --signal=B:USR1@{compute_cfg.dispatch.signal_seconds_before_walltime}" in text
    assert f"#SBATCH --exclude={','.join(compute_cfg.serving.exclude_nodes)}" in text
    assert f'"$rc" -eq {EXIT_DEPS_UNMET}' in text
    assert not re.findall(r"\{[A-Z_]+\}", text)


def test_an_unrendered_placeholder_is_refused(tmp_run_root, study_cfg, compute_cfg, tmp_path):
    template = tmp_path / "broken.sbatch.tmpl"
    template.write_text("#!/bin/bash\n#SBATCH --job-name={JOB_NAME}\necho {NOT_A_KNOWN_KEY}\n")
    manifest = CE.manifest_path(tmp_run_root, Block.E1_CONSTRUCTED)
    CE.write_manifest(manifest, [cell()], block=Block.E1_CONSTRUCTED)
    with pytest.raises(ManifestError, match="unrendered"):
        D.render_array(template, run_root=tmp_run_root, manifest=manifest, lane=Lane.SMALL,
                       run_id="nd1", indices=[0], throttle=1, partition="mit_normal", cpus=1,
                       mem="3G", time_limit="1:00:00", log_dir=tmp_run_root, exclude=(),
                       signal_s=60, config=study_cfg.path)


def test_the_control_jobs_render_with_their_own_intervals(tmp_run_root, study_cfg, compute_cfg):
    for name, template, extra in D._control_jobs("nd1", [Lane.SMALL]):
        path = D._render_control(tmp_run_root, study_cfg, compute_cfg, name=name,
                                 template=template, extra=extra)
        text = path.read_text()
        assert f"#SBATCH --job-name={name}" in text
        assert not re.findall(r"\{[A-Z_]+\}", text)
        assert str(tmp_run_root) in text
