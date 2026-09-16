"""The serving fleet: spec parity, the census key, scale-to-idle, caps and the sampler inserts.

Plan §3.2, D20.  Four of these are named in the brief:
``test_spec_parity_with_legacy_parser``, ``test_census_keys_on_gpu_type_so_h100_and_a100_
entries_do_not_fight``, ``test_fleet_stop_file_prevents_any_launch`` and
``test_scaled_spec_never_exceeds_idle_census``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agents_scaling.nextdyn import fleet as FL
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation, ManifestError
from agents_scaling.nextdyn.kinds import Pool

REPO = Path(__file__).resolve().parents[2]

#: A dozen spec strings covering every shape the grammar admits: four fields and five, a
#: day-prefixed walltime and a plain one, several entries, whitespace, an empty tail.
SPECS = (
    "nd1-8B:1:pi_tpoggio:7-00:00:00",
    "nd1-8B:2:pi_tpoggio:2-12:00:00",
    "nd1-32B:1:ou_bcs_normal:1-00:00:00:h100",
    "nd1-8B:4:ou_bcs_normal:20:00:00:h100",
    "nd1-8B:8:ou_bcs_normal:1-00:00:00",
    "nd1-32B:1:mit_normal_gpu:6:00:00:h200",
    "nd1-8B:2:mit_preemptable:2-00:00:00:h200,nd1-32B:1:mit_preemptable:2-00:00:00:h200",
    "nd1-8B:8:ou_bcs_low:1-00:00:00,nd1-8B:4:ou_bcs_low:1-00:00:00:h100",
    "nd1-32B:2:pi_manoli:3-00:00:00,nd1-32B:1:pi_tpoggio:3-00:00:00",
    "0.6B:1:pi_tpoggio:7-00:00:00",
    " nd1-8B:1:ou_bcs_low:12:00:00 ",
    "nd1-8B:1:ou_bcs_low:12:00:00,",
)


@pytest.fixture(scope="module")
def legacy():
    """``slurm/keepalive.py``'s parser, loaded without importing the 3,941-line module."""
    source = (REPO / "slurm" / "keepalive.py").read_text(encoding="utf-8")
    start = source.index("def parse_spec(")
    end = source.index("def _pool_job_name(")
    namespace: dict[str, object] = {}
    exec("from dataclasses import dataclass\n"                       # noqa: S102 - the vendor
         "@dataclass\n"
         "class Target:\n"
         "    size: str\n    count: int\n    partition: str\n    time_limit: str\n"
         "    gpu_type: str = 'a100'\n\n" + source[start:end], namespace)
    return namespace["parse_spec"]


def test_spec_parity_with_legacy_parser(legacy):
    """The vendored parser answers exactly what ``slurm/keepalive.py:172`` answers.

    Vendored rather than imported because the legacy module reaches the scheduler at import
    and is not on the ``ARD_IMPORTS`` allowlist; the parity test is what makes that safe.
    """
    for spec in SPECS:
        ours = FL.parse_spec(spec, "a100")
        theirs = legacy(spec, "a100")
        assert len(ours) == len(theirs), spec
        for mine, other in zip(ours, theirs):
            assert (mine.size, mine.count, mine.partition, mine.time_limit, mine.gpu_type) == \
                (other.size, other.count, other.partition, other.time_limit, other.gpu_type), spec
    for bad in ("nd1-8B:1:pi_tpoggio", "nd1-8B::pi_tpoggio:1:00:00:00:extra:more"):
        with pytest.raises(ValueError):
            FL.parse_spec(bad, "a100")


def test_a_spec_round_trips_through_its_rendering():
    for spec in SPECS:
        targets = FL.parse_spec(spec, "a100")
        assert FL.parse_spec(FL.render_spec(targets), "a100") == targets


# --------------------------------------------------------------------------- census


def test_census_keys_on_gpu_type_so_h100_and_a100_entries_do_not_fight(tmp_run_root):
    """One profile may hold H100 and A100 replicas on one partition (plan §3.2).

    ARD36's (profile, partition) key made those two entries share a count, so the keepalive
    launched and cancelled in a loop.
    """
    from agents_scaling.serving import registry as REG

    pool_root = FL.pool_run_root(tmp_run_root, Pool.LOW)
    name = REG.serving_job_name(str(pool_root), "nd1-8B")
    rows = "\n".join([
        f"1 {name} ou_bcs_normal RUNNING gpu:a100:1 node3801 1:00:00",
        f"2 {name} ou_bcs_normal RUNNING gpu:h100:1 node1802 0:30:00",
        f"3 {name} ou_bcs_normal PENDING gpu:h100:1 (None) 0:00",
        "4 someone-elses-job ou_bcs_normal RUNNING gpu:a100:8 node3805 5:00:00",
    ])
    census = FL.census(tmp_run_root, Pool.LOW, ["nd1-8B"], runner=lambda cmd: rows)
    assert set(census) == {("nd1-8B", "ou_bcs_normal", "a100"), ("nd1-8B", "ou_bcs_normal", "h100")}
    assert len(census[("nd1-8B", "ou_bcs_normal", "a100")]) == 1
    assert len(census[("nd1-8B", "ou_bcs_normal", "h100")]) == 2      # pending counts as in flight
    assert all(job.name == name for jobs in census.values() for job in jobs)


def test_the_pools_are_sub_run_roots(tmp_run_root, compute_cfg):
    """``registry.servers_dir`` is hard-coded to ``<root>/servers``, so a pool is a root."""
    for pool in Pool:
        root = FL.pool_run_root(tmp_run_root, pool)
        assert root == tmp_run_root / FL.POOLS_DIR / pool.value
        assert FL.port_offset(compute_cfg, pool) == getattr(compute_cfg.pools, pool.value).port_offset
    assert len({FL.port_offset(compute_cfg, p) for p in Pool}) == len(list(Pool))


# --------------------------------------------------------------------------- scale to idle


def test_scaled_spec_never_exceeds_idle_census():
    """plan D20: scale down to what is idle, so the launch preempts nobody.

    Entries are visited in spec order and each consumes the pool it draws from, so two entries
    on one (partition, gpu_type) cannot both be told "there are four free".
    """
    targets = FL.parse_spec("nd1-8B:8:ou_bcs_low:1-00:00:00,nd1-32B:2:ou_bcs_low:1-00:00:00",
                            "a100")
    gpus = {"nd1-8B": 1, "nd1-32B": 2}
    scaled = FL.scale_to_idle(targets, idle={("ou_bcs_low", "a100"): 5}, gpus_per_replica=gpus)
    assert [(t.size, t.count) for t in scaled] == [("nd1-8B", 5)]     # 5 idle, 5 taken, none left
    scaled = FL.scale_to_idle(targets, idle={("ou_bcs_low", "a100"): 12}, gpus_per_replica=gpus)
    assert [(t.size, t.count) for t in scaled] == [("nd1-8B", 8), ("nd1-32B", 2)]
    assert sum(t.count * gpus[t.size] for t in scaled) <= 12
    # nothing idle: nothing launched, and the entry is dropped rather than queued to wait
    assert FL.scale_to_idle(targets, idle={("ou_bcs_low", "a100"): 0}, gpus_per_replica=gpus) == []


def test_free_gpus_skips_drained_and_excluded_nodes():
    sinfo = "\n".join([
        "node3801 gpu:a100:4 gpu:a100:1 idle",
        "node3802 gpu:a100:4 gpu:a100:4 mixed",
        "node3804 gpu:a100:4 gpu:a100:0 idle",        # excluded by compute.yaml
        "node3806 gpu:a100:4 gpu:a100:0 drained",
    ])
    census = FL.free_gpus("ou_bcs_low", "a100", exclude=("node3804",), runner=lambda cmd: sinfo)
    assert census.total == 8 and census.used == 5 and census.free == 3
    assert set(census.drained) == {"node3804", "node3806"}


# --------------------------------------------------------------------------- caps


def test_fleet_up_refuses_a_spec_over_the_captured_cap():
    """plan D20: the LIVE MaxTRESPU from ``preflight/h0.json``, not compute.yaml's snapshot."""
    preflight = {"qos": {"caps": [{"partition": "ou_bcs_normal", "gpu_type": "h100", "gpu": 8},
                                  {"partition": "ou_bcs_normal", "gpu_type": "a100", "gpu": 8}]}}
    gpus = {"nd1-8B": 1, "nd1-32B": 2}
    within = FL.parse_spec("nd1-8B:6:ou_bcs_normal:1-00:00:00:h100", "a100")
    FL.assert_within_caps(within, preflight, gpus_per_replica=gpus)
    over = FL.parse_spec("nd1-32B:5:ou_bcs_normal:1-00:00:00:h100", "a100")
    with pytest.raises(ContractViolation, match="MaxTRESPU"):
        FL.assert_within_caps(over, preflight, gpus_per_replica=gpus)
    # a partition the preflight did not capture is not silently refused; it is simply unknown
    FL.assert_within_caps(FL.parse_spec("nd1-8B:99:pi_manoli:1-00:00:00", "a100"), preflight,
                          gpus_per_replica=gpus)


def test_a_missing_preflight_blocks_the_launch(tmp_run_root):
    with pytest.raises(ManifestError, match="nextdyn_preflight"):
        FL.load_preflight(tmp_run_root)


# --------------------------------------------------------------------------- stop


def test_fleet_stop_file_prevents_any_launch(tmp_run_root, compute_cfg):
    """The keepalive reads ``$RUN/FLEET_STOP`` at the top of every tick.

    ``fleet stop`` writes it *before* any ``scancel``, which is what stops the study_v4 race
    where a keepalive already in flight resurrected a fleet that had just been cancelled.
    """
    targets = FL.parse_spec("nd1-8B:2:ou_bcs_low:1-00:00:00", "a100")
    launched: list[object] = []
    report = FL.tick(tmp_run_root, Pool.LOW, targets, compute=compute_cfg,
                     submit=lambda *a, **k: launched.append(a), runner=lambda cmd: "")
    assert report.launched and launched                       # without the file it launches

    FL.fleet_stop_path(tmp_run_root).write_text("stop\n")
    launched.clear()
    stopped = FL.tick(tmp_run_root, Pool.LOW, targets, compute=compute_cfg,
                      submit=lambda *a, **k: launched.append(a), runner=lambda cmd: "")
    assert stopped.stopped and not stopped.launched and not launched
    assert FL.is_stopped(tmp_run_root)


def test_stop_writes_the_file_before_cancelling_anything(tmp_run_root):
    order: list[str] = []

    def cancel(name: str) -> None:
        order.append(f"cancel:{FL.is_stopped(tmp_run_root)}")

    report = FL.stop(tmp_run_root, run_id="nd1", cancel=cancel)
    assert order and all(entry == "cancel:True" for entry in order)
    assert report.completed == len(order)


def test_job_names_are_exact_and_cover_every_control_job():
    names = FL.job_names("nd1")
    assert names.planner == "asys-nextdyn-nd1-planner"
    assert set(names.keepalives) == {FL.keepalive_job_name("nd1", p) for p in Pool}
    assert all(name.startswith(FL.JOB_PREFIX) for name in names.ordered)
    # teardown order: the keepalives first (they would relaunch), then the planner and drivers
    assert names.ordered.index(names.keepalives[0]) < names.ordered.index(names.planner)
    assert names.serve_prefix.startswith("asys-srv-")


def test_the_tick_never_launches_more_than_the_configured_budget(tmp_run_root, compute_cfg):
    targets = FL.parse_spec("nd1-8B:9:ou_bcs_low:1-00:00:00", "a100")
    launched: list[object] = []
    report = FL.tick(tmp_run_root, Pool.LOW, targets, compute=compute_cfg,
                     submit=lambda *a, **k: launched.append(a), runner=lambda cmd: "")
    assert len(report.launched) == compute_cfg.keepalive.max_launch
    assert report.skipped                                      # and it says what it did not do


# --------------------------------------------------------------------------- serve inserts


@pytest.fixture(scope="module")
def rendered_serve() -> str:
    return (REPO / "slurm" / "serve_qwen.sbatch.tmpl").read_text(encoding="utf-8")


def test_rendered_serve_script_samples_gpus_exactly_once_before_wait(rendered_serve, tmp_run_root):
    """The ARD36 defect: the sampler was never sourced and all 67,105 resource rows were NULL.

    The insert is into the *rendered* script (the ``launch_server.py:703-711`` pattern), the
    sampler starts right after the server does and before the script waits on it, and the EXIT
    trap stops it and captures ``sacct``.
    """
    prepared = FL.prepare_serve_script(rendered_serve, run_root=tmp_run_root, repo=REPO)
    assert prepared.count(FL.SAMPLER_CALL) == 1            # started exactly once
    assert prepared.count(f'source "{REPO / FL.SAMPLER_SCRIPT}"') == 1
    start = prepared.index(FL.SAMPLER_CALL)
    assert prepared.index(FL.ANCHOR_VLLM_PID) < start
    assert start < prepared.index('wait "$VLLM_PID"')
    assert FL.SACCT_CALL in prepared and "sacct -j" in prepared
    assert str(tmp_run_root / FL.RESOURCES_DIR) in prepared
    assert prepared.count("nextdyn_stop_gpu_sampler") == 1
    # idempotent: preparing twice does not double the insert
    assert FL.prepare_serve_script(prepared, run_root=tmp_run_root, repo=REPO) == prepared


def test_a_missing_anchor_is_a_hard_failure_not_a_skipped_insert(tmp_run_root):
    with pytest.raises(ManifestError, match="anchor"):
        FL.inject_gpu_sampler("#!/bin/bash\necho hi\n", run_root=tmp_run_root, repo=REPO)
    with pytest.raises(ManifestError, match="EXIT trap"):
        FL.inject_sacct_capture("#!/bin/bash\nVLLM_PID=$!\n", run_root=tmp_run_root)


def test_the_sampler_script_and_the_reader_agree_on_the_query():
    from agents_scaling.ard import resources as RES

    text = (REPO / "slurm" / FL.SAMPLER_SCRIPT.split("/")[-1]).read_text(encoding="utf-8")
    assert "--query-gpu=" + ",".join(RES.NVIDIA_SMI_FIELDS) in text
    assert FL.SAMPLER_CALL in text


def test_the_keepalive_render_exports_the_pool_offset_after_the_scrub(tmp_run_root, study_cfg,
                                                                     compute_cfg):
    path = FL.render_keepalive(tmp_run_root, Pool.LOW, compute=compute_cfg, cfg=study_cfg,
                               repo=REPO)
    text = path.read_text()
    assert "{" not in text.split("set -euo")[0]
    assert f"export ASYS_PORT_OFFSET={FL.port_offset(compute_cfg, Pool.LOW)}" in text
    assert text.index("common.sh") < text.index("export ASYS_PORT_OFFSET")
    assert f"{FL.FLEET_STOP_FILE}" in text
    assert "--no-requeue" in text and "afterany" in text
    assert "NO_RESUBMIT" in text


def test_the_script_that_reaches_sbatch_is_the_prepared_one(tmp_run_root, compute_cfg):
    """The insert has to happen on the *launch* path, not only in a render test.

    ``fleet._launch`` used to hand the whole job to ``launch_server.submit``, which renders,
    writes and ``sbatch``es in one call with no post-render hook -- so the sampler and the
    ``sacct`` trap were injected nowhere and every ``resources`` row would have been NULL for
    the whole campaign (ARD36's 67,105-row defect).  This drives a real ``tick`` and reads the
    bytes the submitter was handed.
    """
    spooled: list[Path] = []
    targets = FL.parse_spec("nd1-8B:1:ou_bcs_low:1-00:00:00", "a100")
    report = FL.tick(tmp_run_root, Pool.LOW, targets, compute=compute_cfg,
                     submit=lambda path, target, replica: spooled.append(Path(path)) or "4242",
                     runner=lambda cmd: "")
    assert report.launched and spooled
    text = spooled[0].read_text()
    assert text.count(FL.SAMPLER_CALL) == 1
    assert text.count(f'source "{FL.REPO_ROOT / FL.SAMPLER_SCRIPT}"') == 1
    assert FL.SACCT_CALL in text and "sacct -j" in text
    assert str(tmp_run_root / FL.RESOURCES_DIR) in text
    # and the pool's own sub-run-root and port offset travelled with it (plan §3.2)
    assert str(FL.pool_run_root(tmp_run_root, Pool.LOW)) in text
    assert f"export ASYS_PORT_OFFSET={FL.port_offset(compute_cfg, Pool.LOW)}" in text


def test_a_relaunched_replica_does_not_collide_with_its_own_spooled_script(tmp_run_root,
                                                                          compute_cfg):
    """``launch_server`` refuses to overwrite a spooled intent, and the keepalive relaunches the
    same (profile, replica) every time a preemptable job dies -- so each attempt gets its own
    path or the second launch of a replica is a hard failure."""
    target = FL.parse_spec("nd1-8B:1:ou_bcs_low:1-00:00:00", "a100")[0]
    root = FL.pool_run_root(tmp_run_root, Pool.LOW)
    first = FL.serve_script_path(root, target, 0)
    second = FL.serve_script_path(root, target, 0)
    assert first != second
    assert first.parent == second.parent and first.name.startswith("serve_nd1-8B_r0.")


def test_the_serving_profile_is_projected_onto_the_model_size_it_names():
    """A spec entry names a *profile*; ``launch_server`` wants the model size and refuses the
    profile name in its place."""
    assert FL.model_size_of("nd1-8B") == "8B"
    assert FL.model_size_of("nd1-32B") == "32B"
    with pytest.raises(ConfigError, match="no serving profile"):
        FL.model_size_of("nd1-nope")


def test_the_serve_script_never_ships_an_empty_model_contract_path(tmp_run_root, compute_cfg,
                                                                   monkeypatch):
    """The smoke fleet's first launch died in six seconds with the GPU already allocated.

    ``launch_server.render_sbatch`` defaults the contract's path *and* its digest to ambient
    ``ASYS_*`` variables, and the serve template scrubs every ``ASYS_*`` name before it runs --
    so a launch from a shell that had exported the digest but not the path rendered
    ``ASYS_MODEL_CONTRACT=""`` next to a perfectly good ``ASYS_MODEL_CONTRACT_SHA256``.  The
    in-job registrar resolved the empty path to the repo root:
    ``ModelContractError: missing regular model contract``.  Both halves are rendered in now,
    from one read of one file, and this drives the real launch path with the environment
    deliberately unset.
    """
    monkeypatch.delenv(FL.MODEL_CONTRACT_ENV, raising=False)
    monkeypatch.delenv(FL.MODEL_CONTRACT_SHA_ENV, raising=False)
    spooled: list[Path] = []
    targets = FL.parse_spec("nd1-8B:1:ou_bcs_low:1-00:00:00", "a100")
    FL.tick(tmp_run_root, Pool.LOW, targets, compute=compute_cfg,
            submit=lambda path, target, replica: spooled.append(Path(path)) or "4242",
            runner=lambda cmd: "")
    text = spooled[0].read_text()
    contract = FL.model_contract_path()
    assert f'export {FL.MODEL_CONTRACT_ENV}="{contract}"' in text
    assert f'{FL.MODEL_CONTRACT_ENV}=""' not in text
    # the digest travels with the path, and it is the digest *of that file*
    import hashlib

    digest = hashlib.sha256(contract.read_bytes()).hexdigest()
    assert f'{FL.MODEL_CONTRACT_SHA_ENV}="{digest}"' in text


def test_a_digest_inherited_from_the_environment_may_not_disagree_with_the_file(monkeypatch):
    """Naming a different file in the environment than the one being shipped is a hard failure.

    Preferring the inherited digest would ship a script that fails verification inside the job;
    preferring the file's would silently ignore an environment that thinks it is pinning
    something.  Neither is a resolution, so the disagreement is reported.
    """
    monkeypatch.setenv(FL.MODEL_CONTRACT_SHA_ENV, "0" * 64)
    with pytest.raises(ManifestError, match="disagrees with"):
        FL.model_contract_render_args()


def test_a_missing_model_contract_is_refused_before_a_gpu_is_allocated(tmp_path, monkeypatch):
    monkeypatch.setenv(FL.MODEL_CONTRACT_ENV, str(tmp_path / "gone.json"))
    with pytest.raises(ManifestError, match="does not exist"):
        FL.model_contract_path()


def test_a_replica_registered_under_its_profile_name_is_counted(tmp_run_root, compute_cfg,
                                                                monkeypatch):
    """A live pool counted zero, because the lookup used the wrong one of two names.

    ``registry.servers_dir`` names each directory for the **serving profile** (``nd1-8B``),
    while ``compute.serving.max_num_seqs_by_gpu`` is keyed by model *size* (``8B``).  Counting
    endpoints under the size meant a healthy fleet reported no live replicas -- and
    ``gates launch --required-replicas`` reads this number, so the launch gate would have
    failed against a fleet that was up.
    """
    root = FL.pool_run_root(tmp_run_root, Pool.DEADLINE)
    (root / FL.SERVERS_DIR / "nd1-8B").mkdir(parents=True)
    (root / FL.SERVERS_DIR / "nd1-32B").mkdir(parents=True)
    assert FL._served_names(root) == ("nd1-32B", "nd1-8B")
    monkeypatch.setattr(FL.REG, "list_live_servers", lambda r, name, **k: [object()])
    assert FL.live_endpoint_count(tmp_run_root, compute=compute_cfg, pool=Pool.DEADLINE) == 2
    # a pool that nothing registered in is zero, not an error
    assert FL.live_endpoint_count(tmp_run_root, compute=compute_cfg, pool=Pool.AUDIT) == 0
    assert FL._served_names(FL.pool_run_root(tmp_run_root, Pool.LOW)) == ()


def test_a_registry_directory_outlives_the_replica_that_made_it(tmp_run_root, monkeypatch):
    """A preempted replica leaves its record behind, and the pool must not look live.

    ``list_live_servers`` never deletes a shared registry record on a failed observation --
    keepalive owns cleanup -- so after the `low` pool's preemptable replica was preempted, the
    directory still named the dead host.  A check that only listed directory names called the
    pool live, dispatched into it, and took ``exit 2`` from every cell.  Liveness is a probe.
    """
    root = FL.pool_run_root(tmp_run_root, Pool.LOW)
    (root / FL.SERVERS_DIR / "nd1-8B").mkdir(parents=True)
    (root / FL.SERVERS_DIR / "nd1-32B").mkdir(parents=True)
    monkeypatch.setattr(FL.REG, "list_live_servers",
                        lambda r, name, **k: [object()] if name == "nd1-32B" else [])
    assert FL.live_endpoints_by_lane(tmp_run_root, Pool.LOW) == {"32B": 1}
    # the 8B directory is there and names nothing live: the lane is absent, not zero-valued,
    # so `.get(lane)` is falsey for a caller asking "can I dispatch an 8B cell here?"
    assert not FL.live_endpoints_by_lane(tmp_run_root, Pool.LOW).get("8B")
    assert FL.live_endpoints_by_lane(tmp_run_root, Pool.AUDIT) == {}


#: One node of each shape ``sinfo -O "NodeHost:20,Gres:60,GresUsed:60,StateLong:20"`` emits:
#: fully busy, partly busy, idle, and drained.  ``GresUsed`` carries an ``(IDX:…)`` suffix
#: whenever anything is allocated, which is the whole difficulty.
SINFO_FIXTURE = (
    "node1802   gpu:h100:4   gpu:h100:4(IDX:0-3)   mixed-\n"
    "node1803   gpu:h100:4   gpu:h100:2(IDX:1-2)   mixed-\n"
    "node1804   gpu:h100:4   gpu:h100:0(IDX:N/A)   idle\n"
    "node3804   gpu:h100:4   gpu:h100:4(IDX:0-3)   drained\n"
)


def test_an_allocated_gpu_counts_as_used(monkeypatch):
    """``used`` was always zero, so a full partition looked entirely idle.

    ``Gres`` is ``gpu:h100:4`` and ``GresUsed`` is ``gpu:h100:4(IDX:0-3)``.  Reading the count
    as the *last* colon-separated piece gives ``4`` for the first and ``0-3)`` for the second,
    which is not a digit and fell through to 0.  ``scale_to_idle`` then saw every busy GPU as
    free, and `fleet up` would have launched a whole spec on top of other users' jobs -- the
    exact opposite of D20, whose rule is to preempt nobody.
    """
    census = FL.free_gpus("p", "h100", runner=lambda cmd: SINFO_FIXTURE)
    assert (census.total, census.used) == (12, 6)        # the drained node counts for neither
    assert census.drained == ("node3804",)
    # and the scaler now refuses to fill more than the six that are actually free
    scaled = FL.scale_to_idle(FL.parse_spec("nd1-8B:9:p:1-00:00:00", "h100"),
                              idle={("p", "h100"): census.total - census.used},
                              gpus_per_replica={"nd1-8B": 1})
    assert sum(t.count for t in scaled) == 6


def test_the_gres_count_is_the_one_ard36_shipped():
    """Parity with the vendored source (``scripts/ard36_preflight.py`` ``_free_gpus.count``).

    ARD36 read field ``[2]``; the nextdyn copy was transcribed as ``[-1]``.  The VENDORED table
    exists so a transcription that changes behaviour is caught here rather than in a fleet that
    quietly stops respecting idle capacity, so this runs the original text against the copy.
    """
    def ard36_count(field: str, gpu_type: str) -> int:
        for chunk in field.split(","):
            if chunk.startswith(f"gpu:{gpu_type}:"):
                n = chunk.split(":")[2].split("(")[0]
                return int(n) if n.isdigit() else 0
        return 0

    fields = ["gpu:h100:4", "gpu:h100:4(IDX:0-3)", "gpu:h100:0(IDX:N/A)", "gpu:a100:8(IDX:0-7)",
              "gpu:h100:2(IDX:1-2),gpu:a100:1(IDX:0)", "(null)", ""]
    for field in fields:
        for gpu_type in ("h100", "a100"):
            assert FL._gres_count(field, gpu_type) == ard36_count(field, gpu_type), field


#: Four free GPUs of one type, spread one per node -- room for four TP1 replicas and no TP2.
SCATTERED = (
    "node01   gpu:h200:4   gpu:h200:3(IDX:0-2)   mixed\n"
    "node02   gpu:h200:4   gpu:h200:3(IDX:0-2)   mixed\n"
    "node03   gpu:h200:4   gpu:h200:3(IDX:0-2)   mixed\n"
    "node04   gpu:h200:4   gpu:h200:3(IDX:0-2)   mixed\n"
)
#: The same four free GPUs, two nodes with two each -- room for two TP2 replicas.
PAIRED = (
    "node01   gpu:h200:4   gpu:h200:2(IDX:0-1)   mixed\n"
    "node02   gpu:h200:4   gpu:h200:2(IDX:0-1)   mixed\n"
)


def test_a_wide_replica_needs_its_gpus_on_one_node(monkeypatch):
    """"Four GPUs are free" does not mean "two TP2 replicas fit".

    A 32B replica is TP2 and its two GPUs must be on a single node, so a partition total is the
    wrong question for it.  Measured on this cluster while the smoke was queued: `ou_bcs_low`
    H100 had four free and room for exactly one TP2; `mit_preemptable` H200 had seven free and
    room for three.  Scaling on the total would have launched the surplus into
    PENDING/Resources, where it serves nothing while the lane it was meant to fill stays empty
    -- and `fleet up` would have reported them as launched.
    """
    scattered = FL.free_gpus("p", "h200", runner=lambda cmd: SCATTERED)
    paired = FL.free_gpus("p", "h200", runner=lambda cmd: PAIRED)
    assert scattered.free == paired.free == 4                  # the same idle GPU count
    assert (scattered.placements(1), scattered.placements(2)) == (4, 0)
    assert (paired.placements(1), paired.placements(2)) == (4, 2)

    def scaled(census, profile, width):
        targets = FL.parse_spec(f"{profile}:4:p:1-00:00:00", "h200")
        out = FL.scale_to_idle(targets, idle={("p", "h200"): census},
                               gpus_per_replica={profile: width})
        return sum(t.count for t in out)

    assert scaled(scattered, "nd1-32B", 2) == 0, "a TP2 replica cannot straddle two nodes"
    assert scaled(paired, "nd1-32B", 2) == 2
    assert scaled(scattered, "nd1-8B", 1) == 4, "TP1 is happy with scattered GPUs"


def test_the_scaler_still_accepts_a_bare_idle_count():
    """An older caller (or a stub that only knows totals) keeps its previous meaning."""
    targets = FL.parse_spec("nd1-32B:4:p:1-00:00:00", "h200")
    out = FL.scale_to_idle(targets, idle={("p", "h200"): 4}, gpus_per_replica={"nd1-32B": 2})
    assert sum(t.count for t in out) == 2
    # and a pool the spec does not name is left alone rather than scaled to nothing
    other = FL.scale_to_idle(targets, idle={}, gpus_per_replica={"nd1-32B": 2})
    assert sum(t.count for t in other) == 4
