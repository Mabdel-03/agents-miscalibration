from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import schema5_bootstrap_watchdog as watchdog


def test_forced_ssh_uses_absolute_binary_null_config_and_minimal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(_config(tmp_path).read_text(encoding="utf-8"))
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["environment"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"passed": true}\n', stderr=""
        )

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    assert watchdog._ssh(config, "status") == {"passed": True}
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert argv[:3] == ["/usr/bin/ssh", "-F", "/dev/null"]
    assert observed["environment"] == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }


def _config(tmp_path: Path) -> Path:
    identity = tmp_path / "id"
    identity.write_text("private\n", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("cluster key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    state = tmp_path / "state"
    state.mkdir()
    value = {
        "schema_version": 1,
        "protocol": watchdog.PROTOCOL,
        "release_git_commit": "a" * 40,
        "release_tag_object": "b" * 40,
        "chain_id": "c" * 64,
        "chain_manifest_sha256": "d" * 64,
        "anchor_submission_receipt_id": "e" * 64,
        "anchor_submission_receipt_sha256": "f" * 64,
        "harness_environment_binding": {
            "manifest_sha256": "9" * 64,
        },
        "materialization_pilot": {
            "marker": "/sealed/PILOT_COMPLETE.json",
            "marker_sha256": "8" * 64,
            "pilot_id": "7" * 64,
            "pilot_root": "/sealed/pilot",
            "release_worktree": "/sealed/pilot/release-worktree",
            "release_bundle": "/sealed/pilot/release",
            "source_tree_sha256": "6" * 64,
            "release_bundle_id": "5" * 64,
            "release_identity_sha256": "4" * 64,
            "release_completion_sha256": "3" * 64,
        },
        "remote": {
            "host": "cluster",
            "user": "watchdog",
            "identity_file": str(identity),
            "known_hosts_file": str(known_hosts),
        },
        "state_root": str(state),
        "observation_gap_seconds": 60,
        "timer_seconds": 300,
    }
    path = tmp_path / "config.json"
    path.write_bytes(watchdog._canonical(value))
    path.chmod(0o444)
    return path


def _status(
    *,
    observed_at: float,
    cancelled: bool,
    handoff: bool = False,
    generation: int = 0,
    anchor_launch_id: str | None = None,
    release_required: bool = False,
    rearm_required: bool = False,
    receipt_id: str = "e" * 64,
    receipt_sha256: str = "f" * 64,
):
    root_name = "source_checkout" if generation == 0 else "repair-root"
    root_job_id = str(1000 + generation)
    root_held = release_required or rearm_required
    current_job_ids = [
        root_job_id,
        *[
            str(10_000 + generation * 100 + index)
            for index in range(1, watchdog.RECOVERY_JOB_COUNT)
        ],
    ]
    prior_generation_job_ids = [
        [
            str(50_000 + prior_generation * 100 + index)
            for index in range(watchdog.RECOVERY_JOB_COUNT)
        ]
        for prior_generation in range(generation)
    ]
    prior_job_ids = [
        job_id
        for generation_ids in prior_generation_job_ids
        for job_id in generation_ids
    ]
    bound_job_ids = sorted(
        {*prior_job_ids, *current_job_ids}, key=int
    )
    jobs = []
    for index, job_id in enumerate(current_job_ids):
        name = root_name if index == 0 else f"job-{index:02d}"
        jobs.append(
            {
                "name": name,
                "job_id": job_id,
                "comment": (
                    "asys:s5-recovery-v1.2-r13:"
                    f"{'c' * 64}:g{generation:04d}:{name}:"
                    f"{'1' * 16}"
                ),
                "job_name": f"asys-r13-{name}",
                "state": (
                    "CANCELLED"
                    if cancelled
                    else ("PENDING" if root_held else "RUNNING")
                ),
                "active": not cancelled,
                "script_sha256": "1" * 64,
                "submit_line_sha256": "2" * 64,
                "submit_line_exact": True,
                "scontrol_command": f"/sealed/{name}.sbatch",
                "scontrol_requeue": 0,
                "spooled_script_sha256": "1" * 64,
                "spooled_script_exact_match": True,
            }
        )
    descendant_armed = (
        f"/sealed/g{generation:04d}/BOOTSTRAP_DESCENDANT_ARMED.json"
        if generation > 0 and release_required
        else None
    )
    value = {
        "schema_version": 1,
        "protocol": watchdog.STATUS_PROTOCOL,
        "passed": True,
        "observed_at_timestamp": observed_at,
        "release_git_commit": "a" * 40,
        "release_tag_object": "b" * 40,
        "chain_id": "c" * 64,
        "chain_manifest": "/sealed/CHAIN_MANIFEST.json",
        "chain_manifest_sha256": "d" * 64,
        "anchor_submission_receipt_id": "e" * 64,
        "anchor_submission_receipt_sha256": "f" * 64,
        "descendant_chain_validated": True,
        "submission_receipt": (
            f"/sealed/g{generation:04d}/SUBMISSION.json"
        ),
        "submission_receipt_id": receipt_id,
        "submission_receipt_sha256": receipt_sha256,
        "generation_provenance": (
            f"/sealed/g{generation:04d}/"
            "BOOTSTRAP_GENERATION_PROVENANCE.json"
        ),
        "generation_provenance_sha256": "6" * 64,
        "generation_provenance_id": "5" * 64,
        "anchor_submission_receipt": (
            "/sealed/SUBMISSION_COMPLETE.json"
        ),
        "repair_generation": generation,
        "root_name": root_name,
        "root_job_id": root_job_id,
        "root_names": [root_name],
        "root_job_ids": [root_job_id],
        "roots_held": root_held,
        "root_held": root_held,
        "squeue_complete": True,
        "sacct_complete": True,
        "namespace_scan_complete": True,
        "namespace_lineage_receipt_ids": (
            ["e" * 64]
            if generation == 0
            else ["e" * 64, receipt_id]
        ),
        "namespace_lineage_job_ids": [
            *prior_generation_job_ids,
            current_job_ids,
        ],
        "namespace_bound_job_ids": bound_job_ids,
        "job_count": watchdog.RECOVERY_JOB_COUNT,
        "jobs": jobs,
        "ambiguous_jobs": 0,
        "watchdog_scientific_jobs_submitted": 0,
        "recovery_namespace_cancelled": cancelled,
        "anchor_launch_authorized": anchor_launch_id is not None,
        "anchor_launch_id": anchor_launch_id,
        "anchor_armed_authorized": True,
        "anchor_armed_id": "4" * 64,
        "anchor_release_intent_authorized": False,
        "anchor_release_intent_sha256": None,
        "descendant_armed": descendant_armed,
        "descendant_armed_id": (
            "3" * 64 if descendant_armed is not None else None
        ),
        "descendant_rearm_required": rearm_required,
        "descendant_release_required": release_required,
        "handoff_complete": handoff,
    }
    value["observation_id"] = __import__("hashlib").sha256(
        watchdog._canonical(value)
    ).hexdigest()
    value["observation_artifact"] = (
        f"/sealed/observation-{int(observed_at)}.json"
    )
    value["observation_artifact_sha256"] = "9" * 64
    return value


def _rehash_status(value: dict) -> None:
    candidate = dict(value)
    candidate.pop("observation_id", None)
    candidate.pop("observation_artifact", None)
    candidate.pop("observation_artifact_sha256", None)
    value["observation_id"] = __import__("hashlib").sha256(
        watchdog._canonical(candidate)
    ).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "skipped_receipt",
        "lineage_job_union",
        "current_job_binding",
        "unknown_field",
    ],
)
def test_bootstrap_status_requires_exact_ancestry_and_bound_job_ids(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = watchdog.load_config(_config(tmp_path))
    value = _status(
        observed_at=900.0,
        cancelled=False,
        generation=1,
        receipt_id="1" * 64,
        receipt_sha256="2" * 64,
    )
    if mutation == "skipped_receipt":
        value["namespace_lineage_receipt_ids"] = ["1" * 64]
    elif mutation == "lineage_job_union":
        value["namespace_lineage_job_ids"][0][0] = "999999"
    elif mutation == "current_job_binding":
        value["jobs"][0]["job_id"] = "999998"
    else:
        value["unbound"] = True
    _rehash_status(value)

    with pytest.raises(
        watchdog.BootstrapWatchdogError,
        match=(
            "incomplete or ambiguous"
            "|current-job binding is invalid"
            "|roots are not bound"
        ),
    ):
        watchdog._validate_status(value, config)  # noqa: SLF001


def test_external_bootstrap_watchdog_repairs_only_after_two_complete_cuts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    selectors: list[str] = []
    statuses = iter(
        [
            _status(observed_at=1_000.0, cancelled=True),
            _status(observed_at=1_060.0, cancelled=True),
        ]
    )

    def fake_ssh(_config, selector):
        selectors.append(selector)
        if selector.endswith(" status"):
            return next(statuses)
        return {
            "passed": True,
            "status": "bootstrap_repaired_held",
            "repair_generation": 1,
            "submission_receipt": "/sealed/g0001/SUBMISSION.json",
            "submission_receipt_id": "1" * 64,
            "submission_receipt_sha256": "2" * 64,
            "generation_provenance": "/sealed/g0001/provenance.json",
            "generation_provenance_sha256": "3" * 64,
            "generation_provenance_id": "4" * 64,
            "parent_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_sha256": "f" * 64,
            "anchor_launch_authorized": False,
            "anchor_launch_id": None,
            "root_name": "repair-root",
            "root_job_id": "1001",
            "root_names": ["repair-root"],
            "root_job_ids": ["1001"],
            "roots_held": True,
            "root_held": True,
            "root_released": False,
            "root_release_id": None,
            "launch_id": None,
            "watchdog_scientific_jobs_submitted": 0,
        }

    monkeypatch.setattr(watchdog, "_ssh", fake_ssh)
    times = iter([1_000.0, 1_060.0])
    result = watchdog.run_once(
        config,
        sleeper=lambda seconds: None,
        clock=lambda: next(times),
    )

    assert result["action"] == "bootstrap_repair"
    assert selectors == [
        "schema5-bootstrap-watchdog status",
        "schema5-bootstrap-watchdog status",
        "schema5-bootstrap-watchdog repair",
    ]
    assert all("release" not in selector for selector in selectors)
    assert result["recovery_root_release_performed"] is False
    heartbeat = json.loads(
        (
            tmp_path / "state" / "BOOTSTRAP_WATCHDOG_HEARTBEAT.json"
        ).read_text(encoding="utf-8")
    )
    assert heartbeat["heartbeat_id"] == result["heartbeat_id"]


def test_external_bootstrap_watchdog_noops_after_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    selectors: list[str] = []
    statuses = iter(
        [
            _status(
                observed_at=2_000.0,
                cancelled=False,
                handoff=True,
            ),
            _status(
                observed_at=2_060.0,
                cancelled=False,
                handoff=True,
            ),
        ]
    )

    def fake_ssh(_config, selector):
        selectors.append(selector)
        return next(statuses)

    monkeypatch.setattr(watchdog, "_ssh", fake_ssh)
    times = iter([2_000.0, 2_060.0])
    result = watchdog.run_once(
        config,
        sleeper=lambda seconds: None,
        clock=lambda: next(times),
    )

    assert result["action"] == "handoff_complete"
    assert selectors == [
        "schema5-bootstrap-watchdog status",
        "schema5-bootstrap-watchdog status",
    ]


def test_external_bootstrap_watchdog_reconstructs_postlaunch_descendant_held(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    launch_id = "a" * 64
    statuses = iter(
        [
            _status(
                observed_at=3_000.0,
                cancelled=True,
                anchor_launch_id=launch_id,
            ),
            _status(
                observed_at=3_060.0,
                cancelled=True,
                anchor_launch_id=launch_id,
            ),
        ]
    )

    def fake_ssh(_config, selector):
        if selector.endswith(" status"):
            return next(statuses)
        return {
            "passed": True,
            "status": "bootstrap_repaired_held",
            "repair_generation": 1,
            "submission_receipt": "/sealed/g0001/SUBMISSION.json",
            "submission_receipt_id": "1" * 64,
            "submission_receipt_sha256": "2" * 64,
            "generation_provenance": (
                "/sealed/g0001/"
                "BOOTSTRAP_GENERATION_PROVENANCE.json"
            ),
            "generation_provenance_sha256": "6" * 64,
            "generation_provenance_id": "5" * 64,
            "parent_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_sha256": "f" * 64,
            "anchor_launch_authorized": True,
            "anchor_launch_id": launch_id,
            "root_name": "repair-root",
            "root_job_id": "1001",
            "root_names": ["repair-root"],
            "root_job_ids": ["1001"],
            "roots_held": True,
            "root_held": True,
            "root_released": False,
            "root_release_id": None,
            "launch_id": None,
            "watchdog_scientific_jobs_submitted": 0,
        }

    monkeypatch.setattr(watchdog, "_ssh", fake_ssh)
    times = iter([3_000.0, 3_060.0])
    result = watchdog.run_once(
        config,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
    )

    assert result["action"] == "bootstrap_repair"
    assert result["recovery_root_release_performed"] is False
    assert result["production_control_mutated"] is False
    assert result["scientific_jobs_submitted"] == 0


def test_external_bootstrap_watchdog_reconciles_held_postlaunch_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    launch_id = "a" * 64
    receipt_id = "1" * 64
    receipt_sha = "2" * 64
    statuses = iter(
        [
            _status(
                observed_at=4_000.0,
                cancelled=False,
                generation=1,
                anchor_launch_id=launch_id,
                release_required=True,
                receipt_id=receipt_id,
                receipt_sha256=receipt_sha,
            ),
            _status(
                observed_at=4_060.0,
                cancelled=False,
                generation=1,
                anchor_launch_id=launch_id,
                release_required=True,
                receipt_id=receipt_id,
                receipt_sha256=receipt_sha,
            ),
        ]
    )

    def fake_ssh(_config, selector):
        if selector.endswith(" status"):
            return next(statuses)
        return {
            "passed": True,
            "status": "bootstrap_repair_reconciled",
            "repair_generation": 1,
            "submission_receipt": "/sealed/g0001/SUBMISSION.json",
            "submission_receipt_id": receipt_id,
            "submission_receipt_sha256": receipt_sha,
            "generation_provenance": (
                "/sealed/g0001/"
                "BOOTSTRAP_GENERATION_PROVENANCE.json"
            ),
            "generation_provenance_sha256": "6" * 64,
            "generation_provenance_id": "5" * 64,
            "parent_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_sha256": "f" * 64,
            "anchor_launch_authorized": True,
            "anchor_launch_id": launch_id,
            "root_name": "repair-root",
            "root_job_id": "1001",
            "root_names": ["repair-root"],
            "root_job_ids": ["1001"],
            "roots_held": False,
            "root_held": False,
            "root_released": True,
            "root_release_id": "3" * 64,
            "launch_id": "4" * 64,
            "watchdog_scientific_jobs_submitted": 0,
        }

    monkeypatch.setattr(watchdog, "_ssh", fake_ssh)
    times = iter([4_000.0, 4_060.0])
    result = watchdog.run_once(
        config,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
    )
    assert result["action"] == "bootstrap_release_reconcile"
    assert result["recovery_root_release_performed"] is True


def test_external_bootstrap_watchdog_rearms_and_launches_authorized_descendant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    launch_id = "a" * 64
    receipt_id = "1" * 64
    receipt_sha = "2" * 64
    statuses = iter(
        [
            _status(
                observed_at=4_500.0,
                cancelled=False,
                generation=1,
                anchor_launch_id=launch_id,
                rearm_required=True,
                receipt_id=receipt_id,
                receipt_sha256=receipt_sha,
            ),
            _status(
                observed_at=4_560.0,
                cancelled=False,
                generation=1,
                anchor_launch_id=launch_id,
                rearm_required=True,
                receipt_id=receipt_id,
                receipt_sha256=receipt_sha,
            ),
        ]
    )

    def fake_ssh(_config, selector):
        if selector.endswith(" status"):
            return next(statuses)
        return {
            "passed": True,
            "status": "bootstrap_descendant_rearmed_launched",
            "repair_generation": 1,
            "submission_receipt": "/sealed/g0001/SUBMISSION.json",
            "submission_receipt_id": receipt_id,
            "submission_receipt_sha256": receipt_sha,
            "generation_provenance": (
                "/sealed/g0001/"
                "BOOTSTRAP_GENERATION_PROVENANCE.json"
            ),
            "generation_provenance_sha256": "6" * 64,
            "generation_provenance_id": "5" * 64,
            "parent_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_id": "e" * 64,
            "anchor_submission_receipt_sha256": "f" * 64,
            "anchor_launch_authorized": True,
            "anchor_launch_id": launch_id,
            "descendant_armed": (
                "/sealed/g0001/BOOTSTRAP_DESCENDANT_ARMED.json"
            ),
            "descendant_armed_id": "3" * 64,
            "root_name": "repair-root",
            "root_job_id": "1001",
            "root_names": ["repair-root"],
            "root_job_ids": ["1001"],
            "roots_held": False,
            "root_held": False,
            "root_released": True,
            "root_release_id": "7" * 64,
            "launch_id": "8" * 64,
            "watchdog_scientific_jobs_submitted": 0,
        }

    monkeypatch.setattr(watchdog, "_ssh", fake_ssh)
    times = iter([4_500.0, 4_560.0])
    result = watchdog.run_once(
        config,
        sleeper=lambda _seconds: None,
        clock=lambda: next(times),
    )

    assert result["action"] == "bootstrap_descendant_rearm"
    assert result["recovery_root_release_performed"] is True
    assert result["production_control_mutated"] is False


def test_external_bootstrap_watchdog_rejects_mixed_descendant_cuts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    statuses = iter(
        [
            _status(
                observed_at=5_000.0,
                cancelled=True,
                generation=1,
                receipt_id="1" * 64,
            ),
            _status(
                observed_at=5_060.0,
                cancelled=True,
                generation=1,
                receipt_id="2" * 64,
            ),
        ]
    )
    monkeypatch.setattr(
        watchdog,
        "_ssh",
        lambda _config, _selector: next(statuses),
    )
    times = iter([5_000.0, 5_060.0])
    with pytest.raises(
        watchdog.BootstrapWatchdogError,
        match="not at least 60 seconds apart",
    ):
        watchdog.run_once(
            config,
            sleeper=lambda _seconds: None,
            clock=lambda: next(times),
        )
