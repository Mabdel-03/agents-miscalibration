"""Safety and crash-boundary tests for the schema-5 v1.1-r1 recovery DAG."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Sequence

import pytest

from scripts import render_schema5_recovery_chain as chain


COMMIT = "1" * 40
REAL_R1_MUTATION_REJECTION = chain._reject_retired_r1_mutation


@pytest.fixture(autouse=True)
def _historical_r1_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise frozen r1 mechanics without reopening production entrypoints."""

    monkeypatch.setattr(
        chain, "_reject_retired_r1_mutation", lambda _recovery_root: None
    )


def _make_paths(tmp_path: Path) -> chain.RecoveryPaths:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    for run_id in chain.LEGACY_RUN_IDS:
        (results / run_id).mkdir()
    (recovery / "pre_repair_inventory").mkdir()
    hf_home = tmp_path / "hf"
    harness = tmp_path / "source-harness"
    serving = tmp_path / "source-serving"
    for directory in (hf_home, harness, serving):
        directory.mkdir()
    (harness / "bin").mkdir()
    dev_python = harness / "bin" / "python"
    conda = tmp_path / "conda"
    for executable in (dev_python, conda):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    return chain.recovery_paths(
        repository=repository,
        results_root=results,
        recovery_root=recovery,
        hf_home=hf_home,
        dev_python=dev_python,
        source_harness=harness,
        source_serving=serving,
        conda_executable=conda,
    )


@pytest.fixture
def rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> chain.RecoveryPaths:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )
    report = chain.render_chain(paths, slurm_user="tester", apply=True)
    assert report["status"] == "complete"
    return paths


class FakeSlurm:
    def __init__(self) -> None:
        self.jobs: list[dict[str, str]] = []
        self.calls: list[list[str]] = []
        self.next_job_id = 7000
        self.hide_jobs = False
        self.hide_generation_jobs = False
        self.crash_after_accept_once = False
        self.reject_sbatch_once = False
        self.allow_dependency_policy = True
        self.blank_sacct_comments = False

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[:3] == ["scontrol", "show", "config"]:
            value = "kill_invalid_depend" if self.allow_dependency_policy else "(null)"
            return subprocess.CompletedProcess(
                args, 0, f"DependencyParameters = {value}\n", ""
            )
        if args and args[0] in {"squeue", "sacct"}:
            visible = [] if self.hide_jobs else self.jobs
            if self.hide_generation_jobs:
                visible = [job for job in visible if ":g0001:" not in job["comment"]]
            if args[0] == "squeue":
                active = [
                    job
                    for job in visible
                    if job["state"] in chain._ACTIVE_SLURM_STATES
                ]
                output_format = args[args.index("-o") + 1]
                if output_format == "%i|%T|%k|%j":
                    rows = "".join(
                        f"{job['job_id']}|{job['state']}|{job['comment']}|{job['job_name']}\n"
                        for job in active
                    )
                else:
                    rows = "".join(
                        f"{job['job_id']}|{job['comment']}|{job['job_name']}|{job['state']}\n"
                        for job in active
                    )
            elif "-j" in args:
                rows = "".join(
                    f"{job['job_id']}|{job['state']}|0:0|"
                    f"{'(null)' if self.blank_sacct_comments else job['comment']}|"
                    f"{job['job_name']}|{job['submit_line']}\n"
                    for job in visible
                )
            else:
                rows = "".join(
                    f"{job['job_id']}|"
                    f"{'(null)' if self.blank_sacct_comments else job['comment']}|"
                    f"{job['job_name']}|{job['state']}|{job['submit_line']}\n"
                    for job in visible
                )
            return subprocess.CompletedProcess(args, 0, rows, "")
        if args and args[0] == "sbatch":
            if self.reject_sbatch_once:
                self.reject_sbatch_once = False
                return subprocess.CompletedProcess(
                    args, 1, "", "simulated ambiguous sbatch failure"
                )
            comment = next(item.split("=", 1)[1] for item in args if item.startswith("--comment="))
            script = Path(args[-1])
            job_name = next(
                line.split("=", 1)[1]
                for line in script.read_text(encoding="utf-8").splitlines()
                if line.startswith("#SBATCH --job-name=")
            )
            job_id = str(self.next_job_id)
            self.next_job_id += 1
            self.jobs.append(
                {
                    "job_id": job_id,
                    "comment": comment,
                    "job_name": job_name,
                    "state": "PENDING",
                    "submit_line": shlex.join(args),
                }
            )
            if self.crash_after_accept_once:
                self.crash_after_accept_once = False
                raise RuntimeError("simulated death after scheduler acceptance")
            return subprocess.CompletedProcess(args, 0, job_id + ";cluster\n", "")
        raise AssertionError(f"unexpected fake Slurm command: {args}")

    @property
    def sbatch_calls(self) -> list[list[str]]:
        return [args for args in self.calls if args and args[0] == "sbatch"]

    def set_state(self, job_name: str, state: str) -> None:
        matches = [job for job in self.jobs if job["job_name"] == job_name]
        assert len(matches) == 1
        matches[0]["state"] = state

    def set_latest_state(self, job_name: str, state: str) -> None:
        matches = [job for job in self.jobs if job["job_name"] == job_name]
        assert matches
        matches[-1]["state"] = state

    def complete_all(self) -> None:
        for job in self.jobs:
            job["state"] = "COMPLETED"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_r1_failure_envelope(recovery_root: Path) -> Path:
    path = recovery_root / chain.R1_FAILURE_ENVELOPE_NAME
    path.write_text(
        json.dumps(
            {
                "protocol": "schema5-recovery-chain-failure-v1",
                "classification": "requires_superseding_release",
                "retry_same_generation": False,
                "superseded_by": "sweep-recovery-schema5-v1.2",
                "failure_id": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path


def test_r1_is_permanently_blocked_on_fresh_and_evidenced_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chain, "_reject_retired_r1_mutation", REAL_R1_MUTATION_REJECTION
    )
    fresh = _make_paths(tmp_path / "fresh")
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda _repository: pytest.fail("retired render must fail before Git access"),
    )
    with pytest.raises(chain.ChainError, match="permanently disabled"):
        chain.render_chain(fresh, slurm_user="tester")

    _publish_r1_failure_envelope(fresh.recovery_root)
    with pytest.raises(chain.ChainError, match="permanently disabled"):
        chain.render_chain(fresh, slurm_user="tester")

    rendered = _make_paths(tmp_path / "rendered")
    with monkeypatch.context() as historical:
        historical.setattr(
            chain, "_reject_retired_r1_mutation", lambda _root: None
        )
        historical.setattr(
            chain,
            "verify_release_tag",
            lambda _repository: {
                "release_tag": chain.RELEASE_TAG,
                "git_commit": COMMIT,
            },
        )
        chain.render_chain(rendered, slurm_user="tester", apply=True)
    with pytest.raises(chain.ChainError, match="permanently disabled"):
        chain.submit_chain(rendered.chain_manifest)
    with pytest.raises(chain.ChainError, match="permanently disabled"):
        chain.repair_chain(rendered.chain_manifest)


def test_render_dry_run_is_read_only_and_reports_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )
    before = sorted(str(path.relative_to(paths.recovery_root)) for path in paths.recovery_root.rglob("*"))
    report = chain.render_chain(paths, slurm_user="tester", apply=False)
    after = sorted(str(path.relative_to(paths.recovery_root)) for path in paths.recovery_root.rglob("*"))
    assert report["status"] == "dry_run"
    assert report["job_count"] == 20
    assert "pre_repair_snapshot_verify" in report["first_legacy_mutation_ancestors"]
    assert "release_freeze" in report["first_legacy_mutation_ancestors"]
    assert before == after


def test_r1_render_uses_fresh_operational_paths_and_preserves_failed_v11(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_paths(tmp_path)
    recovery = paths.recovery_root
    old_files = [
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_1.json",
        recovery / ".RECOVERY_CHAIN_SCHEMA5_V1_1.submission.json",
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_1_SUBMISSION.json",
        recovery / ".RECOVERY_CHAIN_SCHEMA5_V1_1.render.lock",
        recovery / ".RECOVERY_CHAIN_SCHEMA5_V1_1.submit.lock",
        recovery / "release_source_checkout_v1_1" / "old-checkout.txt",
        recovery / "jobs" / "schema5-v1.1" / "old-job.sbatch",
        recovery / "logs" / "schema5-v1.1" / "old-job.out",
        recovery / "recovery_chain_repairs" / "old-repair.json",
    ]
    for index, path in enumerate(old_files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"cancelled-v1.1-evidence-{index}\n", encoding="utf-8")
    before = {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mode)
        for path in old_files
    }
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )

    report = chain.render_chain(paths, slurm_user="tester", apply=True)

    assert report["status"] == "complete"
    assert chain.RELEASE_ID == "sweep-recovery-schema5-v1.1"
    assert chain.RELEASE_TAG == "sweep-recovery-schema5-v1.1-r1"
    assert paths.source_checkout == recovery / "release_source_checkout_v1_1_r1"
    assert paths.jobs_root == recovery / "jobs" / "schema5-v1.1-r1"
    assert paths.logs_root == recovery / "logs" / "schema5-v1.1-r1"
    assert paths.chain_manifest == recovery / "RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
    assert chain.SUBMISSION_JOURNAL_NAME == (
        ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submission.json"
    )
    assert chain.SUBMISSION_RECEIPT_NAME == (
        "RECOVERY_CHAIN_SCHEMA5_V1_1_R1_SUBMISSION.json"
    )
    assert chain.RENDER_LOCK_NAME == ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.render.lock"
    assert chain.SUBMISSION_LOCK_NAME == ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submit.lock"
    assert chain.REPAIR_ROOT_NAME == "recovery_chain_repairs_v1_1_r1"
    assert {
        path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mode)
        for path in old_files
    } == before


def test_render_verify_and_idempotent_reentry(
    rendered: chain.RecoveryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _read(rendered.chain_manifest)
    assert manifest["chain_id"] == chain.verify_chain(rendered.chain_manifest)["chain_id"]
    assert tuple(row["name"] for row in manifest["jobs"]) == chain.EXPECTED_JOB_ORDER
    assert manifest["state_root"] == str(rendered.state)
    assert manifest["source_harness_prefix"] == str(rendered.source_harness)
    assert not (rendered.jobs_root.stat().st_mode & 0o222)
    assert all("#SBATCH --no-requeue" in path.read_text(encoding="utf-8") for path in rendered.jobs_root.iterdir())
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )
    again = chain.render_chain(rendered, slurm_user="tester", apply=True)
    assert again["status"] == "already_rendered"


def test_preproduction_fleet_is_bound_to_next_generation_runtime_lease(
    rendered: chain.RecoveryPaths,
) -> None:
    manifest = _read(rendered.chain_manifest)
    by_name = {row["name"]: Path(row["script"]) for row in manifest["jobs"]}
    for name in ("fleet_bootstrap", "fleet_readiness"):
        text = by_name[name].read_text(encoding="utf-8")
        assert "with schema5_control.control_lock(state_dir):" in text
        assert 'control.get("desired_state") != "paused"' in text
        assert "target_generation = current_generation + 1" in text
        assert "ensure_runtime_integrity_attestation(" in text
        assert "force_full=False" in text
        assert "validate_runtime_integrity_attestation(" in text
        assert "verify_metadata=True" in text
        assert "schema5_control.production_environment(execution_control)" in text
        for variable in (
            "ASYS_RUNTIME_ATTESTATION",
            "ASYS_RUNTIME_ATTESTATION_SHA256",
            "ASYS_RUNTIME_INTEGRITY_LEASE",
            "ASYS_IMMUTABLE_PINS_SHA256",
            "ASYS_ROLLOUT_GENERATION",
        ):
            assert variable in text
        assert 'subprocess.run([*command, "--once"], check=True, env=environment)' in text

    records = {row["name"]: row for row in manifest["jobs"]}
    assert records["fleet_bootstrap"]["dependencies"] == ["supplementary_cache"]
    assert records["fleet_readiness"]["dependencies"] == ["fleet_bootstrap"]
    assert records["smoke_readiness"]["dependencies"] == ["fleet_readiness"]

    fleet_readiness = by_name["fleet_readiness"].read_text(encoding="utf-8")
    assert '(( remaining < 300 )) && sleep "$remaining" || sleep 300' in fleet_readiness
    smoke = by_name["smoke_readiness"].read_text(encoding="utf-8")
    presmoke = smoke.index("presmoke_deadline=")
    live_probe = smoke.index("attest --gate fleet", presmoke)
    smoke_runner = smoke.index("run_schema5_smokes.py", live_probe)
    assert presmoke < live_probe < smoke_runner
    assert "pre-smoke fleet readiness could not be refreshed within 15 minutes" in smoke

    source_root = Path(chain.__file__).resolve().parents[1]
    smoke_runner_source = (source_root / "scripts/run_schema5_smokes.py").read_text(
        encoding="utf-8"
    )
    assert "runtime_integrity.refresh_generation_lease(" in smoke_runner_source
    assert "process.communicate(timeout=60.0)" in smoke_runner_source
    assert "process.send_signal(signal.SIGUSR1)" in smoke_runner_source


def test_render_recovers_exact_marker_last_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )
    real_atomic = chain._atomic_json
    failed = False

    def crash_before_marker(path: Path, payload: object, *, mode: int = 0o444) -> None:
        nonlocal failed
        if path == paths.chain_manifest and not failed:
            failed = True
            raise RuntimeError("simulated marker publication crash")
        real_atomic(path, payload, mode=mode)

    monkeypatch.setattr(chain, "_atomic_json", crash_before_marker)
    with pytest.raises(RuntimeError, match="marker publication"):
        chain.render_chain(paths, slurm_user="tester", apply=True)
    assert paths.jobs_root.is_dir() and paths.logs_root.is_dir()
    assert not paths.chain_manifest.exists()
    monkeypatch.setattr(chain, "_atomic_json", real_atomic)
    assert chain.render_chain(paths, slurm_user="tester", apply=True)["status"] == "complete"


def test_render_refuses_drifted_crash_remnant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(
        chain,
        "verify_release_tag",
        lambda repository: {"release_tag": chain.RELEASE_TAG, "git_commit": COMMIT},
    )
    specs = chain.job_specs(paths, commit=COMMIT, slurm_user="tester")
    scripts = {spec.name: chain.render_sbatch(spec, paths, partition="mit_preemptable") for spec in specs}
    paths.jobs_root.mkdir(parents=True)
    for spec in specs:
        path = paths.jobs_root / spec.filename
        path.write_bytes(scripts[spec.name])
        path.chmod(0o444)
    paths.jobs_root.chmod(0o550)
    victim = paths.jobs_root / specs[0].filename
    paths.jobs_root.chmod(0o750)
    victim.chmod(0o644)
    victim.write_text("drift\n", encoding="utf-8")
    victim.chmod(0o444)
    paths.jobs_root.chmod(0o550)
    with pytest.raises(chain.ChainError, match="content drifted"):
        chain.render_chain(paths, slurm_user="tester", apply=True)


def test_verify_rejects_manifest_symlink(rendered: chain.RecoveryPaths, tmp_path: Path) -> None:
    link = tmp_path / "manifest-link.json"
    link.symlink_to(rendered.chain_manifest)
    with pytest.raises(chain.ChainError, match="cannot be a symlink"):
        chain.verify_chain(link)


def test_verify_rejects_rehashed_dag_drift(rendered: chain.RecoveryPaths) -> None:
    manifest = _read(rendered.chain_manifest)
    manifest["jobs"][-1]["dependencies"] = ["source_checkout"]
    identity = dict(manifest)
    identity.pop("chain_id")
    manifest["chain_id"] = chain._sha256_bytes(chain._canonical_json(identity))
    rendered.chain_manifest.chmod(0o644)
    rendered.chain_manifest.write_bytes(chain._canonical_json(manifest))
    rendered.chain_manifest.chmod(0o444)
    with pytest.raises(chain.ChainError, match="fixed rendered contract"):
        chain.verify_chain(rendered.chain_manifest)


def test_dag_validator_rejects_missing_gate(rendered: chain.RecoveryPaths) -> None:
    specs = list(chain.job_specs(rendered, commit=COMMIT, slurm_user="tester"))
    resume = specs[-1]
    specs[-1] = chain.JobSpec(
        resume.name,
        resume.filename,
        ("source_checkout",),
        resume.time_limit,
        resume.memory,
        resume.cpus,
        resume.body,
    )
    with pytest.raises(chain.ChainError, match="fixed v1.1-r1 job contract"):
        chain._validate_dag(specs)


def test_submit_dry_run_never_contacts_scheduler(rendered: chain.RecoveryPaths) -> None:
    def forbidden(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(argv)

    report = chain.submit_chain(rendered.chain_manifest, runner=forbidden)
    assert report["status"] == "dry_run"
    assert report["job_count"] == 20
    assert all("${job." in " ".join(row["argv"]) or row["name"] == "source_checkout" for row in report["submission_plan"])


def test_submit_exact_afterok_dag_and_validates_idempotent_receipt(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    report = chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=100.0)
    assert report["status"] == "submitted"
    assert len(slurm.sbatch_calls) == len(chain.EXPECTED_JOB_ORDER)
    submitted: dict[str, str] = {}
    for name, argv, job in zip(chain.EXPECTED_JOB_ORDER, slurm.sbatch_calls, slurm.jobs, strict=True):
        assert "--no-requeue" in argv
        assert "--comment=" + chain._submission_comments(_read(rendered.chain_manifest))[name] in argv
        row = next(row for row in _read(rendered.chain_manifest)["jobs"] if row["name"] == name)
        expected_ids = [submitted[item] for item in row["dependencies"]]
        dependency_flags = [item for item in argv if item.startswith("--dependency=")]
        assert dependency_flags == (["--dependency=afterok:" + ":".join(expected_ids)] if expected_ids else [])
        submitted[name] = job["job_id"]
    receipt = rendered.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    assert not (receipt.stat().st_mode & 0o222)
    journal = rendered.recovery_root / chain.SUBMISSION_JOURNAL_NAME
    assert not (journal.stat().st_mode & 0o222)
    assert report["submission_journal"] == str(journal)
    assert report["submission_journal_sha256"] == chain._sha256(journal)
    no_scheduler = FakeSlurm()
    again = chain.submit_chain(rendered.chain_manifest, apply=True, runner=no_scheduler, now=101.0)
    assert again["status"] == "already_submitted"
    assert no_scheduler.calls == []


def test_sbatch_nonzero_is_treated_as_ambiguous_during_visibility_grace(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    slurm.reject_sbatch_once = True
    with pytest.raises(chain.ChainError, match="sbatch rejected"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=150.0)
    assert len(slurm.sbatch_calls) == 1
    with pytest.raises(chain.ChainError, match="visibility grace"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=151.0)
    assert len(slurm.sbatch_calls) == 1


def test_crash_after_scheduler_acceptance_reconciles_without_duplicate(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError, match="scheduler acceptance"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=100.0)
    journal_path = rendered.recovery_root / chain.SUBMISSION_JOURNAL_NAME
    journal = _read(journal_path)
    first = journal["jobs"]["source_checkout"]
    assert first["attempts"] == 1
    assert first["submission_boundary_state"] == "sbatch_in_flight"
    assert "job_id" not in first

    slurm.hide_jobs = True
    with pytest.raises(chain.ChainError, match="visibility grace"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=101.0)
    assert len(slurm.sbatch_calls) == 1

    slurm.hide_jobs = False
    report = chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=101.0)
    assert report["status"] == "submitted"
    assert len(slurm.jobs) == 20
    assert sum(job["job_name"] == "asys-s5v11r1-checkout" for job in slurm.jobs) == 1


def test_terminal_job_with_blank_sacct_comment_reconciles_from_submit_line(
    rendered: chain.RecoveryPaths,
) -> None:
    """Exercise the cluster's AccountingStoreFlags=(null) behavior."""

    slurm = FakeSlurm()
    slurm.blank_sacct_comments = True
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError, match="scheduler acceptance"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=100.0)
    # The accepted job has already left squeue before recovery.  Its only durable
    # transaction token is now in sacct SubmitLine, not the Comment column.
    slurm.jobs[0]["state"] = "COMPLETED"
    report = chain.submit_chain(
        rendered.chain_manifest,
        apply=True,
        runner=slurm,
        now=401.0,
    )
    assert report["status"] == "submitted"
    assert len(slurm.jobs) == len(chain.EXPECTED_JOB_ORDER)
    assert sum(job["job_name"] == "asys-s5v11r1-checkout" for job in slurm.jobs) == 1


def test_sacct_comment_submit_line_conflict_fails_closed(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError, match="scheduler acceptance"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=100.0)
    slurm.jobs[0]["state"] = "COMPLETED"
    slurm.jobs[0]["submit_line"] = slurm.jobs[0]["submit_line"].replace(
        slurm.jobs[0]["comment"],
        "asys:s5-recovery-v1.1-r1:conflict:g0000:source_checkout",
    )
    with pytest.raises(chain.ChainError, match="comment/SubmitLine conflict"):
        chain.submit_chain(
            rendered.chain_manifest,
            apply=True,
            runner=slurm,
            now=401.0,
        )
    assert len(slurm.jobs) == 1


def test_crash_after_sbatch_return_before_job_id_commit_reconciles(
    rendered: chain.RecoveryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    slurm = FakeSlurm()
    real_atomic = chain._atomic_json
    crashed = False

    def crash_on_job_id(path: Path, payload: object, *, mode: int = 0o444) -> None:
        nonlocal crashed
        if (
            path.name == chain.SUBMISSION_JOURNAL_NAME
            and not crashed
            and isinstance(payload, dict)
            and payload.get("jobs", {}).get("source_checkout", {}).get("job_id")
        ):
            crashed = True
            raise RuntimeError("simulated death before job-id commit")
        real_atomic(path, payload, mode=mode)

    monkeypatch.setattr(chain, "_atomic_json", crash_on_job_id)
    with pytest.raises(RuntimeError, match="job-id commit"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=200.0)
    monkeypatch.setattr(chain, "_atomic_json", real_atomic)
    assert len(slurm.jobs) == 1
    report = chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=201.0)
    assert report["status"] == "submitted"
    assert len(slurm.jobs) == 20


def test_crash_before_receipt_reconciles_all_ids_without_resubmission(
    rendered: chain.RecoveryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    slurm = FakeSlurm()
    real_atomic = chain._atomic_json

    def crash_on_receipt(path: Path, payload: object, *, mode: int = 0o444) -> None:
        if path.name == chain.SUBMISSION_RECEIPT_NAME:
            raise RuntimeError("simulated receipt crash")
        real_atomic(path, payload, mode=mode)

    monkeypatch.setattr(chain, "_atomic_json", crash_on_receipt)
    with pytest.raises(RuntimeError, match="receipt crash"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=300.0)
    assert len(slurm.sbatch_calls) == 20
    monkeypatch.setattr(chain, "_atomic_json", real_atomic)
    report = chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=301.0)
    assert report["status"] == "submitted"
    assert len(slurm.sbatch_calls) == 20


def test_committed_job_missing_from_scheduler_fails_closed(
    rendered: chain.RecoveryPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    slurm = FakeSlurm()
    real_atomic = chain._atomic_json
    crashed = False

    def crash_on_second_intent(path: Path, payload: object, *, mode: int = 0o444) -> None:
        nonlocal crashed
        jobs = payload.get("jobs", {}) if isinstance(payload, dict) else {}
        if path.name == chain.SUBMISSION_JOURNAL_NAME and len(jobs) == 2 and not crashed:
            crashed = True
            raise RuntimeError("stop after first committed job")
        real_atomic(path, payload, mode=mode)

    monkeypatch.setattr(chain, "_atomic_json", crash_on_second_intent)
    with pytest.raises(RuntimeError, match="first committed"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=400.0)
    monkeypatch.setattr(chain, "_atomic_json", real_atomic)
    assert len(slurm.jobs) == 1
    slurm.hide_jobs = True
    with pytest.raises(chain.ChainError, match="maps to 0 Slurm jobs"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=401.0)
    assert len(slurm.jobs) == 1


def test_duplicate_reconciled_jobs_fail_closed(rendered: chain.RecoveryPaths) -> None:
    slurm = FakeSlurm()
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=500.0)
    slurm.jobs.append(dict(slurm.jobs[0], job_id="9999"))
    with pytest.raises(chain.ChainError, match="multiple Slurm jobs"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=501.0)
    assert len(slurm.sbatch_calls) == 1


def test_tampered_receipt_is_not_trusted(rendered: chain.RecoveryPaths) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=600.0)
    receipt_path = rendered.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    receipt = _read(receipt_path)
    receipt["jobs"][0]["job_id"] = "999999"
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(chain._canonical_json(receipt))
    receipt_path.chmod(0o444)
    with pytest.raises(chain.ChainError, match="receipt"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=FakeSlurm(), now=601.0)


def test_receipt_is_bound_to_exact_sealed_submission_journal(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=650.0)
    journal_path = rendered.recovery_root / chain.SUBMISSION_JOURNAL_NAME
    journal = _read(journal_path)
    journal["jobs"]["source_checkout"]["scheduler_state"] = "COMPLETED"
    journal_path.chmod(0o644)
    journal_path.write_bytes(chain._canonical_json(journal))
    journal_path.chmod(0o444)

    with pytest.raises(chain.ChainError, match="receipt identity"):
        chain.submit_chain(
            rendered.chain_manifest,
            apply=True,
            runner=FakeSlurm(),
            now=651.0,
        )


def test_repair_dry_run_and_apply_resubmit_only_failed_suffix(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=900.0)
    slurm.complete_all()
    slurm.set_latest_state("asys-s5v11r1-resume", "FAILED")

    before = len(slurm.sbatch_calls)
    dry_run = chain.repair_chain(
        rendered.chain_manifest, runner=slurm, now=901.0
    )
    assert dry_run == {
        "status": "dry_run",
        "repair_generation": 1,
        "repair_jobs": ["production_resume"],
        "states": {"production_resume": "FAILED"},
        "would_write": str(rendered.recovery_root / chain.REPAIR_ROOT_NAME / "g0001"),
    }
    assert len(slurm.sbatch_calls) == before

    repaired = chain.repair_chain(
        rendered.chain_manifest, apply=True, runner=slurm, now=902.0
    )
    assert repaired["status"] == "repair_submitted"
    assert repaired["repair_jobs"] == ["production_resume"]
    assert len(slurm.sbatch_calls) == before + 1
    repair_argv = slurm.sbatch_calls[-1]
    assert any(":g0001:production_resume" in item for item in repair_argv)
    assert "--dependency=afterok:7018" in repair_argv
    repair_receipt = (
        rendered.recovery_root
        / chain.REPAIR_ROOT_NAME
        / "g0001"
        / chain.SUBMISSION_RECEIPT_NAME
    )
    repair_journal = repair_receipt.parent / chain.SUBMISSION_JOURNAL_NAME
    assert repair_receipt.is_file()
    assert not (repair_receipt.stat().st_mode & 0o222)
    assert not (repair_journal.stat().st_mode & 0o222)
    assert repaired["submission_journal"] == str(repair_journal)
    assert repaired["submission_journal_sha256"] == chain._sha256(repair_journal)

    live = chain.repair_chain(rendered.chain_manifest, runner=slurm, now=903.0)
    assert live["status"] == "in_progress"
    slurm.set_latest_state("asys-s5v11r1-resume", "COMPLETED")
    done = chain.repair_chain(rendered.chain_manifest, runner=slurm, now=904.0)
    assert done["status"] == "complete"
    assert done["receipt"] == str(repair_receipt)


def test_repair_reads_blank_sacct_comment_from_submit_line(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=950.0)
    slurm.complete_all()
    slurm.set_latest_state("asys-s5v11r1-resume", "FAILED")
    slurm.blank_sacct_comments = True

    report = chain.repair_chain(rendered.chain_manifest, runner=slurm, now=951.0)
    assert report["status"] == "dry_run"
    assert report["repair_jobs"] == ["production_resume"]


def test_repair_crash_after_acceptance_observes_visibility_grace(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1000.0)
    slurm.complete_all()
    slurm.set_latest_state("asys-s5v11r1-resume", "FAILED")
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError, match="scheduler acceptance"):
        chain.repair_chain(
            rendered.chain_manifest, apply=True, runner=slurm, now=1001.0
        )
    repair_journal = (
        rendered.recovery_root
        / chain.REPAIR_ROOT_NAME
        / "g0001"
        / chain.SUBMISSION_JOURNAL_NAME
    )
    record = _read(repair_journal)["jobs"]["production_resume"]
    assert record["attempts"] == 1
    assert record["submission_boundary_state"] == "sbatch_in_flight"
    assert "job_id" not in record

    slurm.hide_generation_jobs = True
    with pytest.raises(chain.ChainError, match="visibility grace"):
        chain.repair_chain(
            rendered.chain_manifest, apply=True, runner=slurm, now=1002.0
        )
    assert len(slurm.sbatch_calls) == 21
    slurm.hide_generation_jobs = False
    repaired = chain.repair_chain(
        rendered.chain_manifest, apply=True, runner=slurm, now=1002.0
    )
    assert repaired["status"] == "repair_submitted"
    assert len(slurm.sbatch_calls) == 21


def test_repair_refuses_partial_materialization_before_creating_generation(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1100.0)
    slurm.complete_all()
    manifest = _read(rendered.chain_manifest)
    materialize_index = chain.EXPECTED_JOB_ORDER.index("release_materialize")
    for name in chain.EXPECTED_JOB_ORDER[materialize_index:]:
        job_name = next(row["job_name"] for row in manifest["jobs"] if row["name"] == name)
        slurm.set_latest_state(
            job_name, "FAILED" if name == "release_materialize" else "CANCELLED"
        )
    rendered.release_root.mkdir(parents=True)
    with pytest.raises(chain.ChainError, match="quarantine-materialization --apply"):
        chain.repair_chain(
            rendered.chain_manifest, apply=True, runner=slurm, now=1101.0
        )
    assert not (rendered.recovery_root / chain.REPAIR_ROOT_NAME).exists()


def test_pending_repair_journal_tamper_fails_closed(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1200.0)
    slurm.complete_all()
    slurm.set_latest_state("asys-s5v11r1-resume", "FAILED")
    slurm.crash_after_accept_once = True
    with pytest.raises(RuntimeError):
        chain.repair_chain(
            rendered.chain_manifest, apply=True, runner=slurm, now=1201.0
        )
    journal_path = (
        rendered.recovery_root
        / chain.REPAIR_ROOT_NAME
        / "g0001"
        / chain.SUBMISSION_JOURNAL_NAME
    )
    journal = _read(journal_path)
    journal["repair_jobs"] = ["source_checkout"]
    journal_path.write_bytes(chain._canonical_json(journal))
    with pytest.raises(chain.ChainError, match="repair journal"):
        chain.repair_chain(rendered.chain_manifest, runner=slurm, now=1202.0)


def test_quarantine_rejects_tampered_empty_pending_repair_journal(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1250.0)
    slurm.complete_all()
    manifest = _read(rendered.chain_manifest)
    materialize_index = chain.EXPECTED_JOB_ORDER.index("release_materialize")
    repair_names = list(chain.EXPECTED_JOB_ORDER[materialize_index:])
    for name in repair_names:
        job_name = next(row["job_name"] for row in manifest["jobs"] if row["name"] == name)
        slurm.set_latest_state(
            job_name, "FAILED" if name == "release_materialize" else "CANCELLED"
        )
    rendered.release_root.mkdir(parents=True)

    receipt_path = rendered.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    generation_root = rendered.recovery_root / chain.REPAIR_ROOT_NAME / "g0001"
    generation_root.mkdir(parents=True)
    started = 1251.0
    journal = {
        "schema_version": chain.SUBMISSION_SCHEMA_VERSION,
        "protocol": "schema5-v1.1-r1-recovery-chain-repair-journal",
        "chain_id": "tampered-chain-id",
        "repair_generation": 1,
        "base_receipt": str(receipt_path),
        "base_receipt_sha256": chain._sha256(receipt_path),
        "repair_jobs": repair_names,
        "started_at": "2026-01-01T00:00:00+00:00",
        "started_timestamp": started,
        "scheduler_since": chain._slurm_timestamp(started),
        "jobs": {},
    }
    (generation_root / chain.SUBMISSION_JOURNAL_NAME).write_bytes(
        chain._canonical_json(journal)
    )

    with pytest.raises(chain.ChainError, match="repair journal identity"):
        chain.quarantine_partial_materialization(
            rendered.chain_manifest,
            apply=True,
            runner=slurm,
        )
    assert rendered.release_root.is_dir()


def test_repair_receipt_cannot_rebind_reused_parent_job(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1300.0)
    slurm.complete_all()
    slurm.set_latest_state("asys-s5v11r1-resume", "FAILED")
    chain.repair_chain(rendered.chain_manifest, apply=True, runner=slurm, now=1301.0)
    receipt_path = (
        rendered.recovery_root
        / chain.REPAIR_ROOT_NAME
        / "g0001"
        / chain.SUBMISSION_RECEIPT_NAME
    )
    receipt = _read(receipt_path)
    receipt["jobs"][0]["job_id"] = "999999"
    identity = dict(receipt)
    identity.pop("receipt_id")
    receipt["receipt_id"] = chain._sha256_bytes(chain._canonical_json(identity))
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(chain._canonical_json(receipt))
    receipt_path.chmod(0o444)
    with pytest.raises(chain.ChainError, match="repair receipt job drifted"):
        chain.repair_chain(rendered.chain_manifest, runner=slurm, now=1302.0)


def test_missing_kill_invalid_depend_prevents_any_submission(
    rendered: chain.RecoveryPaths,
) -> None:
    slurm = FakeSlurm()
    slurm.allow_dependency_policy = False
    with pytest.raises(chain.ChainError, match="kill_invalid_depend"):
        chain.submit_chain(rendered.chain_manifest, apply=True, runner=slurm, now=700.0)
    assert slurm.sbatch_calls == []
    assert not (rendered.recovery_root / chain.SUBMISSION_JOURNAL_NAME).exists()


def test_submission_lock_is_nonblocking_and_nonfollowing(rendered: chain.RecoveryPaths) -> None:
    lock = rendered.recovery_root / chain.SUBMISSION_LOCK_NAME
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o640)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(chain.ChainError, match="holds the lock"):
            chain.submit_chain(rendered.chain_manifest, apply=True, runner=FakeSlurm(), now=800.0)
    finally:
        os.close(descriptor)
