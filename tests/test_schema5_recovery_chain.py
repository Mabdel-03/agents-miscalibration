"""Operational invariants for the generation-scoped schema-5 recovery DAG."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import pytest

from scripts import render_schema5_recovery_chain as chain


@pytest.fixture(autouse=True)
def _historical_r1_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep historical tests while the public r1 mutation path stays retired."""

    monkeypatch.setattr(
        chain, "_reject_retired_r1_mutation", lambda _recovery_root: None
    )


def _run(*argv: str, cwd: Path) -> str:
    proc = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _tagged_repository(root: Path) -> Path:
    root.mkdir()
    _run("git", "init", "-q", cwd=root)
    _run("git", "config", "user.email", "schema5-chain@example.invalid", cwd=root)
    _run("git", "config", "user.name", "Schema5 Chain Test", cwd=root)
    (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=root)
    _run("git", "commit", "-q", "-m", "frozen release", cwd=root)
    _run(
        "git",
        "tag",
        "-a",
        chain.RELEASE_TAG,
        "-m",
        "immutable schema5 v1.1-r1 operational retry",
        cwd=root,
    )
    return root


def _paths(tmp_path: Path, repository: Path | None = None) -> chain.RecoveryPaths:
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    for run_id in chain.LEGACY_RUN_IDS:
        (results / run_id).mkdir()
    (recovery / "pre_repair_inventory").mkdir()
    repository = repository or (tmp_path / "repo")
    repository.mkdir(exist_ok=True)
    hf_home = tmp_path / "hf"
    source_serving = tmp_path / "source_serving"
    hf_home.mkdir()
    source_serving.mkdir()
    conda_executable = tmp_path / "conda"
    conda_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    conda_executable.chmod(0o755)
    return chain.recovery_paths(
        repository=repository,
        results_root=results,
        recovery_root=recovery,
        hf_home=hf_home,
        dev_python=Path(sys.executable),
        source_harness=Path(sys.executable).resolve().parent.parent,
        source_serving=source_serving,
        conda_executable=conda_executable,
    )


def _rendered(tmp_path: Path) -> tuple[chain.RecoveryPaths, dict[str, object]]:
    repository = _tagged_repository(tmp_path / "repo")
    paths = _paths(tmp_path, repository)
    dry_run = chain.render_chain(paths, slurm_user="tester", apply=False)
    assert dry_run["status"] == "dry_run"
    assert not paths.jobs_root.exists()
    applied = chain.render_chain(paths, slurm_user="tester", apply=True)
    assert applied["status"] == "complete"
    return paths, json.loads(paths.chain_manifest.read_text(encoding="utf-8"))


def _submit_test_chain(
    paths: chain.RecoveryPaths,
    *,
    first_job_id: int = 1001,
) -> dict[str, object]:
    next_job_id = first_job_id

    def runner(argv):
        nonlocal next_job_id
        argv = list(argv)
        if argv[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv, 0, "DependencyParameters = kill_invalid_depend\n", ""
            )
        if argv and argv[0] in {"squeue", "sacct"}:
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[0] == "sbatch"
        job_id = next_job_id
        next_job_id += 1
        return subprocess.CompletedProcess(argv, 0, f"{job_id}\n", "")

    return chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=runner,
        now=1_800_000_000.0,
    )


def _terminal_scheduler_runner(
    manifest: dict[str, object],
    receipt: dict[str, object],
    states: dict[str, str],
    *,
    submitted_job_id: str | None = None,
):
    manifest_rows = {row["name"]: row for row in manifest["jobs"]}

    def runner(argv):
        argv = list(argv)
        if argv[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv, 0, "DependencyParameters = kill_invalid_depend\n", ""
            )
        if argv and argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv and argv[0] == "sacct" and "-j" in argv:
            lines = []
            for row in receipt["jobs"]:
                name = row["name"]
                state = states.get(name, "COMPLETED")
                exit_code = "0:0" if state == "COMPLETED" else "1:0"
                lines.append(
                    "|".join(
                        (
                            row["job_id"],
                            state,
                            exit_code,
                            row["comment"],
                            manifest_rows[name]["job_name"],
                        )
                    )
                    + "|"
                )
            return subprocess.CompletedProcess(argv, 0, "\n".join(lines) + "\n", "")
        if argv and argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv and argv[0] == "sbatch" and submitted_job_id is not None:
            return subprocess.CompletedProcess(argv, 0, f"{submitted_job_id}\n", "")
        raise AssertionError(argv)

    return runner


def test_rendered_chain_is_fresh_immutable_and_fail_closed(tmp_path: Path):
    paths, manifest = _rendered(tmp_path)

    assert paths.jobs_root == paths.recovery_root / "jobs" / "schema5-v1.1-r1"
    assert paths.logs_root == paths.recovery_root / "logs" / "schema5-v1.1-r1"
    assert paths.chain_manifest.name == "RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
    assert manifest["release_id"] == "sweep-recovery-schema5-v1.1"
    assert manifest["release_tag"] == "sweep-recovery-schema5-v1.1-r1"
    assert manifest["namespace"] == "schema5-v1.1-r1"
    assert manifest["partition"] == "mit_normal"
    assert manifest["source_checkout"].endswith("release_source_checkout_v1_1_r1")
    assert len(manifest["jobs"]) == 20
    assert stat.S_IMODE(paths.chain_manifest.stat().st_mode) == 0o444

    jobs = {row["name"]: row for row in manifest["jobs"]}
    for row in manifest["jobs"]:
        script = Path(row["script"])
        assert stat.S_IMODE(script.stat().st_mode) == 0o444
        text = script.read_text(encoding="utf-8")
        assert text.count("#SBATCH --no-requeue") == 1
        assert re.search(r"^\+  --", text, re.MULTILINE) is None
        syntax = subprocess.run(
            ["bash", "-n", str(script)], text=True, capture_output=True, check=False
        )
        assert syntax.returncode == 0, f"{script}: {syntax.stderr}"

    checkout = Path(jobs["source_checkout"]["script"]).read_text(encoding="utf-8")
    assert "git clone --no-local --no-checkout" in checkout
    assert "checkout --detach" in checkout
    assert "/release_source_checkout/" not in checkout

    fleet = Path(jobs["fleet_readiness"]["script"]).read_text(encoding="utf-8")
    assert jobs["fleet_readiness"]["time_limit"] == "11:00:00"
    assert "+ 36000" in fleet
    assert "seq 1 144" not in fleet

    smoke = Path(jobs["smoke_readiness"]["script"]).read_text(encoding="utf-8")
    assert "print(generation + 1)" in smoke
    assert '--rollout-generation "$next_generation"' in smoke
    assert 'desired_state") != "paused"' in smoke

    resume = Path(jobs["production_resume"]["script"]).read_text(encoding="utf-8")
    assert "#SBATCH --time=11:30:00" in resume
    drill_check = resume.index("drill status --live")
    fleet_refresh = resume.index("fleet_deadline=")
    fleet_reattest = resume.index("attest --gate fleet")
    resume_retry = resume.index('until "${control[@]}" resume')
    assert drill_check < fleet_refresh < fleet_reattest < resume_retry
    assert "+ 900" in resume
    assert "fleet readiness could not be refreshed within 15 minutes" in resume
    assert "resume transaction did not commit within 15 minutes" in resume

    consolidate = Path(jobs["legacy_consolidate"]["script"]).read_text(
        encoding="utf-8"
    )
    assert consolidate.index("freeze_schema5_release.py") < consolidate.index("--apply")
    assert set(jobs["legacy_consolidate"]["dependencies"]) == {
        "release_freeze",
        "pre_repair_snapshot_verify",
    }

    # Every heavyweight pass is an ancestor of the next; none can overlap through a
    # sibling dependency branch.
    dependencies = {
        name: set(row["dependencies"]) for name, row in jobs.items()
    }

    def ancestors(name: str) -> set[str]:
        seen: set[str] = set()
        pending = list(dependencies[name])
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item)
                pending.extend(dependencies[item])
        return seen

    for earlier, later in zip(chain.HEAVY_SERIAL_ORDER, chain.HEAVY_SERIAL_ORDER[1:]):
        assert earlier in ancestors(later)
    assert {
        "static_readiness",
        "context_readiness",
        "email_readiness",
        "fleet_readiness",
        "smoke_readiness",
        "controller_drill",
        "supplementary_cache",
    } <= ancestors("production_resume")

    assert chain.verify_chain(paths.chain_manifest)["passed"] is True


def test_generated_maintenance_body_executes_without_map_tuple_error(tmp_path: Path):
    paths = _paths(tmp_path)
    checkout = paths.source_checkout
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "consolidate_legacy_recovery.py").write_text(
        "def _maintenance_precheck(results_root, recovery_root):\n"
        "    return {'passed': True, 'results': str(results_root), "
        "'recovery': str(recovery_root)}\n",
        encoding="utf-8",
    )
    spec = next(
        item
        for item in chain.job_specs(paths, commit="a" * 40, slurm_user="tester")
        if item.name == "maintenance_preflight"
    )
    script = tmp_path / "maintenance.sbatch"
    script.write_bytes(chain.render_sbatch(spec, paths, partition="mit_preemptable"))
    env = os.environ.copy()
    env["USER"] = "tester"
    proc = subprocess.run(
        ["bash", str(script)], env=env, text=True, capture_output=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["passed"] is True
    assert report["results"] == str(paths.results_root)


def test_context_temporary_expands_exact_slurm_job_id(tmp_path: Path):
    paths = _paths(tmp_path)
    spec = next(
        item
        for item in chain.job_specs(paths, commit="a" * 40, slurm_user="tester")
        if item.name == "context_readiness"
    )
    text = chain.render_sbatch(spec, paths, partition="mit_preemptable").decode()
    assignment = next(line for line in text.splitlines() if line.startswith("dense_tmp="))
    proc = subprocess.run(
        ["bash", "-c", f'SLURM_JOB_ID=12345; {assignment}; printf "%s" "$dense_tmp"'],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.endswith("dense_peer_context.json.12345.tmp")
    assert "${SLURM_JOB_ID}" not in proc.stdout


def test_submission_argv_uses_exact_afterok_and_no_requeue():
    row = {"script": "/recovery/jobs/schema5-v1.1-r1/job.sbatch"}
    argv = chain.submission_argv(
        row,
        dependency_job_ids=["101", "202"],
        comment="asys:s5-recovery-v1.1-r1:abc:job",
    )
    assert argv == [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--comment=asys:s5-recovery-v1.1-r1:abc:job",
        "--dependency=afterok:101:202",
        "/recovery/jobs/schema5-v1.1-r1/job.sbatch",
    ]


def test_transactional_submission_records_exact_dependency_ids(tmp_path: Path):
    paths, manifest = _rendered(tmp_path)
    next_job_id = 1000
    submitted_argv: list[list[str]] = []

    def runner(argv):
        nonlocal next_job_id
        argv = list(argv)
        if argv[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "DependencyParameters = kill_invalid_depend\n",
                "",
            )
        if argv and argv[0] in {"squeue", "sacct"}:
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[0] == "sbatch"
        submitted_argv.append(argv)
        next_job_id += 1
        return subprocess.CompletedProcess(argv, 0, f"{next_job_id}\n", "")

    receipt = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=runner,
        now=1_800_000_000.0,
    )
    assert receipt["status"] == "submitted"
    assert receipt["dependency_policy"] == "afterok+kill_invalid_depend"
    assert len(submitted_argv) == len(manifest["jobs"]) == 20
    by_name = {row["name"]: row for row in receipt["jobs"]}
    for row, argv in zip(receipt["jobs"], submitted_argv):
        assert "--no-requeue" in argv
        expected = row["dependency_job_ids"]
        dependency_args = [item for item in argv if item.startswith("--dependency=")]
        if expected:
            assert dependency_args == ["--dependency=afterok:" + ":".join(expected)]
        else:
            assert dependency_args == []
        assert row["job_id"].isdigit()
    assert by_name["legacy_consolidate"]["dependency_job_ids"] == [
        by_name["release_freeze"]["job_id"],
        by_name["pre_repair_snapshot_verify"]["job_id"],
    ]
    receipt_path = paths.recovery_root / chain.SUBMISSION_RECEIPT_NAME
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    journal = json.loads(
        (paths.recovery_root / chain.SUBMISSION_JOURNAL_NAME).read_text(encoding="utf-8")
    )
    assert all(
        row["submission_boundary_state"] == "committed"
        for row in journal["jobs"].values()
    )


def test_submit_crash_after_sbatch_boundary_adopts_unique_scheduler_job(tmp_path: Path):
    paths, manifest = _rendered(tmp_path)
    first = manifest["jobs"][0]
    first_comment = chain._job_comment(manifest["chain_id"], first["name"], 0)

    def crashing_runner(argv):
        argv = list(argv)
        if argv[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv, 0, "DependencyParameters = kill_invalid_depend\n", ""
            )
        if argv and argv[0] in {"squeue", "sacct"}:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv and argv[0] == "sbatch":
            raise RuntimeError("simulated process death after Slurm accepted job 777")
        raise AssertionError(argv)

    with pytest.raises(RuntimeError, match="simulated process death"):
        chain.submit_chain(
            paths.chain_manifest,
            apply=True,
            runner=crashing_runner,
            now=1_800_000_000.0,
        )
    journal_path = paths.recovery_root / chain.SUBMISSION_JOURNAL_NAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["jobs"][first["name"]]["attempts"] == 1
    assert (
        journal["jobs"][first["name"]]["submission_boundary_state"]
        == "sbatch_in_flight"
    )

    next_job_id = 800
    second_sbatch: list[list[str]] = []

    def recovering_runner(argv):
        nonlocal next_job_id
        argv = list(argv)
        if argv[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv, 0, "DependencyParameters = kill_invalid_depend\n", ""
            )
        if argv and argv[0] == "squeue":
            return subprocess.CompletedProcess(
                argv,
                0,
                f"777|{first_comment}|{first['job_name']}|PENDING\n",
                "",
            )
        if argv and argv[0] == "sacct":
            return subprocess.CompletedProcess(
                argv,
                0,
                f"777|{first_comment}|{first['job_name']}|PENDING|\n",
                "",
            )
        assert argv[0] == "sbatch"
        second_sbatch.append(argv)
        next_job_id += 1
        return subprocess.CompletedProcess(argv, 0, f"{next_job_id}\n", "")

    receipt = chain.submit_chain(
        paths.chain_manifest,
        apply=True,
        runner=recovering_runner,
        now=1_800_000_001.0,
    )
    assert receipt["jobs"][0]["job_id"] == "777"
    assert len(second_sbatch) == 19
    assert all(first_comment not in argv for argv in second_sbatch)


def test_renderer_rejects_lightweight_tag_or_dirty_checkout(tmp_path: Path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.email", "schema5-chain@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "Schema5 Chain Test", cwd=repository)
    (repository / "tracked.txt").write_text("x\n", encoding="utf-8")
    _run("git", "add", "tracked.txt", cwd=repository)
    _run("git", "commit", "-q", "-m", "release", cwd=repository)
    _run("git", "tag", chain.RELEASE_TAG, cwd=repository)
    with pytest.raises(chain.ChainError, match="annotated"):
        chain.verify_release_tag(repository)

    _run("git", "tag", "-d", chain.RELEASE_TAG, cwd=repository)
    _run("git", "tag", "-a", chain.RELEASE_TAG, "-m", "release", cwd=repository)
    (repository / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(chain.ChainError, match="clean"):
        chain.verify_release_tag(repository)


def test_repair_resubmits_only_failed_terminal_suffix(tmp_path: Path):
    paths, manifest = _rendered(tmp_path)
    original = _submit_test_chain(paths)
    states = {"production_resume": "FAILED"}
    runner = _terminal_scheduler_runner(
        manifest,
        original,
        states,
        submitted_job_id="9001",
    )

    dry_run = chain.repair_chain(
        paths.chain_manifest,
        apply=False,
        runner=runner,
        now=1_800_001_000.0,
    )
    assert dry_run["repair_generation"] == 1
    assert dry_run["repair_jobs"] == ["production_resume"]

    repaired = chain.repair_chain(
        paths.chain_manifest,
        apply=True,
        runner=runner,
        now=1_800_001_000.0,
    )
    assert repaired["status"] == "repair_submitted"
    assert repaired["repair_jobs"] == ["production_resume"]
    repaired_rows = {row["name"]: row for row in repaired["jobs"]}
    original_rows = {row["name"]: row for row in original["jobs"]}
    assert repaired_rows["production_resume"]["job_id"] == "9001"
    assert repaired_rows["production_resume"]["generation"] == 1
    assert repaired_rows["production_resume"]["disposition"] == "resubmitted"
    assert repaired_rows["production_resume"]["dependency_job_ids"] == [
        original_rows["controller_drill"]["job_id"]
    ]
    assert repaired_rows["controller_drill"]["job_id"] == original_rows[
        "controller_drill"
    ]["job_id"]
    assert repaired_rows["controller_drill"]["disposition"] == "reused_completed"
    receipt_path = (
        paths.recovery_root
        / chain.REPAIR_ROOT_NAME
        / "g0001"
        / chain.SUBMISSION_RECEIPT_NAME
    )
    assert receipt_path.is_file()
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444


def test_partial_materialization_is_quarantined_without_deletion(tmp_path: Path):
    paths, manifest = _rendered(tmp_path)
    original = _submit_test_chain(paths)
    rows = [row["name"] for row in manifest["jobs"]]
    materialize_index = rows.index("release_materialize")
    states = {
        name: ("TIMEOUT" if name == "release_materialize" else "CANCELLED")
        for name in rows[materialize_index:]
    }
    runner = _terminal_scheduler_runner(manifest, original, states)
    paths.release_root.mkdir(parents=True)
    payload = paths.release_root / "preserved.txt"
    payload.write_text("partial release evidence\n", encoding="utf-8")
    source_inode = paths.release_root.stat().st_ino

    dry_run = chain.quarantine_partial_materialization(
        paths.chain_manifest,
        apply=False,
        runner=runner,
    )
    assert dry_run["status"] == "dry_run"
    assert dry_run["source_inode"] == source_inode

    result = chain.quarantine_partial_materialization(
        paths.chain_manifest,
        apply=True,
        runner=runner,
    )
    destination = Path(result["destination"])
    assert result["status"] == "quarantined"
    assert not paths.release_root.exists()
    assert destination.stat().st_ino == source_inode
    assert (destination / payload.name).read_text(encoding="utf-8") == (
        "partial release evidence\n"
    )
    evidence_root = paths.recovery_root / chain.QUARANTINE_EVIDENCE_ROOT_NAME
    intent = evidence_root / f"partial-job-{result['materialize_job_id']}.intent.json"
    completion = (
        evidence_root
        / f"partial-job-{result['materialize_job_id']}.complete.json"
    )
    assert stat.S_IMODE(intent.stat().st_mode) == 0o444
    assert stat.S_IMODE(completion.stat().st_mode) == 0o444

    repeated = chain.quarantine_partial_materialization(
        paths.chain_manifest,
        apply=True,
        runner=runner,
    )
    assert repeated["status"] == "already_quarantined"
    assert destination.stat().st_ino == source_inode
