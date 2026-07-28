"""Tests for the real, fail-closed protected-capacity evidence transaction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from scripts import build_schema5_protected_capacity_evidence as builder
from scripts import publish_schema5_protected_capacity as publisher
from scripts import run_schema5_throughput_qualification as qualification
from agents_scaling.serving.fleet_contract import (
    expected_replica_id,
    expected_scheduler_job_name,
)


COMMIT = "1" * 40
TAG_OBJECT = "2" * 40
TOKEN = "3" * 32
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLEET_PATH = REPOSITORY_ROOT / "configs" / "schema5_fleet.v1.json"
MODEL_PATH = REPOSITORY_ROOT / "configs" / "model_contracts.v1.json"
FLEET_SHA256 = builder.sha256_file(FLEET_PATH)
MODEL_SHA256 = builder.sha256_file(MODEL_PATH)


def test_builder_scheduler_subprocess_rejects_hostile_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    hostile = {
        "PATH": "/tmp/hostile-bin",
        "BASH_ENV": "/tmp/hostile-bash-env",
        "GIT_DIR": "/tmp/hostile-git",
        "GIT_REPLACE_REF_BASE": "refs/hostile/",
        "LD_PRELOAD": "/tmp/hostile.so",
        "PYTHONPATH": "/tmp/hostile-python",
        "SBATCH_PARTITION": "hostile",
        "SLURM_CONF": "/tmp/hostile-slurm.conf",
        "SQUEUE_FORMAT": "hostile",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)

    def fake_run(argv, **kwargs):
        observed["argv"] = list(argv)
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    result = builder._invoke(None, ["squeue", "--version"], timeout=3.0)

    assert result.returncode == 0
    assert observed["argv"] == ["/usr/bin/squeue", "--version"]
    environment = observed["kwargs"]["env"]
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["LANG"] == environment["LC_ALL"] == "C"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not (set(hostile) - {"PATH"}).intersection(environment)


def test_capacity_allocation_script_has_noninheriting_trusted_bootstrap(
    tmp_path: Path,
) -> None:
    payload = builder._render_script(
        root=tmp_path,
        token=TOKEN,
        role="client",
        chunk_index=0,
        chunk_tasks=24,
        cpus_per_task=1,
        memory_mib_per_task=4096,
        gpus_per_task=0,
        time_limit="12:00:00",
        shape_id="client",
        partition="ou_bcs_normal",
        qos="normal",
    )

    assert payload.count("#SBATCH --export=NONE\n") == 1
    assert "#SBATCH --export=ALL" not in payload
    assert "export PATH=/usr/bin:/bin" in payload
    assert "readonly PATH" in payload
    assert "export GIT_NO_REPLACE_OBJECTS=1" in payload
    assert payload.index("#SBATCH --export=NONE") < payload.index(
        "set -euo pipefail"
    )
    assert payload.index("set -euo pipefail") < payload.index(
        "export PATH=/usr/bin:/bin"
    )
    syntax = subprocess.run(
        ["/bin/bash", "-n"],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def _capacity_authority(parent: Path) -> dict[str, object]:
    """Materialize the real solver-certified zero-delta baseline authority."""

    authority = parent / "capacity-authority"
    authority.mkdir(parents=True, exist_ok=True)
    release_worktree = authority / "release-worktree"
    (release_worktree / "configs").mkdir(parents=True, exist_ok=True)
    (release_worktree / "scripts").mkdir(parents=True, exist_ok=True)
    (release_worktree / "slurm").mkdir(parents=True, exist_ok=True)
    base = release_worktree / "configs" / "schema5_fleet.v1.json"
    base_raw = FLEET_PATH.read_bytes()
    if not base.exists():
        base.write_bytes(base_raw)
        base.chmod(0o444)
    assert base.read_bytes() == base_raw
    model = release_worktree / "configs" / "model_contracts.v1.json"
    model_raw = MODEL_PATH.read_bytes()
    if not model.exists():
        model.write_bytes(model_raw)
        model.chmod(0o444)
    assert model.read_bytes() == model_raw
    for source in (
        FLEET_PATH.with_suffix(".sha256"),
        MODEL_PATH.with_suffix(".sha256"),
    ):
        target = release_worktree / "configs" / source.name
        raw_sidecar = source.read_bytes()
        if not target.exists():
            target.write_bytes(raw_sidecar)
            target.chmod(0o444)
        assert target.read_bytes() == raw_sidecar
    for source in (
        Path(builder.__file__),
        Path(publisher.__file__),
        Path(qualification.__file__),
    ):
        target = release_worktree / "scripts" / source.name
        raw_source = source.read_bytes()
        if not target.exists():
            target.write_bytes(raw_source)
            target.chmod(0o444)
        assert target.read_bytes() == raw_source
    dispatcher_source = REPOSITORY_ROOT / "slurm" / "dispatch_sweeps.py"
    tagged_dispatcher = release_worktree / "slurm" / dispatcher_source.name
    dispatcher_raw = dispatcher_source.read_bytes()
    if not tagged_dispatcher.exists():
        tagged_dispatcher.write_bytes(dispatcher_raw)
        tagged_dispatcher.chmod(0o444)
    assert tagged_dispatcher.read_bytes() == dispatcher_raw
    control_source = REPOSITORY_ROOT / "slurm" / "schema5_control.py"
    tagged_control = release_worktree / "slurm" / control_source.name
    control_raw = control_source.read_bytes()
    if not tagged_control.exists():
        tagged_control.write_bytes(control_raw)
        tagged_control.chmod(0o444)
    assert tagged_control.read_bytes() == control_raw
    source_tree_sha256 = builder.sha256_tree(release_worktree)
    dispatcher_source_sha256 = builder.sha256_file(tagged_dispatcher)
    qualification_runner_source_sha256 = builder.sha256_file(
        release_worktree / "scripts" / Path(qualification.__file__).name
    )
    effective = authority / "schema5_fleet.capacity-v1.json"
    payload = json.loads(FLEET_PATH.read_text(encoding="utf-8"))
    raw = base_raw
    if not effective.exists():
        effective.write_bytes(raw)
        effective.chmod(0o444)
    assert effective.read_bytes() == raw
    effective_sha256 = builder.sha256_file(effective)
    sidecar = effective.with_suffix(".sha256")
    if not sidecar.exists():
        sidecar.write_text(
            f"{effective_sha256}  {effective.name}\n",
            encoding="utf-8",
        )
        sidecar.chmod(0o444)
    assert sidecar.read_text(encoding="utf-8") == (
        f"{effective_sha256}  {effective.name}\n"
    )
    base_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in json.loads(FLEET_PATH.read_text(encoding="utf-8"))[
            "profiles"
        ]
    }
    effective_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in payload["profiles"]
    }
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=1,
        release_git_commit=COMMIT,
        source_tree_sha256=source_tree_sha256,
        release_fleet_contract_sha256=FLEET_SHA256,
        base_fleet_contract_sha256=FLEET_SHA256,
        proposed_effective_fleet_contract_sha256=effective_sha256,
        additive_overlay_contract_sha256=effective_sha256,
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256=dispatcher_source_sha256,
        qualification_runner_source_sha256=(
            qualification_runner_source_sha256
        ),
    )
    certificate_path = (
        authority / builder.runtime_capacity.STATIC_FEASIBILITY_FILENAME
    )
    certificate_raw = builder.canonical_bytes(certificate)
    if not certificate_path.exists():
        certificate_path.write_bytes(certificate_raw)
        certificate_path.chmod(0o444)
    assert certificate_path.read_bytes() == certificate_raw
    return {
        "release_worktree": release_worktree,
        "base_fleet_contract_path": base,
        "base_fleet_contract_sha256": FLEET_SHA256,
        "model_contract_path": model,
        "model_contract_sha256": MODEL_SHA256,
        "effective_fleet_contract_path": effective,
        "effective_fleet_contract_sha256": effective_sha256,
        "additive_overlay_contract_path": effective,
        "additive_overlay_contract_sha256": effective_sha256,
        "static_feasibility_certificate_path": certificate_path,
        "static_feasibility_certificate_sha256": builder.sha256_file(
            certificate_path
        ),
        "static_feasibility_certificate_id": certificate["certificate_id"],
        "capacity_generation": 1,
    }


def _capacity_cli_arguments(parent: Path) -> list[str]:
    authority = _capacity_authority(parent)
    return [
        "--effective-fleet-contract",
        str(authority["effective_fleet_contract_path"]),
        "--effective-fleet-contract-sha256",
        str(authority["effective_fleet_contract_sha256"]),
        "--additive-overlay-contract",
        str(authority["additive_overlay_contract_path"]),
        "--additive-overlay-contract-sha256",
        str(authority["additive_overlay_contract_sha256"]),
        "--static-feasibility-certificate",
        str(authority["static_feasibility_certificate_path"]),
        "--static-feasibility-certificate-sha256",
        str(authority["static_feasibility_certificate_sha256"]),
        "--static-feasibility-certificate-id",
        str(authority["static_feasibility_certificate_id"]),
        "--capacity-generation",
        str(authority["capacity_generation"]),
    ]


def test_capacity_authority_accepts_generation_two_tp2_shortfall(
    tmp_path: Path,
) -> None:
    authority = _capacity_authority(tmp_path)
    payload = json.loads(FLEET_PATH.read_text(encoding="utf-8"))
    profile = next(
        row
        for row in payload["profiles"]
        if row["serving_profile"] == "32B-long"
    )
    index = len(profile["replicas"])
    replica = dict(profile["replicas"][-1])
    replica.update(
        {
            "replica_index": index,
            "replica_id": expected_replica_id("32B-long", index),
            "scheduler_job_name": expected_scheduler_job_name(
                "32B-long", index
            ),
        }
    )
    profile["replicas"].append(replica)
    payload["logical_replica_count"] = sum(
        len(row["replicas"]) for row in payload["profiles"]
    )
    payload["allocated_gpu_count"] = sum(
        int(row["tensor_parallel_size"]) * len(row["replicas"])
        for row in payload["profiles"]
    )
    effective_path = (
        tmp_path / "capacity-authority" / "schema5_fleet.capacity-g2.json"
    )
    effective_path.write_bytes(builder.canonical_bytes(payload))
    effective_path.chmod(0o444)
    effective_sha256 = builder.sha256_file(effective_path)
    effective_path.with_suffix(".sha256").write_text(
        f"{effective_sha256}  {effective_path.name}\n",
        encoding="utf-8",
    )
    effective_path.with_suffix(".sha256").chmod(0o444)

    release_worktree = Path(authority["release_worktree"])
    source_tree_sha256 = builder.sha256_tree(release_worktree)
    dispatcher_source_sha256 = builder.sha256_file(
        release_worktree / "slurm" / "dispatch_sweeps.py"
    )
    qualification_runner_source_sha256 = builder.sha256_file(
        release_worktree
        / "scripts"
        / "run_schema5_throughput_qualification.py"
    )
    base_counts = {
        row["serving_profile"]: (
            len(row["replicas"])
            - (1 if row["serving_profile"] == "32B-long" else 0)
        )
        for row in payload["profiles"]
    }
    effective_counts = {
        row["serving_profile"]: len(row["replicas"])
        for row in payload["profiles"]
    }
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=2,
        release_git_commit=COMMIT,
        source_tree_sha256=source_tree_sha256,
        release_fleet_contract_sha256=FLEET_SHA256,
        base_fleet_contract_sha256=FLEET_SHA256,
        proposed_effective_fleet_contract_sha256=effective_sha256,
        additive_overlay_contract_sha256=effective_sha256,
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256=dispatcher_source_sha256,
        qualification_runner_source_sha256=(
            qualification_runner_source_sha256
        ),
    )
    certificate_path = (
        tmp_path
        / "capacity-authority"
        / "capacity-generations"
        / "c000002"
        / builder.runtime_capacity.STATIC_FEASIBILITY_FILENAME
    )
    certificate_path.parent.mkdir(parents=True)
    certificate_path.write_bytes(builder.canonical_bytes(certificate))
    certificate_path.chmod(0o444)

    base_fleet = builder._load_frozen_fleet(
        fleet_contract_path=Path(authority["base_fleet_contract_path"]),
        fleet_contract_sha256=str(
            authority["base_fleet_contract_sha256"]
        ),
        model_contract_path=Path(authority["model_contract_path"]),
        model_contract_sha256=str(authority["model_contract_sha256"]),
        allow_capacity_layout=False,
    )
    effective_fleet = builder._load_frozen_fleet(
        fleet_contract_path=effective_path,
        fleet_contract_sha256=effective_sha256,
        model_contract_path=Path(authority["model_contract_path"]),
        model_contract_sha256=str(authority["model_contract_sha256"]),
        allow_capacity_layout=True,
    )
    observed = builder._validate_effective_capacity_authority(
        base_fleet=base_fleet,
        effective_fleet=effective_fleet,
        additive_overlay_path=effective_path,
        additive_overlay_sha256=effective_sha256,
        static_feasibility_certificate_path=certificate_path,
        static_feasibility_certificate_sha256=builder.sha256_file(
            certificate_path
        ),
        static_feasibility_certificate_id=certificate["certificate_id"],
        capacity_generation=2,
        release_git_commit=COMMIT,
        source_tree_sha256=source_tree_sha256,
        dispatcher_source_sha256=dispatcher_source_sha256,
        qualification_runner_source_sha256=(
            qualification_runner_source_sha256
        ),
    )

    assert observed.capacity_generation == 2
    assert observed.additive_tp1_logical_replicas == 0
    assert observed.additive_tp2_logical_replicas == 1
    assert observed.wave_passed is False
    assert observed.selected_cell_count == 284
    assert observed.shortfall_cells == 100


def test_sacct_start_times_use_scheduler_local_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This instant falls on the prior local date in EDT, catching both the
    # timestamp and current-day accounting variants.
    observed_at = 1_784_952_540.0
    commands: list[list[str]] = []

    def runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    with monkeypatch.context() as timezone_context:
        timezone_context.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            builder._capture_current_user_occupancy(
                plan={"scheduler_user": "tester"},
                runner=runner,
                observed_at=observed_at,
            )
            builder._submission_truth(
                plan={
                    "scheduler_user": "tester",
                    "roles": {
                        "client": {
                            "chunks": [{"comment": "schema5:test"}],
                        },
                    },
                },
                runner=runner,
                since=observed_at,
            )
        finally:
            timezone_context.undo()
            time.tzset()

    sacct_commands = [argv for argv in commands if argv[0] == "sacct"]
    assert len(sacct_commands) == 2
    assert (
        sacct_commands[0][sacct_commands[0].index("-S") + 1]
        == "2026-07-25"
    )
    assert (
        sacct_commands[1][sacct_commands[1].index("-S") + 1]
        == "2026-07-25T00:04:00"
    )


def _identity_runner(
    argv: list[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    del timeout
    arguments = argv[3:]
    if arguments == ["rev-parse", "HEAD"]:
        output = COMMIT
    elif arguments == [
        "rev-parse",
        f"refs/tags/{publisher.RELEASE_TAG}",
    ]:
        output = TAG_OBJECT
    elif arguments == [
        "rev-parse",
        f"refs/tags/{publisher.RELEASE_TAG}^{{}}",
    ]:
        output = COMMIT
    elif arguments == ["cat-file", "-t", TAG_OBJECT]:
        output = "tag"
    elif arguments == [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]:
        output = ""
    else:
        return subprocess.CompletedProcess(argv, 2, "", "unexpected git")
    return subprocess.CompletedProcess(argv, 0, f"{output}\n", "")


def _mock_release_identity(parent: Path) -> dict[str, str]:
    authority = _capacity_authority(parent)
    certificate = json.loads(
        Path(
            str(authority["static_feasibility_certificate_path"])
        ).read_text(encoding="utf-8")
    )
    return {
        "release_worktree": str(REPOSITORY_ROOT),
        "source_tree_sha256": str(certificate["source_tree_sha256"]),
        "builder_source_path": str(Path(builder.__file__).resolve()),
        "builder_source_sha256": builder.sha256_file(
            Path(builder.__file__).resolve()
        ),
        "publisher_source_path": str(Path(publisher.__file__).resolve()),
        "publisher_source_sha256": builder.sha256_file(
            Path(publisher.__file__).resolve()
        ),
        "dispatcher_source_path": str(
            REPOSITORY_ROOT / "slurm" / "dispatch_sweeps.py"
        ),
        "dispatcher_source_sha256": str(
            certificate["dispatcher_source_sha256"]
        ),
        "qualification_runner_source_path": str(
            Path(qualification.__file__).resolve()
        ),
        "qualification_runner_source_sha256": str(
            certificate["qualification_runner_source_sha256"]
        ),
    }


def _plan(root: Path) -> dict[str, object]:
    capacity = _capacity_authority(root.parent)
    return builder.build_plan(
        root=root,
        release_git_commit=COMMIT,
        release_tag_object=TAG_OBJECT,
        partition="ou_bcs_normal",
        qos="normal",
        scheduler_user="tester",
        token=TOKEN,
        fleet_contract_path=capacity.pop("base_fleet_contract_path"),
        fleet_contract_sha256=capacity.pop("base_fleet_contract_sha256"),
        model_contract_path=capacity.pop("model_contract_path"),
        model_contract_sha256=capacity.pop("model_contract_sha256"),
        release_worktree=capacity.pop("release_worktree"),
        **capacity,
        identity_runner=_identity_runner,
    )


def _prepared(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "capacity"
    plan = _plan(root)
    builder.prepare(plan, apply=True)
    return root, plan


def _mark_other_jobs_submitted(
    root: Path,
    *,
    target_key: str,
    target_state: str,
    target_intent_at: float = 900.0,
) -> tuple[dict[str, object], dict[str, object]]:
    plan = builder._load_plan(root)
    ledger = builder._load_ledger(root, plan=plan)
    next_job_id = 30_000
    for key, record in ledger["jobs"].items():
        if key == target_key:
            record["state"] = target_state
            record["job_id"] = None
            record["submission_intent_at"] = (
                target_intent_at if target_state == "submitting" else None
            )
            record["attempt"] = 1 if target_state == "submitting" else 0
            continue
        next_job_id += 1
        record.update(
            {
                "state": "submitted",
                "job_id": str(next_job_id),
                "submitted_at": 800.0,
                "submission_intent_at": 799.0,
                "submission_receipt_at": 800.0,
                "attempt": 1,
            }
        )
    builder._atomic_json(root / builder.LEDGER_FILENAME, ledger)
    return plan, ledger


class ReconcileScheduler:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.accepted: dict[str, dict[str, object]] = {}
        self.next_job_id = 40_000
        self.sbatch_calls = 0
        self.fail_after_accept = False

    def accept(self, key: str, *, state: str = "RUNNING") -> str:
        plan = builder._load_plan(self.root)
        ledger = builder._load_ledger(self.root, plan=plan)
        self.next_job_id += 1
        job_id = str(self.next_job_id)
        record = dict(ledger["jobs"][key])
        record["scheduler_state"] = state
        self.accepted[job_id] = record
        return job_id

    def _write_ready(self, record: dict[str, object], job_id: str) -> None:
        if record["held"]:
            return
        plan = builder._load_plan(self.root)
        directory = (
            self.root
            / builder.READY_DIRECTORY
            / str(record["role"])
            / f"{int(record['chunk_index']):03d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for task_id in range(int(record["tasks"])):
            receipt = {
                "array_job_id": job_id,
                "plan_id": plan["plan_id"],
                "role": record["role"],
                "script_sha256": record["script_sha256"],
                "shape_id": record["shape_id"],
                "task_id": task_id,
                "token": plan["token"],
            }
            path = directory / str(task_id)
            path.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            path.chmod(0o444)

    def __call__(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        if argv[:2] == ["sbatch", "--parsable"]:
            self.sbatch_calls += 1
            plan = builder._load_plan(self.root)
            ledger = builder._load_ledger(self.root, plan=plan)
            matches = [
                (key, record)
                for key, record in ledger["jobs"].items()
                if record["script"] == argv[-1] and record["state"] == "submitting"
            ]
            assert len(matches) == 1
            key, _record = matches[0]
            job_id = self.accept(key)
            self._write_ready(self.accepted[job_id], job_id)
            if self.fail_after_accept:
                self.fail_after_accept = False
                raise subprocess.TimeoutExpired(argv, 60.0)
            return subprocess.CompletedProcess(argv, 0, f"{job_id}\n", "")
        if argv[:4] == ["squeue", "-h", "-u", "tester"]:
            rows = [
                f"{job_id}|{record['comment']}"
                for job_id, record in self.accepted.items()
                if record["scheduler_state"]
                in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
            ]
            return subprocess.CompletedProcess(argv, 0, "\n".join(rows) + ("\n" if rows else ""), "")
        if argv[:5] == ["sacct", "-nP", "--array", "-u", "tester"]:
            rows = [
                f"{job_id}|{record['comment']}|{record['scheduler_state']}"
                for job_id, record in self.accepted.items()
            ]
            return subprocess.CompletedProcess(argv, 0, "\n".join(rows) + ("\n" if rows else ""), "")
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            job_id = argv[4]
            record = self.accepted[job_id]
            return subprocess.CompletedProcess(
                argv,
                0,
                f"JobId={job_id} Requeue=0 Comment={record['comment']}\n",
                "",
            )
        if (
            len(argv) == 5
            and argv[:3] == ["scontrol", "write", "batch_script"]
        ):
            record = self.accepted[argv[3]]
            return subprocess.CompletedProcess(
                argv,
                0,
                Path(str(record["script"])).read_text(encoding="utf-8"),
                "",
            )
        return subprocess.CompletedProcess(argv, 2, "", "unexpected command")


class FakeScheduler:
    """Closed fake for the builder's exact read-only and submit commands."""

    def __init__(
        self,
        root: Path,
        *,
        client_memory: str = "4G",
        association_max_jobs: int | None = None,
        association_max_submit: int = 448,
        qos_max_jobs: int | None = None,
        qos_max_submit: int | None = None,
        qos_max_wall: str = "1-00:00:00",
        existing_jobs: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self.root = root
        self.client_memory = client_memory
        self.association_max_jobs = association_max_jobs
        self.association_max_submit = association_max_submit
        self.qos_max_jobs = qos_max_jobs
        self.qos_max_submit = qos_max_submit
        self.qos_max_wall = qos_max_wall
        self.existing_jobs = tuple(
            (
                job_id,
                state,
                "account",
                "normal",
                comment,
            )
            if len(row) == 3
            else row
            for row in existing_jobs
            for job_id, state, *rest in (row,)
            for comment in (rest[-1],)
        )
        self.next_job_id = 20_000
        self.submissions: list[list[str]] = []
        self.cancellations: list[str] = []

    def _record(self, job_id: str) -> dict[str, object]:
        ledger = json.loads(
            (self.root / builder.LEDGER_FILENAME).read_text(encoding="utf-8")
        )
        for record in ledger["jobs"].values():
            if str(record["job_id"]) == job_id:
                return record
        raise AssertionError(f"unknown fake job {job_id}")

    @staticmethod
    def _memory_text(memory_mib: int) -> str:
        return (
            f"{memory_mib // 1024}G"
            if memory_mib % 1024 == 0
            else f"{memory_mib}M"
        )

    def _write_ready(self, record: dict[str, object], job_id: str) -> None:
        if record["held"]:
            return
        plan = json.loads(
            (self.root / builder.INTENT_FILENAME).read_text(encoding="utf-8")
        )
        directory = (
            self.root
            / builder.READY_DIRECTORY
            / str(record["role"])
            / f"{int(record['chunk_index']):03d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        for task_id in range(int(record["tasks"])):
            receipt = {
                "array_job_id": job_id,
                "plan_id": plan["plan_id"],
                "role": record["role"],
                "script_sha256": record["script_sha256"],
                "shape_id": record["shape_id"],
                "task_id": task_id,
                "token": plan["token"],
            }
            path = directory / str(task_id)
            path.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            path.chmod(0o444)

    def __call__(
        self,
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        if argv[:2] == ["sbatch", "--parsable"]:
            self.next_job_id += 1
            self.submissions.append(list(argv))
            script = str(argv[-1])
            ledger = json.loads(
                (self.root / builder.LEDGER_FILENAME).read_text(encoding="utf-8")
            )
            matches = [
                record
                for record in ledger["jobs"].values()
                if record["script"] == script and record["state"] == "submitting"
            ]
            assert len(matches) == 1
            self._write_ready(matches[0], str(self.next_job_id))
            return subprocess.CompletedProcess(
                argv, 0, f"{self.next_job_id}\n", ""
            )
        if argv and argv[0] == "scancel":
            self.cancellations.append(argv[1])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                argv, 0, "PreemptType = preempt/partition_prio\n", ""
            )
        if argv[:4] == [
            "scontrol",
            "show",
            "partition",
            "ou_bcs_normal",
        ]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "PartitionName=ou_bcs_normal PreemptMode=OFF State=UP "
                "MaxTime=1-00:00:00 TotalCPUs=3648 "
                "TRES=cpu=3648,mem=26667127M,node=19,gres/gpu=131\n",
                "",
            )
        if argv[:5] == ["sacctmgr", "-nP", "show", "qos", "normal"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "normal|cluster|"
                + (
                    ""
                    if self.qos_max_jobs is None
                    else str(self.qos_max_jobs)
                )
                + "|"
                + (
                    ""
                    if self.qos_max_submit is None
                    else str(self.qos_max_submit)
                )
                + f"|{self.qos_max_wall}"
                + "\n",
                "",
            )
        if argv[:4] == ["sacctmgr", "-nP", "show", "assoc"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                (
                    "cluster|account|tester|normal|"
                    + (
                        ""
                        if self.association_max_jobs is None
                        else str(self.association_max_jobs)
                    )
                    + "|"
                    f"{self.association_max_submit}\n"
                ),
                "",
            )
        if (
            argv[:5] == ["squeue", "-h", "-r", "-u", "tester"]
            and argv[-2:] == ["-o", "%i|%T|%a|%q|%k"]
        ):
            rows = [
                f"{job_id}|{state}|{account}|{qos}|{comment}"
                for job_id, state, account, qos, comment in self.existing_jobs
            ]
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(rows) + ("\n" if rows else ""),
                "",
            )
        if (
            argv
            and argv[0] == "sacct"
            and argv[-2:]
            == ["-o", "JobID,State,Account,QOS,Comment"]
        ):
            requested = (
                None
                if "-j" not in argv
                else set(argv[argv.index("-j") + 1].split(","))
            )
            rows = [
                f"{job_id}|{state}|{account}|{qos}|{comment}"
                for job_id, state, account, qos, comment in self.existing_jobs
                if requested is None or job_id in requested
            ]
            return subprocess.CompletedProcess(
                argv,
                0,
                "\n".join(rows) + ("\n" if rows else ""),
                "",
            )
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            job_id = argv[4]
            record = self._record(job_id)
            return subprocess.CompletedProcess(
                argv,
                0,
                f"JobId={job_id} ArrayJobId={job_id} Requeue=0 "
                f"TimeLimit={record['time_limit']} "
                f"Comment={record['comment']}\n",
                "",
            )
        if (
            len(argv) == 5
            and argv[:3] == ["scontrol", "write", "batch_script"]
            and argv[4] == "-"
        ):
            job_id = argv[3]
            record = self._record(job_id)
            return subprocess.CompletedProcess(
                argv,
                0,
                Path(str(record["script"])).read_text(encoding="utf-8"),
                "",
            )
        if argv and argv[0] in {"squeue", "sacct"}:
            job_id = argv[argv.index("-j") + 1]
            record = self._record(job_id)
            role = str(record["role"])
            tasks = int(record["tasks"])
            state = "PENDING" if role == "reserve" else "RUNNING"
            cpus_per_task = int(record["cpus"]) // tasks
            memory_per_task = int(record["memory_mib"]) // tasks
            gpus_per_task = int(record["gpus"]) // tasks
            memory = self._memory_text(memory_per_task)
            tres = (
                f"gres/gpu:a100:{gpus_per_task}"
                if gpus_per_task
                else "N/A"
            )
            if role == "client":
                memory = self.client_memory
            rows: list[str] = []
            if argv[0] == "sacct":
                assert argv[:4] == ["sacct", "-nP", "--array", "-j"]
                assert argv[-2:] == [
                    "-o",
                    (
                        "JobID,State,Partition,QOS,ReqCPUS,ReqMem,"
                        "ReqTRES,Timelimit,Comment"
                    ),
                ]
                # Real Slurm may retain a consolidated array-parent row alongside
                # the expanded logical task IDs.  The parent is provenance, not an
                # additional allocation element.
                rows.append(
                    f"{job_id}|{state}|ou_bcs_normal|normal|"
                    f"{record['cpus']}|{record['memory_mib']}M|"
                    f"cpu={record['cpus']},mem={record['memory_mib']}M|"
                    f"{record['time_limit']}|"
                    f"{record['comment']}"
                )
            for task_id in range(tasks):
                if argv[0] == "squeue":
                    rows.append(
                        f"{job_id}|{task_id}|{state}|ou_bcs_normal|normal|"
                        f"{cpus_per_task}|{memory}|{tres}|"
                        f"{record['time_limit']}|{record['comment']}"
                    )
                else:
                    req_tres = (
                        f"cpu={cpus_per_task},mem="
                        f"{memory},gres/gpu:a100={gpus_per_task}"
                        if role in {"server_active", "server_warm"}
                        else f"cpu={cpus_per_task},mem={memory}"
                    )
                    rows.append(
                        f"{job_id}_{task_id}|{state}|ou_bcs_normal|normal|"
                        f"{cpus_per_task}|{memory}|{req_tres}|"
                        f"{record['time_limit']}|{record['comment']}"
                    )
            return subprocess.CompletedProcess(
                argv, 0, "\n".join(rows) + "\n", ""
            )
        return subprocess.CompletedProcess(argv, 2, "", "unexpected command")


def test_dry_run_is_nonmutating_and_every_array_is_at_most_24(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capacity"
    plan = _plan(root)

    report = builder.prepare(plan, apply=False)

    assert report["status"] == "dry_run"
    assert not root.exists()
    roles = report["plan"]["roles"]
    expected_chunks = {
        "server_active": 22,
        "server_warm": 3,
        "client": 16,
        "reserve": 2,
    }
    assert {
        role: len(record["chunks"]) for role, record in roles.items()
    } == expected_chunks
    assert roles["server_active"]["tasks"] == 22
    assert roles["server_warm"]["tasks"] == 3
    assert roles["client"]["tasks"] == 384
    assert roles["reserve"]["tasks"] == 39
    assert sum(record["tasks"] for record in roles.values()) == 448
    assert report["plan"]["expected_total_job_elements"] == 448
    release_worktree = Path(str(plan["release_worktree"]))
    assert plan["source_tree_sha256"] == builder.sha256_tree(
        release_worktree
    )
    assert plan["dispatcher_source_sha256"] == builder.sha256_file(
        release_worktree / "slurm" / "dispatch_sweeps.py"
    )
    assert plan["qualification_runner_source_sha256"] == builder.sha256_file(
        release_worktree
        / "scripts"
        / "run_schema5_throughput_qualification.py"
    )
    certificate = json.loads(
        Path(
            str(plan["static_feasibility_certificate"]["path"])
        ).read_text(encoding="utf-8")
    )
    assert certificate["source_tree_sha256"] == plan["source_tree_sha256"]
    assert (
        certificate["dispatcher_source_sha256"]
        == plan["dispatcher_source_sha256"]
    )
    assert (
        certificate["qualification_runner_source_sha256"]
        == plan["qualification_runner_source_sha256"]
    )
    assert report["plan"]["job_element_accounting"] == {
        "cell_job_elements": 384,
        "active_server_job_elements": 22,
        "warm_turnover_job_elements": 3,
        "controller_monitor_other_held_job_elements": 39,
        "total_non_cell_reserve_job_elements": 64,
        "total_canary_job_elements": 448,
    }
    for role, record in roles.items():
        assert sum(chunk["tasks"] for chunk in record["chunks"]) == record["tasks"]
        assert all(
            1 <= chunk["tasks"] <= builder.MAX_ARRAY_TASKS
            for chunk in record["chunks"]
        )
        for chunk in record["chunks"]:
            script = plan["_scripts"][chunk["key"]]
            assert "#SBATCH --partition=ou_bcs_normal" in script
            assert "#SBATCH --qos=normal" in script
            assert "#SBATCH --no-requeue" in script
            assert (
                "#SBATCH --time=1-00:00:00" in script
                if role in {"server_active", "server_warm"}
                else "#SBATCH --time=12:00:00" in script
            )
            assert (
                f"#SBATCH --array=0-{chunk['tasks'] - 1}%{chunk['tasks']}"
                in script
            )


def test_builder_supplies_all_frozen_source_hashes_to_certificate_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []
    original = builder.runtime_capacity.load_static_feasibility_certificate

    def load(*args, **kwargs):
        observed.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        builder.runtime_capacity,
        "load_static_feasibility_certificate",
        load,
    )
    plan = _plan(tmp_path / "capacity")

    assert len(observed) == 1
    assert observed[0]["expected_source_tree_sha256"] == plan[
        "source_tree_sha256"
    ]
    assert observed[0]["expected_dispatcher_source_sha256"] == plan[
        "dispatcher_source_sha256"
    ]
    assert observed[0][
        "expected_qualification_runner_source_sha256"
    ] == plan["qualification_runner_source_sha256"]


def test_prepare_is_idempotent_and_persists_all_chunk_intents(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)

    builder.prepare(plan, apply=True)

    ledger = json.loads(
        (root / builder.LEDGER_FILENAME).read_text(encoding="utf-8")
    )
    assert len(ledger["jobs"]) == 43
    assert len({row["comment"] for row in ledger["jobs"].values()}) == 43
    scripts = sorted((root / "sbatch").glob("*.sbatch"))
    assert len(scripts) == 43
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in scripts)
    intent_path = root / builder.INTENT_FILENAME
    assert stat.S_IMODE(intent_path.stat().st_mode) == 0o444
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    identity = dict(intent)
    plan_id = identity.pop("plan_id")
    assert plan_id == builder.sha256_bytes(builder.canonical_bytes(identity))
    assert intent["builder_source_sha256"] == builder.sha256_file(
        Path(builder.__file__).resolve()
    )
    assert intent["publisher_source_sha256"] == builder.sha256_file(
        Path(publisher.__file__).resolve()
    )
    assert intent["fleet_contract_sha256"] == intent[
        "effective_fleet_contract_sha256"
    ]
    assert intent["base_fleet_contract_sha256"] == FLEET_SHA256


def test_apply_submits_and_observes_full_concurrent_capacity_then_publishes(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)
    scheduler = FakeScheduler(root)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    report = builder.apply_transaction(
        root=root,
        recovery_root=recovery_root,
        runner=scheduler,
        now=lambda: 1_000.0,
        sleep=lambda _seconds: None,
        deadline_seconds=1.0,
        occupancy_observation_interval_seconds=0.0,
    )

    assert report["status"] == "complete"
    assert len(scheduler.submissions) == 43
    assert sum(
        int(chunk["gpus"])
        for record in plan["roles"].values()
        for chunk in record["chunks"]
        if record["gpus"]
    ) == 28
    assert sum(
        int(chunk["tasks"])
        for chunk in plan["roles"]["client"]["chunks"]
    ) == 384
    assert sum(
        int(chunk["tasks"])
        for chunk in plan["roles"]["reserve"]["chunks"]
    ) == 39
    assert sum(
        int(record["tasks"]) for record in plan["roles"].values()
    ) == 448
    assert len(scheduler.cancellations) == 2
    marker = publisher.verify_marker(
        recovery_root,
        expected_release_git_commit=COMMIT,
        expected_release_tag_object=TAG_OBJECT,
        expected_source_tree_sha256=str(plan["source_tree_sha256"]),
        expected_dispatcher_source_sha256=str(
            plan["dispatcher_source_sha256"]
        ),
        expected_qualification_runner_source_sha256=str(
            plan["qualification_runner_source_sha256"]
        ),
    )
    assert marker["active_gpus"] == 24
    assert marker["warm_headroom_gpus"] == 4
    assert marker["cell_ceiling"] == 384
    assert marker["reserve_jobs"] == 64
    assert marker["submit_headroom"] == 448
    scheduler_evidence = json.loads(
        (root / builder.SCHEDULER_EVIDENCE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert scheduler_evidence["scheduler_max_submit_jobs"] == 448
    assert scheduler_evidence["expected_total_job_elements"] == 448
    assert scheduler_evidence["job_element_accounting"] == (
        plan["job_element_accounting"]
    )
    for filename in (
        builder.OCCUPANCY_PREFLIGHT_FILENAME,
        builder.SCHEDULER_EVIDENCE_FILENAME,
        builder.CANARY_EVIDENCE_FILENAME,
    ):
        path = root / filename
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_higher_association_limit_still_runs_exact_448_element_canary(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)
    scheduler = FakeScheduler(root, association_max_submit=500)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    report = builder.apply_transaction(
        root=root,
        recovery_root=recovery_root,
        runner=scheduler,
        now=lambda: 1_000.0,
        sleep=lambda _seconds: None,
        deadline_seconds=1.0,
        occupancy_observation_interval_seconds=0.0,
    )

    assert report["status"] == "complete"
    assert sum(
        int(role["tasks"]) for role in plan["roles"].values()
    ) == 448
    evidence = json.loads(
        (root / builder.SCHEDULER_EVIDENCE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert evidence["scheduler_max_submit_jobs"] == 500
    assert evidence["expected_total_job_elements"] == 448
    marker = publisher.verify_marker(
        recovery_root,
        expected_release_git_commit=COMMIT,
        expected_release_tag_object=TAG_OBJECT,
        expected_source_tree_sha256=str(plan["source_tree_sha256"]),
        expected_dispatcher_source_sha256=str(
            plan["dispatcher_source_sha256"]
        ),
        expected_qualification_runner_source_sha256=str(
            plan["qualification_runner_source_sha256"]
        ),
    )
    assert marker["submit_headroom"] == 448


def test_exact_448_canary_includes_existing_global_occupancy(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    existing = tuple(
        (str(10_000 + index), "RUNNING", f"unrelated:{index}")
        for index in range(52)
    )
    scheduler = FakeScheduler(
        root,
        association_max_submit=500,
        existing_jobs=existing,
    )
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    report = builder.apply_transaction(
        root=root,
        recovery_root=recovery_root,
        runner=scheduler,
        now=lambda: 1_000.0,
        sleep=lambda _seconds: None,
        deadline_seconds=1.0,
        occupancy_observation_interval_seconds=0.0,
    )

    assert report["status"] == "complete"
    assert len(scheduler.submissions) == 43
    ledger = builder._load_ledger(root, plan=builder._load_plan(root))
    preflight = ledger["occupancy_preflight"]
    assert preflight["existing_job_elements"] == 52
    assert preflight["effective_max_submit_jobs"] == 500
    assert (
        preflight["existing_job_elements"]
        + preflight["required_new_job_elements"]
        == 500
    )


@pytest.mark.parametrize(
    ("association_limit", "qos_limit", "existing_count"),
    (
        (500, None, 53),
        (448, None, 1),
        (500, 499, 52),
    ),
)
def test_occupancy_over_effective_limit_fails_before_first_sbatch(
    tmp_path: Path,
    association_limit: int,
    qos_limit: int | None,
    existing_count: int,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    scheduler = FakeScheduler(
        root,
        association_max_submit=association_limit,
        qos_max_submit=qos_limit,
        existing_jobs=tuple(
            (str(11_000 + index), "RUNNING", f"unrelated:{index}")
            for index in range(existing_count)
        ),
    )
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="existing user job elements plus 448",
    ):
        builder.apply_transaction(
            root=root,
            recovery_root=recovery_root,
            runner=scheduler,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            deadline_seconds=1.0,
            occupancy_observation_interval_seconds=0.0,
        )

    assert scheduler.submissions == []


def test_occupancy_preflight_observes_complete_truth_sixty_seconds_apart(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)
    scheduler = FakeScheduler(root, association_max_submit=500)
    clock = [1_000.0]
    sleeps: list[float] = []

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    receipt = builder._verify_submit_headroom_preflight(
        plan=plan,
        runner=scheduler,
        now=lambda: clock[0],
        sleep=advance,
    )

    assert sleeps == [60.0]
    assert (
        receipt["second_observation"]["observed_at"]
        - receipt["first_observation"]["observed_at"]
        == 60.0
    )
    assert receipt["existing_job_elements"] == 0
    assert receipt["effective_max_submit_jobs"] == 500


def test_occupancy_drift_between_complete_observations_fails_closed(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)
    scheduler = FakeScheduler(root, association_max_submit=500)
    queue_observations = 0

    def drifting_runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal queue_observations
        if (
            argv[:5] == ["squeue", "-h", "-r", "-u", "tester"]
            and argv[-2:] == ["-o", "%i|%T|%a|%q|%k"]
        ):
                queue_observations += 1
                if queue_observations == 2:
                    scheduler.existing_jobs = (
                        (
                            "19000",
                            "RUNNING",
                            "account",
                            "normal",
                            "unrelated:new",
                        ),
                    )
        return scheduler(argv, timeout=timeout)

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="occupancy changed",
    ):
        builder._verify_submit_headroom_preflight(
            plan=plan,
            runner=drifting_runner,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            observation_interval_seconds=0.0,
        )


def test_occupancy_uses_expanded_logical_array_ids_and_ignores_parent_summary(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path / "capacity")
    commands: list[list[str]] = []

    def runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        commands.append(list(argv))
        if argv[0] == "squeue":
            assert argv[-2:] == ["-o", "%i|%T|%a|%q|%k"]
            return subprocess.CompletedProcess(
                argv,
                0,
                "700_0|PENDING|account|normal|unrelated:array\n"
                "700_1|RUNNING|account|normal|unrelated:array\n",
                "",
            )
        assert argv[:4] == ["sacct", "-nP", "-X", "--array"]
        assert argv[-2:] == [
            "-o",
            "JobID,State,Account,QOS,Comment",
        ]
        assert "JobIDRaw" not in argv[-1]
        # This is shaped like `sacct --array`: one optional consolidated parent
        # plus the scheduler-logical array_jobid_taskid rows.  Raw allocation IDs
        # can instead be unrelated values and therefore must never be requested.
        return subprocess.CompletedProcess(
            argv,
            0,
            "700|PENDING|account|normal|unrelated:array\n"
            "700_0|PENDING|account|normal|unrelated:array\n"
            "700_1|RUNNING|account|normal|unrelated:array\n",
            "",
        )

    observation = builder._capture_current_user_occupancy(
        plan=plan,
        runner=runner,
        observed_at=1_000.0,
    )

    assert len(commands) == 2
    assert observation["job_elements"] == 2
    assert [row["job_id"] for row in observation["jobs"]] == [
        "700_0",
        "700_1",
    ]


def test_association_limit_below_448_fails_before_any_submission(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    scheduler = FakeScheduler(root, association_max_submit=447)
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="existing user job elements plus 448",
    ):
        builder.apply_transaction(
            root=root,
            recovery_root=recovery_root,
            runner=scheduler,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            deadline_seconds=1.0,
            occupancy_observation_interval_seconds=0.0,
        )

    assert scheduler.submissions == []
    assert not (recovery_root / publisher.MARKER_FILENAME).exists()
    assert not (root / builder.SCHEDULER_EVIDENCE_FILENAME).exists()


def test_raw_client_resource_drift_blocks_evidence_before_publication(
    tmp_path: Path,
) -> None:
    root, plan = _prepared(tmp_path)
    scheduler = FakeScheduler(root, client_memory="2G")
    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="client.*resource request drifted",
    ):
        builder.apply_transaction(
            root=root,
            recovery_root=recovery_root,
            runner=scheduler,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            deadline_seconds=1.0,
            occupancy_observation_interval_seconds=0.0,
        )

    assert not (recovery_root / publisher.MARKER_FILENAME).exists()


def test_cli_plan_is_an_explicit_nonmutating_operation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "capacity"
    monkeypatch.setattr(
        builder,
        "_verify_release_checkout",
        lambda **_kwargs: _mock_release_identity(tmp_path),
    )

    assert (
        builder.main(
            [
                "plan",
                "--root",
                str(root),
                "--release-git-commit",
                COMMIT,
                "--release-tag-object",
                TAG_OBJECT,
                "--scheduler-user",
                "tester",
                "--token",
                TOKEN,
                "--fleet-contract",
                str(FLEET_PATH),
                "--fleet-contract-sha256",
                FLEET_SHA256,
                *_capacity_cli_arguments(tmp_path),
                "--model-contract",
                str(MODEL_PATH),
                "--model-contract-sha256",
                MODEL_SHA256,
                "--release-worktree",
                str(REPOSITORY_ROOT),
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert not root.exists()


def test_cli_run_routes_to_the_resumable_real_operation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "capacity"
    recovery_root = tmp_path / "recovery"
    observed: dict[str, object] = {}

    def fake_run_operation(**kwargs):
        observed.update(kwargs)
        return {"status": "complete", "marker": {"marker_id": "4" * 64}}

    monkeypatch.setattr(builder, "run_operation", fake_run_operation)
    monkeypatch.setattr(
        builder,
        "_verify_release_checkout",
        lambda **_kwargs: _mock_release_identity(tmp_path),
    )

    assert (
        builder.main(
            [
                "run",
                "--root",
                str(root),
                "--recovery-root",
                str(recovery_root),
                "--release-git-commit",
                COMMIT,
                "--release-tag-object",
                TAG_OBJECT,
                "--scheduler-user",
                "tester",
                "--token",
                TOKEN,
                "--fleet-contract",
                str(FLEET_PATH),
                "--fleet-contract-sha256",
                FLEET_SHA256,
                *_capacity_cli_arguments(tmp_path),
                "--model-contract",
                str(MODEL_PATH),
                "--model-contract-sha256",
                MODEL_SHA256,
                "--release-worktree",
                str(REPOSITORY_ROOT),
                "--apply",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert observed["root"] == root
    assert observed["recovery_root"] == recovery_root
    assert observed["partition"] == "ou_bcs_normal"
    assert observed["qos"] == "normal"


def test_run_without_apply_is_nonmutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "capacity"
    recovery_root = tmp_path / "recovery"
    monkeypatch.setattr(
        builder,
        "_verify_release_checkout",
        lambda **_kwargs: _mock_release_identity(tmp_path),
    )
    arguments = [
        "run",
        "--root",
        str(root),
        "--recovery-root",
        str(recovery_root),
        "--release-git-commit",
        COMMIT,
        "--release-tag-object",
        TAG_OBJECT,
        "--scheduler-user",
        "tester",
        "--token",
        TOKEN,
        "--fleet-contract",
        str(FLEET_PATH),
        "--fleet-contract-sha256",
        FLEET_SHA256,
        "--model-contract",
        str(MODEL_PATH),
        "--model-contract-sha256",
        MODEL_SHA256,
        "--release-worktree",
        str(REPOSITORY_ROOT),
        *_capacity_cli_arguments(tmp_path),
    ]

    assert builder.main(arguments) == 0

    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert not root.exists()
    assert not recovery_root.exists()


@pytest.mark.parametrize(
    "filename",
    (
        builder.INTENT_FILENAME,
        builder.SCHEDULER_EVIDENCE_FILENAME,
        builder.RELEASE_FILENAME,
    ),
)
def test_sealed_publication_recovers_linked_interrupted_temporary(
    tmp_path: Path,
    filename: str,
) -> None:
    payload = {"filename": filename, "passed": True}
    target = tmp_path / filename
    interrupted = tmp_path / f".{filename}.crash.publishing"
    interrupted.write_bytes(builder.canonical_bytes(payload))
    interrupted.chmod(0o444)
    os.link(interrupted, target)
    assert target.stat().st_nlink == 2

    builder._publish_sealed_once(target, payload)

    assert not interrupted.exists()
    assert target.read_bytes() == builder.canonical_bytes(payload)
    assert target.stat().st_nlink == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o444


@pytest.mark.parametrize(
    "filename",
    (
        builder.INTENT_FILENAME,
        builder.SCHEDULER_EVIDENCE_FILENAME,
        builder.RELEASE_FILENAME,
    ),
)
def test_sealed_publication_adopts_exact_orphan_temporary(
    tmp_path: Path,
    filename: str,
) -> None:
    payload = {"filename": filename, "passed": True}
    target = tmp_path / filename
    interrupted = tmp_path / f".{filename}.crash.publishing"
    interrupted.write_bytes(builder.canonical_bytes(payload))
    interrupted.chmod(0o444)

    builder._publish_sealed_once(target, payload)

    assert not interrupted.exists()
    assert target.read_bytes() == builder.canonical_bytes(payload)
    assert target.stat().st_nlink == 1


@pytest.mark.parametrize(
    "filename",
    (
        builder.INTENT_FILENAME,
        builder.SCHEDULER_EVIDENCE_FILENAME,
        builder.RELEASE_FILENAME,
    ),
)
def test_sealed_publication_rejects_conflicting_interrupted_temporary(
    tmp_path: Path,
    filename: str,
) -> None:
    payload = {"filename": filename, "passed": True}
    target = tmp_path / filename
    builder._publish_sealed_once(target, payload)
    interrupted = tmp_path / f".{filename}.conflict.publishing"
    interrupted.write_bytes(
        builder.canonical_bytes({"filename": filename, "passed": False})
    )
    interrupted.chmod(0o444)

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="conflicting interrupted",
    ):
        builder._publish_sealed_once(target, payload)

    assert target.read_bytes() == builder.canonical_bytes(payload)
    assert target.stat().st_nlink == 1


@pytest.mark.parametrize("mutation", ("duplicate", "noncanonical", "nan"))
def test_sealed_intent_rejects_noncanonical_or_ambiguous_json(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    path = root / builder.INTENT_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        raw = path.read_text(encoding="utf-8").replace(
            '  "schema_version": 1,\n',
            '  "schema_version": 1,\n  "schema_version": 1,\n',
            1,
        )
    elif mutation == "noncanonical":
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        raw = path.read_text(encoding="utf-8").replace(
            '  "schema_version": 1,\n',
            '  "schema_version": NaN,\n',
            1,
        )
    path.chmod(0o644)
    path.write_text(raw, encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(builder.ProtectedCapacityBuildError):
        builder._load_plan(root)


def test_mutable_ledger_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    plan = builder._load_plan(root)
    path = root / builder.LEDGER_FILENAME
    canonical = path.read_text(encoding="utf-8")
    for raw in (
        canonical.replace(
            '  "schema_version": 1\n',
            '  "schema_version": 1,\n  "schema_version": 1\n',
            1,
        ),
        canonical.replace(
            '  "schema_version": 1\n',
            '  "schema_version": NaN\n',
            1,
        ),
    ):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(builder.ProtectedCapacityBuildError):
            builder._load_ledger(root, plan=plan)
        path.write_text(canonical, encoding="utf-8")


def test_release_worktree_symlink_traversal_is_rejected(tmp_path: Path) -> None:
    release_link = tmp_path / "release"
    release_link.symlink_to(REPOSITORY_ROOT, target_is_directory=True)
    capacity = _capacity_authority(tmp_path)
    capacity.pop("base_fleet_contract_path")
    capacity.pop("base_fleet_contract_sha256")
    capacity.pop("model_contract_path")
    capacity.pop("model_contract_sha256")
    capacity.pop("release_worktree")

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="release worktree traverses a symlink",
    ):
        builder.build_plan(
            root=tmp_path / "capacity",
            release_git_commit=COMMIT,
            release_tag_object=TAG_OBJECT,
            partition="ou_bcs_normal",
            qos="normal",
            scheduler_user="tester",
            token=TOKEN,
            fleet_contract_path=FLEET_PATH,
            fleet_contract_sha256=FLEET_SHA256,
            model_contract_path=MODEL_PATH,
            model_contract_sha256=MODEL_SHA256,
            release_worktree=release_link,
            **capacity,
            identity_runner=_identity_runner,
        )


def test_release_contract_symlink_traversal_is_rejected(tmp_path: Path) -> None:
    fleet_link = tmp_path / "schema5_fleet.v1.json"
    fleet_link.symlink_to(FLEET_PATH)
    capacity = _capacity_authority(tmp_path)
    capacity.pop("base_fleet_contract_path")
    capacity.pop("base_fleet_contract_sha256")
    capacity.pop("model_contract_path")
    capacity.pop("model_contract_sha256")
    capacity.pop("release_worktree")

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="supplied fleet contract traverses a symlink",
    ):
        builder.build_plan(
            root=tmp_path / "capacity",
            release_git_commit=COMMIT,
            release_tag_object=TAG_OBJECT,
            partition="ou_bcs_normal",
            qos="normal",
            scheduler_user="tester",
            token=TOKEN,
            fleet_contract_path=fleet_link,
            fleet_contract_sha256=FLEET_SHA256,
            model_contract_path=MODEL_PATH,
            model_contract_sha256=MODEL_SHA256,
            release_worktree=REPOSITORY_ROOT,
            **capacity,
            identity_runner=_identity_runner,
        )


def test_exact_tag_and_clean_checkout_are_required(tmp_path: Path) -> None:
    def dirty_runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        result = _identity_runner(argv, timeout=timeout)
        if argv[3:] == [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]:
            return subprocess.CompletedProcess(argv, 0, " M dirty.py\n", "")
        return result

    capacity = _capacity_authority(tmp_path)
    capacity.pop("base_fleet_contract_path")
    capacity.pop("base_fleet_contract_sha256")
    capacity.pop("model_contract_path")
    capacity.pop("model_contract_sha256")
    capacity.pop("release_worktree")
    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="exact clean annotated",
    ):
        builder.build_plan(
            root=tmp_path / "capacity",
            release_git_commit=COMMIT,
            release_tag_object=TAG_OBJECT,
            partition="ou_bcs_normal",
            qos="normal",
            scheduler_user="tester",
            token=TOKEN,
            fleet_contract_path=FLEET_PATH,
            fleet_contract_sha256=FLEET_SHA256,
            model_contract_path=MODEL_PATH,
            model_contract_sha256=MODEL_SHA256,
            release_worktree=REPOSITORY_ROOT,
            **capacity,
            identity_runner=dirty_runner,
        )
    assert not (tmp_path / "capacity").exists()


def test_release_identity_replay_rejects_concurrent_tag_ref_move(
    tmp_path: Path,
) -> None:
    authority = _capacity_authority(tmp_path)
    release_worktree = Path(str(authority["release_worktree"]))
    tag_reads = 0

    def moved_tag_runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal tag_reads
        arguments = argv[3:]
        if arguments == [
            "rev-parse",
            f"refs/tags/{publisher.RELEASE_TAG}",
        ]:
            tag_reads += 1
            value = TAG_OBJECT if tag_reads == 1 else "9" * 40
            return subprocess.CompletedProcess(argv, 0, f"{value}\n", "")
        if arguments == ["cat-file", "-t", "9" * 40]:
            return subprocess.CompletedProcess(argv, 0, "tag\n", "")
        return _identity_runner(argv, timeout=timeout)

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="release tree drifted",
    ):
        builder._verify_release_checkout(
            release_worktree=release_worktree,
            release_git_commit=COMMIT,
            release_tag_object=TAG_OBJECT,
            runner=moved_tag_runner,
        )
    assert tag_reads == 2


def test_stale_and_unexpected_ready_receipts_fail_closed(tmp_path: Path) -> None:
    root, _plan_value = _prepared(tmp_path)
    plan, ledger = _mark_other_jobs_submitted(
        root,
        target_key="server_active:000",
        target_state="submitting",
    )
    record = ledger["jobs"]["server_active:000"]
    record.update(
        {
            "state": "submitted",
            "job_id": "44444",
            "submitted_at": 1_000.0,
            "submission_receipt_at": 1_000.0,
        }
    )
    builder._atomic_json(root / builder.LEDGER_FILENAME, ledger)
    ledger = builder._load_ledger(root, plan=plan)
    directory = root / builder.READY_DIRECTORY / "server_active" / "000"
    directory.mkdir(parents=True)
    stale = directory / "0"
    stale.write_text("{}\n", encoding="utf-8")
    stale.chmod(0o444)

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="differs from live ledger",
    ):
        builder._ready_count(
            root,
            "server_active",
            plan=plan,
            ledger=ledger,
        )

    stale.chmod(0o644)
    stale.unlink()
    unexpected = directory / "99"
    unexpected.write_text("{}\n", encoding="utf-8")
    unexpected.chmod(0o444)
    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="unexpected stale receipt",
    ):
        builder._ready_count(
            root,
            "server_active",
            plan=plan,
            ledger=ledger,
        )


def test_accepted_job_is_adopted_after_ambiguous_sbatch_boundary(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    plan, _ledger = _mark_other_jobs_submitted(
        root,
        target_key="server_active:000",
        target_state="prepared",
    )
    scheduler = ReconcileScheduler(root)
    scheduler.fail_after_accept = True

    with pytest.raises(builder.ProtectedCapacityBuildError):
        builder._submit_jobs(
            root,
            runner=scheduler,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
        )
    assert scheduler.sbatch_calls == 1

    ledger = builder._submit_jobs(
        root,
        runner=scheduler,
        now=lambda: 1_001.0,
        sleep=lambda _seconds: None,
        reconciliation_interval_seconds=0.0,
    )

    adopted = ledger["jobs"]["server_active:000"]
    assert adopted["state"] == "submitted"
    assert adopted["job_id"] in scheduler.accepted
    assert scheduler.sbatch_calls == 1
    assert adopted["attempt"] == 1


def test_crash_after_job_id_before_ledger_commit_is_adopted_once(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    _mark_other_jobs_submitted(
        root,
        target_key="server_active:000",
        target_state="prepared",
    )
    scheduler = ReconcileScheduler(root)

    with pytest.raises(RuntimeError, match="crash after ID"):
        builder._submit_jobs(
            root,
            runner=scheduler,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            after_sbatch=lambda _record, _job_id: (_ for _ in ()).throw(
                RuntimeError("crash after ID")
            ),
        )
    assert scheduler.sbatch_calls == 1

    ledger = builder._submit_jobs(
        root,
        runner=scheduler,
        now=lambda: 1_001.0,
        sleep=lambda _seconds: None,
        reconciliation_interval_seconds=0.0,
    )
    assert ledger["jobs"]["server_active:000"]["state"] == "submitted"
    assert scheduler.sbatch_calls == 1


def test_terminal_comment_matched_job_is_never_silently_resubmitted(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    _mark_other_jobs_submitted(
        root,
        target_key="server_active:000",
        target_state="submitting",
    )
    scheduler = ReconcileScheduler(root)
    scheduler.accept("server_active:000", state="FAILED")

    with pytest.raises(
        builder.ProtectedCapacityBuildError,
        match="terminal comment-matched",
    ):
        builder._submit_jobs(
            root,
            runner=scheduler,
            now=lambda: 1_001.0,
            sleep=lambda _seconds: None,
            reconciliation_interval_seconds=0.0,
        )

    assert scheduler.sbatch_calls == 0


def test_cleanup_reconciles_scancel_crash_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root, _plan_value = _prepared(tmp_path)
    plan = builder._load_plan(root)
    ledger = builder._load_ledger(root, plan=plan)
    next_job = 50_000
    for record in ledger["jobs"].values():
        next_job += 1
        record.update(
            {
                "state": "submitted",
                "job_id": str(next_job),
                "submitted_at": 900.0,
                "submission_intent_at": 899.0,
                "submission_receipt_at": 900.0,
                "attempt": 1,
            }
        )
    builder._atomic_json(root / builder.LEDGER_FILENAME, ledger)
    expected_cancel_calls = sum(
        1
        for record in ledger["jobs"].values()
        if record["role"] == "reserve"
    )
    terminal: dict[str, tuple[str, str]] = {}
    scancel_calls: list[str] = []

    def runner(
        argv: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        if argv[0] == "scancel":
            job_id = argv[1]
            scancel_calls.append(job_id)
            record = next(
                row for row in ledger["jobs"].values() if row["job_id"] == job_id
            )
            terminal[job_id] = ("CANCELLED", str(record["comment"]))
            if len(scancel_calls) == 1:
                return subprocess.CompletedProcess(
                    argv, 1, "", "connection dropped after acceptance"
                )
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:4] == ["sacct", "-nP", "--array", "-j"]:
            job_id = argv[4]
            assert argv[-2:] == ["-o", "JobID,State,Comment"]
            state, comment = terminal[job_id]
            return subprocess.CompletedProcess(
                argv, 0, f"{job_id}|{state}|{comment}\n", ""
            )
        return subprocess.CompletedProcess(argv, 2, "", "unexpected")

    marker = {"marker_id": "9" * 64}
    first = builder._cleanup_transaction(
        root=root,
        marker=marker,
        runner=runner,
        now=lambda: 1_000.0,
    )
    call_count = len(scancel_calls)
    second = builder._cleanup_transaction(
        root=root,
        marker=marker,
        runner=runner,
        now=lambda: 1_001.0,
    )

    assert call_count == expected_cancel_calls
    assert len(scancel_calls) == call_count
    assert first == second
    assert {
        record["cleanup_state"] for record in second["jobs"].values()
    } == {"released", "cancelled"}


def test_builder_cli_imports_under_isolated_production_python() -> None:
    interpreter = Path(
        "/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python"
    )
    if not interpreter.is_file():
        pytest.skip("immutable harness pilot interpreter is unavailable")
    result = subprocess.run(
        [
            str(interpreter),
            "-I",
            str(Path(builder.__file__).resolve()),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "run --apply" in result.stdout
