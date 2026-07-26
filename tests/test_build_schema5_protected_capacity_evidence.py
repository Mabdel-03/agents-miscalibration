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


COMMIT = "1" * 40
TAG_OBJECT = "2" * 40
TOKEN = "3" * 32
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLEET_PATH = REPOSITORY_ROOT / "configs" / "schema5_fleet.v1.json"
MODEL_PATH = REPOSITORY_ROOT / "configs" / "model_contracts.v1.json"
FLEET_SHA256 = builder.sha256_file(FLEET_PATH)
MODEL_SHA256 = builder.sha256_file(MODEL_PATH)


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


def _plan(root: Path) -> dict[str, object]:
    return builder.build_plan(
        root=root,
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
        association_max_submit: int = 448,
        qos_max_submit: int | None = None,
        existing_jobs: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        self.root = root
        self.client_memory = client_memory
        self.association_max_submit = association_max_submit
        self.qos_max_submit = qos_max_submit
        self.existing_jobs = existing_jobs
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
                "normal|cluster||"
                + (
                    ""
                    if self.qos_max_submit is None
                    else str(self.qos_max_submit)
                )
                + "\n",
                "",
            )
        if argv[:4] == ["sacctmgr", "-nP", "show", "assoc"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                (
                    "cluster|account|tester|normal||"
                    f"{self.association_max_submit}\n"
                ),
                "",
            )
        if (
            argv[:5] == ["squeue", "-h", "-r", "-u", "tester"]
            and argv[-2:] == ["-o", "%i|%T|%k"]
        ):
            rows = [
                f"{job_id}|{state}|{comment}"
                for job_id, state, comment in self.existing_jobs
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
            and argv[-2:] == ["-o", "JobID,State,Comment"]
        ):
            requested = (
                None
                if "-j" not in argv
                else set(argv[argv.index("-j") + 1].split(","))
            )
            rows = [
                f"{job_id}|{state}|{comment}"
                for job_id, state, comment in self.existing_jobs
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
                        "ReqTRES,Comment"
                    ),
                ]
                # Real Slurm may retain a consolidated array-parent row alongside
                # the expanded logical task IDs.  The parent is provenance, not an
                # additional allocation element.
                rows.append(
                    f"{job_id}|{state}|ou_bcs_normal|normal|"
                    f"{record['cpus']}|{record['memory_mib']}M|"
                    f"cpu={record['cpus']},mem={record['memory_mib']}M|"
                    f"{record['comment']}"
                )
            for task_id in range(tasks):
                if argv[0] == "squeue":
                    rows.append(
                        f"{job_id}|{task_id}|{state}|ou_bcs_normal|normal|"
                        f"{cpus_per_task}|{memory}|{tres}|{record['comment']}"
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
                        f"{record['comment']}"
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
                f"#SBATCH --array=0-{chunk['tasks'] - 1}%{chunk['tasks']}"
                in script
            )


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
    assert intent["fleet_contract_sha256"] == FLEET_SHA256


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
            and argv[-2:] == ["-o", "%i|%T|%k"]
        ):
            queue_observations += 1
            if queue_observations == 2:
                scheduler.existing_jobs = (
                    ("19000", "RUNNING", "unrelated:new"),
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
            assert argv[-2:] == ["-o", "%i|%T|%k"]
            return subprocess.CompletedProcess(
                argv,
                0,
                "700_0|PENDING|unrelated:array\n"
                "700_1|RUNNING|unrelated:array\n",
                "",
            )
        assert argv[:4] == ["sacct", "-nP", "-X", "--array"]
        assert argv[-2:] == ["-o", "JobID,State,Comment"]
        assert "JobIDRaw" not in argv[-1]
        # This is shaped like `sacct --array`: one optional consolidated parent
        # plus the scheduler-logical array_jobid_taskid rows.  Raw allocation IDs
        # can instead be unrelated values and therefore must never be requested.
        return subprocess.CompletedProcess(
            argv,
            0,
            "700|PENDING|unrelated:array\n"
            "700_0|PENDING|unrelated:array\n"
            "700_1|RUNNING|unrelated:array\n",
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
        lambda **_kwargs: {
            "release_worktree": str(REPOSITORY_ROOT),
            "builder_source_path": str(Path(builder.__file__).resolve()),
            "builder_source_sha256": builder.sha256_file(
                Path(builder.__file__).resolve()
            ),
            "publisher_source_path": str(Path(publisher.__file__).resolve()),
            "publisher_source_sha256": builder.sha256_file(
                Path(publisher.__file__).resolve()
            ),
        },
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
        lambda **_kwargs: {
            "release_worktree": str(REPOSITORY_ROOT),
            "builder_source_path": str(Path(builder.__file__).resolve()),
            "builder_source_sha256": builder.sha256_file(
                Path(builder.__file__).resolve()
            ),
            "publisher_source_path": str(Path(publisher.__file__).resolve()),
            "publisher_source_sha256": builder.sha256_file(
                Path(publisher.__file__).resolve()
            ),
        },
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
        lambda **_kwargs: {
            "release_worktree": str(REPOSITORY_ROOT),
            "builder_source_path": str(Path(builder.__file__).resolve()),
            "builder_source_sha256": builder.sha256_file(
                Path(builder.__file__).resolve()
            ),
            "publisher_source_path": str(Path(publisher.__file__).resolve()),
            "publisher_source_sha256": builder.sha256_file(
                Path(publisher.__file__).resolve()
            ),
        },
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
            identity_runner=_identity_runner,
        )


def test_release_contract_symlink_traversal_is_rejected(tmp_path: Path) -> None:
    fleet_link = tmp_path / "schema5_fleet.v1.json"
    fleet_link.symlink_to(FLEET_PATH)

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
            identity_runner=dirty_runner,
        )
    assert not (tmp_path / "capacity").exists()


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

    assert call_count == 2
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
