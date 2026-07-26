"""Durability tests for the real Slurm fleet transaction canary."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import hashlib
import subprocess
import time

import pytest

from scripts import run_schema5_slurm_fleet_canary as canary


def test_dependency_scheduler_since_uses_scheduler_local_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 04:09 UTC is 00:09 EDT. A naive UTC value makes sacct reject a
    # future-looking -S argument on the cluster's America/New_York login nodes.
    timestamp = 1_784_952_540.0
    with monkeypatch.context() as timezone_context:
        timezone_context.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            since = canary._dependency_scheduler_since(timestamp)
            commands: list[list[str]] = []

            def runner(
                argv: list[str],
            ) -> subprocess.CompletedProcess[str]:
                commands.append(list(argv))
                return subprocess.CompletedProcess(argv, 0, "", "")

            canary._dependency_scheduler_snapshot(
                slurm_user="tester",
                since=since,
                runner=runner,
            )
        finally:
            timezone_context.undo()
            time.tzset()

    assert since == "2026-07-25T00:09:00"
    sacct = next(argv for argv in commands if argv[0] == "sacct")
    assert sacct[sacct.index("-S") + 1] == since


def _code_identity():
    canary_path = Path(canary.__file__).resolve()
    transaction_path = Path(canary.tx.__file__).resolve()
    publisher_path = Path(canary.durable_git.__file__).resolve()
    return {
        "schema_version": 1,
        "protocol": canary.CODE_IDENTITY_PROTOCOL,
        "release_tag": canary.REQUIRED_RELEASE_TAG,
        "release_git_commit": "1" * 40,
        "release_tag_object": "2" * 40,
        "canary_script": {
            "git_path": canary.CANARY_GIT_PATH,
            "sha256": hashlib.sha256(canary_path.read_bytes()).hexdigest(),
            "size": canary_path.stat().st_size,
        },
        "fleet_transactions": {
            "git_path": canary.FLEET_TRANSACTIONS_GIT_PATH,
            "sha256": hashlib.sha256(transaction_path.read_bytes()).hexdigest(),
            "size": transaction_path.stat().st_size,
        },
        "durable_git_publisher": {
            "git_path": canary.DURABLE_GIT_PUBLISHER_GIT_PATH,
            "sha256": hashlib.sha256(publisher_path.read_bytes()).hexdigest(),
            "size": publisher_path.stat().st_size,
        },
        "durable_git_release": {
            "path": "/sealed/DURABLE_GIT_RELEASE_COMPLETE.json",
            "sha256": "3" * 64,
            "marker_id": "4" * 64,
            "release_git_commit": "1" * 40,
            "release_tag_object": "2" * 40,
            "bundle_sha256": "5" * 64,
        },
    }


class _Process:
    def __init__(self, *, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Clock:
    def __init__(self, value=1_000.0):
        self.value = float(value)

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += float(seconds)


class _FakeSlurm:
    def __init__(
        self,
        root: Path,
        *,
        crash_after_acceptance=False,
        crash_after_scancel=False,
        spooled_mismatch=False,
    ):
        self.root = root
        self.crash_after_acceptance = crash_after_acceptance
        self.crash_after_scancel = crash_after_scancel
        self.spooled_mismatch = spooled_mismatch
        self.accepted = False
        self.active = False
        self.job_id = "424242"
        self.sbatch_calls = 0
        self.scancel_calls: list[list[str]] = []
        self.scontrol_calls: list[list[str]] = []
        self.effective_state_calls: list[list[str]] = []
        self.comment = ""
        self.sbatch_path = ""
        self.job_name = ""
        self.partition = ""

    def _row(self, *, source):
        if not self.accepted:
            return ""
        state = "RUNNING" if self.active else "CANCELLED"
        node = "node001" if self.active else "node001"
        command = f"sbatch --parsable --comment={self.comment} {self.sbatch_path}"
        comment = self.comment
        return (
            f"{self.job_id}|{self.job_name}|{state}|{self.partition}|{node}|"
            f"{command}|{comment}\n"
        )

    def __call__(self, argv, **_kwargs):
        if argv[0] == "sacct":
            return _Process(stdout=self._row(source="sacct"))
        if argv[0] == "squeue":
            return _Process(stdout=self._row(source="squeue") if self.active else "")
        if argv[0] == "sbatch":
            self.sbatch_calls += 1
            intent_path = self.root / canary.INTENT_FILENAME
            assert intent_path.is_file(), "marker-first intent must precede sbatch"
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            ledger_path = (
                self.root / ".fleet-transactions-v1" / "ledgers" / "g000001.json"
            )
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            attempt = ledger["replicas"][intent["replica_id"]]["attempts"][0]
            assert attempt["state"] == "submitting"
            assert attempt["job_id"] is None
            assert argv == [
                "sbatch",
                "--parsable",
                f"--comment={intent['scheduler_comment']}",
                intent["sbatch_path"],
            ]
            script = Path(intent["sbatch_path"]).read_text(encoding="utf-8")
            assert "#SBATCH --no-requeue\n" in script
            assert "#SBATCH --partition=mit_normal\n" in script
            assert "#SBATCH --time=00:08:00\n" in script
            self.comment = intent["scheduler_comment"]
            self.sbatch_path = intent["sbatch_path"]
            self.job_name = intent["job_name"]
            self.partition = intent["partition"]
            self.accepted = True
            self.active = True
            if self.crash_after_acceptance:
                self.crash_after_acceptance = False
                raise KeyboardInterrupt("crash after Slurm accepted sbatch")
            return _Process(stdout=f"{self.job_id};cluster\n")
        if argv[:3] == ["scontrol", "write", "batch_script"]:
            self.scontrol_calls.append(list(argv))
            assert argv[3] == self.job_id
            body = Path(self.sbatch_path).read_bytes()
            if self.spooled_mismatch:
                body += b"# scheduler drift\n"
            Path(argv[4]).write_bytes(body)
            return _Process()
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            self.effective_state_calls.append(list(argv))
            assert argv == ["scontrol", "show", "job", "-o", self.job_id]
            return _Process(
                stdout=(
                    f"JobId={self.job_id} JobName={self.job_name} "
                    "JobState=RUNNING Requeue=0\n"
                )
            )
        if argv[0] == "scancel":
            self.scancel_calls.append(list(argv))
            assert argv == ["scancel", self.job_id]
            retirement = json.loads(
                (self.root / canary.RETIREMENT_INTENT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            assert retirement["job_id"] == self.job_id
            assert retirement["command"] == ["scancel", self.job_id]
            self.active = False
            if self.crash_after_scancel:
                self.crash_after_scancel = False
                raise KeyboardInterrupt("crash after Slurm accepted scancel")
            return _Process()
        raise AssertionError(f"unexpected command: {argv}")


def _run(root: Path, scheduler: _FakeSlurm, clock: _Clock):
    return canary.run_canary(
        root=root,
        apply=True,
        scheduler_user="tester",
        runner=scheduler,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        poll_seconds=1.0,
        visibility_timeout=20.0,
        terminal_timeout=20.0,
        token_factory=lambda: "1" * 32,
        code_identity=_code_identity(),
    )


def test_default_is_dry_run_and_does_not_create_root(tmp_path):
    root = tmp_path / "isolated-canary"
    result = canary.run_canary(root=root, code_identity=_code_identity())
    assert result["kind"] == "schema5_slurm_fleet_canary_dry_run"
    assert result["apply"] is False
    assert result["partition"] == "mit_normal"
    assert result["no_requeue"] is True
    assert not root.exists()


def test_apply_runs_one_real_transaction_contract_and_seals_marker_last(tmp_path):
    root = tmp_path / "isolated-canary"
    scheduler = _FakeSlurm(root)
    clock = _Clock()
    complete = _run(root, scheduler, clock)

    assert scheduler.sbatch_calls == 1
    assert scheduler.scancel_calls == [["scancel", "424242"]]
    assert len(scheduler.scontrol_calls) == 1
    assert scheduler.effective_state_calls == [
        ["scontrol", "show", "job", "-o", "424242"]
    ]
    assert complete["job_id"] == "424242"
    assert complete["terminal_state"] == "CANCELLED"
    assert complete["spooled_sha256"] == complete["sbatch_sha256"]
    assert complete["effective_requeue"] == 0
    assert canary.COMPLETE_FILENAME not in {
        item["path"] for item in complete["artifact_inventory"]
    }
    ledger = json.loads(
        (root / ".fleet-transactions-v1" / "ledgers" / "g000001.json").read_text(
            encoding="utf-8"
        )
    )
    attempt = next(iter(ledger["replicas"].values()))["attempts"][0]
    assert attempt["state"] == "terminal"
    assert attempt["job_id"] == "424242"
    assert attempt["submission_attempts"] == 1
    assert "#SBATCH --no-requeue" in Path(attempt["sbatch_path"]).read_text()
    for path in [root, *root.rglob("*")]:
        assert not stat.S_IMODE(path.stat().st_mode) & 0o222

    # Completed --apply and --verify calls are purely local and idempotent.
    bomb = lambda *_a, **_k: pytest.fail("sealed verification must not contact Slurm")
    assert canary.run_canary(
        root=root, apply=True, runner=bomb, code_identity=_code_identity()
    ) == complete
    assert canary.run_canary(
        root=root, verify=True, runner=bomb, code_identity=_code_identity()
    ) == complete


def test_crash_after_sbatch_acceptance_is_adopted_without_duplicate(tmp_path):
    root = tmp_path / "isolated-canary"
    scheduler = _FakeSlurm(root, crash_after_acceptance=True)
    clock = _Clock()
    with pytest.raises(KeyboardInterrupt, match="accepted sbatch"):
        _run(root, scheduler, clock)
    assert scheduler.sbatch_calls == 1
    ledger = json.loads(
        (root / ".fleet-transactions-v1" / "ledgers" / "g000001.json").read_text(
            encoding="utf-8"
        )
    )
    attempt = next(iter(ledger["replicas"].values()))["attempts"][0]
    assert attempt["state"] == "submitting"
    assert attempt["job_id"] is None

    complete = _run(root, scheduler, clock)
    assert complete["job_id"] == scheduler.job_id
    assert scheduler.sbatch_calls == 1
    assert scheduler.scancel_calls == [["scancel", scheduler.job_id]]


def test_spooled_script_drift_fails_before_retirement_or_cancel(tmp_path):
    root = tmp_path / "isolated-canary"
    scheduler = _FakeSlurm(root, spooled_mismatch=True)
    clock = _Clock()
    with pytest.raises(
        canary.SlurmFleetCanaryError,
        match="spooled batch script differs",
    ):
        _run(root, scheduler, clock)
    assert scheduler.sbatch_calls == 1
    assert scheduler.scancel_calls == []
    assert not (root / canary.RETIREMENT_INTENT_FILENAME).exists()
    assert not (root / canary.COMPLETE_FILENAME).exists()


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("JobId=42 Requeue=1\n", "Requeue is not disabled"),
        ("JobId=43 Requeue=0\n", "changed exact job identity"),
        ("JobId=42\n", "Requeue is not disabled"),
        (
            "JobId=42 Requeue=0\nJobId=42 Requeue=0\n",
            "exactly one record",
        ),
        ("JobId=42 JobId=42 Requeue=0\n", "duplicate effective Slurm field"),
        ("JobId=42 Requeue=0 Requeue=0\n", "duplicate effective Slurm field"),
    ],
)
def test_effective_requeue_parser_fails_closed(output, message):
    with pytest.raises(canary.SlurmFleetCanaryError, match=message):
        canary._parse_effective_job_state(output, expected_job_id="42")


def test_effective_requeue_receipt_tamper_and_query_failure_are_rejected(
    tmp_path,
):
    root = tmp_path / "isolated-canary"
    root.mkdir()
    receipt = root / canary.EFFECTIVE_JOB_STATE_FILENAME
    raw = "JobId=42 Requeue=1\n"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": canary.SCHEMA_VERSION,
                "kind": "schema5_slurm_fleet_canary_effective_job_state",
                "captured_at": 1.0,
                "command": ["scontrol", "show", "job", "-o", "42"],
                "job_id": "42",
                "effective_requeue": 0,
                "no_requeue": True,
                "raw_output": raw,
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        canary.SlurmFleetCanaryError, match="Requeue is not disabled"
    ):
        canary._ensure_effective_no_requeue(
            root=root,
            job_id="42",
            runner=lambda *_a, **_k: pytest.fail(
                "existing receipt must be validated locally"
            ),
            now=2.0,
        )

    receipt.unlink()
    with pytest.raises(
        canary.SlurmFleetCanaryError, match="query failed rc=1"
    ):
        canary._ensure_effective_no_requeue(
            root=root,
            job_id="42",
            runner=lambda *_a, **_k: _Process(
                returncode=1, stderr="missing job"
            ),
            now=2.0,
        )


def test_crash_after_exact_scancel_is_recovered_from_terminal_truth(tmp_path):
    root = tmp_path / "isolated-canary"
    scheduler = _FakeSlurm(root, crash_after_scancel=True)
    clock = _Clock()
    with pytest.raises(KeyboardInterrupt, match="accepted scancel"):
        _run(root, scheduler, clock)
    assert scheduler.scancel_calls == [["scancel", scheduler.job_id]]
    assert (root / canary.RETIREMENT_INTENT_FILENAME).is_file()
    assert not (root / canary.RETIREMENT_ACCEPTED_FILENAME).exists()

    complete = _run(root, scheduler, clock)
    assert complete["terminal_state"] == "CANCELLED"
    assert scheduler.scancel_calls == [["scancel", scheduler.job_id]]
    accepted = json.loads(
        (root / canary.RETIREMENT_ACCEPTED_FILENAME).read_text(encoding="utf-8")
    )
    assert accepted["evidence"] == "terminal_scheduler_truth"


def test_rejects_canonical_pool_and_parent_before_any_mutation(tmp_path):
    canonical = canary._canonical_pool_root()
    with pytest.raises(canary.SlurmFleetCanaryError, match="canonical server pool"):
        canary.run_canary(root=canonical, apply=True)
    with pytest.raises(canary.SlurmFleetCanaryError, match="contain canonical"):
        canary.run_canary(root=canonical.parent.parent, apply=True)


def test_sealed_inventory_tamper_is_rejected(tmp_path):
    root = tmp_path / "isolated-canary"
    scheduler = _FakeSlurm(root)
    clock = _Clock()
    _run(root, scheduler, clock)
    marker = root / canary.ACTIVE_BINDING_FILENAME
    marker.chmod(0o644)
    with pytest.raises(canary.SlurmFleetCanaryError, match="remains writable"):
        canary.verify_complete(root)


class _FakeTurnoverSlurm:
    def __init__(
        self,
        root: Path,
        *,
        crash_after_turnover_acceptance: bool = False,
    ):
        self.root = root
        self.crash_after_turnover_acceptance = (
            crash_after_turnover_acceptance
        )
        self.next_job_id = 510000
        self.jobs: dict[str, dict] = {}
        self.sbatch_calls: list[list[str]] = []
        self.scancel_calls: list[list[str]] = []
        self.release_calls: list[list[str]] = []
        self.active_counts: list[int] = []

    @staticmethod
    def _directive(script: str, name: str) -> str:
        prefix = f"#SBATCH --{name}="
        return next(
            line[len(prefix) :]
            for line in script.splitlines()
            if line.startswith(prefix)
        )

    def _sacct_row(self, job_id: str, job: dict) -> str:
        state = "RUNNING" if job["active"] else "CANCELLED"
        command = (
            f"sbatch --parsable --comment={job['comment']} {job['path']}"
        )
        return (
            f"{job_id}|{job['name']}|{state}|{job['partition']}|node001|"
            f"{command}|{job['comment']}|2026-07-23T12:00:00|"
            f"2026-07-23T12:08:00|8\n"
        )

    def _dependency_sacct_row(self, job_id: str, job: dict) -> str:
        command = " ".join(job["argv"])
        return (
            f"{job_id}|{job['comment']}|{job['name']}|{job['state']}|"
            f"{job['exit_code']}|{job['reason']}|{job['start']}|{job['end']}|"
            f"{job['elapsed']}|{command}\n"
        )

    def _dependency_squeue_row(self, job_id: str, job: dict) -> str:
        return (
            f"{job_id}|{job['comment']}|{job['name']}|{job['state']}|"
            f"{job['reason']}|{job['start']}|{job['elapsed']}\n"
        )

    def _squeue_row(self, job_id: str, job: dict) -> str:
        command = (
            f"sbatch --parsable --comment={job['comment']} {job['path']}"
        )
        return (
            f"{job_id}|{job['name']}|RUNNING|{job['partition']}|node001|"
            f"{command}|{job['comment']}|2026-07-23T12:00:00|"
            "2026-07-23T12:08:00|00:08:00|(null)\n"
        )

    def __call__(self, argv, **_kwargs):
        if argv[0] == "sacct":
            if any("ExitCode" in value for value in argv):
                return _Process(
                    stdout="".join(
                        self._dependency_sacct_row(job_id, job)
                        for job_id, job in self.jobs.items()
                        if job.get("dependency") is True
                    )
                )
            return _Process(
                stdout="".join(
                    self._sacct_row(job_id, job)
                    for job_id, job in self.jobs.items()
                )
            )
        if argv[0] == "squeue":
            if argv[-1] == "%i|%k|%j|%T|%r|%S|%M":
                return _Process(
                    stdout="".join(
                        self._dependency_squeue_row(job_id, job)
                        for job_id, job in self.jobs.items()
                        if job.get("dependency") is True and job["active"]
                    )
                )
            active = [
                (job_id, job)
                for job_id, job in self.jobs.items()
                if job["active"] and job.get("dependency") is not True
            ]
            self.active_counts.append(len(active))
            return _Process(
                stdout="".join(
                    self._squeue_row(job_id, job)
                    for job_id, job in active
                )
            )
        if argv[0] == "sbatch":
            self.sbatch_calls.append(list(argv))
            path = Path(argv[-1])
            script = path.read_text(encoding="utf-8")
            assert "#SBATCH --no-requeue\n" in script
            assert "#SBATCH --gres" not in script
            job_id = str(self.next_job_id)
            self.next_job_id += 1
            dependency = canary.DEPENDENCY_COMPONENT_DIRECTORY in path.parts
            comment = next(
                value.split("=", 1)[1]
                for value in argv
                if value.startswith("--comment=")
            )
            role = path.stem if dependency else None
            state = "PENDING" if dependency else "RUNNING"
            reason = (
                "JobHeldUser"
                if role == "root"
                else ("Dependency" if dependency else "")
            )
            self.jobs[job_id] = {
                "active": True,
                "comment": comment,
                "path": str(path.resolve()),
                "name": self._directive(script, "job-name"),
                "partition": self._directive(script, "partition"),
                "dependency": dependency,
                "role": role,
                "argv": list(argv),
                "state": state,
                "reason": reason,
                "exit_code": "0:0",
                "start": "Unknown" if dependency else "2026-07-23T12:00:00",
                "end": "Unknown" if dependency else "2026-07-23T12:08:00",
                "elapsed": "00:00:00" if dependency else "00:08:00",
            }
            if not dependency and self.crash_after_turnover_acceptance:
                self.crash_after_turnover_acceptance = False
                raise KeyboardInterrupt(
                    "crash after turnover allocation was accepted"
                )
            return _Process(stdout=f"{job_id};cluster\n")
        if argv[:3] == ["scontrol", "write", "batch_script"]:
            job = self.jobs[argv[3]]
            Path(argv[4]).write_bytes(Path(job["path"]).read_bytes())
            return _Process()
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            job_id = argv[4]
            job = self.jobs[job_id]
            return _Process(
                stdout=(
                    f"JobId={job_id} JobName={job['name']} "
                    f"JobState={job['state'] if job.get('dependency') else ('RUNNING' if job['active'] else 'CANCELLED')} "
                    f"Reason={job.get('reason', 'None')} "
                    "Requeue=0\n"
                )
            )
        if argv == ["scontrol", "show", "config"]:
            return _Process(
                stdout=(
                    "DependencyParameters = disable_remote_singleton_jobs,"
                    "kill_invalid_depend\n"
                )
            )
        if argv[:2] == ["scontrol", "release"]:
            self.release_calls.append(list(argv))
            root_id = argv[2]
            assert self.jobs[root_id]["role"] == "root"
            dependent = {
                job["role"]: (job_id, job)
                for job_id, job in self.jobs.items()
                if job.get("dependency") is True
            }
            root = dependent["root"][1]
            root.update(
                {
                    "active": False,
                    "state": "FAILED",
                    "reason": "NonZeroExitCode",
                    "exit_code": "42:0",
                    "start": "2026-07-23T12:00:00",
                    "end": "2026-07-23T12:00:10",
                    "elapsed": "00:00:10",
                }
            )
            child = dependent["child"][1]
            child.update(
                {
                    "active": False,
                    "state": "CANCELLED",
                    "reason": "DependencyNeverSatisfied",
                    "exit_code": "0:0",
                    "start": "Unknown",
                    "end": "2026-07-23T12:00:10",
                    "elapsed": "00:00:00",
                }
            )
            sentinel_id, sentinel = dependent["sentinel"]
            sentinel.update(
                {
                    "active": False,
                    "state": "COMPLETED",
                    "reason": "None",
                    "exit_code": "0:0",
                    "start": "2026-07-23T12:00:20",
                    "end": "2026-07-23T12:00:21",
                    "elapsed": "00:00:01",
                }
            )
            marker = (
                self.root
                / canary.DEPENDENCY_COMPONENT_DIRECTORY
                / canary.DEPENDENCY_ALERT_MARKER_FILENAME
            )
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "schema5_dependency_alert_executed",
                        "job_id": sentinel_id,
                        "started_timestamp": 1.0,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            return _Process()
        if argv[0] == "scancel":
            self.scancel_calls.append(list(argv))
            job_id = argv[1]
            assert self.jobs[job_id]["active"]
            self.jobs[job_id]["active"] = False
            return _Process()
        raise AssertionError(f"unexpected turnover command: {argv}")


def _successful_probe(endpoint, *, timeout):
    assert endpoint.startswith("http://node001:")
    assert timeout > 0
    return {
        "endpoint": endpoint,
        "health": {
            "status": 200,
            "size": 3,
            "sha256": hashlib.sha256(b"ok\n").hexdigest(),
        },
        "models": {
            "status": 200,
            "size": 41,
            "sha256": hashlib.sha256(
                b'{"data": [{"id": "turnover-canary"}]}'
            ).hexdigest(),
        },
    }


def test_turnover_plan_is_real_executable_and_nonmutating(tmp_path):
    root = tmp_path / "turnover"
    plan = canary.run_turnover_canary(
        root=root,
        code_identity=_code_identity(),
    )
    assert plan["cycles"] == 2
    assert len(plan["allocations"]) == 3
    assert len(plan["transitions"]) == 2
    assert plan["maximum_physical_allocations"] == 2
    assert plan["maximum_overlap_gpus"] == 0
    assert all(
        "#SBATCH --no-requeue\n" in item["script"]
        and "ThreadingHTTPServer" in item["script"]
        and "/v1/models" in item["script"]
        for item in plan["allocations"]
    )
    assert not root.exists()


def test_turnover_ports_are_full_token_derived_unique_and_deterministic():
    token = "0123456789ab" + "0" * 20
    same_prefix_token = "0123456789ab" + "f" * 20
    first = canary.render_turnover_canary_plan(
        cycles=4,
        run_token=token,
    )
    repeated = canary.render_turnover_canary_plan(
        cycles=4,
        run_token=token,
    )
    other = canary.render_turnover_canary_plan(
        cycles=4,
        run_token=same_prefix_token,
    )
    ports = [item["port"] for item in first["allocations"]]
    other_ports = [item["port"] for item in other["allocations"]]

    assert first == repeated
    assert len(ports) == len(set(ports)) == 5
    assert all(
        canary.TURNOVER_PORT_MIN <= port <= canary.TURNOVER_PORT_MAX
        for port in ports
    )
    assert ports != [19_200 + index for index in range(5)]
    # The old identity suffix is deliberately shared here; port selection must use
    # the complete token rather than those first twelve hexadecimal digits.
    assert first["job_name"] == other["job_name"]
    assert ports != other_ports
    assert first["port_derivation"] == {
        "protocol": canary.TURNOVER_PORT_DERIVATION_PROTOCOL,
        "source": "complete_marker_first_128_bit_run_token+allocation_index",
        "minimum_port": canary.TURNOVER_PORT_MIN,
        "maximum_port": canary.TURNOVER_PORT_MAX,
        "collision_resolution": "ascending_wrap_within_plan",
    }
    for allocation, port in zip(first["allocations"], ports, strict=True):
        assert f"PORT={port}\n" in allocation["script"]


def test_turnover_port_derivation_resolves_internal_hash_collisions(
    monkeypatch,
):
    monkeypatch.setattr(
        canary,
        "_turnover_initial_port",
        lambda **_kwargs: canary.TURNOVER_PORT_MAX,
    )
    assert canary._derive_turnover_ports(
        run_token="a" * 32,
        allocation_count=5,
    ) == [
        canary.TURNOVER_PORT_MAX,
        canary.TURNOVER_PORT_MIN,
        canary.TURNOVER_PORT_MIN + 1,
        canary.TURNOVER_PORT_MIN + 2,
        canary.TURNOVER_PORT_MIN + 3,
    ]


def test_turnover_crash_reuses_immutable_token_ports_without_duplicate_sbatch(
    tmp_path,
):
    root = tmp_path / "turnover"
    scheduler = _FakeTurnoverSlurm(
        root,
        crash_after_turnover_acceptance=True,
    )
    clock = _Clock()
    kwargs = {
        "root": root,
        "apply": True,
        "scheduler_user": "tester",
        "runner": scheduler,
        "probe": _successful_probe,
        "now_fn": clock.now,
        "sleep_fn": clock.sleep,
        "poll_seconds": 1.0,
        "visibility_timeout": 20.0,
        "terminal_timeout": 20.0,
        "drain_seconds": 1.0,
        "code_identity": _code_identity(),
    }
    with pytest.raises(
        KeyboardInterrupt,
        match="turnover allocation was accepted",
    ):
        canary.run_turnover_canary(
            **kwargs,
            token_factory=lambda: "e" * 32,
        )
    intent_path = root / canary.TURNOVER_INTENT_FILENAME
    initial_intent = json.loads(intent_path.read_text(encoding="utf-8"))
    initial_ports = [
        item["port"] for item in initial_intent["plan"]["allocations"]
    ]
    assert len(scheduler.sbatch_calls) == 1

    complete = canary.run_turnover_canary(
        **kwargs,
        token_factory=lambda: pytest.fail(
            "crash recovery must reuse the marker-first run token"
        ),
    )
    recovered_intent = json.loads(intent_path.read_text(encoding="utf-8"))
    assert recovered_intent == initial_intent
    assert [
        item["port"] for item in recovered_intent["plan"]["allocations"]
    ] == initial_ports
    assert len(scheduler.sbatch_calls) == 3
    assert complete["allocations_submitted"] == 3


def test_real_dependency_cascade_is_held_receipted_bounded_and_sealed(
    tmp_path,
):
    root = tmp_path / canary.DEPENDENCY_COMPONENT_DIRECTORY
    scheduler = _FakeTurnoverSlurm(tmp_path)
    clock = _Clock()
    complete = canary.run_dependency_cascade_canary(
        root=root,
        apply=True,
        scheduler_user="tester",
        runner=scheduler,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        poll_seconds=1.0,
        visibility_timeout=20.0,
        terminal_timeout=20.0,
        token_factory=lambda: "d" * 32,
        code_identity=_code_identity(),
    )

    assert len(scheduler.sbatch_calls) == 3
    assert "--hold" in scheduler.sbatch_calls[0]
    assert scheduler.release_calls == [["scontrol", "release", "510000"]]
    assert complete["root_initial_hold"] is True
    assert complete["root_state"] == "FAILED"
    assert complete["child_state"] == "CANCELLED"
    assert complete["sentinel_state"] == "COMPLETED"
    assert complete["child_never_started"] is True
    assert complete["kill_invalid_depend"] is True
    assert complete["alert_latency_seconds"] == 10.0
    assert complete["alert_latency_bound_seconds"] == 180.0
    bomb = lambda *_a, **_k: pytest.fail(
        "sealed dependency verification must not contact Slurm"
    )
    assert canary.run_dependency_cascade_canary(
        root=root,
        verify=True,
        runner=bomb,
        code_identity=_code_identity(),
    ) == complete


def test_dependency_cascade_fails_held_before_release_on_policy_drift(
    tmp_path,
):
    root = tmp_path / canary.DEPENDENCY_COMPONENT_DIRECTORY
    scheduler = _FakeTurnoverSlurm(tmp_path)

    def drifted(argv, **kwargs):
        if argv == ["scontrol", "show", "config"]:
            return _Process(stdout="DependencyParameters = (null)\n")
        return scheduler(argv, **kwargs)

    with pytest.raises(
        canary.SlurmFleetCanaryError,
        match="kill_invalid_depend",
    ):
        canary.run_dependency_cascade_canary(
            root=root,
            apply=True,
            scheduler_user="tester",
            runner=drifted,
            token_factory=lambda: "e" * 32,
            code_identity=_code_identity(),
        )
    assert len(scheduler.sbatch_calls) == 3
    assert scheduler.release_calls == []
    assert (root / canary.DEPENDENCY_RECEIPT_FILENAME).is_file()
    assert not (root / canary.DEPENDENCY_COMPLETE_FILENAME).exists()


def test_dependency_cascade_rejects_alert_latency_over_bound(tmp_path):
    root = tmp_path / canary.DEPENDENCY_COMPONENT_DIRECTORY
    scheduler = _FakeTurnoverSlurm(tmp_path)
    clock = _Clock()
    with pytest.raises(
        canary.SlurmFleetCanaryError,
        match="exceeds bound",
    ):
        canary.run_dependency_cascade_canary(
            root=root,
            apply=True,
            scheduler_user="tester",
            runner=scheduler,
            now_fn=clock.now,
            sleep_fn=clock.sleep,
            poll_seconds=1.0,
            visibility_timeout=20.0,
            terminal_timeout=20.0,
            alert_latency_bound_seconds=5.0,
            token_factory=lambda: "f" * 32,
            code_identity=_code_identity(),
        )
    assert scheduler.release_calls == [["scontrol", "release", "510000"]]
    assert not (root / canary.DEPENDENCY_COMPLETE_FILENAME).exists()


def test_two_real_turnover_transactions_seal_continuity_and_overlap_evidence(
    tmp_path,
):
    root = tmp_path / "turnover"
    scheduler = _FakeTurnoverSlurm(root)
    clock = _Clock()
    complete = canary.run_turnover_canary(
        root=root,
        apply=True,
        scheduler_user="tester",
        runner=scheduler,
        probe=_successful_probe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        poll_seconds=1.0,
        visibility_timeout=20.0,
        terminal_timeout=20.0,
        drain_seconds=1.0,
        token_factory=lambda: "a" * 32,
        code_identity=_code_identity(),
    )

    assert len(scheduler.sbatch_calls) == 3
    assert scheduler.scancel_calls == [
        ["scancel", "510000"],
        ["scancel", "510001"],
        ["scancel", "510002"],
    ]
    assert max(scheduler.active_counts) == 2
    assert complete["cycles_completed"] == 2
    assert complete["allocations_submitted"] == 3
    assert complete["maximum_physical_allocations_observed"] == 2
    assert complete["maximum_extra_gpus_observed"] == 0
    assert complete["production_overlap_gpu_ceiling"] == 4
    assert complete["all_effective_requeue"] == 0
    assert complete["continuous_routed_endpoint_evidence"] is True
    assert len(complete["effective_state_receipts"]) == 3
    assert all(
        item["effective_requeue"] == 0
        for item in complete["effective_state_receipts"]
    )
    ledger = json.loads(
        Path(complete["ledger_path"]).read_text(encoding="utf-8")
    )
    assert {
        attempt["allocated_gpus"]
        for replica in ledger["replicas"].values()
        for attempt in replica["attempts"]
    } == {0}
    pointer = json.loads(
        (root / canary.TURNOVER_POINTER_FILENAME).read_text(encoding="utf-8")
    )
    assert pointer["job_id"] == "510002"
    assert pointer["allocation_index"] == 2
    for cycle in (1, 2):
        transition = root / "transitions" / f"c{cycle:02d}"
        assert (transition / "PROBE_1.json").is_file()
        assert (transition / "PROBE_2.json").is_file()
        assert (transition / "POST_PROMOTION_PROBE.json").is_file()
        assert (transition / "PRE_RETIRE_ROUTED_PROBE.json").is_file()
    for path in [root, *root.rglob("*")]:
        assert not stat.S_IMODE(path.stat().st_mode) & 0o222
    bomb = lambda *_a, **_k: pytest.fail(
        "sealed turnover verification must not contact external systems"
    )
    assert canary.run_turnover_canary(
        root=root,
        verify=True,
        runner=bomb,
        probe=bomb,
        code_identity=_code_identity(),
    ) == complete


def test_composite_canary_binds_transaction_and_two_turnovers_marker_last(
    tmp_path,
):
    root = tmp_path / "composite"
    scheduler = _FakeTurnoverSlurm(root)
    clock = _Clock()
    complete = canary.run_composite_canary(
        root=root,
        apply=True,
        scheduler_user="tester",
        runner=scheduler,
        probe=_successful_probe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        poll_seconds=1.0,
        visibility_timeout=20.0,
        terminal_timeout=20.0,
        turnover_drain_seconds=1.0,
        token_factory=lambda: "b" * 32,
        code_identity=_code_identity(),
    )

    assert complete["schema_version"] == canary.SCHEMA_VERSION
    assert (
        complete["kind"]
        == "schema5_slurm_fleet_composite_canary_complete"
    )
    assert complete["turnover_cycles_completed"] == 2
    assert complete["turnover_maximum_physical_allocations_observed"] == 2
    assert complete["turnover_maximum_extra_gpus_observed"] == 0
    assert complete["turnover_all_effective_requeue"] == 0
    assert complete["turnover_continuous_routed_endpoint_evidence"] is True
    assert complete["dependency_kill_invalid_depend"] is True
    assert complete["dependency_root_initial_hold"] is True
    assert complete["dependency_child_never_started"] is True
    assert complete["dependency_alert_latency_seconds"] == 10.0
    assert (
        complete["dependency_alert_latency_seconds"]
        <= complete["dependency_alert_latency_bound_seconds"]
    )
    assert len(scheduler.sbatch_calls) == 7
    assert len(scheduler.scancel_calls) == 4
    assert (
        complete["transaction_marker_sha256"]
        == canary._sha256_file(
            root
            / canary.TRANSACTION_COMPONENT_DIRECTORY
            / canary.COMPLETE_FILENAME
        )
    )
    assert (
        complete["turnover_marker_sha256"]
        == canary._sha256_file(
            root
            / canary.TURNOVER_COMPONENT_DIRECTORY
            / canary.TURNOVER_COMPLETE_FILENAME
        )
    )
    bomb = lambda *_a, **_k: pytest.fail(
        "sealed composite verification must be entirely local"
    )
    assert canary.run_composite_canary(
        root=root,
        verify=True,
        runner=bomb,
        probe=bomb,
        code_identity=_code_identity(),
    ) == complete


def test_turnover_retries_endpoint_startup_within_visibility_deadline(tmp_path):
    root = tmp_path / "turnover"
    scheduler = _FakeTurnoverSlurm(root)
    clock = _Clock()
    failures = 2
    calls = 0

    def delayed_probe(endpoint, *, timeout):
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise canary.SlurmFleetCanaryError("connection refused")
        return _successful_probe(endpoint, timeout=timeout)

    complete = canary.run_turnover_canary(
        root=root,
        apply=True,
        scheduler_user="tester",
        runner=scheduler,
        probe=delayed_probe,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        poll_seconds=1.0,
        visibility_timeout=20.0,
        terminal_timeout=20.0,
        drain_seconds=1.0,
        token_factory=lambda: "c" * 32,
        code_identity=_code_identity(),
    )
    assert calls > failures
    assert complete["cycles_completed"] == 2
