"""Crash-boundary tests for schema-5 serving-fleet transactions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agents_scaling.serving import fleet_transactions as tx


POOL_ID = "schema5-v1"
FLEET_SHA256 = "a" * 64
REPLICA_ID = "schema5-v1--8b--standard--r00"
PROFILE = "8B"
GENERATION = 7
CLUSTER_FIXTURES = Path(__file__).parent / "fixtures" / "slurm_schema5_cluster"


def _process(*, returncode=0, stdout="", stderr=""):
    return type(
        "Process",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


def _open_state(tmp_path, *, now=1.0):
    root = tmp_path / "server_pools" / POOL_ID
    root.mkdir(parents=True, exist_ok=True)
    lock = tx.transaction_lock(root)
    directory = lock.__enter__()
    ledger = tx.load_or_create_ledger(
        directory,
        pool_root=root,
        pool_id=POOL_ID,
        fleet_sha256=FLEET_SHA256,
        rollout_generation=GENERATION,
        replica_ids=[REPLICA_ID],
        now=now,
    )
    return root, lock, directory, ledger


def _prepare(directory, ledger, *, now=1.0):
    return tx.prepare_attempt(
        directory,
        ledger,
        replica_id=REPLICA_ID,
        profile=PROFILE,
        pool_id=POOL_ID,
        fleet_sha256=FLEET_SHA256,
        rollout_generation=GENERATION,
        sbatch_text="#!/bin/bash\n#SBATCH --no-requeue\ntrue\n",
        now=now,
        token_factory=lambda: "1" * 32,
    )


def _committed_handoff(directory, ledger, *, allocated_gpus=1):
    primary = _prepare(directory, ledger, now=1.0)
    primary.update(
        {
            "state": "committed",
            "job_id": "700",
            "submitted_at": 1.0,
            "committed_at": 2.0,
            "last_seen_at": 50.0,
            "allocated_gpus": allocated_gpus,
            "scheduler_start_at": 0.0,
            "scheduler_end_at": 86_400.0,
            "scheduler_time_limit_seconds": 86_400,
        }
    )
    tx.save_ledger(directory, ledger, now=2.0)
    standby = tx.prepare_attempt(
        directory,
        ledger,
        replica_id=REPLICA_ID,
        profile=PROFILE,
        pool_id=POOL_ID,
        fleet_sha256=FLEET_SHA256,
        rollout_generation=GENERATION,
        sbatch_text="#!/bin/bash\n#SBATCH --no-requeue\n--standby\n",
        now=3.0,
        token_factory=lambda: "2" * 32,
        launch_kind="handoff",
        predecessor_job_id="700",
        predecessor_end_at=86_400.0,
        predecessor_attempt=primary,
        allocated_gpus=allocated_gpus,
    )
    standby.update(
        {
            "state": "committed",
            "job_id": "701",
            "submitted_at": 3.0,
            "committed_at": 4.0,
            "last_seen_at": 50.0,
            "last_ready_probe_at": 50.0,
            "ready_probe_count": 1,
            "scheduler_start_at": 3.0,
            "scheduler_end_at": 86_403.0,
            "scheduler_time_limit_seconds": 86_400,
        }
    )
    tx.save_ledger(directory, ledger, now=50.0)
    return primary, standby


def test_pre_sbatch_intent_and_script_are_durable_and_immutable(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        path = Path(attempt["sbatch_path"])
        assert attempt["state"] == "prepared"
        assert path.name == f"{REPLICA_ID}.{'1' * 32}.sbatch"
        assert path.read_text() == "#!/bin/bash\n#SBATCH --no-requeue\ntrue\n"
        assert not path.stat().st_mode & 0o222
        persisted = json.loads(tx.ledger_path(directory, GENERATION).read_text())
        assert persisted["replicas"][REPLICA_ID]["attempts"][0] == attempt
    finally:
        lock.__exit__(None, None, None)


def test_endpoint_admission_exports_only_exact_committed_ledger_attempt(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        attempt.update(
            {
                "state": "committed",
                "job_id": "4242",
                "submitted_at": 1.5,
                "committed_at": 2.0,
            }
        )
        tx.save_ledger(directory, ledger, now=2.0)
        binding = tx.committed_endpoint_admission(
            directory,
            ledger,
            replica_id=REPLICA_ID,
            attempt=attempt,
            slurm_job_id="4242",
        )
        assert binding.intent_token == "1" * 32
        assert binding.rollout_generation == GENERATION
        assert binding.sbatch_sha256 == hashlib.sha256(
            binding.sbatch_path.read_bytes()
        ).hexdigest()
        with pytest.raises(
            tx.FleetTransactionError, match="one exact ledger attempt"
        ):
            tx.committed_endpoint_admission(
                directory,
                ledger,
                replica_id=REPLICA_ID,
                attempt=dict(attempt),
                slurm_job_id="4242",
            )
        with pytest.raises(tx.FleetTransactionError, match="not committed"):
            tx.committed_endpoint_admission(
                directory,
                ledger,
                replica_id=REPLICA_ID,
                attempt=attempt,
                slurm_job_id="9999",
            )
    finally:
        lock.__exit__(None, None, None)


def test_pytest_guard_forbids_live_sbatch_before_state_transition(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        with pytest.raises(tx.FleetTransactionError, match="injected non-Slurm"):
            tx.submit_attempt(
                directory,
                ledger,
                replica_id=REPLICA_ID,
                attempt=attempt,
                now=2.0,
            )
        assert attempt["state"] == "prepared"
        assert attempt["submission_attempts"] == 0
    finally:
        lock.__exit__(None, None, None)


def test_crash_after_sbatch_acceptance_remains_adoptable(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        calls = []

        def accepted_then_killed(argv, **kwargs):
            calls.append((argv, kwargs))
            raise KeyboardInterrupt("controller died after Slurm acceptance")

        with pytest.raises(KeyboardInterrupt, match="after Slurm acceptance"):
            tx.submit_attempt(
                directory,
                ledger,
                replica_id=REPLICA_ID,
                attempt=attempt,
                now=2.0,
                runner=accepted_then_killed,
            )
        persisted = json.loads(tx.ledger_path(directory, GENERATION).read_text())
        durable = persisted["replicas"][REPLICA_ID]["attempts"][0]
        assert durable["state"] == "submitting"
        assert durable["job_id"] is None
        assert durable["submission_attempts"] == 1
        assert calls[0][0][0:2] == ["sbatch", "--parsable"]
        assert calls[0][0][2] == f"--comment={attempt['scheduler_comment']}"
        assert calls[0][0][-1] == attempt["sbatch_path"]
    finally:
        lock.__exit__(None, None, None)


def test_successful_submission_records_exact_job_once(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        job_id = tx.submit_attempt(
            directory,
            ledger,
            replica_id=REPLICA_ID,
            attempt=attempt,
            now=2.0,
            runner=lambda *_a, **_k: _process(stdout="4242;cluster\n"),
        )
        assert job_id == "4242"
        assert attempt["state"] == "submitted"
        assert attempt["job_id"] == "4242"
        with pytest.raises(tx.FleetTransactionError, match="cannot submit"):
            tx.submit_attempt(
                directory,
                ledger,
                replica_id=REPLICA_ID,
                attempt=attempt,
                now=3.0,
                runner=lambda *_a, **_k: pytest.fail("must not call sbatch twice"),
            )
    finally:
        lock.__exit__(None, None, None)


def test_joined_scheduler_uses_sacct_to_recover_terminal_history():
    token = "1" * 32
    comment = tx.intent_comment(
        pool_id=POOL_ID,
        profile=PROFILE,
        replica_id=REPLICA_ID,
        rollout_generation=GENERATION,
        intent_token=token,
        fleet_sha256=FLEET_SHA256,
    )
    path = f"/pool/.fleet-transactions-v1/sbatch/g000007/{REPLICA_ID}.{token}.sbatch"
    accounting = (
        f"4242|asys-s5-serve-8b-s-r00|COMPLETED|ou_bcs_low|(null)|"
        f"sbatch --comment={comment} {path}|\n"
    )
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv[0])
        return _process(stdout=accounting if argv[0] == "sacct" else "")

    snapshot = tx.query_scheduler(runner=runner, user="scientist", now=1000.0)
    assert calls == ["sacct", "squeue"]
    assert snapshot.sacct_ok and snapshot.squeue_ok
    assert [(row.job_id, row.source) for row in snapshot.rows] == [("4242", "sacct")]


def test_scheduler_argv_matches_cluster_supported_fields_exactly():
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return _process()

    tx.query_scheduler(runner=runner, user="scientist", now=1_700_000_000.0)
    assert calls[0][0:4] == ["sacct", "-X", "-u", "scientist"]
    assert calls[0].count("-u") == 1
    assert calls[0][-1] == (
        "--format=JobIDRaw,JobName,State,Partition,NodeList,SubmitLine,"
        "Comment,Start,End,TimelimitRaw"
    )
    assert "Dependency" not in calls[0][-1]
    assert calls[1] == [
        "squeue",
        "-u",
        "scientist",
        "-h",
        "-r",
        "-o",
        "%i|%j|%T|%P|%N|%o|%k|%S|%e|%l|%E",
    ]


def test_scheduler_cluster_time_units_timezone_and_cross_source_agree():
    token = "1" * 32
    comment = tx.intent_comment(
        pool_id=POOL_ID,
        profile=PROFILE,
        replica_id=REPLICA_ID,
        rollout_generation=GENERATION,
        intent_token=token,
        fleet_sha256=FLEET_SHA256,
    )
    path = f"/pool/.fleet-transactions-v1/sbatch/g000007/{REPLICA_ID}.{token}.sbatch"
    submit = f"sbatch --comment={comment} {path}"
    sacct = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|ou_bcs_low|node1|{submit}||"
        "2026-01-01T12:00:00-05:00|Unknown|1440\n"
    )
    squeue = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|ou_bcs_low|node1|{submit}|"
        f"{comment}|2026-01-01T17:00:00Z|2026-01-02T17:00:00Z|"
        "1-00:00:00|(null)\n"
    )

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else squeue)

    [row] = tx.query_scheduler(
        runner=runner, user="scientist", now=1_767_290_400.0
    ).rows
    assert row.start_timestamp == 1_767_286_800.0
    assert row.end_timestamp == row.start_timestamp + 86_400
    assert row.time_limit_seconds == 86_400
    assert row.dependency == ""

    parsed = tx._parse_rows(  # noqa: SLF001 - source-unit contract regression
        (
            f"4243|asys-s5-serve-8b-s-r00|COMPLETED|ou_bcs_low|node1|"
            f"{submit}||2026-01-01T12:00:00-05:00|"
            "2026-01-08T12:00:00-05:00|10080\n"
        ),
        source="sacct",
    )
    assert parsed[0].time_limit_seconds == 7 * 86_400


@pytest.mark.parametrize(
    ("sacct_start", "sacct_limit", "squeue_start", "squeue_limit", "message"),
    [
        (
            "2026-01-01T12:00:00Z",
            "1440",
            "2026-01-01T12:00:03Z",
            "1-00:00:00",
            "start_timestamp conflict",
        ),
        (
            "2026-01-01T12:00:00Z",
            "60",
            "2026-01-01T12:00:00Z",
            "2:00:00",
            "time limit conflict",
        ),
    ],
)
def test_scheduler_rejects_cross_source_timing_drift(
    sacct_start, sacct_limit, squeue_start, squeue_limit, message
):
    token = "1" * 32
    comment = tx.intent_comment(
        pool_id=POOL_ID,
        profile=PROFILE,
        replica_id=REPLICA_ID,
        rollout_generation=GENERATION,
        intent_token=token,
        fleet_sha256=FLEET_SHA256,
    )
    path = f"/pool/{REPLICA_ID}.{token}.sbatch"
    submit = f"sbatch --comment={comment} {path}"
    sacct = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|p|node1|{submit}||"
        f"{sacct_start}|Unknown|{sacct_limit}\n"
    )
    squeue = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|p|node1|{submit}|{comment}|"
        f"{squeue_start}|2026-01-02T12:00:00Z|{squeue_limit}|(null)\n"
    )

    with pytest.raises(tx.FleetTransactionError, match=message):
        tx.query_scheduler(
            runner=lambda argv, **_kwargs: _process(
                stdout=sacct if argv[0] == "sacct" else squeue
            ),
            user="scientist",
            now=1_767_290_400.0,
        )


def test_scheduler_rejects_active_dependency_from_squeue():
    token = "1" * 32
    comment = tx.intent_comment(
        pool_id=POOL_ID,
        profile=PROFILE,
        replica_id=REPLICA_ID,
        rollout_generation=GENERATION,
        intent_token=token,
        fleet_sha256=FLEET_SHA256,
    )
    submit = f"sbatch --comment={comment} /pool/job.sbatch"
    squeue = (
        f"4242|asys-s5-serve-8b-s-r00|PENDING|p|(null)|{submit}|{comment}|"
        "2026-01-01T12:00:00Z|2026-01-02T12:00:00Z|1-00:00:00|afterok:7\n"
    )
    with pytest.raises(tx.FleetTransactionError, match="unexpected dependency"):
        tx.query_scheduler(
            runner=lambda argv, **_kwargs: _process(
                stdout="" if argv[0] == "sacct" else squeue
            ),
            user="scientist",
            now=1_767_290_400.0,
        )


def test_joined_scheduler_recovers_blank_sacct_comment_and_agrees_with_squeue():
    token = "1" * 32
    comment = tx.intent_comment(
        pool_id=POOL_ID,
        profile=PROFILE,
        replica_id=REPLICA_ID,
        rollout_generation=GENERATION,
        intent_token=token,
        fleet_sha256=FLEET_SHA256,
    )
    path = f"/pool/.fleet-transactions-v1/sbatch/g000007/{REPLICA_ID}.{token}.sbatch"
    sacct = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|ou_bcs_low|(null)|"
        f"sbatch --comment={comment} {path}|\n"
    )
    squeue = (
        f"4242|asys-s5-serve-8b-s-r00|RUNNING|ou_bcs_low|node1|"
        f"sbatch --comment={comment} {path}|{comment}\n"
    )

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else squeue)

    snapshot = tx.query_scheduler(runner=runner, user="scientist", now=1000.0)
    assert snapshot.rows[0].comment == comment
    assert snapshot.rows[0].source == "squeue"


def test_recorded_cluster_accounting_contract_is_supported():
    assert "(null)" in (
        CLUSTER_FIXTURES / "accounting_store_flags.txt"
    ).read_text(encoding="utf-8")
    outputs = {
        "sacct": (CLUSTER_FIXTURES / "fleet_sacct_blank_comment.txt").read_text(
            encoding="utf-8"
        ),
        "squeue": (CLUSTER_FIXTURES / "fleet_squeue_comment.txt").read_text(
            encoding="utf-8"
        ),
    }
    snapshot = tx.query_scheduler(
        runner=lambda argv, **_kwargs: _process(stdout=outputs[argv[0]]),
        user="scientist",
        now=1000.0,
    )
    assert len(snapshot.rows) == 1
    assert tx.parse_intent_comment(snapshot.rows[0].comment) == {
        "pool": POOL_ID,
        "profile": PROFILE,
        "replica": REPLICA_ID,
        "generation": str(GENERATION),
        "intent": "1" * 32,
        "fleet": FLEET_SHA256,
    }


def test_sacct_rejects_stored_comment_that_disagrees_with_submit_line():
    sacct = (
        "42|asys-s5-serve-8b-s-r00|RUNNING|p|node1|"
        "sbatch --comment=asys-s5-fleet:from-submit /x.sbatch|"
        "asys-s5-fleet:from-column\n"
    )

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else "")

    with pytest.raises(tx.FleetTransactionError, match="comment/SubmitLine conflict"):
        tx.query_scheduler(runner=runner, user="scientist", now=1000.0)


def test_scheduler_visibility_fails_closed_when_either_source_fails():
    def runner(argv, **_kwargs):
        if argv[0] == "sacct":
            return _process(returncode=1, stderr="accounting unavailable")
        return _process()

    with pytest.raises(tx.FleetTransactionError, match=r"complete squeue\+sacct"):
        tx.query_scheduler(runner=runner, user="scientist", now=1000.0)


def test_scheduler_detects_cross_source_identity_conflict():
    sacct = "42|asys-s5-serve-8b-s-r00|RUNNING|p|(null)|/x.sbatch|comment-a\n"
    squeue = "42|asys-s5-serve-8b-s-r00|RUNNING|p|node1|/x.sbatch|comment-b\n"

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else squeue)

    with pytest.raises(tx.FleetTransactionError, match="identity conflict"):
        tx.query_scheduler(runner=runner, user="scientist", now=1000.0)


def test_scheduler_join_normalizes_slurm_null_comment_spellings():
    sacct = "42|asys-s5-serve-8b-s-r00|RUNNING|p|node1|/x.sbatch|\n"
    squeue = "42|asys-s5-serve-8b-s-r00|RUNNING|p|node1|/x.sbatch|(null)\n"

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else squeue)

    snapshot = tx.query_scheduler(runner=runner, user="scientist", now=1000.0)
    assert snapshot.rows == (
        tx.SchedulerRow(
            "42",
            "asys-s5-serve-8b-s-r00",
            "RUNNING",
            "p",
            "node1",
            "/x.sbatch",
            "",
            "squeue",
        ),
    )


def test_scheduler_join_ignores_unrelated_array_and_step_rows():
    sacct = "41.batch|ordinary-job|COMPLETED||node1|bash|\n"
    squeue = "42_7|ordinary-array|RUNNING|p|node1|worker.sh|(null)\n"

    def runner(argv, **_kwargs):
        return _process(stdout=sacct if argv[0] == "sacct" else squeue)

    snapshot = tx.query_scheduler(runner=runner, user="scientist", now=1000.0)
    assert snapshot.rows == ()


def test_read_only_reconciliation_binds_transaction_comment_ledger_and_script(
    tmp_path,
):
    root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        row = tx.SchedulerRow(
            "4242",
            "asys-s5-serve-8b-s-r00",
            "RUNNING",
            "ou_bcs_low",
            "node001",
            f"sbatch --comment={attempt['scheduler_comment']} "
            f"{attempt['sbatch_path']}",
            attempt["scheduler_comment"],
            "squeue",
        )
        reconciled = tx.reconcile_scheduler_rows(
            [row],
            [ledger],
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            replica_profiles={REPLICA_ID: PROFILE},
            replica_job_names={REPLICA_ID: "asys-s5-serve-8b-s-r00"},
        )
        [allocation] = reconciled.active_allocations
        assert allocation.replica_id == REPLICA_ID
        assert allocation.attempt["intent_token"] == "1" * 32
        assert allocation.ledger_generation == GENERATION
    finally:
        lock.__exit__(None, None, None)


def test_read_only_reconciliation_rejects_duplicate_and_unknown_active_jobs(
    tmp_path,
):
    root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        row = tx.SchedulerRow(
            "4242",
            "asys-s5-serve-8b-s-r00",
            "PENDING",
            "ou_bcs_low",
            "(null)",
            attempt["sbatch_path"],
            attempt["scheduler_comment"],
        )
        kwargs = {
            "pool_id": POOL_ID,
            "fleet_sha256": FLEET_SHA256,
            "replica_profiles": {REPLICA_ID: PROFILE},
            "replica_job_names": {REPLICA_ID: "asys-s5-serve-8b-s-r00"},
        }
        with pytest.raises(tx.FleetTransactionError, match="duplicate fleet jobs"):
            tx.reconcile_scheduler_rows(
                [row, replace(row, job_id="4243")],
                [ledger],
                **kwargs,
            )
        unknown = tx.SchedulerRow(
            "4244",
            row.job_name,
            "RUNNING",
            row.partition,
            "node001",
            "/foreign.sbatch",
            tx.intent_comment(
                pool_id=POOL_ID,
                profile=PROFILE,
                replica_id=REPLICA_ID,
                rollout_generation=GENERATION,
                intent_token="2" * 32,
                fleet_sha256=FLEET_SHA256,
            ),
        )
        with pytest.raises(tx.FleetTransactionError, match="unknown fleet intent"):
            tx.reconcile_scheduler_rows([unknown], [ledger], **kwargs)
    finally:
        lock.__exit__(None, None, None)


def test_generation_ledger_read_is_non_mutating_and_fails_on_writer_intent(
    tmp_path,
):
    root, lock, directory, _ledger = _open_state(tmp_path)
    lock.__exit__(None, None, None)
    before = {
        path: path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    with tx.read_transaction_lock(root) as read_directory:
        ledgers = tx.read_generation_ledgers(
            read_directory,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            current_generation=GENERATION,
            replica_ids=[REPLICA_ID],
        )
    assert len(ledgers) == 1
    assert {
        path: path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    } == before

    marker = directory / tx.SAVE_INTENT_FILENAME
    marker.write_text("writer-owned\n", encoding="utf-8")
    with tx.read_transaction_lock(root) as read_directory:
        with pytest.raises(tx.FleetTransactionError, match="publication is in progress"):
            tx.read_generation_ledgers(
                read_directory,
                pool_root=root,
                pool_id=POOL_ID,
                fleet_sha256=FLEET_SHA256,
                current_generation=GENERATION,
                replica_ids=[REPLICA_ID],
            )


def test_ledger_rejects_external_sbatch_path_and_comment_drift(tmp_path):
    _root, lock, directory, ledger = _open_state(tmp_path)
    try:
        attempt = _prepare(directory, ledger)
        attempt["sbatch_path"] = "/tmp/foreign.sbatch"
        tx.save_ledger(directory, ledger, now=2.0)
    finally:
        lock.__exit__(None, None, None)
    with tx.transaction_lock(_root) as reopened:
        with pytest.raises(tx.FleetTransactionError, match="escapes fleet transaction"):
            tx.load_or_create_ledger(
                reopened,
                pool_root=_root,
                pool_id=POOL_ID,
                fleet_sha256=FLEET_SHA256,
                rollout_generation=GENERATION,
                replica_ids=[REPLICA_ID],
                now=3.0,
            )


def test_generation_index_advances_and_rejects_stale_supervisor(tmp_path):
    root, lock, directory, _ledger = _open_state(tmp_path, now=1.0)
    try:
        generation_eight = tx.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            rollout_generation=GENERATION + 1,
            replica_ids=[REPLICA_ID],
            now=2.0,
        )
        current = json.loads((directory / tx.CURRENT_FILENAME).read_text())
        current_path = tx.ledger_path(directory, GENERATION + 1)
        assert current["current_generation"] == GENERATION + 1
        assert current["ledger_path"] == str(current_path.resolve())
        assert current["ledger_sha256"] == hashlib.sha256(
            current_path.read_bytes()
        ).hexdigest()
        assert generation_eight["rollout_generation"] == GENERATION + 1

        loaded = tx.load_generation_ledgers(
            directory,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            current_generation=GENERATION + 1,
            replica_ids=[REPLICA_ID],
            now=3.0,
        )
        assert [item["rollout_generation"] for item in loaded] == [
            GENERATION,
            GENERATION + 1,
        ]
        with pytest.raises(tx.FleetTransactionError, match="stale generation"):
            tx.load_or_create_ledger(
                directory,
                pool_root=root,
                pool_id=POOL_ID,
                fleet_sha256=FLEET_SHA256,
                rollout_generation=GENERATION,
                replica_ids=[REPLICA_ID],
                now=4.0,
            )
    finally:
        lock.__exit__(None, None, None)


def test_interrupted_ledger_then_current_publication_recovers_from_wal(
    tmp_path, monkeypatch
):
    root, lock, directory, ledger = _open_state(tmp_path, now=1.0)
    try:
        before_index = json.loads((directory / tx.CURRENT_FILENAME).read_text())
        real_write = tx.io.atomic_write_text

        def die_before_current(path, payload):
            if Path(path).name == tx.CURRENT_FILENAME:
                raise KeyboardInterrupt("killed between ledger and CURRENT")
            return real_write(path, payload)

        monkeypatch.setattr(tx.io, "atomic_write_text", die_before_current)
        with pytest.raises(KeyboardInterrupt, match="between ledger and CURRENT"):
            tx.save_ledger(directory, ledger, now=2.0)
        assert (directory / tx.SAVE_INTENT_FILENAME).is_file()
        assert json.loads((directory / tx.CURRENT_FILENAME).read_text()) == before_index
    finally:
        lock.__exit__(None, None, None)

    monkeypatch.setattr(tx.io, "atomic_write_text", real_write)
    with tx.transaction_lock(root) as reopened:
        recovered = tx.load_or_create_ledger(
            reopened,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            rollout_generation=GENERATION,
            replica_ids=[REPLICA_ID],
            now=3.0,
        )
        current = json.loads((reopened / tx.CURRENT_FILENAME).read_text())
        path = tx.ledger_path(reopened, GENERATION)
        assert recovered["updated_at"] == 2.0
        assert current["ledger_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert not (reopened / tx.SAVE_INTENT_FILENAME).exists()


def test_interrupted_pre_ledger_save_rolls_back_wal_without_mutation(
    tmp_path, monkeypatch
):
    root, lock, directory, ledger = _open_state(tmp_path, now=1.0)
    ledger_file = tx.ledger_path(directory, GENERATION)
    before_ledger = ledger_file.read_bytes()
    before_current = (directory / tx.CURRENT_FILENAME).read_bytes()
    real_write = tx.io.atomic_write_text
    try:
        def die_before_ledger(path, payload):
            if Path(path) == ledger_file:
                raise KeyboardInterrupt("killed before ledger")
            return real_write(path, payload)

        monkeypatch.setattr(tx.io, "atomic_write_text", die_before_ledger)
        with pytest.raises(KeyboardInterrupt, match="before ledger"):
            tx.save_ledger(directory, ledger, now=2.0)
        assert (directory / tx.SAVE_INTENT_FILENAME).is_file()
    finally:
        lock.__exit__(None, None, None)

    monkeypatch.setattr(tx.io, "atomic_write_text", real_write)
    with tx.transaction_lock(root) as reopened:
        recovered = tx.load_or_create_ledger(
            reopened,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            rollout_generation=GENERATION,
            replica_ids=[REPLICA_ID],
            now=3.0,
        )
        assert recovered["updated_at"] == 1.0
        assert ledger_file.read_bytes() == before_ledger
        assert (reopened / tx.CURRENT_FILENAME).read_bytes() == before_current
        assert not (reopened / tx.SAVE_INTENT_FILENAME).exists()


def test_health_monitor_never_recovers_writer_save_intent(tmp_path):
    root, lock, directory, _ledger = _open_state(tmp_path)
    try:
        marker = directory / tx.SAVE_INTENT_FILENAME
        marker.write_text("writer-owned\n", encoding="utf-8")
        before = marker.read_bytes()
        with pytest.raises(tx.FleetTransactionError, match="locked recovery"):
            tx.read_health_summary(root)
        assert marker.read_bytes() == before
    finally:
        lock.__exit__(None, None, None)


def test_health_summary_accepts_valid_handoff_and_flags_critical_lead(tmp_path):
    root, lock, directory, ledger = _open_state(tmp_path)
    try:
        _primary, _standby = _committed_handoff(directory, ledger)
    finally:
        lock.__exit__(None, None, None)
    healthy = tx.read_health_summary(root, now=50.0)
    assert healthy["handoff_overlap_gpus"] == 1
    assert healthy["handoff_violations"] == []
    assert healthy["active_handoffs"][0]["job_id"] == "701"

    critical = tx.read_health_summary(root, now=85_000.0)
    assert {
        row["kind"] for row in critical["handoff_violations"]
    } == {"handoff_critical_lead", "handoff_scheduler_stale"}


def test_health_summary_flags_overlap_budget_and_missing_promoted_pointer(
    tmp_path,
):
    root, lock, directory, ledger = _open_state(tmp_path)
    try:
        primary, standby = _committed_handoff(
            directory, ledger, allocated_gpus=5
        )
    finally:
        lock.__exit__(None, None, None)
    over_budget = tx.read_health_summary(root, now=50.0)
    assert over_budget["handoff_overlap_gpus"] == 5
    assert "handoff_overlap_budget_exceeded" in {
        row["kind"] for row in over_budget["handoff_violations"]
    }

    with tx.transaction_lock(root) as directory:
        material = tx.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=POOL_ID,
            fleet_sha256=FLEET_SHA256,
            rollout_generation=GENERATION,
            replica_ids=[REPLICA_ID],
            now=60.0,
        )
        primary, standby = material["replicas"][REPLICA_ID]["attempts"]
        primary.update(
            {
                "state": "terminal",
                "terminal_at": 60.0,
                "lifecycle": "retiring",
            }
        )
        standby.update(
            {
                "lifecycle": "promoted",
                "ready_probe_count": 2,
                "promoted_at": 60.0,
            }
        )
        tx.save_ledger(directory, material, now=60.0)
    missing_pointer = tx.read_health_summary(root, now=61.0)
    assert "invalid_promoted_pointer" in {
        row["kind"] for row in missing_pointer["handoff_violations"]
    }
