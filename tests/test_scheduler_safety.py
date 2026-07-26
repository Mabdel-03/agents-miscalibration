from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agents_scaling.serving import scheduler_safety


CONFIG = """\
KillWait                = 30 sec
PreemptMode             = REQUEUE
PreemptType             = preempt/partition_prio
"""


def _partition(name: str, *, mode: str, grace: int, max_time: str = "1-00:00:00") -> str:
    return (
        f"PartitionName={name} GraceTime={grace} MaxTime={max_time} "
        f"PreemptMode={mode} State=UP TotalNodes=10\n"
    )


def _runner(outputs):
    def run(argv, *, timeout):
        del timeout
        key = tuple(argv)
        assert key in outputs
        return subprocess.CompletedProcess(argv, 0, outputs[key], "")

    return run


def _evidence(*, low_mode: str = "REQUEUE", low_grace: int = 0):
    outputs = {
        ("scontrol", "show", "config"): CONFIG,
        (
            "scontrol",
            "show",
            "partition",
            "ou_bcs_low",
            "-o",
        ): _partition("ou_bcs_low", mode=low_mode, grace=low_grace),
        (
            "scontrol",
            "show",
            "partition",
            "ou_bcs_normal",
            "-o",
        ): _partition("ou_bcs_normal", mode="OFF", grace=0),
    }
    return scheduler_safety.capture_scheduler_safety_evidence(
        ["ou_bcs_low", "ou_bcs_normal"],
        runner=_runner(outputs),
        captured_timestamp=100.0,
    )


def _identity(value):
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_nonpreemptible_partitions_need_no_transport_exception():
    evidence = _evidence(low_mode="OFF")
    policy = scheduler_safety.validate_scheduler_safety_evidence(
        evidence,
        expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
        required_time_limits_seconds={
            "ou_bcs_low": 86_400,
            "ou_bcs_normal": 86_400,
        },
    )
    assert policy["preemptible_partitions"] == []
    assert policy["transport_uncertainty_binding"] is None
    assert {
        value["authorization"] for value in policy["partitions"].values()
    } == {"nonpreemptible"}


def test_preemptible_partition_fails_closed_without_verified_transport_contract():
    evidence = _evidence()
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="immutable no-redraw transport-uncertainty contract",
    ):
        scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
            required_time_limits_seconds={
                "ou_bcs_low": 86_400,
                "ou_bcs_normal": 86_400,
            },
        )


def test_preemptible_partition_accepts_only_exact_validated_binding():
    evidence = _evidence()
    binding = scheduler_safety.expected_transport_uncertainty_binding()
    assert binding["transport_censor_protocol_version"] == 1
    assert binding["transport_censor_protocol_hash"] == (
        "e1a46d605eed904e902f9339be714c1019aa361a11da2cdab98b8edcf0ec3104"
    )
    assert binding["stochastic_chat_sdk_max_retries"] == 0
    assert binding["stochastic_chat_create_calls_per_attempt"] == 1
    assert binding["checkpoint_schema_version"] == 3
    assert binding["replacement_sampling"] == "forbidden"
    assert binding["transport_censor_protocol_spec"][
        "single_attempt_transport"
    ]["openai_sdk_max_retries"] == 0
    assert len(scheduler_safety.transport_uncertainty_binding_sha256(binding)) == 64
    assert scheduler_safety.slurm_time_limit_seconds("1-00:00:00") == 86_400
    assert scheduler_safety.fleet_partition_time_requirements(
        [
            SimpleNamespace(partition="ou_bcs_low", time_limit="12:00:00"),
            SimpleNamespace(partition="ou_bcs_low", time_limit="1-00:00:00"),
            SimpleNamespace(partition="ou_bcs_normal", time_limit="1-00:00:00"),
        ]
    ) == {"ou_bcs_low": 86_400, "ou_bcs_normal": 86_400}

    policy = scheduler_safety.validate_scheduler_safety_evidence(
        evidence,
        expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
        required_time_limits_seconds={
            "ou_bcs_low": 86_400,
            "ou_bcs_normal": 86_400,
        },
        transport_uncertainty_binding=binding,
    )
    assert policy["preemptible_partitions"] == ["ou_bcs_low"]
    assert policy["partitions"]["ou_bcs_low"]["authorization"] == (
        "trusted_transport_censor"
    )
    assert policy["partitions"]["ou_bcs_low"]["grace_time_seconds"] == 0
    assert policy["transport_uncertainty_binding"] == binding
    fresh_evidence = copy.deepcopy(evidence)
    fresh_evidence["captured_timestamp"] = 101.0
    fresh_evidence["evidence_id"] = _identity(
        {
            key: value
            for key, value in fresh_evidence.items()
            if key != "evidence_id"
        }
    )
    fresh_policy = scheduler_safety.validate_scheduler_safety_evidence(
        fresh_evidence,
        expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
        required_time_limits_seconds={
            "ou_bcs_low": 86_400,
            "ou_bcs_normal": 86_400,
        },
        transport_uncertainty_binding=binding,
    )
    assert fresh_policy["scheduler_evidence_id"] != policy["scheduler_evidence_id"]
    assert fresh_policy["policy_id"] != policy["policy_id"]
    assert fresh_policy["policy_contract_id"] == policy["policy_contract_id"]
    drift_policy = scheduler_safety.validate_scheduler_safety_evidence(
        _evidence(low_grace=1),
        expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
        required_time_limits_seconds={
            "ou_bcs_low": 86_400,
            "ou_bcs_normal": 86_400,
        },
        transport_uncertainty_binding=binding,
    )
    assert drift_policy["policy_contract_id"] != policy["policy_contract_id"]

    altered = dict(binding)
    altered["transport_censor_protocol_hash"] = "1" * 64
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="exact schema-5 transport/checkpoint/artifact protocols",
    ):
        scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
            required_time_limits_seconds={
                "ou_bcs_low": 86_400,
                "ou_bcs_normal": 86_400,
            },
            transport_uncertainty_binding=altered,
        )


def test_derived_partition_tamper_is_rejected_after_evidence_id_rehash():
    evidence = _evidence()
    evidence["partitions"][0]["preempt_modes"] = ["OFF"]
    evidence["evidence_id"] = _identity(
        {key: value for key, value in evidence.items() if key != "evidence_id"}
    )
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="differs from its raw query",
    ):
        scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
            required_time_limits_seconds={
                "ou_bcs_low": 86_400,
                "ou_bcs_normal": 86_400,
            },
            transport_uncertainty_binding=(
                scheduler_safety.expected_transport_uncertainty_binding()
            ),
        )


def test_partition_max_time_must_cover_frozen_server_job():
    evidence = _evidence()
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="MaxTime is shorter",
    ):
        scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=["ou_bcs_low", "ou_bcs_normal"],
            required_time_limits_seconds={
                "ou_bcs_low": 86_401,
                "ou_bcs_normal": 86_400,
            },
            transport_uncertainty_binding=(
                scheduler_safety.expected_transport_uncertainty_binding()
            ),
        )


def _client_contract_outputs(*, cpu=96, memory="386G", submit=448):
    return {
        ("scontrol", "show", "config"): CONFIG,
        (
            "scontrol",
            "show",
            "partition",
            "mit_normal",
            "-o",
        ): _partition(
            "mit_normal",
            mode="OFF",
            grace=0,
            max_time="12:00:00",
        ),
        (
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "mit_normal",
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ): f"mit_normal|cpu={cpu},mem={memory}|{submit}\n",
    }


def test_client_capacity_contract_binds_live_partition_and_qos_limits():
    evidence = scheduler_safety.capture_client_capacity_contract(
        runner=_runner(_client_contract_outputs()),
        captured_timestamp=100.0,
    )
    validated = scheduler_safety.validate_client_capacity_contract(evidence)
    assert validated["partition"] == "mit_normal"
    assert validated["cpu_limit"] == 96
    assert validated["memory_limit_mib"] == 386 * 1024
    assert validated["max_submit_jobs"] == 448

    drifted = scheduler_safety.capture_scheduler_safety_evidence(
        ["mit_normal"],
        runner=_runner(
            {
                ("scontrol", "show", "config"): CONFIG,
                (
                    "scontrol",
                    "show",
                    "partition",
                    "mit_normal",
                    "-o",
                ): _partition(
                    "mit_normal",
                    mode="OFF",
                    grace=0,
                    max_time="12:00:00",
                ),
            }
        ),
        captured_timestamp=100.0,
    )
    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        "mit_normal",
        "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
    ]
    qos = scheduler_safety._qos_capacity_fact(  # noqa: SLF001
        raw_output="mit_normal|cpu=95,mem=386G|448\n",
        argv=qos_argv,
    )
    invalid = {
        "schema_version": 1,
        "protocol": scheduler_safety.CLIENT_CAPACITY_PROTOCOL,
        "captured_timestamp": 100.0,
        "scheduler": drifted,
        "qos": qos,
    }
    invalid["evidence_id"] = _identity(invalid)
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="QOS limits drifted",
    ):
        scheduler_safety.validate_client_capacity_contract(invalid)


def test_client_capacity_contract_supports_only_an_explicit_dynamic_authority():
    partition = "schema5_clients_g2"
    outputs = {
        ("scontrol", "show", "config"): CONFIG,
        (
            "scontrol",
            "show",
            "partition",
            partition,
            "-o",
        ): _partition(
            partition,
            mode="OFF",
            grace=0,
            max_time="12:00:00",
        ),
        (
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            partition,
            "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
        ): f"{partition}|cpu=384,mem=1536G|448\n",
    }
    evidence = scheduler_safety.capture_client_capacity_contract(
        partition=partition,
        expected_cpu_limit=384,
        expected_memory_limit_mib=384 * 4096,
        expected_max_submit_jobs=448,
        runner=_runner(outputs),
        captured_timestamp=100.0,
    )
    summary = scheduler_safety.validate_client_capacity_contract(
        evidence,
        expected_partition=partition,
        expected_cpu_limit=384,
        expected_memory_limit_mib=384 * 4096,
        expected_max_submit_jobs=448,
    )
    assert summary == {
        **{
            key: summary[key]
            for key in (
                "evidence_id",
                "scheduler_policy_id",
                "scheduler_policy_contract_id",
            )
        },
            "partition": partition,
            "qos": partition,
            "cpu_limit": 384,
        "memory_limit_mib": 384 * 4096,
        "max_submit_jobs": 448,
    }
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="QOS limits drifted",
    ):
        scheduler_safety.validate_client_capacity_contract(
            evidence,
            expected_partition=partition,
            expected_cpu_limit=383,
            expected_memory_limit_mib=384 * 4096,
            expected_max_submit_jobs=448,
        )


def _number(number, *, isset=True):
    return {"set": isset, "infinite": False, "number": number}


def _usage_job(
    job_id,
    *,
    name,
    cpus,
    memory_mib,
    partition="mit_normal",
    task_id=None,
    comment=None,
):
    return {
        "job_id": job_id,
        "job_state": ["RUNNING"],
        "partition": partition,
        "name": name,
        "comment": (
            (
                f"asys-schema5-intent:{job_id}"
                if name.startswith("asys-dispatch-")
                else ""
            )
            if comment is None
            else comment
        ),
        "qos": partition,
        "cpus": _number(cpus),
        "node_count": _number(1),
        "memory_per_cpu": _number(0, isset=False),
        "memory_per_node": _number(memory_mib),
        "array_task_id": _number(
            0 if task_id is None else task_id,
            isset=task_id is not None,
        ),
    }


def test_user_partition_usage_subtracts_unrelated_tres_and_invisible_intents():
    payload = {
        "jobs": [
            _usage_job(
                10,
                name="unrelated-analysis",
                cpus=2,
                memory_mib=8 * 1024,
            ),
            _usage_job(
                11,
                name="asys-dispatch-test",
                cpus=1,
                memory_mib=4 * 1024,
                task_id=0,
            ),
            _usage_job(
                12,
                name="other-partition",
                cpus=64,
                memory_mib=128 * 1024,
                partition="pi_tpoggio",
            ),
        ],
        "errors": [],
        "warnings": [],
    }
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        "tester",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    usage = scheduler_safety.capture_user_partition_usage(
        user="tester",
        runner=_runner({tuple(argv): json.dumps(payload)}),
        captured_timestamp=100.0,
    )
    assert usage["job_count"] == 2
    assert usage["used_cpus"] == 3
    assert usage["used_memory_mib"] == 12 * 1024
    assert scheduler_safety.client_task_headroom(
        usage,
        cell_cpus=1,
        cell_memory_mib=4 * 1024,
        invisible_reserved_tasks=2,
    ) == 91

    usage["used_cpus"] = 0
    usage["evidence_id"] = _identity(
        {key: value for key, value in usage.items() if key != "evidence_id"}
    )
    with pytest.raises(
        scheduler_safety.SchedulerSafetyError,
        match="differ from the raw scheduler response",
    ):
        scheduler_safety.client_task_headroom(
            usage,
            cell_cpus=1,
            cell_memory_mib=4 * 1024,
        )


def test_user_partition_headroom_uses_the_authorized_dynamic_limits():
    partition = "schema5_clients_g2"
    payload = {
        "jobs": [
            _usage_job(
                20,
                name="asys-dispatch-g2",
                cpus=2,
                memory_mib=8 * 1024,
                partition=partition,
            )
        ],
        "errors": [],
        "warnings": [],
    }
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        "tester",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    usage = scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition=partition,
        runner=_runner({tuple(argv): json.dumps(payload)}),
        captured_timestamp=100.0,
    )
    assert usage["partition"] == partition
    assert scheduler_safety.client_task_headroom(
        usage,
        cell_cpus=1,
        cell_memory_mib=4 * 1024,
        cpu_limit=384,
        memory_limit_mib=384 * 4096,
    ) == 382

    crowded_payload = {
        "jobs": [
            _usage_job(
                job_id,
                name=(
                    "asys-dispatch-g2"
                    if job_id == 20
                    else f"unrelated-{job_id}"
                ),
                cpus=1,
                memory_mib=4096,
                partition=partition,
            )
            for job_id in (20, 21, 22)
        ],
        "errors": [],
        "warnings": [],
    }
    crowded = scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition=partition,
        runner=_runner({tuple(argv): json.dumps(crowded_payload)}),
        captured_timestamp=101.0,
    )
    # MaxSubmitJobs=10 with a four-job reserve and three scheduler-visible jobs
    # leaves three slots. Two visibility-grace intents are absent from squeue and are
    # subtracted exactly once, leaving one safe admission despite ample CPU/memory.
    assert scheduler_safety.client_task_headroom(
        crowded,
        cell_cpus=1,
        cell_memory_mib=4096,
        invisible_reserved_tasks=2,
        cpu_limit=384,
        memory_limit_mib=384 * 4096,
        max_submit_jobs=10,
        reserve_jobs=4,
    ) == 1


def test_inclusive_reserve_with_actual_22_replica_fleet_admits_initial_batch():
    fleet = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "schema5_fleet.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fleet["logical_replica_count"] == 22
    assert fleet["allocated_gpu_count"] == 24
    partition = "ou_bcs_normal"
    server_rows = []
    trusted_servers = {}
    for index in range(fleet["logical_replica_count"]):
        job_id = 20_000 + index
        comment = (
            "asys-s5-fleet:pool=schema5-v1;profile=profile;"
            f"replica=replica-{index};generation=1;intent={'a' * 32};"
            f"fleet={'b' * 64}"
        )
        name = f"asys-s5-serve-{index}"
        server_rows.append(
            _usage_job(
                job_id,
                name=name,
                cpus=8,
                memory_mib=120 * 1024,
                partition=partition,
                comment=comment,
            )
        )
        trusted_servers[str(job_id)] = {
            "job_name": name,
            "comment": comment,
        }
    payload = {"jobs": server_rows, "errors": [], "warnings": []}
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        "tester",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    usage = scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition=partition,
        runner=_runner({tuple(argv): json.dumps(payload)}),
        captured_timestamp=100.0,
    )

    # Two controller chains count globally but run outside the client placement.
    assert scheduler_safety.client_task_headroom(
        usage,
        cell_cpus=1,
        cell_memory_mib=4096,
        cpu_limit=384,
        memory_limit_mib=384 * 4096,
        max_submit_jobs=500,
        reserve_jobs=64,
        cell_ceiling=384,
        absolute_job_ceiling=448,
        live_user_job_elements=24,
        trusted_nonclient_jobs=trusted_servers,
    ) == 384


def test_servers_are_not_double_charged_but_unknown_jobs_reduce_headroom():
    partition = "ou_bcs_normal"
    server_rows = []
    trusted_servers = {}
    for index in range(22):
        job_id = 30_000 + index
        name = f"asys-s5-serve-{index}"
        comment = f"sealed-fleet-comment:{index}"
        server_rows.append(
            _usage_job(
                job_id,
                name=name,
                cpus=8,
                memory_mib=120 * 1024,
                partition=partition,
                comment=comment,
            )
        )
        trusted_servers[str(job_id)] = {
            "job_name": name,
            "comment": comment,
        }
    client_rows = [
        _usage_job(
            40_000 + index,
            name="asys-dispatch-production",
            cpus=1,
            memory_mib=4096,
            partition=partition,
            task_id=index,
            comment="asys-schema5-intent:production",
        )
        for index in range(360)
    ]
    payload = {
        "jobs": server_rows + client_rows,
        "errors": [],
        "warnings": [],
    }
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        "tester",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    usage = scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition=partition,
        runner=_runner({tuple(argv): json.dumps(payload)}),
        captured_timestamp=100.0,
    )
    kwargs = {
        "cell_cpus": 1,
        "cell_memory_mib": 4096,
        "cpu_limit": 384,
        "memory_limit_mib": 384 * 4096,
        "max_submit_jobs": 500,
        "reserve_jobs": 64,
        "cell_ceiling": 384,
        "absolute_job_ceiling": 448,
        # 360 clients + 22 servers + two controllers/unrelated jobs.
        "live_user_job_elements": 384,
        "trusted_nonclient_jobs": trusted_servers,
    }
    assert scheduler_safety.client_task_headroom(usage, **kwargs) == 24

    forged = copy.deepcopy(trusted_servers)
    forged["30000"]["comment"] = "wrong-comment"
    assert (
        scheduler_safety.client_task_headroom(
            usage,
            **{**kwargs, "trusted_nonclient_jobs": forged},
        )
        == 0
    )


def test_absolute_448_gate_applies_even_when_association_limit_is_500():
    payload = {"jobs": [], "errors": [], "warnings": []}
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        "tester",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    usage = scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition="ou_bcs_normal",
        runner=_runner({tuple(argv): json.dumps(payload)}),
        captured_timestamp=100.0,
    )
    common = {
        "cell_cpus": 1,
        "cell_memory_mib": 4096,
        "cpu_limit": 384,
        "memory_limit_mib": 384 * 4096,
        "max_submit_jobs": 500,
        "reserve_jobs": 64,
        "cell_ceiling": 384,
        "absolute_job_ceiling": 448,
    }
    assert scheduler_safety.client_task_headroom(
        usage, live_user_job_elements=447, **common
    ) == 1
    assert scheduler_safety.client_task_headroom(
        usage, live_user_job_elements=448, **common
    ) == 0
