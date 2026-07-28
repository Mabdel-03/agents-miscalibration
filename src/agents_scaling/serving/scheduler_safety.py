"""Fail-closed Slurm partition-preemption evidence for schema-5 serving.

``#SBATCH --no-requeue`` controls what Slurm does *after* terminating a job.  It
does not make an allocation non-preemptible.  In particular, a partition with
``PreemptMode=REQUEUE`` and zero grace can kill a vLLM process after a response has
left the server but before the client has durably journaled it.  Scheduler placement
therefore participates in the scientific no-redraw contract.

This module deliberately separates two concerns:

* capture and reparse exact ``scontrol`` evidence without trusting derived JSON; and
* authorize a preemptible partition only through the exact schema-5
  transport-censor/checkpoint/artifact protocol imported from the production code.

Callers that do not provide that exact binding accept only ``PreemptMode=OFF``.  A
callback, boolean, or environment-variable escape hatch is intentionally not
supported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import re
import shlex
import subprocess
import time
from typing import Any


SCHEDULER_SAFETY_PROTOCOL = (
    "schema5-v1.2-r6-serving-partition-scheduler-safety"
)
SCHEDULER_SAFETY_SCHEMA_VERSION = 1
CLIENT_CAPACITY_PROTOCOL = "schema5-v1.2-r6-mit-normal-client-capacity"
CLIENT_CAPACITY_SCHEMA_VERSION = 1
CLIENT_PARTITION = "mit_normal"
CLIENT_CPU_LIMIT = 96
CLIENT_MEMORY_LIMIT_MIB = 386 * 1024
CLIENT_MAX_SUBMIT_JOBS = 448
CLIENT_JOB_TIME_LIMIT_SECONDS = 12 * 60 * 60
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ACTIVE_PREEMPT_MODES = frozenset({"CANCEL", "GANG", "REQUEUE", "SUSPEND"})
_TRANSPORT_BINDING_FIELDS = frozenset(
    {
        "transport_censor_protocol_version",
        "transport_censor_protocol_hash",
        "transport_censor_protocol_spec",
        "stochastic_chat_sdk_max_retries",
        "stochastic_chat_create_calls_per_attempt",
        "transport_censor_error_message_max_utf8_bytes",
        "transport_censor_error_redaction_policy_version",
        "transport_censor_error_redaction_marker",
        "transport_censor_sensitive_message_pattern",
        "transport_censor_classifications",
        "transport_censor_api_status_error_types",
        "checkpoint_schema_version",
        "artifact_schema_version",
        "self_consistency_protocol_version",
        "self_consistency_protocol_hash",
        "replacement_sampling",
    }
)


class SchedulerSafetyError(RuntimeError):
    """The live scheduler policy cannot safely support the frozen fleet."""


def expected_transport_uncertainty_binding() -> dict[str, Any]:
    """Return the only no-redraw protocol identity accepted by scheduler policy."""

    # Local imports avoid making the lightweight scheduler evidence parser initialize
    # the experiment/checkpoint stack when every partition is non-preemptible.
    from agents_scaling.experiment.qid_checkpoint import CHECKPOINT_SCHEMA_VERSION
    from agents_scaling.experiment.result_schema import (
        ARTIFACT_SCHEMA_VERSION,
        SELF_CONSISTENCY_PROTOCOL_HASH,
        SELF_CONSISTENCY_PROTOCOL_VERSION,
    )
    from agents_scaling.experiment.transport_censor import (
        STOCHASTIC_CHAT_CREATE_CALLS_PER_ATTEMPT,
        STOCHASTIC_CHAT_SDK_MAX_RETRIES,
        TRANSPORT_CENSOR_API_STATUS_ERROR_TYPES,
        TRANSPORT_CENSOR_CLASSIFICATIONS,
        TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES,
        TRANSPORT_CENSOR_ERROR_REDACTION_MARKER,
        TRANSPORT_CENSOR_ERROR_REDACTION_POLICY_VERSION,
        TRANSPORT_CENSOR_PROTOCOL_HASH,
        TRANSPORT_CENSOR_PROTOCOL_VERSION,
        TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN,
        transport_censor_protocol_spec,
    )

    return {
        "transport_censor_protocol_version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
        "transport_censor_protocol_hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
        "transport_censor_protocol_spec": transport_censor_protocol_spec(),
        "stochastic_chat_sdk_max_retries": STOCHASTIC_CHAT_SDK_MAX_RETRIES,
        "stochastic_chat_create_calls_per_attempt": (
            STOCHASTIC_CHAT_CREATE_CALLS_PER_ATTEMPT
        ),
        "transport_censor_error_message_max_utf8_bytes": (
            TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        ),
        "transport_censor_error_redaction_policy_version": (
            TRANSPORT_CENSOR_ERROR_REDACTION_POLICY_VERSION
        ),
        "transport_censor_error_redaction_marker": (
            TRANSPORT_CENSOR_ERROR_REDACTION_MARKER
        ),
        "transport_censor_sensitive_message_pattern": (
            TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN
        ),
        "transport_censor_classifications": sorted(
            TRANSPORT_CENSOR_CLASSIFICATIONS
        ),
        "transport_censor_api_status_error_types": sorted(
            TRANSPORT_CENSOR_API_STATUS_ERROR_TYPES
        ),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "self_consistency_protocol_version": SELF_CONSISTENCY_PROTOCOL_VERSION,
        "self_consistency_protocol_hash": SELF_CONSISTENCY_PROTOCOL_HASH,
        "replacement_sampling": "forbidden",
    }


def validate_transport_uncertainty_binding(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject anything except the exact imported production no-redraw identity."""

    expected = expected_transport_uncertainty_binding()
    if (
        not isinstance(binding, Mapping)
        or set(binding) != _TRANSPORT_BINDING_FIELDS
        or dict(binding) != expected
    ):
        raise SchedulerSafetyError(
            "transport-uncertainty binding differs from the exact schema-5 "
            "transport/checkpoint/artifact protocols"
        )
    return expected


def transport_uncertainty_binding_sha256(
    binding: Mapping[str, Any] | None = None,
) -> str:
    """Return the canonical identity of the exact validated no-redraw binding."""

    validated = validate_transport_uncertainty_binding(
        expected_transport_uncertainty_binding()
        if binding is None
        else binding
    )
    return _identity_sha256(validated)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_key_value_record(
    raw_output: str,
    *,
    description: str,
) -> tuple[dict[str, str], str]:
    if not isinstance(raw_output, str) or "\x00" in raw_output:
        raise SchedulerSafetyError(f"{description} is not safe text")
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SchedulerSafetyError(
            f"{description} must contain exactly one non-empty record"
        )
    try:
        tokens = shlex.split(lines[0], posix=True)
    except (TypeError, ValueError) as exc:
        raise SchedulerSafetyError(f"cannot parse {description}: {exc}") from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise SchedulerSafetyError(
                f"{description} contains duplicate/invalid field {key!r}"
            )
        fields[key] = value.strip('"')
    return fields, lines[0] + "\n"


def _parse_configuration(
    raw_output: str,
) -> dict[str, str]:
    if not isinstance(raw_output, str) or "\x00" in raw_output:
        raise SchedulerSafetyError("scontrol configuration output is not safe text")
    wanted = {"PreemptType", "PreemptMode", "KillWait"}
    fields: dict[str, str] = {}
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in wanted:
            continue
        if key in fields or not value:
            raise SchedulerSafetyError(
                f"scontrol configuration has duplicate/invalid {key}"
            )
        fields[key] = value
    if set(fields) != wanted:
        raise SchedulerSafetyError(
            "scontrol configuration omitted PreemptType/PreemptMode/KillWait"
        )
    return fields


def _parse_duration_seconds(value: str, *, field: str) -> int:
    normalized = str(value).strip()
    if not normalized or normalized.upper() in {"UNLIMITED", "INFINITE"}:
        if normalized.upper() in {"UNLIMITED", "INFINITE"}:
            return 2**63 - 1
        raise SchedulerSafetyError(f"{field} is absent")
    days = 0
    clock = normalized
    if "-" in normalized:
        day_text, clock = normalized.split("-", 1)
        if not day_text.isdigit():
            raise SchedulerSafetyError(f"{field} is invalid: {value!r}")
        days = int(day_text)
    parts = clock.split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise SchedulerSafetyError(f"{field} is invalid: {value!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    else:
        hours, minutes, seconds = 0, 0, numbers[0]
    if minutes >= 60 or seconds >= 60:
        raise SchedulerSafetyError(f"{field} is invalid: {value!r}")
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def slurm_time_limit_seconds(value: str) -> int:
    """Parse one frozen Slurm time-limit string for scheduler-policy validation."""

    return _parse_duration_seconds(value, field="frozen Slurm time limit")


def fleet_partition_time_requirements(
    replicas: Sequence[Any],
) -> dict[str, int]:
    """Derive exact per-partition minimum MaxTime from frozen replica objects."""

    if not replicas:
        raise SchedulerSafetyError("frozen fleet has no replicas")
    requirements: dict[str, int] = {}
    for replica in replicas:
        partition = getattr(replica, "partition", None)
        time_limit = getattr(replica, "time_limit", None)
        if (
            not isinstance(partition, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", partition) is None
            or not isinstance(time_limit, str)
        ):
            raise SchedulerSafetyError(
                "frozen fleet replica has invalid partition/time placement"
            )
        seconds = slurm_time_limit_seconds(time_limit)
        if seconds >= 2**63 - 1:
            raise SchedulerSafetyError(
                "frozen fleet replica time limit must be finite"
            )
        requirements[partition] = max(requirements.get(partition, 0), seconds)
    return dict(sorted(requirements.items()))


def _invoke(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    argv: Sequence[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        process = (
            subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if runner is None
            else runner(list(argv), timeout=timeout)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SchedulerSafetyError(
            f"scheduler safety query failed for {list(argv)!r}: {exc}"
        ) from exc
    if (
        not isinstance(process, subprocess.CompletedProcess)
        or process.returncode != 0
        or not isinstance(process.stdout, str)
        or not isinstance(process.stderr, str)
    ):
        returncode = getattr(process, "returncode", "invalid")
        stderr = str(getattr(process, "stderr", ""))[:500]
        raise SchedulerSafetyError(
            f"scheduler safety query failed rc={returncode}: {stderr}"
        )
    return process


def _partition_fact(
    *,
    partition: str,
    raw_output: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    fields, normalized = _parse_key_value_record(
        raw_output,
        description=f"scontrol partition {partition!r}",
    )
    preempt_mode = fields.get("PreemptMode")
    if (
        fields.get("PartitionName") != partition
        or not isinstance(preempt_mode, str)
        or not preempt_mode
        or fields.get("State") != "UP"
    ):
        raise SchedulerSafetyError(
            f"partition {partition!r} identity/state/preemption policy is invalid"
        )
    modes = tuple(sorted(set(preempt_mode.upper().split(","))))
    if not modes or any(
        mode != "OFF" and mode not in _ACTIVE_PREEMPT_MODES for mode in modes
    ):
        raise SchedulerSafetyError(
            f"partition {partition!r} has unknown PreemptMode={preempt_mode!r}"
        )
    try:
        grace_seconds = int(fields.get("GraceTime", ""))
    except ValueError as exc:
        raise SchedulerSafetyError(
            f"partition {partition!r} has invalid GraceTime"
        ) from exc
    if grace_seconds < 0:
        raise SchedulerSafetyError(
            f"partition {partition!r} has negative GraceTime"
        )
    max_time_seconds = _parse_duration_seconds(
        fields.get("MaxTime", ""),
        field=f"partition {partition!r} MaxTime",
    )
    return {
        "partition": partition,
        "argv": list(argv),
        "raw_output": normalized,
        "raw_output_sha256": _raw_sha256(normalized),
        "preempt_modes": list(modes),
        "grace_time_seconds": grace_seconds,
        "state": fields["State"],
        "max_time": fields["MaxTime"],
        "max_time_seconds": max_time_seconds,
    }


def _configuration_fact(
    *,
    raw_output: str,
    argv: Sequence[str],
) -> dict[str, Any]:
    fields = _parse_configuration(raw_output)
    kill_wait_match = re.fullmatch(r"([0-9]+)(?:\s+sec)?", fields["KillWait"])
    if kill_wait_match is None:
        raise SchedulerSafetyError(
            f"Slurm KillWait is invalid: {fields['KillWait']!r}"
        )
    return {
        "argv": list(argv),
        "raw_output": raw_output,
        "raw_output_sha256": _raw_sha256(raw_output),
        "preempt_type": fields["PreemptType"],
        "default_preempt_mode": fields["PreemptMode"],
        "kill_wait_seconds": int(kill_wait_match.group(1)),
    }


def capture_scheduler_safety_evidence(
    partitions: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Capture exact live scheduler policy for every fleet partition.

    This function only captures and reparses scheduler truth.  Call
    :func:`validate_scheduler_safety_evidence` to apply the no-redraw policy.
    """

    unique = sorted(set(partitions))
    if (
        not unique
        or len(unique) != len(partitions)
        or any(
            not isinstance(partition, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", partition) is None
            for partition in unique
        )
    ):
        raise SchedulerSafetyError(
            "fleet partitions must be unique safe non-empty names"
        )
    config_argv = ["scontrol", "show", "config"]
    config_process = _invoke(runner, config_argv, timeout=30.0)
    configuration = _configuration_fact(
        raw_output=config_process.stdout,
        argv=config_argv,
    )
    partition_facts: list[dict[str, Any]] = []
    for partition in unique:
        argv = ["scontrol", "show", "partition", partition, "-o"]
        process = _invoke(runner, argv, timeout=30.0)
        partition_facts.append(
            _partition_fact(
                partition=partition,
                raw_output=process.stdout,
                argv=argv,
            )
        )
    timestamp = time.time() if captured_timestamp is None else float(captured_timestamp)
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise SchedulerSafetyError("scheduler-safety capture time is invalid")
    evidence: dict[str, Any] = {
        "schema_version": SCHEDULER_SAFETY_SCHEMA_VERSION,
        "protocol": SCHEDULER_SAFETY_PROTOCOL,
        "captured_timestamp": timestamp,
        "configuration": configuration,
        "partitions": partition_facts,
    }
    evidence["evidence_id"] = _identity_sha256(evidence)
    return evidence


def validate_scheduler_safety_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_partitions: Sequence[str],
    required_time_limits_seconds: Mapping[str, int],
    transport_uncertainty_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reparse evidence and authorize every fleet partition.

    ``PreemptMode=OFF`` is intrinsically safe for the observation window.  Any other
    active preemption mode requires the exact imported immutable transport binding
    supplied by the worker/checkpoint implementation.  The validated binding is
    hash-covered in the returned policy record; passing an arbitrary mapping is never
    sufficient.
    """

    expected = sorted(set(expected_partitions))
    if (
        not expected
        or len(expected) != len(expected_partitions)
        or set(required_time_limits_seconds) != set(expected)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in required_time_limits_seconds.values()
        )
    ):
        raise SchedulerSafetyError("expected fleet partition/time contract is invalid")
    required_root = {
        "schema_version",
        "protocol",
        "captured_timestamp",
        "configuration",
        "partitions",
        "evidence_id",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != required_root
        or evidence.get("schema_version") != SCHEDULER_SAFETY_SCHEMA_VERSION
        or evidence.get("protocol") != SCHEDULER_SAFETY_PROTOCOL
        or not isinstance(evidence.get("captured_timestamp"), (int, float))
        or isinstance(evidence.get("captured_timestamp"), bool)
        or not math.isfinite(float(evidence["captured_timestamp"]))
        or float(evidence["captured_timestamp"]) <= 0
        or evidence.get("evidence_id")
        != _identity_sha256(
            {key: value for key, value in evidence.items() if key != "evidence_id"}
        )
    ):
        raise SchedulerSafetyError("scheduler-safety evidence envelope is invalid")

    configuration = evidence.get("configuration")
    config_fields = {
        "argv",
        "raw_output",
        "raw_output_sha256",
        "preempt_type",
        "default_preempt_mode",
        "kill_wait_seconds",
    }
    if not isinstance(configuration, Mapping) or set(configuration) != config_fields:
        raise SchedulerSafetyError("scheduler configuration evidence is malformed")
    raw_config = configuration.get("raw_output")
    if (
        configuration.get("argv") != ["scontrol", "show", "config"]
        or not isinstance(raw_config, str)
        or configuration.get("raw_output_sha256") != _raw_sha256(raw_config)
        or _configuration_fact(
            raw_output=raw_config,
            argv=["scontrol", "show", "config"],
        )
        != dict(configuration)
    ):
        raise SchedulerSafetyError(
            "scheduler configuration evidence differs from its raw query"
        )

    raw_partitions = evidence.get("partitions")
    if not isinstance(raw_partitions, list) or len(raw_partitions) != len(expected):
        raise SchedulerSafetyError("scheduler partition evidence cardinality drifted")
    observed: dict[str, dict[str, Any]] = {}
    for raw in raw_partitions:
        if not isinstance(raw, Mapping):
            raise SchedulerSafetyError("scheduler partition evidence row is malformed")
        partition = raw.get("partition")
        if not isinstance(partition, str) or partition in observed:
            raise SchedulerSafetyError("scheduler partition identity is ambiguous")
        argv = ["scontrol", "show", "partition", partition, "-o"]
        raw_output = raw.get("raw_output")
        if (
            not isinstance(raw_output, str)
            or raw.get("argv") != argv
            or raw.get("raw_output_sha256") != _raw_sha256(raw_output)
            or _partition_fact(
                partition=partition,
                raw_output=raw_output,
                argv=argv,
            )
            != dict(raw)
        ):
            raise SchedulerSafetyError(
                f"partition {partition!r} evidence differs from its raw query"
            )
        observed[partition] = dict(raw)
    if sorted(observed) != expected:
        raise SchedulerSafetyError("scheduler partition set differs from the fleet")

    preemptible = sorted(
        partition
        for partition, fact in observed.items()
        if fact["preempt_modes"] != ["OFF"]
    )
    for partition, fact in observed.items():
        if fact["max_time_seconds"] < required_time_limits_seconds[partition]:
            raise SchedulerSafetyError(
                f"partition {partition!r} MaxTime is shorter than its fleet jobs"
            )

    validated_transport: Mapping[str, Any] | None = None
    if preemptible:
        if transport_uncertainty_binding is None:
            raise SchedulerSafetyError(
                "preemptible fleet partitions require the immutable no-redraw "
                "transport-uncertainty contract: "
                + ", ".join(preemptible)
            )
        try:
            validated_transport = validate_transport_uncertainty_binding(
                transport_uncertainty_binding
            )
        except SchedulerSafetyError as exc:
            raise SchedulerSafetyError(
                "transport-uncertainty contract failed closed for preemptible "
                f"fleet placement: {exc}"
            ) from exc

    partition_policy = {
        partition: {
            "preempt_modes": fact["preempt_modes"],
            "grace_time_seconds": fact["grace_time_seconds"],
            "max_time_seconds": fact["max_time_seconds"],
            "authorization": (
                "nonpreemptible"
                if partition not in preemptible
                else "trusted_transport_censor"
            ),
        }
        for partition, fact in sorted(observed.items())
    }
    policy_contract: dict[str, Any] = {
        "scheduler_evidence_protocol": SCHEDULER_SAFETY_PROTOCOL,
        "preempt_type": configuration["preempt_type"],
        "partitions": partition_policy,
        "preemptible_partitions": preemptible,
        "transport_uncertainty_binding": (
            None if validated_transport is None else dict(validated_transport)
        ),
    }
    policy: dict[str, Any] = {
        "scheduler_evidence_id": evidence["evidence_id"],
        **policy_contract,
        "policy_contract_id": _identity_sha256(policy_contract),
    }
    policy["policy_id"] = _identity_sha256(policy)
    return policy


def _parse_memory_mib(value: str, *, field: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]?)", str(value).strip(), re.IGNORECASE)
    if match is None:
        raise SchedulerSafetyError(f"{field} is invalid: {value!r}")
    amount = int(match.group(1))
    suffix = match.group(2).upper()
    multipliers = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024**2,
        "P": 1024**3,
    }
    converted = amount * multipliers[suffix]
    if converted != int(converted) or converted < 1:
        raise SchedulerSafetyError(f"{field} cannot be represented as positive MiB")
    return int(converted)


def _qos_capacity_fact(
    *,
    raw_output: str,
    argv: Sequence[str],
    qos: str = CLIENT_PARTITION,
) -> dict[str, Any]:
    if not isinstance(raw_output, str) or "\x00" in raw_output:
        raise SchedulerSafetyError("client QOS output is not safe text")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", qos or "") is None:
        raise SchedulerSafetyError("client QOS partition identity is unsafe")
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SchedulerSafetyError(
            "client QOS query must return exactly one non-empty row"
        )
    fields = lines[0].split("|")
    if len(fields) != 3 or fields[0] != qos:
        raise SchedulerSafetyError("client QOS identity/field count drifted")
    tres: dict[str, str] = {}
    for item in fields[1].split(","):
        key, separator, value = item.partition("=")
        if (
            separator != "="
            or not key
            or not value
            or key in tres
        ):
            raise SchedulerSafetyError("client MaxTRESPerUser is malformed")
        tres[key] = value
    if set(tres) != {"cpu", "mem"}:
        raise SchedulerSafetyError(
            "client MaxTRESPerUser must contain exactly cpu and mem"
        )
    try:
        cpu_limit = int(tres["cpu"])
        max_submit = int(fields[2])
    except ValueError as exc:
        raise SchedulerSafetyError("client QOS numeric limits are malformed") from exc
    memory_limit_mib = _parse_memory_mib(
        tres["mem"], field=f"{qos} MaxTRESPerUser mem"
    )
    normalized = lines[0] + "\n"
    return {
        "argv": list(argv),
        "raw_output": normalized,
        "raw_output_sha256": _raw_sha256(normalized),
        "qos": fields[0],
        "cpu_limit": cpu_limit,
        "memory_limit_mib": memory_limit_mib,
        "max_submit_jobs": max_submit,
    }


def capture_client_capacity_contract(
    *,
    partition: str = CLIENT_PARTITION,
    qos: str | None = None,
    expected_cpu_limit: int = CLIENT_CPU_LIMIT,
    expected_memory_limit_mib: int = CLIENT_MEMORY_LIMIT_MIB,
    expected_max_submit_jobs: int = CLIENT_MAX_SUBMIT_JOBS,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Capture the live non-preemptible client partition and QOS limits."""

    evidence = capture_observed_client_capacity_contract(
        partition=partition,
        qos=qos,
        runner=runner,
        captured_timestamp=captured_timestamp,
    )
    validate_client_capacity_contract(
        evidence,
        expected_partition=partition,
        expected_qos=qos,
        expected_cpu_limit=expected_cpu_limit,
        expected_memory_limit_mib=expected_memory_limit_mib,
        expected_max_submit_jobs=expected_max_submit_jobs,
    )
    return evidence


def capture_observed_client_capacity_contract(
    *,
    partition: str,
    qos: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Capture and self-validate the exact live limits for a candidate partition."""

    scheduler = capture_scheduler_safety_evidence(
        [partition],
        runner=runner,
        captured_timestamp=captured_timestamp,
    )
    effective_qos = partition if qos is None else qos
    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        effective_qos,
        "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
    ]
    qos_process = _invoke(runner, qos_argv, timeout=30.0)
    qos = _qos_capacity_fact(
        raw_output=qos_process.stdout,
        argv=qos_argv,
        qos=effective_qos,
    )
    timestamp = float(scheduler["captured_timestamp"])
    evidence: dict[str, Any] = {
        "schema_version": CLIENT_CAPACITY_SCHEMA_VERSION,
        "protocol": CLIENT_CAPACITY_PROTOCOL,
        "captured_timestamp": timestamp,
        "scheduler": scheduler,
        "qos": qos,
    }
    evidence["evidence_id"] = _identity_sha256(evidence)
    validate_client_capacity_contract(
        evidence,
        expected_partition=partition,
        expected_qos=effective_qos,
        expected_cpu_limit=int(qos["cpu_limit"]),
        expected_memory_limit_mib=int(qos["memory_limit_mib"]),
        expected_max_submit_jobs=int(qos["max_submit_jobs"]),
    )
    return evidence


def validate_client_capacity_contract(
    evidence: Mapping[str, Any],
    *,
    expected_partition: str = CLIENT_PARTITION,
    expected_qos: str | None = None,
    expected_cpu_limit: int = CLIENT_CPU_LIMIT,
    expected_memory_limit_mib: int = CLIENT_MEMORY_LIMIT_MIB,
    expected_max_submit_jobs: int = CLIENT_MAX_SUBMIT_JOBS,
) -> dict[str, Any]:
    """Reparse and enforce one exact, externally authorized client contract."""

    effective_qos = expected_partition if expected_qos is None else expected_qos
    if (
        re.fullmatch(r"[A-Za-z0-9_.-]+", expected_partition or "") is None
        or re.fullmatch(r"[A-Za-z0-9_.-]+", effective_qos or "") is None
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in (
                expected_cpu_limit,
                expected_memory_limit_mib,
                expected_max_submit_jobs,
            )
        )
    ):
        raise SchedulerSafetyError("expected client-capacity contract is invalid")

    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "schema_version",
            "protocol",
            "captured_timestamp",
            "scheduler",
            "qos",
            "evidence_id",
        }
        or evidence.get("schema_version") != CLIENT_CAPACITY_SCHEMA_VERSION
        or evidence.get("protocol") != CLIENT_CAPACITY_PROTOCOL
        or not isinstance(evidence.get("captured_timestamp"), (int, float))
        or isinstance(evidence.get("captured_timestamp"), bool)
        or not math.isfinite(float(evidence["captured_timestamp"]))
        or float(evidence["captured_timestamp"]) <= 0
        or evidence.get("evidence_id")
        != _identity_sha256(
            {key: value for key, value in evidence.items() if key != "evidence_id"}
        )
    ):
        raise SchedulerSafetyError("client-capacity evidence envelope is invalid")
    scheduler = evidence.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise SchedulerSafetyError("client-capacity scheduler evidence is missing")
    policy = validate_scheduler_safety_evidence(
        scheduler,
        expected_partitions=[expected_partition],
        required_time_limits_seconds={
            expected_partition: CLIENT_JOB_TIME_LIMIT_SECONDS
        },
    )
    if (
        scheduler.get("captured_timestamp") != evidence["captured_timestamp"]
        or policy["preemptible_partitions"] != []
        or policy["partitions"][expected_partition]["authorization"]
        != "nonpreemptible"
    ):
        raise SchedulerSafetyError(
            f"{expected_partition} is not a bound non-preemptible client partition"
        )
    qos = evidence.get("qos")
    qos_fields = {
        "argv",
        "raw_output",
        "raw_output_sha256",
        "qos",
        "cpu_limit",
        "memory_limit_mib",
        "max_submit_jobs",
    }
    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        effective_qos,
        "format=Name,MaxTRESPerUser,MaxSubmitJobsPerUser",
    ]
    if (
        not isinstance(qos, Mapping)
        or set(qos) != qos_fields
        or qos.get("argv") != qos_argv
        or not isinstance(qos.get("raw_output"), str)
        or qos.get("raw_output_sha256") != _raw_sha256(qos["raw_output"])
        or _qos_capacity_fact(
            raw_output=qos["raw_output"],
            argv=qos_argv,
            qos=effective_qos,
        )
        != dict(qos)
        or qos.get("qos") != effective_qos
        or qos.get("cpu_limit") != expected_cpu_limit
        or qos.get("memory_limit_mib") != expected_memory_limit_mib
        or qos.get("max_submit_jobs") != expected_max_submit_jobs
    ):
        raise SchedulerSafetyError(
            f"{effective_qos} QOS limits drifted from "
            f"cpu={expected_cpu_limit}, mem={expected_memory_limit_mib}MiB, "
            f"submit={expected_max_submit_jobs}"
        )
    return {
        "evidence_id": evidence["evidence_id"],
        "partition": expected_partition,
        "qos": effective_qos,
        "cpu_limit": expected_cpu_limit,
        "memory_limit_mib": expected_memory_limit_mib,
        "max_submit_jobs": expected_max_submit_jobs,
        "scheduler_policy_id": policy["policy_id"],
        "scheduler_policy_contract_id": policy["policy_contract_id"],
    }


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SchedulerSafetyError(f"squeue JSON has duplicate key {key!r}")
        value[key] = item
    return value


def _slurm_number(
    value: Any,
    *,
    field: str,
    allow_unset: bool,
) -> int | None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"set", "infinite", "number"}
        or not isinstance(value.get("set"), bool)
        or not isinstance(value.get("infinite"), bool)
        or not isinstance(value.get("number"), (int, float))
        or isinstance(value.get("number"), bool)
    ):
        raise SchedulerSafetyError(f"squeue job {field} has invalid number wrapper")
    if value["infinite"] is True:
        raise SchedulerSafetyError(f"squeue job {field} is infinite")
    if value["set"] is False:
        if allow_unset and value["number"] == 0:
            return None
        raise SchedulerSafetyError(f"squeue job {field} is unset")
    number = value["number"]
    if int(number) != number or number < 0:
        raise SchedulerSafetyError(f"squeue job {field} is not a non-negative integer")
    return int(number)


def _parse_user_partition_usage_rows(
    raw_output: str,
    *,
    partition: str,
) -> tuple[list[dict[str, Any]], int, int]:
    try:
        payload = json.loads(
            raw_output,
            object_pairs_hook=_json_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SchedulerSafetyError(f"squeue JSON contains {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
        raise SchedulerSafetyError(f"cannot parse squeue JSON usage: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or not {"jobs", "errors", "warnings"}.issubset(payload)
        or not set(payload).issubset(
            {
                "jobs",
                "errors",
                "warnings",
                "meta",
                "last_update",
                "last_backfill",
            }
        )
        or not isinstance(payload.get("jobs"), list)
        or payload.get("errors") != []
        or payload.get("warnings") != []
    ):
        raise SchedulerSafetyError("squeue JSON usage is incomplete or warned")

    jobs: list[dict[str, Any]] = []
    identities: set[str] = set()
    used_cpus = 0
    used_memory_mib = 0
    active_states = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
    for raw in payload["jobs"]:
        if not isinstance(raw, Mapping) or raw.get("partition") != partition:
            continue
        job_id = raw.get("job_id")
        states = raw.get("job_state")
        if (
            not isinstance(job_id, int)
            or isinstance(job_id, bool)
            or job_id < 1
            or not isinstance(states, list)
            or len(states) != 1
            or states[0] not in active_states
            or not isinstance(raw.get("name"), str)
            or not raw["name"]
            or not isinstance(raw.get("qos"), str)
            or not raw["qos"]
            or (
                raw.get("comment") is not None
                and not isinstance(raw.get("comment"), str)
            )
        ):
            raise SchedulerSafetyError(
                "mit_normal squeue job identity/state is malformed"
            )
        task_id = _slurm_number(
            raw.get("array_task_id"),
            field="array_task_id",
            allow_unset=True,
        )
        identity = str(job_id) if task_id is None else f"{job_id}_{task_id}"
        if identity in identities:
            raise SchedulerSafetyError(
                f"mit_normal squeue job {identity} appears more than once"
            )
        identities.add(identity)
        cpus = _slurm_number(raw.get("cpus"), field="cpus", allow_unset=False)
        nodes = _slurm_number(
            raw.get("node_count"), field="node_count", allow_unset=False
        )
        per_cpu = _slurm_number(
            raw.get("memory_per_cpu"),
            field="memory_per_cpu",
            allow_unset=True,
        )
        per_node = _slurm_number(
            raw.get("memory_per_node"),
            field="memory_per_node",
            allow_unset=True,
        )
        assert cpus is not None and nodes is not None
        if cpus < 1 or nodes < 1 or (per_cpu is None) == (per_node is None):
            raise SchedulerSafetyError(
                f"mit_normal squeue job {identity} has ambiguous CPU/memory request"
            )
        memory_mib = (
            int(per_cpu) * cpus
            if per_cpu is not None
            else int(per_node) * nodes
        )
        if memory_mib < 1:
            raise SchedulerSafetyError(
                f"mit_normal squeue job {identity} has no memory reservation"
            )
        jobs.append(
            {
                "job_id": identity,
                "job_name": raw["name"],
                "comment": str(raw.get("comment") or ""),
                "state": states[0],
                "qos": raw["qos"],
                "cpus": cpus,
                "memory_mib": memory_mib,
            }
        )
        used_cpus += cpus
        used_memory_mib += memory_mib
    return (
        sorted(jobs, key=lambda row: row["job_id"]),
        used_cpus,
        used_memory_mib,
    )


def capture_user_partition_usage(
    *,
    user: str,
    partition: str = CLIENT_PARTITION,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Return exact active/pending CPU and memory requests for one user/partition."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+", user or "") is None:
        raise SchedulerSafetyError("scheduler usage requires a safe exact user name")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", partition or "") is None:
        raise SchedulerSafetyError(
            "production client usage requires a safe exact partition"
        )
    argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        user,
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    process = _invoke(runner, argv, timeout=30.0)
    normalized_jobs, used_cpus, used_memory_mib = (
        _parse_user_partition_usage_rows(
            process.stdout,
            partition=partition,
        )
    )
    timestamp = time.time() if captured_timestamp is None else float(captured_timestamp)
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise SchedulerSafetyError("scheduler usage capture time is invalid")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r6-user-partition-tres-usage",
        "captured_timestamp": timestamp,
        "user": user,
        "partition": partition,
        "argv": argv,
        "raw_output": process.stdout,
        "raw_output_sha256": _raw_sha256(process.stdout),
        "jobs": normalized_jobs,
        "job_count": len(normalized_jobs),
        "used_cpus": used_cpus,
        "used_memory_mib": used_memory_mib,
    }
    evidence["evidence_id"] = _identity_sha256(evidence)
    validate_user_partition_usage(evidence)
    return evidence


def validate_user_partition_usage(
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    """Reparse a captured squeue response and return its exact TRES totals."""

    expected_fields = {
        "schema_version",
        "protocol",
        "captured_timestamp",
        "user",
        "partition",
        "argv",
        "raw_output",
        "raw_output_sha256",
        "jobs",
        "job_count",
        "used_cpus",
        "used_memory_mib",
        "evidence_id",
    }
    user = usage.get("user") if isinstance(usage, Mapping) else None
    expected_argv = [
        "squeue",
        "--json",
        "-r",
        "-u",
        user,
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
    ]
    raw_output = usage.get("raw_output") if isinstance(usage, Mapping) else None
    if (
        not isinstance(usage, Mapping)
        or set(usage) != expected_fields
        or usage.get("schema_version") != 1
        or usage.get("protocol")
        != "schema5-v1.2-r6-user-partition-tres-usage"
        or not isinstance(usage.get("captured_timestamp"), (int, float))
        or isinstance(usage.get("captured_timestamp"), bool)
        or not math.isfinite(float(usage["captured_timestamp"]))
        or float(usage["captured_timestamp"]) <= 0
        or not isinstance(user, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", user) is None
        or not isinstance(usage.get("partition"), str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", usage["partition"]) is None
        or usage.get("argv") != expected_argv
        or not isinstance(raw_output, str)
        or usage.get("raw_output_sha256") != _raw_sha256(raw_output)
        or usage.get("evidence_id")
        != _identity_sha256(
            {key: value for key, value in usage.items() if key != "evidence_id"}
        )
    ):
        raise SchedulerSafetyError("client partition usage evidence is invalid")
    jobs, used_cpus, used_memory_mib = _parse_user_partition_usage_rows(
        raw_output,
        partition=str(usage["partition"]),
    )
    if (
        usage.get("jobs") != jobs
        or usage.get("job_count") != len(jobs)
        or usage.get("used_cpus") != used_cpus
        or usage.get("used_memory_mib") != used_memory_mib
    ):
        raise SchedulerSafetyError(
            "client partition usage totals differ from the raw scheduler response"
        )
    return {
        "evidence_id": usage["evidence_id"],
        "jobs": jobs,
        "job_count": len(jobs),
        "used_cpus": used_cpus,
        "used_memory_mib": used_memory_mib,
    }


def _validate_scheduler_bindings(
    bindings: Mapping[str, Mapping[str, str]] | None,
    *,
    description: str,
) -> dict[str, dict[str, str]]:
    """Normalize exact scheduler identities supplied by a durable caller ledger."""

    if bindings is None:
        return {}
    if not isinstance(bindings, Mapping):
        raise SchedulerSafetyError(f"{description} bindings must be a mapping")
    normalized: dict[str, dict[str, str]] = {}
    for identity, binding in bindings.items():
        if (
            not isinstance(identity, str)
            or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identity) is None
            or not isinstance(binding, Mapping)
            or set(binding) != {"job_name", "comment"}
            or not isinstance(binding.get("job_name"), str)
            or not binding["job_name"]
            or not isinstance(binding.get("comment"), str)
            or not binding["comment"]
        ):
            raise SchedulerSafetyError(
                f"{description} scheduler binding is malformed: {identity!r}"
            )
        normalized[identity] = {
            "job_name": str(binding["job_name"]),
            "comment": str(binding["comment"]),
        }
    return normalized


def client_task_headroom(
    usage: Mapping[str, Any],
    *,
    cell_cpus: int,
    cell_memory_mib: int,
    invisible_reserved_tasks: int = 0,
    cpu_limit: int = CLIENT_CPU_LIMIT,
    memory_limit_mib: int = CLIENT_MEMORY_LIMIT_MIB,
    max_submit_jobs: int = CLIENT_MAX_SUBMIT_JOBS,
    reserve_jobs: int = 64,
    cell_ceiling: int | None = None,
    absolute_job_ceiling: int | None = None,
    live_user_job_elements: int | None = None,
    trusted_client_jobs: Mapping[str, Mapping[str, str]] | None = None,
    trusted_nonclient_jobs: Mapping[str, Mapping[str, str]] | None = None,
) -> int:
    """Derive safe new cells under the inclusive 384-client/448-job contract.

    ``cpu_limit`` and ``memory_limit_mib`` describe the *client-only* envelope.
    Provenance-verified serving allocations therefore count toward the absolute
    user-job ceiling but are not charged against that envelope a second time.
    Every same-placement job that is not bound by an exact durable
    ``job_id/name/comment`` tuple is conservatively treated as a client consumer.
    """

    summary = validate_user_partition_usage(usage)
    clients = _validate_scheduler_bindings(
        trusted_client_jobs,
        description="trusted client",
    )
    nonclients = _validate_scheduler_bindings(
        trusted_nonclient_jobs,
        description="trusted non-client",
    )
    if set(clients) & set(nonclients):
        raise SchedulerSafetyError(
            "scheduler identities cannot be both trusted clients and non-clients"
        )
    effective_absolute_ceiling = (
        max_submit_jobs
        if absolute_job_ceiling is None
        else absolute_job_ceiling
    )
    effective_cell_ceiling = (
        effective_absolute_ceiling - reserve_jobs
        if cell_ceiling is None
        else cell_ceiling
    )
    effective_live_jobs = (
        int(summary["job_count"])
        if live_user_job_elements is None
        else live_user_job_elements
    )
    if (
        not isinstance(cell_cpus, int)
        or isinstance(cell_cpus, bool)
        or cell_cpus < 1
        or not isinstance(cell_memory_mib, int)
        or isinstance(cell_memory_mib, bool)
        or cell_memory_mib < 1
        or not isinstance(invisible_reserved_tasks, int)
        or isinstance(invisible_reserved_tasks, bool)
        or invisible_reserved_tasks < 0
        or not isinstance(cpu_limit, int)
        or isinstance(cpu_limit, bool)
        or cpu_limit < 1
        or not isinstance(memory_limit_mib, int)
        or isinstance(memory_limit_mib, bool)
        or memory_limit_mib < 1
        or not isinstance(max_submit_jobs, int)
        or isinstance(max_submit_jobs, bool)
        or max_submit_jobs < 1
        or not isinstance(reserve_jobs, int)
        or isinstance(reserve_jobs, bool)
        or reserve_jobs < 0
        or not isinstance(effective_cell_ceiling, int)
        or isinstance(effective_cell_ceiling, bool)
        or effective_cell_ceiling < 0
        or not isinstance(effective_absolute_ceiling, int)
        or isinstance(effective_absolute_ceiling, bool)
        or effective_absolute_ceiling < 1
        or effective_absolute_ceiling > max_submit_jobs
        or effective_cell_ceiling + reserve_jobs
        > effective_absolute_ceiling
        or not isinstance(effective_live_jobs, int)
        or isinstance(effective_live_jobs, bool)
        or effective_live_jobs < int(summary["job_count"])
    ):
        raise SchedulerSafetyError("client task headroom inputs are invalid")

    client_consumers = 0
    used_client_cpus = 0
    used_client_memory_mib = 0
    for row in summary["jobs"]:
        identity = str(row["job_id"])
        observed = {
            "job_name": str(row["job_name"]),
            "comment": str(row["comment"]),
        }
        if nonclients.get(identity) == observed:
            # This allocation consumes one of the inclusive 64 non-cell slots and is
            # already represented in ``effective_live_jobs``.
            continue
        # Exact trusted clients and unknown/mismatched jobs are both charged to the
        # client envelope.  A forged or stale trusted binding can never create room.
        client_consumers += 1
        used_client_cpus += int(row["cpus"])
        used_client_memory_mib += int(row["memory_mib"])

    remaining_cpus = (
        cpu_limit
        - used_client_cpus
        - invisible_reserved_tasks * cell_cpus
    )
    remaining_memory = (
        memory_limit_mib
        - used_client_memory_mib
        - invisible_reserved_tasks * cell_memory_mib
    )
    remaining_cells = (
        effective_cell_ceiling
        - client_consumers
        - invisible_reserved_tasks
    )
    remaining_submit_jobs = (
        effective_absolute_ceiling
        - effective_live_jobs
        - invisible_reserved_tasks
    )
    visible_headroom = min(
        remaining_cpus // cell_cpus,
        remaining_memory // cell_memory_mib,
        remaining_cells,
        remaining_submit_jobs,
    )
    return max(0, visible_headroom)
