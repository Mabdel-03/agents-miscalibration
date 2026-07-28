#!/usr/bin/env python3
"""External pre-control watchdog for one held schema-5 recovery transaction.

The VM has a distinct restricted SSH key.  The cluster-side forced command accepts
only ``bootstrap-status`` and ``bootstrap-repair``; this runtime cannot release a
root, resume production, clear a hold, or submit scientific work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL = "schema5-v1.2-r4-bootstrap-watchdog-config-v1"
STATUS_PROTOCOL = "schema5-v1.2-r4-bootstrap-status-v1"
RECOVERY_JOB_COUNT = 43
SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT = __import__("re").compile(r"[0-9a-f]{40}\Z")


class BootstrapWatchdogError(RuntimeError):
    """The pre-control watchdog cannot prove a safe, exact action."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _read_sealed(path: Path, description: str) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.resolve(strict=True) != lexical:
        raise BootstrapWatchdogError(f"{description} traverses a symlink")
    metadata = lexical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o222
        or metadata.st_nlink != 1
    ):
        raise BootstrapWatchdogError(f"{description} is not sealed")
    raw = lexical.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapWatchdogError(
            f"{description} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise BootstrapWatchdogError(
            f"{description} is not canonical JSON"
        )
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = _read_sealed(path, "bootstrap watchdog configuration")
    required = {
        "schema_version",
        "protocol",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest_sha256",
        "anchor_submission_receipt_id",
        "anchor_submission_receipt_sha256",
        "harness_environment_binding",
        "materialization_pilot",
        "remote",
        "state_root",
        "observation_gap_seconds",
        "timer_seconds",
    }
    remote = value.get("remote")
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("protocol") != PROTOCOL
        or GIT_OBJECT.fullmatch(
            str(value.get("release_git_commit", ""))
        )
        is None
        or GIT_OBJECT.fullmatch(
            str(value.get("release_tag_object", ""))
        )
        is None
        or any(
            SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "chain_id",
                "chain_manifest_sha256",
                "anchor_submission_receipt_id",
                "anchor_submission_receipt_sha256",
            )
        )
        or not isinstance(value.get("harness_environment_binding"), dict)
        or SHA256.fullmatch(
            str(
                value["harness_environment_binding"].get(
                    "manifest_sha256", ""
                )
            )
        )
        is None
        or not isinstance(value.get("materialization_pilot"), dict)
        or set(value["materialization_pilot"])
        != {
            "marker",
            "marker_sha256",
            "pilot_id",
            "pilot_root",
            "release_worktree",
            "release_bundle",
            "source_tree_sha256",
            "release_bundle_id",
            "release_identity_sha256",
            "release_completion_sha256",
        }
        or any(
            SHA256.fullmatch(
                str(value["materialization_pilot"].get(field, ""))
            )
            is None
            for field in (
                "marker_sha256",
                "pilot_id",
                "source_tree_sha256",
                "release_bundle_id",
                "release_identity_sha256",
                "release_completion_sha256",
            )
        )
        or any(
            not isinstance(
                value["materialization_pilot"].get(field), str
            )
            or not value["materialization_pilot"][field]
            for field in (
                "marker",
                "pilot_root",
                "release_worktree",
                "release_bundle",
            )
        )
        or value.get("observation_gap_seconds") != 60
        or value.get("timer_seconds") != 300
        or not isinstance(remote, dict)
        or set(remote)
        != {
            "host",
            "user",
            "identity_file",
            "known_hosts_file",
        }
    ):
        raise BootstrapWatchdogError(
            "bootstrap watchdog configuration identity is invalid"
        )
    return value


def _ssh(config: Mapping[str, Any], selector: str) -> dict[str, Any]:
    remote = config["remote"]
    argv = [
        "/usr/bin/ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"UserKnownHostsFile={remote['known_hosts_file']}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-i",
        str(remote["identity_file"]),
        f"{remote['user']}@{remote['host']}",
        selector,
    ]
    process = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        },
    )
    if process.returncode != 0:
        raise BootstrapWatchdogError(
            f"forced selector failed: {process.stderr.strip()[:500]}"
        )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapWatchdogError(
            f"forced selector returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BootstrapWatchdogError(
            "forced selector returned a non-object"
        )
    return value


def _validate_status(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    candidate = dict(value)
    observation_id = candidate.pop("observation_id", None)
    observation_artifact = candidate.pop("observation_artifact", None)
    observation_artifact_sha256 = candidate.pop(
        "observation_artifact_sha256", None
    )
    observed_at = value.get("observed_at_timestamp")
    repair_generation = value.get("repair_generation")
    valid_generation = (
        isinstance(repair_generation, int)
        and not isinstance(repair_generation, bool)
        and repair_generation >= 0
    )
    lineage_length = (
        repair_generation + 1 if valid_generation else -1
    )
    required = {
        "schema_version",
        "protocol",
        "passed",
        "observed_at_timestamp",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "anchor_submission_receipt",
        "anchor_submission_receipt_sha256",
        "anchor_submission_receipt_id",
        "descendant_chain_validated",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "root_held",
        "squeue_complete",
        "sacct_complete",
        "namespace_scan_complete",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
        "ambiguous_jobs",
        "job_count",
        "jobs",
        "recovery_namespace_cancelled",
        "anchor_launch_authorized",
        "anchor_launch_id",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "descendant_release_required",
        "handoff_complete",
        "watchdog_scientific_jobs_submitted",
        "observation_id",
        "observation_artifact",
        "observation_artifact_sha256",
    }
    status_jobs = value.get("jobs")
    status_job_ids = (
        [
            str(row.get("job_id", ""))
            for row in status_jobs
            if isinstance(row, Mapping)
        ]
        if isinstance(status_jobs, list)
        else []
    )
    lineage_job_ids = value.get("namespace_lineage_job_ids")
    flattened_lineage_job_ids = (
        [
            str(job_id)
            for generation_ids in lineage_job_ids
            if isinstance(generation_ids, list)
            for job_id in generation_ids
        ]
        if isinstance(lineage_job_ids, list)
        else []
    )
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("protocol") != STATUS_PROTOCOL
        or value.get("passed") is not True
        or value.get("chain_id") != config["chain_id"]
        or value.get("release_git_commit")
        != config["release_git_commit"]
        or value.get("release_tag_object")
        != config["release_tag_object"]
        or value.get("chain_manifest_sha256")
        != config["chain_manifest_sha256"]
        or value.get("anchor_submission_receipt_id")
        != config["anchor_submission_receipt_id"]
        or value.get("anchor_submission_receipt_sha256")
        != config["anchor_submission_receipt_sha256"]
        or value.get("descendant_chain_validated") is not True
        or value.get("squeue_complete") is not True
        or value.get("sacct_complete") is not True
        or value.get("namespace_scan_complete") is not True
        or not isinstance(value.get("namespace_lineage_receipt_ids"), list)
        or len(value["namespace_lineage_receipt_ids"])
        != lineage_length
        or value["namespace_lineage_receipt_ids"][0]
        != config["anchor_submission_receipt_id"]
        or value["namespace_lineage_receipt_ids"][-1]
        != value.get("submission_receipt_id")
        or len(value["namespace_lineage_receipt_ids"])
        != len(set(value["namespace_lineage_receipt_ids"]))
        or any(
            SHA256.fullmatch(str(item)) is None
            for item in value["namespace_lineage_receipt_ids"]
        )
        or not isinstance(lineage_job_ids, list)
        or len(lineage_job_ids) != lineage_length
        or any(
            not isinstance(generation_ids, list)
            or len(generation_ids) != RECOVERY_JOB_COUNT
            or len(set(generation_ids)) != RECOVERY_JOB_COUNT
            or any(
                not isinstance(job_id, str) or not job_id.isdigit()
                for job_id in generation_ids
            )
            for generation_ids in lineage_job_ids
        )
        or (
            isinstance(lineage_job_ids, list)
            and bool(lineage_job_ids)
            and lineage_job_ids[-1] != status_job_ids
        )
        or not isinstance(value.get("namespace_bound_job_ids"), list)
        or any(
            not isinstance(item, str) or not item.isdigit()
            for item in value["namespace_bound_job_ids"]
        )
        or value["namespace_bound_job_ids"]
        != sorted(value["namespace_bound_job_ids"], key=int)
        or len(value["namespace_bound_job_ids"])
        != len(set(value["namespace_bound_job_ids"]))
        or sorted(set(flattened_lineage_job_ids), key=int)
        != value["namespace_bound_job_ids"]
        or value.get("job_count") != RECOVERY_JOB_COUNT
        or not isinstance(status_jobs, list)
        or len(status_jobs) != RECOVERY_JOB_COUNT
        or len(status_job_ids) != RECOVERY_JOB_COUNT
        or len(set(status_job_ids)) != RECOVERY_JOB_COUNT
        or any(not job_id.isdigit() for job_id in status_job_ids)
        or not set(status_job_ids)
        <= set(value["namespace_bound_job_ids"])
        or len(value["namespace_bound_job_ids"]) < value["job_count"]
        or len(value["namespace_bound_job_ids"])
        < RECOVERY_JOB_COUNT
        + (repair_generation if valid_generation else 0)
        or len(value["namespace_bound_job_ids"])
        > RECOVERY_JOB_COUNT
        * (repair_generation + 1 if valid_generation else 0)
        or value.get("ambiguous_jobs") != 0
        or value.get("watchdog_scientific_jobs_submitted") != 0
        or not isinstance(observation_id, str)
        or SHA256.fullmatch(observation_id) is None
        or observation_id
        != hashlib.sha256(_canonical(candidate)).hexdigest()
        or not isinstance(observation_artifact, str)
        or not observation_artifact.startswith("/")
        or SHA256.fullmatch(
            str(observation_artifact_sha256)
        )
        is None
        or not isinstance(observed_at, (int, float))
        or isinstance(observed_at, bool)
        or not math.isfinite(float(observed_at))
        or not valid_generation
        or SHA256.fullmatch(
            str(value.get("submission_receipt_id", ""))
        )
        is None
        or SHA256.fullmatch(
            str(value.get("submission_receipt_sha256", ""))
        )
        is None
        or SHA256.fullmatch(
            str(value.get("generation_provenance_sha256", ""))
        )
        is None
        or SHA256.fullmatch(
            str(value.get("generation_provenance_id", ""))
        )
        is None
        or not isinstance(value.get("generation_provenance"), str)
        or not isinstance(value.get("root_name"), str)
        or not str(value.get("root_job_id", "")).isdigit()
        or not isinstance(value.get("root_names"), list)
        or not value["root_names"]
        or value["root_names"][0] != value.get("root_name")
        or len(value["root_names"]) != len(set(value["root_names"]))
        or not isinstance(value.get("root_job_ids"), list)
        or len(value["root_job_ids"]) != len(value["root_names"])
        or value["root_job_ids"][0] != value.get("root_job_id")
        or any(not str(item).isdigit() for item in value["root_job_ids"])
        or not set(value["root_job_ids"])
        <= set(value["namespace_bound_job_ids"])
        or not isinstance(value.get("roots_held"), bool)
        or value.get("roots_held") is not value.get("root_held")
        or not isinstance(value.get("anchor_launch_authorized"), bool)
        or not isinstance(value.get("anchor_armed_authorized"), bool)
        or not isinstance(
            value.get("anchor_release_intent_authorized"), bool
        )
        or (
            value.get("anchor_armed_authorized") is True
            and SHA256.fullmatch(
                str(value.get("anchor_armed_id", ""))
            )
            is None
        )
        or (
            value.get("anchor_armed_authorized") is False
            and value.get("anchor_armed_id") is not None
        )
        or (
            value.get("anchor_release_intent_authorized") is True
            and SHA256.fullmatch(
                str(value.get("anchor_release_intent_sha256", ""))
            )
            is None
        )
        or (
            value.get("anchor_release_intent_authorized") is False
            and value.get("anchor_release_intent_sha256") is not None
        )
        or (
            value.get("anchor_launch_authorized") is True
            and SHA256.fullmatch(
                str(value.get("anchor_launch_id", ""))
            )
            is None
        )
        or (
            value.get("anchor_launch_authorized") is False
            and value.get("anchor_launch_id") is not None
        )
        or not isinstance(
            value.get("descendant_release_required"), bool
        )
        or not isinstance(
            value.get("descendant_rearm_required"), bool
        )
        or (
            value.get("descendant_rearm_required") is True
            and (
                repair_generation == 0
                or value.get("root_held") is not True
                or value.get("anchor_armed_authorized") is not True
                or value.get("descendant_armed") is not None
                or value.get("descendant_armed_id") is not None
            )
        )
        or (
            value.get("descendant_armed") is not None
            and (
                not isinstance(value.get("descendant_armed"), str)
                or SHA256.fullmatch(
                    str(value.get("descendant_armed_id", ""))
                )
                is None
            )
        )
        or (
            value.get("descendant_release_required") is True
            and (
                repair_generation == 0
                or value.get("root_held") is not True
                or not (
                    value.get("anchor_launch_authorized") is True
                    or (
                        value.get(
                            "anchor_release_intent_authorized"
                        )
                        is True
                        and value.get("descendant_armed") is not None
                    )
                )
            )
        )
    ):
        raise BootstrapWatchdogError(
            "bootstrap scheduler observation is incomplete or ambiguous"
        )
    expected_job_fields = {
        "name",
        "job_id",
        "comment",
        "job_name",
        "state",
        "active",
        "script_sha256",
        "submit_line_sha256",
        "submit_line_exact",
        "scontrol_command",
        "scontrol_requeue",
        "spooled_script_sha256",
        "spooled_script_exact_match",
    }
    jobs_by_name: dict[str, Mapping[str, Any]] = {}
    comment_prefix = (
        f"asys:s5-recovery-v1.2-r4:{config['chain_id']}:g"
    )
    for row in status_jobs:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_job_fields
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or row["name"] in jobs_by_name
            or not str(row.get("job_id", "")).isdigit()
            or not isinstance(row.get("comment"), str)
            or not row["comment"].startswith(comment_prefix)
            or not isinstance(row.get("job_name"), str)
            or not row["job_name"]
            or not isinstance(row.get("state"), str)
            or not row["state"]
            or not isinstance(row.get("active"), bool)
            or any(
                SHA256.fullmatch(str(row.get(field, ""))) is None
                for field in (
                    "script_sha256",
                    "submit_line_sha256",
                    "spooled_script_sha256",
                )
            )
            or row.get("submit_line_exact") is not True
            or not isinstance(row.get("scontrol_command"), str)
            or not row["scontrol_command"].startswith("/")
            or row.get("scontrol_requeue") != 0
            or row.get("spooled_script_exact_match") is not True
        ):
            raise BootstrapWatchdogError(
                "bootstrap status current-job binding is invalid"
            )
        jobs_by_name[str(row["name"])] = row
    if (
        list(value["root_names"])
        != [
            name
            for name in value["root_names"]
            if name in jobs_by_name
        ]
        or [
            str(jobs_by_name[name]["job_id"])
            for name in value["root_names"]
        ]
        != value["root_job_ids"]
    ):
        raise BootstrapWatchdogError(
            "bootstrap status roots are not bound to current receipt jobs"
        )


def _same_descendant(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    fields = (
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "anchor_launch_authorized",
        "anchor_launch_id",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "descendant_release_required",
        "handoff_complete",
    )
    return all(first.get(field) == second.get(field) for field in fields)


def _validate_repair(
    repair: Mapping[str, Any],
    *,
    observation: Mapping[str, Any],
    config: Mapping[str, Any],
    release_required: bool,
    rearm_required: bool = False,
) -> bool:
    """Validate one exact descendant reconstruction/release response."""

    if (
        repair.get("passed") is not True
        or repair.get("watchdog_scientific_jobs_submitted") != 0
        or repair.get("anchor_submission_receipt_id")
        != config["anchor_submission_receipt_id"]
        or repair.get("anchor_submission_receipt_sha256")
        != config["anchor_submission_receipt_sha256"]
        or repair.get("anchor_launch_authorized")
        != observation.get("anchor_launch_authorized")
        or repair.get("anchor_launch_id")
        != observation.get("anchor_launch_id")
        or SHA256.fullmatch(
            str(repair.get("submission_receipt_id", ""))
        )
        is None
        or SHA256.fullmatch(
            str(repair.get("submission_receipt_sha256", ""))
        )
        is None
        or SHA256.fullmatch(
            str(repair.get("generation_provenance_sha256", ""))
        )
        is None
        or SHA256.fullmatch(
            str(repair.get("generation_provenance_id", ""))
        )
        is None
        or not isinstance(repair.get("generation_provenance"), str)
        or not isinstance(repair.get("repair_generation"), int)
        or isinstance(repair.get("repair_generation"), bool)
        or not isinstance(repair.get("root_name"), str)
        or not str(repair.get("root_job_id", "")).isdigit()
        or not isinstance(repair.get("root_names"), list)
        or not repair["root_names"]
        or len(repair["root_names"]) != len(set(repair["root_names"]))
        or repair["root_names"][0] != repair.get("root_name")
        or not isinstance(repair.get("root_job_ids"), list)
        or len(repair["root_job_ids"]) != len(repair["root_names"])
        or repair["root_job_ids"][0] != repair.get("root_job_id")
        or any(not str(item).isdigit() for item in repair["root_job_ids"])
        or not isinstance(repair.get("roots_held"), bool)
        or repair.get("roots_held") is not repair.get("root_held")
    ):
        raise BootstrapWatchdogError(
            "bootstrap repair response lacks exact immutable lineage"
        )
    if rearm_required:
        continuation_authorized = (
            observation.get("anchor_launch_authorized") is True
            or observation.get("anchor_release_intent_authorized") is True
        )
        valid = (
            repair.get("repair_generation")
            == observation.get("repair_generation")
            and repair.get("submission_receipt_id")
            == observation.get("submission_receipt_id")
            and repair.get("submission_receipt_sha256")
            == observation.get("submission_receipt_sha256")
            and repair.get("generation_provenance_id")
            == observation.get("generation_provenance_id")
            and repair.get("root_name") == observation.get("root_name")
            and repair.get("root_job_id")
            == observation.get("root_job_id")
            and repair.get("root_names")
            == observation.get("root_names")
            and repair.get("root_job_ids")
            == observation.get("root_job_ids")
            and SHA256.fullmatch(
                str(repair.get("descendant_armed_id", ""))
            )
            is not None
            and (
                (
                    repair.get("status")
                    == "bootstrap_descendant_rearmed_launched"
                    and repair.get("root_held") is False
                    and repair.get("root_released") is True
                )
                if continuation_authorized
                else (
                    repair.get("status")
                    == "bootstrap_descendant_rearmed_held"
                    and repair.get("root_held") is True
                    and repair.get("root_released") is False
                )
            )
        )
    elif release_required:
        valid = (
            repair.get("status") == "bootstrap_repair_reconciled"
            and repair.get("repair_generation")
            == observation.get("repair_generation")
            and repair.get("submission_receipt_id")
            == observation.get("submission_receipt_id")
            and repair.get("submission_receipt_sha256")
            == observation.get("submission_receipt_sha256")
            and repair.get("generation_provenance")
            == observation.get("generation_provenance")
            and repair.get("generation_provenance_sha256")
            == observation.get("generation_provenance_sha256")
            and repair.get("generation_provenance_id")
            == observation.get("generation_provenance_id")
            and repair.get("root_name") == observation.get("root_name")
            and repair.get("root_job_id") == observation.get("root_job_id")
            and repair.get("root_names")
            == observation.get("root_names")
            and repair.get("root_job_ids")
            == observation.get("root_job_ids")
            and repair.get("root_held") is False
            and repair.get("root_released") is True
        )
    else:
        next_generation = int(observation["repair_generation"]) + 1
        valid = (
            repair.get("repair_generation") == next_generation
            and repair.get("parent_submission_receipt_id")
            == observation.get("submission_receipt_id")
            and repair.get("status") == "bootstrap_repaired_held"
            and repair.get("root_held") is True
            and repair.get("root_released") is False
        )
    if not valid:
        raise BootstrapWatchdogError(
            "bootstrap repair did not produce the exact authorized descendant"
        )
    if repair.get("root_released") is True and (
        SHA256.fullmatch(str(repair.get("root_release_id", ""))) is None
        or SHA256.fullmatch(str(repair.get("launch_id", ""))) is None
    ):
        raise BootstrapWatchdogError(
            "released bootstrap repair lacks marker-last release identities"
        )
    return repair.get("root_released") is True


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run_once(
    config_path: Path,
    *,
    sleeper=time.sleep,
    clock=time.time,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_sha256 = hashlib.sha256(
        Path(config_path).read_bytes()
    ).hexdigest()
    first_at = float(clock())
    first = _ssh(config, "schema5-bootstrap-watchdog status")
    _validate_status(first, config)
    sleeper(60)
    second_at = float(clock())
    second = _ssh(config, "schema5-bootstrap-watchdog status")
    _validate_status(second, config)
    if (
        not math.isfinite(first_at)
        or not math.isfinite(second_at)
        or second_at - first_at < 60
        or float(second["observed_at_timestamp"])
        - float(first["observed_at_timestamp"])
        < 60
        or not _same_descendant(first, second)
    ):
        raise BootstrapWatchdogError(
            "bootstrap observations are not at least 60 seconds apart"
        )
    root_release_performed = False
    if first.get("handoff_complete") or second.get("handoff_complete"):
        action = "handoff_complete"
    elif (
        first.get("recovery_namespace_cancelled") is True
        and second.get("recovery_namespace_cancelled") is True
        and first.get("observation_id") != second.get("observation_id")
    ):
        repair = _ssh(config, "schema5-bootstrap-watchdog repair")
        root_release_performed = _validate_repair(
            repair,
            observation=second,
            config=config,
            release_required=False,
        )
        action = "bootstrap_repair"
    elif (
        first.get("descendant_rearm_required") is True
        and second.get("descendant_rearm_required") is True
        and first.get("observation_id") != second.get("observation_id")
    ):
        repair = _ssh(config, "schema5-bootstrap-watchdog repair")
        root_release_performed = _validate_repair(
            repair,
            observation=second,
            config=config,
            release_required=False,
            rearm_required=True,
        )
        action = "bootstrap_descendant_rearm"
    elif (
        first.get("descendant_release_required") is True
        and second.get("descendant_release_required") is True
        and first.get("observation_id") != second.get("observation_id")
    ):
        repair = _ssh(config, "schema5-bootstrap-watchdog repair")
        root_release_performed = _validate_repair(
            repair,
            observation=second,
            config=config,
            release_required=True,
        )
        action = "bootstrap_release_reconcile"
    else:
        action = "healthy_noop"
    result = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r4-bootstrap-watchdog-heartbeat-v1",
        "passed": True,
        "observed_at_timestamp": second_at,
        "configuration_sha256": config_sha256,
        "release_git_commit": config["release_git_commit"],
        "release_tag_object": config["release_tag_object"],
        "chain_id": config["chain_id"],
        "chain_manifest_sha256": config["chain_manifest_sha256"],
        "anchor_submission_receipt_id": config[
            "anchor_submission_receipt_id"
        ],
        "anchor_submission_receipt_sha256": config[
            "anchor_submission_receipt_sha256"
        ],
        "first_observation_id": first.get("observation_id"),
        "second_observation_id": second.get("observation_id"),
        "action": action,
        "recovery_root_release_performed": root_release_performed,
        "production_control_mutated": False,
        "scientific_jobs_submitted": 0,
        "safety_hold_cleared": False,
    }
    result["heartbeat_id"] = hashlib.sha256(_canonical(result)).hexdigest()
    state_root = Path(str(config["state_root"]))
    _atomic_json(state_root / "BOOTSTRAP_WATCHDOG_HEARTBEAT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_once(args.config)
    except (OSError, subprocess.SubprocessError, BootstrapWatchdogError) as exc:
        print(
            json.dumps({"passed": False, "error": str(exc)}, sort_keys=True),
            file=os.sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
