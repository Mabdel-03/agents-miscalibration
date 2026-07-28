from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import build_schema5_watchdog_deployment as deployment
from scripts import publish_schema5_watchdog_ready as publisher


COMMIT = "a" * 40
TAG_OBJECT = "b" * 40
CONTROL = "c" * 64


def _seal(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deployment._canonical(value))
    path.chmod(0o444)
    return path


def _bootstrap_drill_fixture(tmp_path: Path) -> dict[str, object]:
    canonical_root = tmp_path / "drill"
    root = canonical_root / deployment.BOOTSTRAP_ISOLATED_DRILL_DIRECTORY

    def manifest_for(job_root: Path) -> dict:
        jobs: list[dict] = []
        for index in range(43):
            name = "source_checkout" if index == 0 else f"stage-{index:02d}"
            dependencies = (
                []
                if index == 0
                else (
                    [jobs[0]["name"]]
                    if index == 1
                    else (
                        [jobs[1]["name"]]
                        if index in {2, 3}
                        else [jobs[index - 2]["name"]]
                    )
                )
            )
            jobs.append(
                {
                    "name": name,
                    "job_name": f"asys-r6-{index:02d}",
                    "script": str(
                        job_root / "jobs" / f"{index:02d}.sbatch"
                    ),
                    "script_sha256": hashlib.sha256(
                        f"script-{index}".encode()
                    ).hexdigest(),
                    "dependencies": dependencies,
                    "dependency_type": "afterok",
                }
            )
        value = {
            "schema_version": 1,
            "protocol": "schema5-v1.2-r6-recovery-chain",
            "release_git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "recovery_root": str(job_root),
            "jobs": jobs,
        }
        value["chain_id"] = deployment._self_hash(value, "chain_id")
        return value

    canonical_manifest_path = (
        canonical_root / deployment.BOOTSTRAP_CHAIN_MANIFEST_NAME
    )
    canonical_manifest = manifest_for(canonical_root)
    _seal(canonical_manifest_path, canonical_manifest)
    canonical_manifest_sha = hashlib.sha256(
        canonical_manifest_path.read_bytes()
    ).hexdigest()
    canonical_submitted: dict[str, str] = {}
    canonical_rows: list[dict] = []
    for index, row in enumerate(canonical_manifest["jobs"]):
        job_id = str(1000 + index)
        canonical_rows.append(
            {
                "name": row["name"],
                "job_id": job_id,
                "dependencies": list(row["dependencies"]),
                "dependency_job_ids": [
                    canonical_submitted[item]
                    for item in row["dependencies"]
                ],
                "comment": (
                    "asys:s5-recovery-v1.2-r6:"
                    f"{canonical_manifest['chain_id']}:g0000:"
                    f"{row['name']}:fixture"
                ),
                "script": row["script"],
                "script_sha256": row["script_sha256"],
                "dependency_type": row["dependency_type"],
            }
        )
        canonical_submitted[row["name"]] = job_id
    canonical_anchor_path = (
        canonical_root / deployment.BOOTSTRAP_SUBMISSION_RECEIPT_NAME
    )
    canonical_anchor = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r6-recovery-chain-submission",
        "chain_id": canonical_manifest["chain_id"],
        "manifest": str(canonical_manifest_path),
        "manifest_sha256": canonical_manifest_sha,
        "root_initial_hold": True,
        "no_requeue": True,
        "jobs": canonical_rows,
    }
    canonical_anchor["receipt_id"] = deployment._self_hash(
        canonical_anchor, "receipt_id"
    )
    _seal(canonical_anchor_path, canonical_anchor)

    manifest_path = root / deployment.BOOTSTRAP_CHAIN_MANIFEST_NAME
    manifest = manifest_for(root)
    jobs = manifest["jobs"]
    _seal(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def receipt_rows(
        generation: int,
        base: int,
        *,
        parent_rows: list[dict] | None = None,
        reused_prefix: int = 0,
    ) -> list[dict]:
        rows: list[dict] = []
        submitted: dict[str, str] = {}
        for index, row in enumerate(jobs):
            reused = generation > 0 and index < reused_prefix
            parent = (
                parent_rows[index]
                if reused and parent_rows is not None
                else None
            )
            job_id = (
                str(parent["job_id"])
                if parent is not None
                else str(base + index)
            )
            comment = (
                str(parent["comment"])
                if parent is not None
                else (
                    "asys:s5-recovery-v1.2-r6:"
                    f"{manifest['chain_id']}:g{generation:04d}:"
                    f"{row['name']}:fixture"
                )
            )
            record = {
                "name": row["name"],
                "job_id": job_id,
                "dependencies": list(row["dependencies"]),
                "dependency_job_ids": [
                    submitted[item] for item in row["dependencies"]
                ],
                "comment": comment,
                "script": row["script"],
                "script_sha256": row["script_sha256"],
                "dependency_type": row["dependency_type"],
                **(
                    {}
                    if generation == 0
                    else {
                        "generation": 0 if reused else generation,
                        "disposition": (
                            "reused_completed" if reused else "resubmitted"
                        ),
                    }
                ),
            }
            rows.append(record)
            submitted[row["name"]] = job_id
        return rows

    anchor_path = root / deployment.BOOTSTRAP_SUBMISSION_RECEIPT_NAME
    anchor = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r6-recovery-chain-submission",
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "root_initial_hold": True,
        "no_requeue": True,
        "jobs": receipt_rows(0, 3000),
    }
    anchor["receipt_id"] = deployment._self_hash(anchor, "receipt_id")
    _seal(anchor_path, anchor)
    anchor_raw = anchor_path.read_bytes()

    def provenance(
        *,
        path: Path,
        receipt_path: Path,
        receipt: dict,
        generation: int,
        lineage_ids: list[str],
        lineage_job_ids: list[str],
        parent_path: Path | None,
        parent: dict | None,
    ) -> tuple[Path, dict]:
        current_names = [
            row["name"]
            for row in receipt["jobs"]
            if generation == 0 or row.get("generation", 0) == generation
        ]
        generation_roots = (
            ["source_checkout"]
            if generation == 0
            else list(receipt["held_root_names"])
        )
        rows = [
            {
                "name": manifest_row["name"],
                "job_id": receipt_row["job_id"],
                "comment": receipt_row["comment"],
                "job_name": manifest_row["job_name"],
                "script_sha256": manifest_row["script_sha256"],
                "submit_line_sha256": hashlib.sha256(
                    f"submit-{generation}-{index}".encode()
                ).hexdigest(),
                "submit_line_exact": True,
                "scontrol_command": receipt_row["script"],
                "scontrol_requeue": 0,
                "spooled_script_sha256": manifest_row["script_sha256"],
                "spooled_script_exact_match": True,
                "origin_generation": receipt_row.get("generation", 0),
            }
            for index, (manifest_row, receipt_row) in enumerate(
                zip(jobs, receipt["jobs"], strict=True)
            )
        ]
        value = {
            "schema_version": 1,
            "protocol": (
                "schema5-v1.2-r6-bootstrap-generation-provenance-v1"
            ),
            "passed": True,
            "release_git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "chain_id": manifest["chain_id"],
            "chain_manifest": str(manifest_path),
            "chain_manifest_sha256": manifest_sha,
            "submission_receipt": str(receipt_path),
            "submission_receipt_sha256": hashlib.sha256(
                deployment._canonical(receipt)
            ).hexdigest(),
            "submission_receipt_id": receipt["receipt_id"],
            "repair_generation": generation,
            "parent_provenance": (
                None if parent_path is None else str(parent_path)
            ),
            "parent_provenance_sha256": (
                None
                if parent_path is None
                else hashlib.sha256(parent_path.read_bytes()).hexdigest()
            ),
            "parent_provenance_id": (
                None if parent is None else parent["provenance_id"]
            ),
            "namespace_lineage_receipt_ids": lineage_ids,
            "namespace_bound_job_ids": lineage_job_ids,
            "namespace_scan_complete": True,
            "generation_root_names": generation_roots,
            "current_generation_names": current_names,
            "scheduler_topology_valid": True,
            "jobs": rows,
        }
        value["provenance_id"] = deployment._self_hash(
            value, "provenance_id"
        )
        _seal(path, value)
        return path, value

    anchor_provenance_path, anchor_provenance = provenance(
        path=root / "g0000" / "GENERATION_PROVENANCE.json",
        receipt_path=anchor_path,
        receipt=anchor,
        generation=0,
        lineage_ids=[anchor["receipt_id"]],
        lineage_job_ids=[row["job_id"] for row in anchor["jobs"]],
        parent_path=None,
        parent=None,
    )

    def observation(
        *,
        path: Path,
        timestamp: float,
        receipt_path: Path,
        receipt: dict,
        generation: int,
        provenance_path: Path,
        provenance_value: dict,
        cancelled: bool,
        lineage_ids: list[str],
        lineage_job_ids: list[str],
        completed_names: set[str] | None = None,
    ) -> tuple[Path, dict]:
        lineage_job_ids_by_receipt = [
            [row["job_id"] for row in anchor["jobs"]],
            *(
                []
                if generation == 0
                else [[row["job_id"] for row in receipt["jobs"]]]
            ),
        ]
        assert len(lineage_job_ids_by_receipt) == len(lineage_ids)
        current_names = {
            row["name"]
            for row in receipt["jobs"]
            if generation == 0 or row.get("generation", 0) == generation
        }
        root_names = (
            ["source_checkout"]
            if generation == 0
            else list(receipt["held_root_names"])
        )
        root_job_ids = [
            next(
                row["job_id"]
                for row in receipt["jobs"]
                if row["name"] == name
            )
            for name in root_names
        ]
        status_rows = []
        for manifest_row, receipt_row, provenance_row in zip(
            jobs, receipt["jobs"], provenance_value["jobs"], strict=True
        ):
            active = not cancelled and receipt_row["name"] in current_names
            completed = (
                cancelled
                and completed_names is not None
                and receipt_row["name"] in completed_names
            )
            status_rows.append(
                {
                    "name": receipt_row["name"],
                    "job_id": receipt_row["job_id"],
                    "comment": receipt_row["comment"],
                    "job_name": manifest_row["job_name"],
                    "state": (
                        ("COMPLETED" if completed else "CANCELLED")
                        if cancelled
                        else ("PENDING" if active else "COMPLETED")
                    ),
                    "active": active,
                    "script_sha256": manifest_row["script_sha256"],
                    "submit_line_sha256": provenance_row[
                        "submit_line_sha256"
                    ],
                    "submit_line_exact": True,
                    "scontrol_command": receipt_row["script"],
                    "scontrol_requeue": 0,
                    "spooled_script_sha256": manifest_row["script_sha256"],
                    "spooled_script_exact_match": True,
                }
            )
        value = {
            "schema_version": 1,
            "protocol": "schema5-v1.2-r6-bootstrap-status-v1",
            "passed": True,
            "observed_at_timestamp": timestamp,
            "release_git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "chain_id": manifest["chain_id"],
            "chain_manifest": str(manifest_path),
            "chain_manifest_sha256": manifest_sha,
            "submission_receipt": str(receipt_path),
            "submission_receipt_sha256": hashlib.sha256(
                deployment._canonical(receipt)
            ).hexdigest(),
            "submission_receipt_id": receipt["receipt_id"],
            "generation_provenance": str(provenance_path),
            "generation_provenance_sha256": hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest(),
            "generation_provenance_id": provenance_value["provenance_id"],
            "anchor_submission_receipt": str(anchor_path),
            "anchor_submission_receipt_sha256": hashlib.sha256(
                anchor_raw
            ).hexdigest(),
            "anchor_submission_receipt_id": anchor["receipt_id"],
            "descendant_chain_validated": True,
            "repair_generation": generation,
            "root_name": root_names[0],
            "root_job_id": root_job_ids[0],
            "root_names": root_names,
            "root_job_ids": root_job_ids,
            "roots_held": not cancelled,
            "root_held": not cancelled,
            "squeue_complete": True,
            "sacct_complete": True,
            "namespace_scan_complete": True,
            "namespace_lineage_receipt_ids": lineage_ids,
            "namespace_lineage_job_ids": lineage_job_ids_by_receipt,
            "namespace_bound_job_ids": lineage_job_ids,
            "ambiguous_jobs": 0,
            "job_count": len(jobs),
            "jobs": status_rows,
            "recovery_namespace_cancelled": cancelled,
            "anchor_launch_authorized": False,
            "anchor_launch_id": None,
            "anchor_armed_authorized": False,
            "anchor_armed_id": None,
            "anchor_release_intent_authorized": False,
            "anchor_release_intent_sha256": None,
            "descendant_armed": None,
            "descendant_armed_id": None,
            "descendant_rearm_required": False,
            "descendant_release_required": False,
            "handoff_complete": False,
            "watchdog_scientific_jobs_submitted": 0,
        }
        value["observation_id"] = deployment._self_hash(
            value, "observation_id"
        )
        _seal(path, value)
        return path, value

    cancelled = [
        observation(
            path=root / f"cancelled-{index}.json",
            timestamp=timestamp,
            receipt_path=anchor_path,
            receipt=anchor,
            generation=0,
            provenance_path=anchor_provenance_path,
            provenance_value=anchor_provenance,
            cancelled=True,
            lineage_ids=[anchor["receipt_id"]],
            lineage_job_ids=[
                row["job_id"] for row in anchor["jobs"]
            ],
            completed_names={jobs[0]["name"], jobs[1]["name"]},
        )
        for index, timestamp in enumerate((100.0, 160.0), start=1)
    ]

    repair_root = (
        root
        / deployment.BOOTSTRAP_CANONICAL_REPAIR_DIRECTORY
        / "g0001"
    )
    repair_rows = receipt_rows(
        1,
        4000,
        parent_rows=anchor["jobs"],
        reused_prefix=2,
    )
    resubmitted_rows = [
        row
        for row in repair_rows
        if row["disposition"] == "resubmitted"
    ]
    held_root_names = [jobs[2]["name"], jobs[3]["name"]]
    journal_path = repair_root / "SUBMISSION_JOURNAL.json"
    journal = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r6-recovery-chain-repair-journal",
        "chain_id": manifest["chain_id"],
        "repair_generation": 1,
        "base_receipt": str(anchor_path),
        "base_receipt_sha256": hashlib.sha256(anchor_raw).hexdigest(),
        "repair_jobs": [row["name"] for row in resubmitted_rows],
        "held_root_names": held_root_names,
        "jobs": {
            row["name"]: {
                "name": row["name"],
                "comment": row["comment"],
                "job_id": row["job_id"],
                "submission_boundary_state": "committed",
            }
            for row in resubmitted_rows
        },
    }
    _seal(journal_path, journal)
    repair_receipt_path = (
        repair_root / deployment.BOOTSTRAP_SUBMISSION_RECEIPT_NAME
    )
    repair_receipt = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r6-recovery-chain-repair",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "submission_journal": str(journal_path),
        "submission_journal_sha256": hashlib.sha256(
            journal_path.read_bytes()
        ).hexdigest(),
        "repair_generation": 1,
        "parent_receipt": str(anchor_path),
        "parent_receipt_sha256": hashlib.sha256(anchor_raw).hexdigest(),
        "root_initial_hold": True,
        "held_root_names": held_root_names,
        "no_requeue": True,
        "jobs": repair_rows,
    }
    repair_receipt["receipt_id"] = deployment._self_hash(
        repair_receipt, "receipt_id"
    )
    _seal(repair_receipt_path, repair_receipt)
    repair_provenance_path, repair_provenance = provenance(
        path=repair_root / "GENERATION_PROVENANCE.json",
        receipt_path=repair_receipt_path,
        receipt=repair_receipt,
        generation=1,
        lineage_ids=[
            anchor["receipt_id"],
            repair_receipt["receipt_id"],
        ],
        lineage_job_ids=sorted(
            {
                row["job_id"]
                for receipt in (anchor, repair_receipt)
                for row in receipt["jobs"]
            },
            key=int,
        ),
        parent_path=anchor_provenance_path,
        parent=anchor_provenance,
    )
    repair_result = dict(repair_receipt)
    repair_result.update(
        {
            "status": "bootstrap_repaired_held",
            "repair_jobs": [row["name"] for row in resubmitted_rows],
            "submission_receipt": str(repair_receipt_path),
            "submission_receipt_sha256": hashlib.sha256(
                repair_receipt_path.read_bytes()
            ).hexdigest(),
            "submission_receipt_id": repair_receipt["receipt_id"],
            "generation_provenance": str(repair_provenance_path),
            "generation_provenance_sha256": hashlib.sha256(
                repair_provenance_path.read_bytes()
            ).hexdigest(),
            "generation_provenance_id": repair_provenance[
                "provenance_id"
            ],
            "parent_submission_receipt_id": anchor["receipt_id"],
            "anchor_submission_receipt_id": anchor["receipt_id"],
            "anchor_submission_receipt_sha256": hashlib.sha256(
                anchor_raw
            ).hexdigest(),
            "anchor_launch_authorized": False,
            "anchor_launch_id": None,
            "anchor_armed_authorized": False,
            "anchor_armed_id": None,
            "anchor_release_intent_authorized": False,
            "descendant_armed": None,
            "descendant_armed_id": None,
            "root_name": held_root_names[0],
            "root_job_id": repair_rows[2]["job_id"],
            "root_names": held_root_names,
            "root_job_ids": [
                repair_rows[2]["job_id"],
                repair_rows[3]["job_id"],
            ],
            "roots_held": True,
            "root_held": True,
            "root_released": False,
            "root_release_id": None,
            "launch_id": None,
            "watchdog_scientific_jobs_submitted": 0,
        }
    )
    repair_result_path = _seal(root / "repair-result.json", repair_result)
    recovered_path, recovered = observation(
        path=root / "recovered.json",
        timestamp=220.0,
        receipt_path=repair_receipt_path,
        receipt=repair_receipt,
        generation=1,
        provenance_path=repair_provenance_path,
        provenance_value=repair_provenance,
        cancelled=False,
        lineage_ids=[
            anchor["receipt_id"],
            repair_receipt["receipt_id"],
        ],
        lineage_job_ids=sorted(
            {
                row["job_id"]
                for receipt in (anchor, repair_receipt)
                for row in receipt["jobs"]
            },
            key=int,
        ),
    )

    isolated_contract, _isolated_manifest, _isolated_anchor = (
        deployment._bootstrap_isolated_drill_contract(
            canonical_manifest_path=canonical_manifest_path,
            canonical_manifest=canonical_manifest,
            canonical_receipt_path=canonical_anchor_path,
            canonical_receipt=canonical_anchor,
            isolated_manifest_path=manifest_path,
            isolated_receipt_path=anchor_path,
        )
    )
    bundle = {
        "schema_version": 1,
        "protocol": deployment.BOOTSTRAP_BUNDLE_PROTOCOL,
        "chain_manifest": str(canonical_manifest_path),
        "chain_manifest_sha256": canonical_manifest_sha,
        "chain_id": canonical_manifest["chain_id"],
        "submission_receipt": str(canonical_anchor_path),
        "submission_receipt_sha256": hashlib.sha256(
            canonical_anchor_path.read_bytes()
        ).hexdigest(),
        "submission_receipt_id": canonical_anchor["receipt_id"],
        "isolated_cancellation_drill": isolated_contract,
    }
    bundle["bundle_id"] = deployment._self_hash(bundle, "bundle_id")
    bundle_path = _seal(root / "BUNDLE.json", bundle)
    deployed = {
        "schema_version": 1,
        "protocol": deployment.BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL,
        "passed": True,
        "bundle_id": bundle["bundle_id"],
        "deployment_id": "9" * 64,
    }
    deployed["evidence_id"] = deployment._self_hash(
        deployed, "evidence_id"
    )
    deployment_path = _seal(root / "DEPLOYMENT.json", deployed)
    return {
        "bundle": bundle_path,
        "deployment": deployment_path,
        "cancelled": [item[0] for item in cancelled],
        "repair": repair_result_path,
        "recovered": recovered_path,
        "recovery_seconds": 120.0,
        "output": root / "DRILL_COMPLETE.json",
        "values": {
            "manifest": manifest,
            "anchor": anchor,
            "repair": repair_result,
            "recovered": recovered,
        },
    }


def _source_release(
    tmp_path: Path,
    *,
    release_path: Path | None = None,
    prefix_path: Path | None = None,
    environment_manifest_path: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    release = release_path or (tmp_path / "release")
    for relative in (
        "scripts/build_schema5_watchdog_deployment.py",
        "scripts/schema5_external_watchdog.py",
        "scripts/schema5_watchdog_forced_command.py",
        "src/agents_scaling/serving/external_watchdog.py",
        "slurm/schema5_control.py",
    ):
        path = release / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        path.chmod(0o444)
    prefix = prefix_path or (tmp_path / "harness")
    python = prefix / "bin" / "python"
    target = python.with_name("python3.11")
    python.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o555)
    python.symlink_to(target.name)
    python.parent.chmod(0o555)
    prefix.chmod(0o555)
    entries = [
        {
            "path": "bin",
            "type": "directory",
            "mode": python.parent.stat().st_mode & 0o7777,
        },
        {
            "path": "bin/python",
            "type": "symlink",
            "mode": python.lstat().st_mode & 0o7777,
            "target": target.name,
        },
        {
            "path": "bin/python3.11",
            "type": "file",
            "mode": target.stat().st_mode & 0o7777,
            "size": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
    ]
    inventory = {
        "entries": entries,
        "inventory_sha256": hashlib.sha256(
            deployment._compact_canonical(entries)
        ).hexdigest(),
        "entry_count": len(entries),
        "file_count": 1,
        "directory_count": 1,
        "symlink_count": 1,
        "total_file_bytes": target.stat().st_size,
    }
    environment_manifest = environment_manifest_path or (
        release / "harness_environment.schema5-v1.json"
    )
    environment_manifest.parent.mkdir(parents=True, exist_ok=True)
    environment_manifest.write_bytes(
        deployment._canonical(
            {
                "schema_version": 3,
                "release_id": deployment.RELEASE_ID,
                "role": "harness",
                "prefix": str(prefix),
                "sealed_read_only": True,
                "directory_inventory": inventory,
            }
        )
    )
    environment_manifest.chmod(0o444)
    environment_sha256 = hashlib.sha256(
        environment_manifest.read_bytes()
    ).hexdigest()
    state = tmp_path / "control"
    state.mkdir()
    (state / "control.json").write_bytes(
        deployment._canonical(
            {
                "immutable_sha256": CONTROL,
                "immutable": {
                    "harness_environment_prefix": str(prefix),
                    "harness_environment_manifest_path": str(
                        environment_manifest
                    ),
                    "harness_environment_sha256": environment_sha256,
                },
                "desired_state": "paused",
                "drain_requested": False,
                "finalization": {"state": "idle"},
            }
        )
    )
    key = tmp_path / "watchdog.pub"
    key.write_text("ssh-ed25519 QUJDREVGRw== fixture\n", encoding="ascii")
    return release, python, state, key


def _bundle(tmp_path: Path) -> tuple[dict, Path]:
    release, python, state, key = _source_release(tmp_path)
    identity = tmp_path / "vm" / "id_ed25519"
    identity.parent.mkdir()
    identity.write_text("private fixture\n", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "vm" / "known_hosts"
    known_hosts.write_text("cluster fixture\n", encoding="utf-8")
    known_hosts.chmod(0o644)
    external_state = tmp_path / "vm" / "state"
    external_state.mkdir()
    result = deployment.build_bundle(
        output_root=tmp_path / "bundle",
        release_root=release,
        harness_python=python,
        control_state_dir=state,
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        control_sha256=CONTROL,
        remote_host="cluster.example",
        remote_user="watchdog",
        identity_file=identity,
        known_hosts_file=known_hosts,
        external_state_root=external_state,
        public_key_file=key,
        vm_python=python.resolve(),
        vm_release_root=tmp_path / "vm" / "release",
    )
    return result, Path(result["manifest"])


def test_bootstrap_bundle_is_control_independent_and_uses_distinct_key(
    tmp_path: Path,
) -> None:
    pilot_root = tmp_path / "pilot"
    materialization_root = pilot_root / "materialization"
    release = materialization_root / "release-worktree"
    harness_prefix = materialization_root / "harness-environment"
    release_bundle = materialization_root / "release"
    environment_manifest = (
        release_bundle / "harness_environment.schema5-v1.json"
    )
    release, python, _state, production_key = _source_release(
        tmp_path,
        release_path=release,
        prefix_path=harness_prefix,
        environment_manifest_path=environment_manifest,
    )
    for relative in (
        "scripts/schema5_bootstrap_watchdog.py",
        "scripts/render_schema5_recovery_chain_v12.py",
    ):
        path = release / relative
        path.write_text(f"# {relative}\n", encoding="utf-8")
        path.chmod(0o444)
    for directory in sorted(
        (path for path in release.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    release.chmod(0o555)
    source_tree_sha256 = deployment._sealed_source_tree_sha256(release)
    release_identity = {
        "release_worktree": str(release),
        "worktree_sealed_read_only": True,
        "git": {
            "git_commit": COMMIT,
            "git_tag": deployment.RELEASE_TAG,
            "git_tag_object": TAG_OBJECT,
            "source_tree_sha256": source_tree_sha256,
        },
    }
    release_identity_path = _seal(
        release_bundle / "release_identity.schema5-v1.json",
        release_identity,
    )
    release_completion = {
        "complete": True,
        "publication_protocol": "fsync_verify_marker_last",
        "git_commit": COMMIT,
        "source_tree_sha256": source_tree_sha256,
        "artifacts": {
            release_identity_path.name: {
                "sha256": hashlib.sha256(
                    release_identity_path.read_bytes()
                ).hexdigest(),
                "size": release_identity_path.stat().st_size,
            }
        },
    }
    release_completion["release_bundle_id"] = hashlib.sha256(
        deployment._compact_canonical(release_completion)
    ).hexdigest()
    _seal(release_bundle / "RELEASE_COMPLETE.json", release_completion)
    release_bundle.chmod(0o555)
    layout = {
        "pilot_root": str(pilot_root),
        "environment_capture_root": str(
            pilot_root / "environment-capture"
        ),
        "materialization_root": str(materialization_root),
        "release_worktree": str(release),
        "harness_prefix": str(harness_prefix),
        "serving_prefix": str(
            materialization_root / "serving-environment"
        ),
        "conda_package_cache": str(
            materialization_root / "conda-package-cache"
        ),
        "release_bundle": str(release_bundle),
    }
    pilot_marker = pilot_root / "PILOT_COMPLETE.json"
    pilot = {
        "kind": "schema5-materialization-pilot-completion",
        "complete": True,
        "publication_protocol": (
            "intent_first_stage_evidence_pilot_marker_last"
        ),
        "pilot_root": str(pilot_root),
        "expected_tag": deployment.RELEASE_TAG,
        "expected_commit": COMMIT,
        "git_identity": {
            "git_commit": COMMIT,
            "git_tag": deployment.RELEASE_TAG,
            "git_tag_object": TAG_OBJECT,
            "source_tree_sha256": source_tree_sha256,
            "tag_object": TAG_OBJECT,
            "tag_object_type": "tag",
        },
        "layout": layout,
        "stage_ids": {
            "capture_id": "1" * 64,
            "materialization_id": "2" * 64,
            "release_bundle_id": release_completion["release_bundle_id"],
        },
    }
    pilot["pilot_id"] = hashlib.sha256(
        deployment._compact_canonical(pilot)
    ).hexdigest()
    _seal(pilot_marker, pilot)
    drill_fixture = _bootstrap_drill_fixture(tmp_path)
    drill_bundle = json.loads(
        Path(drill_fixture["bundle"]).read_text(encoding="utf-8")
    )
    isolated_contract = drill_bundle["isolated_cancellation_drill"]
    manifest_path = Path(drill_bundle["chain_manifest"])
    receipt_path = Path(drill_bundle["submission_receipt"])
    bootstrap_key = tmp_path / "bootstrap.pub"
    bootstrap_key.write_text(
        "ssh-ed25519 SEpLTE1OTw== bootstrap\n", encoding="ascii"
    )
    identity = tmp_path / "bootstrap-id"
    identity.write_text("private\n", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("cluster key\n", encoding="utf-8")
    known_hosts.chmod(0o600)
    external_state = tmp_path / "external-state"
    external_state.mkdir()
    result = deployment.build_bootstrap_bundle(
        output_root=tmp_path / "bootstrap-bundle",
        release_root=release,
        harness_python=python,
        harness_environment_manifest=environment_manifest,
        materialization_pilot_marker=pilot_marker,
        chain_manifest=manifest_path,
        submission_receipt=receipt_path,
        isolated_drill_chain_manifest=Path(
            isolated_contract["chain_manifest"]
        ),
        isolated_drill_submission_receipt=Path(
            isolated_contract["anchor_submission_receipt"]
        ),
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        remote_host="cluster",
        remote_user="watchdog",
        identity_file=identity,
        known_hosts_file=known_hosts,
        external_state_root=external_state,
        bootstrap_public_key_file=bootstrap_key,
        production_public_key_file=production_key,
        vm_python=python.resolve(),
        vm_release_root=tmp_path / "bootstrap-vm-release",
        vm_config_path=tmp_path / "bootstrap-vm-config.json",
    )

    bundle = json.loads(
        Path(result["manifest"]).read_text(encoding="utf-8")
    )
    assert bundle["protocol"] == deployment.BOOTSTRAP_BUNDLE_PROTOCOL
    assert bundle["separate_bootstrap_key"] is True
    assert bundle["allowed_operations"] == [
        "bootstrap-status",
        "bootstrap-repair",
        "bootstrap-drill-status",
        "bootstrap-drill-repair",
    ]
    assert bundle["forced_command_argv"][-1] == "bootstrap-dispatch"
    assert (
        str(isolated_contract["chain_manifest"])
        in bundle["forced_command_argv"]
    )
    assert (
        str(isolated_contract["anchor_submission_receipt"])
        in bundle["forced_command_argv"]
    )
    assert all(
        "control.json" not in value
        for value in bundle["forced_command_argv"]
    )
    assert bundle["root_release_authority"] == (
        "armed-descendant-rearm-or-release-intent-continuation-only"
    )
    assert bundle["prelaunch_root_release"] is False
    assert bundle["descendant_rearm_authority"] is True
    assert bundle["release_intent_continuation_authority"] is True
    assert bundle["scientific_admission_direct"] is False
    assert bundle["code_records"] == sorted(
        bundle["code_records"], key=lambda item: item["path"]
    )
    assert {
        "src/agents_scaling/serving/external_watchdog.py",
        "scripts/render_schema5_recovery_chain_v12.py",
    } <= {item["path"] for item in bundle["code_records"]}
    assert {
        item["path"] for item in bundle["runtime_inventory"]
    } == {
        "scripts/build_schema5_watchdog_deployment.py",
        "scripts/schema5_bootstrap_watchdog.py",
        "src/agents_scaling/serving/external_watchdog.py",
    }

    bundle_root = Path(result["manifest"]).parent
    installed_runtime = Path(bundle["vm_release_root"])
    shutil.copytree(bundle_root / "release", installed_runtime)
    for directory in sorted(
        (
            path
            for path in installed_runtime.rglob("*")
            if path.is_dir()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    installed_runtime.chmod(0o555)

    installed_config = Path(bundle["vm_config_path"])
    installed_config.write_bytes(
        (bundle_root / "bootstrap-watchdog.json").read_bytes()
    )
    installed_config.chmod(0o444)
    installed_service = tmp_path / deployment.BOOTSTRAP_SERVICE_NAME
    installed_service.write_bytes(
        (bundle_root / deployment.BOOTSTRAP_SERVICE_NAME).read_bytes()
    )
    installed_service.chmod(0o444)
    installed_timer = tmp_path / deployment.BOOTSTRAP_TIMER_NAME
    installed_timer.write_bytes(
        (bundle_root / deployment.BOOTSTRAP_TIMER_NAME).read_bytes()
    )
    installed_timer.chmod(0o444)
    installed_authorized = tmp_path / "authorized_keys.bootstrap"
    installed_authorized.write_bytes(
        (bundle_root / "authorized_keys.bootstrap.line").read_bytes()
        + b"ssh-ed25519 QUJDREVGRw== unrelated-comment\n"
    )
    installed_authorized.chmod(0o444)
    heartbeat = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r6-bootstrap-watchdog-heartbeat-v1"
        ),
        "passed": True,
        "observed_at_timestamp": 100.0,
        "configuration_sha256": hashlib.sha256(
            installed_config.read_bytes()
        ).hexdigest(),
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_id": bundle["chain_id"],
        "chain_manifest_sha256": bundle["chain_manifest_sha256"],
        "anchor_submission_receipt_id": bundle[
            "submission_receipt_id"
        ],
        "anchor_submission_receipt_sha256": bundle[
            "submission_receipt_sha256"
        ],
        "first_observation_id": "1" * 64,
        "second_observation_id": "2" * 64,
        "action": "healthy_noop",
        "recovery_root_release_performed": False,
        "production_control_mutated": False,
        "scientific_jobs_submitted": 0,
        "safety_hold_cleared": False,
    }
    heartbeat["heartbeat_id"] = deployment._self_hash(
        heartbeat, "heartbeat_id"
    )
    heartbeat_path = _seal(
        tmp_path / "BOOTSTRAP_WATCHDOG_HEARTBEAT.json",
        heartbeat,
    )
    deployment_result = deployment.capture_bootstrap_deployment_evidence(
        bundle_manifest=Path(result["manifest"]),
        installed_release_root=installed_runtime,
        vm_python=python.resolve(),
        installed_config=installed_config,
        installed_service=installed_service,
        installed_timer=installed_timer,
        installed_authorized_keys=installed_authorized,
        service_heartbeat=heartbeat_path,
        output=tmp_path / "BOOTSTRAP_DEPLOYMENT_EVIDENCE.json",
        systemctl_runner=_systemctl,
        clock=lambda: 200.0,
    )
    assert deployment_result["passed"] is True
    deployment_evidence = json.loads(
        Path(deployment_result["deployment_evidence"]).read_text(
            encoding="utf-8"
        )
    )
    assert deployment_evidence["vm_python_path"] == str(
        python.resolve()
    )
    assert deployment_evidence["runtime_file_count"] == 3
    assert deployment_evidence["heartbeat_freshness_seconds"] == 100.0

    stale = dict(heartbeat)
    stale["observed_at_timestamp"] = -1_000.0
    stale.pop("heartbeat_id")
    stale["heartbeat_id"] = deployment._self_hash(
        stale, "heartbeat_id"
    )
    stale_path = _seal(tmp_path / "stale-bootstrap-heartbeat.json", stale)
    with pytest.raises(
        deployment.DeploymentError,
        match="service heartbeat is invalid",
    ):
        deployment.capture_bootstrap_deployment_evidence(
            bundle_manifest=Path(result["manifest"]),
            installed_release_root=installed_runtime,
            vm_python=python.resolve(),
            installed_config=installed_config,
            installed_service=installed_service,
            installed_timer=installed_timer,
            installed_authorized_keys=installed_authorized,
            service_heartbeat=stale_path,
            output=tmp_path / "STALE_DEPLOYMENT_EVIDENCE.json",
            systemctl_runner=_systemctl,
            clock=lambda: 200.0,
        )


def test_ssh_key_identity_ignores_comments_and_rejects_unrestricted_alias() -> None:
    first = deployment._public_key_identity(
        b"ssh-ed25519 QUJDREVGRw== first-comment",
        description="first public key",
    )
    second = deployment._public_key_identity(
        b"ssh-ed25519 QUJDREVGRw== second-comment",
        description="second public key",
    )
    assert first[:2] == second[:2]

    expected = (
        b'restrict,command="/sealed/forced command" '
        b"ssh-ed25519 QUJDREVGRw==\n"
    )
    installed = (
        expected
        + b"ssh-ed25519 QUJDREVGRw== unrestricted-alias\n"
    )
    with pytest.raises(
        deployment.DeploymentError,
        match="exactly one restricted instance",
    ):
        deployment._require_single_authorized_key_identity(
            installed,
            expected,
            description="bootstrap authorized_keys",
        )


def _heartbeat(tmp_path: Path, *, observed: float = 100.0) -> Path:
    value = {
        "schema_version": 1,
        "protocol": deployment.HEARTBEAT_PROTOCOL,
        "observed_timestamp": observed,
        "release_id": deployment.RELEASE_ID,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "first_status_sha256": "1" * 64,
        "second_status_sha256": "2" * 64,
        "action": None,
        "reason": "healthy",
        "desired_state": "paused",
        "finalization_state": "idle",
        "action_result": None,
    }
    value["heartbeat_id"] = hashlib.sha256(
        deployment._canonical(value)
    ).hexdigest()
    path = tmp_path / f"heartbeat-{observed}.json"
    path.write_bytes(deployment._canonical(value))
    path.chmod(0o444)
    return path


def _install_bundle(
    tmp_path: Path, manifest_path: Path
) -> tuple[dict, Path, Path, Path, Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {
        record["name"]: manifest_path.parent / Path(record["path"])
        for record in manifest["files"]
    }
    installed = tmp_path / "installed"
    installed.mkdir()
    config = installed / "watchdog.json"
    service = installed / deployment.SERVICE_NAME
    timer = installed / deployment.TIMER_NAME
    authorized = installed / "authorized_keys"
    for source, target in (
        (by_name["watchdog.json"], config),
        (by_name[deployment.SERVICE_NAME], service),
        (by_name[deployment.TIMER_NAME], timer),
        (by_name["authorized_keys.line"], authorized),
    ):
        shutil.copyfile(source, target)
    shutil.copytree(
        manifest_path.parent / Path(manifest["runtime_root"]),
        Path(manifest["vm_release_root"]),
    )
    runtime_root = Path(manifest["vm_release_root"])
    for path in runtime_root.rglob("*"):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o444)
    runtime_root.chmod(0o555)
    return manifest, config, service, timer, authorized, Path(
        manifest["vm_release_root"]
    )


def _systemctl(argv, **_kwargs):
    assert argv[0] == "systemctl"
    if "--property=LoadState" in argv:
        stdout = "loaded\n"
    elif "--property=Result" in argv:
        stdout = "success\n"
    elif "--property=ExecMainStatus" in argv:
        stdout = "0\n"
    else:
        stdout = ""
    return subprocess.CompletedProcess(argv, 0, stdout, "")


def _successful_probe(argv, **_kwargs):
    return subprocess.CompletedProcess(
        argv,
        0,
        "Run one five-minute external schema-5 watchdog transaction.\n",
        "",
    )


def test_bundle_is_offline_sealed_and_forced_command_only(tmp_path):
    result, manifest_path = _bundle(tmp_path)
    assert result["passed"] is True
    manifest, _ = deployment._sealed_json(
        manifest_path, description="test bundle"
    )
    by_name = {
        record["name"]: manifest_path.parent / Path(record["path"])
        for record in manifest["files"]
    }
    line = by_name["authorized_keys.line"].read_text(encoding="ascii")
    runtime = manifest["cluster_harness_runtime"]
    assert runtime["lexical_path"].endswith("/harness/bin/python")
    assert runtime["resolved_path"].endswith("/harness/bin/python3.11")
    assert line.startswith(f'restrict,command="{runtime["resolved_path"]} -I -u ')
    assert "schema5_watchdog_forced_command.py" in line
    assert "schema5_control.py" not in line
    assert f'--harness-python {runtime["lexical_path"]}' in line
    assert f'--resolved-harness-python {runtime["resolved_path"]}' in line
    assert (
        f'--harness-environment-manifest {runtime["manifest_path"]}' in line
    )
    assert (
        f'--harness-environment-sha256 {runtime["manifest_sha256"]}' in line
    )
    assert "\n" not in line.rstrip("\n")
    assert all(path.stat().st_mode & 0o222 == 0 for path in by_name.values())

    repeated = deployment.build_bundle(
        output_root=tmp_path / "bundle",
        release_root=tmp_path / "release",
        harness_python=tmp_path / "harness" / "bin" / "python",
        control_state_dir=tmp_path / "control",
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        control_sha256=CONTROL,
        remote_host="cluster.example",
        remote_user="watchdog",
        identity_file=tmp_path / "vm" / "id_ed25519",
        known_hosts_file=tmp_path / "vm" / "known_hosts",
        external_state_root=tmp_path / "vm" / "state",
        public_key_file=tmp_path / "watchdog.pub",
        vm_python=tmp_path / "harness" / "bin" / "python3.11",
        vm_release_root=tmp_path / "vm" / "release",
    )
    assert repeated["bundle_id"] == result["bundle_id"]
    assert manifest["runtime_file_count"] == 3
    assert {
        row["path"] for row in manifest["runtime_inventory"]
    } == {
        "scripts/build_schema5_watchdog_deployment.py",
        "scripts/schema5_external_watchdog.py",
        "src/agents_scaling/serving/external_watchdog.py",
    }


@pytest.mark.parametrize("mutation", ("escape", "mutable", "unpinned"))
def test_conda_python_alias_rejects_unsafe_or_unpinned_target(
    tmp_path, mutation
):
    _release, python, state, _key = _source_release(tmp_path)
    target = python.resolve()
    if mutation == "escape":
        outside = tmp_path / "outside-python"
        outside.write_text("#!/bin/sh\n", encoding="utf-8")
        outside.chmod(0o555)
        python.parent.chmod(0o755)
        python.unlink()
        python.symlink_to(outside)
        python.parent.chmod(0o555)
    elif mutation == "mutable":
        target.chmod(0o755)
    else:
        target.chmod(0o755)
        target.write_text("#!/bin/sh\n# drift\n", encoding="utf-8")
        target.chmod(0o555)
    with pytest.raises(deployment.DeploymentError):
        deployment._resolve_inventory_pinned_harness_python(
            control_state_dir=state,
            harness_python=python,
            control_sha256=CONTROL,
        )


def test_deployment_and_human_acknowledged_liveness_evidence(tmp_path):
    _, manifest_path = _bundle(tmp_path)
    manifest, config, service, timer, authorized, runtime = _install_bundle(
        tmp_path, manifest_path
    )
    heartbeat = _heartbeat(tmp_path)

    deployment_path = tmp_path / "DEPLOYMENT_EVIDENCE.json"
    deployed = deployment.capture_deployment_evidence(
        bundle_manifest=manifest_path,
        installed_release_root=runtime,
        vm_python=Path(manifest["vm_python"]),
        installed_config=config,
        installed_service=service,
        installed_timer=timer,
        installed_authorized_keys=authorized,
        service_heartbeat=heartbeat,
        output=deployment_path,
        systemctl_runner=_systemctl,
        probe_runner=_successful_probe,
    )
    assert deployed["forced_command_only"] is True
    assert deployed["isolated_runtime_probe_passed"] is True
    assert deployed["systemd_service_result"] == "success"
    assert deployed["successful_service_heartbeat_id"] == json.loads(
        heartbeat.read_text(encoding="utf-8")
    )["heartbeat_id"]
    assert deployment_path.stat().st_mode & 0o222 == 0

    heartbeat_paths = []
    heartbeat_ids = []
    for index, observed in enumerate((100.0, 160.0), start=1):
        heartbeat = {
            "schema_version": 1,
            "protocol": deployment.HEARTBEAT_PROTOCOL,
            "observed_timestamp": observed,
            "release_id": deployment.RELEASE_ID,
            "git_commit": COMMIT,
            "release_tag_object": TAG_OBJECT,
            "control_sha256": CONTROL,
            "first_status_sha256": "1" * 64,
            "second_status_sha256": "2" * 64,
            "action": None,
            "reason": "healthy",
            "desired_state": "paused",
            "finalization_state": "idle",
            "action_result": None,
        }
        heartbeat["heartbeat_id"] = hashlib.sha256(
            deployment._canonical(heartbeat)
        ).hexdigest()
        path = tmp_path / f"heartbeat-{index}.json"
        path.write_bytes(deployment._canonical(heartbeat))
        path.chmod(0o444)
        heartbeat_paths.append(path)
        heartbeat_ids.append(heartbeat["heartbeat_id"])
    acknowledgement_path = tmp_path / "ACK.json"
    acknowledgement = deployment.acknowledge_liveness_email(
        deployment_evidence=deployment_path,
        heartbeat_paths=heartbeat_paths,
        operator="test-operator",
        output=acknowledgement_path,
        confirm_email_received=True,
        now=lambda: 170.0,
    )
    assert acknowledgement["heartbeat_ids"] == heartbeat_ids
    liveness_path = tmp_path / "LIVENESS_EVIDENCE.json"
    liveness = deployment.capture_liveness_evidence(
        deployment_evidence=deployment_path,
        heartbeat_paths=heartbeat_paths,
        acknowledgement=acknowledgement_path,
        output=liveness_path,
    )
    assert liveness["liveness_email_ack"] is True
    assert liveness["scheduler_observations"] == [100.0, 160.0]


def test_descriptor_safe_reads_reject_symlink_ancestry_and_installed_symlink(
    tmp_path,
):
    _, manifest_path = _bundle(tmp_path)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(manifest_path.parent, target_is_directory=True)
    with pytest.raises(deployment.DeploymentError, match="symlink"):
        deployment._validate_bundle(alias / manifest_path.name)

    manifest, _config, service, timer, authorized, runtime = _install_bundle(
        tmp_path, manifest_path
    )
    by_name = {
        record["name"]: manifest_path.parent / Path(record["path"])
        for record in manifest["files"]
    }
    installed = tmp_path / "installed-links"
    installed.mkdir()
    config_link = installed / "watchdog.json"
    config_link.symlink_to(by_name["watchdog.json"])
    with pytest.raises(deployment.DeploymentError, match="symlink"):
        deployment.capture_deployment_evidence(
            bundle_manifest=manifest_path,
            installed_release_root=runtime,
            vm_python=Path(manifest["vm_python"]),
            installed_config=config_link,
            installed_service=service,
            installed_timer=timer,
            installed_authorized_keys=authorized,
            service_heartbeat=_heartbeat(tmp_path),
            output=tmp_path / "evidence.json",
            systemctl_runner=_systemctl,
            probe_runner=_successful_probe,
        )


def test_deployment_rejects_failed_service_or_runtime_probe(tmp_path):
    _, manifest_path = _bundle(tmp_path)
    manifest, config, service, timer, authorized, runtime = _install_bundle(
        tmp_path, manifest_path
    )
    heartbeat = _heartbeat(tmp_path)
    common = {
        "bundle_manifest": manifest_path,
        "installed_release_root": runtime,
        "vm_python": Path(manifest["vm_python"]),
        "installed_config": config,
        "installed_service": service,
        "installed_timer": timer,
        "installed_authorized_keys": authorized,
        "service_heartbeat": heartbeat,
        "output": tmp_path / "evidence.json",
    }
    with pytest.raises(deployment.DeploymentError, match="runtime probe"):
        deployment.capture_deployment_evidence(
            **common,
            systemctl_runner=_systemctl,
            probe_runner=lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, 1, "", "cannot import runtime"
            ),
        )

    def failed_service(argv, **kwargs):
        result = _systemctl(argv, **kwargs)
        if "--property=Result" in argv:
            return subprocess.CompletedProcess(argv, 0, "exit-code\n", "")
        return result

    with pytest.raises(deployment.DeploymentError, match="not completed"):
        deployment.capture_deployment_evidence(
            **common,
            systemctl_runner=failed_service,
            probe_runner=_successful_probe,
        )


def test_marker_publication_recovers_linked_temp_crash_boundary(tmp_path):
    output = tmp_path / "MARKER.json"
    value = {"schema_version": 1, "passed": True}
    payload = deployment._canonical(value)
    interrupted = tmp_path / ".MARKER.json.crashed.publishing"
    interrupted.write_bytes(payload)
    interrupted.chmod(0o444)
    os_link = __import__("os").link
    os_link(interrupted, output)
    assert output.stat().st_nlink == 2

    path, digest = deployment._publish_once(output, value)
    assert path == output
    assert digest == hashlib.sha256(payload).hexdigest()
    assert output.stat().st_nlink == 1
    assert not interrupted.exists()
    assert output.read_bytes() == payload


def test_evidence_publication_recovers_exact_mode_0600_temp_only(tmp_path):
    output = tmp_path / "ATTESTATION.json"
    value = {"schema_version": 1, "passed": True}
    payload = deployment._canonical(value)
    interrupted = tmp_path / ".ATTESTATION.json.crashed.publishing"
    interrupted.write_bytes(payload)
    interrupted.chmod(0o600)
    deployment._publish_once(output, value)
    assert output.read_bytes() == payload
    assert output.stat().st_mode & 0o777 == 0o444
    assert output.stat().st_nlink == 1
    assert not interrupted.exists()

    conflicting_output = tmp_path / "ARMED.json"
    conflicting = tmp_path / ".ARMED.json.crashed.publishing"
    conflicting.write_bytes(deployment._canonical({"passed": False}))
    conflicting.chmod(0o600)
    with pytest.raises(
        deployment.DeploymentError, match="conflicting interrupted"
    ):
        deployment._publish_once(conflicting_output, value)


def test_bootstrap_drill_binds_and_revalidates_exact_scheduler_lineage(
    tmp_path,
):
    fixture = _bootstrap_drill_fixture(tmp_path)
    result = deployment.capture_bootstrap_drill_evidence(
        bundle_manifest=fixture["bundle"],
        deployment_evidence=fixture["deployment"],
        cancelled_observations=fixture["cancelled"],
        repair_result=fixture["repair"],
        recovered_observation=fixture["recovered"],
        recovery_seconds=fixture["recovery_seconds"],
        output=fixture["output"],
    )
    assert result["passed"] is True
    evidence = json.loads(
        Path(result["drill_evidence"]).read_text(encoding="utf-8")
    )
    assert evidence["cancelled_observation_gap_seconds"] == 60.0
    assert (
        evidence["recovery_namespace_cancellation_recovery_seconds"] == 120.0
    )
    assert evidence["repair_generation"] == 1
    assert evidence["held_root_names"] == ["stage-02", "stage-03"]
    assert evidence["held_root_job_ids"] == ["4002", "4003"]
    assert evidence["isolated_cancellation_drill"] is True
    assert evidence["canonical_job_id_overlap"] == 0
    assert evidence["canonical_comment_overlap"] == 0
    assert evidence["canonical_control_paths_absent"] is True
    assert evidence["squeue_complete"] is True
    assert evidence["sacct_complete"] is True
    assert evidence["duplicate_jobs"] == 0
    assert evidence["duplicate_submission_intents"] == 0
    assert evidence["root_remained_held"] is True
    assert evidence["scientific_jobs_started"] == 0
    assert evidence["namespace_scan_complete"] is True
    bundle = json.loads(Path(fixture["bundle"]).read_text(encoding="utf-8"))
    deployed = json.loads(
        Path(fixture["deployment"]).read_text(encoding="utf-8")
    )
    inspected = deployment._validate_bootstrap_drill_preimages(
        drill=evidence,
        bundle=bundle,
        deployment=deployed,
    )
    assert inspected["repair_result_id"] == evidence["repair_result_id"]
    publisher._revalidate_bootstrap_drill_preimages(
        drill=evidence,
        bundle=bundle,
        deployment=deployed,
    )


def test_ordered_repair_frontier_excludes_transitive_selected_ancestor() -> None:
    manifest = {
        "jobs": [
            {"name": "root", "dependencies": []},
            {"name": "completed_bridge", "dependencies": ["root"]},
            {"name": "descendant", "dependencies": ["completed_bridge"]},
            {"name": "independent", "dependencies": []},
        ]
    }

    assert deployment._ordered_repair_frontier(
        manifest,
        ["root", "descendant", "independent"],
    ) == ["root", "independent"]


def test_bootstrap_drill_rejects_fabricated_or_unrelated_observation(
    tmp_path,
):
    fixture = _bootstrap_drill_fixture(tmp_path)
    original = json.loads(
        Path(fixture["cancelled"][1]).read_text(encoding="utf-8")
    )
    original["chain_id"] = "f" * 64
    original["observation_id"] = deployment._self_hash(
        original, "observation_id"
    )
    unrelated = _seal(
        Path(fixture["cancelled"][1]).with_name(
            "unrelated-cancelled-observation.json"
        ),
        original,
    )
    with pytest.raises(
        deployment.DeploymentError, match="observation identity drifted"
    ):
        deployment.capture_bootstrap_drill_evidence(
            bundle_manifest=fixture["bundle"],
            deployment_evidence=fixture["deployment"],
            cancelled_observations=[
                fixture["cancelled"][0],
                unrelated,
            ],
            repair_result=fixture["repair"],
            recovered_observation=fixture["recovered"],
            recovery_seconds=fixture["recovery_seconds"],
            output=fixture["output"],
        )


def test_bootstrap_drill_rejects_observation_not_bound_to_provenance(
    tmp_path,
):
    fixture = _bootstrap_drill_fixture(tmp_path)
    fabricated = json.loads(
        Path(fixture["cancelled"][1]).read_text(encoding="utf-8")
    )
    fabricated["jobs"][0]["submit_line_sha256"] = "f" * 64
    fabricated["observation_id"] = deployment._self_hash(
        fabricated, "observation_id"
    )
    fabricated_path = _seal(
        Path(fixture["cancelled"][1]).with_name(
            "fabricated-provenance-observation.json"
        ),
        fabricated,
    )

    with pytest.raises(
        deployment.DeploymentError,
        match="scheduler job evidence drifted",
    ):
        deployment.capture_bootstrap_drill_evidence(
            bundle_manifest=fixture["bundle"],
            deployment_evidence=fixture["deployment"],
            cancelled_observations=[
                fixture["cancelled"][0],
                fabricated_path,
            ],
            repair_result=fixture["repair"],
            recovered_observation=fixture["recovered"],
            recovery_seconds=fixture["recovery_seconds"],
            output=fixture["output"],
        )


def test_bootstrap_drill_rejects_unrelated_repair_and_rebound_evidence(
    tmp_path,
):
    fixture = _bootstrap_drill_fixture(tmp_path)
    repair = json.loads(
        Path(fixture["repair"]).read_text(encoding="utf-8")
    )
    repair["parent_submission_receipt_id"] = "f" * 64
    unrelated_repair = _seal(
        Path(fixture["repair"]).with_name("unrelated-repair.json"),
        repair,
    )
    with pytest.raises(
        deployment.DeploymentError, match="receipt lineage is invalid"
    ):
        deployment.capture_bootstrap_drill_evidence(
            bundle_manifest=fixture["bundle"],
            deployment_evidence=fixture["deployment"],
            cancelled_observations=fixture["cancelled"],
            repair_result=unrelated_repair,
            recovered_observation=fixture["recovered"],
            recovery_seconds=fixture["recovery_seconds"],
            output=fixture["output"],
        )

    deployment.capture_bootstrap_drill_evidence(
        bundle_manifest=fixture["bundle"],
        deployment_evidence=fixture["deployment"],
        cancelled_observations=fixture["cancelled"],
        repair_result=fixture["repair"],
        recovered_observation=fixture["recovered"],
        recovery_seconds=fixture["recovery_seconds"],
        output=fixture["output"],
    )
    evidence = json.loads(
        Path(fixture["output"]).read_text(encoding="utf-8")
    )
    evidence["recovered_observation"] = dict(
        evidence["cancelled_observations"][0]
    )
    evidence["evidence_id"] = deployment._self_hash(
        evidence, "evidence_id"
    )
    bundle = json.loads(Path(fixture["bundle"]).read_text(encoding="utf-8"))
    deployed = json.loads(
        Path(fixture["deployment"]).read_text(encoding="utf-8")
    )
    with pytest.raises(deployment.DeploymentError):
        deployment._validate_bootstrap_drill_preimages(
            drill=evidence,
            bundle=bundle,
            deployment=deployed,
        )
    with pytest.raises(
        publisher.WatchdogEvidenceError,
        match="sealed preimages are invalid",
    ):
        publisher._revalidate_bootstrap_drill_preimages(
            drill=evidence,
            bundle=bundle,
            deployment=deployed,
        )
