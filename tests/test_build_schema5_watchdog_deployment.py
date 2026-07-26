from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import build_schema5_watchdog_deployment as deployment


COMMIT = "a" * 40
TAG_OBJECT = "b" * 40
CONTROL = "c" * 64


def _source_release(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    release = tmp_path / "release"
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
    prefix = tmp_path / "harness"
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
    environment_manifest = release / "harness_environment.schema5-v1.json"
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
