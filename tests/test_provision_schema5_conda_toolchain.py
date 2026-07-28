from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from scripts import provision_schema5_conda_toolchain as provision


def _contract(installer: Path, *, version: str = "25.11.0-test"):
    return provision.InstallerContract(
        filename=installer.name,
        release="Miniforge3-test",
        sha256=hashlib.sha256(installer.read_bytes()).hexdigest(),
        conda_version=version,
    )


def _fake_installer(
    tmp_path: Path,
    *,
    unsafe_symlink: bool = False,
    mutate_on_probe: bool = False,
    fail_once: bool = False,
    unresolved_cache_symlink: bool = False,
    report_devnull_config: bool = False,
) -> tuple[Path, Path]:
    installer = tmp_path / "Miniforge3-test.sh"
    execution_marker = tmp_path / "INSTALLER_EXECUTED"
    failure_marker = tmp_path / "INSTALLER_FAILED_ONCE"
    unsafe = (
        'ln -s /etc/passwd "$prefix/lib/unsafe-link"\n'
        if unsafe_symlink
        else 'ln -s ../lib "$prefix/share/lib-link"\n'
    )
    mutation = (
        'touch "$prefix/PROBE_MUTATION"\n' if mutate_on_probe else ""
    )
    cache_link = (
        'ln -s missing-package-target "$prefix/pkgs/cache/deferred-link"\n'
        if unresolved_cache_symlink
        else ""
    )
    reported_config = "/dev/null" if report_devnull_config else "$prefix/.condarc"
    fail = (
        f"""
if [ ! -f {failure_marker!s} ]; then
  : > {failure_marker!s}
  mkdir -p "$prefix"
  printf 'partial\\n' > "$prefix/PARTIAL_INSTALL"
  exit 17
fi
"""
        if fail_once
        else ""
    )
    installer.write_text(
        f"""#!/bin/bash
set -eu
: > {execution_marker!s}
prefix=
while [ "$#" -gt 0 ]; do
  case "$1" in
    -b) shift ;;
    -p) prefix="$2"; shift 2 ;;
    *) exit 93 ;;
  esac
done
test -n "$prefix"
{fail}
mkdir -p "$prefix/bin" "$prefix/lib/python3.12/site-packages/conda" \\
  "$prefix/share" "$prefix/pkgs/cache" "$prefix/conda-meta"
cp /bin/sh "$prefix/bin/python"
printf 'fixture\\n' > "$prefix/lib/python3.12/site-packages/conda/__init__.py"
printf 'offline: true\\n' > "$prefix/.condarc"
printf '{{"name":"conda","version":"25.11.0-test"}}\\n' \\
  > "$prefix/conda-meta/conda-test.json"
cat > "$prefix/bin/conda" <<EOF
#!$prefix/bin/python
set -eu
prefix='$prefix'
{mutation}if [ "\\$1" = "--version" ]; then
  printf 'conda 25.11.0-test\\\\n'
  exit 0
fi
if [ "\\$1" = "info" ]; then
  cat <<JSON
{{"conda_version":"25.11.0-test","python_version":"3.12.12.test.0","root_prefix":"$prefix","conda_prefix":"$prefix","root_writable":false,"offline":true,"conda_location":"$prefix/lib/python3.12/site-packages/conda","config_files":["{reported_config}"]}}
JSON
  exit 0
fi
exit 94
EOF
chmod 755 "$prefix/bin/python" "$prefix/bin/conda"
{unsafe}
{cache_link}
""",
        encoding="utf-8",
    )
    installer.chmod(0o755)
    return installer, execution_marker


def _provision(
    tmp_path: Path,
    installer: Path,
    *,
    apply: bool,
    contract: provision.InstallerContract | None = None,
):
    namespace = tmp_path / "release-namespace"
    namespace.mkdir(exist_ok=True)
    return provision.provision_conda_toolchain(
        installer=installer,
        namespace_root=namespace,
        forbidden_prefixes=(),
        contract=contract or _contract(installer),
        apply=apply,
        installer_timeout_seconds=30,
        probe_timeout_seconds=10,
    )


def _target(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "release-namespace"
        / provision.TOOLCHAIN_DIRECTORY_NAME
    )


def test_dry_run_binds_installer_without_execution_or_writes(tmp_path):
    installer, executed = _fake_installer(tmp_path)
    before = installer.stat()

    report = _provision(tmp_path, installer, apply=False)

    after = installer.stat()
    assert report["action"] == "would_provision"
    assert report["installer"]["sha256"] == hashlib.sha256(
        installer.read_bytes()
    ).hexdigest()
    assert report["would_invoke_existing_conda"] is False
    assert report["would_query_live_prefixes"] is False
    assert report["would_publish_marker_last"] is True
    assert report["portable_shebang"][
        "absolute_base_prefix_interpreter_required"
    ] is True
    assert (
        report["portable_shebang"]["shebang_bytes"]
        <= provision.MAX_PORTABLE_SHEBANG_BYTES
    )
    assert not executed.exists()
    assert not _target(tmp_path).exists()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def test_provision_seals_and_replays_complete_toolchain(tmp_path):
    installer, executed = _fake_installer(tmp_path)
    contract = _contract(installer)

    report = _provision(
        tmp_path, installer, apply=True, contract=contract
    )
    target = _target(tmp_path)
    marker = target / provision.MARKER_NAME

    assert report["action"] == "provisioned"
    assert executed.is_file()
    assert marker.is_file()
    assert report["installer"]["sha256"] == contract.sha256
    assert report["runtime_identity"]["validated_twice"] is True
    assert (
        report["complete_prefix_inventory"]["before"]
        == report["complete_prefix_inventory"]["after"]
    )
    assert report["read_only_operation"]["probe_count"] == 2
    assert report["read_only_operation"]["target_prefix_writable"] is False
    assert report["symlink_audit"]["destination_external_symlink_count"] == 0
    assert not stat.S_IMODE(target.stat().st_mode) & 0o222
    assert not stat.S_IMODE(marker.stat().st_mode) & 0o222
    assert os.stat(installer).st_ino != os.stat(
        target / "inputs" / installer.name
    ).st_ino

    verified = provision.verify_conda_toolchain(
        target, contract=contract, exercise=True, probe_timeout_seconds=10
    )
    assert verified["marker_id"] == report["marker_id"]
    assert (
        provision.verified_conda_executable(
            target, contract=contract, exercise=False
        )
        == target / "base/bin/conda"
    )
    binding = provision.verified_conda_toolchain_binding(
        target, contract=contract, exercise=False
    )
    assert set(binding) == {
        "schema_version",
        "protocol",
        "release_tag",
        "chain_namespace",
        "toolchain_root",
        "base_prefix",
        "portable_shebang",
        "completion_marker",
        "marker_id",
        "installer_contract",
        "intent_id",
        "conda_executable",
        "runtime_identity_sha256",
        "complete_prefix_inventory_sha256",
        "read_only_probes",
        "binding_id",
    }
    assert binding["toolchain_root"] == str(target)
    assert binding["portable_shebang"] == report["portable_shebang"]
    assert binding["completion_marker"]["path"] == str(marker)
    assert binding["completion_marker"]["sha256"] == hashlib.sha256(
        marker.read_bytes()
    ).hexdigest()
    assert binding["completion_marker"]["size"] == marker.stat().st_size
    assert binding["marker_id"] == report["marker_id"]
    assert binding["installer_contract"] == contract.as_dict()
    assert binding["intent_id"] == report["intent"]["intent_id"]
    assert binding["conda_executable"]["path"] == str(
        target / "base/bin/conda"
    )
    assert (
        binding["complete_prefix_inventory_sha256"]
        == report["complete_prefix_inventory"]["before"]["inventory_sha256"]
    )
    replay_binding = provision.verified_conda_toolchain_binding(
        target, contract=contract, exercise=False
    )
    assert replay_binding == binding

    replay = _provision(
        tmp_path, installer, apply=True, contract=contract
    )
    assert replay["action"] == "already_complete"
    assert replay["marker_id"] == report["marker_id"]


def test_probe_accepts_only_the_explicit_devnull_external_config(tmp_path):
    installer, _ = _fake_installer(tmp_path, report_devnull_config=True)
    report = _provision(tmp_path, installer, apply=True)

    assert report["read_only_operation"]["probe_count"] == 2
    assert (
        report["read_only_operation"]["summary"][
            "external_config_file_count"
        ]
        == 0
    )


def test_wrong_installer_digest_fails_before_output_or_execution(tmp_path):
    installer, executed = _fake_installer(tmp_path)
    contract = _contract(installer)
    contract = provision.InstallerContract(
        filename=contract.filename,
        release=contract.release,
        sha256="0" * 64,
        conda_version=contract.conda_version,
    )

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="does not match the immutable contract",
    ):
        _provision(
            tmp_path, installer, apply=True, contract=contract
        )

    assert not executed.exists()
    assert not _target(tmp_path).exists()


def test_overlong_interpreter_path_fails_before_output_or_execution(tmp_path):
    installer, executed = _fake_installer(tmp_path)
    namespace = tmp_path / ("long-prefix-" + "x" * 100)
    namespace.mkdir()

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="portable shebang limit",
    ):
        provision.provision_conda_toolchain(
            installer=installer,
            namespace_root=namespace,
            forbidden_prefixes=(),
            contract=_contract(installer),
            apply=True,
        )

    assert not executed.exists()
    assert not (namespace / provision.TOOLCHAIN_DIRECTORY_NAME).exists()


def test_unsafe_installer_symlink_is_rejected_before_marker(tmp_path):
    installer, executed = _fake_installer(tmp_path, unsafe_symlink=True)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="symlink|external",
    ):
        _provision(tmp_path, installer, apply=True)

    assert executed.is_file()
    assert not (_target(tmp_path) / provision.MARKER_NAME).exists()


def test_unresolved_declared_package_cache_link_is_inventoried_not_executed(
    tmp_path,
):
    installer, _ = _fake_installer(
        tmp_path, unresolved_cache_symlink=True
    )

    report = _provision(tmp_path, installer, apply=True)

    complete = report["complete_prefix_inventory"]["before"]
    assert complete["unresolved_cache_symlink_count"] == 1
    assert complete["all_runtime_symlinks_resolve_inside_prefix"] is True
    assert report["symlink_audit"]["unresolvable_symlink_count"] == 0


def test_read_only_probe_detects_attempted_target_mutation(tmp_path):
    installer, _ = _fake_installer(tmp_path, mutate_on_probe=True)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="version probe failed",
    ):
        _provision(tmp_path, installer, apply=True)

    assert not (_target(tmp_path) / provision.MARKER_NAME).exists()
    assert not (_target(tmp_path) / "base/PROBE_MUTATION").exists()


def test_failed_install_is_preserved_and_retry_uses_fresh_prefix(tmp_path):
    installer, _ = _fake_installer(tmp_path, fail_once=True)
    contract = _contract(installer)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="installer failed rc=17",
    ):
        _provision(
            tmp_path, installer, apply=True, contract=contract
        )
    target = _target(tmp_path)
    assert (target / "base/PARTIAL_INSTALL").is_file()
    assert not (target / provision.MARKER_NAME).exists()

    report = _provision(
        tmp_path, installer, apply=True, contract=contract
    )

    assert report["action"] == "provisioned"
    quarantine = (
        tmp_path
        / "release-namespace"
        / provision.TRANSACTION_DIRECTORY_NAME
        / "quarantine/g0001"
    )
    assert (quarantine / "base/PARTIAL_INSTALL").is_file()
    receipt = json.loads(
        (quarantine / "QUARANTINE_COMPLETE.json").read_text(encoding="utf-8")
    )
    assert receipt["moved"] == ["base"]
    assert report["installation"]["quarantined_incomplete_predecessor"][
        "generation"
    ] == "g0001"


def test_output_scope_cannot_overlap_forbidden_prefix(tmp_path):
    installer, executed = _fake_installer(tmp_path)
    namespace = tmp_path / "blocked"
    namespace.mkdir()

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="overlaps a live or shared forbidden prefix",
    ):
        provision.provision_conda_toolchain(
            installer=installer,
            namespace_root=namespace,
            forbidden_prefixes=(namespace,),
            contract=_contract(installer),
            apply=True,
        )

    assert not executed.exists()


def test_production_binding_rejects_known_live_or_shared_prefix_overlap(
    tmp_path, monkeypatch
):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    monkeypatch.setattr(
        provision,
        "DEFAULT_FORBIDDEN_PREFIXES",
        (target.parent,),
    )

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="known live or shared prefix",
    ):
        provision.verified_conda_toolchain_binding(
            target,
            contract=contract,
            exercise=False,
        )


def test_verify_rejects_writable_or_unexpected_runtime_mutation(tmp_path):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    payload = target / "base/lib/python3.12/site-packages/conda/__init__.py"
    payload.chmod(0o600)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="writable entry",
    ):
        provision.verify_conda_toolchain(
            target, contract=contract, exercise=False
        )


def test_verify_rejects_marker_tampering(tmp_path):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    marker = target / provision.MARKER_NAME
    marker.chmod(0o600)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["sealed_read_only"] = False
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker.chmod(0o400)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="completion marker contract",
    ):
        provision.verify_conda_toolchain(
            target, contract=contract, exercise=False
        )


def test_completed_apply_repairs_only_interrupted_root_seal(tmp_path):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    first = _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    target.chmod(0o755)

    replay = _provision(
        tmp_path, installer, apply=True, contract=contract
    )

    assert replay["action"] == "already_complete"
    assert replay["marker_id"] == first["marker_id"]
    assert not stat.S_IMODE(target.stat().st_mode) & 0o222


def test_verifier_is_independent_of_cached_installer_after_completion(tmp_path):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    installer.rename(tmp_path / "cached-installer-moved-away")

    report = provision.verify_conda_toolchain(
        target, contract=contract, exercise=True, probe_timeout_seconds=10
    )

    assert report["installer"]["sha256"] == contract.sha256


def test_complete_inventory_rejects_inode_shared_outside_prefix(tmp_path):
    prefix = tmp_path / "base"
    prefix.mkdir()
    external = tmp_path / "external"
    external.write_bytes(b"shared\n")
    os.link(external, prefix / "shared")

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="inodes shared outside",
    ):
        provision._complete_prefix_inventory(prefix)


def test_runtime_symlink_audit_rejects_external_hop_that_reenters(tmp_path):
    prefix = tmp_path / "base"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    interpreter = prefix / "bin/python"
    interpreter.write_bytes(Path("/bin/sh").resolve().read_bytes())
    interpreter.chmod(0o755)
    conda = prefix / "bin/conda"
    conda.write_text(
        f"#!{interpreter}\nprintf 'fixture\\n'\n", encoding="utf-8"
    )
    conda.chmod(0o755)
    target = prefix / "lib/target"
    target.write_text("runtime\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "link-back").symlink_to(target)
    (prefix / "lib/bounce").symlink_to(outside / "link-back")

    # A final-endpoint-only check sees ``target`` inside the prefix. The r9 audit
    # must still reject its dependency on the external ``outside/link-back`` hop.
    assert (prefix / "lib/bounce").resolve(strict=True) == target
    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="external absolute path",
    ):
        provision._runtime_symlink_audit(prefix)


def test_verifier_rejects_unexpected_sealed_root_entry(tmp_path):
    installer, _ = _fake_installer(tmp_path)
    contract = _contract(installer)
    _provision(tmp_path, installer, apply=True, contract=contract)
    target = _target(tmp_path)
    target.chmod(0o755)
    unexpected = target / "UNEXPECTED"
    unexpected.write_text("not in marker\n", encoding="utf-8")
    unexpected.chmod(0o400)
    target.chmod(0o555)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="unexpected or missing root entries",
    ):
        provision.verify_conda_toolchain(
            target, contract=contract, exercise=False
        )


def test_installer_source_must_be_outside_release_namespace(tmp_path):
    namespace = tmp_path / "release-namespace"
    namespace.mkdir()
    installer, executed = _fake_installer(namespace)

    with pytest.raises(
        provision.CondaToolchainProvisionError,
        match="installer must be outside",
    ):
        provision.provision_conda_toolchain(
            installer=installer,
            namespace_root=namespace,
            contract=_contract(installer),
            apply=True,
        )

    assert not executed.exists()
