from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import publish_schema5_durable_git_release as publisher


def test_git_environment_rejects_hostile_repository_and_scheduler_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PATH": "/tmp/attacker-bin",
        "BASH_ENV": "/tmp/attacker-env",
        "LD_PRELOAD": "/tmp/attacker.so",
        "GIT_DIR": "/tmp/attacker-git",
        "GIT_OBJECT_DIRECTORY": "/tmp/attacker-objects",
        "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/attacker-hooks'",
        "GIT_REPLACE_REF_BASE": "refs/attacker",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/tmp/attacker-hooks",
        "SBATCH_PARTITION": "attacker",
        "SQUEUE_FORMAT": "attacker",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    environment = publisher._sanitized_process_environment()
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not (set(hostile) - {"PATH"}).intersection(environment)


def _git(cwd: Path, *argv: str) -> str:
    result = subprocess.run(
        ["git", *argv],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _repositories(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare")
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Schema Five Test")
    _git(repository, "config", "user.email", "schema5@example.invalid")
    (repository / "release.txt").write_text("schema five\n", encoding="utf-8")
    _git(repository, "add", "release.txt")
    _git(repository, "commit", "-m", "release")
    _git(
        repository,
        "tag",
        "-a",
        publisher.RELEASE_TAG,
        "-m",
        "schema-5 v1.2-r6",
    )
    _git(repository, "remote", "add", "durable", str(remote))
    _git(
        repository,
        "push",
        "durable",
        f"HEAD:{publisher.DURABLE_COMMIT_REF}",
    )
    _git(repository, "push", "durable", f"refs/tags/{publisher.RELEASE_TAG}")
    return repository, remote


def test_durable_release_is_dry_by_default_and_marker_last(tmp_path):
    repository, _remote = _repositories(tmp_path)
    recovery = tmp_path / "recovery"
    dry = publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
    )
    assert dry["status"] == "dry_run"
    assert not recovery.exists()

    complete = publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    assert complete["status"] == "complete"
    marker = recovery / publisher.MARKER_NAME
    binding = publisher.marker_binding(marker)
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert binding["marker_id"] == value["marker_id"]
    assert value["remote_query_read_only"] is True
    assert value["remote_tag_object"] == value["release_tag_object"]
    assert value["remote_peeled_commit"] == value["release_git_commit"]
    assert marker.stat().st_mode & 0o222 == 0
    assert Path(value["bundle_path"]).stat().st_mode & 0o222 == 0
    assert Path(value["checksum_path"]).stat().st_mode & 0o222 == 0

    repeated = publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    assert repeated["status"] == "already_complete"
    assert repeated["marker_id"] == complete["marker_id"]


def test_durable_release_rejects_dirty_or_unpushed_identity(tmp_path):
    repository, _remote = _repositories(tmp_path)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(publisher.DurableGitReleaseError, match="clean"):
        publisher.publish(
            repository=repository,
            recovery_root=tmp_path / "recovery",
            remote="durable",
            remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        )
    (repository / "untracked.txt").unlink()
    _git(repository, "tag", "-d", publisher.RELEASE_TAG)
    _git(
        repository,
        "tag",
        "-a",
        publisher.RELEASE_TAG,
        "-m",
        "different tag object",
    )
    with pytest.raises(publisher.DurableGitReleaseError, match="remote does not"):
        publisher.publish(
            repository=repository,
            recovery_root=tmp_path / "recovery",
            remote="durable",
            remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        )


def test_durable_release_rejects_git_replacement_refs(tmp_path):
    repository, _remote = _repositories(tmp_path)
    trusted_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "release.txt").write_text("substituted\n", encoding="utf-8")
    _git(repository, "add", "release.txt")
    _git(repository, "commit", "-m", "replacement")
    replacement_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "reset", "--hard", trusted_commit)
    _git(repository, "replace", trusted_commit, replacement_commit)
    assert _git(repository, "show", f"{trusted_commit}:release.txt") == "substituted"

    with pytest.raises(
        publisher.DurableGitReleaseError,
        match="forbidden Git replacement refs",
    ):
        publisher.publish(
            repository=repository,
            recovery_root=tmp_path / "recovery",
            remote="durable",
            remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        )
    assert (
        subprocess.run(
            ["/usr/bin/git", "show", f"{trusted_commit}:release.txt"],
            cwd=repository,
            env={
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "LC_ALL": "C",
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == "schema five\n"
    )


def test_marker_binding_rejects_bundle_tamper(tmp_path):
    repository, _remote = _repositories(tmp_path)
    recovery = tmp_path / "recovery"
    publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    marker = recovery / publisher.MARKER_NAME
    value = json.loads(marker.read_text(encoding="utf-8"))
    bundle = Path(value["bundle_path"])
    bundle.chmod(0o644)
    bundle.write_bytes(bundle.read_bytes() + b"tamper")
    bundle.chmod(0o444)
    with pytest.raises(publisher.DurableGitReleaseError, match="drifted"):
        publisher.marker_binding(marker)


def test_publication_recovers_bundle_and_checksum_before_marker_boundary(
    tmp_path,
):
    repository, _remote = _repositories(tmp_path)
    recovery = tmp_path / "recovery"
    publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    marker = recovery / publisher.MARKER_NAME
    original = json.loads(marker.read_text(encoding="utf-8"))
    marker.unlink()

    recovered = publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    assert recovered["status"] == "complete"
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert value["bundle_sha256"] == original["bundle_sha256"]
    assert publisher.marker_binding(marker)["marker_id"] == value["marker_id"]


def test_publication_never_launders_writable_pre_marker_bundle(tmp_path):
    repository, _remote = _repositories(tmp_path)
    recovery = tmp_path / "recovery"
    publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    marker = recovery / publisher.MARKER_NAME
    value = json.loads(marker.read_text(encoding="utf-8"))
    bundle = Path(value["bundle_path"])
    marker.unlink()
    bundle.chmod(0o644)

    with pytest.raises(
        publisher.DurableGitReleaseError,
        match="mutable, linked, or changed",
    ):
        publisher.publish(
            repository=repository,
            recovery_root=recovery,
            remote="durable",
            remote_commit_ref=publisher.DURABLE_COMMIT_REF,
            apply=True,
        )

    assert bundle.stat().st_mode & 0o222
    assert not marker.exists()


def test_completed_publication_requires_same_remote_ref_identity(tmp_path):
    repository, _remote = _repositories(tmp_path)
    recovery = tmp_path / "recovery"
    publisher.publish(
        repository=repository,
        recovery_root=recovery,
        remote="durable",
        remote_commit_ref=publisher.DURABLE_COMMIT_REF,
        apply=True,
    )
    _git(repository, "push", "durable", "HEAD:refs/heads/alternate-release")

    with pytest.raises(
        publisher.DurableGitReleaseError,
        match="exact refs/heads/schema5-v1.2-r6",
    ):
        publisher.publish(
            repository=repository,
            recovery_root=recovery,
            remote="durable",
            remote_commit_ref="refs/heads/alternate-release",
            apply=True,
        )
