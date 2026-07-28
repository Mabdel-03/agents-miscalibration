from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import seal_schema5_r7_prelaunch_failure as seal


def _git(*arguments: object) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", *map(str, arguments)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _proof_fixture(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repository"
    bundle = tmp_path / "release.bundle"
    _git("init", repository)
    _git("-C", repository, "config", "user.name", "Schema Five Test")
    _git("-C", repository, "config", "user.email", "schema5@example.invalid")
    (repository / "tracked.txt").write_text("immutable\n", encoding="utf-8")
    _git("-C", repository, "add", "tracked.txt")
    _git("-C", repository, "commit", "-m", "fixture")
    commit = _git("-C", repository, "rev-parse", "HEAD")
    _git("-C", repository, "tag", "-a", "proof-tag", "-m", "proof")
    tag_object = _git("-C", repository, "rev-parse", "refs/tags/proof-tag")
    _git("-C", repository, "bundle", "create", bundle, "--all")
    monkeypatch.setattr(seal, "R7_COMMIT", commit)
    monkeypatch.setattr(seal, "R7_TAG", "proof-tag")
    monkeypatch.setattr(seal, "R7_TAG_OBJECT", tag_object)
    release = {"durable_bundle": seal._file_ref(bundle, description="test bundle")}
    return release


def test_index_refresh_proof_is_same_bytes_different_inode_and_sealed(
    tmp_path, monkeypatch
):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    release = _proof_fixture(tmp_path, monkeypatch)
    contract = seal._proof_contract(evidence, release)

    proof = seal._execute_proof(evidence, {"proof": contract})

    assert proof["index_before"]["inode"] != proof["index_after"]["inode"]
    assert proof["index_before"]["sha256"] == proof["index_after"]["sha256"]
    assert proof["index_before"]["size"] == proof["index_after"]["size"]
    assert (
        proof["inventory_before_excluding_index"]
        == proof["inventory_after_excluding_index"]
    )
    assert proof["corrected_query_preserved_index"] is True
    assert not (
        stat.S_IMODE(
            (evidence / seal.REPRODUCTION_CHECKOUT_NAME).stat().st_mode
        )
        & 0o222
    )
    seal._verify_proof(evidence, {"proof": contract}, proof)


def test_r7_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.EVIDENCE_RELATIVE_ROOT
    intent = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "sealed_r7_toolchain": {},
        "proof": {"expected_results": []},
        "scientific_state": {},
        "scheduler": {"scheduler_user": "researcher"},
    }
    intent["intent_id"] = seal._identity(intent, "intent_id")
    monkeypatch.setattr(
        seal,
        "_intent_payload",
        lambda *_args: json.loads(json.dumps(intent)),
    )

    dry = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=False,
    )
    assert dry["action"] == "would_seal"
    assert not evidence.exists()

    monkeypatch.setattr(
        seal,
        "_execute_proof",
        lambda *_args: {"sealed_test_proof": True},
    )
    monkeypatch.setattr(
        seal,
        "verify_failure_seal",
        lambda *_args, **_kwargs: {"passed": True},
    )
    applied = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    assert applied["action"] == "sealed"
    marker = evidence / seal.MARKER_NAME
    assert marker.is_file()
    assert not stat.S_IMODE(marker.stat().st_mode) & 0o222
    assert not stat.S_IMODE(evidence.stat().st_mode) & 0o222


def test_r7_and_r8_git_environments_differ_only_by_optional_locks():
    r7 = seal._git_environment(optional_locks=True)
    r8 = seal._git_environment(optional_locks=False)
    assert "GIT_OPTIONAL_LOCKS" not in r7
    assert r8 == {**r7, "GIT_OPTIONAL_LOCKS": "0"}


def test_r7_probe_tree_binds_both_absent_envelopes_and_must_remain_empty(
    tmp_path, monkeypatch
):
    probe_root = tmp_path / "probe"
    probe_root.mkdir(mode=0o700)
    monkeypatch.setattr(seal, "_r7_probe_root", lambda _user: probe_root)

    report = seal._empty_r7_probe_tree("researcher")
    assert report["entries"] == []
    absent = seal._permanently_absent_paths(
        tmp_path / "results/recovery/schema5-v1", "researcher"
    )
    assert [
        str(probe_root / name) for name in seal.R7_PROBE_ENVELOPE_NAMES
    ] == absent[-2:]

    (probe_root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(seal.R7FailureSealError, match="is not empty"):
        seal._empty_r7_probe_tree("researcher")
