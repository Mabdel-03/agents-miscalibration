from __future__ import annotations

import json
from pathlib import Path
import stat
import types

import pytest

from scripts import seal_schema5_r9_prelaunch_failure as seal


def _diagnostic_contract(tmp_path: Path) -> dict:
    base = tmp_path / "toolchain/base"
    destination = tmp_path / "reproduction/offline-clone-destination"
    return {
        "argv": [
            str(base / "bin/conda"),
            "create",
            "--yes",
            "--offline",
            "--clone",
            str(base),
            "--prefix",
            str(destination),
        ],
        "destination": str(destination),
    }


def _diagnostic_output(contract: dict) -> tuple[str, str]:
    stdout = (
        f"Source:      {contract['argv'][5]}\n"
        f"Destination: {contract['destination']}\n"
        "Packages: 89\n"
        "Files: 3\n\n"
        "Downloading and Extracting Packages: ...working..."
        "\rpython-3.12 | | 0% \x1b[A done\n"
    )
    stderr = (
        "\nOfflineError: EnforceUnusedAdapter called with url "
        "https://conda.anaconda.org/conda-forge/linux-64/python.conda.\n"
        "This command is using a remote connection in offline mode.\n"
        "OfflineError: EnforceUnusedAdapter called with url "
        "https://conda.anaconda.org/conda-forge/noarch/pip.conda.\n"
        "This command is using a remote connection in offline mode.\n"
    )
    return stdout, stderr


def test_r9_equivalent_diagnostic_accepts_multiple_canonical_fetches(tmp_path):
    contract = _diagnostic_contract(tmp_path)
    stdout, stderr = _diagnostic_output(contract)

    urls = seal._validate_structural_diagnostic(
        stdout=stdout, stderr=stderr, contract=contract
    )

    assert len(urls) == 2
    assert urls[0].endswith("/linux-64/python.conda")
    assert urls[1].endswith("/noarch/pip.conda")
    assert seal._validate_structural_diagnostic(
        stdout=stdout,
        stderr=stderr + "\n",
        contract=contract,
    ) == urls
    with pytest.raises(seal.R9FailureSealError, match="stream drifted"):
        seal._validate_structural_diagnostic(
            stdout=stdout,
            stderr=stderr + "\n\n",
            contract=contract,
        )
    with pytest.raises(seal.R9FailureSealError, match="stream drifted"):
        seal._validate_structural_diagnostic(
            stdout=stdout,
            stderr=stderr + "fabricated suffix\n",
            contract=contract,
        )
    with pytest.raises(seal.R9FailureSealError, match="progress output"):
        seal._validate_structural_diagnostic(
            stdout="fabricated prefix\n" + stdout,
            stderr=stderr,
            contract=contract,
        )


def test_r9_validator_rejection_is_exact(tmp_path, monkeypatch):
    class HistoricalEvidenceError(RuntimeError):
        pass

    module = types.SimpleNamespace(
        EvidenceError=HistoricalEvidenceError,
        _validate_r3_probe_failure_signature=lambda **_kwargs: (
            (_ for _ in ()).throw(
                HistoricalEvidenceError(seal.EXPECTED_R9_REJECTION)
            )
        ),
    )
    monkeypatch.setattr(
        seal, "_load_r9_evidence_module", lambda _recovery: module
    )

    result = seal._r9_rejection(tmp_path, stdout="progress", stderr="errors")

    assert result == {
        "rejected": True,
        "error_class": "EvidenceError",
        "error": seal.EXPECTED_R9_REJECTION,
    }


def test_r9_original_partial_probe_tree_is_bound_without_offline_envelope(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    probe = tmp_path / "probe"
    probe.mkdir()
    broken = probe / seal.BROKEN_ENVELOPE_NAME
    broken.write_text("{}\n", encoding="utf-8")
    broken.with_suffix(".json.sha256").write_text(
        "fixture\n", encoding="utf-8"
    )
    cache = probe / "empty-conda-pkgs"
    cache.mkdir()
    (cache / "python-3.12.conda.partial").touch()
    (cache / "urls").touch()
    destination = probe / "offline-clone-destination"
    destination.mkdir()
    (destination / ".condarc").write_text("offline: true\n", encoding="utf-8")
    (destination / "micromamba").write_bytes(b"binary")
    expected_target = recovery / seal.R9_TOOLCHAIN_RELATIVE / "base/micromamba"
    (destination / "_conda").symlink_to(expected_target)
    module = types.SimpleNamespace(
        _validate_r3_prelaunch_failure_envelope=lambda _path: (
            {
                "classification": "unsafe_recorded_broken_internal_symlink",
                "scheduler_job_ids": [],
                "pre_scheduler_submission": True,
                "failure_id": "f" * 64,
            },
            b"{}\n",
            {"path": str(broken), "sha256": "a" * 64, "size": 3},
        )
    )
    monkeypatch.setattr(seal, "R9_PROBE_ROOT", probe)
    monkeypatch.setattr(
        seal, "_load_r9_evidence_module", lambda _recovery: module
    )

    report = seal._validate_original_probe_tree(recovery)

    assert report["offline_envelope_published"] is False
    assert report["partial_artifact_count"] == 1
    assert report["inventory"]["entry_count"] == 10


def test_r9_volatile_probe_loss_requires_absence_and_is_not_called_proof(
    tmp_path, monkeypatch
):
    probe = tmp_path / "externally-cleaned-r9-probe"
    monkeypatch.setattr(seal, "R9_PROBE_ROOT", probe)

    report = seal._validate_volatile_probe_absence()

    assert report["present_at_seal"] is False
    assert (
        report["loss_classification"]
        == "external_volatile_tmp_cleanup_before_durable_seal"
    )
    assert report["prior_live_audit"]["authoritative_seal_evidence"] is False
    assert (
        report["replacement_evidence_policy"]
        == "reproduce_both_exact_r9_recorder_commands_and_archive_marker_first"
    )
    probe.mkdir()
    with pytest.raises(
        seal.R9FailureSealError, match="reappeared before controlled reproduction"
    ):
        seal._validate_volatile_probe_absence()


def test_r9_exact_recorder_commands_bind_both_historical_probes(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    probe = tmp_path / "schema5-r3-prelaunch-researcher-r9"
    monkeypatch.setattr(seal, "R9_PROBE_ROOT", probe)

    broken = seal._r9_recorder_argv(
        recovery,
        classification="unsafe_recorded_broken_internal_symlink",
        apply=True,
    )
    offline = seal._r9_recorder_argv(
        recovery,
        classification="offline_clone_unseeded_release_local_cache",
        apply=True,
    )

    assert broken[:3] == [seal.sys.executable, "-I", "-B"]
    assert "--apply" in broken and "--apply" in offline
    assert "recorded-shared-conda-base" in broken
    assert "recorded-shared-conda-base" not in offline
    assert "CONDA_PKGS_DIRS" in offline
    assert str(probe / "empty-conda-pkgs") in offline
    assert str(probe / seal.BROKEN_ENVELOPE_NAME) in broken
    assert str(probe / seal.OFFLINE_ENVELOPE_NAME) in offline
    assert broken[-2:] == [
        "--conda-executable",
        "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda",
    ]
    assert offline[-7:] == [
        "create",
        "--yes",
        "--offline",
        "--clone",
        str(recovery / seal.R9_TOOLCHAIN_RELATIVE / "base"),
        "--prefix",
        str(probe / "offline-clone-destination"),
    ]


def test_r9_reproduction_accepts_only_the_exact_prefixed_cli_error(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    probe = tmp_path / "schema5-r3-prelaunch-reproduction"
    monkeypatch.setattr(seal, "R9_PROBE_ROOT", probe)
    monkeypatch.setattr(
        seal, "_validate_volatile_probe_absence", lambda: {"present": False}
    )
    records = [
        {
            "returncode": 0,
            "stdout": "{}\n",
            "stderr": "",
        }
        for _index in range(4)
    ] + [
        {
            "returncode": 2,
            "stdout": "",
            "stderr": seal.EXPECTED_R9_CLI_STDERR,
        }
    ]
    monkeypatch.setattr(
        seal, "_run_r9_recorder", lambda *_args, **_kwargs: records.pop(0)
    )
    state = {"inventory": {"entry_count": 2}}
    monkeypatch.setattr(
        seal, "_validate_original_probe_tree", lambda _recovery: state
    )

    reproduced, transcript = seal._reproduce_volatile_probe_tree(recovery)

    assert reproduced == state
    assert transcript["records"][-1]["stderr"].startswith(
        "[schema5-evidence] ERROR:"
    )

    second_probe = tmp_path / "schema5-r3-prelaunch-bare-error"
    monkeypatch.setattr(seal, "R9_PROBE_ROOT", second_probe)
    bare_records = [
        {
            "returncode": 0,
            "stdout": "{}\n",
            "stderr": "",
        }
        for _index in range(4)
    ] + [
        {
            "returncode": 2,
            "stdout": "",
            "stderr": f"ERROR: {seal.EXPECTED_R9_REJECTION}\n",
        }
    ]
    monkeypatch.setattr(
        seal, "_run_r9_recorder", lambda *_args, **_kwargs: bare_records.pop(0)
    )
    with pytest.raises(
        seal.R9FailureSealError, match="recorder rejection reproduction drifted"
    ):
        seal._reproduce_volatile_probe_tree(recovery)


def test_r9_transcript_reference_preserves_complete_file_metadata(
    tmp_path,
):
    transcript = tmp_path / "transcript.json"
    transcript.write_text('{"transcript_id":"fixture"}\n', encoding="utf-8")

    reference = seal._recorder_transcript_ref(transcript, "fixture")

    assert reference["path"] == str(transcript)
    assert reference["transcript_id"] == "fixture"
    assert reference["size"] == transcript.stat().st_size
    assert reference["mode"] == stat.S_IMODE(transcript.stat().st_mode)
    assert reference["link_count"] == 1
    assert len(reference["sha256"]) == 64


def test_r9_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.EVIDENCE_RELATIVE_ROOT
    source = {
        "path": str(seal.R9_PROBE_ROOT),
        "present_at_seal": False,
        "loss_classification": (
            "external_volatile_tmp_cleanup_before_durable_seal"
        ),
        "prior_live_audit": seal.PRIOR_VOLATILE_AUDIT,
        "replacement_evidence_policy": (
            "reproduce_both_exact_r9_recorder_commands_and_archive_marker_first"
        ),
    }
    intent = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "sealed_r9_toolchain": {},
        "original_probe_state": source,
        "reproduction": {},
        "scientific_state": {},
        "permanently_absent_r9_execution_paths": [],
        "scheduler": {"scheduler_user": "researcher"},
        "evidence_root": str(evidence),
    }
    intent["intent_id"] = seal._identity(intent, "intent_id")
    monkeypatch.setattr(
        seal, "_validate_volatile_probe_absence", lambda: source
    )
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
