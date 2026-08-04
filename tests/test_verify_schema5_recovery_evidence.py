"""Protocol-dispatch tests for immutable recovery-chain evidence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import verify_schema5_recovery_evidence as evidence


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("protocol", "module_name"),
    [
        (evidence.R1_PROTOCOL, "render_schema5_recovery_chain"),
        (evidence.R15_PROTOCOL, "render_schema5_recovery_chain_v12"),
    ],
)
def test_dispatches_chain_and_receipt_to_native_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    module_name: str,
) -> None:
    manifest = tmp_path / "chain.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text(
        json.dumps({"protocol": protocol, "chain_id": "chain"}) + "\n",
        encoding="utf-8",
    )
    receipt.write_text(
        json.dumps(
            {
                "protocol": f"{protocol}-submission",
                "receipt_id": "receipt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    receipt.chmod(0o444)
    calls: list[object] = []

    def verify_chain(path: Path) -> dict[str, object]:
        calls.append(("chain", path))
        return {"passed": True}

    def comments(payload: dict[str, object]) -> dict[str, str]:
        calls.append(("comments", payload["protocol"]))
        return {"job": "comment"}

    def validate_receipt(
        path: Path,
        *,
        manifest: dict[str, object],
        manifest_path: Path,
        comments: dict[str, str],
    ) -> dict[str, str]:
        calls.append(("receipt", path, manifest_path, comments))
        assert manifest["protocol"] == protocol
        return {"receipt_id": "receipt"}

    renderer = SimpleNamespace(
        verify_chain=verify_chain,
        _submission_comments=comments,
        _validate_submission_receipt=validate_receipt,
    )
    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda requested: renderer
        if requested == module_name
        else pytest.fail(f"wrong renderer: {requested}"),
    )
    before = (manifest.read_bytes(), receipt.read_bytes())
    report = evidence.verify_recovery_evidence(manifest, receipt)

    assert report["chain_protocol"] == protocol
    assert report["renderer"] == module_name
    assert report["manifest"]["protocol"] == protocol
    assert report["submission_receipt"] == {"receipt_id": "receipt"}
    assert [call[0] for call in calls] == ["chain", "comments", "receipt"]
    assert (manifest.read_bytes(), receipt.read_bytes()) == before


def test_r15_repair_receipt_dispatches_to_generation_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "chain.json"
    parent = tmp_path / "base.json"
    receipt = tmp_path / "g0001" / "receipt.json"
    receipt.parent.mkdir()
    manifest.write_text(
        json.dumps({"protocol": evidence.R15_PROTOCOL, "chain_id": "chain"}) + "\n",
        encoding="utf-8",
    )
    parent.write_text("{}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "protocol": f"{evidence.R15_PROTOCOL}-repair",
                "repair_generation": 1,
                "parent_receipt": str(parent.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[object, ...]] = []

    def validate(path: Path, **kwargs):
        calls.append((path, kwargs["generation"], kwargs["parent_path"]))
        return {"receipt_id": "repair"}

    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda _name: SimpleNamespace(
            verify_chain=lambda _path: {"passed": True},
            _validate_repair_receipt=validate,
        ),
    )
    report = evidence.verify_recovery_evidence(manifest, receipt)
    assert report["submission_receipt"] == {"receipt_id": "repair"}
    assert calls == [(receipt.resolve(), 1, parent.resolve())]


def test_rejects_unknown_protocol_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "chain.json"
    manifest.write_text('{"protocol":"schema5-future"}\n', encoding="utf-8")
    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda _name: pytest.fail("unsupported evidence selected a renderer"),
    )
    with pytest.raises(evidence.EvidenceVerificationError, match="unsupported"):
        evidence.verify_recovery_evidence(manifest)


def test_historical_r2_uses_read_only_native_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_root = tmp_path / "jobs"
    jobs_root.mkdir()
    script = jobs_root / "00_root.sbatch"
    script.write_bytes(b"#!/bin/bash\ntrue\n")
    script.chmod(0o444)
    manifest_path = tmp_path / "r2-chain.json"
    manifest = {
        "schema_version": 1,
        "protocol": evidence.R2_PROTOCOL,
        "namespace": "schema5-v1.2-r2",
        "release_tag": "sweep-recovery-schema5-v1.2-r2",
        "jobs_root": str(jobs_root),
        "jobs": [
            {
                "name": "source_checkout",
                "job_name": "asys-r2-root",
                "script": str(script),
                "script_sha256": hashlib.sha256(
                    script.read_bytes()
                ).hexdigest(),
                "dependencies": [],
                "dependency_type": "afterok",
                "no_requeue": True,
            }
        ],
    }
    manifest["chain_id"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))
    manifest_path.chmod(0o444)
    receipt_path = tmp_path / "r2-receipt.json"
    receipt = {
        "schema_version": 1,
        "protocol": f"{evidence.R2_PROTOCOL}-submission",
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "root_initial_hold": True,
        "no_requeue": True,
        "jobs": [
            {
                "name": "source_checkout",
                "job_id": "123",
                "dependencies": [],
                "dependency_job_ids": [],
                "comment": (
                    "asys:s5-recovery-v1.2-r2:"
                    f"{manifest['chain_id']}:g0000:source_checkout"
                ),
                "script": str(script),
                "script_sha256": manifest["jobs"][0]["script_sha256"],
            }
        ],
    }
    receipt["receipt_id"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))
    receipt_path.chmod(0o444)
    monkeypatch.setattr(
        evidence,
        "HISTORICAL_R2_EVIDENCE_SHA256",
        {
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(): frozenset(
                {hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
            )
        },
    )
    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda _name: pytest.fail("historical r2 imported the active renderer"),
    )

    report = evidence.verify_recovery_evidence(
        manifest_path, receipt_path
    )

    assert report["chain_protocol"] == evidence.R2_PROTOCOL
    assert report["renderer"] == "historical_r2_read_only_verifier"
    assert (
        report["chain_report"]["status"]
        == "verified_historical_r2_read_only"
    )


def test_rejects_symlink_discriminator(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps({"protocol": evidence.R1_PROTOCOL}) + "\n",
        encoding="utf-8",
    )
    link = tmp_path / "chain.json"
    link.symlink_to(target)
    with pytest.raises(
        evidence.EvidenceVerificationError, match="traverses a symlink"
    ):
        evidence.verify_recovery_evidence(link)


def test_rejects_duplicate_discriminator_key(tmp_path: Path) -> None:
    manifest = tmp_path / "chain.json"
    manifest.write_text(
        '{"protocol":"schema5-v1.2-r15-recovery-chain",'
        '"protocol":"schema5-v1.2-r2-recovery-chain"}\n',
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    with pytest.raises(
        evidence.EvidenceVerificationError, match="duplicate JSON key"
    ):
        evidence.verify_recovery_evidence(manifest)


def test_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    manifest = real / "chain.json"
    manifest.write_text(
        json.dumps({"protocol": evidence.R15_PROTOCOL}) + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        evidence.EvidenceVerificationError, match="traverses a symlink"
    ):
        evidence.verify_recovery_evidence(alias / "chain.json")


def test_lexical_alias_is_canonicalized_before_native_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "chain.json"
    manifest.write_text(
        json.dumps({"protocol": evidence.R15_PROTOCOL}) + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    (tmp_path / "unused").mkdir()
    observed: list[Path] = []
    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda _name: SimpleNamespace(
            verify_chain=lambda path: observed.append(path)
            or {"passed": True}
        ),
    )

    evidence.verify_recovery_evidence(
        tmp_path / "unused" / ".." / "chain.json"
    )

    assert observed == [manifest]


def test_cli_is_read_only_and_reports_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "chain.json"
    manifest.write_text(
        json.dumps({"protocol": evidence.R15_PROTOCOL, "chain_id": "chain"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evidence,
        "_renderer_module",
        lambda _name: SimpleNamespace(
            verify_chain=lambda _path: {"passed": True}
        ),
    )
    assert evidence.main(["--chain-manifest", str(manifest)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["renderer"] == "render_schema5_recovery_chain_v12"


def test_flat_bundle_imports_under_python_isolated_mode(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in (
        "schema5_recovery_sentinel.py",
        "verify_schema5_recovery_evidence.py",
    ):
        shutil.copyfile(Path("scripts") / name, bundle / name)
    for name in (
        "schema5_recovery_sentinel.py",
        "verify_schema5_recovery_evidence.py",
    ):
        result = subprocess.run(
            [sys.executable, "-I", str(bundle / name), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
