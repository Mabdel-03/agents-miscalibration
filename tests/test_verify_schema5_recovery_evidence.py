"""Protocol-dispatch tests for immutable recovery-chain evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import verify_schema5_recovery_evidence as evidence


@pytest.mark.parametrize(
    ("protocol", "module_name"),
    [
        (evidence.R1_PROTOCOL, "render_schema5_recovery_chain"),
        (evidence.R2_PROTOCOL, "render_schema5_recovery_chain_v12"),
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


def test_r2_repair_receipt_dispatches_to_generation_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "chain.json"
    parent = tmp_path / "base.json"
    receipt = tmp_path / "g0001" / "receipt.json"
    receipt.parent.mkdir()
    manifest.write_text(
        json.dumps({"protocol": evidence.R2_PROTOCOL, "chain_id": "chain"}) + "\n",
        encoding="utf-8",
    )
    parent.write_text("{}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "protocol": f"{evidence.R2_PROTOCOL}-repair",
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


def test_rejects_symlink_discriminator(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text(
        json.dumps({"protocol": evidence.R1_PROTOCOL}) + "\n",
        encoding="utf-8",
    )
    link = tmp_path / "chain.json"
    link.symlink_to(target)
    with pytest.raises(evidence.EvidenceVerificationError, match="non-symlink"):
        evidence.verify_recovery_evidence(link)


def test_cli_is_read_only_and_reports_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "chain.json"
    manifest.write_text(
        json.dumps({"protocol": evidence.R2_PROTOCOL, "chain_id": "chain"}) + "\n",
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
