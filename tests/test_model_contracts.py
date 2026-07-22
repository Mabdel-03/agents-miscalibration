"""Frozen model/tokenizer contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import pytest

from agents_scaling.serving import model_contracts as contracts


def _copy_contract(tmp_path: Path) -> Path:
    source = contracts.default_model_contract_path()
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    checksum = tmp_path / contracts.MODEL_CONTRACT_CHECKSUM_FILENAME
    checksum.write_text(
        f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}\n",
        encoding="utf-8",
    )
    return target


def test_checked_in_model_contract_is_frozen_complete_dense_ladder():
    frozen = contracts.load_model_contracts()

    assert tuple(frozen.models) == contracts.MODEL_SIZES
    assert frozen.offline_required is True
    assert frozen.for_size("32B").model_revision == (
        "9216db5781bf21249d130ec9da846c4624c16137"
    )
    assert len(frozen.sha256) == 64


def test_runtime_identity_drift_fails_closed():
    frozen = contracts.load_model_contracts()
    expected = frozen.for_size("8B")
    frozen.verify_identity(
        size="8B",
        hf_id=expected.hf_id,
        model_revision=expected.model_revision,
        tokenizer_id=expected.tokenizer_id,
        tokenizer_revision=expected.tokenizer_revision,
    )

    with pytest.raises(contracts.ModelContractError, match="drift"):
        frozen.verify_identity(
            size="8B",
            hf_id=expected.hf_id,
            model_revision="0" * 40,
            tokenizer_id=expected.tokenizer_id,
            tokenizer_revision=expected.tokenizer_revision,
        )


def test_byte_or_checksum_drift_is_rejected(tmp_path):
    target = _copy_contract(tmp_path)
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(contracts.ModelContractError, match="checksum mismatch"):
        contracts.load_model_contracts(target)


def test_policy_expected_hash_is_enforced(tmp_path):
    target = _copy_contract(tmp_path)

    with pytest.raises(contracts.ModelContractError, match="policy mismatch"):
        contracts.load_model_contracts(target, expected_sha256="f" * 64)


def test_offline_environment_is_explicit():
    assert contracts.offline_environment() == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_installed_runtime_resolves_contract_from_explicit_release_worktree(
    tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[1] / "configs"
    release = tmp_path / "immutable-release"
    config_dir = release / "configs"
    config_dir.mkdir(parents=True)
    for name in (
        contracts.MODEL_CONTRACT_FILENAME,
        contracts.MODEL_CONTRACT_CHECKSUM_FILENAME,
    ):
        shutil.copy2(source / name, config_dir / name)

    monkeypatch.setenv("ASYS_RELEASE_WORKTREE", str(release.resolve()))
    assert contracts.default_model_contract_path() == (
        release.resolve() / "configs" / contracts.MODEL_CONTRACT_FILENAME
    )
    assert contracts.load_model_contracts().path.parent == config_dir.resolve()

    monkeypatch.setenv("ASYS_RELEASE_WORKTREE", "relative-release")
    with pytest.raises(contracts.ModelContractError, match="must be absolute"):
        contracts.load_model_contracts()
