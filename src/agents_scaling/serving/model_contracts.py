"""Immutable model and tokenizer identities for the schema-5 production fleet.

The model-contract digest is the SHA-256 of the exact JSON sidecar bytes.  This is the
value stored in run policy and result provenance.  The loader is deliberately strict:
it rejects a missing checksum, duplicate JSON keys, non-finite numbers, unknown fields,
an incomplete dense ladder, and any revision that is not an exact Hugging Face commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


MODEL_CONTRACT_SCHEMA_VERSION = 1
MODEL_CONTRACT_FILENAME = "model_contracts.v1.json"
MODEL_CONTRACT_CHECKSUM_FILENAME = "model_contracts.v1.sha256"
MODEL_SIZES = ("0.6B", "1.7B", "4B", "8B", "14B", "32B")
_ROOT_FIELDS = frozenset({"schema_version", "family", "offline_required", "models"})
_MODEL_FIELDS = frozenset(
    {"hf_id", "model_revision", "tokenizer_id", "tokenizer_revision"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ModelContractError(RuntimeError):
    """A frozen model contract is absent, malformed, or does not match runtime."""


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def default_model_contract_path() -> Path:
    """Return the checked-in model contract without consulting the process cwd.

    Production installs the package into an immutable environment, so ``__file__`` is
    below ``site-packages`` rather than the separately frozen release worktree.  The
    Slurm entry points export an already verified absolute release root; use only the
    canonical contract location below it.  Source-checkout and legacy callers retain
    the historical package-relative fallback.
    """

    release_value = os.environ.get("ASYS_RELEASE_WORKTREE")
    if release_value:
        raw_release = Path(release_value).expanduser()
        if not raw_release.is_absolute():
            raise ModelContractError("ASYS_RELEASE_WORKTREE must be absolute")
        release = raw_release.resolve()
        if raw_release.is_symlink() or not release.is_dir():
            raise ModelContractError(
                "ASYS_RELEASE_WORKTREE is missing or symlinked"
            )
        contract = release / "configs" / MODEL_CONTRACT_FILENAME
        if contract.is_symlink() or not contract.is_file():
            raise ModelContractError(
                f"release worktree lacks a regular model contract: {contract}"
            )
        return contract

    return Path(__file__).resolve().parents[3] / "configs" / MODEL_CONTRACT_FILENAME


@dataclass(frozen=True)
class FrozenModelIdentity:
    size: str
    hf_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str


@dataclass(frozen=True)
class FrozenModelContracts:
    path: Path
    checksum_path: Path
    sha256: str
    family: str
    offline_required: bool
    models: Mapping[str, FrozenModelIdentity]

    def for_size(self, size: str) -> FrozenModelIdentity:
        try:
            return self.models[size]
        except KeyError as exc:
            raise ModelContractError(
                f"model size {size!r} is outside frozen contract {self.sha256}"
            ) from exc

    def verify_identity(
        self,
        *,
        size: str,
        hf_id: str,
        model_revision: str,
        tokenizer_id: str,
        tokenizer_revision: str,
    ) -> FrozenModelIdentity:
        """Fail closed unless all runtime identities equal their frozen pins."""

        expected = self.for_size(size)
        observed = {
            "hf_id": hf_id,
            "model_revision": model_revision,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        }
        wanted = {
            "hf_id": expected.hf_id,
            "model_revision": expected.model_revision,
            "tokenizer_id": expected.tokenizer_id,
            "tokenizer_revision": expected.tokenizer_revision,
        }
        if observed != wanted:
            differences = ", ".join(
                f"{key}: expected {wanted[key]!r}, observed {observed[key]!r}"
                for key in wanted
                if wanted[key] != observed[key]
            )
            raise ModelContractError(
                f"model/tokenizer drift for {size} under {self.sha256}: {differences}"
            )
        return expected

    def verify_contract_hash(self, observed_sha256: str) -> None:
        if observed_sha256 != self.sha256:
            raise ModelContractError(
                "model-contract hash drift: "
                f"expected {self.sha256}, observed {observed_sha256}"
            )


def _decode(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelContractError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelContractError(f"{path} must contain one JSON object")
    return value


def _read_checksum(path: Path, expected_filename: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ModelContractError(f"missing regular model-contract checksum: {path}")
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise ModelContractError(f"cannot read {path}: {exc}") from exc
    if (
        len(fields) != 2
        or _SHA256_RE.fullmatch(fields[0]) is None
        or fields[1] != expected_filename
    ):
        raise ModelContractError(f"invalid model-contract checksum record: {path}")
    return fields[0]


def load_model_contracts(
    path: str | Path | None = None, *, expected_sha256: str | None = None
) -> FrozenModelContracts:
    """Load and validate the exact-byte frozen Qwen contract."""

    contract_path = default_model_contract_path() if path is None else Path(path)
    checksum_path = contract_path.with_name(MODEL_CONTRACT_CHECKSUM_FILENAME)
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ModelContractError(f"missing regular model contract: {contract_path}")
    raw = contract_path.read_bytes()
    observed_digest = hashlib.sha256(raw).hexdigest()
    recorded_digest = _read_checksum(checksum_path, contract_path.name)
    if recorded_digest != observed_digest:
        raise ModelContractError(
            f"model-contract checksum mismatch: expected {recorded_digest}, "
            f"observed {observed_digest}"
        )
    if expected_sha256 is not None and expected_sha256 != observed_digest:
        raise ModelContractError(
            f"model-contract policy mismatch: expected {expected_sha256}, "
            f"observed {observed_digest}"
        )

    payload = _decode(raw, contract_path)
    if set(payload) != _ROOT_FIELDS:
        raise ModelContractError("model contract has the wrong root fields")
    if payload["schema_version"] != MODEL_CONTRACT_SCHEMA_VERSION:
        raise ModelContractError(
            f"unsupported model contract schema {payload['schema_version']!r}"
        )
    if payload["family"] != "Qwen3-dense":
        raise ModelContractError("model contract family must be Qwen3-dense")
    if payload["offline_required"] is not True:
        raise ModelContractError("schema-5 production requires offline model loading")
    raw_models = payload["models"]
    if not isinstance(raw_models, dict) or tuple(raw_models) != MODEL_SIZES:
        raise ModelContractError(
            f"model contract must contain the ordered dense ladder {MODEL_SIZES}"
        )

    models: dict[str, FrozenModelIdentity] = {}
    for size in MODEL_SIZES:
        row = raw_models[size]
        if not isinstance(row, dict) or set(row) != _MODEL_FIELDS:
            raise ModelContractError(f"model contract entry {size} has wrong fields")
        if any(not isinstance(row[field], str) or not row[field] for field in row):
            raise ModelContractError(f"model contract entry {size} has invalid text")
        if _COMMIT_RE.fullmatch(row["model_revision"]) is None:
            raise ModelContractError(f"{size} model revision is not an exact commit")
        if _COMMIT_RE.fullmatch(row["tokenizer_revision"]) is None:
            raise ModelContractError(f"{size} tokenizer revision is not an exact commit")
        if row["tokenizer_id"] != row["hf_id"]:
            raise ModelContractError(f"{size} tokenizer/model repository drift")
        models[size] = FrozenModelIdentity(size=size, **row)

    return FrozenModelContracts(
        path=contract_path.resolve(),
        checksum_path=checksum_path.resolve(),
        sha256=observed_digest,
        family=payload["family"],
        offline_required=True,
        models=MappingProxyType(models),
    )


def offline_environment() -> dict[str, str]:
    """Environment required by launchers after the cache audit has passed."""

    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
