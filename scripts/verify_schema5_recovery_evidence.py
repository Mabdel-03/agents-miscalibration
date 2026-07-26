#!/usr/bin/env python3
"""Verify immutable schema-5 recovery evidence with its native protocol.

The v1.1-r1 recovery chain is sealed historical evidence.  It must continue to be
verified by the r1 renderer's frozen contract; interpreting it as v1.2-r2 would
silently rewrite the meaning of that evidence.  This small dispatcher reads only the
protocol discriminator and delegates all substantive validation to the matching
renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any


# ``python -I`` deliberately removes the script directory from the implicit import
# path.  The r2 renderer publishes this verifier and both native renderers as one
# checksummed flat bundle, so opt into that exact immutable directory explicitly.
_BUNDLE_DIRECTORY = str(Path(__file__).resolve().parent)
if _BUNDLE_DIRECTORY not in sys.path:
    sys.path.insert(0, _BUNDLE_DIRECTORY)


R1_PROTOCOL = "schema5-v1.1-r1-recovery-chain"
R2_PROTOCOL = "schema5-v1.2-r2-recovery-chain"
_RENDERERS = {
    R1_PROTOCOL: "render_schema5_recovery_chain",
    R2_PROTOCOL: "render_schema5_recovery_chain_v12",
}
_MAX_DISCRIMINATOR_BYTES = 16 * 1024 * 1024


class EvidenceVerificationError(RuntimeError):
    """Recovery evidence is unsupported, mutable, or fails its native contract."""


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_discriminator(path: Path) -> dict[str, Any]:
    """Read only enough untrusted input to select the authoritative verifier."""

    path = _absolute(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EvidenceVerificationError(
            f"recovery-chain manifest is unavailable: {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceVerificationError(
            f"recovery-chain manifest must be one regular non-symlink file: {path}"
        )
    if metadata.st_size > _MAX_DISCRIMINATOR_BYTES:
        raise EvidenceVerificationError(
            f"recovery-chain manifest exceeds {_MAX_DISCRIMINATOR_BYTES} bytes"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(
            f"cannot parse recovery-chain manifest discriminator: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceVerificationError("recovery-chain manifest is not a JSON object")
    return payload


def _renderer_module(module_name: str) -> ModuleType:
    """Load a sibling script both as a package module and under ``python -I``."""

    try:
        return importlib.import_module(f"scripts.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", f"scripts.{module_name}"}:
            raise
        return importlib.import_module(module_name)


def verify_recovery_evidence(
    chain_manifest: str | Path,
    submission_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a chain and optional base/repair receipt without mutating either."""

    manifest_path = _absolute(chain_manifest)
    discriminator = _read_discriminator(manifest_path)
    protocol = discriminator.get("protocol")
    if not isinstance(protocol, str) or protocol not in _RENDERERS:
        raise EvidenceVerificationError(
            f"unsupported recovery-chain protocol: {protocol!r}"
        )

    module_name = _RENDERERS[protocol]
    renderer = _renderer_module(module_name)
    try:
        chain_report = renderer.verify_chain(manifest_path)
    except Exception as exc:
        raise EvidenceVerificationError(
            f"{protocol} chain verification failed: {exc}"
        ) from exc

    # Re-read only after the native renderer has proved the complete schema, identity,
    # canonical paths, modes, and rendered scripts.
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(
            f"verified recovery-chain manifest became unreadable: {exc}"
        ) from exc
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-recovery-evidence-protocol-dispatch",
        "chain_protocol": protocol,
        "renderer": module_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "chain_report": chain_report,
        "submission_receipt_path": None,
        "submission_receipt_sha256": None,
        "submission_receipt": None,
    }

    if submission_receipt is not None:
        receipt_path = _absolute(submission_receipt)
        try:
            receipt_discriminator = _read_discriminator(receipt_path)
            receipt_protocol = receipt_discriminator.get("protocol")
            if receipt_protocol == f"{protocol}-submission":
                comments = renderer._submission_comments(manifest)
                receipt = renderer._validate_submission_receipt(
                    receipt_path,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    comments=comments,
                )
            elif (
                protocol == R2_PROTOCOL
                and receipt_protocol == f"{protocol}-repair"
            ):
                generation = receipt_discriminator.get("repair_generation")
                parent_value = receipt_discriminator.get("parent_receipt")
                if (
                    not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or generation <= 0
                    or not isinstance(parent_value, str)
                ):
                    raise EvidenceVerificationError(
                        "repair receipt has invalid generation/parent provenance"
                    )
                parent_path = _absolute(parent_value)
                receipt = renderer._validate_repair_receipt(
                    receipt_path,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    generation=generation,
                    parent_path=parent_path,
                )
            else:
                raise EvidenceVerificationError(
                    f"unsupported receipt protocol for {protocol}: "
                    f"{receipt_protocol!r}"
                )
        except Exception as exc:
            if isinstance(exc, EvidenceVerificationError):
                raise
            raise EvidenceVerificationError(
                f"{protocol} submission-receipt verification failed: {exc}"
            ) from exc
        result.update(
            {
                "submission_receipt_path": str(receipt_path),
                "submission_receipt_sha256": _sha256(receipt_path),
                "submission_receipt": receipt,
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify immutable schema-5 recovery evidence with the renderer selected "
            "by its frozen protocol."
        )
    )
    parser.add_argument("--chain-manifest", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_recovery_evidence(
            args.chain_manifest,
            args.submission_receipt,
        )
    except EvidenceVerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    printable = {
        key: value
        for key, value in report.items()
        if key not in {"manifest", "submission_receipt"}
    }
    printable["passed"] = True
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
