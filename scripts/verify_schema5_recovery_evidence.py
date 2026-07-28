#!/usr/bin/env python3
"""Verify immutable schema-5 recovery evidence with its native protocol.

The v1.1-r1 and v1.2-r2 recovery chains are sealed historical evidence. They retain
their native protocol semantics; neither is reinterpreted as the active v1.2-r11
chain. This dispatcher reads only the protocol discriminator and selects either the
matching renderer or the deliberately read-only historical-r2 validator below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
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
R11_PROTOCOL = "schema5-v1.2-r11-recovery-chain"
_RENDERERS = {
    R1_PROTOCOL: "render_schema5_recovery_chain",
    R11_PROTOCOL: "render_schema5_recovery_chain_v12",
}
_MAX_DISCRIMINATOR_BYTES = 16 * 1024 * 1024
# No native r2 chain was admitted into the current recovery root. Historical r2
# validation is therefore deny-by-default until reviewed immutable byte identities
# are entered as ``manifest_sha256: frozenset(receipt_sha256)``.
HISTORICAL_R2_EVIDENCE_SHA256: dict[str, frozenset[str]] = {}


class EvidenceVerificationError(RuntimeError):
    """Recovery evidence is unsupported, mutable, or fails its native contract."""


def _absolute(path: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceVerificationError(
            f"recovery evidence path is unavailable: {lexical}: {exc}"
        ) from exc
    if lexical != resolved:
        raise EvidenceVerificationError(
            "recovery evidence path is noncanonical or traverses a symlink: "
            f"{lexical}"
        )
    return lexical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_discriminator_bytes(
    path: Path, *, require_canonical: bool = False
) -> tuple[dict[str, Any], bytes]:
    """Stably read a duplicate-free discriminator through a no-follow fd."""

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
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceVerificationError(
            f"cannot open recovery-chain discriminator: {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > _MAX_DISCRIMINATOR_BYTES:
                raise EvidenceVerificationError(
                    "recovery-chain manifest exceeds "
                    f"{_MAX_DISCRIMINATOR_BYTES} bytes"
                )
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731 - stable inode tuple
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(after)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise EvidenceVerificationError(
            f"recovery-chain discriminator changed or is unsafe: {path}"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceVerificationError(
            f"cannot parse recovery-chain manifest discriminator: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceVerificationError("recovery-chain manifest is not a JSON object")
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if require_canonical and raw != canonical:
        raise EvidenceVerificationError(
            "historical r2 recovery evidence is not canonical JSON"
        )
    return payload, raw


def _read_discriminator(path: Path) -> dict[str, Any]:
    """Read only enough untrusted input to select the authoritative verifier."""

    return _read_discriminator_bytes(path)[0]


def _renderer_module(module_name: str) -> ModuleType:
    """Load a sibling script both as a package module and under ``python -I``."""

    try:
        return importlib.import_module(f"scripts.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name not in {"scripts", f"scripts.{module_name}"}:
            raise
        return importlib.import_module(module_name)


def _historical_r2_chain(
    manifest_path: Path,
    receipt_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Validate immutable r2 bytes without importing the active r11 renderer."""

    manifest, manifest_raw = _read_discriminator_bytes(
        manifest_path, require_canonical=True
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    allowed_receipts = HISTORICAL_R2_EVIDENCE_SHA256.get(manifest_sha256)
    if allowed_receipts is None:
        raise EvidenceVerificationError(
            "historical r2 manifest bytes are not in the reviewed immutable "
            "identity allowlist"
        )
    identity = dict(manifest)
    chain_id = identity.pop("chain_id", None)
    jobs = manifest.get("jobs")
    jobs_root = Path(str(manifest.get("jobs_root", "")))
    if (
        manifest_path.stat().st_nlink != 1
        or stat.S_IMODE(manifest_path.stat().st_mode) & 0o222
        or manifest.get("protocol") != R2_PROTOCOL
        or manifest.get("namespace") != "schema5-v1.2-r2"
        or manifest.get("release_tag") != "sweep-recovery-schema5-v1.2-r2"
        or not isinstance(chain_id, str)
        or len(chain_id) != 64
        or any(character not in "0123456789abcdef" for character in chain_id)
        or chain_id
        != hashlib.sha256(
            json.dumps(
                identity,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
        or not jobs_root.is_absolute()
        or not isinstance(jobs, list)
        or not jobs
    ):
        raise EvidenceVerificationError(
            "historical r2 recovery-chain identity is invalid"
        )
    seen_names: set[str] = set()
    scripts: dict[str, tuple[str, str]] = {}
    for row in jobs:
        dependencies = row.get("dependencies") if isinstance(row, dict) else None
        name = row.get("name") if isinstance(row, dict) else None
        script = Path(str(row.get("script", ""))) if isinstance(row, dict) else Path()
        script_hash = row.get("script_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or name in seen_names
            or not isinstance(dependencies, list)
            or any(
                not isinstance(item, str) or item not in seen_names
                for item in dependencies
            )
            or row.get("dependency_type") not in {"afterok", "afterany"}
            or row.get("no_requeue") is not True
            or not script.is_absolute()
            or script.parent != jobs_root
            or script.is_symlink()
            or not script.is_file()
            or script.stat().st_nlink != 1
            or stat.S_IMODE(script.stat().st_mode) & 0o222
            or not isinstance(script_hash, str)
            or _sha256(script) != script_hash
        ):
            raise EvidenceVerificationError(
                "historical r2 rendered-job topology or bytes drifted"
            )
        seen_names.add(name)
        scripts[name] = (str(script), script_hash)
    report = {
        "status": "verified_historical_r2_read_only",
        "chain_id": chain_id,
        "job_count": len(jobs),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    if receipt_path is None:
        return report, manifest, None
    receipt, receipt_raw = _read_discriminator_bytes(
        receipt_path, require_canonical=True
    )
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if receipt_sha256 not in allowed_receipts:
        raise EvidenceVerificationError(
            "historical r2 receipt bytes are not in the reviewed immutable "
            "identity allowlist"
        )
    receipt_identity = dict(receipt)
    receipt_id = receipt_identity.pop("receipt_id", None)
    receipt_jobs = receipt.get("jobs")
    protocol = receipt.get("protocol")
    if (
        receipt_path.stat().st_nlink != 1
        or stat.S_IMODE(receipt_path.stat().st_mode) & 0o222
        or protocol
        not in {f"{R2_PROTOCOL}-submission", f"{R2_PROTOCOL}-repair"}
        or receipt.get("chain_id") != chain_id
        or receipt.get("manifest") != str(manifest_path)
        or receipt.get("manifest_sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or receipt.get("root_initial_hold") is not True
        or receipt.get("no_requeue") is not True
        or not isinstance(receipt_id, str)
        or receipt_id
        != hashlib.sha256(
            json.dumps(
                receipt_identity,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
        or not isinstance(receipt_jobs, list)
        or len(receipt_jobs) != len(jobs)
    ):
        raise EvidenceVerificationError(
            "historical r2 submission receipt identity is invalid"
        )
    seen_ids: set[str] = set()
    submitted: dict[str, str] = {}
    for manifest_row, receipt_row in zip(jobs, receipt_jobs, strict=True):
        name = manifest_row["name"]
        job_id = receipt_row.get("job_id")
        dependencies = manifest_row["dependencies"]
        if (
            receipt_row.get("name") != name
            or not isinstance(job_id, str)
            or not job_id.isdigit()
            or job_id in seen_ids
            or receipt_row.get("dependencies") != dependencies
            or receipt_row.get("dependency_job_ids")
            != [submitted[item] for item in dependencies]
            or receipt_row.get("script") != scripts[name][0]
            or receipt_row.get("script_sha256") != scripts[name][1]
            or not isinstance(receipt_row.get("comment"), str)
            or not receipt_row["comment"].startswith(
                f"asys:s5-recovery-v1.2-r2:{chain_id}:g"
            )
        ):
            raise EvidenceVerificationError(
                "historical r2 scheduler receipt topology drifted"
            )
        seen_ids.add(job_id)
        submitted[name] = job_id
    report["submission_receipt_sha256"] = hashlib.sha256(
        receipt_raw
    ).hexdigest()
    return report, manifest, receipt


def verify_recovery_evidence(
    chain_manifest: str | Path,
    submission_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a chain and optional base/repair receipt without mutating either."""

    manifest_path = _absolute(chain_manifest)
    discriminator = _read_discriminator(manifest_path)
    protocol = discriminator.get("protocol")
    if (
        not isinstance(protocol, str)
        or (
            protocol not in _RENDERERS
            and protocol != R2_PROTOCOL
        )
    ):
        raise EvidenceVerificationError(
            f"unsupported recovery-chain protocol: {protocol!r}"
        )

    receipt_path = (
        None
        if submission_receipt is None
        else _absolute(submission_receipt)
    )
    historical_receipt: dict[str, Any] | None = None
    if protocol == R2_PROTOCOL:
        module_name = "historical_r2_read_only_verifier"
        chain_report, manifest, historical_receipt = _historical_r2_chain(
            manifest_path, receipt_path
        )
        renderer = None
    else:
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
    if protocol != R2_PROTOCOL:
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
        assert receipt_path is not None
        if protocol == R2_PROTOCOL:
            assert historical_receipt is not None
            result.update(
                {
                    "submission_receipt_path": str(receipt_path),
                    "submission_receipt_sha256": _sha256(receipt_path),
                    "submission_receipt": historical_receipt,
                }
            )
            return result
        try:
            assert renderer is not None
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
                protocol == R11_PROTOCOL
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
