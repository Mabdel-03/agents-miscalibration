"""One dependency vocabulary for every cell (plan D16, §3.4).

ARD36 had ``cell:`` as ordering *advice* (``study/runner.py:132`` logs and proceeds) and
``controller:``/``state:`` as hard gates resolved by ``ard/runner.py:123-226``; the sealer that
should have written the ``state:`` registers was never called, so the fork stage had a
dependency mechanism at both ends and no middle (``ard/seal.py`` docstring).  Here every
dependency is hard, every register is content-addressed, and the reader and the writer are
the same module:

``cell:<cell_id>``
    ``<run_root>/cells/<cell_id>/meta.json`` exists **and** the directory is owned by that
    cell: ``meta.cell_id`` equals the directory name, ``meta.n_items`` equals the number of
    item files, every item file claims the cell, no ``incomplete/`` items, no
    ``SUSPENDED.json`` (``ard/runner.py:415 _assert_directory_is_this_cells``, defect D-04).
``bundle:<family>/<bucket>``
    the label-free root bundle ``<run_root>/plan/root_teams/<family>/<bucket>.json`` whose
    ``.sha256`` sidecar matches its bytes.
``register:<kind>/<sha>``
    ``<run_root>/registers/<kind>/<sha>/REGISTER.json`` for ``pool`` (sealed selection pools),
    ``packets`` (E2a fixed packets), ``panel`` (capture panel); content-addressed over the
    canonical JSON of the body (the ``ard/runner.py:freeze_register`` rule), and never empty
    (the ``ard/seal.py`` rule: sealing nothing hands a consumer a dependency that resolves
    but contains no parents).

A tampered artifact is reported as ``tampered``, never as met and never as merely missing,
so ``plan --repair-registers`` can regenerate exactly the missing ones.  Every write is
tmp + fsync + rename.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn.errors import DependencyUnmet, ManifestError
from agents_scaling.study import identity

CELL_PREFIX: Final[str] = "cell:"
BUNDLE_PREFIX: Final[str] = "bundle:"
REGISTER_PREFIX: Final[str] = "register:"
PREFIXES: Final[tuple[str, ...]] = (CELL_PREFIX, BUNDLE_PREFIX, REGISTER_PREFIX)

REGISTER_POOL: Final[str] = "pool"
REGISTER_PACKETS: Final[str] = "packets"
REGISTER_PANEL: Final[str] = "panel"
REGISTER_KINDS: Final[tuple[str, ...]] = (REGISTER_POOL, REGISTER_PACKETS, REGISTER_PANEL)

REGISTERS_DIR: Final[str] = "registers"
REGISTER_FILE: Final[str] = "REGISTER.json"
BUNDLES_DIR: Final[tuple[str, ...]] = ("plan", "root_teams")
CELLS_DIR: Final[str] = "cells"
META_FILE: Final[str] = "meta.json"
SUSPENDED_FILE: Final[str] = "SUSPENDED.json"
REGISTER_SHA_FIELD: Final[str] = "register_sha256"
REGISTER_ENTRIES_FIELD: Final[str] = "entries"

STATUS_MET: Final[str] = "met"
STATUS_UNMET: Final[str] = "unmet"
STATUS_TAMPERED: Final[str] = "tampered"
STATUS_MISSING: Final[str] = "missing"

_SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,180}$")
_FAMILY_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_BUCKET_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


# --------------------------------------------------------------------------- vocabulary


@dataclass(frozen=True)
class ParsedDependency:
    kind: str        # "cell" | "bundle" | "register"
    value: str       # cell_id | "<family>/<bucket>" | "<register kind>/<sha>"

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.value.split("/"))


def cell_dependency(cell_id: str) -> str:
    if not _CELL_ID_RE.match(str(cell_id)):
        raise ManifestError(f"bad cell id in a dependency: {cell_id!r}")
    return CELL_PREFIX + str(cell_id)


def bundle_dependency(family: str, bucket: str | int) -> str:
    return BUNDLE_PREFIX + f"{_family(family)}/{bucket_name(bucket)}"


def register_dependency(kind: str, sha: str) -> str:
    return REGISTER_PREFIX + f"{_register_kind(kind)}/{_sha(sha)}"


def bucket_name(bucket: str | int) -> str:
    """Bucket tokens: an int ``k`` becomes ``b<k:04d>``; a string must already be a token."""
    if isinstance(bucket, bool):
        raise ManifestError("bucket must be an int or a token, not a bool")
    if isinstance(bucket, int):
        if bucket < 0:
            raise ManifestError(f"bucket index must be >= 0, got {bucket}")
        return f"b{bucket:04d}"
    if not _BUCKET_RE.match(str(bucket)):
        raise ManifestError(f"bad bucket token {bucket!r}")
    return str(bucket)


def parse_dependency(dep: str) -> ParsedDependency:
    """``cell:``/``bundle:``/``register:`` only.  A bare id or another prefix is an error:
    there is no advisory dependency in nextdyn (plan D16)."""
    if not isinstance(dep, str):
        raise DependencyUnmet(f"dependency must be a string, got {type(dep).__name__}")
    if dep.startswith(CELL_PREFIX):
        cell_id = dep[len(CELL_PREFIX):]
        if not _CELL_ID_RE.match(cell_id):
            raise DependencyUnmet(f"bad cell id in dependency {dep!r}")
        return ParsedDependency("cell", cell_id)
    if dep.startswith(BUNDLE_PREFIX):
        body = dep[len(BUNDLE_PREFIX):]
        family, sep, bucket = body.partition("/")
        if not sep or not _FAMILY_RE.match(family) or not _BUCKET_RE.match(bucket):
            raise DependencyUnmet(f"bundle dependency must be 'bundle:<family>/<bucket>', got {dep!r}")
        return ParsedDependency("bundle", body)
    if dep.startswith(REGISTER_PREFIX):
        body = dep[len(REGISTER_PREFIX):]
        kind, sep, sha = body.partition("/")
        if not sep or kind not in REGISTER_KINDS or not _SHA_RE.match(sha):
            raise DependencyUnmet(
                f"register dependency must be 'register:<{'|'.join(REGISTER_KINDS)}>/<sha>', got {dep!r}")
        return ParsedDependency("register", body)
    raise DependencyUnmet(
        f"unknown dependency {dep!r}: the vocabulary is {PREFIXES} (no bare ids, no advice)")


def _sha(sha: str) -> str:
    text = str(sha).lower()
    if not _SHA_RE.match(text):
        raise ManifestError(f"a register sha is 8..64 lowercase hex, got {sha!r}")
    return text


def _register_kind(kind: str) -> str:
    if kind not in REGISTER_KINDS:
        raise ManifestError(f"unknown register kind {kind!r}; expected one of {REGISTER_KINDS}")
    return kind


def _family(family: str) -> str:
    if not _FAMILY_RE.match(str(family)):
        raise ManifestError(f"bad family token {family!r}")
    return str(family)


# --------------------------------------------------------------------------- atomic io


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """tmp + fsync + rename + directory fsync (``ard/freeze.py:_atomic_write_bytes``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical bytes of a register body.  Registers hold no floats (ids, hashes, ints), so
    JCS applies; a float is a design error, not something to round."""
    return identity.jcs(dict(payload))


# --------------------------------------------------------------------------- registers


def register_path(run_root: str | os.PathLike, kind: str, sha: str) -> Path:
    return Path(run_root) / REGISTERS_DIR / _register_kind(kind) / _sha(sha) / REGISTER_FILE


def write_register(run_root: str | os.PathLike, kind: str, payload: Mapping[str, Any]) -> str:
    """Seal ``payload`` as ``register:<kind>/<sha>`` and return the sha.

    ``payload["entries"]`` is the sealed set and must be non-empty; the body (payload plus
    ``kind``, minus any ``register_sha256``) is hashed as canonical JSON, the sha is stamped
    into the file, and an identical existing file is left alone (idempotent re-seal).
    """
    kind = _register_kind(kind)
    if not isinstance(payload, Mapping) or not payload:
        raise ManifestError("refusing to seal an empty register")
    entries = payload.get(REGISTER_ENTRIES_FIELD)
    if not isinstance(entries, (list, tuple)) or len(entries) == 0:
        raise ManifestError(
            f"refusing to seal an empty {kind} register: payload['{REGISTER_ENTRIES_FIELD}'] "
            "must be a non-empty list (a register that resolves but contains nothing is the "
            "ard/seal.py failure mode)")
    body = {k: v for k, v in dict(payload).items() if k != REGISTER_SHA_FIELD}
    if body.get("kind", kind) != kind:
        raise ManifestError(f"payload declares kind {body['kind']!r} but is sealed as {kind!r}")
    body["kind"] = kind
    try:
        sha = identity.sha256_hex(_canonical_json_bytes(body))
    except TypeError as exc:
        raise ManifestError(f"{kind} register payload is not canonical-JSON safe ({exc}); registers hold "
                            "ids, hashes and ints only -- a float (a confidence, a weight) is a design "
                            "error, not something to round") from exc
    path = register_path(run_root, kind, sha)
    text = json.dumps({**body, REGISTER_SHA_FIELD: sha}, sort_keys=True, indent=2,
                      ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        status = verify_register(run_root, kind, sha)
        if status.status == STATUS_MET:
            return sha
        raise ManifestError(f"{path} exists but is {status.status}: {status.detail}")
    atomic_write_bytes(path, text.encode("utf-8"))
    return sha


@dataclass(frozen=True)
class RegisterStatus:
    status: str       # met | tampered | missing
    detail: str = ""

    @property
    def met(self) -> bool:
        return self.status == STATUS_MET


def read_register(run_root: str | os.PathLike, kind: str, sha: str) -> Mapping[str, Any]:
    """The verified body of a register (raises :class:`DependencyUnmet` unless met)."""
    status = verify_register(run_root, kind, sha)
    if not status.met:
        raise DependencyUnmet(f"register:{kind}/{sha} is {status.status}: {status.detail}")
    return json.loads(register_path(run_root, kind, sha).read_text(encoding="utf-8"))


def verify_register(run_root: str | os.PathLike, kind: str, sha: str) -> RegisterStatus:
    try:
        path = register_path(run_root, kind, sha)
    except ManifestError as exc:
        return RegisterStatus(STATUS_TAMPERED, str(exc))
    if not path.exists():
        return RegisterStatus(STATUS_MISSING, f"{path} does not exist")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (ValueError, OSError) as exc:
        return RegisterStatus(STATUS_TAMPERED, f"{path}: unreadable ({exc})")
    if not isinstance(payload, dict):
        return RegisterStatus(STATUS_TAMPERED, f"{path}: not a JSON object")
    declared = payload.get(REGISTER_SHA_FIELD)
    if not isinstance(declared, str):
        return RegisterStatus(STATUS_TAMPERED, f"{path}: no {REGISTER_SHA_FIELD}")
    body = {k: v for k, v in payload.items() if k != REGISTER_SHA_FIELD}
    if body.get("kind") != kind:
        return RegisterStatus(STATUS_TAMPERED, f"{path}: sealed as {body.get('kind')!r}, read as {kind!r}")
    entries = body.get(REGISTER_ENTRIES_FIELD)
    if not isinstance(entries, list) or not entries:
        return RegisterStatus(STATUS_TAMPERED, f"{path}: empty register")
    try:
        computed = identity.sha256_hex(_canonical_json_bytes(body))
    except TypeError as exc:
        return RegisterStatus(STATUS_TAMPERED, f"{path}: body is not canonical-JSON safe ({exc})")
    if computed != declared.lower():
        return RegisterStatus(STATUS_TAMPERED,
                              f"{path}: declared {declared[:12]} != content {computed[:12]}")
    if not computed.startswith(sha.lower()):
        return RegisterStatus(STATUS_TAMPERED,
                              f"{path}: content {computed[:12]} does not match the frozen {sha[:12]}")
    return RegisterStatus(STATUS_MET, f"content-addressed, {len(entries)} entries")


# --------------------------------------------------------------------------- bundles


def bundle_path(run_root: str | os.PathLike, family: str, bucket: str | int) -> Path:
    return Path(run_root).joinpath(*BUNDLES_DIR) / _family(family) / f"{bucket_name(bucket)}.json"


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_bundle(run_root: str | os.PathLike, family: str, bucket: str | int,
                 payload: Mapping[str, Any]) -> str:
    """Write a root bundle + ``.sha256`` sidecar; return the sha.  Refuses a differing overwrite.

    The label-free check (no composition, no grades) is ``bundles.assert_label_free`` (WP3) and
    runs before this call; this function only guarantees atomicity and the sidecar.
    """
    if not isinstance(payload, Mapping) or not payload:
        raise ManifestError("refusing to write an empty root bundle")
    path = bundle_path(run_root, family, bucket)
    raw = (json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
           + "\n").encode("utf-8")
    sha = identity.sha256_hex(raw)
    if path.exists():
        existing = _sidecar(path).read_text(encoding="utf-8").split()[0] if _sidecar(path).exists() else ""
        if existing == sha and identity.sha256_hex(path.read_bytes()) == sha:
            return sha
        raise ManifestError(f"{path} exists with different content; a bundle is written once "
                            "(a new team set is a new bucket, never an edit)")
    atomic_write_bytes(path, raw)
    atomic_write_bytes(_sidecar(path), f"{sha}  {path.name}\n".encode("utf-8"))
    return sha


def verify_bundle(run_root: str | os.PathLike, family: str, bucket: str | int) -> RegisterStatus:
    try:
        path = bundle_path(run_root, family, bucket)
    except ManifestError as exc:
        return RegisterStatus(STATUS_TAMPERED, str(exc))
    if not path.exists():
        return RegisterStatus(STATUS_MISSING, f"{path} does not exist")
    sidecar = _sidecar(path)
    if not sidecar.exists():
        return RegisterStatus(STATUS_TAMPERED, f"{sidecar} is missing")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower() if sidecar.read_text(encoding="utf-8").split() else ""
    actual = identity.sha256_hex(path.read_bytes())
    if expected != actual:
        return RegisterStatus(STATUS_TAMPERED, f"sidecar {expected[:12]} != bytes {actual[:12]}")
    return RegisterStatus(STATUS_MET, actual)


def read_bundle(run_root: str | os.PathLike, family: str, bucket: str | int) -> tuple[Mapping[str, Any], str]:
    """``(payload, export_hash)`` of a verified bundle; raises unless met."""
    status = verify_bundle(run_root, family, bucket)
    if not status.met:
        raise DependencyUnmet(f"bundle:{family}/{bucket_name(bucket)} is {status.status}: {status.detail}")
    return json.loads(bundle_path(run_root, family, bucket).read_text(encoding="utf-8")), status.detail


# --------------------------------------------------------------------------- cells


def cell_dir(run_root: str | os.PathLike, cell_id: str) -> Path:
    return Path(run_root) / CELLS_DIR / cell_id


def _cell_status(run_root: str | os.PathLike, cell_id: str) -> RegisterStatus:
    directory = cell_dir(run_root, cell_id)
    meta_path = directory / META_FILE
    if not meta_path.exists():
        return RegisterStatus(STATUS_UNMET, f"{meta_path} does not exist (cell not complete)")
    if (directory / SUSPENDED_FILE).exists():
        return RegisterStatus(STATUS_UNMET, f"{directory / SUSPENDED_FILE} present")
    incomplete = directory / "incomplete"
    if incomplete.exists() and any(incomplete.glob("*.json")):
        return RegisterStatus(STATUS_UNMET, f"{incomplete} holds in-flight items")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return RegisterStatus(STATUS_TAMPERED, f"{meta_path}: not valid JSON ({exc})")
    if not isinstance(meta, dict) or meta.get("cell_id") != cell_id:
        return RegisterStatus(STATUS_TAMPERED,
                              f"{meta_path} claims cell {meta.get('cell_id') if isinstance(meta, dict) else None!r}, "
                              f"directory is {cell_id!r}")
    items = sorted((directory / "items").glob("*.json")) if (directory / "items").exists() else []
    n_items = meta.get("n_items")
    if not isinstance(n_items, int) or n_items != len(items):
        return RegisterStatus(STATUS_TAMPERED,
                              f"{directory}: meta.n_items={n_items!r} but {len(items)} item files present "
                              "(two manifests may share this cell id, defect D-04)")
    if n_items == 0:
        return RegisterStatus(STATUS_TAMPERED, f"{directory}: a complete cell with zero items")
    for item in items:
        try:
            with open(item, "r", encoding="utf-8") as handle:
                claimed = json.load(handle).get("cell_id")
        except (ValueError, OSError, AttributeError) as exc:
            return RegisterStatus(STATUS_TAMPERED, f"{item}: unreadable ({exc})")
        if claimed != cell_id:
            return RegisterStatus(STATUS_TAMPERED,
                                  f"{item.name} belongs to cell {claimed!r}, directory is {cell_id!r}")
    return RegisterStatus(STATUS_MET, f"{n_items} items, directory owned by {cell_id}")


@dataclass(frozen=True)
class DependencyStatus:
    """One resolved dependency (``ard/runner.py:DependencyStatus`` shape, nextdyn vocabulary)."""

    dep: str
    kind: str            # cell | bundle | register
    path: str
    status: str          # met | unmet | tampered
    detail: str = ""

    @property
    def met(self) -> bool:
        return self.status == STATUS_MET


def dependency_status(run_root: str | os.PathLike, dep: str) -> DependencyStatus:
    parsed = parse_dependency(dep)
    if parsed.kind == "cell":
        status = _cell_status(run_root, parsed.value)
        return DependencyStatus(dep, "cell", str(cell_dir(run_root, parsed.value) / META_FILE),
                                status.status, status.detail)
    if parsed.kind == "bundle":
        family, bucket = parsed.parts
        status = verify_bundle(run_root, family, bucket)
        return DependencyStatus(dep, "bundle", str(bundle_path(run_root, family, bucket)),
                                STATUS_UNMET if status.status == STATUS_MISSING else status.status,
                                status.detail)
    kind, sha = parsed.parts
    status = verify_register(run_root, kind, sha)
    return DependencyStatus(dep, "register", str(register_path(run_root, kind, sha)),
                            STATUS_UNMET if status.status == STATUS_MISSING else status.status,
                            status.detail)


def unmet(run_root: str | os.PathLike, deps: Sequence[str]) -> list[DependencyStatus]:
    """Every dependency that is not met, in manifest order (tampered ones included)."""
    return [status for status in (dependency_status(run_root, d) for d in deps) if not status.met]


__all__ = [
    "BUNDLE_PREFIX", "CELL_PREFIX", "DependencyStatus", "ParsedDependency", "PREFIXES",
    "REGISTER_KINDS", "REGISTER_PACKETS", "REGISTER_PANEL", "REGISTER_POOL", "REGISTER_PREFIX",
    "RegisterStatus", "STATUS_MET", "STATUS_MISSING", "STATUS_TAMPERED", "STATUS_UNMET",
    "atomic_write_bytes", "bucket_name", "bundle_dependency", "bundle_path", "cell_dependency",
    "cell_dir", "dependency_status", "parse_dependency", "read_bundle", "read_register",
    "register_dependency", "register_path", "unmet", "verify_bundle", "verify_register",
    "write_bundle", "write_register",
]
