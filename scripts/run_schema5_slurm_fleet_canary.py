#!/usr/bin/env python3
"""Exercise schema-5 admission, dependency failure, and turnover against Slurm.

The default command is a composite canary outside the canonical serving pool.  It
first submits one short CPU-only allocation to prove the admission transaction,
proves the cluster's fail-closed dependency cascade with three held/released jobs,
then performs two real warm turnovers with short CPU-only HTTP endpoints:

* an immutable static intent precedes every other canary artifact;
* :mod:`agents_scaling.serving.fleet_transactions` durably records the admission
  intent before ``sbatch``;
* a crash after scheduler acceptance is recovered by the intent comment, without
  allocating a second job;
* joined ``squeue`` and ``sacct`` truth binds the job id, comment, immutable batch
  path, and ledger;
* ``scontrol write batch_script`` returns the exact submitted bytes;
* every allocation proves effective ``Requeue=0`` before retirement;
* a deliberately failed, initially held root cancels its never-started ``afterok``
  child and makes an aggregate ``afterany`` sentinel run within a fixed bound;
* the dependency proof reparses and seals the live
  ``DependencyParameters=kill_invalid_depend`` scheduler configuration;
* two consecutive dual endpoint probes precede each atomic promotion;
* the routed endpoint remains available while physical overlap stays at two jobs
  and zero GPUs; and
* an immutable exact-id retirement intent precedes ``scancel``; and
* terminal accounting truth is observed before a checksummed, read-only
  ``CANARY_COMPLETE.json`` is published last.

The command is non-mutating unless ``--apply`` is supplied.  Repeating ``--apply``
after completion performs sealed verification only.  ``--verify`` also verifies a
completed canary without contacting Slurm.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for _value in (REPO, SOURCE_ROOT):
    if str(_value) not in sys.path:
        sys.path.insert(0, str(_value))

from agents_scaling.serving import fleet_transactions as tx  # noqa: E402
from scripts import publish_schema5_durable_git_release as durable_git  # noqa: E402

SCHEMA_VERSION = 4
INTENT_FILENAME = "CANARY_INTENT.json"
ACTIVE_BINDING_FILENAME = "SCHEDULER_ACTIVE_BINDING.json"
SPOOLED_SCRIPT_FILENAME = "SPOOLED_BATCH_SCRIPT.sbatch"
EFFECTIVE_JOB_STATE_FILENAME = "EFFECTIVE_JOB_STATE.json"
RETIREMENT_INTENT_FILENAME = "RETIREMENT_INTENT.json"
RETIREMENT_ACCEPTED_FILENAME = "RETIREMENT_ACCEPTED.json"
TERMINAL_BINDING_FILENAME = "SCHEDULER_TERMINAL_BINDING.json"
COMPLETE_FILENAME = "CANARY_COMPLETE.json"
COMPOSITE_INTENT_FILENAME = "COMPOSITE_CANARY_INTENT.json"
TRANSACTION_COMPONENT_DIRECTORY = "transaction"
DEPENDENCY_COMPONENT_DIRECTORY = "dependency-cascade"
TURNOVER_COMPONENT_DIRECTORY = "turnover"
DEFAULT_PARTITION = "mit_normal"
DEFAULT_TIME_LIMIT = "00:08:00"
DEFAULT_VISIBILITY_TIMEOUT = 180.0
DEFAULT_TERMINAL_TIMEOUT = 180.0
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_TURNOVER_CYCLES = 2
DEFAULT_TURNOVER_DRAIN_SECONDS = 5.0
DEFAULT_DEPENDENCY_ALERT_LATENCY_SECONDS = 180.0
DEPENDENCY_INTENT_FILENAME = "DEPENDENCY_CASCADE_INTENT.json"
DEPENDENCY_RECEIPT_FILENAME = "DEPENDENCY_CASCADE_SUBMISSION.json"
DEPENDENCY_RELEASE_INTENT_FILENAME = "ROOT_RELEASE_INTENT.json"
DEPENDENCY_RELEASE_COMPLETE_FILENAME = "ROOT_RELEASE_COMPLETE.json"
DEPENDENCY_TERMINAL_EVIDENCE_FILENAME = "TERMINAL_SCHEDULER_EVIDENCE.json"
DEPENDENCY_ALERT_MARKER_FILENAME = "ALERT_EXECUTED.json"
DEPENDENCY_FORBIDDEN_CHILD_MARKER_FILENAME = "CHILD_STARTED.INVALID"
DEPENDENCY_COMPLETE_FILENAME = "DEPENDENCY_CASCADE_COMPLETE.json"
PRODUCTION_HANDOFF_DRAIN_SECONDS = 660
TURNOVER_INTENT_FILENAME = "TURNOVER_INTENT.json"
TURNOVER_POINTER_FILENAME = "PROMOTED_ENDPOINT.json"
TURNOVER_COMPLETE_FILENAME = "TURNOVER_COMPLETE.json"
TURNOVER_PROFILE = "turnover-canary"
TURNOVER_PORT_DERIVATION_PROTOCOL = (
    "schema5-v1.2-r2-turnover-run-token-port-derivation-v1"
)
# Use a broad unprivileged namespace instead of one fixed three-port tuple.  The
# complete marker-first 128-bit run token and allocation index select the initial
# port.  Deterministic open addressing makes the ports unique within one plan even
# if two truncated hashes collide, while preserving exact crash/restart replay.
TURNOVER_PORT_MIN = 20_000
TURNOVER_PORT_MAX = 64_999
TURNOVER_MAX_ALLOCATIONS = 5
ROLLOUT_GENERATION = 1
PROFILE = "canary-cpu"
REQUIRED_RELEASE_TAG = "sweep-recovery-schema5-v1.2-r2"
CANARY_GIT_PATH = "scripts/run_schema5_slurm_fleet_canary.py"
FLEET_TRANSACTIONS_GIT_PATH = (
    "src/agents_scaling/serving/fleet_transactions.py"
)
DURABLE_GIT_PUBLISHER_GIT_PATH = (
    "scripts/publish_schema5_durable_git_release.py"
)
CODE_IDENTITY_PROTOCOL = "schema5-v1.2-r2-slurm-canary-code-v3"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}\Z")
_JOB_NAME_RE = re.compile(r"asys-s5-serve-[A-Za-z0-9_.-]{1,96}\Z")


def _default_canary_root() -> Path:
    return (
        REPO.parent
        / "agents_scaling_results"
        / "recovery"
        / "schema5-v1"
        / "slurm_canaries"
        / "schema5-v1.2-r2"
    )


def _default_turnover_root() -> Path:
    return _default_canary_root() / TURNOVER_COMPONENT_DIRECTORY


class SlurmFleetCanaryError(RuntimeError):
    """The canary cannot proceed without weakening transaction guarantees."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SlurmFleetCanaryError(
            f"cannot open canary artifact {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SlurmFleetCanaryError(f"canary artifact is not regular: {path}")
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise SlurmFleetCanaryError(f"canary artifact changed while hashing: {path}")
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO), *arguments),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise SlurmFleetCanaryError(f"cannot execute Git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise SlurmFleetCanaryError(
            f"Git command failed ({completed.returncode}): {arguments!r}: {detail}"
        )
    return completed.stdout.strip()


def _current_code_files() -> dict[str, Path]:
    canary_path = Path(__file__).resolve()
    transaction_path = Path(tx.__file__).resolve()
    expected = {
        CANARY_GIT_PATH: (REPO / CANARY_GIT_PATH).resolve(),
        FLEET_TRANSACTIONS_GIT_PATH: (
            REPO / FLEET_TRANSACTIONS_GIT_PATH
        ).resolve(),
        DURABLE_GIT_PUBLISHER_GIT_PATH: (
            REPO / DURABLE_GIT_PUBLISHER_GIT_PATH
        ).resolve(),
    }
    observed = {
        CANARY_GIT_PATH: canary_path,
        FLEET_TRANSACTIONS_GIT_PATH: transaction_path,
        DURABLE_GIT_PUBLISHER_GIT_PATH: Path(
            durable_git.__file__
        ).resolve(),
    }
    if observed != expected:
        raise SlurmFleetCanaryError(
            "canary code was imported outside its repository checkout"
        )
    for git_path, path in observed.items():
        if path.is_symlink() or not path.is_file():
            raise SlurmFleetCanaryError(
                f"canary code source is missing or unsafe: {git_path}={path}"
            )
    return observed


def _validate_code_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "canary_script",
        "fleet_transactions",
        "durable_git_publisher",
        "durable_git_release",
    }
    if not isinstance(identity, Mapping) or set(identity) != required:
        raise SlurmFleetCanaryError("canary code identity fields drifted")
    files = _current_code_files()
    records = {
        "canary_script": CANARY_GIT_PATH,
        "fleet_transactions": FLEET_TRANSACTIONS_GIT_PATH,
        "durable_git_publisher": DURABLE_GIT_PUBLISHER_GIT_PATH,
    }
    if (
        identity.get("schema_version") != 1
        or identity.get("protocol") != CODE_IDENTITY_PROTOCOL
        or identity.get("release_tag") != REQUIRED_RELEASE_TAG
        or _COMMIT_RE.fullmatch(str(identity.get("release_git_commit", "")))
        is None
        or _COMMIT_RE.fullmatch(str(identity.get("release_tag_object", "")))
        is None
    ):
        raise SlurmFleetCanaryError("canary release code identity is invalid")
    for field, git_path in records.items():
        record = identity.get(field)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"git_path", "sha256", "size"}
            or record.get("git_path") != git_path
            or record.get("sha256") != _sha256_file(files[git_path])
            or record.get("size") != files[git_path].stat().st_size
        ):
            raise SlurmFleetCanaryError(
                f"canary code identity drifted for {git_path}"
            )
    durable_binding = identity.get("durable_git_release")
    if (
        not isinstance(durable_binding, Mapping)
        or set(durable_binding)
        != {
            "path",
            "sha256",
            "marker_id",
            "release_git_commit",
            "release_tag_object",
            "bundle_sha256",
        }
        or not isinstance(durable_binding.get("path"), str)
        or not Path(durable_binding["path"]).is_absolute()
        or any(
            _SHA256_RE.fullmatch(str(durable_binding.get(field, ""))) is None
            for field in ("sha256", "marker_id", "bundle_sha256")
        )
        or durable_binding.get("release_git_commit")
        != identity.get("release_git_commit")
        or durable_binding.get("release_tag_object")
        != identity.get("release_tag_object")
    ):
        raise SlurmFleetCanaryError(
            "canary durable Git release binding is invalid"
        )
    return json.loads(json.dumps(identity, sort_keys=True))


def derive_code_identity(
    durable_git_release_marker: Path | None = None,
) -> dict[str, Any]:
    """Bind a production canary to the exact clean annotated r2 checkout."""

    files = _current_code_files()
    tag_ref = f"refs/tags/{REQUIRED_RELEASE_TAG}"
    if _git("cat-file", "-t", tag_ref) != "tag":
        raise SlurmFleetCanaryError(
            f"{REQUIRED_RELEASE_TAG} must be an annotated tag"
        )
    tag_object = _git("rev-parse", "--verify", tag_ref)
    tag_commit = _git("rev-parse", "--verify", f"{tag_ref}^{{commit}}")
    head = _git("rev-parse", "--verify", "HEAD")
    if (
        _COMMIT_RE.fullmatch(tag_object) is None
        or _COMMIT_RE.fullmatch(tag_commit) is None
        or head != tag_commit
        or _git("status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise SlurmFleetCanaryError(
            "production canary requires the exact clean r2 tagged checkout"
        )
    if durable_git_release_marker is None:
        raise SlurmFleetCanaryError(
            "production canary requires the durable Git release marker"
        )
    try:
        durable_binding = durable_git.marker_binding(
            durable_git_release_marker
        )
    except durable_git.DurableGitReleaseError as exc:
        raise SlurmFleetCanaryError(str(exc)) from exc
    if (
        durable_binding["release_git_commit"] != tag_commit
        or durable_binding["release_tag_object"] != tag_object
    ):
        raise SlurmFleetCanaryError(
            "durable Git release marker belongs to another release identity"
        )
    identity = {
        "schema_version": 1,
        "protocol": CODE_IDENTITY_PROTOCOL,
        "release_tag": REQUIRED_RELEASE_TAG,
        "release_git_commit": tag_commit,
        "release_tag_object": tag_object,
        "canary_script": {
            "git_path": CANARY_GIT_PATH,
            "sha256": _sha256_file(files[CANARY_GIT_PATH]),
            "size": files[CANARY_GIT_PATH].stat().st_size,
        },
        "fleet_transactions": {
            "git_path": FLEET_TRANSACTIONS_GIT_PATH,
            "sha256": _sha256_file(files[FLEET_TRANSACTIONS_GIT_PATH]),
            "size": files[FLEET_TRANSACTIONS_GIT_PATH].stat().st_size,
        },
        "durable_git_publisher": {
            "git_path": DURABLE_GIT_PUBLISHER_GIT_PATH,
            "sha256": _sha256_file(files[DURABLE_GIT_PUBLISHER_GIT_PATH]),
            "size": files[DURABLE_GIT_PUBLISHER_GIT_PATH].stat().st_size,
        },
        "durable_git_release": durable_binding,
    }
    return _validate_code_identity(identity)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SlurmFleetCanaryError(f"{description} is missing or unsafe: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SlurmFleetCanaryError(
            f"cannot parse {description} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SlurmFleetCanaryError(f"{description} must be one JSON object")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable_once(
    path: Path,
    payload: Mapping[str, Any] | bytes,
    *,
    description: str,
    mode: int = 0o444,
) -> None:
    encoded = (
        bytes(payload) if isinstance(payload, bytes) else _canonical_bytes(payload)
    )
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise SlurmFleetCanaryError(
                f"existing {description} differs from the durable transaction: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError(f"existing {description} is writable: {path}")
        return
    if path.parent.is_symlink():
        raise SlurmFleetCanaryError(f"{description} parent is symlinked: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_pool_root() -> Path:
    return (
        REPO.parent / "agents_scaling_results" / "server_pools" / "schema5-v1"
    ).resolve()


def _validate_isolated_root(root: Path) -> Path:
    supplied = root.expanduser()
    if supplied.is_symlink():
        raise SlurmFleetCanaryError(f"canary root is symlinked: {supplied}")
    resolved = supplied.resolve()
    canonical = _canonical_pool_root()
    try:
        resolved.relative_to(canonical)
    except ValueError:
        pass
    else:
        raise SlurmFleetCanaryError(
            f"canary root must not touch canonical server pool {canonical}"
        )
    try:
        canonical.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise SlurmFleetCanaryError(
            f"canary root may not contain canonical server pool {canonical}"
        )
    if "/server_pools/schema5-v1" in resolved.as_posix():
        raise SlurmFleetCanaryError("canary root aliases the canonical pool namespace")
    return resolved


def _job_script(*, job_name: str, partition: str, time_limit: str) -> str:
    if not _JOB_NAME_RE.fullmatch(job_name):
        raise SlurmFleetCanaryError(f"unsafe canary job name: {job_name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
        raise SlurmFleetCanaryError(f"unsafe Slurm partition: {partition!r}")
    match = re.fullmatch(r"00:0([1-9]):00", time_limit)
    if match is None:
        raise SlurmFleetCanaryError(
            "canary time limit must be a whole number of minutes from 1 through 9"
        )
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time_limit}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=128M\n"
        "#SBATCH --no-requeue\n"
        "#SBATCH --output=/dev/null\n"
        "#SBATCH --error=/dev/null\n"
        "set -euo pipefail\n"
        "trap 'exit 0' TERM INT\n"
        "while :; do\n"
        "  sleep 30\n"
        "done\n"
    )


def _turnover_job_script(
    *,
    job_name: str,
    partition: str,
    time_limit: str,
    port: int,
) -> str:
    """Render a real short-walltime Slurm allocation with two health endpoints."""

    if (
        not isinstance(port, int)
        or isinstance(port, bool)
        or not TURNOVER_PORT_MIN <= port <= TURNOVER_PORT_MAX
    ):
        raise SlurmFleetCanaryError(
            "turnover port must be an integer inside the frozen unprivileged range"
        )
    # Reuse the strict resource/time/name validation from the transaction canary.
    base = _job_script(
        job_name=job_name,
        partition=partition,
        time_limit=time_limit,
    )
    header = base.split("set -euo pipefail\n", 1)[0]
    return (
        header
        + "set -euo pipefail\n"
        + f"PORT={port}\n"
        + "python3 - \"$PORT\" <<'PY'\n"
        + "import json\n"
        + "import sys\n"
        + "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        + "class Handler(BaseHTTPRequestHandler):\n"
        + "    def do_GET(self):\n"
        + "        if self.path == '/health':\n"
        + "            body = b'ok\\n'\n"
        + "        elif self.path == '/v1/models':\n"
        + "            body = json.dumps({'data': [{'id': 'turnover-canary'}]}).encode()\n"
        + "        else:\n"
        + "            self.send_response(404); self.end_headers(); return\n"
        + "        self.send_response(200)\n"
        + "        self.send_header('Content-Type', 'application/json')\n"
        + "        self.send_header('Content-Length', str(len(body)))\n"
        + "        self.end_headers(); self.wfile.write(body)\n"
        + "    def log_message(self, *_args): pass\n"
        + "ThreadingHTTPServer(('0.0.0.0', int(sys.argv[1])), Handler).serve_forever()\n"
        + "PY\n"
    )


def _turnover_initial_port(*, run_token: str, allocation_index: int) -> int:
    """Map the complete run token and allocation index to one candidate port."""

    if _TOKEN_RE.fullmatch(str(run_token)) is None:
        raise SlurmFleetCanaryError(
            "turnover run token must be 32 lowercase hex digits"
        )
    if (
        not isinstance(allocation_index, int)
        or isinstance(allocation_index, bool)
        or not 0 <= allocation_index < TURNOVER_MAX_ALLOCATIONS
    ):
        raise SlurmFleetCanaryError("turnover allocation index is out of range")
    material = (
        f"{TURNOVER_PORT_DERIVATION_PROTOCOL}\0{run_token}\0"
        f"{allocation_index}"
    ).encode("ascii")
    digest = hashlib.sha256(material).digest()
    span = TURNOVER_PORT_MAX - TURNOVER_PORT_MIN + 1
    return TURNOVER_PORT_MIN + int.from_bytes(digest[:8], "big") % span


def _derive_turnover_ports(
    *, run_token: str, allocation_count: int
) -> list[int]:
    """Derive unique, deterministic ports with bounded collision resolution."""

    if _TOKEN_RE.fullmatch(str(run_token)) is None:
        raise SlurmFleetCanaryError(
            "turnover run token must be 32 lowercase hex digits"
        )
    if (
        not isinstance(allocation_count, int)
        or isinstance(allocation_count, bool)
        or not 1 <= allocation_count <= TURNOVER_MAX_ALLOCATIONS
    ):
        raise SlurmFleetCanaryError(
            "turnover allocation count is outside the frozen port contract"
        )
    span = TURNOVER_PORT_MAX - TURNOVER_PORT_MIN + 1
    selected: list[int] = []
    used: set[int] = set()
    for allocation_index in range(allocation_count):
        initial = _turnover_initial_port(
            run_token=run_token,
            allocation_index=allocation_index,
        )
        for offset in range(span):
            candidate = TURNOVER_PORT_MIN + (
                initial - TURNOVER_PORT_MIN + offset
            ) % span
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                break
        else:  # pragma: no cover - impossible under the bounded count contract
            raise SlurmFleetCanaryError(
                "turnover deterministic port namespace is exhausted"
            )
    return selected


def render_turnover_canary_plan(
    *,
    partition: str = DEFAULT_PARTITION,
    time_limit: str = DEFAULT_TIME_LIMIT,
    cycles: int = DEFAULT_TURNOVER_CYCLES,
    drain_seconds: float = DEFAULT_TURNOVER_DRAIN_SECONDS,
    run_token: str = "0" * 32,
) -> dict[str, Any]:
    """Render a real, executable short-walltime warm-turnover Slurm plan."""

    if not isinstance(cycles, int) or isinstance(cycles, bool) or not 2 <= cycles <= 4:
        raise SlurmFleetCanaryError("turnover cycles must be an integer from 2 through 4")
    if (
        not isinstance(drain_seconds, (int, float))
        or isinstance(drain_seconds, bool)
        or not 0 < float(drain_seconds) < 60
    ):
        raise SlurmFleetCanaryError(
            "turnover canary drain must be positive and shorter than 60 seconds"
        )
    if _TOKEN_RE.fullmatch(str(run_token)) is None:
        raise SlurmFleetCanaryError("turnover run token must be 32 lowercase hex digits")
    suffix = str(run_token)[:12]
    pool_id = f"schema5-v12-turnover-{suffix}"
    replica_id = f"schema5-v12-turnover-{suffix}"
    job_name = f"asys-s5-serve-turnover-{suffix}"
    ports = _derive_turnover_ports(
        run_token=str(run_token),
        allocation_count=cycles + 1,
    )
    port_derivation = {
        "protocol": TURNOVER_PORT_DERIVATION_PROTOCOL,
        "source": "complete_marker_first_128_bit_run_token+allocation_index",
        "minimum_port": TURNOVER_PORT_MIN,
        "maximum_port": TURNOVER_PORT_MAX,
        "collision_resolution": "ascending_wrap_within_plan",
    }
    contract = {
        "schema_version": 1,
        "kind": "schema5_warm_handoff_turnover_canary_contract",
        "pool_id": pool_id,
        "replica_id": replica_id,
        "profile": TURNOVER_PROFILE,
        "job_name": job_name,
        "partition": partition,
        "time_limit": time_limit,
        "cycles": cycles,
        "ports": ports,
        "port_derivation": port_derivation,
        "cpus_per_allocation": 1,
        "gpus_per_allocation": 0,
        "no_requeue": True,
        "canary_drain_seconds": float(drain_seconds),
        "production_drain_seconds": PRODUCTION_HANDOFF_DRAIN_SECONDS,
    }
    fleet_sha256 = _sha256_bytes(_canonical_compact_bytes(contract))
    allocations: list[dict[str, Any]] = []
    for index in range(cycles + 1):
        script = _turnover_job_script(
            job_name=job_name,
            partition=partition,
            time_limit=time_limit,
            port=ports[index],
        )
        token = hashlib.sha256(
            f"{run_token}\0{partition}\0{time_limit}\0{cycles}\0{index}".encode(
                "utf-8"
            )
        ).hexdigest()[:32]
        comment = tx.intent_comment(
            pool_id=pool_id,
            profile=TURNOVER_PROFILE,
            replica_id=replica_id,
            rollout_generation=ROLLOUT_GENERATION,
            intent_token=token,
            fleet_sha256=fleet_sha256,
        )
        allocations.append(
            {
                "index": index,
                "role": "primary" if index == 0 else "standby",
                "job_name": job_name,
                "scheduler_comment": comment,
                "time_limit": time_limit,
                "port": ports[index],
                "script": script,
                "script_sha256": _sha256_bytes(script.encode("utf-8")),
                "submit_argv": [
                    "sbatch",
                    "--parsable",
                    f"--comment={comment}",
                    f"<immutable-script-c{index:02d}>",
                ],
                "effective_state_argv": [
                    "scontrol",
                    "show",
                    "job",
                    "-o",
                    f"<job-id-c{index:02d}>",
                ],
            }
        )
    transitions = [
        {
            "cycle": cycle,
            "predecessor_index": cycle - 1,
            "standby_index": cycle,
            "gates": [
                "complete squeue+sacct intent reconciliation",
                "effective scontrol Requeue=0 receipt",
                "exact spooled-script hash",
                "first simultaneous /health and /v1/models success",
                "second simultaneous /health and /v1/models success",
                "atomic promoted-pointer replacement",
                f"{float(drain_seconds):g}-second canary predecessor drain",
                "exact-ID predecessor scancel",
                "terminal sacct truth",
            ],
        }
        for cycle in range(1, cycles + 1)
    ]
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "kind": "schema5_warm_handoff_turnover_canary_plan",
        "render_only": True,
        "submitted": False,
        "run_token": str(run_token),
        "pool_id": pool_id,
        "fleet_sha256": fleet_sha256,
        "replica_id": replica_id,
        "profile": TURNOVER_PROFILE,
        "job_name": job_name,
        "partition": partition,
        "time_limit": time_limit,
        "cycles": cycles,
        "logical_replicas": 1,
        "maximum_physical_allocations": 2,
        "maximum_overlap_gpus": 0,
        "readiness_probes_per_promotion": 2,
        "canary_drain_seconds": float(drain_seconds),
        "production_drain_seconds": PRODUCTION_HANDOFF_DRAIN_SECONDS,
        "port_derivation": port_derivation,
        "allocations": allocations,
        "transitions": transitions,
        "acceptance": [
            "one promoted logical endpoint at every stable boundary",
            "no more than one typed standby overlap",
            "no coordinate routes to a standby before atomic promotion",
            "every allocation proves effective Requeue=0",
            "two complete turnover cycles without duplicate submission",
        ],
    }
    envelope["plan_sha256"] = _sha256_bytes(_canonical_compact_bytes(envelope))
    return envelope


def _validate_turnover_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise SlurmFleetCanaryError("turnover plan must be one JSON object")
    try:
        expected = render_turnover_canary_plan(
            partition=str(plan["partition"]),
            time_limit=str(plan["time_limit"]),
            cycles=int(plan["cycles"]),
            drain_seconds=float(plan["canary_drain_seconds"]),
            run_token=str(plan["run_token"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SlurmFleetCanaryError(f"turnover plan is malformed: {exc}") from exc
    observed = json.loads(json.dumps(plan, sort_keys=True))
    if observed != expected:
        raise SlurmFleetCanaryError("turnover plan identity or script bytes drifted")
    return expected


def _load_or_create_turnover_intent(
    *,
    root: Path,
    partition: str,
    time_limit: str,
    cycles: int,
    drain_seconds: float,
    code_identity: Mapping[str, Any],
    token_factory: Callable[[], str],
    now: float,
) -> dict[str, Any]:
    path = root / TURNOVER_INTENT_FILENAME
    if path.exists() or path.is_symlink():
        intent = _read_json(path, description="warm-turnover marker-first intent")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError("warm-turnover intent remains writable")
        required = {
            "schema_version",
            "kind",
            "created_at",
            "canary_root",
            "plan",
            "code_identity",
        }
        if (
            set(intent) != required
            or intent.get("schema_version") != SCHEMA_VERSION
            or intent.get("kind")
            != "schema5_warm_handoff_turnover_canary_intent"
            or intent.get("canary_root") != str(root)
            or not isinstance(intent.get("created_at"), (int, float))
            or isinstance(intent.get("created_at"), bool)
            or _validate_turnover_plan(intent.get("plan", {})) != intent["plan"]
            or _validate_code_identity(intent.get("code_identity", {}))
            != intent["code_identity"]
        ):
            raise SlurmFleetCanaryError("warm-turnover intent is invalid")
        plan = intent["plan"]
        if (
            plan["partition"] != partition
            or plan["time_limit"] != time_limit
            or plan["cycles"] != cycles
            or plan["canary_drain_seconds"] != float(drain_seconds)
            or intent["code_identity"] != code_identity
        ):
            raise SlurmFleetCanaryError(
                "requested turnover contract differs from marker-first intent"
            )
        return intent
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise SlurmFleetCanaryError(f"turnover canary root is unsafe: {root}")
        entries = list(root.iterdir())
        if entries:
            raise SlurmFleetCanaryError(
                "new turnover root contains artifacts before marker-first intent: "
                + ", ".join(sorted(item.name for item in entries))
            )
    else:
        root.mkdir(parents=True)
    run_token = token_factory()
    if not isinstance(run_token, str) or _TOKEN_RE.fullmatch(run_token) is None:
        raise SlurmFleetCanaryError(
            "turnover token factory returned an invalid token"
        )
    plan = render_turnover_canary_plan(
        partition=partition,
        time_limit=time_limit,
        cycles=cycles,
        drain_seconds=drain_seconds,
        run_token=run_token,
    )
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_turnover_canary_intent",
        "created_at": float(now),
        "canary_root": str(root),
        "plan": plan,
        "code_identity": dict(code_identity),
    }
    _write_immutable_once(
        path,
        intent,
        description="warm-turnover marker-first intent",
    )
    return intent


def _write_atomic_turnover_pointer(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically install the one routable endpoint; immutable receipts preserve history."""

    encoded = _canonical_bytes(payload)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise SlurmFleetCanaryError("turnover promoted pointer is unsafe")
        if path.read_bytes() == encoded:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _turnover_pointer_payload(
    *,
    plan: Mapping[str, Any],
    allocation: Mapping[str, Any],
    job_id: str,
    node: str,
    promoted_at: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_promoted_endpoint",
        "pool_id": plan["pool_id"],
        "fleet_sha256": plan["fleet_sha256"],
        "replica_id": plan["replica_id"],
        "profile": plan["profile"],
        "allocation_index": allocation["index"],
        "intent_token": allocation["scheduler_comment"].split(";intent=", 1)[1].split(
            ";", 1
        )[0],
        "job_id": str(job_id),
        "host": str(node),
        "port": allocation["port"],
        "endpoint": f"http://{node}:{allocation['port']}",
        "promoted_at": float(promoted_at),
    }


def _default_endpoint_probe(
    endpoint: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Probe both serving endpoints and return hash-bound observations."""

    results: dict[str, Any] = {}
    for path in ("/health", "/v1/models"):
        url = endpoint.rstrip("/") + path
        try:
            with urllib_request.urlopen(url, timeout=timeout) as response:
                status = int(response.status)
                body = response.read(1024 * 1024)
        except (OSError, urllib_error.URLError) as exc:
            raise SlurmFleetCanaryError(
                f"turnover endpoint probe failed for {url}: {exc}"
            ) from exc
        if status != 200:
            raise SlurmFleetCanaryError(
                f"turnover endpoint {url} returned HTTP {status}"
            )
        results[path] = {
            "status": status,
            "size": len(body),
            "sha256": _sha256_bytes(body),
        }
        if path == "/v1/models":
            try:
                model_payload = json.loads(body.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise SlurmFleetCanaryError(
                    f"turnover models response is invalid JSON: {exc}"
                ) from exc
            if model_payload != {"data": [{"id": "turnover-canary"}]}:
                raise SlurmFleetCanaryError(
                    "turnover models response has unexpected identity"
                )
    return {
        "endpoint": endpoint,
        "health": results["/health"],
        "models": results["/v1/models"],
    }


EndpointProbe = Callable[..., Mapping[str, Any]]


def _probe_turnover_pair(
    *,
    root: Path,
    cycle: int,
    sequence: int,
    predecessor_pointer: Mapping[str, Any],
    standby_pointer: Mapping[str, Any],
    probe: EndpointProbe,
    timeout: float,
    now: float,
) -> dict[str, Any]:
    predecessor = dict(
        probe(str(predecessor_pointer["endpoint"]), timeout=timeout)
    )
    standby = dict(probe(str(standby_pointer["endpoint"]), timeout=timeout))
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_dual_probe",
        "captured_at": float(now),
        "cycle": cycle,
        "sequence": sequence,
        "routed_job_id": predecessor_pointer["job_id"],
        "standby_job_id": standby_pointer["job_id"],
        "promoted_pointer_sha256": _sha256_file(
            root / TURNOVER_POINTER_FILENAME
        ),
        "predecessor": predecessor,
        "standby": standby,
    }
    path = root / "transitions" / f"c{cycle:02d}" / f"PROBE_{sequence}.json"
    _write_immutable_once(path, record, description="turnover dual-probe receipt")
    return record


def _turnover_scheduler_rows(
    snapshot: tx.SchedulerSnapshot,
    plan: Mapping[str, Any],
) -> tuple[tx.SchedulerRow, ...]:
    selected: list[tx.SchedulerRow] = []
    for row in snapshot.rows:
        parsed = tx.parse_intent_comment(row.comment)
        exact_name = row.job_name == plan["job_name"]
        related = parsed is not None and (
            parsed["pool"] == plan["pool_id"]
            or parsed["replica"] == plan["replica_id"]
            or parsed["fleet"] == plan["fleet_sha256"]
        )
        if not exact_name and not related:
            continue
        if (
            parsed is None
            or row.job_name != plan["job_name"]
            or parsed["pool"] != plan["pool_id"]
            or parsed["profile"] != plan["profile"]
            or parsed["replica"] != plan["replica_id"]
            or parsed["generation"] != str(ROLLOUT_GENERATION)
            or parsed["fleet"] != plan["fleet_sha256"]
        ):
            raise SlurmFleetCanaryError(
                "foreign or malformed scheduler allocation overlaps turnover "
                f"identity: job {row.job_id}"
            )
        selected.append(row)
    return tuple(selected)


def _turnover_reconcile(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    runner: Runner,
    scheduler_user: str | None,
    now: float,
) -> tuple[tx.SchedulerSnapshot, tx.FleetReconciliation]:
    snapshot = tx.query_scheduler(runner=runner, user=scheduler_user, now=now)
    rows = _turnover_scheduler_rows(snapshot, plan)
    try:
        reconciled = tx.reconcile_scheduler_rows(
            rows,
            [ledger],
            pool_id=str(plan["pool_id"]),
            fleet_sha256=str(plan["fleet_sha256"]),
            replica_profiles={str(plan["replica_id"]): str(plan["profile"])},
            replica_job_names={
                str(plan["replica_id"]): str(plan["job_name"])
            },
        )
    except tx.FleetTransactionError as exc:
        raise SlurmFleetCanaryError(str(exc)) from exc
    active = reconciled.active_allocations
    if len(active) > 2:
        raise SlurmFleetCanaryError(
            "turnover canary exceeded two physical allocations"
        )
    return snapshot, reconciled


def _turnover_allocation_for_token(
    reconciliation: tx.FleetReconciliation,
    token: str,
) -> tx.ReconciledFleetAllocation | None:
    matches = [
        item
        for item in reconciliation.allocations
        if item.attempt.get("intent_token") == token
    ]
    if len(matches) > 1:
        raise SlurmFleetCanaryError(
            f"turnover intent {token} maps to multiple scheduler jobs"
        )
    return None if not matches else matches[0]


def _turnover_adopt(
    *,
    directory: Path,
    ledger: dict[str, Any],
    plan: Mapping[str, Any],
    attempt: dict[str, Any],
    allocation: tx.ReconciledFleetAllocation,
    now: float,
) -> None:
    row = allocation.row
    limit_match = re.fullmatch(r"00:0([1-9]):00", str(plan["time_limit"]))
    if limit_match is None:
        raise SlurmFleetCanaryError("turnover plan walltime is invalid")
    expected_limit = int(limit_match.group(1)) * 60
    if (
        row.partition != plan["partition"]
        or row.job_name != plan["job_name"]
        or row.time_limit_seconds != expected_limit
        or (not tx.terminal_state(row.state) and row.dependency != "")
        or (
            not tx.terminal_state(row.state)
            and (row.start_timestamp is None or row.end_timestamp is None)
        )
    ):
        raise SlurmFleetCanaryError(
            f"turnover job {row.job_id} scheduler contract drifted"
        )
    if attempt["job_id"] not in {None, row.job_id}:
        raise SlurmFleetCanaryError("turnover intent changed exact job identity")
    attempt["job_id"] = row.job_id
    attempt["submitted_at"] = attempt["submitted_at"] or float(now)
    attempt["last_seen_at"] = float(now)
    attempt["missing_since"] = None
    attempt["scheduler_start_at"] = row.start_timestamp
    attempt["scheduler_end_at"] = row.end_timestamp
    attempt["scheduler_time_limit_seconds"] = row.time_limit_seconds
    if tx.terminal_state(row.state):
        attempt["state"] = "terminal"
        attempt["terminal_at"] = attempt["terminal_at"] or float(now)
    else:
        attempt["state"] = "committed"
        attempt["committed_at"] = attempt["committed_at"] or float(now)
    tx.save_ledger(directory, ledger, now=now)


def _wait_turnover_allocation(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    token: str,
    runner: Runner,
    scheduler_user: str | None,
    timeout: float,
    poll_seconds: float,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    require_terminal: bool = False,
) -> tuple[
    tx.SchedulerSnapshot,
    tx.FleetReconciliation,
    tx.ReconciledFleetAllocation,
]:
    deadline = now_fn() + timeout
    while True:
        snapshot, reconciled = _turnover_reconcile(
            plan=plan,
            ledger=ledger,
            runner=runner,
            scheduler_user=scheduler_user,
            now=now_fn(),
        )
        allocation = _turnover_allocation_for_token(reconciled, token)
        if allocation is not None:
            terminal = tx.terminal_state(allocation.row.state)
            running = allocation.row.state.upper().startswith("RUNNING")
            node_ready = allocation.row.node not in {"", "(null)", "None", "N/A"}
            if (require_terminal and terminal) or (
                not require_terminal and running and node_ready
            ):
                return snapshot, reconciled, allocation
            if terminal and not require_terminal:
                raise SlurmFleetCanaryError(
                    f"turnover allocation {allocation.row.job_id} terminated "
                    "before readiness"
                )
        if now_fn() >= deadline:
            target = "terminal" if require_terminal else "running"
            raise SlurmFleetCanaryError(
                f"turnover allocation {token} did not become {target}"
            )
        sleep_fn(poll_seconds)


def _ensure_turnover_attempt_running(
    *,
    directory: Path,
    ledger: dict[str, Any],
    plan: Mapping[str, Any],
    allocation_contract: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
    runner: Runner,
    scheduler_user: str | None,
    visibility_timeout: float,
    poll_seconds: float,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any], tx.ReconciledFleetAllocation, int]:
    attempts = ledger["replicas"][plan["replica_id"]]["attempts"]
    index = int(allocation_contract["index"])
    token = tx.parse_intent_comment(
        str(allocation_contract["scheduler_comment"])
    )["intent"]
    matching = [item for item in attempts if item["intent_token"] == token]
    if len(matching) > 1:
        raise SlurmFleetCanaryError("turnover ledger duplicates an intent token")
    if matching:
        attempt = matching[0]
    else:
        if any(
            item["intent_token"]
            == tx.parse_intent_comment(
                str(later["scheduler_comment"])
            )["intent"]
            for later in plan["allocations"][index + 1 :]
            for item in attempts
        ):
            raise SlurmFleetCanaryError(
                "turnover ledger skipped an earlier allocation"
            )
        attempt = tx.prepare_attempt(
            directory,
            ledger,
            replica_id=str(plan["replica_id"]),
            profile=str(plan["profile"]),
            pool_id=str(plan["pool_id"]),
            fleet_sha256=str(plan["fleet_sha256"]),
            rollout_generation=ROLLOUT_GENERATION,
            sbatch_text=str(allocation_contract["script"]),
            now=now_fn(),
            token_factory=lambda: token,
            launch_kind="primary" if predecessor is None else "handoff",
            predecessor_job_id=(
                None if predecessor is None else str(predecessor["job_id"])
            ),
            predecessor_end_at=(
                None
                if predecessor is None
                else float(predecessor["scheduler_end_at"])
            ),
            # This isolated turnover canary launches CPU-only HTTP endpoints.
            # Recording one GPU here would turn control-plane evidence into a
            # fabricated scientific-capacity claim.
            allocated_gpus=0,
        )
    if (
        attempt["scheduler_comment"] != allocation_contract["scheduler_comment"]
        or attempt["sbatch_sha256"] != allocation_contract["script_sha256"]
    ):
        raise SlurmFleetCanaryError(
            "turnover ledger differs from the immutable allocation contract"
        )
    _snapshot, reconciled = _turnover_reconcile(
        plan=plan,
        ledger=ledger,
        runner=runner,
        scheduler_user=scheduler_user,
        now=now_fn(),
    )
    observed = _turnover_allocation_for_token(reconciled, token)
    if observed is None:
        if attempt["state"] in {"prepared", "submission_failed"}:
            tx.submit_attempt(
                directory,
                ledger,
                replica_id=str(plan["replica_id"]),
                attempt=attempt,
                now=now_fn(),
                runner=runner,
            )
        elif attempt["state"] == "submitting":
            basis = float(attempt["submit_started_at"] or attempt["created_at"])
            if now_fn() - basis >= tx.DEFAULT_VISIBILITY_GRACE_SECONDS:
                tx.submit_attempt(
                    directory,
                    ledger,
                    replica_id=str(plan["replica_id"]),
                    attempt=attempt,
                    now=now_fn(),
                    runner=runner,
                )
        elif attempt["state"] in {"submitted", "committed", "missing"}:
            raise SlurmFleetCanaryError(
                "accepted turnover intent is absent from complete scheduler truth"
            )
        elif attempt["state"] == "terminal":
            raise SlurmFleetCanaryError(
                "turnover allocation is already terminal before its transition"
            )
    snapshot, reconciled, observed = _wait_turnover_allocation(
        plan=plan,
        ledger=ledger,
        token=token,
        runner=runner,
        scheduler_user=scheduler_user,
        timeout=visibility_timeout,
        poll_seconds=poll_seconds,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    _turnover_adopt(
        directory=directory,
        ledger=ledger,
        plan=plan,
        attempt=attempt,
        allocation=observed,
        now=now_fn(),
    )
    return attempt, observed, len(reconciled.active_allocations)


def _turnover_allocation_directory(root: Path, index: int) -> Path:
    return root / "allocations" / f"c{index:02d}"


def _turnover_write_binding(
    *,
    root: Path,
    allocation_contract: Mapping[str, Any],
    allocation: tx.ReconciledFleetAllocation,
    snapshot: tx.SchedulerSnapshot,
) -> dict[str, Any]:
    row = allocation.row
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_allocation_binding",
        "captured_at": float(snapshot.captured_at),
        "complete_squeue_truth": snapshot.squeue_ok,
        "complete_sacct_truth": snapshot.sacct_ok,
        "allocation_index": allocation_contract["index"],
        "role": allocation_contract["role"],
        "job_id": row.job_id,
        "job_name": row.job_name,
        "state": row.state,
        "partition": row.partition,
        "node": row.node,
        "command": row.command,
        "comment": row.comment,
        "start_timestamp": row.start_timestamp,
        "end_timestamp": row.end_timestamp,
        "time_limit_seconds": row.time_limit_seconds,
        "dependency": row.dependency,
    }
    _write_immutable_once(
        _turnover_allocation_directory(
            root, int(allocation_contract["index"])
        )
        / "SCHEDULER_ACTIVE_BINDING.json",
        record,
        description="turnover active scheduler binding",
    )
    return record


def _turnover_write_spooled(
    *,
    root: Path,
    allocation_contract: Mapping[str, Any],
    job_id: str,
    runner: Runner,
) -> Path:
    directory = _turnover_allocation_directory(
        root, int(allocation_contract["index"])
    )
    destination = directory / "SPOOLED_BATCH_SCRIPT.sbatch"
    expected = str(allocation_contract["script"]).encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != expected
            or stat.S_IMODE(destination.stat().st_mode) & 0o222
        ):
            raise SlurmFleetCanaryError(
                "persisted turnover spooled batch script is invalid"
            )
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ".SPOOLED_BATCH_SCRIPT.pending"
    try:
        if not pending.exists() and not pending.is_symlink():
            proc = runner(
                [
                    "scontrol",
                    "write",
                    "batch_script",
                    str(job_id),
                    str(pending),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=60.0,
            )
            if proc.returncode != 0:
                raise SlurmFleetCanaryError(
                    "turnover scontrol write batch_script failed "
                    f"rc={proc.returncode}: {proc.stderr.strip()[:500]}"
                )
        if (
            pending.is_symlink()
            or not pending.is_file()
            or pending.read_bytes() != expected
        ):
            raise SlurmFleetCanaryError(
                "turnover Slurm spooled bytes differ from immutable script"
            )
        os.chmod(pending, 0o444)
        os.replace(pending, destination)
        _fsync_directory(directory)
    finally:
        try:
            pending.unlink()
        except FileNotFoundError:
            pass
    return destination


def _turnover_effective_state(
    *,
    root: Path,
    allocation_contract: Mapping[str, Any],
    job_id: str,
    runner: Runner,
    now: float,
) -> dict[str, Any]:
    path = (
        _turnover_allocation_directory(root, int(allocation_contract["index"]))
        / "EFFECTIVE_JOB_STATE.json"
    )
    command = ["scontrol", "show", "job", "-o", str(job_id)]
    if path.exists() or path.is_symlink():
        record = _read_json(path, description="turnover effective Slurm state")
    else:
        proc = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
        if proc.returncode != 0:
            raise SlurmFleetCanaryError(
                f"turnover effective Slurm query failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        parsed = _parse_effective_job_state(
            proc.stdout, expected_job_id=str(job_id)
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_warm_handoff_effective_job_state",
            "captured_at": float(now),
            "allocation_index": allocation_contract["index"],
            "command": command,
            **parsed,
        }
        _write_immutable_once(
            path,
            record,
            description="turnover effective no-requeue evidence",
        )
    if (
        set(record)
        != {
            "schema_version",
            "kind",
            "captured_at",
            "allocation_index",
            "command",
            "job_id",
            "effective_requeue",
            "no_requeue",
            "raw_output",
            "raw_output_sha256",
        }
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind")
        != "schema5_warm_handoff_effective_job_state"
        or record.get("allocation_index") != allocation_contract["index"]
        or record.get("command") != command
        or record.get("job_id") != str(job_id)
        or record.get("effective_requeue") != 0
        or record.get("no_requeue") is not True
        or record.get("raw_output_sha256")
        != _sha256_bytes(str(record.get("raw_output", "")).encode("utf-8"))
        or _parse_effective_job_state(
            str(record.get("raw_output", "")),
            expected_job_id=str(job_id),
        )["effective_requeue"]
        != 0
    ):
        raise SlurmFleetCanaryError(
            "turnover effective Slurm no-requeue evidence is invalid"
        )
    return record


def _ensure_turnover_allocation_provenance(
    *,
    root: Path,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    allocation_contract: Mapping[str, Any],
    attempt: Mapping[str, Any],
    allocation: tx.ReconciledFleetAllocation,
    runner: Runner,
    scheduler_user: str | None,
    now: float,
) -> dict[str, Any]:
    snapshot, reconciled = _turnover_reconcile(
        plan=plan,
        ledger=ledger,
        runner=runner,
        scheduler_user=scheduler_user,
        now=now,
    )
    observed = _turnover_allocation_for_token(
        reconciled, str(attempt["intent_token"])
    )
    if (
        observed is None
        or observed.row.job_id != allocation.row.job_id
        or tx.terminal_state(observed.row.state)
    ):
        raise SlurmFleetCanaryError(
            "turnover allocation vanished before provenance verification"
        )
    binding = _turnover_write_binding(
        root=root,
        allocation_contract=allocation_contract,
        allocation=observed,
        snapshot=snapshot,
    )
    spooled = _turnover_write_spooled(
        root=root,
        allocation_contract=allocation_contract,
        job_id=observed.row.job_id,
        runner=runner,
    )
    effective = _turnover_effective_state(
        root=root,
        allocation_contract=allocation_contract,
        job_id=observed.row.job_id,
        runner=runner,
        now=now,
    )
    return {
        "binding": binding,
        "spooled_path": str(spooled.resolve()),
        "spooled_sha256": _sha256_file(spooled),
        "effective": effective,
        "effective_path": str(
            (
                _turnover_allocation_directory(
                    root, int(allocation_contract["index"])
                )
                / "EFFECTIVE_JOB_STATE.json"
            ).resolve()
        ),
    }


def _probe_single_turnover_endpoint(
    *,
    root: Path,
    relative_path: Path,
    pointer: Mapping[str, Any],
    probe: EndpointProbe,
    timeout: float,
    now: float,
    kind: str,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "captured_at": float(now),
        "job_id": pointer["job_id"],
        "allocation_index": pointer["allocation_index"],
        "observation": dict(
            probe(str(pointer["endpoint"]), timeout=timeout)
        ),
    }
    _write_immutable_once(
        root / relative_path,
        record,
        description="turnover endpoint probe receipt",
    )
    return record


def _read_or_probe_single_turnover_endpoint(
    *,
    root: Path,
    relative_path: Path,
    pointer: Mapping[str, Any],
    probe: EndpointProbe,
    timeout: float,
    now: float,
    kind: str,
) -> dict[str, Any]:
    path = root / relative_path
    if path.exists() or path.is_symlink():
        record = _read_json(path, description="turnover endpoint probe receipt")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("kind") != kind
            or record.get("job_id") != pointer["job_id"]
            or record.get("allocation_index") != pointer["allocation_index"]
            or set(record.get("observation", {}))
            != {"endpoint", "health", "models"}
        ):
            raise SlurmFleetCanaryError(
                "turnover endpoint probe receipt is invalid"
            )
        return record
    return _probe_single_turnover_endpoint(
        root=root,
        relative_path=relative_path,
        pointer=pointer,
        probe=probe,
        timeout=timeout,
        now=now,
        kind=kind,
    )


def _turnover_pointer_from_binding(
    *,
    root: Path,
    plan: Mapping[str, Any],
    allocation_contract: Mapping[str, Any],
    attempt: Mapping[str, Any],
    promoted_at: float,
) -> dict[str, Any]:
    binding = _read_json(
        _turnover_allocation_directory(
            root, int(allocation_contract["index"])
        )
        / "SCHEDULER_ACTIVE_BINDING.json",
        description="turnover active scheduler binding",
    )
    return _turnover_pointer_payload(
        plan=plan,
        allocation=allocation_contract,
        job_id=str(attempt["job_id"]),
        node=str(binding["node"]),
        promoted_at=promoted_at,
    )


def _ensure_initial_turnover_pointer(
    *,
    root: Path,
    pointer: Mapping[str, Any],
    probe_receipts: Sequence[Mapping[str, Any]],
    now: float,
) -> dict[str, Any]:
    intent_path = root / "INITIAL_PROMOTION_INTENT.json"
    proposed = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_initial_promotion_intent",
        "created_at": float(now),
        "pointer": dict(pointer),
        "probe_receipts": [
            {
                "path": str(
                    (root / "initial" / f"PROBE_{sequence}.json").resolve()
                ),
                "sha256": _sha256_file(
                    root / "initial" / f"PROBE_{sequence}.json"
                ),
            }
            for sequence in range(1, len(probe_receipts) + 1)
        ],
    }
    if intent_path.exists() or intent_path.is_symlink():
        intent = _read_json(
            intent_path, description="initial turnover promotion intent"
        )
        comparable = dict(proposed)
        comparable["created_at"] = intent.get("created_at")
        comparable["pointer"]["promoted_at"] = intent.get("pointer", {}).get(
            "promoted_at"
        )
        if intent != comparable:
            raise SlurmFleetCanaryError(
                "initial turnover promotion intent drifted"
            )
    else:
        intent = proposed
        _write_immutable_once(
            intent_path,
            intent,
            description="initial turnover promotion intent",
        )
    pointer_path = root / TURNOVER_POINTER_FILENAME
    if pointer_path.exists() or pointer_path.is_symlink():
        observed = _read_json(
            pointer_path, description="initial turnover promoted pointer"
        )
        if observed != intent["pointer"]:
            raise SlurmFleetCanaryError(
                "initial turnover pointer differs from durable intent"
            )
    else:
        _write_atomic_turnover_pointer(pointer_path, intent["pointer"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_initial_promotion_accepted",
        "recorded_at": float(now),
        "promotion_intent_path": str(intent_path.resolve()),
        "promotion_intent_sha256": _sha256_file(intent_path),
        "job_id": intent["pointer"]["job_id"],
        "promoted_pointer_path": str(pointer_path.resolve()),
        "promoted_pointer_sha256": _sha256_file(pointer_path),
    }
    _write_immutable_once(
        root / "INITIAL_PROMOTION_ACCEPTED.json",
        receipt,
        description="initial turnover promotion acceptance",
    )
    return intent["pointer"]


def _read_or_probe_turnover_pair(
    *,
    root: Path,
    cycle: int,
    sequence: int,
    predecessor_pointer: Mapping[str, Any],
    standby_pointer: Mapping[str, Any],
    probe: EndpointProbe,
    timeout: float,
    now: float,
) -> dict[str, Any]:
    path = root / "transitions" / f"c{cycle:02d}" / f"PROBE_{sequence}.json"
    if path.exists() or path.is_symlink():
        record = _read_json(path, description="turnover dual-probe receipt")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("kind") != "schema5_warm_handoff_dual_probe"
            or record.get("cycle") != cycle
            or record.get("sequence") != sequence
            or record.get("routed_job_id") != predecessor_pointer["job_id"]
            or record.get("standby_job_id") != standby_pointer["job_id"]
            or set(record.get("predecessor", {}))
            != {"endpoint", "health", "models"}
            or set(record.get("standby", {}))
            != {"endpoint", "health", "models"}
        ):
            raise SlurmFleetCanaryError("turnover dual-probe receipt is invalid")
        return record
    return _probe_turnover_pair(
        root=root,
        cycle=cycle,
        sequence=sequence,
        predecessor_pointer=predecessor_pointer,
        standby_pointer=standby_pointer,
        probe=probe,
        timeout=timeout,
        now=now,
    )


def _promote_turnover_standby(
    *,
    root: Path,
    directory: Path,
    ledger: dict[str, Any],
    plan: Mapping[str, Any],
    cycle: int,
    predecessor_attempt: dict[str, Any],
    predecessor_pointer: Mapping[str, Any],
    standby_attempt: dict[str, Any],
    standby_pointer: Mapping[str, Any],
    probe_records: Sequence[Mapping[str, Any]],
    now: float,
) -> dict[str, Any]:
    transition = root / "transitions" / f"c{cycle:02d}"
    intent_path = transition / "PROMOTION_INTENT.json"
    successor_pointer = dict(standby_pointer)
    successor_pointer["promoted_at"] = float(now)
    proposed = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_promotion_intent",
        "created_at": float(now),
        "cycle": cycle,
        "predecessor_job_id": predecessor_pointer["job_id"],
        "successor_job_id": standby_pointer["job_id"],
        "predecessor_pointer_sha256": _sha256_bytes(
            _canonical_bytes(predecessor_pointer)
        ),
        "successor_pointer": successor_pointer,
        "probe_receipts": [
            {
                "path": str(
                    (
                        transition / f"PROBE_{sequence}.json"
                    ).resolve()
                ),
                "sha256": _sha256_file(
                    transition / f"PROBE_{sequence}.json"
                ),
            }
            for sequence in range(1, len(probe_records) + 1)
        ],
    }
    if intent_path.exists() or intent_path.is_symlink():
        intent = _read_json(intent_path, description="turnover promotion intent")
        # ``created_at``/``promoted_at`` are selected once at the crash boundary.
        comparable = dict(proposed)
        comparable["created_at"] = intent.get("created_at")
        comparable["successor_pointer"]["promoted_at"] = intent.get(
            "successor_pointer", {}
        ).get("promoted_at")
        if intent != comparable:
            raise SlurmFleetCanaryError("turnover promotion intent drifted")
    else:
        intent = proposed
        _write_immutable_once(
            intent_path,
            intent,
            description="turnover promotion intent",
        )
    pointer_path = root / TURNOVER_POINTER_FILENAME
    pointer = _read_json(pointer_path, description="turnover promoted pointer")
    predecessor_matches = (
        pointer.get("job_id") == predecessor_pointer["job_id"]
        and pointer.get("allocation_index")
        == predecessor_pointer["allocation_index"]
    )
    successor_matches = pointer == intent["successor_pointer"]
    if not predecessor_matches and not successor_matches:
        raise SlurmFleetCanaryError(
            "turnover promoted pointer is neither predecessor nor intended successor"
        )
    if predecessor_attempt["state"] != "terminal":
        predecessor_attempt["lifecycle"] = "retiring"
    standby_attempt["ready_probe_count"] = max(
        int(standby_attempt["ready_probe_count"]), len(probe_records)
    )
    standby_attempt["last_ready_probe_at"] = float(now)
    if predecessor_matches:
        tx.save_ledger(directory, ledger, now=now)
        _write_atomic_turnover_pointer(
            pointer_path, intent["successor_pointer"]
        )
    standby_attempt["lifecycle"] = "promoted"
    standby_attempt["promoted_at"] = float(
        intent["successor_pointer"]["promoted_at"]
    )
    tx.save_ledger(directory, ledger, now=now)
    pointer = _read_json(pointer_path, description="turnover promoted pointer")
    if pointer != intent["successor_pointer"]:
        raise SlurmFleetCanaryError(
            "turnover pointer did not atomically promote intended successor"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_promotion_accepted",
        "recorded_at": float(now),
        "cycle": cycle,
        "predecessor_job_id": predecessor_pointer["job_id"],
        "successor_job_id": standby_pointer["job_id"],
        "promotion_intent_path": str(intent_path.resolve()),
        "promotion_intent_sha256": _sha256_file(intent_path),
        "promoted_pointer_path": str(pointer_path.resolve()),
        "promoted_pointer_sha256": _sha256_file(pointer_path),
        "ledger_path": str(
            tx.ledger_path(directory, ROLLOUT_GENERATION).resolve()
        ),
        "ledger_sha256": _sha256_file(
            tx.ledger_path(directory, ROLLOUT_GENERATION)
        ),
    }
    receipt_path = transition / "PROMOTION_ACCEPTED.json"
    _write_immutable_once(
        receipt_path,
        receipt,
        description="turnover promotion acceptance",
    )
    return receipt


def _retire_turnover_allocation(
    *,
    root: Path,
    directory: Path,
    ledger: dict[str, Any],
    plan: Mapping[str, Any],
    retirement_name: str,
    attempt: dict[str, Any],
    allocation_contract: Mapping[str, Any],
    routed_pointer: Mapping[str, Any],
    runner: Runner,
    scheduler_user: str | None,
    terminal_timeout: float,
    poll_seconds: float,
    drain_seconds: float,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    probe: EndpointProbe,
    probe_timeout: float,
    teardown: bool,
) -> tuple[dict[str, Any], int]:
    directory_path = root / retirement_name
    directory_path.mkdir(parents=True, exist_ok=True)
    effective_path = (
        _turnover_allocation_directory(
            root, int(allocation_contract["index"])
        )
        / "EFFECTIVE_JOB_STATE.json"
    )
    spooled_path = (
        _turnover_allocation_directory(
            root, int(allocation_contract["index"])
        )
        / "SPOOLED_BATCH_SCRIPT.sbatch"
    )
    if not effective_path.is_file() or not spooled_path.is_file():
        raise SlurmFleetCanaryError(
            "turnover retirement lacks spooled/effective provenance"
        )
    job_id = str(attempt["job_id"])
    if teardown:
        if routed_pointer.get("job_id") != job_id:
            raise SlurmFleetCanaryError(
                "turnover final teardown does not target routed allocation"
            )
    elif routed_pointer.get("job_id") == job_id:
        raise SlurmFleetCanaryError(
            "turnover predecessor remained routed at retirement"
        )
    intent_path = directory_path / "RETIREMENT_INTENT.json"
    proposed = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_retirement_intent",
        "created_at": float(now_fn()),
        "teardown": teardown,
        "job_id": job_id,
        "command": ["scancel", job_id],
        "allocation_index": allocation_contract["index"],
        "scheduler_comment": allocation_contract["scheduler_comment"],
        "routed_job_id": routed_pointer["job_id"],
        "drain_seconds": float(drain_seconds),
        "not_before": float(now_fn()) + float(drain_seconds),
        "effective_state_path": str(effective_path.resolve()),
        "effective_state_sha256": _sha256_file(effective_path),
        "spooled_path": str(spooled_path.resolve()),
        "spooled_sha256": _sha256_file(spooled_path),
    }
    if intent_path.exists() or intent_path.is_symlink():
        intent = _read_json(intent_path, description="turnover retirement intent")
        comparable = dict(proposed)
        comparable["created_at"] = intent.get("created_at")
        comparable["not_before"] = intent.get("not_before")
        if intent != comparable:
            raise SlurmFleetCanaryError("turnover retirement intent drifted")
    else:
        intent = proposed
        _write_immutable_once(
            intent_path,
            intent,
            description="turnover exact-id retirement intent",
        )
    remaining = float(intent["not_before"]) - now_fn()
    if remaining > 0:
        sleep_fn(remaining)
    _read_or_probe_single_turnover_endpoint(
        root=root,
        relative_path=Path(retirement_name) / "PRE_RETIRE_ROUTED_PROBE.json",
        pointer=routed_pointer,
        probe=probe,
        timeout=probe_timeout,
        now=now_fn(),
        kind="schema5_warm_handoff_pre_retirement_routed_probe",
    )
    # Recheck both scheduler identity and the effective no-requeue receipt immediately
    # before crossing the destructive boundary.
    _turnover_effective_state(
        root=root,
        allocation_contract=allocation_contract,
        job_id=job_id,
        runner=runner,
        now=now_fn(),
    )
    snapshot, reconciliation = _turnover_reconcile(
        plan=plan,
        ledger=ledger,
        runner=runner,
        scheduler_user=scheduler_user,
        now=now_fn(),
    )
    observed = _turnover_allocation_for_token(
        reconciliation, str(attempt["intent_token"])
    )
    terminal_observed = (
        observed is not None and tx.terminal_state(observed.row.state)
    )
    accepted_path = directory_path / "RETIREMENT_ACCEPTED.json"
    if accepted_path.exists() or accepted_path.is_symlink():
        accepted = _read_json(
            accepted_path, description="turnover retirement acceptance"
        )
    else:
        if terminal_observed:
            evidence = "terminal_scheduler_truth"
        else:
            if observed is None or observed.row.job_id != job_id:
                raise SlurmFleetCanaryError(
                    "turnover predecessor vanished before exact-id retirement"
                )
            attempt["retire_requested_at"] = (
                attempt["retire_requested_at"] or float(now_fn())
            )
            attempt["retire_attempts"] += 1
            attempt["last_retire_attempt_at"] = float(now_fn())
            attempt["retire_error"] = None
            tx.save_ledger(directory, ledger, now=now_fn())
            proc = runner(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                check=False,
                timeout=60.0,
            )
            if proc.returncode != 0:
                attempt["retire_error"] = (
                    f"scancel rc={proc.returncode}: {proc.stderr.strip()[:500]}"
                )
                tx.save_ledger(directory, ledger, now=now_fn())
                raise SlurmFleetCanaryError(attempt["retire_error"])
            evidence = "scancel_returncode_zero"
        accepted = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_warm_handoff_retirement_accepted",
            "recorded_at": float(now_fn()),
            "job_id": job_id,
            "command": ["scancel", job_id],
            "retirement_intent_path": str(intent_path.resolve()),
            "retirement_intent_sha256": _sha256_file(intent_path),
            "effective_state_sha256": _sha256_file(effective_path),
            "evidence": evidence,
        }
        _write_immutable_once(
            accepted_path,
            accepted,
            description="turnover retirement acceptance",
        )
    if (
        accepted.get("job_id") != job_id
        or accepted.get("command") != ["scancel", job_id]
        or accepted.get("effective_state_sha256") != _sha256_file(effective_path)
        or accepted.get("evidence")
        not in {"scancel_returncode_zero", "terminal_scheduler_truth"}
    ):
        raise SlurmFleetCanaryError("turnover retirement acceptance is invalid")
    if not terminal_observed:
        snapshot, reconciliation, observed = _wait_turnover_allocation(
            plan=plan,
            ledger=ledger,
            token=str(attempt["intent_token"]),
            runner=runner,
            scheduler_user=scheduler_user,
            timeout=terminal_timeout,
            poll_seconds=poll_seconds,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            require_terminal=True,
        )
    assert observed is not None
    _turnover_adopt(
        directory=directory,
        ledger=ledger,
        plan=plan,
        attempt=attempt,
        allocation=observed,
        now=now_fn(),
    )
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_terminal_binding",
        "captured_at": float(snapshot.captured_at),
        "job_id": job_id,
        "state": observed.row.state,
        "allocation_index": allocation_contract["index"],
        "comment": observed.row.comment,
        "retirement_accepted_path": str(accepted_path.resolve()),
        "retirement_accepted_sha256": _sha256_file(accepted_path),
    }
    _write_immutable_once(
        directory_path / "SCHEDULER_TERMINAL_BINDING.json",
        terminal,
        description="turnover terminal scheduler binding",
    )
    return terminal, len(reconciliation.active_allocations)


def _turnover_artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.name == TURNOVER_COMPLETE_FILENAME and path.parent == root:
            continue
        if path.is_symlink():
            raise SlurmFleetCanaryError(
                f"turnover canary contains a symlink: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise SlurmFleetCanaryError(
                f"turnover canary contains an unsafe entry: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _seal_and_publish_turnover_complete(
    *,
    root: Path,
    intent: Mapping[str, Any],
    ledger: Mapping[str, Any],
    max_physical_allocations: int,
    now: float,
) -> dict[str, Any]:
    complete_path = root / TURNOVER_COMPLETE_FILENAME
    if complete_path.exists() or complete_path.is_symlink():
        return verify_turnover_complete(root)
    plan = intent["plan"]
    attempts = ledger["replicas"][plan["replica_id"]]["attempts"]
    if (
        len(attempts) != int(plan["cycles"]) + 1
        or any(item["state"] != "terminal" for item in attempts)
        or any(not str(item["job_id"] or "").isdigit() for item in attempts)
        or len({str(item["job_id"]) for item in attempts}) != len(attempts)
    ):
        raise SlurmFleetCanaryError(
            "turnover completion requires one terminal allocation per generation"
        )
    pointer_path = root / TURNOVER_POINTER_FILENAME
    pointer = _read_json(pointer_path, description="final turnover pointer")
    if (
        pointer.get("allocation_index") != plan["cycles"]
        or pointer.get("job_id") != attempts[-1]["job_id"]
    ):
        raise SlurmFleetCanaryError(
            "final turnover pointer does not bind the last allocation"
        )
    transition_receipts: list[dict[str, Any]] = []
    for cycle in range(1, int(plan["cycles"]) + 1):
        transition = root / "transitions" / f"c{cycle:02d}"
        for name in (
            "PROMOTION_INTENT.json",
            "PROMOTION_ACCEPTED.json",
            "RETIREMENT_INTENT.json",
            "RETIREMENT_ACCEPTED.json",
            "SCHEDULER_TERMINAL_BINDING.json",
            "PRE_RETIRE_ROUTED_PROBE.json",
        ):
            path = transition / name
            if not path.is_file() or path.is_symlink():
                raise SlurmFleetCanaryError(
                    f"turnover transition receipt is missing: {path}"
                )
            transition_receipts.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                }
            )
    effective_receipts: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        path = (
            _turnover_allocation_directory(root, index)
            / "EFFECTIVE_JOB_STATE.json"
        )
        record = _turnover_effective_state(
            root=root,
            allocation_contract=plan["allocations"][index],
            job_id=str(attempt["job_id"]),
            runner=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("completion must use persisted effective evidence")
            ),
            now=now,
        )
        effective_receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "job_id": record["job_id"],
                "effective_requeue": record["effective_requeue"],
            }
        )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SlurmFleetCanaryError(
                f"cannot seal symlinked turnover artifact: {path}"
            )
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    inventory = _turnover_artifact_inventory(root)
    ledger_path = tx.ledger_path(
        tx.state_directory(root), ROLLOUT_GENERATION
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_warm_handoff_turnover_canary_complete",
        "completed_at": float(now),
        "canary_root": str(root),
        "intent_path": str((root / TURNOVER_INTENT_FILENAME).resolve()),
        "intent_sha256": _sha256_file(root / TURNOVER_INTENT_FILENAME),
        "plan_sha256": plan["plan_sha256"],
        "code_identity": intent["code_identity"],
        "pool_id": plan["pool_id"],
        "fleet_sha256": plan["fleet_sha256"],
        "replica_id": plan["replica_id"],
        "cycles_completed": int(plan["cycles"]),
        "allocations_submitted": len(attempts),
        "job_ids": [str(item["job_id"]) for item in attempts],
        "all_effective_requeue": 0,
        "effective_state_receipts": effective_receipts,
        "transition_receipts": transition_receipts,
        "promoted_pointer_path": str(pointer_path.resolve()),
        "promoted_pointer_sha256": _sha256_file(pointer_path),
        "final_teardown_terminal": True,
        "maximum_physical_allocations_observed": int(max_physical_allocations),
        "maximum_extra_gpus_observed": 0,
        "production_overlap_gpu_ceiling": 4,
        "canary_drain_seconds": plan["canary_drain_seconds"],
        "production_drain_seconds": plan["production_drain_seconds"],
        "readiness_probes_per_promotion": 2,
        "continuous_routed_endpoint_evidence": True,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": _sha256_file(ledger_path),
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": _sha256_bytes(
            _canonical_compact_bytes(inventory)
        ),
    }
    payload["canary_id"] = _sha256_bytes(_canonical_bytes(payload))
    _write_immutable_once(
        complete_path,
        payload,
        description="marker-last turnover completion",
    )
    for child in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        child.chmod(stat.S_IMODE(child.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
    return payload


def verify_turnover_complete(root: Path) -> dict[str, Any]:
    """Verify sealed turnover evidence without Git, Slurm, HTTP, or mutable sources."""

    root = _validate_isolated_root(root)
    complete = _read_json(
        root / TURNOVER_COMPLETE_FILENAME,
        description="warm-turnover completion",
    )
    intent = _read_json(
        root / TURNOVER_INTENT_FILENAME,
        description="warm-turnover marker-first intent",
    )
    plan = _validate_turnover_plan(intent.get("plan", {}))
    identity_payload = dict(complete)
    canary_id = identity_payload.pop("canary_id", None)
    inventory = _turnover_artifact_inventory(root)
    required = {
        "schema_version",
        "kind",
        "completed_at",
        "canary_root",
        "intent_path",
        "intent_sha256",
        "plan_sha256",
        "code_identity",
        "pool_id",
        "fleet_sha256",
        "replica_id",
        "cycles_completed",
        "allocations_submitted",
        "job_ids",
        "all_effective_requeue",
        "effective_state_receipts",
        "transition_receipts",
        "promoted_pointer_path",
        "promoted_pointer_sha256",
        "final_teardown_terminal",
        "maximum_physical_allocations_observed",
        "maximum_extra_gpus_observed",
        "production_overlap_gpu_ceiling",
        "canary_drain_seconds",
        "production_drain_seconds",
        "readiness_probes_per_promotion",
        "continuous_routed_endpoint_evidence",
        "ledger_path",
        "ledger_sha256",
        "artifact_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "canary_id",
    }
    if (
        set(complete) != required
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind")
        != "schema5_warm_handoff_turnover_canary_complete"
        or complete.get("canary_root") != str(root)
        or complete.get("intent_path")
        != str((root / TURNOVER_INTENT_FILENAME).resolve())
        or complete.get("intent_sha256")
        != _sha256_file(root / TURNOVER_INTENT_FILENAME)
        or complete.get("plan_sha256") != plan["plan_sha256"]
        or complete.get("pool_id") != plan["pool_id"]
        or complete.get("fleet_sha256") != plan["fleet_sha256"]
        or complete.get("replica_id") != plan["replica_id"]
        or complete.get("cycles_completed") != plan["cycles"]
        or complete.get("cycles_completed", 0) < 2
        or complete.get("allocations_submitted") != plan["cycles"] + 1
        or len(complete.get("job_ids", [])) != plan["cycles"] + 1
        or len(set(complete.get("job_ids", []))) != plan["cycles"] + 1
        or not all(str(job_id).isdigit() for job_id in complete.get("job_ids", []))
        or complete.get("all_effective_requeue") != 0
        or len(complete.get("effective_state_receipts", []))
        != plan["cycles"] + 1
        or complete.get("final_teardown_terminal") is not True
        or not 1
        <= complete.get("maximum_physical_allocations_observed", 0)
        <= 2
        or complete.get("maximum_extra_gpus_observed") != 0
        or complete.get("production_overlap_gpu_ceiling") != 4
        or complete.get("canary_drain_seconds")
        != plan["canary_drain_seconds"]
        or complete.get("production_drain_seconds")
        != PRODUCTION_HANDOFF_DRAIN_SECONDS
        or complete.get("readiness_probes_per_promotion") != 2
        or complete.get("continuous_routed_endpoint_evidence") is not True
        or complete.get("artifact_count") != len(inventory)
        or complete.get("artifact_inventory") != inventory
        or complete.get("artifact_inventory_sha256")
        != _sha256_bytes(_canonical_compact_bytes(inventory))
        or canary_id != _sha256_bytes(_canonical_bytes(identity_payload))
    ):
        raise SlurmFleetCanaryError(
            "turnover completion identity or inventory drifted"
        )
    pointer_path = root / TURNOVER_POINTER_FILENAME
    pointer = _read_json(pointer_path, description="final turnover pointer")
    if (
        complete.get("promoted_pointer_path") != str(pointer_path.resolve())
        or complete.get("promoted_pointer_sha256") != _sha256_file(pointer_path)
        or pointer.get("job_id") != complete["job_ids"][-1]
        or pointer.get("allocation_index") != plan["cycles"]
    ):
        raise SlurmFleetCanaryError("final turnover pointer evidence drifted")
    for index, receipt in enumerate(complete["effective_state_receipts"]):
        path = (
            _turnover_allocation_directory(root, index)
            / "EFFECTIVE_JOB_STATE.json"
        )
        if (
            receipt.get("path") != str(path.resolve())
            or receipt.get("sha256") != _sha256_file(path)
            or receipt.get("job_id") != complete["job_ids"][index]
            or receipt.get("effective_requeue") != 0
        ):
            raise SlurmFleetCanaryError(
                "turnover effective Requeue receipt drifted"
            )
        _turnover_effective_state(
            root=root,
            allocation_contract=plan["allocations"][index],
            job_id=complete["job_ids"][index],
            runner=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("sealed verification must not contact Slurm")
            ),
            now=float(complete["completed_at"]),
        )
        spooled = (
            _turnover_allocation_directory(root, index)
            / "SPOOLED_BATCH_SCRIPT.sbatch"
        )
        if (
            spooled.read_bytes()
            != str(plan["allocations"][index]["script"]).encode("utf-8")
        ):
            raise SlurmFleetCanaryError(
                "sealed turnover spooled script bytes drifted"
            )
    for receipt in complete["transition_receipts"]:
        path = Path(str(receipt.get("path", "")))
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SlurmFleetCanaryError(
                "turnover receipt escapes canary root"
            ) from exc
        if receipt.get("sha256") != _sha256_file(path):
            raise SlurmFleetCanaryError("turnover transition receipt drifted")
    ledger_path = Path(str(complete.get("ledger_path", "")))
    if (
        ledger_path
        != tx.ledger_path(tx.state_directory(root), ROLLOUT_GENERATION)
        or complete.get("ledger_sha256") != _sha256_file(ledger_path)
    ):
        raise SlurmFleetCanaryError("turnover final ledger drifted")
    with tx.read_transaction_lock(root) as directory:
        ledgers = tx.read_generation_ledgers(
            directory,
            pool_root=root,
            pool_id=str(plan["pool_id"]),
            fleet_sha256=str(plan["fleet_sha256"]),
            current_generation=ROLLOUT_GENERATION,
            replica_ids=[str(plan["replica_id"])],
        )
    attempts = ledgers[-1]["replicas"][plan["replica_id"]]["attempts"]
    if (
        len(attempts) != plan["cycles"] + 1
        or [str(item["job_id"]) for item in attempts] != complete["job_ids"]
        or any(item["state"] != "terminal" for item in attempts)
        or any(item["submission_attempts"] != 1 for item in attempts)
    ):
        raise SlurmFleetCanaryError(
            "turnover ledger does not prove exactly one allocation per cycle"
        )
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise SlurmFleetCanaryError(
                f"sealed turnover canary contains symlink: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError(
                f"sealed turnover artifact remains writable: {path}"
            )
    return complete


def run_turnover_canary(
    *,
    root: Path,
    apply: bool = False,
    verify: bool = False,
    partition: str = DEFAULT_PARTITION,
    time_limit: str = DEFAULT_TIME_LIMIT,
    cycles: int = DEFAULT_TURNOVER_CYCLES,
    drain_seconds: float = DEFAULT_TURNOVER_DRAIN_SECONDS,
    scheduler_user: str | None = None,
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    terminal_timeout: float = DEFAULT_TERMINAL_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    probe_timeout: float = 10.0,
    runner: Runner = subprocess.run,
    probe: EndpointProbe = _default_endpoint_probe,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    token_factory: Callable[[], str] = lambda: os.urandom(16).hex(),
    code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute and seal at least two real short-walltime warm turnovers."""

    root = _validate_isolated_root(root)
    if apply and verify:
        raise SlurmFleetCanaryError("--apply and --verify are mutually exclusive")
    if (root / TURNOVER_COMPLETE_FILENAME).exists() or (
        root / TURNOVER_COMPLETE_FILENAME
    ).is_symlink():
        complete = verify_turnover_complete(root)
        if code_identity is not None and complete.get(
            "code_identity"
        ) != _validate_code_identity(code_identity):
            raise SlurmFleetCanaryError(
                "sealed turnover canary belongs to another code identity"
            )
        return complete
    if verify:
        raise SlurmFleetCanaryError(
            f"turnover completion is missing: {root / TURNOVER_COMPLETE_FILENAME}"
        )
    if not apply:
        return render_turnover_canary_plan(
            partition=partition,
            time_limit=time_limit,
            cycles=cycles,
            drain_seconds=drain_seconds,
        )
    if (
        visibility_timeout <= 0
        or terminal_timeout <= 0
        or poll_seconds <= 0
        or probe_timeout <= 0
    ):
        raise SlurmFleetCanaryError(
            "turnover polling, probe, and timeout values must be positive"
        )
    raw_probe = probe

    def bounded_probe(endpoint: str, *, timeout: float) -> Mapping[str, Any]:
        deadline = now_fn() + visibility_timeout
        last_error: Exception | None = None
        while True:
            try:
                return raw_probe(endpoint, timeout=timeout)
            except Exception as exc:  # readiness is expected to lag RUNNING briefly
                last_error = exc
            if now_fn() >= deadline:
                raise SlurmFleetCanaryError(
                    f"turnover endpoint {endpoint} did not become ready within "
                    f"{visibility_timeout:g}s: {last_error}"
                ) from last_error
            sleep_fn(min(poll_seconds, max(0.0, deadline - now_fn())))

    probe = bounded_probe
    bound_code_identity = (
        derive_code_identity()
        if code_identity is None
        else _validate_code_identity(code_identity)
    )
    intent = _load_or_create_turnover_intent(
        root=root,
        partition=partition,
        time_limit=time_limit,
        cycles=cycles,
        drain_seconds=drain_seconds,
        code_identity=bound_code_identity,
        token_factory=token_factory,
        now=now_fn(),
    )
    plan = intent["plan"]
    max_physical = 0
    with tx.transaction_lock(root) as directory:
        ledger = tx.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=str(plan["pool_id"]),
            fleet_sha256=str(plan["fleet_sha256"]),
            rollout_generation=ROLLOUT_GENERATION,
            replica_ids=[str(plan["replica_id"])],
            now=now_fn(),
        )
        attempts = ledger["replicas"][plan["replica_id"]]["attempts"]
        pointer_path = root / TURNOVER_POINTER_FILENAME
        if not pointer_path.exists() and not pointer_path.is_symlink():
            primary_attempt, primary_allocation, active_count = (
                _ensure_turnover_attempt_running(
                    directory=directory,
                    ledger=ledger,
                    plan=plan,
                    allocation_contract=plan["allocations"][0],
                    predecessor=None,
                    runner=runner,
                    scheduler_user=scheduler_user,
                    visibility_timeout=visibility_timeout,
                    poll_seconds=poll_seconds,
                    now_fn=now_fn,
                    sleep_fn=sleep_fn,
                )
            )
            max_physical = max(max_physical, active_count)
            _ensure_turnover_allocation_provenance(
                root=root,
                plan=plan,
                ledger=ledger,
                allocation_contract=plan["allocations"][0],
                attempt=primary_attempt,
                allocation=primary_allocation,
                runner=runner,
                scheduler_user=scheduler_user,
                now=now_fn(),
            )
            primary_pointer = _turnover_pointer_payload(
                plan=plan,
                allocation=plan["allocations"][0],
                job_id=primary_allocation.row.job_id,
                node=primary_allocation.row.node,
                promoted_at=now_fn(),
            )
            initial_probes = [
                _read_or_probe_single_turnover_endpoint(
                    root=root,
                    relative_path=Path("initial") / f"PROBE_{sequence}.json",
                    pointer=primary_pointer,
                    probe=probe,
                    timeout=probe_timeout,
                    now=now_fn(),
                    kind="schema5_warm_handoff_initial_readiness_probe",
                )
                for sequence in (1, 2)
            ]
            _ensure_initial_turnover_pointer(
                root=root,
                pointer=primary_pointer,
                probe_receipts=initial_probes,
                now=now_fn(),
            )
        pointer = _read_json(
            pointer_path, description="turnover promoted pointer"
        )
        for cycle in range(1, int(plan["cycles"]) + 1):
            predecessor_contract = plan["allocations"][cycle - 1]
            successor_contract = plan["allocations"][cycle]
            attempts = ledger["replicas"][plan["replica_id"]]["attempts"]
            predecessor_token = tx.parse_intent_comment(
                predecessor_contract["scheduler_comment"]
            )["intent"]
            predecessor_attempts = [
                item
                for item in attempts
                if item["intent_token"] == predecessor_token
            ]
            if len(predecessor_attempts) != 1:
                raise SlurmFleetCanaryError(
                    f"turnover cycle {cycle} lacks one predecessor intent"
                )
            predecessor_attempt = predecessor_attempts[0]
            pointer = _read_json(
                pointer_path, description="turnover promoted pointer"
            )
            if int(pointer["allocation_index"]) < cycle:
                if int(pointer["allocation_index"]) != cycle - 1:
                    raise SlurmFleetCanaryError(
                        "turnover pointer skipped a transition"
                    )
                successor_attempt, successor_allocation, active_count = (
                    _ensure_turnover_attempt_running(
                        directory=directory,
                        ledger=ledger,
                        plan=plan,
                        allocation_contract=successor_contract,
                        predecessor=predecessor_attempt,
                        runner=runner,
                        scheduler_user=scheduler_user,
                        visibility_timeout=visibility_timeout,
                        poll_seconds=poll_seconds,
                        now_fn=now_fn,
                        sleep_fn=sleep_fn,
                    )
                )
                max_physical = max(max_physical, active_count)
                _ensure_turnover_allocation_provenance(
                    root=root,
                    plan=plan,
                    ledger=ledger,
                    allocation_contract=successor_contract,
                    attempt=successor_attempt,
                    allocation=successor_allocation,
                    runner=runner,
                    scheduler_user=scheduler_user,
                    now=now_fn(),
                )
                standby_pointer = _turnover_pointer_payload(
                    plan=plan,
                    allocation=successor_contract,
                    job_id=successor_allocation.row.job_id,
                    node=successor_allocation.row.node,
                    promoted_at=now_fn(),
                )
                probes = [
                    _read_or_probe_turnover_pair(
                        root=root,
                        cycle=cycle,
                        sequence=sequence,
                        predecessor_pointer=pointer,
                        standby_pointer=standby_pointer,
                        probe=probe,
                        timeout=probe_timeout,
                        now=now_fn(),
                    )
                    for sequence in (1, 2)
                ]
                _promote_turnover_standby(
                    root=root,
                    directory=directory,
                    ledger=ledger,
                    plan=plan,
                    cycle=cycle,
                    predecessor_attempt=predecessor_attempt,
                    predecessor_pointer=pointer,
                    standby_attempt=successor_attempt,
                    standby_pointer=standby_pointer,
                    probe_records=probes,
                    now=now_fn(),
                )
                pointer = _read_json(
                    pointer_path, description="turnover promoted pointer"
                )
                _read_or_probe_single_turnover_endpoint(
                    root=root,
                    relative_path=Path("transitions")
                    / f"c{cycle:02d}"
                    / "POST_PROMOTION_PROBE.json",
                    pointer=pointer,
                    probe=probe,
                    timeout=probe_timeout,
                    now=now_fn(),
                    kind="schema5_warm_handoff_post_promotion_probe",
                )
            elif int(pointer["allocation_index"]) == cycle:
                successor_token = tx.parse_intent_comment(
                    successor_contract["scheduler_comment"]
                )["intent"]
                successor_matches = [
                    item
                    for item in attempts
                    if item["intent_token"] == successor_token
                ]
                if len(successor_matches) != 1:
                    raise SlurmFleetCanaryError(
                        "promoted turnover pointer lacks successor ledger intent"
                    )
                successor_attempt = successor_matches[0]
                if successor_attempt["lifecycle"] == "standby":
                    predecessor_pointer = _turnover_pointer_from_binding(
                        root=root,
                        plan=plan,
                        allocation_contract=predecessor_contract,
                        attempt=predecessor_attempt,
                        promoted_at=float(
                            predecessor_attempt.get("promoted_at")
                            or predecessor_attempt["committed_at"]
                        ),
                    )
                    probes = [
                        _read_or_probe_turnover_pair(
                            root=root,
                            cycle=cycle,
                            sequence=sequence,
                            predecessor_pointer=predecessor_pointer,
                            standby_pointer=pointer,
                            probe=probe,
                            timeout=probe_timeout,
                            now=now_fn(),
                        )
                        for sequence in (1, 2)
                    ]
                    _promote_turnover_standby(
                        root=root,
                        directory=directory,
                        ledger=ledger,
                        plan=plan,
                        cycle=cycle,
                        predecessor_attempt=predecessor_attempt,
                        predecessor_pointer=predecessor_pointer,
                        standby_attempt=successor_attempt,
                        standby_pointer=pointer,
                        probe_records=probes,
                        now=now_fn(),
                    )
            else:
                # A later pointer proves this transition completed far enough to
                # prepare another handoff; its immutable receipts are checked below.
                successor_attempt = next(
                    (
                        item
                        for item in attempts
                        if item["intent_token"]
                        == tx.parse_intent_comment(
                            successor_contract["scheduler_comment"]
                        )["intent"]
                    ),
                    None,
                )
                if successor_attempt is None:
                    raise SlurmFleetCanaryError(
                        "turnover pointer advanced without prior successor intent"
                    )
            pointer = _read_json(
                pointer_path, description="turnover promoted pointer"
            )
            _terminal, active_count = _retire_turnover_allocation(
                root=root,
                directory=directory,
                ledger=ledger,
                plan=plan,
                retirement_name=f"transitions/c{cycle:02d}",
                attempt=predecessor_attempt,
                allocation_contract=predecessor_contract,
                routed_pointer=pointer,
                runner=runner,
                scheduler_user=scheduler_user,
                terminal_timeout=terminal_timeout,
                poll_seconds=poll_seconds,
                drain_seconds=drain_seconds,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                probe=probe,
                probe_timeout=probe_timeout,
                teardown=False,
            )
            max_physical = max(max_physical, active_count)
        pointer = _read_json(
            pointer_path, description="final turnover promoted pointer"
        )
        attempts = ledger["replicas"][plan["replica_id"]]["attempts"]
        final_attempt = attempts[-1]
        _read_or_probe_single_turnover_endpoint(
            root=root,
            relative_path=Path("teardown") / "FINAL_ROUTED_PROBE.json",
            pointer=pointer,
            probe=probe,
            timeout=probe_timeout,
            now=now_fn(),
            kind="schema5_warm_handoff_final_routed_probe",
        )
        _terminal, active_count = _retire_turnover_allocation(
            root=root,
            directory=directory,
            ledger=ledger,
            plan=plan,
            retirement_name="teardown",
            attempt=final_attempt,
            allocation_contract=plan["allocations"][-1],
            routed_pointer=pointer,
            runner=runner,
            scheduler_user=scheduler_user,
            terminal_timeout=terminal_timeout,
            poll_seconds=poll_seconds,
            drain_seconds=drain_seconds,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            probe=probe,
            probe_timeout=probe_timeout,
            teardown=True,
        )
        max_physical = max(max_physical, active_count)
        if any(
            (
                root
                / "transitions"
                / f"c{cycle:02d}"
                / "PROMOTION_ACCEPTED.json"
            ).is_file()
            for cycle in range(1, int(plan["cycles"]) + 1)
        ):
            # A promotion receipt is hash-bound to simultaneous predecessor/standby
            # probes, so it proves the two-allocation peak even after crash/restart.
            max_physical = max(max_physical, 2)
        _seal_and_publish_turnover_complete(
            root=root,
            intent=intent,
            ledger=ledger,
            max_physical_allocations=max_physical,
            now=now_fn(),
        )
    return verify_turnover_complete(root)


def _new_static_intent(
    *,
    root: Path,
    partition: str,
    time_limit: str,
    code_identity: Mapping[str, Any],
    token_factory: Callable[[], str],
    now: float,
) -> dict[str, Any]:
    token = token_factory()
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise SlurmFleetCanaryError("intent token factory returned an invalid token")
    suffix = token[:12]
    pool_id = f"schema5-v12-slurm-canary-{suffix}"
    replica_id = f"schema5-v12-canary-{suffix}"
    job_name = f"asys-s5-serve-canary-{suffix}"
    script = _job_script(
        job_name=job_name,
        partition=partition,
        time_limit=time_limit,
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_canary_contract",
        "pool_id": pool_id,
        "replica_id": replica_id,
        "profile": PROFILE,
        "job_name": job_name,
        "partition": partition,
        "time_limit": time_limit,
        "cpus": 1,
        "memory": "128M",
        "no_requeue": True,
        "job_script_sha256": _sha256_bytes(script.encode("utf-8")),
        "code_identity": dict(code_identity),
    }
    fleet_sha256 = _sha256_bytes(_canonical_compact_bytes(contract))
    comment = tx.intent_comment(
        pool_id=pool_id,
        profile=PROFILE,
        replica_id=replica_id,
        rollout_generation=ROLLOUT_GENERATION,
        intent_token=token,
        fleet_sha256=fleet_sha256,
    )
    sbatch_path = (
        tx.state_directory(root)
        / "sbatch"
        / f"g{ROLLOUT_GENERATION:06d}"
        / f"{replica_id}.{token}.sbatch"
    ).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_canary_intent",
        "created_at": float(now),
        "canary_root": str(root),
        "pool_id": pool_id,
        "fleet_sha256": fleet_sha256,
        "rollout_generation": ROLLOUT_GENERATION,
        "replica_id": replica_id,
        "profile": PROFILE,
        "intent_token": token,
        "scheduler_comment": comment,
        "job_name": job_name,
        "partition": partition,
        "time_limit": time_limit,
        "cpus": 1,
        "memory": "128M",
        "no_requeue": True,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": _sha256_bytes(script.encode("utf-8")),
        "script": script,
        "code_identity": dict(code_identity),
    }


def _validate_static_intent(intent: Mapping[str, Any], *, root: Path) -> None:
    required = {
        "schema_version",
        "kind",
        "created_at",
        "canary_root",
        "pool_id",
        "fleet_sha256",
        "rollout_generation",
        "replica_id",
        "profile",
        "intent_token",
        "scheduler_comment",
        "job_name",
        "partition",
        "time_limit",
        "cpus",
        "memory",
        "no_requeue",
        "sbatch_path",
        "sbatch_sha256",
        "script",
        "code_identity",
    }
    if not isinstance(intent, Mapping) or set(intent) != required:
        raise SlurmFleetCanaryError("canary static-intent fields drifted")
    if (
        intent["schema_version"] != SCHEMA_VERSION
        or intent["kind"] != "schema5_slurm_fleet_canary_intent"
        or intent["canary_root"] != str(root)
        or intent["rollout_generation"] != ROLLOUT_GENERATION
        or intent["profile"] != PROFILE
        or intent["cpus"] != 1
        or intent["memory"] != "128M"
        or intent["no_requeue"] is not True
        or not isinstance(intent["created_at"], (int, float))
        or isinstance(intent["created_at"], bool)
        or _TOKEN_RE.fullmatch(str(intent["intent_token"])) is None
        or _SHA256_RE.fullmatch(str(intent["fleet_sha256"])) is None
        or _SHA256_RE.fullmatch(str(intent["sbatch_sha256"])) is None
        or not _JOB_NAME_RE.fullmatch(str(intent["job_name"]))
    ):
        raise SlurmFleetCanaryError("canary static-intent identity is invalid")
    expected_script = _job_script(
        job_name=str(intent["job_name"]),
        partition=str(intent["partition"]),
        time_limit=str(intent["time_limit"]),
    )
    expected_comment = tx.intent_comment(
        pool_id=str(intent["pool_id"]),
        profile=PROFILE,
        replica_id=str(intent["replica_id"]),
        rollout_generation=ROLLOUT_GENERATION,
        intent_token=str(intent["intent_token"]),
        fleet_sha256=str(intent["fleet_sha256"]),
    )
    expected_contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_canary_contract",
        "pool_id": intent["pool_id"],
        "replica_id": intent["replica_id"],
        "profile": PROFILE,
        "job_name": intent["job_name"],
        "partition": intent["partition"],
        "time_limit": intent["time_limit"],
        "cpus": 1,
        "memory": "128M",
        "no_requeue": True,
        "job_script_sha256": _sha256_bytes(expected_script.encode("utf-8")),
        "code_identity": intent["code_identity"],
    }
    expected_fleet_sha256 = _sha256_bytes(_canonical_compact_bytes(expected_contract))
    expected_path = (
        tx.state_directory(root)
        / "sbatch"
        / f"g{ROLLOUT_GENERATION:06d}"
        / f"{intent['replica_id']}.{intent['intent_token']}.sbatch"
    ).resolve()
    if (
        intent["script"] != expected_script
        or intent["fleet_sha256"] != expected_fleet_sha256
        or intent["sbatch_sha256"] != _sha256_bytes(expected_script.encode("utf-8"))
        or intent["scheduler_comment"] != expected_comment
        or intent["sbatch_path"] != str(expected_path)
        or _validate_code_identity(intent["code_identity"])
        != intent["code_identity"]
    ):
        raise SlurmFleetCanaryError("canary static-intent provenance drifted")


def _load_or_create_static_intent(
    *,
    root: Path,
    partition: str,
    time_limit: str,
    code_identity: Mapping[str, Any],
    token_factory: Callable[[], str],
    now: float,
) -> dict[str, Any]:
    path = root / INTENT_FILENAME
    if path.exists() or path.is_symlink():
        intent = _read_json(path, description="canary static intent")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError("existing canary static intent is writable")
        _validate_static_intent(intent, root=root)
        if intent["partition"] != partition or intent["time_limit"] != time_limit:
            raise SlurmFleetCanaryError(
                "requested Slurm parameters differ from the immutable canary intent"
            )
        if intent["code_identity"] != code_identity:
            raise SlurmFleetCanaryError(
                "requested code identity differs from the immutable canary intent"
            )
        return intent
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise SlurmFleetCanaryError(f"canary root is unsafe: {root}")
        entries = list(root.iterdir())
        if entries:
            raise SlurmFleetCanaryError(
                "new canary root contains artifacts before its marker-first intent: "
                + ", ".join(sorted(item.name for item in entries))
            )
    else:
        root.mkdir(parents=True)
    intent = _new_static_intent(
        root=root,
        partition=partition,
        time_limit=time_limit,
        code_identity=code_identity,
        token_factory=token_factory,
        now=now,
    )
    _write_immutable_once(path, intent, description="canary static intent")
    return intent


def _scheduler_rows_for_intent(
    snapshot: tx.SchedulerSnapshot, intent: Mapping[str, Any]
) -> tuple[tx.SchedulerRow, ...]:
    selected: list[tx.SchedulerRow] = []
    for row in snapshot.rows:
        parsed = tx.parse_intent_comment(row.comment)
        exact_name = row.job_name == intent["job_name"]
        related = parsed is not None and (
            parsed["pool"] == intent["pool_id"]
            or parsed["intent"] == intent["intent_token"]
            or parsed["replica"] == intent["replica_id"]
        )
        if not exact_name and not related:
            continue
        if (
            parsed is None
            or row.job_name != intent["job_name"]
            or row.comment != intent["scheduler_comment"]
        ):
            raise SlurmFleetCanaryError(
                f"foreign or malformed scheduler allocation overlaps canary identity: "
                f"job {row.job_id}"
            )
        selected.append(row)
    return tuple(selected)


def _query_bound_allocation(
    *,
    intent: Mapping[str, Any],
    ledger: Mapping[str, Any],
    runner: Runner,
    scheduler_user: str | None,
    now: float,
) -> tuple[tx.SchedulerSnapshot, tx.ReconciledFleetAllocation | None]:
    snapshot = tx.query_scheduler(runner=runner, user=scheduler_user, now=now)
    selected = _scheduler_rows_for_intent(snapshot, intent)
    try:
        reconciled = tx.reconcile_scheduler_rows(
            selected,
            [ledger],
            pool_id=str(intent["pool_id"]),
            fleet_sha256=str(intent["fleet_sha256"]),
            replica_profiles={str(intent["replica_id"]): PROFILE},
            replica_job_names={str(intent["replica_id"]): str(intent["job_name"])},
        )
    except tx.FleetTransactionError as exc:
        raise SlurmFleetCanaryError(str(exc)) from exc
    if len(reconciled.allocations) > 1:
        raise SlurmFleetCanaryError("canary intent maps to more than one Slurm job")
    return snapshot, (None if not reconciled.allocations else reconciled.allocations[0])


def _adopt_allocation(
    *,
    directory: Path,
    ledger: dict[str, Any],
    intent: Mapping[str, Any],
    allocation: tx.ReconciledFleetAllocation,
    now: float,
) -> dict[str, Any]:
    attempt = ledger["replicas"][intent["replica_id"]]["attempts"][0]
    row = allocation.row
    if row.partition != intent["partition"]:
        raise SlurmFleetCanaryError(
            f"canary job {row.job_id} ran in unexpected partition {row.partition!r}"
        )
    if attempt["job_id"] not in {None, row.job_id}:
        raise SlurmFleetCanaryError("canary intent changed Slurm job id")
    attempt["job_id"] = row.job_id
    if attempt["submitted_at"] is None:
        attempt["submitted_at"] = float(now)
    attempt["last_seen_at"] = float(now)
    attempt["missing_since"] = None
    if tx.terminal_state(row.state):
        attempt["state"] = "terminal"
        if attempt["terminal_at"] is None:
            attempt["terminal_at"] = float(now)
    else:
        attempt["state"] = "committed"
        if attempt["committed_at"] is None:
            attempt["committed_at"] = float(now)
    tx.save_ledger(directory, ledger, now=now)
    return attempt


def _binding_payload(
    *,
    kind: str,
    snapshot: tx.SchedulerSnapshot,
    allocation: tx.ReconciledFleetAllocation,
    attempt: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    row = allocation.row
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "captured_at": float(snapshot.captured_at),
        "complete_squeue_truth": snapshot.squeue_ok,
        "complete_sacct_truth": snapshot.sacct_ok,
        "job_id": row.job_id,
        "job_name": row.job_name,
        "state": row.state,
        "partition": row.partition,
        "node": row.node,
        "command": row.command,
        "comment": row.comment,
        "scheduler_source": row.source,
        "pool_id": intent["pool_id"],
        "replica_id": intent["replica_id"],
        "profile": intent["profile"],
        "rollout_generation": intent["rollout_generation"],
        "intent_token": intent["intent_token"],
        "fleet_sha256": intent["fleet_sha256"],
        "sbatch_path": attempt["sbatch_path"],
        "sbatch_sha256": attempt["sbatch_sha256"],
    }


def _wait_for_allocation(
    *,
    intent: Mapping[str, Any],
    ledger: Mapping[str, Any],
    runner: Runner,
    scheduler_user: str | None,
    timeout: float,
    poll_seconds: float,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> tuple[tx.SchedulerSnapshot, tx.ReconciledFleetAllocation]:
    deadline = now_fn() + timeout
    while True:
        snapshot, allocation = _query_bound_allocation(
            intent=intent,
            ledger=ledger,
            runner=runner,
            scheduler_user=scheduler_user,
            now=now_fn(),
        )
        if allocation is not None:
            return snapshot, allocation
        if now_fn() >= deadline:
            raise SlurmFleetCanaryError(
                "canary allocation did not become visible in complete squeue+sacct truth"
            )
        sleep_fn(poll_seconds)


def _write_and_verify_spooled_script(
    *,
    root: Path,
    job_id: str,
    expected: bytes,
    runner: Runner,
) -> Path:
    destination = root / SPOOLED_SCRIPT_FILENAME
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != expected
            or stat.S_IMODE(destination.stat().st_mode) & 0o222
        ):
            raise SlurmFleetCanaryError("persisted spooled batch script is invalid")
        return destination
    # A fixed, transaction-local pending path makes the scontrol boundary recoverable:
    # if the caller dies after Slurm writes the file, the next invocation validates
    # and promotes those exact bytes instead of issuing another write blindly.
    temporary = root / ".SPOOLED_BATCH_SCRIPT.pending"
    try:
        if not temporary.exists() and not temporary.is_symlink():
            proc = runner(
                ["scontrol", "write", "batch_script", job_id, str(temporary)],
                capture_output=True,
                text=True,
                check=False,
                timeout=60.0,
            )
            if proc.returncode != 0:
                raise SlurmFleetCanaryError(
                    f"scontrol write batch_script failed rc={proc.returncode}: "
                    f"{proc.stderr.strip()[:500]}"
                )
        if temporary.is_symlink() or not temporary.is_file():
            raise SlurmFleetCanaryError(
                "scontrol did not create a regular spooled script"
            )
        observed = temporary.read_bytes()
        if observed != expected:
            raise SlurmFleetCanaryError(
                "Slurm spooled batch script differs from immutable submitted bytes"
            )
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        _fsync_directory(root)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _parse_effective_job_state(
    output: str,
    *,
    expected_job_id: str,
) -> dict[str, Any]:
    """Parse one exact ``scontrol show job -o`` result and require Requeue=0."""

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise SlurmFleetCanaryError(
            "effective Slurm job query must return exactly one record"
        )
    try:
        tokens = shlex.split(lines[0], posix=True)
    except (TypeError, ValueError) as exc:
        raise SlurmFleetCanaryError(
            f"cannot parse effective Slurm job state: {exc}"
        ) from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise SlurmFleetCanaryError(
                f"duplicate effective Slurm field {key!r}"
            )
        fields[key] = value.strip('"')
    if fields.get("JobId") != str(expected_job_id):
        raise SlurmFleetCanaryError(
            "effective Slurm job state changed exact job identity"
        )
    if fields.get("Requeue") != "0":
        raise SlurmFleetCanaryError(
            f"effective Slurm Requeue is not disabled: {fields.get('Requeue')!r}"
        )
    return {
        "job_id": str(expected_job_id),
        "effective_requeue": 0,
        "no_requeue": True,
        "raw_output": lines[0] + "\n",
        "raw_output_sha256": _sha256_bytes((lines[0] + "\n").encode("utf-8")),
    }


def _ensure_effective_no_requeue(
    *,
    root: Path,
    job_id: str,
    runner: Runner,
    now: float,
) -> dict[str, Any]:
    """Persist effective (not merely scripted) Requeue=0 before any retirement."""

    path = root / EFFECTIVE_JOB_STATE_FILENAME
    if path.exists() or path.is_symlink():
        record = _read_json(path, description="effective Slurm job state")
    else:
        command = ["scontrol", "show", "job", "-o", str(job_id)]
        proc = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
        if proc.returncode != 0:
            raise SlurmFleetCanaryError(
                f"effective Slurm job query failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        parsed = _parse_effective_job_state(
            proc.stdout, expected_job_id=str(job_id)
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_slurm_fleet_canary_effective_job_state",
            "captured_at": float(now),
            "command": command,
            **parsed,
        }
        _write_immutable_once(
            path,
            record,
            description="effective Slurm no-requeue evidence",
        )
    required = {
        "schema_version",
        "kind",
        "captured_at",
        "command",
        "job_id",
        "effective_requeue",
        "no_requeue",
        "raw_output",
        "raw_output_sha256",
    }
    if (
        set(record) != required
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind")
        != "schema5_slurm_fleet_canary_effective_job_state"
        or record.get("command")
        != ["scontrol", "show", "job", "-o", str(job_id)]
        or record.get("job_id") != str(job_id)
        or record.get("effective_requeue") != 0
        or record.get("no_requeue") is not True
        or record.get("raw_output_sha256")
        != _sha256_bytes(str(record.get("raw_output", "")).encode("utf-8"))
        or _parse_effective_job_state(
            str(record.get("raw_output", "")),
            expected_job_id=str(job_id),
        )["no_requeue"]
        is not True
    ):
        raise SlurmFleetCanaryError(
            "effective Slurm no-requeue evidence is invalid"
        )
    return record


def _retirement_payload(
    *,
    root: Path,
    intent: Mapping[str, Any],
    attempt: Mapping[str, Any],
    job_id: str,
    spooled_path: Path,
    now: float,
) -> dict[str, Any]:
    ledger = tx.ledger_path(tx.state_directory(root), ROLLOUT_GENERATION)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_canary_retirement_intent",
        "created_at": float(now),
        "job_id": job_id,
        "command": ["scancel", job_id],
        "job_name": intent["job_name"],
        "scheduler_comment": intent["scheduler_comment"],
        "pool_id": intent["pool_id"],
        "replica_id": intent["replica_id"],
        "intent_token": intent["intent_token"],
        "fleet_sha256": intent["fleet_sha256"],
        "sbatch_path": attempt["sbatch_path"],
        "sbatch_sha256": attempt["sbatch_sha256"],
        "spooled_path": str(spooled_path.resolve()),
        "spooled_sha256": _sha256_file(spooled_path),
        "effective_job_state_path": str(
            (root / EFFECTIVE_JOB_STATE_FILENAME).resolve()
        ),
        "effective_job_state_sha256": _sha256_file(
            root / EFFECTIVE_JOB_STATE_FILENAME
        ),
        "ledger_path": str(ledger.resolve()),
        "ledger_sha256": _sha256_file(ledger),
    }


def _validate_retirement(
    retirement: Mapping[str, Any],
    *,
    root: Path,
    intent: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "kind",
        "created_at",
        "job_id",
        "command",
        "job_name",
        "scheduler_comment",
        "pool_id",
        "replica_id",
        "intent_token",
        "fleet_sha256",
        "sbatch_path",
        "sbatch_sha256",
        "spooled_path",
        "spooled_sha256",
        "effective_job_state_path",
        "effective_job_state_sha256",
        "ledger_path",
        "ledger_sha256",
    }
    job_id = str(retirement.get("job_id", ""))
    if (
        set(retirement) != required
        or retirement.get("schema_version") != SCHEMA_VERSION
        or retirement.get("kind") != "schema5_slurm_fleet_canary_retirement_intent"
        or not job_id.isdigit()
        or retirement.get("command") != ["scancel", job_id]
        or retirement.get("job_name") != intent["job_name"]
        or retirement.get("scheduler_comment") != intent["scheduler_comment"]
        or retirement.get("pool_id") != intent["pool_id"]
        or retirement.get("replica_id") != intent["replica_id"]
        or retirement.get("intent_token") != intent["intent_token"]
        or retirement.get("fleet_sha256") != intent["fleet_sha256"]
        or retirement.get("sbatch_path") != intent["sbatch_path"]
        or retirement.get("sbatch_sha256") != intent["sbatch_sha256"]
        or retirement.get("spooled_path")
        != str((root / SPOOLED_SCRIPT_FILENAME).resolve())
        or retirement.get("spooled_sha256") != intent["sbatch_sha256"]
        or retirement.get("effective_job_state_path")
        != str((root / EFFECTIVE_JOB_STATE_FILENAME).resolve())
        or retirement.get("effective_job_state_sha256")
        != _sha256_file(root / EFFECTIVE_JOB_STATE_FILENAME)
        or retirement.get("ledger_path")
        != str(tx.ledger_path(tx.state_directory(root), ROLLOUT_GENERATION).resolve())
        or not _SHA256_RE.fullmatch(str(retirement.get("ledger_sha256", "")))
        or not isinstance(retirement.get("created_at"), (int, float))
        or isinstance(retirement.get("created_at"), bool)
    ):
        raise SlurmFleetCanaryError("exact-id retirement intent is invalid")


def _ensure_retirement_accepted(
    *,
    root: Path,
    intent: Mapping[str, Any],
    retirement: Mapping[str, Any],
    terminal_observed: bool,
    runner: Runner,
    now: float,
) -> None:
    path = root / RETIREMENT_ACCEPTED_FILENAME
    if path.exists() or path.is_symlink():
        record = _read_json(path, description="retirement acceptance")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("kind") != "schema5_slurm_fleet_canary_retirement_accepted"
            or record.get("job_id") != retirement["job_id"]
            or record.get("command") != ["scancel", retirement["job_id"]]
            or record.get("scheduler_comment") != intent["scheduler_comment"]
            or record.get("evidence")
            not in {"scancel_returncode_zero", "terminal_scheduler_truth"}
        ):
            raise SlurmFleetCanaryError("retirement acceptance drifted")
        return
    evidence = "terminal_scheduler_truth"
    if not terminal_observed:
        proc = runner(
            ["scancel", str(retirement["job_id"])],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
        )
        if proc.returncode != 0:
            raise SlurmFleetCanaryError(
                f"exact-id scancel failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        evidence = "scancel_returncode_zero"
    _write_immutable_once(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_slurm_fleet_canary_retirement_accepted",
            "recorded_at": float(now),
            "job_id": retirement["job_id"],
            "command": ["scancel", retirement["job_id"]],
            "scheduler_comment": intent["scheduler_comment"],
            "evidence": evidence,
        },
        description="retirement acceptance",
    )


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.name == COMPLETE_FILENAME and path.parent == root:
            continue
        if path.is_symlink():
            raise SlurmFleetCanaryError(f"canary contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SlurmFleetCanaryError(f"canary contains unsafe entry: {relative}")
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _validate_scheduler_binding(
    binding: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    kind: str,
    terminal: bool,
) -> None:
    required = {
        "schema_version",
        "kind",
        "captured_at",
        "complete_squeue_truth",
        "complete_sacct_truth",
        "job_id",
        "job_name",
        "state",
        "partition",
        "node",
        "command",
        "comment",
        "scheduler_source",
        "pool_id",
        "replica_id",
        "profile",
        "rollout_generation",
        "intent_token",
        "fleet_sha256",
        "sbatch_path",
        "sbatch_sha256",
    }
    state = str(binding.get("state", ""))
    if (
        set(binding) != required
        or binding.get("schema_version") != SCHEMA_VERSION
        or binding.get("kind") != kind
        or not isinstance(binding.get("captured_at"), (int, float))
        or isinstance(binding.get("captured_at"), bool)
        or binding.get("complete_squeue_truth") is not True
        or binding.get("complete_sacct_truth") is not True
        or not str(binding.get("job_id", "")).isdigit()
        or binding.get("job_name") != intent["job_name"]
        or binding.get("partition") != intent["partition"]
        or binding.get("comment") != intent["scheduler_comment"]
        or binding.get("scheduler_source") not in {"squeue", "sacct"}
        or binding.get("pool_id") != intent["pool_id"]
        or binding.get("replica_id") != intent["replica_id"]
        or binding.get("profile") != intent["profile"]
        or binding.get("rollout_generation") != intent["rollout_generation"]
        or binding.get("intent_token") != intent["intent_token"]
        or binding.get("fleet_sha256") != intent["fleet_sha256"]
        or binding.get("sbatch_path") != intent["sbatch_path"]
        or binding.get("sbatch_sha256") != intent["sbatch_sha256"]
        or tx.terminal_state(state) is not terminal
        or not tx.command_binds_sbatch(
            str(binding.get("command", "")), str(intent["sbatch_path"])
        )
    ):
        raise SlurmFleetCanaryError(f"{kind} scheduler binding is invalid")


def _seal_and_publish_complete(
    *,
    root: Path,
    intent: Mapping[str, Any],
    terminal: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    complete_path = root / COMPLETE_FILENAME
    if complete_path.exists() or complete_path.is_symlink():
        return verify_complete(root)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SlurmFleetCanaryError(f"cannot seal symlinked canary entry: {path}")
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    inventory = _artifact_inventory(root)
    inventory_sha256 = _sha256_bytes(_canonical_compact_bytes(inventory))
    ledger_path = tx.ledger_path(tx.state_directory(root), ROLLOUT_GENERATION)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_transaction_canary_complete",
        "completed_at": float(now),
        "canary_root": str(root),
        "intent_path": str((root / INTENT_FILENAME).resolve()),
        "intent_sha256": _sha256_file(root / INTENT_FILENAME),
        "pool_id": intent["pool_id"],
        "replica_id": intent["replica_id"],
        "profile": intent["profile"],
        "rollout_generation": intent["rollout_generation"],
        "intent_token": intent["intent_token"],
        "fleet_sha256": intent["fleet_sha256"],
        "job_id": terminal["job_id"],
        "terminal_state": terminal["state"],
        "scheduler_comment": intent["scheduler_comment"],
        "job_name": intent["job_name"],
        "partition": intent["partition"],
        "no_requeue": True,
        "sbatch_path": intent["sbatch_path"],
        "sbatch_sha256": intent["sbatch_sha256"],
        "spooled_path": str((root / SPOOLED_SCRIPT_FILENAME).resolve()),
        "spooled_sha256": _sha256_file(root / SPOOLED_SCRIPT_FILENAME),
        "effective_job_state_path": str(
            (root / EFFECTIVE_JOB_STATE_FILENAME).resolve()
        ),
        "effective_job_state_sha256": _sha256_file(
            root / EFFECTIVE_JOB_STATE_FILENAME
        ),
        "effective_requeue": 0,
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": _sha256_file(ledger_path),
        "retirement_intent_path": str((root / RETIREMENT_INTENT_FILENAME).resolve()),
        "terminal_binding_path": str((root / TERMINAL_BINDING_FILENAME).resolve()),
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": inventory_sha256,
        "code_identity": intent["code_identity"],
    }
    payload["canary_id"] = _sha256_bytes(_canonical_bytes(payload))
    _write_immutable_once(
        complete_path,
        payload,
        description="marker-last canary completion",
    )
    # The completion marker is the final file.  Directory-mode sealing follows without
    # creating or rewriting any evidence.
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        directory.chmod(stat.S_IMODE(directory.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
    return payload


def verify_complete(root: Path) -> dict[str, Any]:
    root = _validate_isolated_root(root)
    complete = _read_json(root / COMPLETE_FILENAME, description="canary completion")
    intent = _read_json(root / INTENT_FILENAME, description="canary static intent")
    _validate_static_intent(intent, root=root)
    inventory = _artifact_inventory(root)
    complete_required = {
        "schema_version",
        "kind",
        "completed_at",
        "canary_root",
        "intent_path",
        "intent_sha256",
        "pool_id",
        "replica_id",
        "profile",
        "rollout_generation",
        "intent_token",
        "fleet_sha256",
        "job_id",
        "terminal_state",
        "scheduler_comment",
        "job_name",
        "partition",
        "no_requeue",
        "sbatch_path",
        "sbatch_sha256",
        "spooled_path",
        "spooled_sha256",
        "effective_job_state_path",
        "effective_job_state_sha256",
        "effective_requeue",
        "ledger_path",
        "ledger_sha256",
        "retirement_intent_path",
        "terminal_binding_path",
        "artifact_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "code_identity",
        "canary_id",
    }
    identity_payload = dict(complete)
    canary_id = identity_payload.pop("canary_id", None)
    if (
        set(complete) != complete_required
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != "schema5_slurm_fleet_transaction_canary_complete"
        or complete.get("canary_root") != str(root)
        or complete.get("intent_path") != str((root / INTENT_FILENAME).resolve())
        or complete.get("intent_sha256") != _sha256_file(root / INTENT_FILENAME)
        or complete.get("pool_id") != intent["pool_id"]
        or complete.get("replica_id") != intent["replica_id"]
        or complete.get("profile") != intent["profile"]
        or complete.get("rollout_generation") != ROLLOUT_GENERATION
        or complete.get("intent_token") != intent["intent_token"]
        or complete.get("fleet_sha256") != intent["fleet_sha256"]
        or complete.get("scheduler_comment") != intent["scheduler_comment"]
        or complete.get("job_name") != intent["job_name"]
        or complete.get("partition") != intent["partition"]
        or complete.get("no_requeue") is not True
        or complete.get("sbatch_path") != intent["sbatch_path"]
        or complete.get("sbatch_sha256") != intent["sbatch_sha256"]
        or complete.get("spooled_sha256") != intent["sbatch_sha256"]
        or complete.get("spooled_path")
        != str((root / SPOOLED_SCRIPT_FILENAME).resolve())
        or complete.get("effective_job_state_path")
        != str((root / EFFECTIVE_JOB_STATE_FILENAME).resolve())
        or complete.get("effective_job_state_sha256")
        != _sha256_file(root / EFFECTIVE_JOB_STATE_FILENAME)
        or complete.get("effective_requeue") != 0
        or complete.get("retirement_intent_path")
        != str((root / RETIREMENT_INTENT_FILENAME).resolve())
        or complete.get("terminal_binding_path")
        != str((root / TERMINAL_BINDING_FILENAME).resolve())
        or complete.get("artifact_count") != len(inventory)
        or complete.get("artifact_inventory") != inventory
        or complete.get("artifact_inventory_sha256")
        != _sha256_bytes(_canonical_compact_bytes(inventory))
        or complete.get("code_identity") != intent.get("code_identity")
        or _validate_code_identity(complete.get("code_identity", {}))
        != complete.get("code_identity")
        or not isinstance(canary_id, str)
        or _SHA256_RE.fullmatch(canary_id) is None
        or canary_id != _sha256_bytes(_canonical_bytes(identity_payload))
        or not str(complete.get("job_id", "")).isdigit()
        or not tx.terminal_state(str(complete.get("terminal_state", "")))
    ):
        raise SlurmFleetCanaryError("canary completion identity or inventory drifted")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SlurmFleetCanaryError(f"sealed canary contains symlink: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError(f"sealed canary entry remains writable: {path}")
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise SlurmFleetCanaryError("sealed canary root remains writable")
    retirement = _read_json(
        root / RETIREMENT_INTENT_FILENAME,
        description="exact-id retirement intent",
    )
    _validate_retirement(retirement, root=root, intent=intent)
    effective_job_state = _ensure_effective_no_requeue(
        root=root,
        job_id=str(complete["job_id"]),
        runner=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("sealed verification must not contact Slurm")
        ),
        now=float(complete["completed_at"]),
    )
    if (
        effective_job_state["effective_requeue"] != 0
        or _sha256_file(root / EFFECTIVE_JOB_STATE_FILENAME)
        != retirement["effective_job_state_sha256"]
    ):
        raise SlurmFleetCanaryError(
            "retirement is not bound to effective no-requeue evidence"
        )
    active = _read_json(
        root / ACTIVE_BINDING_FILENAME,
        description="active scheduler binding",
    )
    terminal = _read_json(
        root / TERMINAL_BINDING_FILENAME,
        description="terminal scheduler binding",
    )
    _validate_scheduler_binding(
        active,
        intent=intent,
        kind="schema5_slurm_fleet_canary_active_binding",
        terminal=False,
    )
    _validate_scheduler_binding(
        terminal,
        intent=intent,
        kind="schema5_slurm_fleet_canary_terminal_binding",
        terminal=True,
    )
    retirement_accepted = _read_json(
        root / RETIREMENT_ACCEPTED_FILENAME,
        description="retirement acceptance",
    )
    if (
        active.get("job_id") != complete["job_id"]
        or terminal.get("job_id") != complete["job_id"]
        or terminal.get("state") != complete["terminal_state"]
        or retirement.get("job_id") != complete["job_id"]
        or set(retirement_accepted)
        != {
            "schema_version",
            "kind",
            "recorded_at",
            "job_id",
            "command",
            "scheduler_comment",
            "evidence",
        }
        or retirement_accepted.get("schema_version") != SCHEMA_VERSION
        or retirement_accepted.get("kind")
        != "schema5_slurm_fleet_canary_retirement_accepted"
        or retirement_accepted.get("job_id") != complete["job_id"]
        or retirement_accepted.get("command") != ["scancel", complete["job_id"]]
        or retirement_accepted.get("scheduler_comment") != intent["scheduler_comment"]
        or retirement_accepted.get("evidence")
        not in {"scancel_returncode_zero", "terminal_scheduler_truth"}
    ):
        raise SlurmFleetCanaryError("scheduler or retirement evidence is invalid")
    ledger_path = Path(str(complete.get("ledger_path", "")))
    if ledger_path != tx.ledger_path(
        tx.state_directory(root), ROLLOUT_GENERATION
    ) or _sha256_file(ledger_path) != complete.get("ledger_sha256"):
        raise SlurmFleetCanaryError("final canary ledger is not hash-bound")
    with tx.read_transaction_lock(root) as directory:
        ledgers = tx.read_generation_ledgers(
            directory,
            pool_root=root,
            pool_id=str(intent["pool_id"]),
            fleet_sha256=str(intent["fleet_sha256"]),
            current_generation=ROLLOUT_GENERATION,
            replica_ids=[str(intent["replica_id"])],
        )
    attempt = ledgers[-1]["replicas"][intent["replica_id"]]["attempts"][0]
    if (
        len(ledgers[-1]["replicas"][intent["replica_id"]]["attempts"]) != 1
        or attempt["intent_token"] != intent["intent_token"]
        or attempt["job_id"] != complete["job_id"]
        or attempt["state"] != "terminal"
        or attempt["terminal_at"] is None
        or attempt["sbatch_path"] != intent["sbatch_path"]
        or attempt["sbatch_sha256"] != intent["sbatch_sha256"]
        or attempt["scheduler_comment"] != intent["scheduler_comment"]
    ):
        raise SlurmFleetCanaryError(
            "final canary ledger does not prove one transaction"
        )
    return complete


def run_canary(
    *,
    root: Path,
    apply: bool = False,
    verify: bool = False,
    partition: str = DEFAULT_PARTITION,
    time_limit: str = DEFAULT_TIME_LIMIT,
    scheduler_user: str | None = None,
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    terminal_timeout: float = DEFAULT_TERMINAL_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    runner: Runner = subprocess.run,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    token_factory: Callable[[], str] = lambda: os.urandom(16).hex(),
    code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _validate_isolated_root(root)
    if apply and verify:
        raise SlurmFleetCanaryError("--apply and --verify are mutually exclusive")
    complete_path = root / COMPLETE_FILENAME
    if complete_path.exists() or complete_path.is_symlink():
        complete = verify_complete(root)
        if code_identity is not None and complete.get(
            "code_identity"
        ) != _validate_code_identity(code_identity):
            raise SlurmFleetCanaryError(
                "sealed canary belongs to a different code identity"
            )
        return complete
    if verify:
        raise SlurmFleetCanaryError(f"canary completion is missing: {complete_path}")
    if not apply:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_slurm_fleet_canary_dry_run",
            "apply": False,
            "canary_root": str(root),
            "partition": partition,
            "time_limit": time_limit,
            "cpus": 1,
            "memory": "128M",
            "no_requeue": True,
            "canonical_pool_excluded": str(_canonical_pool_root()),
            "code_identity": (
                None
                if code_identity is None
                else _validate_code_identity(code_identity)
            ),
            "operations": [
                "publish marker-first static intent",
                "prepare immutable fleet admission intent before sbatch",
                "submit one CPU-only Slurm job",
                "join complete squeue+sacct truth",
                "verify exact scontrol-spooled batch bytes",
                "publish exact-id retirement intent",
                "cancel only the recorded job id",
                "observe terminal sacct truth",
                "publish checksummed read-only CANARY_COMPLETE last",
            ],
        }
    if visibility_timeout <= 0 or terminal_timeout <= 0 or poll_seconds <= 0:
        raise SlurmFleetCanaryError("poll and timeout values must be positive")
    bound_code_identity = (
        derive_code_identity()
        if code_identity is None
        else _validate_code_identity(code_identity)
    )
    intent = _load_or_create_static_intent(
        root=root,
        partition=partition,
        time_limit=time_limit,
        code_identity=bound_code_identity,
        token_factory=token_factory,
        now=now_fn(),
    )
    with tx.transaction_lock(root) as directory:
        ledger = tx.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=str(intent["pool_id"]),
            fleet_sha256=str(intent["fleet_sha256"]),
            rollout_generation=ROLLOUT_GENERATION,
            replica_ids=[str(intent["replica_id"])],
            now=now_fn(),
        )
        attempts = ledger["replicas"][intent["replica_id"]]["attempts"]
        if not attempts:
            attempt = tx.prepare_attempt(
                directory,
                ledger,
                replica_id=str(intent["replica_id"]),
                profile=PROFILE,
                pool_id=str(intent["pool_id"]),
                fleet_sha256=str(intent["fleet_sha256"]),
                rollout_generation=ROLLOUT_GENERATION,
                sbatch_text=str(intent["script"]),
                now=now_fn(),
                token_factory=lambda: str(intent["intent_token"]),
            )
        elif len(attempts) == 1:
            attempt = attempts[0]
        else:
            raise SlurmFleetCanaryError(
                "canary ledger contains multiple admission intents"
            )
        if (
            attempt["intent_token"] != intent["intent_token"]
            or attempt["scheduler_comment"] != intent["scheduler_comment"]
            or attempt["sbatch_path"] != intent["sbatch_path"]
            or attempt["sbatch_sha256"] != intent["sbatch_sha256"]
        ):
            raise SlurmFleetCanaryError(
                "canary ledger drifted from marker-first intent"
            )

        snapshot, allocation = _query_bound_allocation(
            intent=intent,
            ledger=ledger,
            runner=runner,
            scheduler_user=scheduler_user,
            now=now_fn(),
        )
        if allocation is None:
            if attempt["state"] in {"prepared", "submission_failed"}:
                tx.submit_attempt(
                    directory,
                    ledger,
                    replica_id=str(intent["replica_id"]),
                    attempt=attempt,
                    now=now_fn(),
                    runner=runner,
                )
            elif attempt["state"] == "submitting":
                basis = float(attempt["submit_started_at"] or attempt["created_at"])
                if now_fn() - basis < tx.DEFAULT_VISIBILITY_GRACE_SECONDS:
                    # The scheduler may have accepted the job before the prior process
                    # died.  Never cross sbatch again within the ambiguity window.
                    pass
                else:
                    # Complete joined absence after the fleet visibility grace permits
                    # retry of this same immutable token/path.
                    tx.submit_attempt(
                        directory,
                        ledger,
                        replica_id=str(intent["replica_id"]),
                        attempt=attempt,
                        now=now_fn(),
                        runner=runner,
                    )
            elif attempt["state"] == "submitted":
                raise SlurmFleetCanaryError(
                    f"accepted canary job {attempt['job_id']} is absent from complete "
                    "squeue+sacct truth; refusing a duplicate"
                )
            elif attempt["state"] in {"committed", "missing"}:
                raise SlurmFleetCanaryError(
                    "previously visible canary allocation disappeared; refusing replacement"
                )
            elif attempt["state"] == "terminal":
                raise SlurmFleetCanaryError(
                    "canary allocation terminated before spooled provenance completed"
                )
            snapshot, allocation = _wait_for_allocation(
                intent=intent,
                ledger=ledger,
                runner=runner,
                scheduler_user=scheduler_user,
                timeout=visibility_timeout,
                poll_seconds=poll_seconds,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
            )
        attempt = _adopt_allocation(
            directory=directory,
            ledger=ledger,
            intent=intent,
            allocation=allocation,
            now=now_fn(),
        )
        terminal_observed = tx.terminal_state(allocation.row.state)
        if terminal_observed:
            # Recovery after an accepted exact-id cancellation is valid only if active
            # binding, spooled bytes, and retirement intent were already durable.
            active_binding = _read_json(
                root / ACTIVE_BINDING_FILENAME,
                description="active scheduler binding",
            )
            _validate_scheduler_binding(
                active_binding,
                intent=intent,
                kind="schema5_slurm_fleet_canary_active_binding",
                terminal=False,
            )
            if active_binding.get("job_id") != allocation.row.job_id:
                raise SlurmFleetCanaryError(
                    "terminal recovery lacks the prior active scheduler binding"
                )
            spooled_path = root / SPOOLED_SCRIPT_FILENAME
            if (
                spooled_path.is_symlink()
                or not spooled_path.is_file()
                or spooled_path.read_bytes() != str(intent["script"]).encode("utf-8")
            ):
                raise SlurmFleetCanaryError(
                    "canary terminated before durable spooled-script verification"
                )
        else:
            active_binding = _binding_payload(
                kind="schema5_slurm_fleet_canary_active_binding",
                snapshot=snapshot,
                allocation=allocation,
                attempt=attempt,
                intent=intent,
            )
            _validate_scheduler_binding(
                active_binding,
                intent=intent,
                kind="schema5_slurm_fleet_canary_active_binding",
                terminal=False,
            )
            _write_immutable_once(
                root / ACTIVE_BINDING_FILENAME,
                active_binding,
                description="active scheduler binding",
            )
            spooled_path = _write_and_verify_spooled_script(
                root=root,
                job_id=allocation.row.job_id,
                expected=str(intent["script"]).encode("utf-8"),
                runner=runner,
            )
        effective_path = root / EFFECTIVE_JOB_STATE_FILENAME
        if terminal_observed and not effective_path.is_file():
            raise SlurmFleetCanaryError(
                "terminal recovery lacks pre-retirement effective Requeue=0 evidence"
            )
        _ensure_effective_no_requeue(
            root=root,
            job_id=allocation.row.job_id,
            runner=runner,
            now=now_fn(),
        )
        retirement_path = root / RETIREMENT_INTENT_FILENAME
        if retirement_path.exists() or retirement_path.is_symlink():
            retirement = _read_json(
                retirement_path, description="exact-id retirement intent"
            )
        else:
            retirement = _retirement_payload(
                root=root,
                intent=intent,
                attempt=attempt,
                job_id=allocation.row.job_id,
                spooled_path=spooled_path,
                now=now_fn(),
            )
            _write_immutable_once(
                retirement_path,
                retirement,
                description="exact-id retirement intent",
            )
        _validate_retirement(retirement, root=root, intent=intent)
        if retirement["job_id"] != allocation.row.job_id:
            raise SlurmFleetCanaryError("retirement intent job id changed")

        _ensure_retirement_accepted(
            root=root,
            intent=intent,
            retirement=retirement,
            terminal_observed=terminal_observed,
            runner=runner,
            now=now_fn(),
        )
        if terminal_observed:
            terminal_snapshot, terminal_allocation = snapshot, allocation
        else:
            deadline = now_fn() + terminal_timeout
            while True:
                terminal_snapshot, terminal_allocation = _query_bound_allocation(
                    intent=intent,
                    ledger=ledger,
                    runner=runner,
                    scheduler_user=scheduler_user,
                    now=now_fn(),
                )
                if terminal_allocation is not None and tx.terminal_state(
                    terminal_allocation.row.state
                ):
                    break
                if now_fn() >= deadline:
                    raise SlurmFleetCanaryError(
                        f"exact job {retirement['job_id']} did not reach terminal "
                        "sacct truth"
                    )
                sleep_fn(poll_seconds)
        terminal_attempt = _adopt_allocation(
            directory=directory,
            ledger=ledger,
            intent=intent,
            allocation=terminal_allocation,
            now=now_fn(),
        )
        terminal_binding = _binding_payload(
            kind="schema5_slurm_fleet_canary_terminal_binding",
            snapshot=terminal_snapshot,
            allocation=terminal_allocation,
            attempt=terminal_attempt,
            intent=intent,
        )
        _validate_scheduler_binding(
            terminal_binding,
            intent=intent,
            kind="schema5_slurm_fleet_canary_terminal_binding",
            terminal=True,
        )
        _write_immutable_once(
            root / TERMINAL_BINDING_FILENAME,
            terminal_binding,
            description="terminal scheduler binding",
        )
        _seal_and_publish_complete(
            root=root,
            intent=intent,
            terminal=terminal_binding,
            now=now_fn(),
        )
    # The writer lock has been released before sealed verification takes a read lock.
    return verify_complete(root)


def _dependency_job_scripts(
    *,
    root: Path,
    partition: str,
    time_limit: str,
) -> dict[str, bytes]:
    """Render the exact three-job dependency-cascade experiment."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
        raise SlurmFleetCanaryError(
            f"unsafe dependency-canary partition: {partition!r}"
        )
    if re.fullmatch(r"00:0[1-9]:00", time_limit) is None:
        raise SlurmFleetCanaryError(
            "dependency-canary time limit must be 1 through 9 whole minutes"
        )
    alert_marker = root / DEPENDENCY_ALERT_MARKER_FILENAME
    child_marker = root / DEPENDENCY_FORBIDDEN_CHILD_MARKER_FILENAME
    common = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition}\n"
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --mem=256M\n"
        f"#SBATCH --time={time_limit}\n"
        "#SBATCH --no-requeue\n"
        "set -euo pipefail\n"
        "umask 027\n"
    )
    root_script = (
        common
        + "#SBATCH --job-name=asys-s5-serve-dep-root\n"
        + "exit 42\n"
    )
    child_script = (
        common
        + "#SBATCH --job-name=asys-s5-serve-dep-child\n"
        + f"printf 'INVALID\\n' > {shlex.quote(str(child_marker))}\n"
        + "exit 99\n"
    )
    marker_program = (
        "import json, os, pathlib, sys, tempfile, time\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "job_id = os.environ.get('SLURM_JOB_ID', '')\n"
        "if not job_id.isdigit():\n"
        "    raise SystemExit('missing numeric SLURM_JOB_ID')\n"
        "payload = {'schema_version': 1, "
        "'kind': 'schema5_dependency_alert_executed', "
        "'job_id': job_id, 'started_timestamp': time.time()}\n"
        "fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', "
        "suffix='.tmp', dir=path.parent)\n"
        "try:\n"
        "    with os.fdopen(fd, 'w', encoding='utf-8') as handle:\n"
        "        json.dump(payload, handle, sort_keys=True, "
        "separators=(',', ':'), allow_nan=False)\n"
        "        handle.write('\\n')\n"
        "        handle.flush()\n"
        "        os.fsync(handle.fileno())\n"
        "    os.replace(name, path)\n"
        "    dfd = os.open(path.parent, os.O_RDONLY | "
        "getattr(os, 'O_DIRECTORY', 0))\n"
        "    try:\n"
        "        os.fsync(dfd)\n"
        "    finally:\n"
        "        os.close(dfd)\n"
        "finally:\n"
        "    try:\n"
        "        os.unlink(name)\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
    )
    sentinel_script = (
        common
        + "#SBATCH --job-name=asys-s5-serve-dep-sentinel\n"
        + f"{shlex.quote(sys.executable)} - "
        + f"{shlex.quote(str(alert_marker))} <<'PY'\n"
        + marker_program
        + "PY\n"
    )
    return {
        "root": root_script.encode("utf-8"),
        "child": child_script.encode("utf-8"),
        "sentinel": sentinel_script.encode("utf-8"),
    }


def _dependency_scheduler_since(timestamp: float) -> str:
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        raise SlurmFleetCanaryError("dependency-canary timestamp is invalid")
    return datetime.fromtimestamp(
        float(timestamp), tz=timezone.utc
    ).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _dependency_submit_comment(submit_line: str) -> str:
    try:
        tokens = shlex.split(submit_line)
    except ValueError as exc:
        raise SlurmFleetCanaryError(
            f"invalid dependency-canary SubmitLine: {exc}"
        ) from exc
    comments: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            comments.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise SlurmFleetCanaryError(
                    "dependency-canary SubmitLine has empty --comment"
                )
            comments.append(tokens[index + 1])
    if len(comments) > 1:
        raise SlurmFleetCanaryError(
            "dependency-canary SubmitLine has duplicate comments"
        )
    return comments[0] if comments else ""


def _dependency_state(value: str) -> str:
    return value.strip().split()[0].split("+", 1)[0].upper() if value.strip() else ""


def _dependency_scheduler_snapshot(
    *,
    slurm_user: str,
    since: str,
    runner: Runner,
) -> dict[str, Any]:
    commands = {
        "squeue": [
            "squeue",
            "-u",
            slurm_user,
            "-h",
            "-o",
            "%i|%k|%j|%T|%r|%S|%M",
        ],
        "sacct": [
            "sacct",
            "-u",
            slurm_user,
            "-X",
            "-n",
            "-P",
            "-S",
            since,
            (
                "--format=JobIDRaw,Comment%256,JobName%64,State,ExitCode,"
                "Reason,Start,End,Elapsed,SubmitLine"
            ),
        ],
    }
    raw: dict[str, dict[str, Any]] = {}
    live: dict[str, dict[str, str]] = {}
    history: dict[str, dict[str, str]] = {}
    for source, argv in commands.items():
        proc = runner(argv)
        record = {
            "argv": list(argv),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        }
        raw[source] = record
        if proc.returncode != 0:
            raise SlurmFleetCanaryError(
                f"dependency-canary {source} query failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        for line in proc.stdout.splitlines():
            fields = line.rstrip("\n").split("|")
            if source == "squeue":
                if len(fields) != 7:
                    if line.strip():
                        raise SlurmFleetCanaryError(
                            f"malformed dependency-canary squeue row: {line!r}"
                        )
                    continue
                job_id, comment, job_name, state, reason, start, elapsed = fields
                if not job_id.isdigit():
                    continue
                row = {
                    "job_id": job_id,
                    "comment": comment.strip(),
                    "job_name": job_name.strip(),
                    "state": _dependency_state(state),
                    "exit_code": "",
                    "reason": reason.strip(),
                    "start": start.strip(),
                    "end": "",
                    "elapsed": elapsed.strip(),
                    "submit_line": "",
                    "active": "true",
                }
                previous = live.get(job_id)
                if previous is not None and previous != row:
                    raise SlurmFleetCanaryError(
                        f"ambiguous dependency-canary squeue rows for {job_id}"
                    )
                live[job_id] = row
                continue
            if len(fields) < 10:
                if line.strip():
                    raise SlurmFleetCanaryError(
                        f"malformed dependency-canary sacct row: {line!r}"
                    )
                continue
            (
                job_id,
                stored_comment,
                job_name,
                state,
                exit_code,
                reason,
                start,
                end,
                elapsed,
            ) = fields[:9]
            submit_line = "|".join(fields[9:])
            if not job_id.isdigit():
                continue
            normalized_comment = stored_comment.strip()
            derived_comment = _dependency_submit_comment(submit_line)
            if (
                normalized_comment.lower() in {"", "(null)", "null", "none"}
            ):
                normalized_comment = derived_comment
            elif derived_comment and normalized_comment != derived_comment:
                raise SlurmFleetCanaryError(
                    f"dependency-canary accounting comment conflict for {job_id}"
                )
            row = {
                "job_id": job_id,
                "comment": normalized_comment,
                "job_name": job_name.strip(),
                "state": _dependency_state(state),
                "exit_code": exit_code.strip(),
                "reason": reason.strip(),
                "start": start.strip(),
                "end": end.strip(),
                "elapsed": elapsed.strip(),
                "submit_line": submit_line.strip(),
                "active": "false",
            }
            previous = history.get(job_id)
            if previous is not None and previous != row:
                raise SlurmFleetCanaryError(
                    f"ambiguous dependency-canary sacct rows for {job_id}"
                )
            history[job_id] = row
    joined: dict[str, dict[str, str]] = {}
    for job_id in set(live) | set(history):
        active = live.get(job_id)
        terminal = history.get(job_id)
        if active is not None and terminal is not None and (
            active["comment"] != terminal["comment"]
            or active["job_name"] != terminal["job_name"]
        ):
            raise SlurmFleetCanaryError(
                f"dependency-canary scheduler identity conflicts for {job_id}"
            )
        joined[job_id] = active or terminal  # type: ignore[assignment]
    return {
        "complete_squeue_truth": True,
        "complete_sacct_truth": True,
        "raw": raw,
        "live": live,
        "history": history,
        "joined": joined,
    }


def _dependency_config_check(
    *,
    root: Path,
    runner: Runner,
    now: float,
) -> dict[str, Any]:
    argv = ["scontrol", "show", "config"]
    proc = runner(argv)
    match = re.search(
        r"^DependencyParameters\s*=\s*(.*?)\s*$", proc.stdout, re.MULTILINE
    )
    parameters = (
        sorted(
            value
            for value in re.split(r"[,:\s]+", match.group(1).strip())
            if value
        )
        if match
        else []
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_config_check",
        "checked_at": float(now),
        "argv": argv,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        "dependency_parameters": parameters,
        "kill_invalid_depend": "kill_invalid_depend" in parameters,
    }
    if proc.returncode != 0 or payload["kill_invalid_depend"] is not True:
        raise SlurmFleetCanaryError(
            "live Slurm must report DependencyParameters=kill_invalid_depend"
        )
    directory = root / "dependency_config_checks"
    index = len(list(directory.glob("check-*.json"))) + 1 if directory.exists() else 1
    path = directory / f"check-{index:04d}.json"
    _write_immutable_once(
        path,
        payload,
        description="dependency configuration evidence",
    )
    return payload | {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def _load_or_create_dependency_intent(
    *,
    root: Path,
    partition: str,
    time_limit: str,
    latency_bound_seconds: float,
    code_identity: Mapping[str, Any],
    now: float,
    token_factory: Callable[[], str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    scripts = _dependency_job_scripts(
        root=root, partition=partition, time_limit=time_limit
    )
    path = root / DEPENDENCY_INTENT_FILENAME
    if path.exists() or path.is_symlink():
        intent = _read_json(path, description="dependency-cascade intent")
        if (
            intent.get("schema_version") != SCHEMA_VERSION
            or intent.get("kind") != "schema5_dependency_cascade_intent"
            or intent.get("canary_root") != str(root)
            or intent.get("partition") != partition
            or intent.get("time_limit") != time_limit
            or intent.get("alert_latency_bound_seconds")
            != float(latency_bound_seconds)
            or intent.get("code_identity")
            != _validate_code_identity(code_identity)
            or not isinstance(intent.get("scheduler_since"), str)
            or _TOKEN_RE.fullmatch(str(intent.get("token", ""))) is None
        ):
            raise SlurmFleetCanaryError(
                "existing dependency-cascade intent differs from requested contract"
            )
        expected_jobs = {
            role: {
                "script": str((root / "jobs" / f"{role}.sbatch").resolve()),
                "script_sha256": _sha256_bytes(payload),
                "job_name": f"asys-s5-serve-dep-{role}",
                "comment": f"asys:s5-dependency-canary:{intent['token']}:{role}",
            }
            for role, payload in scripts.items()
        }
        if intent.get("jobs") != expected_jobs:
            raise SlurmFleetCanaryError(
                "dependency-cascade intent job contract drifted"
            )
        return intent, scripts
    if root.exists():
        if root.is_symlink() or not root.is_dir() or list(root.iterdir()):
            raise SlurmFleetCanaryError(
                "new dependency-cascade root is unsafe or nonempty"
            )
    else:
        root.mkdir(parents=True)
    token = token_factory()
    if _TOKEN_RE.fullmatch(token) is None:
        raise SlurmFleetCanaryError(
            "dependency-cascade token factory returned an invalid token"
        )
    intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_cascade_intent",
        "created_at": float(now),
        "scheduler_since": _dependency_scheduler_since(now),
        "canary_root": str(root),
        "partition": partition,
        "time_limit": time_limit,
        "alert_latency_bound_seconds": float(latency_bound_seconds),
        "token": token,
        "code_identity": _validate_code_identity(code_identity),
        "jobs": {
            role: {
                "script": str((root / "jobs" / f"{role}.sbatch").resolve()),
                "script_sha256": _sha256_bytes(payload),
                "job_name": f"asys-s5-serve-dep-{role}",
                "comment": f"asys:s5-dependency-canary:{token}:{role}",
            }
            for role, payload in scripts.items()
        },
    }
    _write_immutable_once(
        path, intent, description="marker-first dependency-cascade intent"
    )
    return intent, scripts


def _ensure_dependency_scripts(
    *, root: Path, intent: Mapping[str, Any], scripts: Mapping[str, bytes]
) -> None:
    for role, payload in scripts.items():
        path = Path(intent["jobs"][role]["script"])
        _write_immutable_once(
            path,
            payload,
            description=f"dependency-cascade {role} sbatch",
            mode=0o555,
        )
        if _sha256_file(path) != intent["jobs"][role]["script_sha256"]:
            raise SlurmFleetCanaryError(
                f"dependency-cascade {role} script hash drifted"
            )


def _dependency_submission_argv(
    *,
    role: str,
    intent: Mapping[str, Any],
    job_ids: Mapping[str, str],
) -> list[str]:
    record = intent["jobs"][role]
    argv = [
        "sbatch",
        "--parsable",
        "--no-requeue",
    ]
    if role == "root":
        argv.append("--hold")
    elif role == "child":
        argv.append(f"--dependency=afterok:{job_ids['root']}")
    elif role == "sentinel":
        argv.append(
            f"--dependency=afterany:{job_ids['root']}:{job_ids['child']}"
        )
    else:
        raise SlurmFleetCanaryError(f"unknown dependency-canary role: {role}")
    argv.extend(
        [
            f"--comment={record['comment']}",
            str(record["script"]),
        ]
    )
    return argv


def _wait_dependency_job_visible(
    *,
    intent: Mapping[str, Any],
    role: str,
    job_id: str | None,
    slurm_user: str,
    runner: Runner,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    timeout: float,
    poll_seconds: float,
) -> dict[str, str]:
    deadline = now_fn() + timeout
    expected = intent["jobs"][role]
    while True:
        snapshot = _dependency_scheduler_snapshot(
            slurm_user=slurm_user,
            since=str(intent["scheduler_since"]),
            runner=runner,
        )
        matches = [
            row
            for row in snapshot["joined"].values()
            if row["comment"] == expected["comment"]
        ]
        if len(matches) > 1:
            raise SlurmFleetCanaryError(
                f"dependency-cascade {role} intent maps to duplicate jobs"
            )
        if len(matches) == 1:
            row = matches[0]
            if (
                row["job_name"] != expected["job_name"]
                or (job_id is not None and row["job_id"] != job_id)
            ):
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} scheduler identity drifted"
                )
            return row
        if now_fn() >= deadline:
            raise SlurmFleetCanaryError(
                f"dependency-cascade {role} job did not become visible"
            )
        sleep_fn(poll_seconds)


def _submit_dependency_jobs(
    *,
    root: Path,
    intent: Mapping[str, Any],
    slurm_user: str,
    runner: Runner,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    visibility_timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    receipt_path = root / DEPENDENCY_RECEIPT_FILENAME
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read_json(
            receipt_path, description="dependency-cascade submission receipt"
        )
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("kind")
            != "schema5_dependency_cascade_submission"
            or receipt.get("intent_sha256")
            != _sha256_file(root / DEPENDENCY_INTENT_FILENAME)
        ):
            raise SlurmFleetCanaryError(
                "dependency-cascade submission receipt drifted"
            )
        return receipt
    job_ids: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for role in ("root", "child", "sentinel"):
        submission_root = root / "submissions" / role
        job_intent_path = submission_root / "INTENT.json"
        accepted_path = submission_root / "ACCEPTED.json"
        argv = _dependency_submission_argv(
            role=role, intent=intent, job_ids=job_ids
        )
        job_intent = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_dependency_job_submission_intent",
            "role": role,
            "comment": intent["jobs"][role]["comment"],
            "job_name": intent["jobs"][role]["job_name"],
            "script": intent["jobs"][role]["script"],
            "script_sha256": intent["jobs"][role]["script_sha256"],
            "argv": argv,
            "created_at": float(now_fn()),
        }
        if job_intent_path.exists() or job_intent_path.is_symlink():
            persisted = _read_json(
                job_intent_path,
                description=f"dependency-cascade {role} submission intent",
            )
            comparable = dict(job_intent)
            comparable["created_at"] = persisted.get("created_at")
            if persisted != comparable:
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} submission intent drifted"
                )
            job_intent = persisted
        else:
            _write_immutable_once(
                job_intent_path,
                job_intent,
                description=f"dependency-cascade {role} submission intent",
            )
        if accepted_path.exists() or accepted_path.is_symlink():
            accepted = _read_json(
                accepted_path,
                description=f"dependency-cascade {role} acceptance",
            )
            job_id = str(accepted.get("job_id", ""))
            if (
                accepted.get("schema_version") != SCHEMA_VERSION
                or accepted.get("kind")
                != "schema5_dependency_job_submission_accepted"
                or accepted.get("role") != role
                or accepted.get("intent_sha256") != _sha256_file(job_intent_path)
                or not job_id.isdigit()
            ):
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} acceptance drifted"
                )
            job_ids[role] = job_id
            records.append(accepted)
            continue
        visible: dict[str, str] | None = None
        try:
            visible = _wait_dependency_job_visible(
                intent=intent,
                role=role,
                job_id=None,
                slurm_user=slurm_user,
                runner=runner,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                timeout=0.0,
                poll_seconds=poll_seconds,
            )
        except SlurmFleetCanaryError as exc:
            if "did not become visible" not in str(exc):
                raise
        submission_result: dict[str, Any] | None = None
        if visible is None:
            proc = runner(argv)
            submission_result = {
                "argv": argv,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
                "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
            }
            if proc.returncode != 0:
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} sbatch failed rc={proc.returncode}: "
                    f"{proc.stderr.strip()[:500]}"
                )
            job_id = proc.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} sbatch returned invalid job id"
                )
            visible = _wait_dependency_job_visible(
                intent=intent,
                role=role,
                job_id=job_id,
                slurm_user=slurm_user,
                runner=runner,
                now_fn=now_fn,
                sleep_fn=sleep_fn,
                timeout=visibility_timeout,
                poll_seconds=poll_seconds,
            )
        job_id = visible["job_id"]
        spool_path = submission_root / "SPOOLED_BATCH_SCRIPT.sbatch"
        with tempfile.NamedTemporaryFile(
            prefix=".spooled.",
            suffix=".sbatch",
            dir=submission_root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            proc = runner(
                [
                    "scontrol",
                    "write",
                    "batch_script",
                    job_id,
                    str(temporary),
                ]
            )
            if proc.returncode != 0:
                raise SlurmFleetCanaryError(
                    f"dependency-cascade {role} spooled-script query failed: "
                    f"{proc.stderr.strip()[:500]}"
                )
            payload = temporary.read_bytes()
            _write_immutable_once(
                spool_path,
                payload,
                description=f"dependency-cascade {role} spooled script",
            )
        finally:
            temporary.unlink(missing_ok=True)
        if (
            _sha256_file(spool_path) != intent["jobs"][role]["script_sha256"]
            or spool_path.read_bytes()
            != Path(intent["jobs"][role]["script"]).read_bytes()
        ):
            raise SlurmFleetCanaryError(
                f"dependency-cascade {role} spooled script differs"
            )
        accepted = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_dependency_job_submission_accepted",
            "role": role,
            "job_id": job_id,
            "intent": str(job_intent_path.resolve()),
            "intent_sha256": _sha256_file(job_intent_path),
            "comment": intent["jobs"][role]["comment"],
            "job_name": intent["jobs"][role]["job_name"],
            "script": intent["jobs"][role]["script"],
            "script_sha256": intent["jobs"][role]["script_sha256"],
            "spooled_script": str(spool_path.resolve()),
            "spooled_script_sha256": _sha256_file(spool_path),
            "reconciled_after_boundary": submission_result is None,
            "submission_result": submission_result,
            "recorded_at": float(now_fn()),
        }
        _write_immutable_once(
            accepted_path,
            accepted,
            description=f"dependency-cascade {role} acceptance",
        )
        job_ids[role] = job_id
        records.append(accepted)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_cascade_submission",
        "created_at": float(now_fn()),
        "intent": str((root / DEPENDENCY_INTENT_FILENAME).resolve()),
        "intent_sha256": _sha256_file(root / DEPENDENCY_INTENT_FILENAME),
        "all_jobs_initially_fenced": True,
        "root_initial_hold": True,
        "jobs": records,
    }
    receipt["receipt_id"] = _sha256_bytes(_canonical_bytes(receipt))
    _write_immutable_once(
        receipt_path,
        receipt,
        description="dependency-cascade submission receipt",
    )
    return receipt


def _dependency_show_job(
    *, job_id: str, runner: Runner
) -> dict[str, Any]:
    argv = ["scontrol", "show", "job", "-o", job_id]
    proc = runner(argv)
    fields: dict[str, str] = {}
    if proc.returncode == 0:
        for token in proc.stdout.strip().split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
    return {
        "argv": argv,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        "fields": fields,
    }


def _release_dependency_root(
    *,
    root: Path,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    slurm_user: str,
    runner: Runner,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    visibility_timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    path = root / DEPENDENCY_RELEASE_COMPLETE_FILENAME
    if path.exists() or path.is_symlink():
        return _read_json(path, description="dependency root release completion")
    job_by_role = {row["role"]: row for row in receipt["jobs"]}
    root_id = str(job_by_role["root"]["job_id"])
    release_intent_path = root / DEPENDENCY_RELEASE_INTENT_FILENAME
    release_intent = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_root_release_intent",
        "created_at": float(now_fn()),
        "job_id": root_id,
        "comment": intent["jobs"]["root"]["comment"],
        "submission_receipt": str(
            (root / DEPENDENCY_RECEIPT_FILENAME).resolve()
        ),
        "submission_receipt_sha256": _sha256_file(
            root / DEPENDENCY_RECEIPT_FILENAME
        ),
        "command": ["scontrol", "release", root_id],
    }
    if release_intent_path.exists() or release_intent_path.is_symlink():
        persisted = _read_json(
            release_intent_path, description="dependency root release intent"
        )
        comparable = dict(release_intent)
        comparable["created_at"] = persisted.get("created_at")
        if persisted != comparable:
            raise SlurmFleetCanaryError(
                "dependency root release intent drifted"
            )
        release_intent = persisted
    else:
        _write_immutable_once(
            release_intent_path,
            release_intent,
            description="marker-first dependency root release intent",
        )
    config = _dependency_config_check(
        root=root, runner=runner, now=now_fn()
    )
    before = _dependency_show_job(job_id=root_id, runner=runner)
    reason = str(before["fields"].get("Reason", ""))
    state = _dependency_state(str(before["fields"].get("JobState", "")))
    attempt_directory = root / "release_attempts"
    prior_intents = sorted(attempt_directory.glob("attempt-*.intent.json"))
    performed = False
    result: dict[str, Any] | None = None
    if state == "PENDING" and reason.lower() == "jobhelduser":
        index = len(prior_intents) + 1
        attempt_path = attempt_directory / f"attempt-{index:04d}.intent.json"
        attempt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_dependency_root_release_attempt_intent",
            "attempt": index,
            "created_at": float(now_fn()),
            "job_id": root_id,
            "command": release_intent["command"],
            "config_check": config["path"],
            "config_check_sha256": config["sha256"],
        }
        _write_immutable_once(
            attempt_path,
            attempt,
            description="dependency root release attempt intent",
        )
        proc = runner(release_intent["command"])
        performed = True
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_dependency_root_release_attempt_result",
            "attempt": index,
            "completed_at": float(now_fn()),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
            "intent": str(attempt_path.resolve()),
            "intent_sha256": _sha256_file(attempt_path),
        }
        _write_immutable_once(
            attempt_directory / f"attempt-{index:04d}.result.json",
            result,
            description="dependency root release attempt result",
        )
        if proc.returncode != 0:
            raise SlurmFleetCanaryError(
                f"dependency root release failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
    elif not prior_intents:
        raise SlurmFleetCanaryError(
            "dependency root was not held and has no durable release attempt"
        )
    deadline = now_fn() + visibility_timeout
    after = _dependency_show_job(job_id=root_id, runner=runner)
    while (
        after["returncode"] == 0
        and _dependency_state(str(after["fields"].get("JobState", "")))
        == "PENDING"
        and str(after["fields"].get("Reason", "")).lower() == "jobhelduser"
    ):
        if now_fn() >= deadline:
            raise SlurmFleetCanaryError(
                "dependency root remained held after exact release"
            )
        sleep_fn(poll_seconds)
        after = _dependency_show_job(job_id=root_id, runner=runner)
    snapshot = _dependency_scheduler_snapshot(
        slurm_user=slurm_user,
        since=str(intent["scheduler_since"]),
        runner=runner,
    )
    observed = snapshot["joined"].get(root_id)
    if observed is None:
        raise SlurmFleetCanaryError(
            "released dependency root disappeared from squeue+sacct truth"
        )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_root_release_complete",
        "completed_at": float(now_fn()),
        "job_id": root_id,
        "release_intent": str(release_intent_path.resolve()),
        "release_intent_sha256": _sha256_file(release_intent_path),
        "config_check": config["path"],
        "config_check_sha256": config["sha256"],
        "dependency_parameters": config["dependency_parameters"],
        "kill_invalid_depend": config["kill_invalid_depend"],
        "state_before": before,
        "state_after": after,
        "scheduler_observation": observed,
        "release_performed_this_run": performed,
        "release_result": result,
        "reconciled_after_boundary": not performed,
    }
    _write_immutable_once(
        path,
        completion,
        description="dependency root release completion",
    )
    return completion


def _parse_dependency_time(value: str) -> float:
    normalized = value.strip()
    if normalized.lower() in {
        "",
        "unknown",
        "none",
        "n/a",
        "(null)",
    }:
        raise SlurmFleetCanaryError(
            f"dependency-canary scheduler timestamp is absent: {value!r}"
        )
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SlurmFleetCanaryError(
            f"invalid dependency-canary scheduler timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _wait_dependency_terminal(
    *,
    root: Path,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    slurm_user: str,
    runner: Runner,
    now_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    terminal_timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    path = root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME
    if path.exists() or path.is_symlink():
        return _read_json(path, description="dependency terminal evidence")
    ids = {row["role"]: str(row["job_id"]) for row in receipt["jobs"]}
    deadline = now_fn() + terminal_timeout
    snapshot: dict[str, Any]
    histories: dict[str, dict[str, str]]
    active_states = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
    }
    while True:
        snapshot = _dependency_scheduler_snapshot(
            slurm_user=slurm_user,
            since=str(intent["scheduler_since"]),
            runner=runner,
        )
        histories = {}
        for role, job_id in ids.items():
            row = snapshot["history"].get(job_id)
            if row is not None and row["state"] not in active_states:
                histories[role] = row
        marker_ready = (
            root / DEPENDENCY_ALERT_MARKER_FILENAME
        ).is_file()
        if len(histories) == 3 and marker_ready:
            break
        if now_fn() >= deadline:
            raise SlurmFleetCanaryError(
                "dependency-cascade jobs did not reach terminal truth and alert "
                "within the configured timeout"
            )
        sleep_fn(poll_seconds)
    root_row = histories["root"]
    child_row = histories["child"]
    sentinel_row = histories["sentinel"]
    if root_row["state"] != "FAILED" or root_row["exit_code"] != "42:0":
        raise SlurmFleetCanaryError(
            "dependency-cascade root did not fail with the deliberate exit 42"
        )
    reason = child_row["reason"].lower().replace("_", "")
    if (
        child_row["state"] != "CANCELLED"
        or "dependency" not in reason
        or "never" not in reason
        or child_row["start"].strip().lower()
        not in {"", "unknown", "none", "n/a", "(null)"}
        or child_row["elapsed"].strip()
        not in {"", "00:00:00", "0:00", "00:00"}
    ):
        raise SlurmFleetCanaryError(
            "afterok child was not a never-started dependency cancellation"
        )
    if sentinel_row["state"] != "COMPLETED" or sentinel_row["exit_code"] != "0:0":
        raise SlurmFleetCanaryError(
            "aggregate afterany dependency sentinel did not complete"
        )
    child_marker = root / DEPENDENCY_FORBIDDEN_CHILD_MARKER_FILENAME
    if child_marker.exists() or child_marker.is_symlink():
        raise SlurmFleetCanaryError(
            "dependency-cascade afterok child executed unexpectedly"
        )
    alert_path = root / DEPENDENCY_ALERT_MARKER_FILENAME
    alert = _read_json(alert_path, description="dependency alert marker")
    if (
        set(alert)
        != {"schema_version", "kind", "job_id", "started_timestamp"}
        or alert.get("schema_version") != 1
        or alert.get("kind") != "schema5_dependency_alert_executed"
        or alert.get("job_id") != ids["sentinel"]
        or not isinstance(alert.get("started_timestamp"), (int, float))
        or isinstance(alert.get("started_timestamp"), bool)
    ):
        raise SlurmFleetCanaryError(
            "dependency alert marker identity is invalid"
        )
    root_end = _parse_dependency_time(root_row["end"])
    sentinel_start = _parse_dependency_time(sentinel_row["start"])
    latency = sentinel_start - root_end
    bound = float(intent["alert_latency_bound_seconds"])
    if latency < 0 or latency > bound:
        raise SlurmFleetCanaryError(
            f"dependency alert latency {latency:.3f}s exceeds bound {bound:.3f}s"
        )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_terminal_scheduler_evidence",
        "recorded_at": float(now_fn()),
        "jobs": histories,
        "complete_squeue_truth": snapshot["complete_squeue_truth"],
        "complete_sacct_truth": snapshot["complete_sacct_truth"],
        "raw_scheduler_queries": snapshot["raw"],
        "root_terminal_timestamp": root_end,
        "sentinel_start_timestamp": sentinel_start,
        "alert_latency_seconds": latency,
        "alert_latency_bound_seconds": bound,
        "alert_marker": str(alert_path.resolve()),
        "alert_marker_sha256": _sha256_file(alert_path),
        "child_never_started": True,
    }
    _write_immutable_once(
        path,
        evidence,
        description="dependency-cascade terminal scheduler evidence",
    )
    return evidence


def _dependency_inventory(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if path.name == DEPENDENCY_COMPLETE_FILENAME and path.parent == root:
            continue
        if path.is_symlink():
            raise SlurmFleetCanaryError(
                f"dependency-cascade contains symlink: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise SlurmFleetCanaryError(
                f"dependency-cascade contains unsafe entry: {relative}"
            )
        result.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return result


def _seal_dependency_complete(
    *,
    root: Path,
    intent: Mapping[str, Any],
    receipt: Mapping[str, Any],
    release: Mapping[str, Any],
    terminal: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    path = root / DEPENDENCY_COMPLETE_FILENAME
    if path.exists() or path.is_symlink():
        return verify_dependency_cascade_complete(root)
    for artifact in root.rglob("*"):
        if artifact.is_symlink():
            raise SlurmFleetCanaryError(
                f"dependency-cascade contains symlink: {artifact}"
            )
        if artifact.is_file():
            artifact.chmod(stat.S_IMODE(artifact.stat().st_mode) & ~0o222)
    inventory = _dependency_inventory(root)
    jobs = {row["role"]: row["job_id"] for row in receipt["jobs"]}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_dependency_cascade_complete",
        "completed_at": float(now),
        "canary_root": str(root),
        "code_identity": intent["code_identity"],
        "intent_path": str((root / DEPENDENCY_INTENT_FILENAME).resolve()),
        "intent_sha256": _sha256_file(root / DEPENDENCY_INTENT_FILENAME),
        "submission_receipt_path": str(
            (root / DEPENDENCY_RECEIPT_FILENAME).resolve()
        ),
        "submission_receipt_sha256": _sha256_file(
            root / DEPENDENCY_RECEIPT_FILENAME
        ),
        "submission_receipt_id": receipt["receipt_id"],
        "release_complete_path": str(
            (root / DEPENDENCY_RELEASE_COMPLETE_FILENAME).resolve()
        ),
        "release_complete_sha256": _sha256_file(
            root / DEPENDENCY_RELEASE_COMPLETE_FILENAME
        ),
        "terminal_evidence_path": str(
            (root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME).resolve()
        ),
        "terminal_evidence_sha256": _sha256_file(
            root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME
        ),
        "dependency_parameters": release["dependency_parameters"],
        "kill_invalid_depend": release["kill_invalid_depend"],
        "root_initial_hold": receipt["root_initial_hold"],
        "root_job_id": jobs["root"],
        "child_job_id": jobs["child"],
        "sentinel_job_id": jobs["sentinel"],
        "root_state": terminal["jobs"]["root"]["state"],
        "child_state": terminal["jobs"]["child"]["state"],
        "sentinel_state": terminal["jobs"]["sentinel"]["state"],
        "child_never_started": terminal["child_never_started"],
        "alert_latency_seconds": terminal["alert_latency_seconds"],
        "alert_latency_bound_seconds": terminal[
            "alert_latency_bound_seconds"
        ],
        "alert_marker_sha256": terminal["alert_marker_sha256"],
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": _sha256_bytes(
            _canonical_compact_bytes(inventory)
        ),
    }
    payload["canary_id"] = _sha256_bytes(_canonical_bytes(payload))
    _write_immutable_once(
        path,
        payload,
        description="marker-last dependency-cascade completion",
    )
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(stat.S_IMODE(directory.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
    return payload


def verify_dependency_cascade_complete(root: Path) -> dict[str, Any]:
    root = _validate_isolated_root(root)
    complete = _read_json(
        root / DEPENDENCY_COMPLETE_FILENAME,
        description="dependency-cascade completion",
    )
    intent = _read_json(
        root / DEPENDENCY_INTENT_FILENAME,
        description="dependency-cascade intent",
    )
    receipt = _read_json(
        root / DEPENDENCY_RECEIPT_FILENAME,
        description="dependency-cascade submission receipt",
    )
    release = _read_json(
        root / DEPENDENCY_RELEASE_COMPLETE_FILENAME,
        description="dependency root release completion",
    )
    terminal = _read_json(
        root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME,
        description="dependency terminal evidence",
    )
    inventory = _dependency_inventory(root)
    identity = dict(complete)
    canary_id = identity.pop("canary_id", None)
    required = {
        "schema_version",
        "kind",
        "completed_at",
        "canary_root",
        "code_identity",
        "intent_path",
        "intent_sha256",
        "submission_receipt_path",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "release_complete_path",
        "release_complete_sha256",
        "terminal_evidence_path",
        "terminal_evidence_sha256",
        "dependency_parameters",
        "kill_invalid_depend",
        "root_initial_hold",
        "root_job_id",
        "child_job_id",
        "sentinel_job_id",
        "root_state",
        "child_state",
        "sentinel_state",
        "child_never_started",
        "alert_latency_seconds",
        "alert_latency_bound_seconds",
        "alert_marker_sha256",
        "artifact_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "canary_id",
    }
    jobs = {row["role"]: row["job_id"] for row in receipt.get("jobs", [])}
    if (
        set(complete) != required
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind") != "schema5_dependency_cascade_complete"
        or complete.get("canary_root") != str(root)
        or complete.get("code_identity") != intent.get("code_identity")
        or _validate_code_identity(complete.get("code_identity", {}))
        != complete.get("code_identity")
        or complete.get("intent_path")
        != str((root / DEPENDENCY_INTENT_FILENAME).resolve())
        or complete.get("intent_sha256")
        != _sha256_file(root / DEPENDENCY_INTENT_FILENAME)
        or complete.get("submission_receipt_path")
        != str((root / DEPENDENCY_RECEIPT_FILENAME).resolve())
        or complete.get("submission_receipt_sha256")
        != _sha256_file(root / DEPENDENCY_RECEIPT_FILENAME)
        or complete.get("submission_receipt_id") != receipt.get("receipt_id")
        or complete.get("release_complete_path")
        != str((root / DEPENDENCY_RELEASE_COMPLETE_FILENAME).resolve())
        or complete.get("release_complete_sha256")
        != _sha256_file(root / DEPENDENCY_RELEASE_COMPLETE_FILENAME)
        or complete.get("terminal_evidence_path")
        != str((root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME).resolve())
        or complete.get("terminal_evidence_sha256")
        != _sha256_file(root / DEPENDENCY_TERMINAL_EVIDENCE_FILENAME)
        or complete.get("dependency_parameters")
        != release.get("dependency_parameters")
        or complete.get("kill_invalid_depend") is not True
        or release.get("kill_invalid_depend") is not True
        or complete.get("root_initial_hold") is not True
        or receipt.get("root_initial_hold") is not True
        or set(jobs) != {"root", "child", "sentinel"}
        or complete.get("root_job_id") != jobs.get("root")
        or complete.get("child_job_id") != jobs.get("child")
        or complete.get("sentinel_job_id") != jobs.get("sentinel")
        or complete.get("root_state") != "FAILED"
        or complete.get("child_state") != "CANCELLED"
        or complete.get("sentinel_state") != "COMPLETED"
        or complete.get("child_never_started") is not True
        or terminal.get("child_never_started") is not True
        or complete.get("alert_latency_seconds")
        != terminal.get("alert_latency_seconds")
        or complete.get("alert_latency_bound_seconds")
        != terminal.get("alert_latency_bound_seconds")
        or not isinstance(complete.get("alert_latency_seconds"), (int, float))
        or isinstance(complete.get("alert_latency_seconds"), bool)
        or float(complete["alert_latency_seconds"]) < 0
        or float(complete["alert_latency_seconds"])
        > float(complete.get("alert_latency_bound_seconds", -1))
        or complete.get("alert_marker_sha256")
        != terminal.get("alert_marker_sha256")
        or complete.get("artifact_count") != len(inventory)
        or complete.get("artifact_inventory") != inventory
        or complete.get("artifact_inventory_sha256")
        != _sha256_bytes(_canonical_compact_bytes(inventory))
        or canary_id != _sha256_bytes(_canonical_bytes(identity))
    ):
        raise SlurmFleetCanaryError(
            "dependency-cascade completion identity or evidence drifted"
        )
    for artifact in [root, *root.rglob("*")]:
        if artifact.is_symlink():
            raise SlurmFleetCanaryError(
                f"sealed dependency-cascade contains symlink: {artifact}"
            )
        if stat.S_IMODE(artifact.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError(
                f"sealed dependency-cascade entry remains writable: {artifact}"
            )
    return complete


def run_dependency_cascade_canary(
    *,
    root: Path,
    apply: bool = False,
    verify: bool = False,
    partition: str = DEFAULT_PARTITION,
    time_limit: str = DEFAULT_TIME_LIMIT,
    alert_latency_bound_seconds: float = DEFAULT_DEPENDENCY_ALERT_LATENCY_SECONDS,
    scheduler_user: str | None = None,
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    terminal_timeout: float = DEFAULT_TERMINAL_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    runner: Runner = subprocess.run,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    token_factory: Callable[[], str] = lambda: os.urandom(16).hex(),
    code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the exact fail-closed dependency behavior used by recovery."""

    root = _validate_isolated_root(root)
    if apply and verify:
        raise SlurmFleetCanaryError("--apply and --verify are mutually exclusive")
    if (
        not isinstance(alert_latency_bound_seconds, (int, float))
        or isinstance(alert_latency_bound_seconds, bool)
        or float(alert_latency_bound_seconds) <= 0
    ):
        raise SlurmFleetCanaryError(
            "dependency alert latency bound must be positive"
        )
    marker = root / DEPENDENCY_COMPLETE_FILENAME
    if marker.exists() or marker.is_symlink():
        complete = verify_dependency_cascade_complete(root)
        if code_identity is not None and complete.get(
            "code_identity"
        ) != _validate_code_identity(code_identity):
            raise SlurmFleetCanaryError(
                "sealed dependency canary belongs to another code identity"
            )
        return complete
    if verify:
        raise SlurmFleetCanaryError(
            f"dependency-cascade completion is missing: {marker}"
        )
    scripts = _dependency_job_scripts(
        root=root, partition=partition, time_limit=time_limit
    )
    if not apply:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_dependency_cascade_dry_run",
            "apply": False,
            "canary_root": str(root),
            "partition": partition,
            "time_limit": time_limit,
            "alert_latency_bound_seconds": float(
                alert_latency_bound_seconds
            ),
            "job_count": 3,
            "scripts": {
                role: payload.decode("utf-8")
                for role, payload in scripts.items()
            },
            "root_initial_hold": True,
            "dependency_contract": {
                "child": "afterok:root",
                "sentinel": "afterany:root:child",
            },
        }
    bound_code_identity = (
        derive_code_identity()
        if code_identity is None
        else _validate_code_identity(code_identity)
    )
    user = scheduler_user or os.environ.get("USER", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise SlurmFleetCanaryError(
            "dependency-canary scheduler user is absent or unsafe"
        )
    intent, scripts = _load_or_create_dependency_intent(
        root=root,
        partition=partition,
        time_limit=time_limit,
        latency_bound_seconds=float(alert_latency_bound_seconds),
        code_identity=bound_code_identity,
        now=now_fn(),
        token_factory=token_factory,
    )
    _ensure_dependency_scripts(root=root, intent=intent, scripts=scripts)
    receipt = _submit_dependency_jobs(
        root=root,
        intent=intent,
        slurm_user=user,
        runner=runner,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        visibility_timeout=visibility_timeout,
        poll_seconds=poll_seconds,
    )
    release = _release_dependency_root(
        root=root,
        intent=intent,
        receipt=receipt,
        slurm_user=user,
        runner=runner,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        visibility_timeout=visibility_timeout,
        poll_seconds=poll_seconds,
    )
    terminal = _wait_dependency_terminal(
        root=root,
        intent=intent,
        receipt=receipt,
        slurm_user=user,
        runner=runner,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        terminal_timeout=terminal_timeout,
        poll_seconds=poll_seconds,
    )
    _seal_dependency_complete(
        root=root,
        intent=intent,
        receipt=receipt,
        release=release,
        terminal=terminal,
        now=now_fn(),
    )
    return verify_dependency_cascade_complete(root)


def _load_or_create_composite_intent(
    *,
    root: Path,
    partition: str,
    time_limit: str,
    turnover_cycles: int,
    turnover_drain_seconds: float,
    dependency_alert_latency_seconds: float,
    code_identity: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    path = root / COMPOSITE_INTENT_FILENAME
    proposed = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_composite_canary_intent",
        "created_at": float(now),
        "canary_root": str(root),
        "transaction_root": str(
            (root / TRANSACTION_COMPONENT_DIRECTORY).resolve()
        ),
        "dependency_root": str(
            (root / DEPENDENCY_COMPONENT_DIRECTORY).resolve()
        ),
        "turnover_root": str((root / TURNOVER_COMPONENT_DIRECTORY).resolve()),
        "partition": partition,
        "time_limit": time_limit,
        "turnover_cycles": turnover_cycles,
        "turnover_drain_seconds": float(turnover_drain_seconds),
        "dependency_alert_latency_seconds": float(
            dependency_alert_latency_seconds
        ),
        "code_identity": dict(code_identity),
    }
    if path.exists() or path.is_symlink():
        intent = _read_json(path, description="composite canary intent")
        comparable = dict(proposed)
        comparable["created_at"] = intent.get("created_at")
        if (
            intent != comparable
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise SlurmFleetCanaryError(
                "existing composite canary intent differs from requested contract"
            )
        return intent
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise SlurmFleetCanaryError(f"composite canary root is unsafe: {root}")
        entries = list(root.iterdir())
        if entries:
            raise SlurmFleetCanaryError(
                "new composite canary root contains artifacts before marker-first "
                "intent: "
                + ", ".join(sorted(item.name for item in entries))
            )
    else:
        root.mkdir(parents=True)
    _write_immutable_once(
        path,
        proposed,
        description="composite canary marker-first intent",
    )
    return proposed


def _seal_and_publish_composite_complete(
    *,
    root: Path,
    intent: Mapping[str, Any],
    transaction: Mapping[str, Any],
    dependency: Mapping[str, Any],
    turnover: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    path = root / COMPLETE_FILENAME
    if path.exists() or path.is_symlink():
        return verify_composite_complete(root)
    transaction_root = root / TRANSACTION_COMPONENT_DIRECTORY
    dependency_root = root / DEPENDENCY_COMPONENT_DIRECTORY
    turnover_root = root / TURNOVER_COMPONENT_DIRECTORY
    transaction_marker = transaction_root / COMPLETE_FILENAME
    dependency_marker = dependency_root / DEPENDENCY_COMPLETE_FILENAME
    turnover_marker = turnover_root / TURNOVER_COMPLETE_FILENAME
    if (
        transaction.get("schema_version") != SCHEMA_VERSION
        or transaction.get("kind")
        != "schema5_slurm_fleet_transaction_canary_complete"
        or dependency.get("schema_version") != SCHEMA_VERSION
        or dependency.get("kind") != "schema5_dependency_cascade_complete"
        or turnover.get("schema_version") != SCHEMA_VERSION
        or turnover.get("kind")
        != "schema5_warm_handoff_turnover_canary_complete"
        or turnover.get("cycles_completed", 0) < 2
        or transaction.get("code_identity") != intent["code_identity"]
        or dependency.get("code_identity") != intent["code_identity"]
        or turnover.get("code_identity") != intent["code_identity"]
        or dependency.get("kill_invalid_depend") is not True
        or dependency.get("child_never_started") is not True
        or dependency.get("root_initial_hold") is not True
        or dependency.get("alert_latency_seconds", float("inf"))
        > dependency.get("alert_latency_bound_seconds", -1)
    ):
        raise SlurmFleetCanaryError(
            "component evidence cannot satisfy the composite canary contract"
        )
    for artifact in root.rglob("*"):
        if artifact.is_symlink():
            raise SlurmFleetCanaryError(
                f"composite canary contains a symlink: {artifact}"
            )
        if artifact.is_file():
            artifact.chmod(stat.S_IMODE(artifact.stat().st_mode) & ~0o222)
    inventory = _artifact_inventory(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "schema5_slurm_fleet_composite_canary_complete",
        "completed_at": float(now),
        "canary_root": str(root),
        "composite_intent_path": str(
            (root / COMPOSITE_INTENT_FILENAME).resolve()
        ),
        "composite_intent_sha256": _sha256_file(
            root / COMPOSITE_INTENT_FILENAME
        ),
        "code_identity": intent["code_identity"],
        "transaction_root": str(transaction_root.resolve()),
        "transaction_marker_path": str(transaction_marker.resolve()),
        "transaction_marker_sha256": _sha256_file(transaction_marker),
        "transaction_canary_id": transaction["canary_id"],
        "transaction_job_id": transaction["job_id"],
        "dependency_root": str(dependency_root.resolve()),
        "dependency_marker_path": str(dependency_marker.resolve()),
        "dependency_marker_sha256": _sha256_file(dependency_marker),
        "dependency_canary_id": dependency["canary_id"],
        "dependency_parameters": dependency["dependency_parameters"],
        "dependency_kill_invalid_depend": dependency["kill_invalid_depend"],
        "dependency_root_initial_hold": dependency["root_initial_hold"],
        "dependency_child_never_started": dependency["child_never_started"],
        "dependency_alert_latency_seconds": dependency[
            "alert_latency_seconds"
        ],
        "dependency_alert_latency_bound_seconds": dependency[
            "alert_latency_bound_seconds"
        ],
        "turnover_root": str(turnover_root.resolve()),
        "turnover_marker_path": str(turnover_marker.resolve()),
        "turnover_marker_sha256": _sha256_file(turnover_marker),
        "turnover_canary_id": turnover["canary_id"],
        "turnover_cycles_completed": turnover["cycles_completed"],
        "turnover_allocations_submitted": turnover["allocations_submitted"],
        "turnover_maximum_physical_allocations_observed": turnover[
            "maximum_physical_allocations_observed"
        ],
        "turnover_maximum_extra_gpus_observed": turnover[
            "maximum_extra_gpus_observed"
        ],
        "turnover_all_effective_requeue": turnover[
            "all_effective_requeue"
        ],
        "turnover_continuous_routed_endpoint_evidence": turnover[
            "continuous_routed_endpoint_evidence"
        ],
        "production_overlap_gpu_ceiling": turnover[
            "production_overlap_gpu_ceiling"
        ],
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": _sha256_bytes(
            _canonical_compact_bytes(inventory)
        ),
    }
    payload["canary_id"] = _sha256_bytes(_canonical_bytes(payload))
    _write_immutable_once(
        path,
        payload,
        description="marker-last composite canary completion",
    )
    for child in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        child.chmod(stat.S_IMODE(child.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
    return payload


def verify_composite_complete(root: Path) -> dict[str, Any]:
    """Verify the canonical transaction+turnover canary as one sealed contract."""

    root = _validate_isolated_root(root)
    complete = _read_json(
        root / COMPLETE_FILENAME, description="composite canary completion"
    )
    intent = _read_json(
        root / COMPOSITE_INTENT_FILENAME,
        description="composite canary intent",
    )
    transaction_root = root / TRANSACTION_COMPONENT_DIRECTORY
    dependency_root = root / DEPENDENCY_COMPONENT_DIRECTORY
    turnover_root = root / TURNOVER_COMPONENT_DIRECTORY
    transaction = verify_complete(transaction_root)
    dependency = verify_dependency_cascade_complete(dependency_root)
    turnover = verify_turnover_complete(turnover_root)
    inventory = _artifact_inventory(root)
    identity = dict(complete)
    canary_id = identity.pop("canary_id", None)
    required = {
        "schema_version",
        "kind",
        "completed_at",
        "canary_root",
        "composite_intent_path",
        "composite_intent_sha256",
        "code_identity",
        "transaction_root",
        "transaction_marker_path",
        "transaction_marker_sha256",
        "transaction_canary_id",
        "transaction_job_id",
        "dependency_root",
        "dependency_marker_path",
        "dependency_marker_sha256",
        "dependency_canary_id",
        "dependency_parameters",
        "dependency_kill_invalid_depend",
        "dependency_root_initial_hold",
        "dependency_child_never_started",
        "dependency_alert_latency_seconds",
        "dependency_alert_latency_bound_seconds",
        "turnover_root",
        "turnover_marker_path",
        "turnover_marker_sha256",
        "turnover_canary_id",
        "turnover_cycles_completed",
        "turnover_allocations_submitted",
        "turnover_maximum_physical_allocations_observed",
        "turnover_maximum_extra_gpus_observed",
        "turnover_all_effective_requeue",
        "turnover_continuous_routed_endpoint_evidence",
        "production_overlap_gpu_ceiling",
        "artifact_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "canary_id",
    }
    transaction_marker = transaction_root / COMPLETE_FILENAME
    dependency_marker = dependency_root / DEPENDENCY_COMPLETE_FILENAME
    turnover_marker = turnover_root / TURNOVER_COMPLETE_FILENAME
    if (
        set(complete) != required
        or complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("kind")
        != "schema5_slurm_fleet_composite_canary_complete"
        or complete.get("canary_root") != str(root)
        or complete.get("composite_intent_path")
        != str((root / COMPOSITE_INTENT_FILENAME).resolve())
        or complete.get("composite_intent_sha256")
        != _sha256_file(root / COMPOSITE_INTENT_FILENAME)
        or complete.get("code_identity") != intent.get("code_identity")
        or complete.get("code_identity") != transaction.get("code_identity")
        or complete.get("code_identity") != dependency.get("code_identity")
        or complete.get("code_identity") != turnover.get("code_identity")
        or _validate_code_identity(complete.get("code_identity", {}))
        != complete.get("code_identity")
        or complete.get("transaction_root") != str(transaction_root.resolve())
        or complete.get("transaction_marker_path")
        != str(transaction_marker.resolve())
        or complete.get("transaction_marker_sha256")
        != _sha256_file(transaction_marker)
        or complete.get("transaction_canary_id") != transaction["canary_id"]
        or complete.get("transaction_job_id") != transaction["job_id"]
        or complete.get("dependency_root") != str(dependency_root.resolve())
        or complete.get("dependency_marker_path")
        != str(dependency_marker.resolve())
        or complete.get("dependency_marker_sha256")
        != _sha256_file(dependency_marker)
        or complete.get("dependency_canary_id") != dependency["canary_id"]
        or complete.get("dependency_parameters")
        != dependency["dependency_parameters"]
        or complete.get("dependency_kill_invalid_depend") is not True
        or dependency.get("kill_invalid_depend") is not True
        or complete.get("dependency_root_initial_hold") is not True
        or dependency.get("root_initial_hold") is not True
        or complete.get("dependency_child_never_started") is not True
        or dependency.get("child_never_started") is not True
        or complete.get("dependency_alert_latency_seconds")
        != dependency["alert_latency_seconds"]
        or complete.get("dependency_alert_latency_bound_seconds")
        != dependency["alert_latency_bound_seconds"]
        or complete.get("dependency_alert_latency_seconds", float("inf"))
        > complete.get("dependency_alert_latency_bound_seconds", -1)
        or complete.get("turnover_root") != str(turnover_root.resolve())
        or complete.get("turnover_marker_path")
        != str(turnover_marker.resolve())
        or complete.get("turnover_marker_sha256")
        != _sha256_file(turnover_marker)
        or complete.get("turnover_canary_id") != turnover["canary_id"]
        or complete.get("turnover_cycles_completed")
        != turnover["cycles_completed"]
        or complete.get("turnover_cycles_completed", 0) < 2
        or complete.get("turnover_allocations_submitted")
        != turnover["allocations_submitted"]
        or complete.get("turnover_maximum_physical_allocations_observed")
        != turnover["maximum_physical_allocations_observed"]
        or complete.get("turnover_maximum_physical_allocations_observed") != 2
        or complete.get("turnover_maximum_extra_gpus_observed") != 0
        or complete.get("turnover_all_effective_requeue") != 0
        or complete.get("turnover_continuous_routed_endpoint_evidence")
        is not True
        or complete.get("production_overlap_gpu_ceiling") != 4
        or complete.get("artifact_count") != len(inventory)
        or complete.get("artifact_inventory") != inventory
        or complete.get("artifact_inventory_sha256")
        != _sha256_bytes(_canonical_compact_bytes(inventory))
        or canary_id != _sha256_bytes(_canonical_bytes(identity))
    ):
        raise SlurmFleetCanaryError(
            "composite canary identity, turnover, or inventory evidence drifted"
        )
    for artifact in [root, *root.rglob("*")]:
        if artifact.is_symlink():
            raise SlurmFleetCanaryError(
                f"sealed composite canary contains symlink: {artifact}"
            )
        if stat.S_IMODE(artifact.stat().st_mode) & 0o222:
            raise SlurmFleetCanaryError(
                f"sealed composite canary entry remains writable: {artifact}"
            )
    return complete


def run_composite_canary(
    *,
    root: Path,
    apply: bool = False,
    verify: bool = False,
    partition: str = DEFAULT_PARTITION,
    time_limit: str = DEFAULT_TIME_LIMIT,
    turnover_cycles: int = DEFAULT_TURNOVER_CYCLES,
    turnover_drain_seconds: float = DEFAULT_TURNOVER_DRAIN_SECONDS,
    dependency_alert_latency_seconds: float = (
        DEFAULT_DEPENDENCY_ALERT_LATENCY_SECONDS
    ),
    scheduler_user: str | None = None,
    visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
    terminal_timeout: float = DEFAULT_TERMINAL_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    probe_timeout: float = 10.0,
    runner: Runner = subprocess.run,
    probe: EndpointProbe = _default_endpoint_probe,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    token_factory: Callable[[], str] = lambda: os.urandom(16).hex(),
    code_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _validate_isolated_root(root)
    if apply and verify:
        raise SlurmFleetCanaryError("--apply and --verify are mutually exclusive")
    marker = root / COMPLETE_FILENAME
    if marker.exists() or marker.is_symlink():
        complete = verify_composite_complete(root)
        if code_identity is not None and complete.get(
            "code_identity"
        ) != _validate_code_identity(code_identity):
            raise SlurmFleetCanaryError(
                "sealed composite canary belongs to another code identity"
            )
        return complete
    if verify:
        raise SlurmFleetCanaryError(
            f"composite completion is missing: {marker}"
        )
    if not apply:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "schema5_slurm_fleet_composite_canary_dry_run",
            "apply": False,
            "canary_root": str(root),
            "transaction_root": str(
                (root / TRANSACTION_COMPONENT_DIRECTORY).resolve()
            ),
            "dependency_root": str(
                (root / DEPENDENCY_COMPONENT_DIRECTORY).resolve()
            ),
            "turnover_root": str(
                (root / TURNOVER_COMPONENT_DIRECTORY).resolve()
            ),
            "dependency_plan": run_dependency_cascade_canary(
                root=root / DEPENDENCY_COMPONENT_DIRECTORY,
                partition=partition,
                time_limit=time_limit,
                alert_latency_bound_seconds=dependency_alert_latency_seconds,
                code_identity=code_identity,
            ),
            "turnover_cycles": turnover_cycles,
            "turnover_plan": render_turnover_canary_plan(
                partition=partition,
                time_limit=time_limit,
                cycles=turnover_cycles,
                drain_seconds=turnover_drain_seconds,
            ),
            "marker_last": COMPLETE_FILENAME,
            "code_identity": (
                None
                if code_identity is None
                else _validate_code_identity(code_identity)
            ),
        }
    bound_code_identity = (
        derive_code_identity()
        if code_identity is None
        else _validate_code_identity(code_identity)
    )
    intent = _load_or_create_composite_intent(
        root=root,
        partition=partition,
        time_limit=time_limit,
        turnover_cycles=turnover_cycles,
        turnover_drain_seconds=turnover_drain_seconds,
        dependency_alert_latency_seconds=dependency_alert_latency_seconds,
        code_identity=bound_code_identity,
        now=now_fn(),
    )
    transaction = run_canary(
        root=root / TRANSACTION_COMPONENT_DIRECTORY,
        apply=True,
        partition=partition,
        time_limit=time_limit,
        scheduler_user=scheduler_user,
        visibility_timeout=visibility_timeout,
        terminal_timeout=terminal_timeout,
        poll_seconds=poll_seconds,
        runner=runner,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        token_factory=token_factory,
        code_identity=bound_code_identity,
    )
    dependency = run_dependency_cascade_canary(
        root=root / DEPENDENCY_COMPONENT_DIRECTORY,
        apply=True,
        partition=partition,
        time_limit=time_limit,
        alert_latency_bound_seconds=dependency_alert_latency_seconds,
        scheduler_user=scheduler_user,
        visibility_timeout=visibility_timeout,
        terminal_timeout=terminal_timeout,
        poll_seconds=poll_seconds,
        runner=runner,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        token_factory=token_factory,
        code_identity=bound_code_identity,
    )
    turnover = run_turnover_canary(
        root=root / TURNOVER_COMPONENT_DIRECTORY,
        apply=True,
        partition=partition,
        time_limit=time_limit,
        cycles=turnover_cycles,
        drain_seconds=turnover_drain_seconds,
        scheduler_user=scheduler_user,
        visibility_timeout=visibility_timeout,
        terminal_timeout=terminal_timeout,
        poll_seconds=poll_seconds,
        probe_timeout=probe_timeout,
        runner=runner,
        probe=probe,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        token_factory=token_factory,
        code_identity=bound_code_identity,
    )
    _seal_and_publish_composite_complete(
        root=root,
        intent=intent,
        transaction=transaction,
        dependency=dependency,
        turnover=turnover,
        now=now_fn(),
    )
    return verify_composite_complete(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="run the genuine isolated composite Slurm canary (default is dry-run)",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify sealed composite evidence without contacting Slurm",
    )
    component = parser.add_mutually_exclusive_group()
    component.add_argument(
        "--transaction-only",
        action="store_true",
        help="diagnostic: run only the single admission transaction component",
    )
    component.add_argument(
        "--turnover-only",
        action="store_true",
        help="diagnostic: run only the warm-turnover component",
    )
    component.add_argument(
        "--dependency-only",
        action="store_true",
        help="diagnostic: run only the fail-closed dependency component",
    )
    parser.add_argument("--canary-root", type=Path, default=_default_canary_root())
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--scheduler-user")
    parser.add_argument(
        "--visibility-timeout",
        type=float,
        default=DEFAULT_VISIBILITY_TIMEOUT,
    )
    parser.add_argument(
        "--terminal-timeout",
        type=float,
        default=DEFAULT_TERMINAL_TIMEOUT,
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--turnover-cycles",
        type=int,
        default=DEFAULT_TURNOVER_CYCLES,
        help="number of real warm turnovers (must be 2-4)",
    )
    parser.add_argument(
        "--turnover-drain-seconds",
        type=float,
        default=DEFAULT_TURNOVER_DRAIN_SECONDS,
        help="short canary drain; production remains fixed at 660 seconds",
    )
    parser.add_argument(
        "--dependency-alert-latency-seconds",
        type=float,
        default=DEFAULT_DEPENDENCY_ALERT_LATENCY_SECONDS,
        help="maximum accepted root-failure to afterany-sentinel start latency",
    )
    parser.add_argument("--probe-timeout", type=float, default=10.0)
    parser.add_argument(
        "--durable-git-release-marker",
        type=Path,
        required=True,
        help="sealed marker-last proof that the exact commit and tag are durable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        code_identity = derive_code_identity(args.durable_git_release_marker)
        common = {
            "root": args.canary_root,
            "apply": args.apply,
            "verify": args.verify,
            "partition": args.partition,
            "time_limit": args.time_limit,
            "scheduler_user": args.scheduler_user,
            "visibility_timeout": args.visibility_timeout,
            "terminal_timeout": args.terminal_timeout,
            "poll_seconds": args.poll_seconds,
            "code_identity": code_identity,
        }
        if args.transaction_only:
            result = run_canary(**common)
        elif args.turnover_only:
            result = run_turnover_canary(
                **common,
                cycles=args.turnover_cycles,
                drain_seconds=args.turnover_drain_seconds,
                probe_timeout=args.probe_timeout,
            )
        elif args.dependency_only:
            result = run_dependency_cascade_canary(
                **common,
                alert_latency_bound_seconds=(
                    args.dependency_alert_latency_seconds
                ),
            )
        else:
            result = run_composite_canary(
                **common,
                turnover_cycles=args.turnover_cycles,
                turnover_drain_seconds=args.turnover_drain_seconds,
                dependency_alert_latency_seconds=(
                    args.dependency_alert_latency_seconds
                ),
                probe_timeout=args.probe_timeout,
            )
    except (SlurmFleetCanaryError, tx.FleetTransactionError) as exc:
        print(f"schema-5 Slurm fleet canary failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
