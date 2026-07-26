#!/usr/bin/env python3
"""Render, verify, and transactionally submit the schema-5 v1.1-r1 recovery DAG.

The recovery jobs are deliberately generated outside the Git checkout.  A successful
``render --apply`` publishes immutable generation-specific sbatch files first and the
chain manifest last.  ``submit --apply`` persists one intent before every ``sbatch``
boundary and publishes a separate submission receipt only after every exact
``afterok`` dependency has been accepted.

No command in this module mutates a legacy run.  The first generated job that can do
so is ``legacy_consolidate``; its DAG ancestors and its own preflight both prove the
sealed pre-repair snapshot and immutable release before ``--apply`` is invoked.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


RELEASE_ID = "sweep-recovery-schema5-v1.1"
RELEASE_TAG = "sweep-recovery-schema5-v1.1-r1"
CHAIN_NAMESPACE = "schema5-v1.1-r1"
CHAIN_MANIFEST_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
R1_FAILURE_ENVELOPE_NAME = "FAILED_RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json"
SUBMISSION_JOURNAL_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submission.json"
SUBMISSION_RECEIPT_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_1_R1_SUBMISSION.json"
REPAIR_ROOT_NAME = "recovery_chain_repairs_v1_1_r1"
QUARANTINE_ROOT_NAME = "quarantine"
QUARANTINE_EVIDENCE_ROOT_NAME = "materialization_quarantines"
RENDER_LOCK_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.render.lock"
SUBMISSION_LOCK_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_1_R1.submit.lock"
CHAIN_SCHEMA_VERSION = 2
SUBMISSION_SCHEMA_VERSION = 1
VISIBILITY_GRACE_SECONDS = 300.0
LEGACY_RUN_IDS = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
)
PRODUCTION_RUN_IDS = (
    "full_sweep_schema5_v1",
    "full_sweep_agent_counts_schema5_v1",
    "full_sweep_agent_count_7_schema5_v1",
)
SMOKE_RUN_IDS = (
    "schema5_smoke_32b_long_v1",
    "schema5_smoke_selective_long_v1",
    "schema5_smoke_standard_canaries_v1",
)
HEAVY_SERIAL_ORDER = (
    "pre_repair_snapshot",
    "pre_repair_snapshot_verify",
    "release_materialize",
    "release_freeze",
    "legacy_consolidate",
    "legacy_consolidated_snapshot",
    "legacy_consolidated_verify",
)
FIRST_LEGACY_MUTATION = "legacy_consolidate"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# This deliberately duplicates the operational contract encoded by ``job_specs``.
# Verification must compare a rendered manifest against an independent, fixed DAG
# contract rather than merely proving that the manifest is internally consistent.
EXPECTED_JOB_CONTRACT: tuple[
    tuple[str, str, tuple[str, ...], str, str, int], ...
] = (
    ("source_checkout", "00_source_checkout.sbatch", (), "01:00:00", "4G", 1),
    ("maintenance_preflight", "01_maintenance_preflight.sbatch", ("source_checkout",), "02:00:00", "8G", 1),
    ("pre_repair_snapshot", "02_pre_repair_snapshot.sbatch", ("maintenance_preflight",), "11:30:00", "8G", 1),
    ("pre_repair_snapshot_verify", "03_pre_repair_snapshot_verify.sbatch", ("pre_repair_snapshot",), "11:30:00", "8G", 1),
    ("release_materialize", "04_release_materialize.sbatch", ("pre_repair_snapshot_verify",), "11:30:00", "12G", 2),
    ("release_freeze", "05_release_freeze.sbatch", ("release_materialize",), "11:30:00", "12G", 2),
    ("legacy_consolidate", "06_legacy_consolidate.sbatch", ("release_freeze", "pre_repair_snapshot_verify"), "11:30:00", "16G", 2),
    ("legacy_consolidated_snapshot", "07_legacy_consolidated_snapshot.sbatch", ("legacy_consolidate",), "11:30:00", "8G", 1),
    ("legacy_consolidated_verify", "08_legacy_consolidated_verify.sbatch", ("legacy_consolidated_snapshot",), "11:30:00", "8G", 1),
    ("legacy_retire", "09_legacy_retire.sbatch", ("legacy_consolidated_verify",), "06:00:00", "8G", 1),
    ("schema5_initialize", "10_schema5_initialize.sbatch", ("legacy_retire",), "11:30:00", "16G", 2),
    ("static_readiness", "11_static_readiness.sbatch", ("schema5_initialize",), "11:30:00", "8G", 1),
    ("context_readiness", "13_context_readiness.sbatch", ("static_readiness",), "11:30:00", "32G", 4),
    ("email_readiness", "14_email_readiness.sbatch", ("static_readiness",), "01:00:00", "2G", 1),
    ("supplementary_cache", "15_supplementary_cache.sbatch", ("context_readiness",), "11:30:00", "64G", 4),
    ("fleet_bootstrap", "12_fleet_bootstrap.sbatch", ("supplementary_cache",), "02:00:00", "4G", 1),
    ("fleet_readiness", "16_fleet_readiness.sbatch", ("fleet_bootstrap",), "11:00:00", "8G", 1),
    ("smoke_readiness", "17_smoke_readiness.sbatch", ("fleet_readiness",), "11:30:00", "8G", 1),
    ("controller_drill", "18_controller_drill.sbatch", ("smoke_readiness", "email_readiness"), "02:00:00", "8G", 1),
    ("production_resume", "19_production_resume.sbatch", ("controller_drill",), "11:30:00", "8G", 1),
)
EXPECTED_JOB_ORDER = tuple(row[0] for row in EXPECTED_JOB_CONTRACT)


class ChainError(RuntimeError):
    """The recovery chain cannot be rendered, verified, or safely submitted."""


@dataclass(frozen=True)
class RecoveryPaths:
    repository: Path
    results_root: Path
    recovery_root: Path
    source_checkout: Path
    release_root: Path
    worktree: Path
    identity: Path
    harness: Path
    serving: Path
    state: Path
    pool: Path
    hf_home: Path
    dev_python: Path
    source_harness: Path
    source_serving: Path
    conda_executable: Path
    jobs_root: Path
    logs_root: Path
    chain_manifest: Path
    immutable_pins: Path
    readiness: Path


@dataclass(frozen=True)
class JobSpec:
    name: str
    filename: str
    dependencies: tuple[str, ...]
    time_limit: str
    memory: str
    cpus: int
    body: str


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _job_comment(chain_id: str, name: str, generation: int) -> str:
    if generation < 0:
        raise ChainError("recovery-chain submission generation cannot be negative")
    return f"asys:s5-recovery-v1.1-r1:{chain_id}:g{generation:04d}:{name}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slurm_timestamp(timestamp: float) -> str:
    """Return the timezone-free ISO form accepted by this cluster's ``sacct -S``."""

    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without following its final symlink."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_canonical_path(
    path: Path,
    *,
    description: str,
    kind: str | None = None,
) -> Path:
    """Reject symlink-mediated or non-canonical paths used as trust anchors."""

    lexical = _lexical_absolute(path)
    if "\n" in str(lexical) or "\r" in str(lexical):
        raise ChainError(f"{description} is not a safe path: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ChainError(f"{description} is missing or unsafe: {lexical}: {exc}") from exc
    if resolved != lexical:
        raise ChainError(f"{description} traverses a symlink: {lexical}")
    if kind == "file" and not lexical.is_file():
        raise ChainError(f"{description} is not a regular file: {lexical}")
    if kind == "directory" and not lexical.is_dir():
        raise ChainError(f"{description} is not a directory: {lexical}")
    return lexical


def _manifest_path(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ChainError(f"{description} is not one safe absolute path")
    raw = Path(value)
    if not raw.is_absolute():
        raise ChainError(f"{description} is not absolute: {value!r}")
    lexical = _lexical_absolute(raw)
    if str(lexical) != value:
        raise ChainError(f"{description} is not lexically canonical: {value!r}")
    try:
        resolved = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ChainError(f"{description} is unsafe: {lexical}: {exc}") from exc
    if resolved != lexical:
        raise ChainError(f"{description} traverses a symlink: {lexical}")
    return lexical


@contextmanager
def _exclusive_lock(path: Path, *, description: str) -> Iterable[None]:
    """Acquire a non-following, cross-process publication lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o640)
    except OSError as exc:
        raise ChainError(f"cannot open {description} lock {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ChainError(f"{description} lock is not one regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ChainError(f"another {description} process holds the lock") from exc
        yield
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ChainError(f"{description} is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChainError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ChainError(f"{description} is not one JSON object: {path}")
    return value


def _reject_retired_r1_mutation(recovery_root: Path) -> None:
    """Unconditionally reject mutation by the retired r1 protocol.

    Historical verification and quarantine inspection remain available, but the
    superseding release must never render, submit, or repair r1—even on a fresh or
    incorrectly selected recovery root without the historical failure envelope.  If
    the envelope exists, validate it before reporting permanent retirement so corrupt
    forensic evidence is not silently ignored.
    """

    envelope_path = recovery_root / R1_FAILURE_ENVELOPE_NAME
    if envelope_path.exists() or envelope_path.is_symlink():
        envelope = _read_json(
            envelope_path, description="retired r1 failure envelope"
        )
        if (
            stat.S_IMODE(envelope_path.stat().st_mode) & 0o222
            or envelope.get("protocol") != "schema5-recovery-chain-failure-v1"
            or envelope.get("classification") != "requires_superseding_release"
            or envelope.get("retry_same_generation") is not False
            or envelope.get("superseded_by") != "sweep-recovery-schema5-v1.2"
            or _SHA256.fullmatch(str(envelope.get("failure_id", ""))) is None
        ):
            raise ChainError(
                "r1 mutation is blocked because its failure envelope is unsafe "
                "or invalid"
            )
    raise ChainError(
        "schema5-v1.1-r1 is retired and requires the superseding v1.2 release; "
        "render, submit, and repair are permanently disabled on every recovery root"
    )


def _absolute(path: Path, *, description: str) -> Path:
    if not path.is_absolute() or "\n" in str(path) or "\r" in str(path):
        raise ChainError(f"{description} must be a safe absolute path: {path}")
    if path.is_symlink():
        raise ChainError(f"{description} cannot be a symlink: {path}")
    return path.resolve(strict=False)


def _environment_executable(
    path: Path, *, environment_prefix: Path, description: str
) -> Path:
    """Resolve a normal Conda executable alias without trusting an external target."""

    lexical = _lexical_absolute(path)
    prefix = _absolute(environment_prefix, description=f"{description} environment")
    if not lexical.is_absolute() or "\n" in str(lexical) or "\r" in str(lexical):
        raise ChainError(f"{description} must be a safe absolute path: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(prefix.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ChainError(
            f"{description} must resolve inside {prefix}: {lexical}: {exc}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ChainError(f"{description} target is not executable: {resolved}")
    return resolved


def recovery_paths(
    *,
    repository: Path,
    results_root: Path,
    recovery_root: Path,
    hf_home: Path,
    dev_python: Path,
    source_harness: Path,
    source_serving: Path,
    conda_executable: Path,
) -> RecoveryPaths:
    repository = _absolute(repository, description="repository")
    results_root = _absolute(results_root, description="results root")
    recovery_root = _absolute(recovery_root, description="recovery root")
    hf_home = _absolute(hf_home, description="HF home")
    expected_recovery = results_root / "recovery" / "schema5-v1"
    if recovery_root != expected_recovery:
        raise ChainError(
            f"recovery root must be the canonical {expected_recovery}, got {recovery_root}"
        )
    release = recovery_root / "releases" / RELEASE_ID
    source_harness = _absolute(source_harness, description="source harness")
    source_serving = _absolute(source_serving, description="source serving prefix")
    for description, directory in (
        ("repository", repository),
        ("results root", results_root),
        ("recovery root", recovery_root),
        ("HF home", hf_home),
        ("source harness", source_harness),
        ("source serving prefix", source_serving),
    ):
        if not directory.is_dir():
            raise ChainError(f"{description} is not an existing directory: {directory}")
    dev_python = _environment_executable(
        dev_python,
        environment_prefix=source_harness,
        description="development Python",
    )
    conda_executable = _absolute(
        conda_executable, description="Conda executable"
    )
    if not conda_executable.is_file() or not os.access(conda_executable, os.X_OK):
        raise ChainError(f"Conda executable is missing or not executable: {conda_executable}")
    return RecoveryPaths(
        repository=repository,
        results_root=results_root,
        recovery_root=recovery_root,
        source_checkout=recovery_root / "release_source_checkout_v1_1_r1",
        release_root=release,
        worktree=release / "worktree",
        identity=release / "identity",
        harness=release / "environments" / "harness",
        serving=release / "environments" / "serving",
        state=results_root / ".dispatcher-schema5-v1",
        pool=results_root / "server_pools" / "schema5-v1",
        hf_home=hf_home,
        dev_python=dev_python,
        source_harness=source_harness,
        source_serving=source_serving,
        conda_executable=conda_executable,
        jobs_root=recovery_root / "jobs" / CHAIN_NAMESPACE,
        logs_root=recovery_root / "logs" / CHAIN_NAMESPACE,
        chain_manifest=recovery_root / CHAIN_MANIFEST_NAME,
        immutable_pins=recovery_root / "immutable_pins.schema5-v1.json",
        readiness=recovery_root / "readiness",
    )


def _run_checked(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            list(argv), cwd=cwd, text=True, capture_output=True, check=False
        )
    except OSError as exc:
        raise ChainError(f"cannot execute {argv[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise ChainError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}: "
            f"{proc.stderr.strip()[:1000]}"
        )
    return proc.stdout.strip()


def verify_release_tag(repository: Path) -> dict[str, str]:
    if not repository.is_dir() or not (repository / ".git").exists():
        raise ChainError(f"source repository is not a Git checkout: {repository}")
    tag_type = _run_checked(
        ["git", "cat-file", "-t", f"refs/tags/{RELEASE_TAG}"], cwd=repository
    )
    if tag_type != "tag":
        raise ChainError(f"{RELEASE_TAG} must be an annotated immutable tag")
    commit = _run_checked(
        ["git", "rev-parse", f"refs/tags/{RELEASE_TAG}^{{commit}}"], cwd=repository
    )
    head = _run_checked(["git", "rev-parse", "HEAD"], cwd=repository)
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or head != commit:
        raise ChainError(
            f"development HEAD must equal {RELEASE_TAG}: head={head}, tag={commit}"
        )
    status = _run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repository
    )
    if status:
        raise ChainError("development checkout must be clean before chain rendering")
    return {"release_tag": RELEASE_TAG, "git_commit": commit}


def _q(path_or_value: object) -> str:
    return shlex.quote(str(path_or_value))


def _common_exports(paths: RecoveryPaths) -> str:
    return f"""\
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH LD_PRELOAD
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export ASYS_RESULTS_ROOT={_q(paths.results_root)}
export ASYS_RELEASE_WORKTREE={_q(paths.worktree)}
export HF_HOME={_q(paths.hf_home)}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"""


def _source_checkout_body(paths: RecoveryPaths, commit: str) -> str:
    return f"""\
repository={_q(paths.repository)}
target={_q(paths.source_checkout)}
release_tag={_q(RELEASE_TAG)}
expected_commit={_q(commit)}
temporary="${{target}}.clone.${{SLURM_JOB_ID:-manual}}"
verify_target() {{
  [[ -d "$target" && ! -L "$target" ]]
  [[ ! -s "$target/.git/objects/info/alternates" ]]
  [[ "$(git -C "$target" cat-file -t "refs/tags/$release_tag")" == tag ]]
  [[ "$(git -C "$target" rev-parse HEAD)" == "$expected_commit" ]]
  [[ "$(git -C "$target" rev-parse "${{release_tag}}^{{commit}}")" == "$expected_commit" ]]
  [[ -z "$(git -C "$target" status --porcelain=v1 --untracked-files=all)" ]]
  git -C "$target" fsck --full --strict
}}
if [[ -e "$target" || -L "$target" ]]; then
  verify_target
  exit 0
fi
[[ ! -e "$temporary" && ! -L "$temporary" ]]
git -C "$repository" cat-file -e "${{release_tag}}^{{commit}}"
[[ "$(git -C "$repository" cat-file -t "refs/tags/$release_tag")" == tag ]]
[[ "$(git -C "$repository" rev-parse "${{release_tag}}^{{commit}}")" == "$expected_commit" ]]
git clone --no-local --no-checkout -- "$repository" "$temporary"
[[ ! -s "$temporary/.git/objects/info/alternates" ]]
git -C "$temporary" checkout --detach "$release_tag"
[[ "$(git -C "$temporary" rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git -C "$temporary" status --porcelain=v1 --untracked-files=all)" ]]
git -C "$temporary" fsck --full --strict
mv -- "$temporary" "$target"
verify_target
"""


def _maintenance_body(paths: RecoveryPaths, slurm_user: str) -> str:
    return _common_exports(paths) + f"""\
exec {_q(paths.dev_python)} -I - {_q(paths.source_checkout)} {_q(paths.results_root)} {_q(paths.recovery_root)} {_q(slurm_user)} <<'PY'
import json
import os
from pathlib import Path
import sys

checkout, results, recovery = map(Path, sys.argv[1:4])
expected_user = sys.argv[4]
if os.environ.get("USER") != expected_user:
    raise SystemExit(f"scheduler user drift: {{os.environ.get('USER')!r}} != {{expected_user!r}}")
sys.path.insert(0, str(checkout))
from scripts.consolidate_legacy_recovery import _maintenance_precheck
report = _maintenance_precheck(results_root=results, recovery_root=recovery)
print(json.dumps(report, indent=2, sort_keys=True))
PY
"""


def _snapshot_body(paths: RecoveryPaths) -> str:
    tool = paths.source_checkout / "scripts" / "create_recovery_snapshot.py"
    args = " \\\n  ".join(
        [
            f"--snapshot-root {_q(paths.recovery_root / 'pre_repair')}",
            f"--source {_q('full_sweep_v1=' + str(paths.results_root / 'full_sweep_v1'))}",
            f"--source {_q('full_sweep_agent_counts_v1=' + str(paths.results_root / 'full_sweep_agent_counts_v1'))}",
            f"--source {_q('full_sweep_agent_count_7_v1=' + str(paths.results_root / 'full_sweep_agent_count_7_v1'))}",
            f"--source {_q('dispatcher_v3=' + str(paths.results_root / '.dispatcher-v3'))}",
            f"--source {_q('recovery_evidence=' + str(paths.recovery_root / 'pre_repair_inventory'))}",
        ]
    )
    return _common_exports(paths) + f"""\
exec ionice -c 2 -n 7 nice -n 10 {_q(paths.dev_python)} -I {_q(tool)} \\
  {args}
"""


def _snapshot_verify_body(paths: RecoveryPaths) -> str:
    tool = paths.source_checkout / "scripts" / "create_recovery_snapshot.py"
    return _common_exports(paths) + f"""\
exec ionice -c 2 -n 7 nice -n 10 {_q(paths.dev_python)} -I {_q(tool)} \\
  --verify-only \\
  --snapshot-root {_q(paths.recovery_root / 'pre_repair')} \\
  --attestation-path {_q(paths.recovery_root / 'pre_repair.attestation.json')}
"""


def _materialize_body(paths: RecoveryPaths) -> str:
    tool = paths.source_checkout / "scripts" / "materialize_schema5_release.py"
    command = f"""\
{_q(paths.dev_python)} -I {_q(tool)} materialize \\
  --output-root {_q(paths.release_root)} \\
  --source-repository {_q(paths.source_checkout)} \\
  --release-worktree {_q(paths.worktree)} \\
  --source-harness-prefix {_q(paths.source_harness)} \\
  --source-serving-prefix {_q(paths.source_serving)} \\
  --harness-prefix {_q(paths.harness)} \\
  --serving-prefix {_q(paths.serving)} \\
  --conda-executable {_q(paths.conda_executable)}"""
    return _common_exports(paths) + f"""\
if [[ -e {_q(paths.release_root)} || -L {_q(paths.release_root)} ]]; then
  if [[ -f {_q(paths.release_root / 'MATERIALIZATION_COMPLETE.json')} && \
        ! -L {_q(paths.release_root / 'MATERIALIZATION_COMPLETE.json')} ]]; then
    exec ionice -c 2 -n 7 nice -n 10 {_q(paths.dev_python)} -I {_q(tool)} verify \
      --output-root {_q(paths.release_root)}
  fi
  echo "partial materialization is preserved at {_q(paths.release_root)}; run the explicit quarantine-materialization command before repair" >&2
  exit 3
fi
ionice -c 2 -n 7 nice -n 10 {command}
ionice -c 2 -n 7 nice -n 10 {command} --apply
exec ionice -c 2 -n 7 nice -n 10 {_q(paths.dev_python)} -I {_q(tool)} verify \\
  --output-root {_q(paths.release_root)}
"""


def _freeze_body(paths: RecoveryPaths) -> str:
    materializer = paths.source_checkout / "scripts" / "materialize_schema5_release.py"
    freezer = paths.worktree / "scripts" / "freeze_schema5_release.py"
    command = f"""\
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(freezer)} create \\
  --output-root {_q(paths.identity)} \\
  --release-worktree {_q(paths.worktree)} \\
  --harness-prefix {_q(paths.harness)} \\
  --serving-prefix {_q(paths.serving)} \\
  --model-contract {_q(paths.worktree / 'configs/model_contracts.v1.json')} \\
  --fleet-contract {_q(paths.worktree / 'configs/schema5_fleet.v1.json')} \\
  --conda-executable {_q(paths.conda_executable)}"""
    return _common_exports(paths) + f"""\
ionice -c 2 -n 7 nice -n 10 {_q(paths.dev_python)} -I {_q(materializer)} verify \\
  --output-root {_q(paths.release_root)}
ionice -c 2 -n 7 nice -n 10 {command}
ionice -c 2 -n 7 nice -n 10 {command} \\
  --apply --seal-worktree --seal-environments --seal-output-root
exec ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I {_q(freezer)} verify \\
  --output-root {_q(paths.identity)}
"""


def _consolidate_body(paths: RecoveryPaths) -> str:
    freezer = paths.worktree / "scripts" / "freeze_schema5_release.py"
    tool = paths.worktree / "scripts" / "consolidate_legacy_recovery.py"
    command = f"""\
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(tool)} \\
  --results-root {_q(paths.results_root)} \\
  --recovery-root {_q(paths.recovery_root)} \\
  --pre-repair-attestation {_q(paths.recovery_root / 'pre_repair.attestation.json')}"""
    return _common_exports(paths) + f"""\
# Re-prove the immutable release immediately before the first legacy mutation.
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I \\
  {_q(freezer)} verify --output-root {_q(paths.identity)}
# The complete dry-run re-verifies the snapshot, maintenance state, exact would-change
# set, and before/after hashes.  Only its success permits the apply invocation.
{command}
exec {command} --apply
"""


def _consolidated_snapshot_body(paths: RecoveryPaths) -> str:
    tool = paths.worktree / "scripts" / "create_recovery_snapshot.py"
    args = " \\\n  ".join(
        [
            f"--snapshot-root {_q(paths.recovery_root / 'legacy_consolidated')}",
            f"--source {_q('full_sweep_v1=' + str(paths.results_root / 'full_sweep_v1'))}",
            f"--source {_q('full_sweep_agent_counts_v1=' + str(paths.results_root / 'full_sweep_agent_counts_v1'))}",
            f"--source {_q('full_sweep_agent_count_7_v1=' + str(paths.results_root / 'full_sweep_agent_count_7_v1'))}",
            f"--source {_q('dispatcher_v3=' + str(paths.results_root / '.dispatcher-v3'))}",
            f"--source {_q('legacy_cleanup_evidence=' + str(paths.recovery_root / 'operations/legacy_consolidation'))}",
            f"--source {_q('legacy_cleanup_complete=' + str(paths.recovery_root / 'LEGACY_CLEANUP_COMPLETE.json'))}",
        ]
    )
    return _common_exports(paths) + f"""\
exec ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I {_q(tool)} \\
  {args}
"""


def _consolidated_verify_body(paths: RecoveryPaths) -> str:
    tool = paths.worktree / "scripts" / "create_recovery_snapshot.py"
    return _common_exports(paths) + f"""\
exec ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I {_q(tool)} \\
  --verify-only \\
  --snapshot-root {_q(paths.recovery_root / 'legacy_consolidated')} \\
  --attestation-path {_q(paths.recovery_root / 'legacy_consolidated.attestation.json')}
"""


def _retire_body(paths: RecoveryPaths) -> str:
    tool = paths.worktree / "scripts" / "retire_legacy_runs.py"
    command = f"""\
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(tool)} \\
  --results-root {_q(paths.results_root)} --recovery-root {_q(paths.recovery_root)}"""
    return _common_exports(paths) + f"""\
{command}
exec {command} --apply
"""


def _initialize_body(paths: RecoveryPaths) -> str:
    fragment = paths.identity / "release_identity.schema5-v1.json"
    return _common_exports(paths) + f"""\
fragment={_q(fragment)}
release_id="$(jq -er '.control_pin_fragment.release_id' "$fragment")"
git_commit="$(jq -er '.control_pin_fragment.git_commit' "$fragment")"
source_tree="$(jq -er '.control_pin_fragment.source_tree_sha256' "$fragment")"
model_contract="$(jq -er '.control_pin_fragment.model_contract_path' "$fragment")"
harness_hash="$(jq -er '.control_pin_fragment.harness_environment_sha256' "$fragment")"
serving_hash="$(jq -er '.control_pin_fragment.serving_environment_sha256' "$fragment")"
[[ "$release_id" == {_q(RELEASE_ID)} ]]

clone=(
  {_q(paths.harness / 'bin/python')} -I {_q(paths.worktree / 'scripts/clone_schema5_manifests.py')}
  --results-root {_q(paths.results_root)} --model-contract "$model_contract"
  --release-id "$release_id" --git-commit "$git_commit"
  --source-tree-sha256 "$source_tree" --harness-env-sha256 "$harness_hash"
  --serving-env-sha256 "$serving_hash"
)
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "${{clone[@]}}"
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "${{clone[@]}}" --apply

smoke=(
  {_q(paths.harness / 'bin/python')} -I {_q(paths.worktree / 'scripts/init_schema5_smokes.py')}
  --results-root {_q(paths.results_root)} --release-worktree {_q(paths.worktree)}
  --model-contract "$model_contract" --release-id "$release_id"
  --git-commit "$git_commit" --source-tree-sha256 "$source_tree"
  --harness-environment-sha256 "$harness_hash"
  --serving-environment-sha256 "$serving_hash"
)
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "${{smoke[@]}}"
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "${{smoke[@]}}" --apply

env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I \\
  {_q(paths.worktree / 'slurm/schema5_control.py')} --state-dir {_q(paths.state)} prepare-pins \\
  --release-bundle-root {_q(paths.identity)} --hf-home {_q(paths.hf_home)} \\
  --output {_q(paths.immutable_pins)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I \\
  {_q(paths.worktree / 'slurm/schema5_control.py')} --state-dir {_q(paths.state)} init \\
  --pins-json {_q(paths.immutable_pins)}
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I \\
  {_q(paths.worktree / 'slurm/schema5_control.py')} --state-dir {_q(paths.state)} \\
  reconcile --all --no-admit
"""


def _static_readiness_body(paths: RecoveryPaths) -> str:
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    return _common_exports(paths) + f"""\
mkdir -p -- {_q(paths.readiness)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'snapshot.json')} snapshot \\
  --pre-repair-attestation {_q(paths.recovery_root / 'pre_repair.attestation.json')} \\
  --legacy-consolidated-attestation {_q(paths.recovery_root / 'legacy_consolidated.attestation.json')}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate snapshot --evidence {_q(paths.readiness / 'snapshot.json')}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'migrations.json')} migrations \\
  --cleanup-marker {_q(paths.recovery_root / 'LEGACY_CLEANUP_COMPLETE.json')}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate migrations --evidence {_q(paths.readiness / 'migrations.json')}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'semantic_audit.json')} semantic-audit \\
  --cleanup-marker {_q(paths.recovery_root / 'LEGACY_CLEANUP_COMPLETE.json')}
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate semantic_audit \\
  --evidence {_q(paths.readiness / 'semantic_audit.json')}
"""


def _fleet_once_python(paths: RecoveryPaths) -> str:
    return f"""\
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I - \
  {_q(paths.state)} {_q(paths.worktree)} <<'PY'
import copy
import os
from pathlib import Path
import subprocess
import sys

state_dir = Path(sys.argv[1]).resolve()
worktree = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(worktree))
from slurm import schema5_control

# Fleet bootstrap is preproduction: control remains paused at generation g while the
# servers are launched for the exact next generation g+1.  Hold the production control
# lock across attestation, lease refresh, and supervisor submission so a concurrent
# resume cannot change that generation between check and use.
with schema5_control.control_lock(state_dir):
    control = schema5_control.load_control(state_dir, verify_files=True)
    if control.get("desired_state") != "paused" or control.get("drain_requested") is not False:
        raise SystemExit("preproduction fleet launch requires paused, non-draining control")
    current_generation = control.get("rollout_generation")
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 0
    ):
        raise SystemExit("control has an invalid rollout generation")
    target_generation = current_generation + 1
    attestation = schema5_control.ensure_runtime_integrity_attestation(
        state_dir,
        control,
        generation=target_generation,
        force_full=False,
    )
    execution_control = copy.deepcopy(control)
    execution_control["rollout_generation"] = target_generation
    execution_control[schema5_control.RUNTIME_ATTESTATION_STATE_KEY] = attestation
    verified = schema5_control.validate_runtime_integrity_attestation(
        execution_control,
        verify_metadata=True,
    )
    if verified != attestation:
        raise SystemExit("next-generation runtime attestation projection drifted")
    production_environment = schema5_control.production_environment(execution_control)
    required = {{
        "ASYS_RUNTIME_ATTESTATION",
        "ASYS_RUNTIME_ATTESTATION_SHA256",
        "ASYS_RUNTIME_INTEGRITY_LEASE",
        "ASYS_IMMUTABLE_PINS_SHA256",
        "ASYS_ROLLOUT_GENERATION",
    }}
    if not required <= set(production_environment):
        raise SystemExit("next-generation production environment is incomplete")
    if production_environment["ASYS_ROLLOUT_GENERATION"] != str(target_generation):
        raise SystemExit("next-generation production environment has the wrong generation")
    command = control["immutable"].get("fleet_supervisor_command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise SystemExit("immutable control lacks a valid fleet_supervisor_command")
    environment = dict(os.environ)
    environment.update(production_environment)
    subprocess.run([*command, "--once"], check=True, env=environment)
PY
"""


def _bootstrap_fleet_body(paths: RecoveryPaths) -> str:
    return _common_exports(paths) + "exec " + _fleet_once_python(paths)


def _context_body(paths: RecoveryPaths) -> str:
    audit = paths.worktree / "scripts" / "audit_context_capacity.py"
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    run_id = "full_sweep_agent_count_7_schema5_v1"
    dense = paths.readiness / "dense_peer_context.json"
    seven = paths.readiness / "seven_agent_context.json"
    return _common_exports(paths) + f"""\
mkdir -p -- {_q(paths.readiness)}
dense_tmp={_q(dense)}".${{SLURM_JOB_ID}}.tmp"
seven_tmp={_q(seven)}".${{SLURM_JOB_ID}}.tmp"
trap 'test ! -f "$dense_tmp" || mv -- "$dense_tmp" "$dense_tmp.failed"; test ! -f "$seven_tmp" || mv -- "$seven_tmp" "$seven_tmp.failed"' EXIT
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(audit)} \\
  --run-id {run_id} --run-root {_q(paths.results_root / run_id)} --n-agents 7 \\
  --reasoning b2048 b8192 unlimited --prompt-level 3 --topology decentralized \\
  --context-share-level plus_cot --all-routed-profiles >"$dense_tmp"
mv -- "$dense_tmp" {_q(dense)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(audit)} \\
  --run-id {run_id} --run-root {_q(paths.results_root / run_id)} --n-agents 7 \\
  --reasoning unlimited --context-share-level plus_cot --all-routed-profiles >"$seven_tmp"
mv -- "$seven_tmp" {_q(seven)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'context_audit.json')} context-audit \\
  --dense-peer-audit {_q(dense)} --seven-agent-audit {_q(seven)}
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate context_audit \\
  --evidence {_q(paths.readiness / 'context_audit.json')}
"""


def _email_body(paths: RecoveryPaths) -> str:
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    return _common_exports(paths) + f"""\
mkdir -p -- {_q(paths.readiness)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'email_test.json')} email-test --apply
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate email_test \\
  --evidence {_q(paths.readiness / 'email_test.json')}
"""


def _supplementary_body(paths: RecoveryPaths) -> str:
    output = paths.recovery_root / "analysis_cache" / "supplementary_legacy"
    return _common_exports(paths) + f"""\
export SCHEMA5_CONTROL_STATE_DIR={_q(paths.state)}
export ANALYSIS_MODE=supplementary-legacy
export PRE_REPAIR_SNAPSHOT_ROOT={_q(paths.recovery_root / 'pre_repair')}
export OUT_DIR={_q(output)}
exec bash {_q(paths.worktree / 'analysis/refresh.sh')}
"""


def _fleet_wait_body(paths: RecoveryPaths) -> str:
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    return _common_exports(paths) + f"""\
mkdir -p -- {_q(paths.readiness)}
deadline=$(( $(date +%s) + 36000 ))
attempt=0
while (( $(date +%s) < deadline )); do
  attempt=$(( attempt + 1 ))
  echo "[fleet-readiness] attempt=$attempt timestamp=$(date --iso-8601=seconds)"
  {_fleet_once_python(paths)}
  if env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
      --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'fleet.json')} fleet --probe-timeout 10; then
    exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
      --state-dir {_q(paths.state)} attest --gate fleet --evidence {_q(paths.readiness / 'fleet.json')}
  fi
  remaining=$(( deadline - $(date +%s) ))
  (( remaining > 0 )) || break
  (( remaining < 300 )) && sleep "$remaining" || sleep 300
done
echo "fleet did not satisfy readiness within the bounded 10-hour window" >&2
exit 2
"""


def _smoke_body(paths: RecoveryPaths) -> str:
    runner = paths.worktree / "scripts" / "run_schema5_smokes.py"
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    return _common_exports(paths) + f"""\
# The dependency gate may have completed long before Slurm starts this allocation.
# Re-adopt and probe the exact g+1 fleet inside the smoke allocation so queue latency
# cannot stale the runtime lease before the first estimand-excluded draw.
fleet_once() {{
{_fleet_once_python(paths)}}}
presmoke_deadline=$(( $(date +%s) + 900 ))
presmoke_fleet_ready=0
while (( $(date +%s) < presmoke_deadline )); do
  if fleet_once && \
     env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \
       --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'fleet.json')} fleet --probe-timeout 10 && \
     env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \
       --state-dir {_q(paths.state)} attest --gate fleet \
       --evidence {_q(paths.readiness / 'fleet.json')}; then
    presmoke_fleet_ready=1
    break
  fi
  sleep 30
done
(( presmoke_fleet_ready == 1 )) || {{ echo "pre-smoke fleet readiness could not be refreshed within 15 minutes" >&2; exit 2; }}

readarray -t smoke_pins < <(env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I - {_q(paths.state / 'control.json')} <<'PY'
import json
from pathlib import Path
import sys

control = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if control.get("desired_state") != "paused" or control.get("drain_requested") is not False:
    raise SystemExit("smoke launch requires paused, non-draining control")
generation = control.get("rollout_generation")
if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
    raise SystemExit("control has invalid rollout_generation")
immutable = control.get("immutable_sha256")
if not isinstance(immutable, str) or len(immutable) != 64:
    raise SystemExit("control has invalid immutable_sha256")
print(immutable)
print(generation + 1)
PY
)
[[ "${{#smoke_pins[@]}}" -eq 2 ]]
immutable_sha="${{smoke_pins[0]}}"
next_generation="${{smoke_pins[1]}}"
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(runner)} \\
  --results-root {_q(paths.results_root)} --server-pool-root {_q(paths.pool)} \\
  --release-worktree {_q(paths.worktree)} --harness-prefix {_q(paths.harness)} \\
  --state-root {_q(paths.state)} --immutable-pins-sha256 "$immutable_sha" \\
  --rollout-generation "$next_generation" --apply
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate smoke_runs \\
  --evidence {_q(paths.state / 'readiness/smoke_runs.json')}
"""


def _drill_body(paths: RecoveryPaths) -> str:
    control = paths.worktree / "slurm" / "schema5_control.py"
    prefix = (
        f"env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "
        f"{_q(paths.harness / 'bin/python')} -I {_q(control)} "
        f"--state-dir {_q(paths.state)}"
    )
    return _common_exports(paths) + f"""\
control=( {prefix} )
if [[ -f {_q(paths.state / 'CONTROLLER_KILL_DRILL_COMPLETE.json')} && \
      ! -L {_q(paths.state / 'CONTROLLER_KILL_DRILL_COMPLETE.json')} ]]; then
  exec "${{control[@]}}" drill status --live
fi
"${{control[@]}}" reconcile --all --no-admit
"${{control[@]}}" drill start
deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" drill status --live; do
  (( $(date +%s) < deadline )) || {{ echo "initial drill readiness exceeded 15 minutes" >&2; exit 2; }}
  sleep 5
done
"${{control[@]}}" drill kill --role dispatcher
"${{control[@]}}" drill wait --role dispatcher --timeout 900
deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" drill status --live; do
  (( $(date +%s) < deadline )) || {{ echo "dispatcher recovery did not become globally ready" >&2; exit 2; }}
  sleep 5
done
"${{control[@]}}" drill kill --role fleet_supervisor
"${{control[@]}}" drill wait --role fleet_supervisor --timeout 900
"${{control[@]}}" drill finish --timeout 900
exec "${{control[@]}}" drill status --live
"""


def _resume_body(paths: RecoveryPaths) -> str:
    control = paths.worktree / "slurm" / "schema5_control.py"
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    prefix = (
        f"env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "
        f"{_q(paths.harness / 'bin/python')} -I {_q(control)} "
        f"--state-dir {_q(paths.state)}"
    )
    return _common_exports(paths) + f"""\
control=( {prefix} )
"${{control[@]}}" drill status --live
"${{control[@]}}" reconcile --all --no-admit

# Fleet readiness is deliberately short-lived (<=600 seconds).  The serial smokes and
# controller drill make the earlier gate stale, so rebuild it after the drill and as
# close as possible to resume.  Readiness and transition history are excluded from the
# drill's production-state baseline; this refresh cannot mask a drill mutation.
fleet_once() {{
{_fleet_once_python(paths)}}}
fleet_deadline=$(( $(date +%s) + 900 ))
fleet_refreshed=0
while (( $(date +%s) < fleet_deadline )); do
  if fleet_once && \
     env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
       --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'fleet.json')} fleet --probe-timeout 10 && \
     "${{control[@]}}" attest --gate fleet --evidence {_q(paths.readiness / 'fleet.json')}; then
    fleet_refreshed=1
    break
  fi
  sleep 30
done
(( fleet_refreshed == 1 )) || {{ echo "fleet readiness could not be refreshed within 15 minutes" >&2; exit 2; }}

# Resume persists scheduler intents before sbatch and can intentionally return a
# visibility-pending error.  Retry the same idempotent transaction; never construct a
# second state directory or bypass its scheduler reconciliation.
resume_deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" resume; do
  (( $(date +%s) < resume_deadline )) || {{ echo "resume transaction did not commit within 15 minutes" >&2; exit 2; }}
  sleep 5
done
deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" status --live; do
  (( $(date +%s) < deadline )) || {{ echo "production controllers failed live health within 15 minutes" >&2; exit 2; }}
  sleep 5
done
"${{control[@]}}" status --live
"""


def job_specs(paths: RecoveryPaths, *, commit: str, slurm_user: str) -> tuple[JobSpec, ...]:
    if not _SAFE_NAME.fullmatch(slurm_user):
        raise ChainError(f"unsafe Slurm user name: {slurm_user!r}")
    return (
        JobSpec("source_checkout", "00_source_checkout.sbatch", (), "01:00:00", "4G", 1, _source_checkout_body(paths, commit)),
        JobSpec("maintenance_preflight", "01_maintenance_preflight.sbatch", ("source_checkout",), "02:00:00", "8G", 1, _maintenance_body(paths, slurm_user)),
        JobSpec("pre_repair_snapshot", "02_pre_repair_snapshot.sbatch", ("maintenance_preflight",), "11:30:00", "8G", 1, _snapshot_body(paths)),
        JobSpec("pre_repair_snapshot_verify", "03_pre_repair_snapshot_verify.sbatch", ("pre_repair_snapshot",), "11:30:00", "8G", 1, _snapshot_verify_body(paths)),
        JobSpec("release_materialize", "04_release_materialize.sbatch", ("pre_repair_snapshot_verify",), "11:30:00", "12G", 2, _materialize_body(paths)),
        JobSpec("release_freeze", "05_release_freeze.sbatch", ("release_materialize",), "11:30:00", "12G", 2, _freeze_body(paths)),
        JobSpec("legacy_consolidate", "06_legacy_consolidate.sbatch", ("release_freeze", "pre_repair_snapshot_verify"), "11:30:00", "16G", 2, _consolidate_body(paths)),
        JobSpec("legacy_consolidated_snapshot", "07_legacy_consolidated_snapshot.sbatch", ("legacy_consolidate",), "11:30:00", "8G", 1, _consolidated_snapshot_body(paths)),
        JobSpec("legacy_consolidated_verify", "08_legacy_consolidated_verify.sbatch", ("legacy_consolidated_snapshot",), "11:30:00", "8G", 1, _consolidated_verify_body(paths)),
        JobSpec("legacy_retire", "09_legacy_retire.sbatch", ("legacy_consolidated_verify",), "06:00:00", "8G", 1, _retire_body(paths)),
        JobSpec("schema5_initialize", "10_schema5_initialize.sbatch", ("legacy_retire",), "11:30:00", "16G", 2, _initialize_body(paths)),
        JobSpec("static_readiness", "11_static_readiness.sbatch", ("schema5_initialize",), "11:30:00", "8G", 1, _static_readiness_body(paths)),
        JobSpec("context_readiness", "13_context_readiness.sbatch", ("static_readiness",), "11:30:00", "32G", 4, _context_body(paths)),
        JobSpec("email_readiness", "14_email_readiness.sbatch", ("static_readiness",), "01:00:00", "2G", 1, _email_body(paths)),
        JobSpec("supplementary_cache", "15_supplementary_cache.sbatch", ("context_readiness",), "11:30:00", "64G", 4, _supplementary_body(paths)),
        JobSpec("fleet_bootstrap", "12_fleet_bootstrap.sbatch", ("supplementary_cache",), "02:00:00", "4G", 1, _bootstrap_fleet_body(paths)),
        JobSpec("fleet_readiness", "16_fleet_readiness.sbatch", ("fleet_bootstrap",), "11:00:00", "8G", 1, _fleet_wait_body(paths)),
        JobSpec("smoke_readiness", "17_smoke_readiness.sbatch", ("fleet_readiness",), "11:30:00", "8G", 1, _smoke_body(paths)),
        JobSpec("controller_drill", "18_controller_drill.sbatch", ("smoke_readiness", "email_readiness"), "02:00:00", "8G", 1, _drill_body(paths)),
        JobSpec("production_resume", "19_production_resume.sbatch", ("controller_drill",), "11:30:00", "8G", 1, _resume_body(paths)),
    )


def _job_name(name: str) -> str:
    shortened = {
        "pre_repair_snapshot_verify": "snapshot-verify",
        "legacy_consolidated_snapshot": "legacy-snapshot",
        "legacy_consolidated_verify": "legacy-verify",
        "maintenance_preflight": "maintenance",
        "release_materialize": "materialize",
        "release_freeze": "freeze",
        "legacy_consolidate": "consolidate",
        "schema5_initialize": "initialize",
        "static_readiness": "static",
        "fleet_bootstrap": "fleet-bootstrap",
        "context_readiness": "context",
        "email_readiness": "email",
        "supplementary_cache": "legacy-cache",
        "fleet_readiness": "fleet-ready",
        "smoke_readiness": "smokes",
        "controller_drill": "drill",
        "production_resume": "resume",
        "source_checkout": "checkout",
        "pre_repair_snapshot": "snapshot",
        "legacy_retire": "retire",
    }[name]
    return f"asys-s5v11r1-{shortened}"


def render_sbatch(spec: JobSpec, paths: RecoveryPaths, *, partition: str) -> bytes:
    if not _SAFE_NAME.fullmatch(partition):
        raise ChainError(f"unsafe Slurm partition: {partition!r}")
    log = paths.logs_root / f"{spec.name}_%j.out"
    text = f"""#!/bin/bash
#SBATCH --job-name={_job_name(spec.name)}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={spec.cpus}
#SBATCH --mem={spec.memory}
#SBATCH --time={spec.time_limit}
#SBATCH --no-requeue
#SBATCH --output={log}

set -euo pipefail
umask 027
{spec.body.rstrip()}
"""
    return text.encode("utf-8")


def _topological_specs(specs: Sequence[JobSpec]) -> tuple[JobSpec, ...]:
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ChainError("duplicate recovery job name")
    seen: set[str] = set()
    for spec in specs:
        unknown = set(spec.dependencies) - set(by_name)
        if unknown:
            raise ChainError(f"{spec.name} has unknown dependencies: {sorted(unknown)}")
        if any(dependency not in seen for dependency in spec.dependencies):
            raise ChainError(f"recovery jobs are not topologically ordered at {spec.name}")
        seen.add(spec.name)
    return tuple(specs)


def _ancestors(name: str, specs: Mapping[str, JobSpec]) -> set[str]:
    result: set[str] = set()
    pending = list(specs[name].dependencies)
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(specs[dependency].dependencies)
    return result


def _manifest_ancestors(
    name: str, jobs: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    """Return transitive ancestors from the already-verified manifest DAG."""

    if name not in jobs:
        raise ChainError(f"unknown recovery-chain job: {name}")
    result: set[str] = set()
    pending = list(jobs[name]["dependencies"])
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        if dependency not in jobs:
            raise ChainError(
                f"recovery-chain job {name} has unknown ancestor {dependency}"
            )
        result.add(dependency)
        pending.extend(jobs[dependency]["dependencies"])
    return result


def _validate_dag(specs: Sequence[JobSpec]) -> None:
    ordered = _topological_specs(specs)
    observed_contract = tuple(
        (
            spec.name,
            spec.filename,
            spec.dependencies,
            spec.time_limit,
            spec.memory,
            spec.cpus,
        )
        for spec in ordered
    )
    if observed_contract != EXPECTED_JOB_CONTRACT:
        raise ChainError("recovery DAG differs from the fixed v1.1-r1 job contract")
    by_name = {spec.name: spec for spec in ordered}
    for earlier, later in zip(HEAVY_SERIAL_ORDER, HEAVY_SERIAL_ORDER[1:]):
        if earlier not in _ancestors(later, by_name):
            raise ChainError(f"heavy I/O is not serialized: {earlier} -> {later}")
    mutation_ancestors = _ancestors(FIRST_LEGACY_MUTATION, by_name)
    required = {"pre_repair_snapshot_verify", "release_freeze"}
    if not required <= mutation_ancestors:
        raise ChainError(
            "legacy mutation is not fenced by snapshot and release proofs: "
            f"missing={sorted(required - mutation_ancestors)}"
        )
    final_ancestors = _ancestors("production_resume", by_name)
    required_final = {
        "static_readiness",
        "context_readiness",
        "email_readiness",
        "fleet_readiness",
        "smoke_readiness",
        "controller_drill",
        "supplementary_cache",
    }
    if not required_final <= final_ancestors:
        raise ChainError(
            f"production resume lacks gates: {sorted(required_final - final_ancestors)}"
        )


def _preflight_fresh_destinations(paths: RecoveryPaths) -> None:
    for description, path in (
        ("v1.1-r1 source checkout", paths.source_checkout),
        ("v1.1 production release root", paths.release_root),
        ("schema-5 control state", paths.state),
    ):
        if path.exists() or path.is_symlink():
            raise ChainError(f"{description} must be fresh and absent: {path}")
    for run_id in (*PRODUCTION_RUN_IDS, *SMOKE_RUN_IDS):
        path = paths.results_root / run_id
        if path.exists() or path.is_symlink():
            raise ChainError(f"new run root must be fresh and absent: {path}")
    for run_id in LEGACY_RUN_IDS:
        path = paths.results_root / run_id
        if path.is_symlink() or not path.is_dir():
            raise ChainError(f"legacy source root is missing or unsafe: {path}")
    inventory = paths.recovery_root / "pre_repair_inventory"
    if inventory.is_symlink() or not inventory.is_dir():
        raise ChainError(f"pre-repair evidence source is missing or unsafe: {inventory}")


def _verify_jobs_namespace(
    jobs_root: Path,
    specs: Sequence[JobSpec],
    scripts: Mapping[str, bytes],
) -> None:
    if jobs_root.is_symlink() or not jobs_root.is_dir():
        raise ChainError(f"rendered jobs namespace is missing or unsafe: {jobs_root}")
    if jobs_root.resolve(strict=True) != jobs_root:
        raise ChainError(f"rendered jobs namespace traverses a symlink: {jobs_root}")
    if stat.S_IMODE(jobs_root.stat().st_mode) & 0o222:
        raise ChainError(f"rendered jobs namespace must be read-only: {jobs_root}")
    expected_names = {spec.filename for spec in specs}
    observed_names = {item.name for item in jobs_root.iterdir()}
    if observed_names != expected_names:
        raise ChainError(
            "rendered jobs namespace contents drifted: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    for spec in specs:
        path = jobs_root / spec.filename
        if path.is_symlink() or not path.is_file():
            raise ChainError(f"rendered job is missing or a symlink: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ChainError(f"rendered job must be read-only: {path}")
        if _sha256(path) != _sha256_bytes(scripts[spec.name]):
            raise ChainError(f"rendered job content drifted: {path}")


def _verify_logs_namespace(logs_root: Path, *, require_empty: bool) -> None:
    if logs_root.is_symlink() or not logs_root.is_dir():
        raise ChainError(f"recovery log namespace is missing or unsafe: {logs_root}")
    if logs_root.resolve(strict=True) != logs_root:
        raise ChainError(f"recovery log namespace traverses a symlink: {logs_root}")
    if require_empty and any(logs_root.iterdir()):
        raise ChainError(
            f"unsealed recovery log namespace is unexpectedly non-empty: {logs_root}"
        )


def _manifest_payload(
    paths: RecoveryPaths,
    specs: Sequence[JobSpec],
    scripts: Mapping[str, bytes],
    *,
    partition: str,
    git_identity: Mapping[str, str],
    slurm_user: str,
) -> dict[str, Any]:
    jobs = []
    for spec in specs:
        script = paths.jobs_root / spec.filename
        jobs.append(
            {
                "name": spec.name,
                "job_name": _job_name(spec.name),
                "script": str(script),
                "script_sha256": _sha256_bytes(scripts[spec.name]),
                "dependencies": list(spec.dependencies),
                "dependency_type": "afterok",
                "no_requeue": True,
                "time_limit": spec.time_limit,
                "memory": spec.memory,
                "cpus": spec.cpus,
            }
        )
    identity = {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "protocol": "schema5-v1.1-r1-recovery-chain",
        "namespace": CHAIN_NAMESPACE,
        "release_id": RELEASE_ID,
        "release_tag": git_identity["release_tag"],
        "release_git_commit": git_identity["git_commit"],
        "repository": str(paths.repository),
        "results_root": str(paths.results_root),
        "recovery_root": str(paths.recovery_root),
        "source_checkout": str(paths.source_checkout),
        "release_root": str(paths.release_root),
        "state_root": str(paths.state),
        "server_pool_root": str(paths.pool),
        "hf_home": str(paths.hf_home),
        "dev_python": str(paths.dev_python),
        "source_harness_prefix": str(paths.source_harness),
        "source_serving_prefix": str(paths.source_serving),
        "conda_executable": str(paths.conda_executable),
        "jobs_root": str(paths.jobs_root),
        "logs_root": str(paths.logs_root),
        "immutable_pins": str(paths.immutable_pins),
        "readiness_root": str(paths.readiness),
        "partition": partition,
        "slurm_user": slurm_user,
        "heavy_io_serial_order": list(HEAVY_SERIAL_ORDER),
        "first_legacy_mutation": FIRST_LEGACY_MUTATION,
        "jobs": jobs,
    }
    identity["chain_id"] = _sha256_bytes(_canonical_json(identity))
    return identity


def render_chain(
    paths: RecoveryPaths,
    *,
    partition: str = "mit_normal",
    slurm_user: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    _reject_retired_r1_mutation(paths.recovery_root)
    if partition != "mit_normal":
        raise ChainError(
            "durable recovery wrappers require the non-preempting mit_normal partition"
        )
    slurm_user = getpass.getuser() if slurm_user is None else slurm_user
    git_identity = verify_release_tag(paths.repository)
    specs = job_specs(paths, commit=git_identity["git_commit"], slurm_user=slurm_user)
    _validate_dag(specs)
    scripts = {
        spec.name: render_sbatch(spec, paths, partition=partition) for spec in specs
    }
    manifest = _manifest_payload(
        paths,
        specs,
        scripts,
        partition=partition,
        git_identity=git_identity,
        slurm_user=slurm_user,
    )
    report = {
        "status": "rendered" if apply else "dry_run",
        "chain_manifest": str(paths.chain_manifest),
        "chain_id": manifest["chain_id"],
        "job_count": len(specs),
        "jobs": [spec.name for spec in specs],
        "heavy_io_serial_order": list(HEAVY_SERIAL_ORDER),
        "first_legacy_mutation_ancestors": sorted(
            _ancestors(FIRST_LEGACY_MUTATION, {spec.name: spec for spec in specs})
        ),
    }
    if not apply:
        _preflight_fresh_destinations(paths)
        for description, path in (
            ("v1.1-r1 job namespace", paths.jobs_root),
            ("v1.1-r1 log namespace", paths.logs_root),
            ("v1.1-r1 chain manifest", paths.chain_manifest),
        ):
            if path.exists() or path.is_symlink():
                raise ChainError(f"{description} must be fresh and absent: {path}")
        return report

    render_lock = paths.recovery_root / RENDER_LOCK_NAME
    with _exclusive_lock(render_lock, description="recovery-chain renderer"):
        if paths.chain_manifest.exists() or paths.chain_manifest.is_symlink():
            verified = verify_chain(paths.chain_manifest)
            existing_manifest = _read_json(
                paths.chain_manifest, description="recovery-chain manifest"
            )
            if existing_manifest != manifest:
                raise ChainError(
                    "existing recovery-chain manifest does not match this render"
                )
            return report | {"status": "already_rendered", "verified": verified}

        _preflight_fresh_destinations(paths)
        paths.jobs_root.parent.mkdir(parents=True, exist_ok=True)
        paths.logs_root.parent.mkdir(parents=True, exist_ok=True)
        _require_canonical_path(
            paths.jobs_root.parent,
            description="recovery jobs parent",
            kind="directory",
        )
        _require_canonical_path(
            paths.logs_root.parent,
            description="recovery logs parent",
            kind="directory",
        )

        # Marker-last publication is restartable.  A crash may leave either namespace
        # behind; only byte-exact scripts and an empty, safe log directory are adopted.
        jobs_present = paths.jobs_root.exists() or paths.jobs_root.is_symlink()
        logs_present = paths.logs_root.exists() or paths.logs_root.is_symlink()
        if jobs_present:
            _verify_jobs_namespace(paths.jobs_root, specs, scripts)
        if logs_present:
            _verify_logs_namespace(paths.logs_root, require_empty=True)

        temporary_jobs: Path | None = None
        temporary_logs: Path | None = None
        try:
            if not jobs_present:
                temporary_jobs = paths.jobs_root.parent / (
                    f".{CHAIN_NAMESPACE}.render.{os.getpid()}.{uuid.uuid4().hex}"
                )
                temporary_jobs.mkdir(mode=0o750)
                for spec in specs:
                    target = temporary_jobs / spec.filename
                    descriptor = os.open(
                        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(scripts[spec.name])
                        handle.flush()
                        os.fsync(handle.fileno())
                os.chmod(temporary_jobs, 0o550)
                _fsync_directory(temporary_jobs)
                os.rename(temporary_jobs, paths.jobs_root)
                temporary_jobs = None
                _fsync_directory(paths.jobs_root.parent)

            if not logs_present:
                temporary_logs = paths.logs_root.parent / (
                    f".{CHAIN_NAMESPACE}.render.{os.getpid()}.{uuid.uuid4().hex}"
                )
                temporary_logs.mkdir(mode=0o750)
                _fsync_directory(temporary_logs)
                os.rename(temporary_logs, paths.logs_root)
                temporary_logs = None
                _fsync_directory(paths.logs_root.parent)

            _verify_jobs_namespace(paths.jobs_root, specs, scripts)
            _verify_logs_namespace(paths.logs_root, require_empty=True)
            # This immutable manifest is the only successful-render signal.
            _atomic_json(paths.chain_manifest, manifest, mode=0o444)
        finally:
            for temporary in (temporary_jobs, temporary_logs):
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)
        verified = verify_chain(paths.chain_manifest)
        return report | {"status": "complete", "verified": verified}


def verify_chain(manifest_path: Path) -> dict[str, Any]:
    manifest_path = _lexical_absolute(manifest_path)
    if manifest_path.is_symlink():
        raise ChainError(f"recovery-chain manifest cannot be a symlink: {manifest_path}")
    manifest_path = _require_canonical_path(
        manifest_path,
        description="recovery-chain manifest",
        kind="file",
    )
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    expected_fields = {
        "schema_version",
        "protocol",
        "namespace",
        "release_id",
        "release_tag",
        "release_git_commit",
        "repository",
        "results_root",
        "recovery_root",
        "source_checkout",
        "release_root",
        "state_root",
        "server_pool_root",
        "hf_home",
        "dev_python",
        "source_harness_prefix",
        "source_serving_prefix",
        "conda_executable",
        "jobs_root",
        "logs_root",
        "immutable_pins",
        "readiness_root",
        "partition",
        "slurm_user",
        "heavy_io_serial_order",
        "first_legacy_mutation",
        "jobs",
        "chain_id",
    }
    if set(manifest) != expected_fields:
        raise ChainError(
            f"chain manifest fields drifted: {sorted(set(manifest) ^ expected_fields)}"
        )
    identity = dict(manifest)
    chain_id = identity.pop("chain_id")
    if (
        manifest["schema_version"] != CHAIN_SCHEMA_VERSION
        or manifest["protocol"] != "schema5-v1.1-r1-recovery-chain"
        or manifest["namespace"] != CHAIN_NAMESPACE
        or manifest["release_id"] != RELEASE_ID
        or manifest["release_tag"] != RELEASE_TAG
        or not isinstance(chain_id, str)
        or not _SHA256.fullmatch(chain_id)
        or chain_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("recovery-chain immutable identity is invalid")
    if stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
        raise ChainError("recovery-chain manifest must be read-only")

    path_names = (
        "repository",
        "results_root",
        "recovery_root",
        "source_checkout",
        "release_root",
        "state_root",
        "server_pool_root",
        "hf_home",
        "dev_python",
        "source_harness_prefix",
        "source_serving_prefix",
        "conda_executable",
        "jobs_root",
        "logs_root",
        "immutable_pins",
        "readiness_root",
    )
    paths_by_name = {
        name: _manifest_path(manifest[name], description=f"manifest {name}")
        for name in path_names
    }
    results_root = paths_by_name["results_root"]
    recovery_root = paths_by_name["recovery_root"]
    expected_paths = {
        "recovery_root": results_root / "recovery" / "schema5-v1",
        "source_checkout": recovery_root / "release_source_checkout_v1_1_r1",
        "release_root": recovery_root / "releases" / RELEASE_ID,
        "state_root": results_root / ".dispatcher-schema5-v1",
        "server_pool_root": results_root / "server_pools" / "schema5-v1",
        "jobs_root": recovery_root / "jobs" / CHAIN_NAMESPACE,
        "logs_root": recovery_root / "logs" / CHAIN_NAMESPACE,
        "immutable_pins": recovery_root / "immutable_pins.schema5-v1.json",
        "readiness_root": recovery_root / "readiness",
    }
    if any(paths_by_name[name] != expected for name, expected in expected_paths.items()):
        raise ChainError("recovery-chain canonical path contract drifted")
    if manifest_path != recovery_root / CHAIN_MANIFEST_NAME:
        raise ChainError("recovery-chain manifest is outside its canonical recovery root")
    if (
        not isinstance(manifest["release_git_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["release_git_commit"])
        or manifest["partition"] != "mit_normal"
        or not isinstance(manifest["slurm_user"], str)
        or not _SAFE_NAME.fullmatch(manifest["slurm_user"])
        or manifest["first_legacy_mutation"] != FIRST_LEGACY_MUTATION
        or manifest["heavy_io_serial_order"] != list(HEAVY_SERIAL_ORDER)
    ):
        raise ChainError("recovery-chain scalar provenance is invalid")

    reconstructed = RecoveryPaths(
        repository=paths_by_name["repository"],
        results_root=results_root,
        recovery_root=recovery_root,
        source_checkout=paths_by_name["source_checkout"],
        release_root=paths_by_name["release_root"],
        worktree=paths_by_name["release_root"] / "worktree",
        identity=paths_by_name["release_root"] / "identity",
        harness=paths_by_name["release_root"] / "environments" / "harness",
        serving=paths_by_name["release_root"] / "environments" / "serving",
        state=paths_by_name["state_root"],
        pool=paths_by_name["server_pool_root"],
        hf_home=paths_by_name["hf_home"],
        dev_python=paths_by_name["dev_python"],
        source_harness=paths_by_name["source_harness_prefix"],
        source_serving=paths_by_name["source_serving_prefix"],
        conda_executable=paths_by_name["conda_executable"],
        jobs_root=paths_by_name["jobs_root"],
        logs_root=paths_by_name["logs_root"],
        chain_manifest=manifest_path,
        immutable_pins=paths_by_name["immutable_pins"],
        readiness=paths_by_name["readiness_root"],
    )
    expected_specs = job_specs(
        reconstructed,
        commit=manifest["release_git_commit"],
        slurm_user=manifest["slurm_user"],
    )
    _validate_dag(expected_specs)
    expected_scripts = {
        spec.name: render_sbatch(
            spec, reconstructed, partition=manifest["partition"]
        )
        for spec in expected_specs
    }
    expected_manifest = _manifest_payload(
        reconstructed,
        expected_specs,
        expected_scripts,
        partition=manifest["partition"],
        git_identity={
            "release_tag": RELEASE_TAG,
            "git_commit": manifest["release_git_commit"],
        },
        slurm_user=manifest["slurm_user"],
    )
    if manifest != expected_manifest:
        raise ChainError("recovery-chain manifest differs from the fixed rendered contract")

    jobs_root = _require_canonical_path(
        reconstructed.jobs_root,
        description="immutable jobs namespace",
        kind="directory",
    )
    if stat.S_IMODE(jobs_root.stat().st_mode) & 0o222:
        raise ChainError("immutable jobs namespace must be read-only")
    _require_canonical_path(
        reconstructed.logs_root,
        description="recovery logs namespace",
        kind="directory",
    )
    if {item.name for item in jobs_root.iterdir()} != {
        spec.filename for spec in expected_specs
    }:
        raise ChainError("immutable jobs namespace contains unexpected files")
    by_name: dict[str, Mapping[str, Any]] = {}
    for spec, record in zip(expected_specs, manifest["jobs"], strict=True):
        script = jobs_root / spec.filename
        _require_canonical_path(script, description=f"{spec.name} script", kind="file")
        if stat.S_IMODE(script.stat().st_mode) & 0o222:
            raise ChainError(f"job script is writable: {script}")
        payload = script.read_bytes()
        if payload != expected_scripts[spec.name] or _sha256(script) != record["script_sha256"]:
            raise ChainError(f"job script content drifted: {script}")
        _run_checked(["bash", "-n", str(script)])
        by_name[spec.name] = record

    fleet_text = (jobs_root / "16_fleet_readiness.sbatch").read_text(encoding="utf-8")
    if "36000" not in fleet_text or "seq 1 144" in fleet_text:
        raise ChainError("fleet readiness does not use the bounded 10-hour deadline")
    smoke_text = (jobs_root / "17_smoke_readiness.sbatch").read_text(encoding="utf-8")
    if "generation + 1" not in smoke_text:
        raise ChainError("smoke job does not derive current rollout generation + 1")
    checkout_text = (jobs_root / "00_source_checkout.sbatch").read_text(encoding="utf-8")
    if (
        "git clone --no-local --no-checkout" not in checkout_text
        or "checkout --detach" not in checkout_text
    ):
        raise ChainError("source checkout is not a fresh detached no-local clone")
    consolidation_text = (jobs_root / "06_legacy_consolidate.sbatch").read_text(
        encoding="utf-8"
    )
    apply_offset = consolidation_text.find("--apply")
    release_verify_offset = consolidation_text.find("freeze_schema5_release.py")
    if release_verify_offset < 0 or apply_offset < release_verify_offset:
        raise ChainError("legacy consolidation is not fenced by live release verification")
    return {
        "passed": True,
        "chain_id": chain_id,
        "manifest": str(manifest_path),
        "job_count": len(expected_specs),
        "release_git_commit": manifest["release_git_commit"],
        "production_resume_dependencies_verified": True,
    }


def submission_argv(
    record: Mapping[str, Any],
    *,
    dependency_job_ids: Sequence[str],
    comment: str,
) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9:._-]{1,256}", comment):
        raise ChainError(f"unsafe Slurm recovery-chain comment: {comment!r}")
    if any(not str(job_id).isdigit() for job_id in dependency_job_ids):
        raise ChainError("dependency job IDs must be numeric")
    argv = [
        "sbatch",
        "--parsable",
        "--no-requeue",
        f"--comment={comment}",
    ]
    if dependency_job_ids:
        argv.append("--dependency=afterok:" + ":".join(dependency_job_ids))
    argv.append(str(record["script"]))
    return argv


def _submit_line_comment(command: str) -> str | None:
    """Recover one exact ``--comment`` value from Slurm's stored SubmitLine.

    This cluster does not persist ``JobComment`` in ``sacct`` because
    ``AccountingStoreFlags`` omits it.  Slurm does retain the original submission
    command in ``SubmitLine``, however.  Recovery must use that durable field when a
    job has already left ``squeue``; otherwise a crash after scheduler acceptance can
    look like a missing job and eventually cause a duplicate submission.
    """

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise ChainError(f"invalid sacct SubmitLine quoting: {exc}") from exc
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            values.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise ChainError("sacct SubmitLine has a valueless --comment")
            values.append(tokens[index + 1])
    if len(values) > 1:
        raise ChainError("sacct SubmitLine has duplicate --comment options")
    return values[0] if values else None


def _accounting_comment(stored: str, submit_line: str, *, job_id: str) -> str:
    """Resolve and cross-check a recovery job's accounting identity."""

    normalized = "" if stored.strip().lower() in {"", "(null)", "null", "none"} else stored.strip()
    derived = _submit_line_comment(submit_line.strip())
    if normalized and derived and normalized != derived:
        raise ChainError(
            f"sacct comment/SubmitLine conflict for recovery job {job_id}"
        )
    return normalized or derived or ""


def _query_comment_jobs(
    comment: str,
    *,
    slurm_user: str,
    since: str,
    runner: Runner,
) -> list[dict[str, str]]:
    commands = (
        ["squeue", "-u", slurm_user, "-h", "-o", "%i|%k|%j|%T"],
        [
            "sacct",
            "-u",
            slurm_user,
            "-X",
            "-n",
            "-P",
            "-S",
            since,
            "--format=JobIDRaw,Comment%256,JobName%64,State,SubmitLine",
        ],
    )
    by_id: dict[str, dict[str, str]] = {}
    for command in commands:
        proc = runner(command)
        if proc.returncode != 0:
            raise ChainError(
                f"scheduler reconciliation failed: {shlex.join(command)}: "
                f"{proc.stderr.strip()[:500]}"
            )
        source = command[0]
        for raw in proc.stdout.splitlines():
            fields = raw.rstrip("\n").split("|")
            minimum_fields = 5 if source == "sacct" else 4
            if len(fields) < minimum_fields:
                if raw.strip():
                    raise ChainError(
                        f"malformed scheduler reconciliation row: {raw[:300]!r}"
                    )
                continue
            job_id, observed_comment, job_name, state = (
                field.strip() for field in fields[:4]
            )
            if source == "sacct":
                observed_comment = _accounting_comment(
                    observed_comment,
                    fields[4],
                    job_id=job_id,
                )
            if observed_comment != comment:
                continue
            if "." in job_id:
                continue
            if not job_id.isdigit() or not job_name or not state:
                raise ChainError(
                    f"invalid scheduler identity for comment {comment}: {raw[:300]!r}"
                )
            candidate = {
                "job_id": job_id,
                "comment": observed_comment,
                "job_name": job_name,
                "state": state,
            }
            previous = by_id.get(job_id)
            if previous is not None and (
                previous["comment"] != candidate["comment"]
                or previous["job_name"] != candidate["job_name"]
            ):
                raise ChainError(
                    f"squeue/sacct identity conflict for recovery job {job_id}"
                )
            # The later accounting pass is allowed to supply a terminal state, but it
            # may not rebind the immutable comment/name identity.
            by_id[job_id] = candidate
    return sorted(by_id.values(), key=lambda row: int(row["job_id"]))


_ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
_REPAIRABLE_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def _normalize_slurm_state(value: str) -> str:
    return value.strip().split()[0].rstrip("+") if value.strip() else ""


def _query_receipt_job_states(
    *,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    runner: Runner,
) -> dict[str, dict[str, Any]]:
    """Join squeue and sacct for every exact receipt job, rejecting ambiguity."""

    receipt_by_id = {str(row["job_id"]): row for row in receipt["jobs"]}
    manifest_by_name = {str(row["name"]): row for row in manifest["jobs"]}
    requested_ids = set(receipt_by_id)
    squeue = runner(
        [
            "squeue",
            "-u",
            str(manifest["slurm_user"]),
            "-h",
            "-o",
            "%i|%T|%k|%j",
        ]
    )
    if squeue.returncode != 0:
        raise ChainError(f"cannot query active recovery jobs: {squeue.stderr[:500]}")
    active: dict[str, dict[str, str]] = {}
    for raw in squeue.stdout.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 4 or fields[0] not in requested_ids:
            continue
        job_id, state, comment, job_name = fields[:4]
        if job_id in active:
            raise ChainError(f"receipt job appears more than once in squeue: {job_id}")
        active[job_id] = {
            "state": _normalize_slurm_state(state),
            "comment": comment,
            "job_name": job_name,
        }
    accounting = runner(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(sorted(requested_ids, key=int)),
            "--format=JobIDRaw,State,ExitCode,Comment%256,JobName%64,SubmitLine",
        ]
    )
    if accounting.returncode != 0:
        raise ChainError(f"cannot query recovery job history: {accounting.stderr[:500]}")
    historical: dict[str, dict[str, str]] = {}
    for raw in accounting.stdout.splitlines():
        fields = raw.strip().split("|")
        if len(fields) < 6:
            continue
        job_id, state, exit_code, comment, job_name = (
            field.strip() for field in fields[:5]
        )
        if job_id not in requested_ids or "." in job_id:
            continue
        comment = _accounting_comment(comment, fields[5], job_id=job_id)
        candidate = {
            "state": _normalize_slurm_state(state),
            "exit_code": exit_code,
            "comment": comment,
            "job_name": job_name,
        }
        previous = historical.get(job_id)
        if previous is not None and previous != candidate:
            raise ChainError(f"receipt job has ambiguous sacct rows: {job_id}")
        historical[job_id] = candidate

    result: dict[str, dict[str, Any]] = {}
    for job_id, receipt_row in receipt_by_id.items():
        name = str(receipt_row["name"])
        expected_name = str(manifest_by_name[name]["job_name"])
        expected_comment = str(receipt_row["comment"])
        live = active.get(job_id)
        history = historical.get(job_id)
        observed = live or history
        if observed is None:
            raise ChainError(f"receipt job is absent from both squeue and sacct: {job_id}")
        if (
            observed["comment"] != expected_comment
            or observed["job_name"] != expected_name
        ):
            raise ChainError(f"scheduler identity drifted for receipt job {job_id}")
        if history is not None and (
            history["comment"] != expected_comment
            or history["job_name"] != expected_name
        ):
            raise ChainError(f"accounting identity drifted for receipt job {job_id}")
        state = live["state"] if live is not None else history["state"]
        if live is not None and state not in _ACTIVE_SLURM_STATES:
            raise ChainError(f"squeue reported non-active state {state!r} for {job_id}")
        if state not in _ACTIVE_SLURM_STATES | _REPAIRABLE_SLURM_STATES | {"COMPLETED"}:
            raise ChainError(f"unclassified scheduler state {state!r} for {job_id}")
        result[name] = {
            "job_id": job_id,
            "state": state,
            "active": live is not None,
            "exit_code": None if history is None else history.get("exit_code"),
            "comment": expected_comment,
            "job_name": expected_name,
        }
    return result


def _dependency_config_allows_fail_closed(runner: Runner) -> None:
    proc = runner(["scontrol", "show", "config"])
    if proc.returncode != 0:
        raise ChainError(f"cannot read Slurm dependency policy: {proc.stderr[:500]}")
    match = re.search(r"^DependencyParameters\s*=\s*(.*?)\s*$", proc.stdout, re.M)
    values = set(re.split(r"[,:\s]+", match.group(1).strip())) if match else set()
    if "kill_invalid_depend" not in values:
        raise ChainError(
            "Slurm must enable DependencyParameters=kill_invalid_depend so a failed "
            "ancestor cannot leave held recovery successors indefinitely"
        )


def _submission_comments(
    manifest: Mapping[str, Any], *, generation: int = 0
) -> dict[str, str]:
    chain_id = manifest["chain_id"]
    return {
        row["name"]: _job_comment(chain_id, row["name"], generation)
        for row in manifest["jobs"]
    }


def _validate_submission_journal(
    journal: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    comments: Mapping[str, str],
) -> None:
    expected_fields = {
        "schema_version",
        "protocol",
        "chain_id",
        "manifest",
        "started_at",
        "started_timestamp",
        "scheduler_since",
        "slurm_user",
        "jobs",
    }
    if set(journal) != expected_fields:
        raise ChainError("submission journal fields drifted")
    started_timestamp = journal["started_timestamp"]
    if (
        journal["schema_version"] != SUBMISSION_SCHEMA_VERSION
        or journal["protocol"] != "schema5-v1.1-r1-recovery-chain-submission"
        or journal["chain_id"] != manifest["chain_id"]
        or journal["manifest"] != str(manifest_path)
        or journal["slurm_user"] != manifest["slurm_user"]
        or not isinstance(journal["started_at"], str)
        or not isinstance(started_timestamp, (int, float))
        or isinstance(started_timestamp, bool)
        or not math.isfinite(float(started_timestamp))
        or not isinstance(journal["scheduler_since"], str)
        or journal["scheduler_since"] != _slurm_timestamp(float(started_timestamp))
        or not isinstance(journal["jobs"], dict)
    ):
        raise ChainError("submission journal identity is invalid")

    manifest_names = [row["name"] for row in manifest["jobs"]]
    journal_name_set = set(journal["jobs"])
    journal_names = manifest_names[: len(journal_name_set)]
    if journal_name_set != set(journal_names):
        raise ChainError("submission journal is not a topological job prefix")
    submitted_ids: dict[str, str] = {}
    all_job_ids: set[str] = set()
    allowed_record_fields = {
        "state",
        "name",
        "comment",
        "dependencies",
        "dependency_job_ids",
        "argv",
        "intent_created_at",
        "intent_created_timestamp",
        "attempts",
        "last_attempt_at",
        "last_attempt_timestamp",
        "last_returncode",
        "last_stderr",
        "last_stdout",
        "last_submission_rejected",
        "submission_boundary_state",
        "job_id",
        "submitted_at",
        "scheduler_state",
    }
    required_record_fields = {
        "state",
        "name",
        "comment",
        "dependencies",
        "dependency_job_ids",
        "argv",
        "intent_created_at",
        "intent_created_timestamp",
        "attempts",
    }
    rows = {row["name"]: row for row in manifest["jobs"]}
    for name in journal_names:
        record = journal["jobs"][name]
        row = rows[name]
        if (
            not isinstance(record, dict)
            or not required_record_fields <= set(record) <= allowed_record_fields
            or record["name"] != name
            or record["comment"] != comments[name]
            or record["dependencies"] != row["dependencies"]
        ):
            raise ChainError(f"submission journal record drifted for {name}")
        dependency_ids = [submitted_ids[item] for item in row["dependencies"]]
        expected_argv = submission_argv(
            row,
            dependency_job_ids=dependency_ids,
            comment=comments[name],
        )
        attempts = record["attempts"]
        intent_timestamp = record["intent_created_timestamp"]
        if (
            record["dependency_job_ids"] != dependency_ids
            or record["argv"] != expected_argv
            or not isinstance(record["intent_created_at"], str)
            or not isinstance(intent_timestamp, (int, float))
            or isinstance(intent_timestamp, bool)
            or not math.isfinite(float(intent_timestamp))
            or not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 0
        ):
            raise ChainError(f"submission journal transaction data is invalid for {name}")
        if "last_attempt_timestamp" in record:
            attempt_timestamp = record["last_attempt_timestamp"]
            if (
                not isinstance(attempt_timestamp, (int, float))
                or isinstance(attempt_timestamp, bool)
                or not math.isfinite(float(attempt_timestamp))
            ):
                raise ChainError(f"invalid attempt timestamp for {name}")
        job_id = record.get("job_id")
        if job_id is None:
            if name != journal_names[-1]:
                raise ChainError("only the final journal intent may be uncommitted")
            continue
        if not isinstance(job_id, str) or not job_id.isdigit() or job_id in all_job_ids:
            raise ChainError(f"invalid or duplicate journal job ID for {name}")
        if record["state"] not in {"submitted", "submitted_reconciled"}:
            raise ChainError(f"committed journal job has invalid state for {name}")
        all_job_ids.add(job_id)
        submitted_ids[name] = job_id


def _validate_submission_receipt(
    receipt_path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    comments: Mapping[str, str],
) -> dict[str, Any]:
    receipt_path = _require_canonical_path(
        receipt_path,
        description="chain submission receipt",
        kind="file",
    )
    if stat.S_IMODE(receipt_path.stat().st_mode) & 0o222:
        raise ChainError("chain submission receipt must be read-only")
    receipt = _read_json(receipt_path, description="chain submission receipt")
    journal_path = _require_canonical_path(
        manifest_path.parent / SUBMISSION_JOURNAL_NAME,
        description="sealed chain submission journal",
        kind="file",
    )
    if stat.S_IMODE(journal_path.stat().st_mode) & 0o222:
        raise ChainError("completed chain submission journal must be read-only")
    journal = _read_json(journal_path, description="sealed chain submission journal")
    _validate_submission_journal(
        journal,
        manifest=manifest,
        manifest_path=manifest_path,
        comments=comments,
    )
    expected_fields = {
        "schema_version",
        "protocol",
        "passed",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_journal",
        "submission_journal_sha256",
        "submitted_at",
        "dependency_policy",
        "no_requeue",
        "jobs",
        "receipt_id",
    }
    if set(receipt) != expected_fields:
        raise ChainError("chain submission receipt fields drifted")
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id")
    if (
        receipt["schema_version"] != SUBMISSION_SCHEMA_VERSION
        or receipt["protocol"] != "schema5-v1.1-r1-recovery-chain-submission"
        or receipt["passed"] is not True
        or receipt["chain_id"] != manifest["chain_id"]
        or receipt["manifest"] != str(manifest_path)
        or receipt["manifest_sha256"] != _sha256(manifest_path)
        or receipt["submission_journal"] != str(journal_path)
        or receipt["submission_journal_sha256"] != _sha256(journal_path)
        or not isinstance(receipt["submitted_at"], str)
        or receipt["dependency_policy"] != "afterok+kill_invalid_depend"
        or receipt["no_requeue"] is not True
        or not isinstance(receipt_id, str)
        or not _SHA256.fullmatch(receipt_id)
        or receipt_id != _sha256_bytes(_canonical_json(identity))
        or not isinstance(receipt["jobs"], list)
        or len(receipt["jobs"]) != len(manifest["jobs"])
    ):
        raise ChainError("chain submission receipt identity is invalid")
    submitted: dict[str, str] = {}
    observed_ids: set[str] = set()
    expected_job_fields = {
        "name",
        "job_id",
        "dependencies",
        "dependency_job_ids",
        "comment",
        "script",
        "script_sha256",
    }
    for row, record in zip(manifest["jobs"], receipt["jobs"], strict=True):
        if not isinstance(record, dict) or set(record) != expected_job_fields:
            raise ChainError("chain submission receipt job fields drifted")
        job_id = record["job_id"]
        dependency_ids = [submitted[item] for item in row["dependencies"]]
        if (
            record["name"] != row["name"]
            or not isinstance(job_id, str)
            or not job_id.isdigit()
            or job_id in observed_ids
            or record["dependencies"] != row["dependencies"]
            or record["dependency_job_ids"] != dependency_ids
            or record["comment"] != comments[row["name"]]
            or record["script"] != row["script"]
            or record["script_sha256"] != row["script_sha256"]
            or journal["jobs"].get(row["name"], {}).get("job_id") != job_id
        ):
            raise ChainError(f"chain submission receipt job drifted: {row['name']}")
        observed_ids.add(job_id)
        submitted[row["name"]] = job_id
    return receipt


def submit_chain(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    manifest_path = _lexical_absolute(manifest_path)
    _reject_retired_r1_mutation(manifest_path.parent)
    verified = verify_chain(manifest_path)
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    jobs = manifest["jobs"]
    comments = _submission_comments(manifest)
    symbolic = []
    for row in jobs:
        dependencies = [f"${{job.{name}}}" for name in row["dependencies"]]
        symbolic.append(
            {
                "name": row["name"],
                "argv": submission_argv(
                    row, dependency_job_ids=(), comment=comments[row["name"]]
                )[:-1]
                + (
                    ["--dependency=afterok:" + ":".join(dependencies)]
                    if dependencies
                    else []
                )
                + [str(row["script"])],
            }
        )
    if not apply:
        return {
            "status": "dry_run",
            "chain_id": manifest["chain_id"],
            "job_count": len(jobs),
            "submission_plan": symbolic,
        }
    runner = (
        (lambda argv: subprocess.run(argv, text=True, capture_output=True, check=False))
        if runner is None
        else runner
    )
    fixed_timestamp = now is not None
    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp):
        raise ChainError("submission timestamp must be finite")

    def boundary_timestamp() -> float:
        return timestamp if fixed_timestamp else time.time()
    recovery_root = manifest_path.parent
    journal_path = recovery_root / SUBMISSION_JOURNAL_NAME
    receipt_path = recovery_root / SUBMISSION_RECEIPT_NAME
    lock_path = recovery_root / SUBMISSION_LOCK_NAME
    with _exclusive_lock(lock_path, description="recovery-chain submitter"):
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = _validate_submission_receipt(
                receipt_path,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=comments,
            )
            return receipt | {"status": "already_submitted", "verified": verified}

        _dependency_config_allows_fail_closed(runner)
        if journal_path.exists() or journal_path.is_symlink():
            journal = _read_json(journal_path, description="chain submission journal")
            _validate_submission_journal(
                journal,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=comments,
            )
        else:
            journal = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": "schema5-v1.1-r1-recovery-chain-submission",
                "chain_id": manifest["chain_id"],
                "manifest": str(manifest_path),
                "started_at": _utc_now(),
                "started_timestamp": timestamp,
                "scheduler_since": _slurm_timestamp(timestamp),
                "slurm_user": manifest["slurm_user"],
                "jobs": {},
            }
            _atomic_json(journal_path, journal, mode=0o640)
        submitted: dict[str, str] = {}
        for row in jobs:
            name = row["name"]
            existing = journal["jobs"].get(name)
            if isinstance(existing, dict) and str(existing.get("job_id", "")).isdigit():
                matches = _query_comment_jobs(
                    comments[name],
                    slurm_user=str(manifest["slurm_user"]),
                    since=str(journal["scheduler_since"]),
                    runner=runner,
                )
                if len(matches) != 1:
                    raise ChainError(
                        f"committed submission {name} maps to {len(matches)} Slurm jobs"
                    )
                if (
                    matches[0]["job_id"] != str(existing["job_id"])
                    or matches[0]["job_name"] != row["job_name"]
                ):
                    raise ChainError(
                        f"committed submission {name} has scheduler identity drift"
                    )
                existing["scheduler_state"] = matches[0]["state"]
                _atomic_json(journal_path, journal, mode=0o640)
                submitted[name] = matches[0]["job_id"]
                continue
            dependency_ids = [submitted[dependency] for dependency in row["dependencies"]]
            comment = comments[name]
            argv = submission_argv(
                row, dependency_job_ids=dependency_ids, comment=comment
            )
            if not isinstance(existing, dict):
                intent_timestamp = boundary_timestamp()
                existing = {
                    "state": "submitting",
                    "name": name,
                    "comment": comment,
                    "dependencies": list(row["dependencies"]),
                    "dependency_job_ids": dependency_ids,
                    "argv": argv,
                    "intent_created_at": _utc_now(),
                    "intent_created_timestamp": intent_timestamp,
                    "attempts": 0,
                }
                journal["jobs"][name] = existing
                _atomic_json(journal_path, journal, mode=0o640)
            else:
                # Complete validation at function entry ensures this intent and its
                # dependency IDs exactly match the immutable manifest.
                if existing["argv"] != argv or existing["dependency_job_ids"] != dependency_ids:
                    raise ChainError(f"submission intent drifted for {name}")
            matches = _query_comment_jobs(
                comment,
                slurm_user=str(manifest["slurm_user"]),
                since=str(journal["scheduler_since"]),
                runner=runner,
            )
            if len(matches) > 1:
                raise ChainError(f"submission intent {name} maps to multiple Slurm jobs")
            if len(matches) == 1:
                if matches[0]["job_name"] != row["job_name"]:
                    raise ChainError(f"submission intent {name} has a job-name mismatch")
                job_id = matches[0]["job_id"]
                existing["state"] = "submitted_reconciled"
                existing["job_id"] = job_id
                existing["scheduler_state"] = matches[0]["state"]
                existing["submission_boundary_state"] = "committed"
                _atomic_json(journal_path, journal, mode=0o640)
                submitted[name] = job_id
                continue
            last_boundary = float(
                existing.get(
                    "last_attempt_timestamp", existing["intent_created_timestamp"]
                )
            )
            attempt_timestamp = boundary_timestamp()
            boundary_age = attempt_timestamp - last_boundary
            if (
                existing["attempts"]
                and boundary_age < VISIBILITY_GRACE_SECONDS
            ):
                raise ChainError(
                    f"submission intent {name} is inside scheduler visibility grace; retry later"
                )
            existing["attempts"] = int(existing["attempts"]) + 1
            existing["last_attempt_at"] = _utc_now()
            existing["last_attempt_timestamp"] = attempt_timestamp
            existing["last_submission_rejected"] = False
            existing["submission_boundary_state"] = "sbatch_in_flight"
            # Persist before the external boundary.  If this process dies after Slurm
            # accepts the job, the next submitter reconciles the unique comment through
            # both squeue and sacct instead of issuing a duplicate.
            _atomic_json(journal_path, journal, mode=0o640)
            proc = runner(argv)
            existing["last_returncode"] = int(proc.returncode)
            existing["last_stderr"] = proc.stderr.strip()[:1000]
            existing["last_stdout"] = proc.stdout.strip()[:1000]
            if proc.returncode != 0:
                existing["last_submission_rejected"] = True
                existing["submission_boundary_state"] = "rejected"
                _atomic_json(journal_path, journal, mode=0o640)
                raise ChainError(
                    f"sbatch rejected recovery job {name}: {proc.stderr.strip()[:500]}"
                )
            job_id = proc.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                existing["submission_boundary_state"] = "ambiguous_response"
                _atomic_json(journal_path, journal, mode=0o640)
                raise ChainError(f"sbatch returned an invalid job ID for {name}: {job_id!r}")
            existing["state"] = "submitted"
            existing["job_id"] = job_id
            existing["submitted_at"] = _utc_now()
            existing["last_submission_rejected"] = False
            existing["submission_boundary_state"] = "committed"
            _atomic_json(journal_path, journal, mode=0o640)
            submitted[name] = job_id
        _validate_submission_journal(
            journal,
            manifest=manifest,
            manifest_path=manifest_path,
            comments=comments,
        )
        os.chmod(journal_path, 0o444)
        _fsync_directory(journal_path.parent)
        receipt = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "protocol": "schema5-v1.1-r1-recovery-chain-submission",
            "passed": True,
            "chain_id": manifest["chain_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "submission_journal": str(journal_path),
            "submission_journal_sha256": _sha256(journal_path),
            "submitted_at": _utc_now(),
            "dependency_policy": "afterok+kill_invalid_depend",
            "no_requeue": True,
            "jobs": [
                {
                    "name": row["name"],
                    "job_id": submitted[row["name"]],
                    "dependencies": list(row["dependencies"]),
                    "dependency_job_ids": [submitted[item] for item in row["dependencies"]],
                    "comment": comments[row["name"]],
                    "script": row["script"],
                    "script_sha256": row["script_sha256"],
                }
                for row in jobs
            ],
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        _atomic_json(receipt_path, receipt, mode=0o444)
        validated_receipt = _validate_submission_receipt(
            receipt_path,
            manifest=manifest,
            manifest_path=manifest_path,
            comments=comments,
        )
        return validated_receipt | {"status": "submitted", "verified": verified}


def _validate_repair_receipt(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    generation: int,
    parent_path: Path,
) -> dict[str, Any]:
    path = _require_canonical_path(path, description="repair receipt", kind="file")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError("repair receipt must be read-only")
    receipt = _read_json(path, description="repair receipt")
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id", None)
    parent = _read_json(parent_path, description="parent chain receipt")
    if not isinstance(parent.get("jobs"), list):
        raise ChainError("parent chain receipt has no valid jobs")
    journal_path = _require_canonical_path(
        path.parent / SUBMISSION_JOURNAL_NAME,
        description="sealed repair submission journal",
        kind="file",
    )
    if stat.S_IMODE(journal_path.stat().st_mode) & 0o222:
        raise ChainError("completed repair submission journal must be read-only")
    journal = _read_json(journal_path, description="sealed repair submission journal")
    repair_names = _validate_repair_journal(
        journal,
        manifest=manifest,
        base=parent,
        base_path=parent_path,
        generation=generation,
    )
    repair_set = set(repair_names)
    expected_fields = {
        "schema_version", "protocol", "passed", "chain_id", "manifest",
        "manifest_sha256", "submission_journal", "submission_journal_sha256",
        "repair_generation", "parent_receipt", "parent_receipt_sha256",
        "submitted_at", "dependency_policy", "no_requeue", "jobs", "receipt_id",
    }
    parent_by_name = {
        str(record.get("name")): record
        for record in parent["jobs"]
        if isinstance(record, dict)
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or receipt.get("protocol") != "schema5-v1.1-r1-recovery-chain-repair"
        or receipt.get("passed") is not True
        or receipt.get("chain_id") != manifest["chain_id"]
        or receipt.get("manifest") != str(manifest_path)
        or receipt.get("manifest_sha256") != _sha256(manifest_path)
        or receipt.get("submission_journal") != str(journal_path)
        or receipt.get("submission_journal_sha256") != _sha256(journal_path)
        or receipt.get("repair_generation") != generation
        or receipt.get("parent_receipt") != str(parent_path)
        or receipt.get("parent_receipt_sha256") != _sha256(parent_path)
        or not isinstance(receipt.get("submitted_at"), str)
        or receipt.get("dependency_policy") != "afterok+kill_invalid_depend"
        or receipt.get("no_requeue") is not True
        or not isinstance(receipt_id, str)
        or not _SHA256.fullmatch(receipt_id)
        or receipt_id != _sha256_bytes(_canonical_json(identity))
        or not isinstance(receipt.get("jobs"), list)
        or len(receipt["jobs"]) != len(manifest["jobs"])
    ):
        raise ChainError("repair receipt identity is invalid")
    submitted: dict[str, str] = {}
    observed_ids: set[str] = set()
    fields = {
        "name", "job_id", "dependencies", "dependency_job_ids", "comment",
        "script", "script_sha256", "generation", "disposition",
    }
    for row, record in zip(manifest["jobs"], receipt["jobs"], strict=True):
        if not isinstance(record, dict) or set(record) != fields:
            raise ChainError("repair receipt job fields drifted")
        job_generation = record["generation"]
        job_id = record["job_id"]
        dependencies = [submitted[item] for item in row["dependencies"]]
        prior = parent_by_name.get(row["name"])
        prior_generation = (
            prior.get("generation", 0) if isinstance(prior, dict) else None
        )
        reused = record["disposition"] == "reused_completed"
        expected_reused = row["name"] not in repair_set
        repair_record = journal["jobs"].get(row["name"])
        if (
            record["name"] != row["name"]
            or not isinstance(job_generation, int)
            or isinstance(job_generation, bool)
            or not 0 <= job_generation <= generation
            or record["disposition"] not in {"reused_completed", "resubmitted"}
            or reused != expected_reused
            or (record["disposition"] == "resubmitted" and job_generation != generation)
            or prior is None
            or (
                reused
                and (
                    job_id != prior.get("job_id")
                    or record["comment"] != prior.get("comment")
                    or job_generation != prior_generation
                )
            )
            or (not reused and job_id == prior.get("job_id"))
            or (
                not reused
                and (
                    not isinstance(repair_record, dict)
                    or repair_record.get("job_id") != job_id
                )
            )
            or not isinstance(job_id, str)
            or not job_id.isdigit()
            or job_id in observed_ids
            or record["dependencies"] != row["dependencies"]
            or record["dependency_job_ids"] != dependencies
            or record["comment"]
            != _job_comment(manifest["chain_id"], row["name"], job_generation)
            or record["script"] != row["script"]
            or record["script_sha256"] != row["script_sha256"]
        ):
            raise ChainError(f"repair receipt job drifted: {row['name']}")
        observed_ids.add(job_id)
        submitted[row["name"]] = job_id
    return receipt


def _latest_chain_receipt(
    *, manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], Path, int, Path | None]:
    recovery_root = manifest_path.parent
    original_path = recovery_root / SUBMISSION_RECEIPT_NAME
    original = _validate_submission_receipt(
        original_path,
        manifest=manifest,
        manifest_path=manifest_path,
        comments=_submission_comments(manifest, generation=0),
    )
    latest: dict[str, Any] = original
    latest_path = original_path
    latest_generation = 0
    pending: Path | None = None
    repair_root = recovery_root / REPAIR_ROOT_NAME
    if not repair_root.exists():
        return latest, latest_path, latest_generation, pending
    repair_root = _require_canonical_path(
        repair_root, description="repair root", kind="directory"
    )
    entries = sorted(repair_root.iterdir(), key=lambda item: item.name)
    expected_generation = 1
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir() or entry.name != f"g{expected_generation:04d}":
            raise ChainError(f"repair generations are not contiguous and canonical: {entry}")
        receipt_path = entry / SUBMISSION_RECEIPT_NAME
        if not receipt_path.exists():
            if entry != entries[-1]:
                raise ChainError("only the latest repair generation may be incomplete")
            pending = entry
            break
        latest = _validate_repair_receipt(
            receipt_path,
            manifest=manifest,
            manifest_path=manifest_path,
            generation=expected_generation,
            parent_path=latest_path,
        )
        latest_path = receipt_path
        latest_generation = expected_generation
        expected_generation += 1
    return latest, latest_path, latest_generation, pending


def _validate_repair_journal(
    journal: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    base: Mapping[str, Any],
    base_path: Path,
    generation: int,
) -> list[str]:
    expected_fields = {
        "schema_version",
        "protocol",
        "chain_id",
        "repair_generation",
        "base_receipt",
        "base_receipt_sha256",
        "repair_jobs",
        "started_at",
        "started_timestamp",
        "scheduler_since",
        "jobs",
    }
    started = journal.get("started_timestamp")
    repair_names = journal.get("repair_jobs")
    if (
        set(journal) != expected_fields
        or journal.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or journal.get("protocol") != "schema5-v1.1-r1-recovery-chain-repair-journal"
        or journal.get("chain_id") != manifest["chain_id"]
        or journal.get("repair_generation") != generation
        or journal.get("base_receipt") != str(base_path)
        or journal.get("base_receipt_sha256") != _sha256(base_path)
        or not isinstance(journal.get("started_at"), str)
        or not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not math.isfinite(float(started))
        or journal.get("scheduler_since") != _slurm_timestamp(float(started))
        or not isinstance(repair_names, list)
        or not repair_names
        or len(repair_names) != len(set(repair_names))
        or not isinstance(journal.get("jobs"), dict)
    ):
        raise ChainError("repair journal identity is invalid")
    manifest_order = [row["name"] for row in manifest["jobs"]]
    if any(not isinstance(name, str) or name not in manifest_order for name in repair_names):
        raise ChainError("repair journal names an unknown job")
    expected_repair_order = [name for name in manifest_order if name in set(repair_names)]
    if repair_names != expected_repair_order:
        raise ChainError("repair journal jobs are not in immutable DAG order")

    journal_name_set = set(journal["jobs"])
    journal_names = repair_names[: len(journal_name_set)]
    if journal_name_set != set(journal_names):
        raise ChainError("repair journal intents are not a topological repair prefix")
    base_by_name = {row["name"]: row for row in base["jobs"]}
    manifest_by_name = {row["name"]: row for row in manifest["jobs"]}
    repair_set = set(repair_names)
    submitted: dict[str, str] = {}
    observed_ids = {str(row["job_id"]) for row in base["jobs"]}
    allowed = {
        "name",
        "comment",
        "dependency_job_ids",
        "argv",
        "attempts",
        "intent_created_timestamp",
        "last_attempt_timestamp",
        "submission_boundary_state",
        "last_returncode",
        "last_submission_rejected",
        "last_stderr",
        "last_stdout",
        "job_id",
        "state",
        "scheduler_state",
    }
    required = {
        "name",
        "comment",
        "dependency_job_ids",
        "argv",
        "attempts",
        "intent_created_timestamp",
    }
    for name in manifest_order:
        row = manifest_by_name[name]
        if name not in repair_set:
            submitted[name] = str(base_by_name[name]["job_id"])
            continue
        if name not in journal_name_set:
            continue
        record = journal["jobs"][name]
        dependency_ids = [submitted[item] for item in row["dependencies"]]
        comment = _job_comment(manifest["chain_id"], name, generation)
        attempts = record.get("attempts") if isinstance(record, dict) else None
        intent_timestamp = (
            record.get("intent_created_timestamp")
            if isinstance(record, dict)
            else None
        )
        if (
            not isinstance(record, dict)
            or not required <= set(record) <= allowed
            or record.get("name") != name
            or record.get("comment") != comment
            or record.get("dependency_job_ids") != dependency_ids
            or record.get("argv")
            != submission_argv(
                row, dependency_job_ids=dependency_ids, comment=comment
            )
            or not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 0
            or not isinstance(intent_timestamp, (int, float))
            or isinstance(intent_timestamp, bool)
            or not math.isfinite(float(intent_timestamp))
        ):
            raise ChainError(f"repair journal transaction data drifted for {name}")
        if "last_attempt_timestamp" in record:
            value = record["last_attempt_timestamp"]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ChainError(f"repair journal attempt timestamp is invalid for {name}")
        job_id = record.get("job_id")
        if job_id is None:
            if name != journal_names[-1]:
                raise ChainError("only the final repair intent may be uncommitted")
            continue
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or job_id in observed_ids
            or record.get("state") not in {"submitted", "submitted_reconciled"}
        ):
            raise ChainError(f"repair journal job ID/state is invalid for {name}")
        observed_ids.add(job_id)
        submitted[name] = job_id
    return list(repair_names)


def _preflight_repair_stage_artifacts(
    manifest: Mapping[str, Any], repair_names: Sequence[str], manifest_path: Path
) -> None:
    if "source_checkout" in repair_names:
        checkout = Path(manifest["source_checkout"])
        if checkout.exists() or checkout.is_symlink():
            raise ChainError(
                f"failed source checkout left {checkout}; preserve and remove or adopt "
                "it explicitly before repair"
            )
    if "release_materialize" in repair_names:
        release_root = Path(manifest["release_root"])
        if release_root.exists() or release_root.is_symlink():
            raise ChainError(
                f"failed materialization is preserved at {release_root}; run "
                "quarantine-materialization --apply before repair"
            )


def repair_chain(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Resubmit only the non-completed DAG suffix under a new durable generation."""

    manifest_path = _lexical_absolute(manifest_path)
    _reject_retired_r1_mutation(manifest_path.parent)
    verify_chain(manifest_path)
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    runner = (
        (lambda argv: subprocess.run(argv, text=True, capture_output=True, check=False))
        if runner is None else runner
    )
    fixed_timestamp = now is not None
    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp):
        raise ChainError("repair timestamp must be finite")

    def boundary_timestamp() -> float:
        return timestamp if fixed_timestamp else time.time()
    lock_path = manifest_path.parent / SUBMISSION_LOCK_NAME
    with _exclusive_lock(lock_path, description="recovery-chain repair"):
        base, base_path, base_generation, pending = _latest_chain_receipt(
            manifest=manifest, manifest_path=manifest_path
        )
        states = _query_receipt_job_states(
            receipt=base, manifest=manifest, runner=runner
        )
        active = sorted(name for name, item in states.items() if item["active"])
        failed = [
            row["name"]
            for row in manifest["jobs"]
            if states[row["name"]]["state"] in _REPAIRABLE_SLURM_STATES
        ]
        if active:
            report = {"status": "in_progress", "active_jobs": active, "repair_jobs": []}
            if apply:
                raise ChainError(f"cannot repair while receipt jobs are active: {active}")
            return report
        if not failed:
            if pending is not None:
                raise ChainError("pending repair exists although its parent DAG is complete")
            return {"status": "complete", "repair_jobs": [], "receipt": str(base_path)}
        failed_set = set(failed)
        by_name = {row["name"]: row for row in manifest["jobs"]}
        for name, item in states.items():
            if item["state"] == "COMPLETED" and (
                _manifest_ancestors(name, by_name) & failed_set
            ):
                raise ChainError(f"completed job {name} has a failed ancestor")

        if pending is None:
            generation = base_generation + 1
            repair_names = failed
            repair_root = manifest_path.parent / REPAIR_ROOT_NAME
            generation_root = repair_root / f"g{generation:04d}"
            if not apply:
                return {
                    "status": "dry_run", "repair_generation": generation,
                    "repair_jobs": repair_names,
                    "states": {name: states[name]["state"] for name in repair_names},
                    "would_write": str(generation_root),
                }
            _dependency_config_allows_fail_closed(runner)
            _preflight_repair_stage_artifacts(
                manifest, repair_names, manifest_path
            )
            repair_root.mkdir(parents=True, exist_ok=True)
            _require_canonical_path(
                repair_root, description="repair root", kind="directory"
            )
            journal = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": "schema5-v1.1-r1-recovery-chain-repair-journal",
                "chain_id": manifest["chain_id"], "repair_generation": generation,
                "base_receipt": str(base_path), "base_receipt_sha256": _sha256(base_path),
                "repair_jobs": repair_names, "started_at": _utc_now(),
                "started_timestamp": timestamp, "scheduler_since": _slurm_timestamp(timestamp),
                "jobs": {},
            }
            temporary_generation = manifest_path.parent / (
                f".{REPAIR_ROOT_NAME}.g{generation:04d}.{os.getpid()}.{uuid.uuid4().hex}"
            )
            temporary_generation.mkdir(mode=0o750)
            try:
                _atomic_json(
                    temporary_generation / SUBMISSION_JOURNAL_NAME,
                    journal,
                    mode=0o640,
                )
                _fsync_directory(temporary_generation)
                os.rename(temporary_generation, generation_root)
                _fsync_directory(repair_root)
            finally:
                if temporary_generation.exists():
                    shutil.rmtree(temporary_generation)
        else:
            generation_root = pending
            generation = int(pending.name[1:])
            journal = _read_json(
                generation_root / SUBMISSION_JOURNAL_NAME,
                description="repair submission journal",
            )
            repair_names = _validate_repair_journal(
                journal,
                manifest=manifest,
                base=base,
                base_path=base_path,
                generation=generation,
            )
            if repair_names != failed:
                raise ChainError(
                    "pending repair set no longer equals its parent's failed job set"
                )
            if not apply:
                return {"status": "pending_repair", "repair_generation": generation, "repair_jobs": repair_names}
            _dependency_config_allows_fail_closed(runner)
            _preflight_repair_stage_artifacts(
                manifest, repair_names, manifest_path
            )

        base_by_name = {row["name"]: row for row in base["jobs"]}
        repair_set = set(repair_names)
        submitted: dict[str, str] = {}
        comments = _submission_comments(manifest, generation=generation)
        journal_path = generation_root / SUBMISSION_JOURNAL_NAME
        _validate_repair_journal(
            journal,
            manifest=manifest,
            base=base,
            base_path=base_path,
            generation=generation,
        )
        for row in manifest["jobs"]:
            name = row["name"]
            if name not in repair_set:
                submitted[name] = str(base_by_name[name]["job_id"])
                continue
            dependency_ids = [submitted[item] for item in row["dependencies"]]
            comment = comments[name]
            argv = submission_argv(row, dependency_job_ids=dependency_ids, comment=comment)
            record = journal["jobs"].get(name)
            if not isinstance(record, dict):
                intent_timestamp = boundary_timestamp()
                record = {
                    "name": name, "comment": comment, "dependency_job_ids": dependency_ids,
                    "argv": argv, "attempts": 0,
                    "intent_created_timestamp": intent_timestamp,
                }
                journal["jobs"][name] = record
                _atomic_json(journal_path, journal, mode=0o640)
            if record.get("argv") != argv or record.get("dependency_job_ids") != dependency_ids:
                raise ChainError(f"repair intent drifted for {name}")
            matches = _query_comment_jobs(
                comment, slurm_user=manifest["slurm_user"],
                since=journal["scheduler_since"], runner=runner,
            )
            if len(matches) > 1:
                raise ChainError(f"repair intent {name} maps to multiple jobs")
            if len(matches) == 1:
                if (
                    matches[0]["job_name"] != row["job_name"]
                    or (
                        record.get("job_id") is not None
                        and matches[0]["job_id"] != record["job_id"]
                    )
                ):
                    raise ChainError(f"repair scheduler identity drifted for {name}")
                record["job_id"] = matches[0]["job_id"]
                record["state"] = "submitted_reconciled"
                record["scheduler_state"] = matches[0]["state"]
                record["submission_boundary_state"] = "committed"
                _atomic_json(journal_path, journal, mode=0o640)
                submitted[name] = matches[0]["job_id"]
                continue
            if record.get("job_id"):
                raise ChainError(f"committed repair job disappeared from scheduler: {name}")
            attempt_timestamp = boundary_timestamp()
            boundary_age = attempt_timestamp - float(
                record.get("last_attempt_timestamp", record["intent_created_timestamp"])
            )
            if record["attempts"] and boundary_age < VISIBILITY_GRACE_SECONDS:
                raise ChainError(f"repair intent {name} is inside scheduler visibility grace")
            record["attempts"] = int(record["attempts"]) + 1
            record["last_attempt_timestamp"] = attempt_timestamp
            record["last_submission_rejected"] = False
            record["submission_boundary_state"] = "sbatch_in_flight"
            _atomic_json(journal_path, journal, mode=0o640)
            proc = runner(argv)
            record["last_returncode"] = int(proc.returncode)
            record["last_stderr"] = proc.stderr.strip()[:1000]
            record["last_stdout"] = proc.stdout.strip()[:1000]
            record["last_submission_rejected"] = proc.returncode != 0
            if proc.returncode != 0:
                record["submission_boundary_state"] = "rejected"
                _atomic_json(journal_path, journal, mode=0o640)
                raise ChainError(f"sbatch rejected repair job {name}: {proc.stderr[:500]}")
            job_id = proc.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                record["submission_boundary_state"] = "ambiguous_response"
                _atomic_json(journal_path, journal, mode=0o640)
                raise ChainError(f"sbatch returned invalid repair job ID {job_id!r}")
            record.update(job_id=job_id, state="submitted", submission_boundary_state="committed")
            _atomic_json(journal_path, journal, mode=0o640)
            submitted[name] = job_id

        _validate_repair_journal(
            journal,
            manifest=manifest,
            base=base,
            base_path=base_path,
            generation=generation,
        )
        os.chmod(journal_path, 0o444)
        _fsync_directory(journal_path.parent)

        receipt_jobs = []
        for row in manifest["jobs"]:
            name = row["name"]
            if name in repair_set:
                job_generation = generation
                disposition = "resubmitted"
                comment = comments[name]
            else:
                prior = base_by_name[name]
                job_generation = int(prior.get("generation", 0))
                disposition = "reused_completed"
                comment = prior["comment"]
            receipt_jobs.append({
                "name": name, "job_id": submitted[name],
                "dependencies": list(row["dependencies"]),
                "dependency_job_ids": [submitted[item] for item in row["dependencies"]],
                "comment": comment, "script": row["script"],
                "script_sha256": row["script_sha256"], "generation": job_generation,
                "disposition": disposition,
            })
        receipt = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "protocol": "schema5-v1.1-r1-recovery-chain-repair", "passed": True,
            "chain_id": manifest["chain_id"], "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "submission_journal": str(journal_path),
            "submission_journal_sha256": _sha256(journal_path),
            "repair_generation": generation,
            "parent_receipt": str(base_path), "parent_receipt_sha256": _sha256(base_path),
            "submitted_at": _utc_now(), "dependency_policy": "afterok+kill_invalid_depend",
            "no_requeue": True, "jobs": receipt_jobs,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        receipt_path = generation_root / SUBMISSION_RECEIPT_NAME
        _atomic_json(receipt_path, receipt, mode=0o444)
        validated = _validate_repair_receipt(
            receipt_path, manifest=manifest, manifest_path=manifest_path,
            generation=generation, parent_path=base_path,
        )
        return validated | {"status": "repair_submitted", "repair_jobs": repair_names}


def quarantine_partial_materialization(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Atomically preserve a failed materialization so its job can be retried.

    This operation never deletes or rewrites the partial tree.  It proves the exact
    materialization job is terminal, journals the source inode first, and then uses a
    same-filesystem rename into a deterministic quarantine path.
    """

    manifest_path = _lexical_absolute(manifest_path)
    verify_chain(manifest_path)
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    runner = (
        (lambda argv: subprocess.run(argv, text=True, capture_output=True, check=False))
        if runner is None
        else runner
    )
    recovery_root = _require_canonical_path(
        manifest_path.parent,
        description="recovery root",
        kind="directory",
    )
    release_root = _lexical_absolute(Path(manifest["release_root"]))
    expected_release_root = recovery_root / "releases" / RELEASE_ID
    if release_root != expected_release_root:
        raise ChainError(
            f"release root is not the pinned v1.1 destination: {release_root}"
        )

    lock_path = recovery_root / SUBMISSION_LOCK_NAME
    with _exclusive_lock(lock_path, description="materialization quarantine"):
        receipt, receipt_path, _generation, pending = _latest_chain_receipt(
            manifest=manifest,
            manifest_path=manifest_path,
        )
        if pending is not None:
            pending_journal = _read_json(
                pending / SUBMISSION_JOURNAL_NAME,
                description="pending repair journal",
            )
            pending_generation = int(pending.name[1:])
            _validate_repair_journal(
                pending_journal,
                manifest=manifest,
                base=receipt,
                base_path=receipt_path,
                generation=pending_generation,
            )
            if pending_journal.get("jobs") != {}:
                raise ChainError(
                    "cannot quarantine while a repair generation has scheduler intents"
                )

        states = _query_receipt_job_states(
            receipt=receipt,
            manifest=manifest,
            runner=runner,
        )
        active = sorted(name for name, row in states.items() if row["active"])
        if active:
            raise ChainError(
                f"cannot quarantine while recovery-chain jobs are active: {active}"
            )
        materialize_state = states["release_materialize"]
        if materialize_state["state"] not in _REPAIRABLE_SLURM_STATES:
            raise ChainError(
                "release_materialize must have a terminal failed state before "
                f"quarantine, got {materialize_state['state']}"
            )
        materialize_job_id = str(materialize_state["job_id"])
        quarantine_parent = release_root.parent / QUARANTINE_ROOT_NAME
        destination = (
            quarantine_parent
            / f"{RELEASE_ID}.partial-job-{materialize_job_id}"
        )
        evidence_root = recovery_root / QUARANTINE_EVIDENCE_ROOT_NAME
        for description, path in (
            ("materialization quarantine directory", quarantine_parent),
            ("materialization quarantine evidence directory", evidence_root),
        ):
            if path.exists() or path.is_symlink():
                _require_canonical_path(path, description=description, kind="directory")
        evidence_stem = f"partial-job-{materialize_job_id}"
        intent_path = evidence_root / f"{evidence_stem}.intent.json"
        completion_path = evidence_root / f"{evidence_stem}.complete.json"

        source_exists = release_root.exists() or release_root.is_symlink()
        destination_exists = destination.exists() or destination.is_symlink()
        if source_exists and destination_exists:
            raise ChainError(
                "partial materialization exists at both source and quarantine paths"
            )
        if source_exists:
            materialized_path = _require_canonical_path(
                release_root,
                description="partial materialization",
                kind="directory",
            )
            for forbidden_marker in (
                materialized_path / "MATERIALIZATION_COMPLETE.json",
                materialized_path / "identity" / "RELEASE_COMPLETE.json",
            ):
                if forbidden_marker.exists() or forbidden_marker.is_symlink():
                    raise ChainError(
                        "refusing to quarantine a materialization with a completion "
                        f"marker: {forbidden_marker}"
                    )
        elif destination_exists:
            materialized_path = _require_canonical_path(
                destination,
                description="quarantined partial materialization",
                kind="directory",
            )
        else:
            raise ChainError(
                f"no partial materialization exists at {release_root} or {destination}"
            )
        source_stat = os.lstat(materialized_path)

        expected_intent = {
            "schema_version": 1,
            "protocol": "schema5-v1.1-r1-partial-materialization-quarantine-intent",
            "chain_id": manifest["chain_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "release_id": RELEASE_ID,
            "materialize_job_id": materialize_job_id,
            "materialize_job_state": materialize_state["state"],
            "source": str(release_root),
            "destination": str(destination),
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
        }

        if intent_path.exists() or intent_path.is_symlink():
            intent = _read_json(intent_path, description="materialization quarantine intent")
            if stat.S_IMODE(intent_path.stat().st_mode) & 0o222:
                raise ChainError("materialization quarantine intent must be read-only")
            intent_identity = dict(intent)
            intent_id = intent_identity.pop("intent_id", None)
            created_at = intent_identity.pop("created_at", None)
            if (
                intent_identity != expected_intent
                or not isinstance(created_at, str)
                or not isinstance(intent_id, str)
                or not _SHA256.fullmatch(intent_id)
                or intent_id
                != _sha256_bytes(
                    _canonical_json(expected_intent | {"created_at": created_at})
                )
            ):
                raise ChainError("materialization quarantine intent identity drifted")
        else:
            if destination_exists:
                raise ChainError(
                    "quarantined tree exists without its marker-first intent"
                )
            intent = expected_intent | {"created_at": _utc_now()}
            intent["intent_id"] = _sha256_bytes(_canonical_json(intent))

        if completion_path.exists() or completion_path.is_symlink():
            completion = _read_json(
                completion_path,
                description="materialization quarantine completion",
            )
            if stat.S_IMODE(completion_path.stat().st_mode) & 0o222:
                raise ChainError("materialization quarantine completion must be read-only")
            completion_identity = dict(completion)
            completion_id = completion_identity.pop("completion_id", None)
            expected_completion = {
                "schema_version": 1,
                "protocol": "schema5-v1.1-r1-partial-materialization-quarantine",
                "passed": True,
                "release_id": RELEASE_ID,
                "materialize_job_id": materialize_job_id,
                "materialize_job_state": materialize_state["state"],
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
                "intent": str(intent_path),
                "intent_sha256": _sha256(intent_path),
                "intent_id": intent["intent_id"],
                "source": str(release_root),
                "destination": str(destination),
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
            }
            completed_at = completion_identity.pop("completed_at", None)
            if (
                completion_identity != expected_completion
                or not isinstance(completed_at, str)
                or not isinstance(completion_id, str)
                or not _SHA256.fullmatch(completion_id)
                or completion_id
                != _sha256_bytes(
                    _canonical_json(expected_completion | {"completed_at": completed_at})
                )
                or source_exists
                or not destination_exists
            ):
                raise ChainError("materialization quarantine completion drifted")
            return completion | {"status": "already_quarantined"}

        report = {
            "status": "dry_run" if not apply else "quarantining",
            "materialize_job_id": materialize_job_id,
            "materialize_job_state": materialize_state["state"],
            "source": str(release_root),
            "destination": str(destination),
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
            "would_resume_incomplete_rename": destination_exists,
        }
        if not apply:
            return report

        evidence_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        quarantine_parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        _require_canonical_path(
            evidence_root,
            description="materialization quarantine evidence directory",
            kind="directory",
        )
        _require_canonical_path(
            quarantine_parent,
            description="materialization quarantine directory",
            kind="directory",
        )
        if os.lstat(quarantine_parent).st_dev != source_stat.st_dev:
            raise ChainError("partial materialization quarantine is not same-filesystem")
        if not intent_path.exists():
            _atomic_json(intent_path, intent, mode=0o444)

        if source_exists:
            # Recheck exact scheduler truth and the source inode at the destructive
            # boundary.  The operation is a recoverable rename, never a deletion.
            current_states = _query_receipt_job_states(
                receipt=receipt,
                manifest=manifest,
                runner=runner,
            )
            if any(row["active"] for row in current_states.values()):
                raise ChainError("a recovery-chain job became active before quarantine")
            current_materialize = current_states["release_materialize"]
            if (
                current_materialize["job_id"] != materialize_job_id
                or current_materialize["state"] not in _REPAIRABLE_SLURM_STATES
            ):
                raise ChainError("materialization job state changed before quarantine")
            current_stat = os.lstat(release_root)
            if (
                current_stat.st_dev != source_stat.st_dev
                or current_stat.st_ino != source_stat.st_ino
            ):
                raise ChainError("partial materialization inode changed before quarantine")
            for forbidden_marker in (
                release_root / "MATERIALIZATION_COMPLETE.json",
                release_root / "identity" / "RELEASE_COMPLETE.json",
            ):
                if forbidden_marker.exists() or forbidden_marker.is_symlink():
                    raise ChainError("completion marker appeared before quarantine")
            os.rename(release_root, destination)
            _fsync_directory(release_root.parent)
            _fsync_directory(quarantine_parent)

        destination_stat = os.lstat(destination)
        if (
            destination_stat.st_dev != source_stat.st_dev
            or destination_stat.st_ino != source_stat.st_ino
            or release_root.exists()
            or release_root.is_symlink()
        ):
            raise ChainError("materialization quarantine rename did not preserve identity")
        completion = {
            "schema_version": 1,
            "protocol": "schema5-v1.1-r1-partial-materialization-quarantine",
            "passed": True,
            "release_id": RELEASE_ID,
            "materialize_job_id": materialize_job_id,
            "materialize_job_state": materialize_state["state"],
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "intent": str(intent_path),
            "intent_sha256": _sha256(intent_path),
            "intent_id": intent["intent_id"],
            "source": str(release_root),
            "destination": str(destination),
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
            "completed_at": _utc_now(),
        }
        completion["completion_id"] = _sha256_bytes(_canonical_json(completion))
        _atomic_json(completion_path, completion, mode=0o444)
        return completion | {"status": "quarantined"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="dry-run or publish the v1.1-r1 chain")
    render.add_argument("--repository", required=True, type=Path)
    render.add_argument("--results-root", required=True, type=Path)
    render.add_argument("--recovery-root", required=True, type=Path)
    render.add_argument("--hf-home", required=True, type=Path)
    render.add_argument("--dev-python", required=True, type=Path)
    render.add_argument("--source-harness-prefix", required=True, type=Path)
    render.add_argument("--source-serving-prefix", required=True, type=Path)
    render.add_argument("--conda-executable", required=True, type=Path)
    render.add_argument("--partition", choices=("mit_normal",), default="mit_normal")
    render.add_argument("--slurm-user", default=getpass.getuser())
    render.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser("verify", help="verify an immutable rendered chain")
    verify.add_argument("--chain-manifest", required=True, type=Path)
    submit = subparsers.add_parser("submit", help="dry-run or transactionally submit")
    submit.add_argument("--chain-manifest", required=True, type=Path)
    submit.add_argument("--apply", action="store_true")
    repair = subparsers.add_parser(
        "repair",
        aliases=["repair-chain"],
        help="reconcile a submitted chain and transactionally repair terminal jobs",
    )
    repair.add_argument("--chain-manifest", required=True, type=Path)
    repair.add_argument("--apply", action="store_true")
    quarantine = subparsers.add_parser(
        "quarantine-materialization",
        help="preserve a failed partial release tree before repairing the chain",
    )
    quarantine.add_argument("--chain-manifest", required=True, type=Path)
    quarantine.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "render":
            paths = recovery_paths(
                repository=args.repository,
                results_root=args.results_root,
                recovery_root=args.recovery_root,
                hf_home=args.hf_home,
                dev_python=args.dev_python,
                source_harness=args.source_harness_prefix,
                source_serving=args.source_serving_prefix,
                conda_executable=args.conda_executable,
            )
            result = render_chain(
                paths,
                partition=args.partition,
                slurm_user=args.slurm_user,
                apply=args.apply,
            )
        elif args.command == "verify":
            result = verify_chain(args.chain_manifest)
        elif args.command == "submit":
            result = submit_chain(args.chain_manifest, apply=args.apply)
        elif args.command in {"repair", "repair-chain"}:
            result = repair_chain(args.chain_manifest, apply=args.apply)
        elif args.command == "quarantine-materialization":
            result = quarantine_partial_materialization(
                args.chain_manifest,
                apply=args.apply,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (OSError, ValueError, ChainError) as exc:
        print(f"[schema5-chain] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
