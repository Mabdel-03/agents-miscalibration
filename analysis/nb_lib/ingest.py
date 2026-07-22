"""Raw run cells -> analysis caches (items, agent-rounds, deduped cell aggregates).

Reads $ASYS_RESULTS_ROOT/<run_id>/cells/*/ with the same robust tolerances as
analysis/examples/findings_summary.py (skip malformed JSONL lines and corrupt meta.json
from preemption partial-writes), plus one fix that script lacks: results.jsonl rows are
DEDUPED on qid (keep first) — resumed cells append in place, which inflated n_questions
up to 546 in ~200 cells and biased their aggregates.

Outputs (written atomically: *.tmp then os.replace) under --out-dir:
  items_v1.parquet        one row per (cell_id, qid), all four benchmarks
  agents_v1.parquet       one row per (cell_id, qid, agent_id, round)
  cells_dedup_v1.parquet  cell-level re-aggregation: findings-parquet schema parity
                          (plug-in 15-bin ECEs via the package helpers) + audit columns
                          + pre-registered recomputed calibration (equal-mass / exact-atom
                          / debiased ECE, Brier reliability, Cox) + fixed-bin joint-qid
                          bootstrap SEs for the ECEs and deltas
  incident_items_scientifically_excluded_v1.parquet
                          hash-verified rows sealed by protocol-v4 reset incidents
  incident_agents_scientifically_excluded_v1.parquet
                          corresponding per-agent rows; never part of estimands
  ingest_manifest_v1.json provenance + data-integrity counters

The default is the authoritative homogeneous schema-5 dataset and is deliberately
fail-closed: all three production run IDs, their frozen manifests, and their exact
artifact-policy pins must be present.  Historical data is available only through the
explicit ``supplementary-legacy`` mode.  It is always labelled mixed-protocol and rows
without artifact-schema provenance can never populate exact-token columns.  Valid rows
from partial/active/retryable manifested cells are retained in item/agent tables with
their coverage state, but only semantically complete cells enter cell aggregates.

CLI:
  python analysis/nb_lib/ingest.py --out-dir analysis/cache
  python analysis/nb_lib/ingest.py --mode supplementary-legacy \
      --include-unmanifested --out-dir analysis/cache/supplementary_legacy
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import agents_scaling

_ANALYSIS = Path(__file__).resolve().parents[1]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from agents_scaling.efficiency.metrics import coordination_metrics  # noqa: E402
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.config import ExperimentCell  # noqa: E402
# Private package helpers, pinned here ONLY (analysis plan: an upstream rename must
# break loudly in one file). They define the parquet-parity plug-in ECEs.
from agents_scaling.experiment.analyze import (  # noqa: E402
    _censor_accounting,
    _per_agent_calibration,
    _system_calibration,
)
from agents_scaling.experiment import io as run_io  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    get_completion_status,
    read_canonical_results,
    reasoning_token_summary,
)
from agents_scaling.experiment.manifest import load_manifest, manifested_cell_dirs  # noqa: E402
from agents_scaling.experiment.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.result_schema import (  # noqa: E402
    SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)
from agents_scaling.serving.profiles import serving_profile_for_cell  # noqa: E402
from agents_scaling.models import get_model  # noqa: E402
from agents_scaling.serving.model_contracts import (  # noqa: E402
    ModelContractError,
    load_model_contracts,
)

from nb_lib import calib  # noqa: E402

SYSTEM_CONF_KEYS = [
    "vote_fraction", "mean_agreeing_logprob", "mean_agreeing_verbal", "mean_all_logprob",
    "final_producer_logprob", "final_producer_verbal", "mean_producer_logprob",
    "orchestrator_logprob", "orchestrator_verbal",
]
CFG_KEYS = ["benchmark", "model_size", "topology", "context_share_level",
            "prompt_complexity_level", "reasoning_level", "seed"]
EXPECTED_N = {"gpqa": 198, "mmlu_pro": 200, "truthfulqa": 200, "math": 200}
BOOT_B = 500
CHUNK_CELLS = 400

PRIMARY_MODE = "primary-schema5"
SUPPLEMENTARY_MODE = "supplementary-legacy"
PRIMARY_SCHEMA5_RUN_IDS = (
    "full_sweep_schema5_v1",
    "full_sweep_agent_counts_schema5_v1",
    "full_sweep_agent_count_7_schema5_v1",
)
LEGACY_RUN_IDS = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
)

DISCARDED_RESPONSE_INCIDENT_KIND = "discarded_server_response_protocol_v4"
DISCARDED_RESPONSE_INCIDENT_ROOT = (
    Path("incidents") / DISCARDED_RESPONSE_INCIDENT_KIND
)
INCIDENT_ITEMS_FILENAME = "incident_items_scientifically_excluded_v1.parquet"
INCIDENT_AGENTS_FILENAME = "incident_agents_scientifically_excluded_v1.parquet"
LEGACY_CONSOLIDATED_EXPECTED_ACTIVE_QIDS = 888_068
LEGACY_CONSOLIDATED_EXPECTED_SEALED_QIDS = 1_064
LEGACY_CONSOLIDATED_EXPECTED_TOTAL_QIDS = 889_132
LEGACY_PREEXISTING_INCIDENTS = 14
LEGACY_PREEXISTING_INCIDENT_QIDS = 355
LEGACY_NEWLY_SEALED_INCIDENTS = 8
LEGACY_ALL_INCIDENTS = 22
LEGACY_ALL_INCIDENT_QIDS = (
    LEGACY_CONSOLIDATED_EXPECTED_SEALED_QIDS
    + LEGACY_PREEXISTING_INCIDENT_QIDS
)
PRE_REPAIR_SNAPSHOT_COMPLETE = "SNAPSHOT_COMPLETE.json"
PRE_REPAIR_SNAPSHOT_INVENTORY = "SNAPSHOT_INVENTORY.sha256"
CACHE_MANIFEST_FILENAME = "ingest_manifest_v1.json"
CACHE_BUILD_MARKER_FILENAME = ".cache_generation_in_progress.json"
CACHE_BUILD_LOCK_FILENAME = ".cache_generation.lock"
CACHE_ARTIFACT_FILENAMES = (
    "items_v1.parquet",
    "agents_v1.parquet",
    "cells_dedup_v1.parquet",
    INCIDENT_ITEMS_FILENAME,
    INCIDENT_AGENTS_FILENAME,
)
SUPPLEMENTARY_PRESERVED_STATES = frozenset(
    {
        "complete",
        "partial",
        "active",
        "retryable",
    }
)


def _analysis_runtime_authority(mode: str) -> dict[str, Any] | None:
    """Validate the immutable implementation/environment used for primary ingest."""

    release_value = os.environ.get("ASYS_RELEASE_WORKTREE")
    if not release_value:
        if mode == PRIMARY_MODE:
            raise RuntimeError(
                "primary schema-5 ingest requires ASYS_RELEASE_WORKTREE and the "
                "immutable harness; run analysis/refresh.sh"
            )
        return None
    required = {
        "harness_prefix": os.environ.get("ASYS_HARNESS_ENVIRONMENT_PREFIX"),
        "harness_environment_sha256": os.environ.get(
            "ASYS_HARNESS_ENVIRONMENT_SHA256"
        ),
        "immutable_pins_sha256": os.environ.get("ASYS_IMMUTABLE_PINS_SHA256"),
        "model_contract_path": os.environ.get("ASYS_MODEL_CONTRACT"),
        "model_contract_sha256": os.environ.get("ASYS_MODEL_CONTRACT_SHA256"),
        "release_id": os.environ.get("ASYS_RELEASE_ID"),
        "git_commit": os.environ.get("ASYS_RELEASE_GIT_COMMIT"),
        "source_tree_sha256": os.environ.get("ASYS_SOURCE_TREE_SHA256"),
    }
    missing = [field for field, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "immutable analysis runtime is missing pins: " + ", ".join(missing)
        )
    raw_release = Path(release_value).expanduser()
    raw_harness = Path(str(required["harness_prefix"])).expanduser()
    raw_model = Path(str(required["model_contract_path"])).expanduser()
    if not all(path.is_absolute() for path in (raw_release, raw_harness, raw_model)):
        raise RuntimeError("immutable analysis runtime paths must be absolute")
    release = raw_release.resolve()
    harness = raw_harness.resolve()
    model_path = raw_model.resolve()
    if raw_release.is_symlink() or not release.is_dir():
        raise RuntimeError("immutable analysis release is missing or symlinked")
    if raw_harness.is_symlink() or not harness.is_dir():
        raise RuntimeError("immutable analysis harness is missing or symlinked")
    expected_model = release / "configs" / "model_contracts.v1.json"
    if model_path != expected_model or raw_model.is_symlink() or not raw_model.is_file():
        raise RuntimeError("analysis model contract is outside the frozen release")
    if Path(__file__).resolve().is_relative_to(release) is False:
        raise RuntimeError("primary ingest implementation is not from the frozen release")
    executable = (harness / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != executable or Path(sys.prefix).resolve() != harness:
        raise RuntimeError("analysis is not executing with the immutable harness")
    if sys.flags.isolated != 1:
        raise RuntimeError("immutable analysis harness must run with python -I")
    package_path = Path(agents_scaling.__file__).resolve()
    try:
        package_path.relative_to(harness)
    except ValueError as exc:
        raise RuntimeError("agents_scaling was imported outside the immutable harness") from exc
    sha_fields = (
        "harness_environment_sha256",
        "immutable_pins_sha256",
        "model_contract_sha256",
        "source_tree_sha256",
    )
    if any(
        not isinstance(required[field], str)
        or len(str(required[field])) != 64
        or any(character not in "0123456789abcdef" for character in str(required[field]))
        for field in sha_fields
    ):
        raise RuntimeError("immutable analysis SHA-256 pins are malformed")
    try:
        contracts = load_model_contracts(
            model_path,
            expected_sha256=str(required["model_contract_sha256"]),
        )
    except ModelContractError as exc:
        raise RuntimeError(f"analysis model contract failed closed: {exc}") from exc
    return {
        "release_worktree": str(release),
        "release_id": str(required["release_id"]),
        "git_commit": str(required["git_commit"]),
        "source_tree_sha256": str(required["source_tree_sha256"]),
        "harness_prefix": str(harness),
        "harness_environment_sha256": str(required["harness_environment_sha256"]),
        "immutable_pins_sha256": str(required["immutable_pins_sha256"]),
        "model_contract_path": str(model_path),
        "model_contract_sha256": contracts.sha256,
        "python": str(executable),
        "agents_scaling_module": str(package_path),
        "ingest_script": str(Path(__file__).resolve()),
        "isolated": True,
    }


class SupplementaryIntegrityError(RuntimeError):
    """Legacy evidence cannot be represented without silently losing provenance."""


@dataclass(frozen=True)
class VerifiedDiscardedIncident:
    """Cryptographically verified, canonical rows from one sealed reset incident."""

    run_id: str
    cell_id: str
    archive_path: Path
    incident_sha256: str
    results_sha256: str | None
    expected_qids: tuple[str, ...]
    records: tuple[dict[str, Any], ...]

    @property
    def valid_qids(self) -> tuple[str, ...]:
        return tuple(str(record["qid"]) for record in self.records)

    @property
    def missing_qids(self) -> tuple[str, ...]:
        valid = set(self.valid_qids)
        return tuple(qid for qid in self.expected_qids if qid not in valid)


@dataclass(frozen=True)
class PreRepairIncidentMembership:
    """Incident reset markers that already existed at the frozen recovery boundary."""

    snapshot_root: Path
    snapshot_id: str
    inventory_sha256: str
    cells: frozenset[tuple[str, str]]

    def is_preexisting(self, run_id: str, cell_id: str) -> bool:
        return (run_id, cell_id) in self.cells


def _parse_snapshot_inventory(payload: bytes) -> dict[str, str]:
    """Strictly parse the recovery snapshot's sorted SHA-256 inventory."""

    records: dict[str, str] = {}
    previous: str | None = None
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SupplementaryIntegrityError(
            f"pre-repair snapshot inventory is not UTF-8: {exc}"
        ) from exc
    for line_number, line in enumerate(lines, 1):
        digest, separator, logical = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not logical
            or logical.startswith("/")
            or "\n" in logical
            or "\r" in logical
            or any(part in {"", ".", ".."} for part in Path(logical).parts)
        ):
            raise SupplementaryIntegrityError(
                f"invalid pre-repair snapshot inventory line {line_number}"
            )
        if previous is not None and logical <= previous:
            raise SupplementaryIntegrityError(
                "pre-repair snapshot inventory is unsorted or contains duplicates"
            )
        records[logical] = digest
        previous = logical
    return records


def load_pre_repair_incident_membership(
    snapshot_root: Path,
    *,
    run_ids: Sequence[str] = LEGACY_RUN_IDS,
) -> PreRepairIncidentMembership:
    """Authenticate which response incidents predated the frozen baseline.

    The cleanup intentionally seals eight *new* incidents containing 1,064 rows.  Fourteen
    earlier reset archives contain another 355 rows, but those rows predate—and therefore
    are outside—the frozen 889,132-outcome baseline.  The sealed pre-repair snapshot is
    the non-temporal source of truth for this distinction: marker membership is checked
    against its cryptographic inventory, rather than inferred from mutable mtimes.
    """

    root = Path(snapshot_root)
    if root.is_symlink() or not root.is_dir():
        raise SupplementaryIntegrityError(
            f"missing or unsafe pre-repair snapshot: {root}"
        )
    complete_path = root / PRE_REPAIR_SNAPSHOT_COMPLETE
    inventory_path = root / PRE_REPAIR_SNAPSHOT_INVENTORY
    if (
        complete_path.is_symlink()
        or not complete_path.is_file()
        or inventory_path.is_symlink()
        or not inventory_path.is_file()
    ):
        raise SupplementaryIntegrityError(
            f"pre-repair snapshot is not sealed: {root}"
        )
    complete = _strict_json_object(
        complete_path.read_bytes(), label=str(complete_path)
    )
    inventory_bytes = inventory_path.read_bytes()
    inventory_sha = _sha256_bytes(inventory_bytes)
    if (
        complete.get("schema_version") != 1
        or complete.get("verified") is not True
        or complete.get("read_only") is not True
        or not isinstance(complete.get("snapshot_id"), str)
        or complete.get("snapshot_inventory_sha256") != inventory_sha
    ):
        raise SupplementaryIntegrityError(
            f"pre-repair snapshot completion contract drift: {complete_path}"
        )
    inventory = _parse_snapshot_inventory(inventory_bytes)
    allowed_runs = set(run_ids)
    cells: set[tuple[str, str]] = set()
    suffix = (
        "/incidents/"
        f"{DISCARDED_RESPONSE_INCIDENT_KIND}/"
    )
    marker_tail = "/reset_complete.json"
    for logical, expected_sha in inventory.items():
        if suffix not in logical or not logical.endswith(marker_tail):
            continue
        run_id, separator, remainder = logical.partition(suffix)
        cell_id = remainder[: -len(marker_tail)]
        if (
            not separator
            or run_id not in allowed_runs
            or not cell_id
            or Path(cell_id).name != cell_id
        ):
            continue
        marker_path = root / logical
        if (
            marker_path.is_symlink()
            or not marker_path.is_file()
            or _sha256_bytes(marker_path.read_bytes()) != expected_sha
        ):
            raise SupplementaryIntegrityError(
                f"pre-repair incident marker failed inventory verification: {logical}"
            )
        cells.add((run_id, cell_id))
    return PreRepairIncidentMembership(
        snapshot_root=root,
        snapshot_id=str(complete["snapshot_id"]),
        inventory_sha256=inventory_sha,
        cells=frozenset(cells),
    )


def resolve_ingest_scope(
    mode: str,
    requested_run_ids: list[str] | tuple[str, ...] | None,
    *,
    include_unmanifested: bool,
) -> tuple[str, ...]:
    """Return one canonical, auditable run set or fail before reading any artifacts."""

    if mode not in {PRIMARY_MODE, SUPPLEMENTARY_MODE}:
        raise ValueError(f"unsupported analysis mode {mode!r}")
    expected = PRIMARY_SCHEMA5_RUN_IDS if mode == PRIMARY_MODE else LEGACY_RUN_IDS
    observed = tuple(requested_run_ids) if requested_run_ids is not None else expected
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError(
            f"{mode} requires exactly these run IDs: {', '.join(expected)}; "
            f"observed: {', '.join(observed) if observed else '<none>'}"
        )
    if mode == PRIMARY_MODE and include_unmanifested:
        raise ValueError(
            "authoritative schema-5 ingestion cannot include unmanifested directories"
        )
    # Canonical order makes cache identity independent of command-line ordering.
    return expected


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file_and_parent(path: Path) -> None:
    """Durably publish a generated cache member on local or shared filesystems."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError:
        # Some shared filesystems reject fsync on a read descriptor after a completed
        # close.  Atomic same-directory replacement remains the publication boundary.
        pass
    finally:
        os.close(descriptor)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Decode one finite, duplicate-key-free JSON object.

    Incident hashes authenticate bytes, not their interpretation.  Strict decoding is
    therefore part of the scientific integrity boundary: two duplicate keys must not
    be interpreted differently by different JSON implementations.
    """

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token!r}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SupplementaryIntegrityError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SupplementaryIntegrityError(f"{label} must contain one JSON object")
    return decoded


def _safe_regular_archive_file(
    archive_root: Path,
    relative: str,
    *,
    label: str,
) -> Path:
    """Resolve a sealed-archive member without following a symlink or ``..``."""

    rel = Path(relative)
    if (
        not relative
        or rel.is_absolute()
        or any(part in {"", ".", ".."} for part in rel.parts)
    ):
        raise SupplementaryIntegrityError(f"unsafe {label} path: {relative!r}")
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise SupplementaryIntegrityError(
            f"unsafe discarded-response incident directory: {archive_root}"
        )
    current = archive_root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise SupplementaryIntegrityError(f"refusing symlink {label}: {current}")
    try:
        current.resolve().relative_to(archive_root.resolve())
    except ValueError as exc:
        raise SupplementaryIntegrityError(
            f"{label} escapes discarded-response archive: {relative!r}"
        ) from exc
    if not current.is_file():
        raise SupplementaryIntegrityError(f"missing/non-regular {label}: {current}")
    return current


def _verified_indexed_file(
    archive_root: Path,
    row: Mapping[str, Any],
    *,
    path_key: str,
    label: str,
) -> tuple[Path, bytes]:
    relative = row.get(path_key)
    expected_sha = row.get("sha256")
    expected_size = row.get("size")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise SupplementaryIntegrityError(f"invalid cryptographic index for {label}")
    path = _safe_regular_archive_file(
        archive_root, relative, label=label
    )
    payload = path.read_bytes()
    observed_sha = _sha256_bytes(payload)
    if len(payload) != expected_size or observed_sha != expected_sha:
        raise SupplementaryIntegrityError(
            f"cryptographic verification failed for {label}: {path}"
        )
    return path, payload


def _verify_incident_evidence_files(
    archive_root: Path,
    incident: Mapping[str, Any],
) -> None:
    """Verify every archived dispatcher log/batch referenced by an incident."""

    discovery = incident.get("discovery")
    events = discovery.get("dispatcher_events") if isinstance(discovery, dict) else None
    if not isinstance(events, list):
        raise SupplementaryIntegrityError(
            f"invalid dispatcher evidence index in {archive_root}"
        )
    seen: set[str] = set()
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise SupplementaryIntegrityError(
                f"invalid dispatcher event {event_index} in {archive_root}"
            )
        for field in ("log", "batch_manifest"):
            row = event.get(field)
            if not isinstance(row, dict):
                raise SupplementaryIntegrityError(
                    f"invalid archived {field} index in {archive_root}"
                )
            path, _ = _verified_indexed_file(
                archive_root,
                row,
                path_key="archived_path",
                label=f"dispatcher {field}",
            )
            relative = str(path.relative_to(archive_root))
            if relative in seen:
                raise SupplementaryIntegrityError(
                    f"duplicate archived evidence path in {archive_root}: {relative}"
                )
            seen.add(relative)


def verify_discarded_response_incident(
    *,
    run_id: str,
    run_root: Path,
    snapshot,
    catalog: VerifiedQuestionCatalog,
    archive_path: Path,
) -> VerifiedDiscardedIncident:
    """Authenticate one sealed protocol-v4 reset and return all canonical rows.

    The reset marker authenticates ``incident.json``; the incident in turn authenticates
    every archived active artifact and dispatcher evidence file.  The archived JSONL is
    then validated against the same frozen manifest and benchmark contracts as active
    legacy rows.  Any malformed, invalid, unexpected, duplicated, or omitted JSONL row
    fails the cache build instead of being silently dropped.
    """

    if archive_path.parent != run_root / DISCARDED_RESPONSE_INCIDENT_ROOT:
        raise SupplementaryIntegrityError(
            f"incident archive is outside the expected run namespace: {archive_path}"
        )
    if archive_path.is_symlink() or not archive_path.is_dir():
        raise SupplementaryIntegrityError(f"unsafe incident archive: {archive_path}")
    incident_path = _safe_regular_archive_file(
        archive_path, "incident.json", label="incident record"
    )
    marker_path = _safe_regular_archive_file(
        archive_path, "reset_complete.json", label="incident reset marker"
    )
    incident_bytes = incident_path.read_bytes()
    incident_sha = _sha256_bytes(incident_bytes)
    incident = _strict_json_object(incident_bytes, label=str(incident_path))
    marker = _strict_json_object(marker_path.read_bytes(), label=str(marker_path))
    expected_marker = {
        "reset_marker_schema_version": 1,
        "incident_sha256": incident_sha,
        "target_artifact_schema_version": 5,
        "active_cell_state_removed": True,
    }
    if marker != expected_marker:
        raise SupplementaryIntegrityError(
            f"reset marker does not authenticate incident record: {archive_path}"
        )
    if (
        incident.get("incident_schema_version") != 1
        or incident.get("incident_type") != DISCARDED_RESPONSE_INCIDENT_KIND
        or incident.get("source_protocol_error_type")
        != "ServerResponseProtocolError"
        or incident.get("run_id") != run_id
        or incident.get("target_artifact_schema_version") != 5
    ):
        raise SupplementaryIntegrityError(
            f"discarded-response incident identity drift: {archive_path}"
        )
    manifest = incident.get("manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != snapshot.sha256
        or manifest.get("cell_count") != len(snapshot.cells)
    ):
        raise SupplementaryIntegrityError(
            f"incident manifest contract drift: {archive_path}"
        )
    cell_identity = incident.get("cell")
    if not isinstance(cell_identity, dict):
        raise SupplementaryIntegrityError(f"invalid incident cell identity: {archive_path}")
    cell_id = cell_identity.get("cell_id")
    manifest_index = cell_identity.get("manifest_index")
    if (
        not isinstance(cell_id, str)
        or archive_path.name != cell_id
        or not isinstance(manifest_index, int)
        or isinstance(manifest_index, bool)
        or not 0 <= manifest_index < len(snapshot.cells)
    ):
        raise SupplementaryIntegrityError(f"incident cell/index drift: {archive_path}")
    cell = snapshot.cells[manifest_index]
    expected_identity = {
        "cell_id": cell.cell_id,
        "manifest_index": manifest_index,
        "config_hash": cell.config_hash(),
        "config": cell.to_dict(),
    }
    if cell_identity != expected_identity:
        raise SupplementaryIntegrityError(
            f"incident configuration does not match frozen manifest: {archive_path}"
        )

    artifact_rows = incident.get("active_artifacts")
    if not isinstance(artifact_rows, list):
        raise SupplementaryIntegrityError(
            f"incident active-artifact index is invalid: {archive_path}"
        )
    indexed_sources: set[str] = set()
    indexed_archive_paths: set[str] = set()
    results_path: Path | None = None
    results_sha: str | None = None
    for row_index, row in enumerate(artifact_rows):
        if not isinstance(row, dict):
            raise SupplementaryIntegrityError(
                f"invalid active artifact row {row_index}: {archive_path}"
            )
        source = row.get("source_relative_path")
        source_path = Path(source) if isinstance(source, str) else None
        source_allowed = bool(
            isinstance(source, str)
            and (
                source in {"results.jsonl", "meta.json", "failure.json"}
                or (
                    source_path is not None
                    and len(source_path.parts) == 2
                    and source_path.parts[0] == ".qid_checkpoints"
                    and source_path.name not in {"", ".", ".."}
                )
            )
        )
        if not source_allowed or source in indexed_sources:
            raise SupplementaryIntegrityError(
                f"duplicate/invalid active artifact source: {archive_path}"
            )
        path, _ = _verified_indexed_file(
            archive_path,
            row,
            path_key="archived_path",
            label=f"active artifact {source}",
        )
        relative = str(path.relative_to(archive_path))
        if relative in indexed_archive_paths:
            raise SupplementaryIntegrityError(
                f"duplicate active artifact archive path: {archive_path}/{relative}"
            )
        indexed_sources.add(source)
        indexed_archive_paths.add(relative)
        if source == "results.jsonl":
            results_path = path
            results_sha = str(row["sha256"])
    expected_reset_contract = {
        "scope": "entire_active_cell_state",
        "active_files": ["results.jsonl", "meta.json", "failure.json"],
        "checkpoint_directory": ".qid_checkpoints",
        "rerun_from_empty_cell_under_schema": 5,
    }
    if incident.get("reset_contract") != expected_reset_contract:
        raise SupplementaryIntegrityError(
            f"discarded-response reset contract drift: {archive_path}"
        )
    _verify_incident_evidence_files(archive_path, incident)
    questions = catalog.questions_for(cell)
    expected_qids = tuple(question.qid for question in questions)
    if results_path is None:
        # Some affected cells failed before their first QID was journaled.  Their
        # incident remains part of the verified sealed history but contributes zero
        # scientifically-excluded outcomes.
        return VerifiedDiscardedIncident(
            run_id=run_id,
            cell_id=cell_id,
            archive_path=archive_path,
            incident_sha256=incident_sha,
            results_sha256=None,
            expected_qids=expected_qids,
            records=(),
        )
    assert results_sha is not None
    parsed = read_canonical_results(
        cell,
        results_path.parent,
        expected_qids=expected_qids,
        expected_questions=questions,
        verified_benchmark_contracts=catalog.frozen,
        verified_manifest=catalog.snapshot,
    )
    if parsed.has_corruption or parsed.raw_rows != len(parsed.records):
        details = "; ".join(parsed.validation_errors[:3]) or (
            f"raw_rows={parsed.raw_rows}, canonical_rows={len(parsed.records)}, "
            f"duplicates={len(parsed.duplicate_qids)}, "
            f"unexpected={len(parsed.unexpected_qids)}"
        )
        raise SupplementaryIntegrityError(
            f"sealed results are not entirely canonical in {archive_path}: {details}"
        )
    return VerifiedDiscardedIncident(
        run_id=run_id,
        cell_id=cell_id,
        archive_path=archive_path,
        incident_sha256=incident_sha,
        results_sha256=results_sha,
        expected_qids=expected_qids,
        records=parsed.records,
    )


def load_verified_discarded_response_incidents(
    *,
    run_id: str,
    run_root: Path,
    snapshot,
    catalog: VerifiedQuestionCatalog,
) -> tuple[VerifiedDiscardedIncident, ...]:
    """Discover all sealed response incidents for one run in stable cell-id order."""

    root = run_root / DISCARDED_RESPONSE_INCIDENT_ROOT
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise SupplementaryIntegrityError(f"unsafe incident root: {root}")
    archives: list[VerifiedDiscardedIncident] = []
    seen_cells: set[str] = set()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_dir():
            raise SupplementaryIntegrityError(f"unsafe incident-root entry: {path}")
        verified = verify_discarded_response_incident(
            run_id=run_id,
            run_root=run_root,
            snapshot=snapshot,
            catalog=catalog,
            archive_path=path,
        )
        if verified.cell_id in seen_cells:
            raise SupplementaryIntegrityError(
                f"duplicate sealed incident for {run_id}/{verified.cell_id}"
            )
        seen_cells.add(verified.cell_id)
        archives.append(verified)
    return tuple(archives)


def _qid_contract_annotation(
    *,
    completion_state: str,
    expected_qids: Sequence[str] | None,
    valid_qids: Sequence[str],
    expected_count: int | None = None,
    semantically_complete: bool,
    manifested: bool,
) -> dict[str, Any]:
    """Columns repeated on every item/agent so partial rows cannot be mistaken as full."""

    valid = tuple(str(qid) for qid in valid_qids)
    if expected_qids is None:
        expected = None
        missing = None
        resolved_expected_count = expected_count
        missing_count = (
            None
            if expected_count is None
            else max(int(expected_count) - len(valid), 0)
        )
    else:
        expected = tuple(str(qid) for qid in expected_qids)
        valid_set = set(valid)
        missing = tuple(qid for qid in expected if qid not in valid_set)
        resolved_expected_count = len(expected)
        missing_count = len(missing)
    return {
        "cell_completion_state": completion_state,
        "cell_semantically_complete": bool(semantically_complete),
        "cell_manifested": bool(manifested),
        "cell_qid_contract_verified": expected is not None,
        "cell_expected_qid_count": resolved_expected_count,
        "cell_valid_qid_count": len(valid),
        "cell_missing_qid_count": missing_count,
        "cell_expected_qids_json": (
            None if expected is None else json.dumps(expected, separators=(",", ":"))
        ),
        "cell_valid_qids_json": json.dumps(valid, separators=(",", ":")),
        "cell_missing_qids_json": (
            None if missing is None else json.dumps(missing, separators=(",", ":"))
        ),
    }


def _analysis_provenance_annotation(
    *,
    mode: str,
    row: Mapping[str, Any],
    scientifically_excluded: bool,
    exclusion_reason: str | None = None,
) -> dict[str, Any]:
    row_exact = _row_reasoning_tokens_exact(dict(row))
    supplementary = mode == SUPPLEMENTARY_MODE
    if supplementary and not row_exact:
        token_provenance = "legacy-nonexact-word-count"
    elif supplementary:
        token_provenance = "legacy-row-exact-nonauthoritative"
    else:
        token_provenance = "schema5-exact" if row_exact else "schema5-no-reasoning-span"
    return {
        "analysis_mode": mode,
        "mixed_protocol": supplementary,
        "protocol_provenance": (
            "legacy-mixed-protocol" if supplementary else "schema5-homogeneous"
        ),
        "artifact_schema_provenance": str(row.get("schema_version", "legacy")),
        "token_provenance": token_provenance,
        # A row-local native span can be inspected in the supplement, but it is never
        # promoted to a homogeneous/authoritative cross-run exact-token claim.
        "authoritative_exact_token_claim": bool(row_exact and not supplementary),
        "scientifically_excluded": bool(scientifically_excluded),
        "scientific_exclusion_reason": exclusion_reason,
    }


def _manifested_ingest_decision(
    mode: str,
    completion_state: str,
) -> tuple[bool, bool]:
    """Return ``(preserve_rows, include_complete_cell_aggregate)``.

    This deliberately makes row preservation and aggregate eligibility separate
    decisions.  In particular, a partial legacy cell contributes already-observed QIDs
    to the supplementary item/agent evidence tables but never masquerades as a complete
    experimental unit.
    """

    if mode == PRIMARY_MODE:
        complete = completion_state == "complete"
        return complete, complete
    if mode != SUPPLEMENTARY_MODE:
        raise ValueError(f"unsupported analysis mode {mode!r}")
    if completion_state in {"corrupt", "permanent"}:
        raise SupplementaryIntegrityError(
            "legacy consolidation requires zero unresolved "
            f"{completion_state} cells"
        )
    preserve = completion_state in SUPPLEMENTARY_PRESERVED_STATES
    return preserve, completion_state == "complete"


def legacy_consolidated_acceptance(
    *,
    active_validated_qids: int,
    sealed_scientifically_excluded_qids: int,
    include_unmanifested: bool,
    newly_sealed_incidents: int | None = None,
    preexisting_historical_incidents: int | None = None,
    preexisting_historical_qids: int | None = None,
) -> dict[str, Any]:
    """Machine-readable acceptance record for the frozen legacy consolidation."""

    observed_total = (
        int(active_validated_qids) + int(sealed_scientifically_excluded_qids)
    )
    applicable = not include_unmanifested
    baseline_passed = bool(
        active_validated_qids == LEGACY_CONSOLIDATED_EXPECTED_ACTIVE_QIDS
        and sealed_scientifically_excluded_qids
        == LEGACY_CONSOLIDATED_EXPECTED_SEALED_QIDS
        and observed_total == LEGACY_CONSOLIDATED_EXPECTED_TOTAL_QIDS
    )
    incident_partition_supplied = all(
        value is not None
        for value in (
            newly_sealed_incidents,
            preexisting_historical_incidents,
            preexisting_historical_qids,
        )
    )
    incident_partition_passed = bool(
        not incident_partition_supplied
        or (
            newly_sealed_incidents == LEGACY_NEWLY_SEALED_INCIDENTS
            and preexisting_historical_incidents == LEGACY_PREEXISTING_INCIDENTS
            and preexisting_historical_qids
            == LEGACY_PREEXISTING_INCIDENT_QIDS
        )
    )
    passed = bool(applicable and baseline_passed and incident_partition_passed)
    return {
        "applicable": applicable,
        "expected_active_artifact_validated_qids": (
            LEGACY_CONSOLIDATED_EXPECTED_ACTIVE_QIDS
        ),
        "expected_sealed_scientifically_excluded_qids": (
            LEGACY_CONSOLIDATED_EXPECTED_SEALED_QIDS
        ),
        "expected_total_preserved_qids": LEGACY_CONSOLIDATED_EXPECTED_TOTAL_QIDS,
        "observed_active_artifact_validated_qids": int(active_validated_qids),
        "observed_sealed_scientifically_excluded_qids": int(
            sealed_scientifically_excluded_qids
        ),
        "observed_total_preserved_qids": observed_total,
        "baseline_counts_passed": baseline_passed,
        "incident_partition_supplied": incident_partition_supplied,
        "incident_partition_passed": incident_partition_passed,
        "expected_newly_sealed_incidents": LEGACY_NEWLY_SEALED_INCIDENTS,
        "expected_preexisting_historical_incidents": LEGACY_PREEXISTING_INCIDENTS,
        "expected_preexisting_historical_qids": (
            LEGACY_PREEXISTING_INCIDENT_QIDS
        ),
        "observed_newly_sealed_incidents": newly_sealed_incidents,
        "observed_preexisting_historical_incidents": (
            preexisting_historical_incidents
        ),
        "observed_preexisting_historical_qids": preexisting_historical_qids,
        "passed": passed,
    }


def _run_contract_record(
    *,
    run_id: str,
    run_root: Path,
    snapshot,
    catalog: VerifiedQuestionCatalog,
    mode: str,
) -> dict:
    """Validate and summarize the immutable inputs that define one cache run."""

    try:
        policy = load_artifact_policy(
            run_root,
            required=mode == PRIMARY_MODE,
        )
    except ArtifactPolicyError as exc:
        raise RuntimeError(f"cannot trust analysis run {run_id}: {exc}") from exc
    if mode == PRIMARY_MODE:
        assert policy is not None
        if policy.accepted_manifest_sha256 != snapshot.sha256:
            raise RuntimeError(
                f"schema-5 policy/manifest mismatch for {run_id}: "
                f"{policy.accepted_manifest_sha256} != {snapshot.sha256}"
            )
        if policy.accepted_benchmark_contracts_sha256 != catalog.sidecar_sha256:
            raise RuntimeError(
                f"schema-5 policy/benchmark mismatch for {run_id}: "
                f"{policy.accepted_benchmark_contracts_sha256} != "
                f"{catalog.sidecar_sha256}"
            )
    elif policy is not None:
        raise RuntimeError(
            f"supplementary legacy run unexpectedly has an authoritative policy: {run_id}"
        )
    return {
        "run_id": run_id,
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "artifact_policy_sha256": policy.file_sha256 if policy is not None else None,
        "artifact_policy_id": policy.policy_id if policy is not None else None,
        "model_contract_sha256": (
            policy.accepted_model_contract_sha256 if policy is not None else None
        ),
        "release_id": policy.release.release_id if policy is not None else None,
        "git_commit": policy.release.git_commit if policy is not None else None,
        "source_tree_sha256": (
            policy.release.source_tree_sha256 if policy is not None else None
        ),
        "harness_environment_sha256": (
            policy.environment.harness_sha256 if policy is not None else None
        ),
        "serving_environment_sha256": (
            policy.environment.serving_sha256 if policy is not None else None
        ),
        "required_artifact_schema_version": (
            policy.required_artifact_schema_version if policy is not None else None
        ),
        "authoritative": bool(policy is not None and policy.authoritative),
    }


def _f(x) -> float:
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _actual_n_agents(cfg: dict, rows: list[dict] | None = None, row: dict | None = None) -> int:
    """Executed agent count; fixes legacy single-agent configs that stored n_agents=3."""
    candidates = []
    if row is not None:
        candidates.append(row)
    if rows is not None:
        candidates.extend(rows)
    for r in candidates:
        eff = r.get("efficiency_raw") or {}
        if eff.get("n_agents") is not None:
            return int(eff["n_agents"])
    if cfg.get("topology") == "single_agent":
        return 1
    return int(cfg.get("n_agents", 3))


def _row_reasoning_tokens_exact(row: dict) -> bool:
    """Whether every per-agent reasoning count is an exact native token-ID span."""
    agents = row.get("per_agent") or []
    return bool(agents) and row.get("schema_version") in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS and all(
        (agent or {}).get("reasoning_token_source")
        in {"vllm_native_token_ids", "none"}
        for agent in agents
    )


def _reasoning_sources(row: dict) -> list[str]:
    return sorted(
        {
            str((agent or {}).get("reasoning_token_source") or "legacy_word_count")
            for agent in (row.get("per_agent") or [])
        }
    )


def read_cell(cell_path: Path) -> tuple[dict | None, list[dict], int, int]:
    """(meta, deduped rows, n_bad_lines, n_dupes). meta=None -> skip cell."""
    meta_p, res_p = cell_path / "meta.json", cell_path / "results.jsonl"
    if not meta_p.exists() or not res_p.exists():
        return None, [], 0, 0
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return None, [], 0, 0
    rows, bad, dupes = [], 0, 0
    seen: set[str] = set()
    for line in res_p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        qid = r.get("qid")
        if qid in seen:
            dupes += 1
            continue
        seen.add(qid)
        rows.append(r)
    return meta, rows, bad, dupes


def item_record(run_id: str, cell_id: str, cfg: dict, r: dict) -> dict:
    sc = r.get("self_consistency") or {}
    sc_aux = sc.get("auxiliary_efficiency_raw") or {}
    eff = r.get("efficiency_raw") or {}
    sys_conf = r.get("system_conf") or {}
    key = r.get("answer_key")
    pa_confs, pa_verbals = [], []
    for a in r.get("per_agent") or []:
        lp = a.get("option_logprobs") or {}
        if a.get("answer") is not None and lp:
            pa_confs.append(_f(lp.get(a["answer"], 0.0)))
        if a.get("verbalized_conf") is not None:
            pa_verbals.append(_f(a["verbalized_conf"]))
    samples = sc.get("samples") or []
    first_sample_answer = None
    first_sample_observed = False
    if samples:
        first = samples[0]
        if isinstance(first, dict):
            if first.get("termination_status") == TERMINATION_COMPLETED:
                output = first.get("agent_output") or {}
                first_sample_answer = output.get("answer")
                first_sample_observed = True
        else:  # explicit legacy schema
            first_sample_answer = first
            first_sample_observed = True
    reasoning_exact = _row_reasoning_tokens_exact(r)
    observed_reasoning = _f(eff.get("total_reasoning_tokens"))
    rec = {
        "run_id": run_id,
        "cell_id": cell_id,
        **{k: cfg.get(k) for k in CFG_KEYS},
        "n_agents": _actual_n_agents(cfg, row=r),
        "qid": r.get("qid"),
        "termination_status": r.get("termination_status", TERMINATION_COMPLETED),
        "length_censored": (
            r.get("termination_status") == TERMINATION_LENGTH_CENSORED
        ),
        "protocol_censored": (
            r.get("termination_status") == TERMINATION_PROTOCOL_CENSORED
        ),
        "correct": bool(r.get("correct")),
        "final_answer": (
            None
            if r.get("final_answer") is None
            else str(r.get("final_answer"))
        ),
        "answer_key": str(key),
        **{k: _f(sys_conf.get(k)) for k in SYSTEM_CONF_KEYS},
        "sc_conf": _f(sc.get("self_consistency_conf")),
        "sc_sem_entropy_conf": _f(sc.get("semantic_entropy_conf")),
        "sc_sem_entropy": _f(sc.get("semantic_entropy")),
        "sc_n_samples": len(samples),
        "sc_n_completed_samples": _f(sc.get("completed_sample_count")),
        "sc_n_length_censored_samples": _f(
            sc.get("length_censored_sample_count")
        ),
        "sc_n_protocol_censored_samples": _f(
            sc.get("protocol_censored_sample_count")
        ),
        "sc_length_censor_rate": (
            _f(sc.get("length_censored_sample_count")) / len(samples)
            if samples else np.nan
        ),
        "sc_protocol_censor_rate": (
            _f(sc.get("protocol_censored_sample_count")) / len(samples)
            if samples else np.nan
        ),
        "sc_first_correct": (
            float(first_sample_answer == key) if first_sample_observed else np.nan
        ),
        "sc_aux_prompt_tokens": _f(sc_aux.get("total_prompt_tokens")),
        "sc_aux_completion_tokens": _f(sc_aux.get("total_completion_tokens")),
        "sc_aux_reasoning_tokens": _f(sc_aux.get("total_reasoning_tokens")),
        "sc_aux_wall_ms": _f(sc_aux.get("wall_ms")),
        "pa_mean_conf": float(np.mean(pa_confs)) if pa_confs else np.nan,
        "pa_mean_verbal": float(np.mean(pa_verbals)) if pa_verbals else np.nan,
        "n_agent_entries": len(r.get("per_agent") or []),
        "n_turns": _f(eff.get("n_turns")),
        "n_messages": _f(eff.get("n_messages")),
        "total_prompt_tokens": _f(eff.get("total_prompt_tokens")),
        "total_completion_tokens": _f(eff.get("total_completion_tokens")),
        # The primary token column is exact-only.  Historical rows stored whitespace
        # counts under this name; retain those in a clearly non-token-ID column.
        "total_reasoning_tokens": observed_reasoning if reasoning_exact else np.nan,
        "total_reasoning_tokens_legacy_or_mixed": (
            np.nan if reasoning_exact else observed_reasoning
        ),
        "reasoning_tokens_exact": reasoning_exact,
        "reasoning_token_sources_json": json.dumps(_reasoning_sources(r)),
        "wall_ms": _f(eff.get("wall_ms")),
    }
    return rec


def agent_records(run_id: str, cell_id: str, cfg: dict, r: dict, full_logprobs: bool) -> list[dict]:
    key = r.get("answer_key")
    out = []
    row_exact = _row_reasoning_tokens_exact(r)
    for a in r.get("per_agent") or []:
        lp = a.get("option_logprobs") or {}
        ans = a.get("answer")
        rec = {
            "run_id": run_id,
            "cell_id": cell_id,
            "benchmark": cfg.get("benchmark"), "model_size": cfg.get("model_size"),
            "topology": cfg.get("topology"), "reasoning_level": cfg.get("reasoning_level"),
            "n_agents": _actual_n_agents(cfg, row=r),
            "seed": cfg.get("seed"),
            "qid": r.get("qid"),
            "agent_id": str(a.get("agent_id")),
            "round": int(a.get("round") or 0),
            "answer": str(ans),
            "correct": bool(ans == key) if ans is not None else False,
            "conf_chosen": _f(lp.get(ans)) if (ans is not None and lp) else np.nan,
            "prob_answer_key": _f(lp.get(key)) if lp else np.nan,
            "verbalized_conf": _f(a.get("verbalized_conf")),
            "prompt_tokens": _f(a.get("prompt_tokens")),
            "completion_tokens": _f(a.get("completion_tokens")),
            "reasoning_tokens": (
                _f(a.get("reasoning_tokens")) if row_exact else np.nan
            ),
            "reasoning_tokens_legacy_nonexact": (
                np.nan if row_exact else _f(a.get("reasoning_tokens"))
            ),
            "reasoning_tokens_exact": row_exact,
            "reasoning_token_source": str(
                a.get("reasoning_token_source") or "legacy_word_count"
            ),
            "cot_len": len(a.get("cot_text") or ""),
            "reasoning_len": len(a.get("reasoning_text") or ""),
        }
        if full_logprobs:
            rec["option_logprobs_json"] = json.dumps(lp) if lp else None
        out.append(rec)
    return out


def _boot_ece_ses(rows: list[dict], rng: np.random.Generator) -> dict:
    """Joint-qid bootstrap SEs for the primary ECEs and deltas (fixed-bin approximation).

    Bin/atom assignments are computed once on the observed sample; replicates only
    re-weight items (multinomial over qids). This keeps B=500 x ~3.4k cells vectorized;
    treating bin edges as fixed is a standard approximation adequate for SEs.
    """
    censor_accounting = _censor_accounting(rows)
    n = censor_accounting["n_questions"]
    if n == 0:
        return {
            "se_pa_ece": np.nan,
            "se_vote_ece": np.nan,
            "se_fp_ece": np.nan,
            "se_delta_vote": np.nan,
            "se_delta_fp": np.nan,
        }
    correct = np.array([bool(r.get("correct")) for r in rows], float)
    sys_conf = {k: np.array([_f((r.get("system_conf") or {}).get(k)) for r in rows])
                for k in ("vote_fraction", "final_producer_logprob")}
    # Same degenerate-signal guard as cell_record: math logs fp as identically 0.0 —
    # without this the SE would be a real-looking number attached to a NaN estimate.
    if np.nanmax(sys_conf["final_producer_logprob"], initial=0.0) <= 0.0:
        sys_conf["final_producer_logprob"] = np.full(n, np.nan)
    pa_conf, pa_corr, pa_item = [], [], []
    for i, r in enumerate(rows):
        key = r.get("answer_key")
        for a in r.get("per_agent") or []:
            lp = a.get("option_logprobs") or {}
            ans = a.get("answer")
            if ans is None or not lp:
                continue
            pa_conf.append(_f(lp.get(ans, 0.0)))
            pa_corr.append(float(ans == key))
            pa_item.append(i)
    pa_conf = np.asarray(pa_conf, float)
    pa_corr = np.asarray(pa_corr, float)
    pa_item = np.asarray(pa_item, int)

    def prep(conf, corr, discrete):
        ok = np.isfinite(conf) & np.isfinite(corr)
        if ok.sum() < 10:
            return None
        c, y = conf[ok], corr[ok]
        if discrete:
            _, g = np.unique(np.round(c, 6), return_inverse=True)
        else:
            g = calib.bin_index(c, calib.equal_mass_edges(c, 10))
        return c, y, g, np.flatnonzero(ok)

    sys_prep = {k: prep(v, correct, k in calib.DISCRETE_SIGNALS)
                for k, v in sys_conf.items()}
    pa_ok = np.isfinite(pa_conf)
    pa_prep = None
    if pa_ok.sum() >= 10:
        c = pa_conf[pa_ok]
        pa_prep = (c, pa_corr[pa_ok],
                   calib.bin_index(c, calib.equal_mass_edges(c, 10)),
                   pa_item[pa_ok])

    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=BOOT_B).astype(float)

    def ece_reps(prep) -> np.ndarray | None:
        """Vectorized fixed-group weighted ECE across all replicates at once."""
        if prep is None:
            return None
        c, y, g, idx = prep
        n_g = int(g.max()) + 1
        M = np.zeros((len(c), n_g))
        M[np.arange(len(c)), g] = 1.0
        W = counts[:, idx]                      # B x entries
        wts = W @ M                             # B x G
        conf_g = W @ (M * c[:, None])
        acc_g = W @ (M * y[:, None])
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(conf_g / wts - acc_g / wts)
        gap[wts == 0] = 0.0
        return np.sum(wts / wts.sum(axis=1, keepdims=True) * gap, axis=1)

    reps = {"pa": ece_reps(pa_prep),
            "vote": ece_reps(sys_prep["vote_fraction"]),
            "fp": ece_reps(sys_prep["final_producer_logprob"])}

    def sd(x) -> float:
        return float(np.std(x)) if x is not None else np.nan

    def sd_diff(a, b) -> float:
        return float(np.std(a - b)) if (a is not None and b is not None) else np.nan

    return {"se_pa_ece": sd(reps["pa"]), "se_vote_ece": sd(reps["vote"]),
            "se_fp_ece": sd(reps["fp"]),
            "se_delta_vote": sd_diff(reps["vote"], reps["pa"]),
            "se_delta_fp": sd_diff(reps["fp"], reps["pa"])}


def cell_record(run_id: str, meta: dict, rows: list[dict], n_bad: int, n_dupes: int, n_raw: int,
                rng: np.random.Generator) -> dict:
    cfg = meta["config"]
    censor_accounting = _censor_accounting(rows)
    n = censor_accounting["n_questions"]
    correct = np.array([bool(r.get("correct")) for r in rows], float)
    acc = float(correct.mean())
    completed_rows = [
        row
        for row in rows
        if row.get("termination_status", TERMINATION_COMPLETED)
        == TERMINATION_COMPLETED
    ]
    n_censored = censor_accounting["n_length_censored_questions"]
    n_protocol_censored = censor_accounting["n_protocol_censored_questions"]
    n_any_censored = n_censored + n_protocol_censored
    eff_raw = [r.get("efficiency_raw") or {} for r in completed_rows]
    mean_turns = (
        float(np.mean([_f(e.get("n_turns")) for e in eff_raw]))
        if eff_raw else np.nan
    )
    mean_msgs = (
        float(np.mean([_f(e.get("n_messages")) for e in eff_raw]))
        if eff_raw else np.nan
    )
    mean_tokens = (
        float(np.mean([_f(e.get("total_prompt_tokens"))
                       + _f(e.get("total_completion_tokens"))
                       for e in eff_raw]))
        if eff_raw else np.nan
    )
    observed_mean_rt = (
        float(np.nanmean([
            _f(e.get("total_reasoning_tokens", 0)) or 0.0 for e in eff_raw
        ]))
        if eff_raw else np.nan
    )
    reasoning_summary = reasoning_token_summary(rows)
    completed_reasoning_summary = reasoning_token_summary(completed_rows)
    reasoning_sources = sorted(
        {source for row in rows for source in _reasoning_sources(row)}
    )

    # Parquet-parity plug-in calibration (package definitions, 15-bin equal-width).
    pa_cal = _per_agent_calibration(completed_rows)
    vote_cal = _system_calibration(completed_rows, "vote_fraction")
    fp_cal = _system_calibration(completed_rows, "final_producer_logprob")
    conditional_pa_ece = float(pa_cal["ece"]) if pa_cal else np.nan
    conditional_sys_ece_vote = float(vote_cal["ece"]) if vote_cal else np.nan
    conditional_sys_ece_fp = float(fp_cal["ece"]) if fp_cal else np.nan
    calibration_defined = n_any_censored == 0
    pa_ece = conditional_pa_ece if calibration_defined else np.nan
    sys_ece_vote = conditional_sys_ece_vote if calibration_defined else np.nan
    sys_ece_fp = conditional_sys_ece_fp if calibration_defined else np.nan

    # Pre-registered recomputed calibration.
    pa_conf, pa_corr = [], []
    for r in completed_rows:
        key = r.get("answer_key")
        for a in r.get("per_agent") or []:
            lp = a.get("option_logprobs") or {}
            if a.get("answer") is not None and lp:
                pa_conf.append(_f(lp.get(a["answer"], 0.0)))
                pa_corr.append(float(a["answer"] == key))
    pa_c = calib.cell_calibration(np.array(pa_conf), np.array(pa_corr), "per_agent")
    completed_correct = np.array(
        [bool(r.get("correct")) for r in completed_rows], float
    )
    vote_conf = np.array([_f((r.get("system_conf") or {}).get("vote_fraction"))
                          for r in completed_rows])
    fp_conf = np.array([_f((r.get("system_conf") or {}).get("final_producer_logprob"))
                        for r in completed_rows])
    # Degenerate-signal guard: math logs final_producer_logprob as identically 0.0 (no
    # option-logprob path for free-form answers), which would make fp "ECE" = accuracy.
    # A confidence signal with no positive mass carries no information — treat as absent.
    if np.nanmax(fp_conf, initial=0.0) <= 0.0:
        fp_conf = np.full_like(fp_conf, np.nan)
    vote_c = calib.cell_calibration(vote_conf, completed_correct, "vote_fraction")
    fp_c = calib.cell_calibration(fp_conf, completed_correct, "final_producer_logprob")

    def first_auxiliary_correct(row: dict) -> float:
        samples = (row.get("self_consistency") or {}).get("samples") or []
        if not samples:
            return np.nan
        sample = samples[0]
        if isinstance(sample, dict):
            if sample.get("termination_status") != TERMINATION_COMPLETED:
                return np.nan
            answer = (sample.get("agent_output") or {}).get("answer")
        else:  # explicit legacy schema
            answer = sample
        return float(answer == row.get("answer_key"))

    first = np.array([first_auxiliary_correct(row) for row in completed_rows])
    acc_first = float(np.nanmean(first)) if np.isfinite(first).any() else np.nan

    primary_reasoning_defined = bool(
        completed_reasoning_summary.all_exact
        and censor_accounting["top_level_uncensored"]
        and len(completed_rows) == n
    )

    rec = {
        "run_id": run_id,
        "cell_id": meta["cell_id"],
        "model_size": cfg["model_size"],
        "param_count": get_model(cfg["model_size"]).param_count,
        "topology": cfg["topology"],
        "n_agents": _actual_n_agents(cfg, rows=rows),
        "context_share_level": cfg["context_share_level"],
        "prompt_complexity_level": cfg["prompt_complexity_level"],
        "reasoning_level": cfg.get("reasoning_level", "off"),
        "benchmark": cfg["benchmark"],
        "seed": cfg["seed"],
        **{
            key: (np.nan if value is None else value)
            for key, value in censor_accounting.items()
        },
        "accuracy": acc,
        "error_rate": 1.0 - acc,
        "mean_reasoning_tokens": (
            _f(completed_reasoning_summary.mean_reasoning_tokens)
            if primary_reasoning_defined
            else np.nan
        ),
        "mean_reasoning_tokens_completed_only": _f(
            completed_reasoning_summary.mean_reasoning_tokens
        ),
        "mean_reasoning_tokens_exact_subset": _f(
            reasoning_summary.mean_reasoning_tokens_exact
        ),
        "mean_reasoning_tokens_legacy_or_mixed": (
            observed_mean_rt
            if (
                censor_accounting["top_level_uncensored"]
                and not completed_reasoning_summary.all_exact
            )
            else np.nan
        ),
        "mean_reasoning_tokens_legacy_or_mixed_completed_only": (
            observed_mean_rt
            if not completed_reasoning_summary.all_exact
            else np.nan
        ),
        "mean_reasoning_word_count_legacy": (
            reasoning_summary.mean_reasoning_word_count_legacy
        ),
        "reasoning_tokens_exact": primary_reasoning_defined,
        "reasoning_tokens_completed_only_exact": (
            completed_reasoning_summary.all_exact
        ),
        "reasoning_metric_primary_defined": primary_reasoning_defined,
        "exact_reasoning_question_count": reasoning_summary.exact_question_count,
        "nonexact_reasoning_question_count": reasoning_summary.nonexact_question_count,
        "reasoning_token_sources_json": json.dumps(reasoning_sources),
        "mean_total_tokens": mean_tokens,
        "mean_turns": mean_turns,
        "mean_messages": mean_msgs,
        "pa_ece": pa_ece, "sys_ece_vote": sys_ece_vote, "sys_ece_fp": sys_ece_fp,
        "delta_vote": sys_ece_vote - pa_ece, "delta_fp": sys_ece_fp - pa_ece,
        "calibration_primary_defined": calibration_defined,
        "pa_ece_completed_only": conditional_pa_ece,
        "sys_ece_vote_completed_only": conditional_sys_ece_vote,
        "sys_ece_fp_completed_only": conditional_sys_ece_fp,
        "is_mas": cfg["topology"] != "single_agent",
        # audit
        "n_raw_rows": n_raw, "n_dupes_dropped": n_dupes, "n_bad_lines": n_bad,
        "started_at": _f(meta.get("started_at")),
        "finished_at": _f(meta.get("finished_at")),
        "prompt_token_count": _f(meta.get("prompt_token_count")),
        "acc_first_sample": acc_first,
        # recomputed (pre-registered primary estimators; *_ece_prim = equal-mass for
        # continuous signals, exact-atom for vote_fraction — see calib.py)
        **{
            f"pa_{'ece_prim' if k == 'ece' else k}": (
                v if calibration_defined else np.nan
            )
            for k, v in pa_c.items()
        },
        **{
            f"vote_{'ece_prim' if k == 'ece' else k}": (
                v if calibration_defined else np.nan
            )
            for k, v in vote_c.items()
        },
        **{
            f"fp_{'ece_prim' if k == 'ece' else k}": (
                v if calibration_defined else np.nan
            )
            for k, v in fp_c.items()
        },
    }
    # Primary deltas mix estimators on purpose: each signal uses ITS pre-registered
    # estimator (vote = exact-atom, pa/fp = equal-mass), per plan §3.
    rec["delta_vote_prim"] = (
        _f(vote_c.get("ece")) - _f(pa_c.get("ece"))
        if calibration_defined
        else np.nan
    )
    rec["delta_fp_prim"] = (
        _f(fp_c.get("ece")) - _f(pa_c.get("ece"))
        if calibration_defined
        else np.nan
    )
    boot = _boot_ece_ses(completed_rows, rng)
    rec.update(
        boot
        if calibration_defined
        else {name: np.nan for name in boot}
    )
    return rec


class ChunkWriter:
    """Schema-stable incremental parquet writer (chunked; atomic final rename)."""

    def __init__(self, out_path: Path, columns: list[str] | None = None):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        self.columns = columns
        self.writer: pq.ParquetWriter | None = None

    def write(self, records: list[dict]):
        if not records:
            return
        df = pd.DataFrame.from_records(records)
        if self.columns is None:
            self.columns = list(df.columns)
        df = df.reindex(columns=self.columns)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.tmp_path, table.schema)
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        else:
            # Never leave an older, populated cache in place when a clean production
            # sweep currently has zero complete cells.
            pq.write_table(
                pa.Table.from_pandas(
                    pd.DataFrame(columns=self.columns or []), preserve_index=False
                ),
                self.tmp_path,
            )
        _fsync_file_and_parent(self.tmp_path)
        os.replace(self.tmp_path, self.out_path)
        _fsync_file_and_parent(self.out_path)


class CacheGeneration:
    """Fail-closed, marker-last publication of a complete cache generation.

    Parquet members are individually atomically replaced.  Before any replacement, the
    prior manifest is withdrawn and an in-progress marker is durably published under a
    nonblocking directory lock.  Therefore a killed refresh can leave recoverable cache
    members, but never a manifest that authenticates a mixture of generations.  The new
    manifest binds every member's hash and is published last; the in-progress marker is
    removed only after that durable write.
    """

    def __init__(self, out_dir: Path, *, mode: str, run_ids: Sequence[str]):
        self.out_dir = Path(out_dir)
        self.mode = mode
        self.run_ids = tuple(run_ids)
        self.lock_descriptor: int | None = None
        self.started_at = time.time()

    @property
    def marker_path(self) -> Path:
        return self.out_dir / CACHE_BUILD_MARKER_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / CACHE_MANIFEST_FILENAME

    def begin(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.out_dir / CACHE_BUILD_LOCK_FILENAME
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise RuntimeError(
                f"another cache refresh holds {lock_path}"
            ) from exc
        self.lock_descriptor = descriptor
        run_io.write_json(
            self.marker_path,
            {
                "cache_build_schema_version": 1,
                "status": "in_progress",
                "analysis_mode": self.mode,
                "run_ids": list(self.run_ids),
                "pid": os.getpid(),
                "started_at": self.started_at,
            },
        )
        if self.manifest_path.exists():
            previous = self.out_dir / "ingest_manifest_v1.previous.json"
            run_io.move_file(self.manifest_path, previous)

    def publish(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if self.lock_descriptor is None:
            raise RuntimeError("cache generation was not begun")
        artifact_contracts: dict[str, dict[str, Any]] = {}
        for filename in CACHE_ARTIFACT_FILENAMES:
            path = self.out_dir / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"cache generation lacks {filename}")
            artifact_contracts[filename] = {
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        published = dict(manifest)
        published["cache_artifacts"] = artifact_contracts
        published["cache_generation_id"] = _sha256_json(
            {
                "analysis_mode": published["analysis_mode"],
                "run_ids": published["run_ids"],
                "cache_contract_sha256": published["cache_contract_sha256"],
                "cache_artifacts": artifact_contracts,
            }
        )
        # This is the only current-manifest publication.  The marker remains present
        # while it is written, so a cooperating reader can never accept an in-flight
        # generation.
        run_io.write_json(self.manifest_path, published)
        run_io.remove_file(self.marker_path)
        self.release()
        return published

    def release(self) -> None:
        descriptor = self.lock_descriptor
        self.lock_descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __del__(self) -> None:
        self.release()


def add_efficiency(cells: pd.DataFrame) -> pd.DataFrame:
    """Kim efficiency vs matched SAS baseline, post-dedupe.

    Prefers the SAME-PROMPT single-agent baseline; falls back to the package's
    (last-wins) pick when none exists. The package keys only on (size, benchmark, seed,
    reasoning) — 209 of those keys have SAS cells at 2-4 prompt levels, so package
    Ec/Ae/Opct silently mix prompt levels; this version does not.
    """
    key = ["model_size", "benchmark", "seed", "reasoning_level"]
    sas_rows = cells[cells["topology"] == "single_agent"]
    sas = {tuple(r[k] for k in key): r for _, r in sas_rows.iterrows()}
    sas_prompt = {tuple(r[k] for k in key) + (r["prompt_complexity_level"],): r
                  for _, r in sas_rows.iterrows()}
    ec, ae, opct = [], [], []
    for _, r in cells.iterrows():
        base = sas_prompt.get(tuple(r[k] for k in key) + (r["prompt_complexity_level"],),
                              sas.get(tuple(r[k] for k in key)))
        if (
            base is None
            or r.get("n_length_censored_questions", 0) > 0
            or r.get("n_protocol_censored_questions", 0) > 0
            or base.get("n_length_censored_questions", 0) > 0
            or base.get("n_protocol_censored_questions", 0) > 0
        ):
            ec.append(np.nan), ae.append(np.nan), opct.append(np.nan)
            continue
        eff = coordination_metrics(
            success_rate=r["accuracy"], error_rate=r["error_rate"],
            mean_turns=r["mean_turns"], mean_messages=r["mean_messages"],
            sas_turns=base["mean_turns"], sas_error_rate=base["error_rate"],
            mean_total_tokens=r["mean_total_tokens"]).to_dict()
        ec.append(eff.get("coordination_efficiency"))
        ae.append(eff.get("error_amplification"))
        opct.append(eff.get("overhead_pct"))
    cells["Ec"], cells["Ae"], cells["Opct"] = ec, ae, opct
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=(PRIMARY_MODE, SUPPLEMENTARY_MODE),
        default=PRIMARY_MODE,
        help="authoritative schema-5 primary cache or opt-in mixed legacy supplement",
    )
    ap.add_argument(
        "--run-id",
        nargs="+",
        default=None,
        help=(
            "optional explicit run set; it must exactly match the selected mode's "
            "three frozen run IDs"
        ),
    )
    ap.add_argument("--out-dir", default=str(_ANALYSIS / "cache"))
    ap.add_argument("--full-logprobs", action="store_true",
                    help="keep per-agent option_logprobs as a JSON column (bigger file)")
    ap.add_argument(
        "--include-unmanifested",
        action="store_true",
        help="include legacy/pre-trim cell directories absent from cells.json (supplementary only)",
    )
    ap.add_argument(
        "--pre-repair-snapshot-root",
        type=Path,
        default=None,
        help=(
            "sealed schema5-v1 pre-repair snapshot used to distinguish the 14 "
            "preexisting incidents from the 8 frozen-baseline resets; supplementary "
            "mode defaults to $ASYS_RESULTS_ROOT/recovery/schema5-v1/pre_repair"
        ),
    )
    args = ap.parse_args()

    analysis_runtime = _analysis_runtime_authority(args.mode)
    model_contract_path = (
        None
        if analysis_runtime is None
        else analysis_runtime["model_contract_path"]
    )
    out_dir = Path(args.out_dir)
    try:
        run_ids = list(
            resolve_ingest_scope(
                args.mode,
                args.run_id,
                include_unmanifested=args.include_unmanifested,
            )
        )
    except ValueError as exc:
        ap.error(str(exc))
    mixed_protocol = args.mode == SUPPLEMENTARY_MODE
    pre_repair_membership: PreRepairIncidentMembership | None = None
    if mixed_protocol:
        snapshot_root = args.pre_repair_snapshot_root or (
            run_io.results_root() / "recovery" / "schema5-v1" / "pre_repair"
        )
        pre_repair_membership = load_pre_repair_incident_membership(
            snapshot_root,
            run_ids=run_ids,
        )
        if len(pre_repair_membership.cells) != LEGACY_PREEXISTING_INCIDENTS:
            raise SupplementaryIntegrityError(
                "pre-repair incident membership drift: expected "
                f"{LEGACY_PREEXISTING_INCIDENTS}, observed "
                f"{len(pre_repair_membership.cells)}"
            )
    cell_dirs: list[tuple[str, Path, ExperimentCell | None]] = []
    manifest_counts: dict[str, int] = {}
    stale_counts: dict[str, int] = {}
    missing_dir_counts: dict[str, int] = {}
    question_catalogs: dict[str, VerifiedQuestionCatalog] = {}
    run_contracts: dict[str, dict] = {}
    run_roots: dict[str, Path] = {}
    run_snapshots: dict[str, Any] = {}
    for run_id in run_ids:
        run_root = run_io.results_root() / run_id
        paths, stale_ids, missing_ids = manifested_cell_dirs(
            run_root, include_unmanifested=args.include_unmanifested
        )
        snapshot = load_manifest(run_root)
        run_roots[run_id] = run_root
        run_snapshots[run_id] = snapshot
        question_catalogs[run_id] = VerifiedQuestionCatalog(
            run_root,
            snapshot=snapshot,
        )
        run_contracts[run_id] = _run_contract_record(
            run_id=run_id,
            run_root=run_root,
            snapshot=snapshot,
            catalog=question_catalogs[run_id],
            mode=args.mode,
        )
        if args.mode == PRIMARY_MODE:
            assert analysis_runtime is not None
            runtime_checks = {
                "model_contract_sha256": analysis_runtime[
                    "model_contract_sha256"
                ],
                "release_id": analysis_runtime["release_id"],
                "git_commit": analysis_runtime["git_commit"],
                "source_tree_sha256": analysis_runtime["source_tree_sha256"],
                "harness_environment_sha256": analysis_runtime[
                    "harness_environment_sha256"
                ],
            }
            drift = {
                field: (run_contracts[run_id].get(field), expected)
                for field, expected in runtime_checks.items()
                if run_contracts[run_id].get(field) != expected
            }
            if drift:
                raise RuntimeError(
                    f"analysis runtime differs from {run_id} policy: {drift}"
                )
        cell_by_id = {cell.cell_id: cell for cell in snapshot.cells}
        manifest_counts[run_id] = len(paths) + len(missing_ids) - (
            len(stale_ids) if args.include_unmanifested else 0
        )
        stale_counts[run_id] = len(stale_ids)
        missing_dir_counts[run_id] = len(missing_ids)
        cell_dirs.extend((run_id, p, cell_by_id.get(p.name)) for p in paths)
    rng = np.random.default_rng(0)

    cache_generation = CacheGeneration(
        out_dir,
        mode=args.mode,
        run_ids=run_ids,
    )
    cache_generation.begin()

    items_w = ChunkWriter(out_dir / "items_v1.parquet")
    agents_w = ChunkWriter(out_dir / "agents_v1.parquet")
    incident_items_w = ChunkWriter(out_dir / INCIDENT_ITEMS_FILENAME)
    incident_agents_w = ChunkWriter(out_dir / INCIDENT_AGENTS_FILENAME)
    cell_recs: list[dict] = []
    qid_sets: dict[tuple, frozenset] = {}
    qid_mismatch: list[str] = []
    n_bad_lines = n_bad_cells = n_dupes_total = n_cells_with_dupes = 0
    n_unexpected_n: list[str] = []
    completion_states: dict[str, int] = {
        "missing": sum(missing_dir_counts.values())
    }
    artifact_schema_counts: dict[str, int] = {}
    validated_qids_by_completion_state: dict[str, int] = {}
    n_manifested_active_artifact_qids = 0
    n_unmanifested_qids = 0
    items_buf: list[dict] = []
    agents_buf: list[dict] = []

    t0 = time.time()
    for i, (run_id, cell_path, manifest_cell) in enumerate(cell_dirs):
        aggregate_eligible = False
        if manifest_cell is not None:
            profile = serving_profile_for_cell(manifest_cell)
            questions = question_catalogs[run_id].questions_for(manifest_cell)
            expected_qids = tuple(question.qid for question in questions)
            status = get_completion_status(
                manifest_cell,
                cell_path,
                expected_qids=expected_qids,
                expected_questions=questions,
                verified_benchmark_contracts=question_catalogs[run_id].frozen,
                verified_manifest=question_catalogs[run_id].snapshot,
                check_active=mixed_protocol,
                serving_profile=profile.name,
                model_contract_path=model_contract_path,
            )
            state = status.status.value
            completion_states[state] = completion_states.get(state, 0) + 1
            try:
                preserve_rows, aggregate_eligible = _manifested_ingest_decision(
                    args.mode, state
                )
            except SupplementaryIntegrityError as exc:
                raise SupplementaryIntegrityError(
                    f"{exc}; observed {run_id}/{manifest_cell.cell_id}"
                ) from exc
            if not preserve_rows:
                continue
            parsed = read_canonical_results(
                manifest_cell,
                cell_path,
                expected_qids=expected_qids,
                expected_questions=questions,
                verified_benchmark_contracts=question_catalogs[run_id].frozen,
                verified_manifest=question_catalogs[run_id].snapshot,
            )
            rows = list(parsed.records)
            if status.valid_count != len(rows):
                raise SupplementaryIntegrityError(
                    "completion/canonical row count drift for "
                    f"{run_id}/{manifest_cell.cell_id}: "
                    f"{status.valid_count} != {len(rows)}"
                )
            if not rows:
                continue
            meta = (
                json.loads((cell_path / "meta.json").read_text())
                if status.is_complete
                else None
            )
            cfg = manifest_cell.to_dict()
            cell_id = manifest_cell.cell_id
            valid_qids = tuple(str(row["qid"]) for row in rows)
            qid_annotation = _qid_contract_annotation(
                completion_state=state,
                expected_qids=expected_qids,
                valid_qids=valid_qids,
                semantically_complete=status.is_complete,
                manifested=True,
            )
            bad = parsed.malformed_lines + parsed.invalid_rows
            dupes = len(parsed.duplicate_qids)
            if aggregate_eligible != status.is_complete:
                raise SupplementaryIntegrityError(
                    f"completion decision drift for {run_id}/{manifest_cell.cell_id}"
                )
            if mixed_protocol:
                n_manifested_active_artifact_qids += len(rows)
            validated_qids_by_completion_state[state] = (
                validated_qids_by_completion_state.get(state, 0) + len(rows)
            )
        else:
            # Explicit supplementary mode for legacy/pre-trim directories.
            meta, rows, bad, dupes = read_cell(cell_path)
            if meta is None or not rows:
                n_bad_lines += bad
                n_bad_cells += bool(meta is None and (cell_path / "meta.json").exists())
                continue
            cfg = meta["config"]
            cell_id = meta["cell_id"]
            valid_qids = tuple(
                str(row.get("qid")) for row in rows if row.get("qid") is not None
            )
            qid_annotation = _qid_contract_annotation(
                completion_state="unmanifested_legacy_unverified",
                expected_qids=None,
                expected_count=(
                    int(cfg["n_questions"])
                    if isinstance(cfg.get("n_questions"), int)
                    else None
                ),
                valid_qids=valid_qids,
                semantically_complete=False,
                manifested=False,
            )
            n_unmanifested_qids += len(rows)
        n_bad_lines += bad
        if not rows:
            n_bad_cells += bool(meta is None and (cell_path / "meta.json").exists())
            continue
        n_dupes_total += dupes
        n_cells_with_dupes += dupes > 0

        exp = EXPECTED_N.get(cfg["benchmark"])
        if aggregate_eligible and exp and not (exp * 0.5 <= len(rows) <= exp + 12):
            n_unexpected_n.append(f"{cell_id}:{len(rows)}")

        # qid-set consistency within (benchmark, seed) — matched-pair precondition.
        if aggregate_eligible:
            qkey = (cfg["benchmark"], cfg["seed"])
            qset = frozenset(r.get("qid") for r in rows)
            ref = qid_sets.setdefault(qkey, qset)
            if qset != ref:
                qid_mismatch.append(cell_id)

        for r in rows:
            schema_label = str(r.get("schema_version", "legacy"))
            artifact_schema_counts[schema_label] = (
                artifact_schema_counts.get(schema_label, 0) + 1
            )
            item = item_record(run_id, cell_id, cfg, r)
            item.update(qid_annotation)
            item.update(
                _analysis_provenance_annotation(
                    mode=args.mode,
                    row=r,
                    scientifically_excluded=manifest_cell is None,
                    exclusion_reason=(
                        "unmanifested_legacy_directory"
                        if manifest_cell is None
                        else None
                    ),
                )
            )
            items_buf.append(item)
            agent_rows = agent_records(
                run_id, cell_id, cfg, r, args.full_logprobs
            )
            for agent_row in agent_rows:
                agent_row.update(qid_annotation)
                agent_row.update(
                    _analysis_provenance_annotation(
                        mode=args.mode,
                        row=r,
                        scientifically_excluded=manifest_cell is None,
                        exclusion_reason=(
                            "unmanifested_legacy_directory"
                            if manifest_cell is None
                            else None
                        ),
                    )
                )
            agents_buf.extend(agent_rows)
        if aggregate_eligible:
            assert meta is not None
            cell_rec = cell_record(
                run_id, meta, rows, bad, dupes, len(rows) + dupes, rng
            )
            cell_rec.update(qid_annotation)
            cell_rec.update(
                analysis_mode=args.mode,
                mixed_protocol=mixed_protocol,
                protocol_provenance=(
                    "legacy-mixed-protocol"
                    if mixed_protocol
                    else "schema5-homogeneous"
                ),
                authoritative_exact_token_claim=(
                    args.mode == PRIMARY_MODE
                    and bool(cell_rec.get("reasoning_tokens_exact"))
                ),
                scientifically_excluded=False,
                scientific_exclusion_reason=None,
            )
            cell_recs.append(cell_rec)

        if (i + 1) % CHUNK_CELLS == 0:
            items_w.write(items_buf)
            agents_w.write(agents_buf)
            items_buf, agents_buf = [], []
            print(f"[ingest] {i + 1}/{len(cell_dirs)} dirs "
                  f"({len(cell_recs)} complete cells, {time.time() - t0:.0f}s)",
                  flush=True)

    incident_count = 0
    n_newly_sealed_incidents = 0
    n_preexisting_historical_incidents = 0
    n_newly_sealed_incident_qids = 0
    n_preexisting_historical_qids = 0
    n_all_sealed_incident_agent_rows = 0
    n_newly_sealed_incident_agent_rows = 0
    n_preexisting_historical_agent_rows = 0
    incident_artifact_schema_counts: dict[str, int] = {}
    incident_contracts: list[dict[str, Any]] = []
    incident_items_buf: list[dict] = []
    incident_agents_buf: list[dict] = []
    if args.mode == SUPPLEMENTARY_MODE:
        for run_id in run_ids:
            snapshot = run_snapshots[run_id]
            cell_by_id = {cell.cell_id: cell for cell in snapshot.cells}
            incidents = load_verified_discarded_response_incidents(
                run_id=run_id,
                run_root=run_roots[run_id],
                snapshot=snapshot,
                catalog=question_catalogs[run_id],
            )
            for incident in incidents:
                incident_count += 1
                assert pre_repair_membership is not None
                preexisting_history = pre_repair_membership.is_preexisting(
                    run_id, incident.cell_id
                )
                frozen_baseline_member = not preexisting_history
                if preexisting_history:
                    n_preexisting_historical_incidents += 1
                else:
                    n_newly_sealed_incidents += 1
                cell = cell_by_id[incident.cell_id]
                cfg = cell.to_dict()
                qid_annotation = _qid_contract_annotation(
                    completion_state="sealed_discarded_response_protocol_v4",
                    expected_qids=incident.expected_qids,
                    valid_qids=incident.valid_qids,
                    semantically_complete=False,
                    manifested=True,
                )
                incident_columns = {
                    "incident_type": DISCARDED_RESPONSE_INCIDENT_KIND,
                    "incident_sha256": incident.incident_sha256,
                    "incident_results_sha256": incident.results_sha256,
                    "incident_preexisting_at_pre_repair_snapshot": (
                        preexisting_history
                    ),
                    "incident_frozen_baseline_member": frozen_baseline_member,
                    "pre_repair_snapshot_id": pre_repair_membership.snapshot_id,
                    "incident_archive_relative_path": str(
                        incident.archive_path.relative_to(run_roots[run_id])
                    ),
                }
                for row in incident.records:
                    schema_label = str(row.get("schema_version", "legacy"))
                    incident_artifact_schema_counts[schema_label] = (
                        incident_artifact_schema_counts.get(schema_label, 0) + 1
                    )
                    item = item_record(run_id, incident.cell_id, cfg, row)
                    item.update(qid_annotation)
                    item.update(
                        _analysis_provenance_annotation(
                            mode=args.mode,
                            row=row,
                            scientifically_excluded=True,
                            exclusion_reason=DISCARDED_RESPONSE_INCIDENT_KIND,
                        )
                    )
                    item.update(incident_columns)
                    incident_items_buf.append(item)
                    agents = agent_records(
                        run_id,
                        incident.cell_id,
                        cfg,
                        row,
                        args.full_logprobs,
                    )
                    for agent in agents:
                        agent.update(qid_annotation)
                        agent.update(
                            _analysis_provenance_annotation(
                                mode=args.mode,
                                row=row,
                                scientifically_excluded=True,
                                exclusion_reason=DISCARDED_RESPONSE_INCIDENT_KIND,
                            )
                        )
                        agent.update(incident_columns)
                    incident_agents_buf.extend(agents)
                    n_all_sealed_incident_agent_rows += len(agents)
                    if frozen_baseline_member:
                        n_newly_sealed_incident_agent_rows += len(agents)
                    else:
                        n_preexisting_historical_agent_rows += len(agents)
                if frozen_baseline_member:
                    n_newly_sealed_incident_qids += len(incident.records)
                else:
                    n_preexisting_historical_qids += len(incident.records)
                incident_contracts.append(
                    {
                        "run_id": run_id,
                        "cell_id": incident.cell_id,
                        "incident_sha256": incident.incident_sha256,
                        "results_sha256": incident.results_sha256,
                        "validated_qids": len(incident.records),
                        "preexisting_at_pre_repair_snapshot": (
                            preexisting_history
                        ),
                        "frozen_baseline_member": frozen_baseline_member,
                    }
                )
                if len(incident_items_buf) >= 1_000:
                    incident_items_w.write(incident_items_buf)
                    incident_agents_w.write(incident_agents_buf)
                    incident_items_buf, incident_agents_buf = [], []

    observed_preserved_total = (
        n_manifested_active_artifact_qids + n_newly_sealed_incident_qids
    )
    consolidated_acceptance = legacy_consolidated_acceptance(
        active_validated_qids=n_manifested_active_artifact_qids,
        sealed_scientifically_excluded_qids=n_newly_sealed_incident_qids,
        include_unmanifested=(
            args.include_unmanifested or args.mode != SUPPLEMENTARY_MODE
        ),
        newly_sealed_incidents=(
            n_newly_sealed_incidents if mixed_protocol else None
        ),
        preexisting_historical_incidents=(
            n_preexisting_historical_incidents if mixed_protocol else None
        ),
        preexisting_historical_qids=(
            n_preexisting_historical_qids if mixed_protocol else None
        ),
    )
    if mixed_protocol and not (
        consolidated_acceptance["baseline_counts_passed"]
        and consolidated_acceptance["incident_partition_passed"]
        and incident_count == LEGACY_ALL_INCIDENTS
        and n_newly_sealed_incident_qids + n_preexisting_historical_qids
        == LEGACY_ALL_INCIDENT_QIDS
    ):
        # No ChunkWriter has published its temporary file yet.  A failed acceptance
        # therefore leaves the previous cache withdrawn and cannot masquerade as a
        # successfully consolidated supplement.
        raise SupplementaryIntegrityError(
            "legacy frozen-baseline preservation failed: "
            f"active={n_manifested_active_artifact_qids}, "
            f"newly_sealed={n_newly_sealed_incident_qids}, "
            f"historical={n_preexisting_historical_qids}, "
            f"incidents={incident_count}, acceptance={consolidated_acceptance}"
        )

    items_w.write(items_buf)
    agents_w.write(agents_buf)
    incident_items_w.write(incident_items_buf)
    incident_agents_w.write(incident_agents_buf)
    items_w.close()
    agents_w.close()
    incident_items_w.close()
    incident_agents_w.close()

    cells = pd.DataFrame.from_records(cell_recs)
    if not cells.empty:
        cells = add_efficiency(cells)
    tmp = out_dir / "cells_dedup_v1.parquet.tmp"
    cells.to_parquet(tmp)
    _fsync_file_and_parent(tmp)
    os.replace(tmp, out_dir / "cells_dedup_v1.parquet")
    _fsync_file_and_parent(out_dir / "cells_dedup_v1.parquet")

    manifest = {
        "cache_schema_version": 3,
        "analysis_mode": args.mode,
        "mixed_protocol": mixed_protocol,
        "exact_token_policy": (
            "schema5-required"
            if args.mode == PRIMARY_MODE
            else "row-provenance-only; schema-less rows excluded"
        ),
        "run_ids": run_ids,
        "run_id": run_ids[0] if len(run_ids) == 1 else None,
        "manifest_scoped": not args.include_unmanifested,
        "manifest_cells_by_run": manifest_counts,
        "benchmark_contracts_sha256_by_run": {
            run_id: catalog.sidecar_sha256
            for run_id, catalog in sorted(question_catalogs.items())
        },
        "run_contracts": {
            run_id: run_contracts[run_id] for run_id in sorted(run_contracts)
        },
        "analysis_runtime": analysis_runtime,
        "stale_dirs_by_run": stale_counts,
        "missing_dirs_by_run": missing_dir_counts,
        "completion_states": completion_states,
        "validated_qids_by_completion_state": {
            key: validated_qids_by_completion_state[key]
            for key in sorted(validated_qids_by_completion_state)
        },
        "artifact_schema_counts": {
            key: artifact_schema_counts[key] for key in sorted(artifact_schema_counts)
        },
        "incident_artifact_schema_counts": {
            key: incident_artifact_schema_counts[key]
            for key in sorted(incident_artifact_schema_counts)
        },
        "supplementary_preservation": {
            "active_artifact_validated_qids": n_manifested_active_artifact_qids,
            # Only incidents absent from the pre-repair snapshot belong to the frozen
            # 889,132-row baseline.  All incident evidence is still retained below.
            "sealed_scientifically_excluded_qids": (
                n_newly_sealed_incident_qids
            ),
            "total_preserved_qids": observed_preserved_total,
            "unmanifested_opt_in_qids_not_in_acceptance": n_unmanifested_qids,
            "sealed_incident_count": incident_count,
            "newly_sealed_incident_count": n_newly_sealed_incidents,
            "preexisting_historical_incident_count": (
                n_preexisting_historical_incidents
            ),
            "preexisting_historical_incident_qids": (
                n_preexisting_historical_qids
            ),
            "all_sealed_incident_qids": (
                n_newly_sealed_incident_qids
                + n_preexisting_historical_qids
            ),
            "all_sealed_incident_agent_rows": (
                n_all_sealed_incident_agent_rows
            ),
            "newly_sealed_incident_agent_rows": (
                n_newly_sealed_incident_agent_rows
            ),
            "preexisting_historical_incident_agent_rows": (
                n_preexisting_historical_agent_rows
            ),
            "sealed_incident_items_file": INCIDENT_ITEMS_FILENAME,
            "sealed_incident_agents_file": INCIDENT_AGENTS_FILENAME,
            "pre_repair_snapshot": (
                None
                if pre_repair_membership is None
                else {
                    "path": str(pre_repair_membership.snapshot_root),
                    "snapshot_id": pre_repair_membership.snapshot_id,
                    "snapshot_inventory_sha256": (
                        pre_repair_membership.inventory_sha256
                    ),
                    "preexisting_incident_cells": len(
                        pre_repair_membership.cells
                    ),
                }
            ),
            "incident_contracts": incident_contracts,
            "legacy_consolidated_acceptance": consolidated_acceptance,
        },
        "timestamp": time.time(),
        "n_cell_dirs": len(cell_dirs),
        # Only semantically complete manifested cells enter this aggregate table.
        "n_cells_complete": len(cells),
        "n_bad_lines": n_bad_lines,
        "n_bad_cells": n_bad_cells,
        "n_dupes_dropped": n_dupes_total,
        "n_cells_with_dupes": n_cells_with_dupes,
        "n_unexpected_n": len(n_unexpected_n),
        "unexpected_n_cells": n_unexpected_n[:50],
        "qid_mismatch_cells": qid_mismatch[:50],
        "boot_B": BOOT_B,
        "full_logprobs": bool(args.full_logprobs),
    }
    manifest["cache_contract_sha256"] = _sha256_json(
        {
            "analysis_mode": manifest["analysis_mode"],
            "mixed_protocol": manifest["mixed_protocol"],
            "run_ids": manifest["run_ids"],
            "run_contracts": manifest["run_contracts"],
            "analysis_runtime": manifest["analysis_runtime"],
            "manifest_scoped": manifest["manifest_scoped"],
            "incident_contracts": incident_contracts,
            "active_artifact_validated_qids": n_manifested_active_artifact_qids,
            "sealed_scientifically_excluded_qids": (
                n_newly_sealed_incident_qids
            ),
            "preexisting_historical_incident_qids": (
                n_preexisting_historical_qids
            ),
        }
    )
    cache_generation.publish(manifest)

    print(f"[ingest] done in {time.time() - t0:.0f}s — {len(cells)} cells, "
          f"{n_dupes_total} duplicate rows dropped across {n_cells_with_dupes} cells, "
          f"{n_bad_lines} malformed lines skipped.")
    if qid_mismatch:
        print(f"[ingest] WARNING: qid-set mismatch within (benchmark, seed) for "
              f"{len(qid_mismatch)} cells — matched-pair tests must align on shared qids.")


if __name__ == "__main__":
    main()
