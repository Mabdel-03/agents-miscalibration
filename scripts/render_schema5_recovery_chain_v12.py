#!/usr/bin/env python3
"""Render, verify, and transactionally submit the schema-5 v1.2-r7 recovery DAG.

The recovery jobs are deliberately generated outside the Git checkout.  A successful
``render --apply`` publishes immutable generation-specific sbatch files first and the
chain manifest last.  ``submit --apply`` persists one intent before every ``sbatch``
boundary and publishes a separate submission receipt only after every exact
configured dependency has been accepted.  Production stages use ``afterok``;
the marker-last failure sentinel uses ``afterany`` across every production stage.

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

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from scripts import publish_schema5_durable_git_release as durable_git  # noqa: E402
from scripts import provision_schema5_conda_toolchain as conda_toolchain  # noqa: E402
from scripts import seal_recovery_evidence as recovery_evidence  # noqa: E402
from scripts import seal_schema5_r4_toolchain_failure as r4_toolchain_failure  # noqa: E402
from scripts import seal_schema5_r5_prelaunch_failure as r5_prelaunch_failure  # noqa: E402
from scripts import seal_schema5_r6_prelaunch_failure as r6_prelaunch_failure  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    trusted_generation_catalog_errors,
)
from agents_scaling.serving.generation_catalog import (  # noqa: E402
    GenerationCatalogError,
    TrustedGenerationCatalog,
    validate_trusted_generation_catalog,
)
from agents_scaling.serving import protected_capacity  # noqa: E402


RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r7"
CHAIN_NAMESPACE = "schema5-v1.2-r7"
CHAIN_MANIFEST_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_2_R7.json"
SUBMISSION_JOURNAL_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_2_R7.submission.json"
SUBMISSION_RECEIPT_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_2_R7_SUBMISSION.json"
DEPENDENCY_POLICY_CHECKS_ROOT_NAME = "dependency_policy_checks"
ROOT_RELEASE_INTENT_NAME = "RECOVERY_CHAIN_ROOT_RELEASE_INTENT.json"
ROOT_RELEASE_COMPLETE_NAME = "RECOVERY_CHAIN_ROOT_RELEASE_COMPLETE.json"
LAUNCH_COMPLETE_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_2_R7_LAUNCHED.json"
BOOTSTRAP_WATCHDOG_READY_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_READY.json"
)
BOOTSTRAP_WATCHDOG_READY_PROTOCOL = (
    "schema5-v1.2-r7-bootstrap-watchdog-deployment-ready-v1"
)
BOOTSTRAP_WATCHDOG_ARM_INTENT_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_ARM_INTENT.json"
)
BOOTSTRAP_WATCHDOG_ARMED_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_ARMED.json"
)
BOOTSTRAP_WATCHDOG_ARMED_PROTOCOL = (
    "schema5-v1.2-r7-bootstrap-watchdog-armed-v1"
)
BOOTSTRAP_WATCHDOG_HANDOFF_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_HANDOFF_COMPLETE.json"
)
BOOTSTRAP_WATCHDOG_HANDOFF_PROTOCOL = (
    "schema5-v1.2-r7-bootstrap-watchdog-handoff-v1"
)
BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS = 600
BOOTSTRAP_OBSERVATIONS_ROOT_NAME = "bootstrap_watchdog_observations"
BOOTSTRAP_ISOLATED_DRILL_DIRECTORY = "isolated_cancellation_drill"
BOOTSTRAP_ISOLATED_RENDER_LOCK_NAME = (
    ".RECOVERY_CHAIN_SCHEMA5_V1_2_R7.isolated-drill-render.lock"
)
BOOTSTRAP_REPAIR_RESULT_NAME = "BOOTSTRAP_REPAIR_RESULT.json"
BOOTSTRAP_ISOLATED_SCRIPT_EXIT_CODE = 78
BOOTSTRAP_GENERATION_PROVENANCE_NAME = (
    "BOOTSTRAP_GENERATION_PROVENANCE.json"
)
BOOTSTRAP_DESCENDANT_ARM_INTENT_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_DESCENDANT_ARM_INTENT.json"
)
BOOTSTRAP_DESCENDANT_ARMED_NAME = (
    "RECOVERY_CHAIN_BOOTSTRAP_DESCENDANT_ARMED.json"
)
BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL = (
    "schema5-v1.2-r7-bootstrap-descendant-armed-v1"
)
SCHEDULER_ACCEPTANCE_COMPLETE_NAME = (
    "RECOVERY_CHAIN_SCHEDULER_ACCEPTANCE_COMPLETE.json"
)
SCHEDULER_ACCEPTANCE_PROTOCOL = (
    "schema5-v1.2-r7-recovery-scheduler-acceptance-v1"
)
SCHEDULER_ACCEPTANCE_INTENT_NAME = (
    "RECOVERY_CHAIN_SCHEDULER_ACCEPTANCE_INTENT.json"
)
SCHEDULER_ACCEPTANCE_ARTIFACT_ROOT_NAME = "scheduler_acceptance_artifacts"
SCHEDULER_ACCEPTANCE_STAGING_ROOT_NAME = "scheduler_acceptance_staging"
DEPENDENCY_POLICY_CONTRACT = (
    "afterok+per_stage_afterany+aggregate_afterany_sentinel+kill_invalid_depend"
    "+held_root+sealed_dependency_cascade"
)
REPAIR_ROOT_NAME = "recovery_chain_repairs_v1_2_r7"
CAPACITY_TRANSIENT_ROOT_NAME = "fleet_capacity_transients_v1_2_r7"
CAPACITY_TRANSIENT_MARKER_NAME = "CAPACITY_TRANSIENT_COMPLETE.json"
PROTECTED_CAPACITY_MARKER_NAME = "PROTECTED_CAPACITY_COMPLETE.json"
PROTECTED_CAPACITY_PROTOCOL = protected_capacity.PROTOCOL
PROTECTED_CAPACITY_SOURCE = protected_capacity.CAPACITY_SOURCE
PROTECTED_MINIMUM_SCIENTIFIC_WALL_SECONDS = (
    protected_capacity.MIN_SCIENTIFIC_WALL_SECONDS
)
PROTECTED_CLIENT_WALL_SECONDS = protected_capacity.CLIENT_WALL_SECONDS
PROTECTED_RUNNING_SCIENTIFIC_JOBS = (
    protected_capacity.CLIENT_JOB_ELEMENTS
    + protected_capacity.BASE_LOGICAL_REPLICAS
    + protected_capacity.RETAINED_WARM_TURNOVER_JOB_ELEMENTS
)
DURABLE_GIT_RELEASE_MARKER_NAME = durable_git.MARKER_NAME
DURABLE_GIT_RELEASE_PROTOCOL = durable_git.PROTOCOL
SUPERSEDED_R2_CANARY_FAILURE_RELATIVE_PATH = (
    Path("canary_failures")
    / "schema5-v1.2-r2"
    / "CANARY_FAILURE_SEALED.json"
)
SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL = (
    "schema5-v1.2-r2-partial-canary-failure-seal-v1"
)
SUPERSEDED_R2_RELEASE_TAG = "sweep-recovery-schema5-v1.2-r2"
SUPERSEDED_R2_RELEASE_COMMIT = "f06baaad024abb237a077e81edcd49a3c3167f9f"
SUPERSEDED_R2_RELEASE_TAG_OBJECT = "23b1a6d3d299b941bbd47d5900c8ff8b8ca0c398"
SUPERSEDED_R2_CANARY_JOB_ID = "18889366"
SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID = (
    "7528c410d8a496abfc6051e70454aa4a429b1f9176c408b3fec6128f94f5fa7f"
)
SUPERSEDED_R2_CANARY_FAILURE_FILE_COUNT = 18
SUPERSEDED_R2_CANARY_FAILURE_TOTAL_BYTES = 28_960
SUPERSEDED_R3_RELEASE_TAG = "sweep-recovery-schema5-v1.2-r3"
SUPERSEDED_R3_RELEASE_COMMIT = (
    "acd723ba9a99d88e77f7d752268bc31205c3a808"
)
SUPERSEDED_R3_RELEASE_TAG_OBJECT = (
    "fa87b283974afc1fa48fcb22b61e6ae7eec4bb36"
)
SUPERSEDED_R3_CHAIN_NAMESPACE = "schema5-v1.2-r3"
SUPERSEDED_R3_FAILURE_CLASSIFICATIONS = (
    "unsafe_recorded_broken_internal_symlink",
    "offline_clone_unseeded_release_local_cache",
)
SUPERSEDED_R4_TOOLCHAIN_FAILURE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r4-toolchain"
)
SUPERSEDED_R5_PRELAUNCH_FAILURE_RELATIVE_ROOT = (
    r5_prelaunch_failure.EVIDENCE_RELATIVE_ROOT
)
SUPERSEDED_R6_PRELAUNCH_FAILURE_RELATIVE_ROOT = (
    r6_prelaunch_failure.EVIDENCE_RELATIVE_ROOT
)
WATCHDOG_READY_MARKER_NAME = "WATCHDOG_READY.json"
WATCHDOG_READY_PROTOCOL = "schema5-v1.2-r7-external-watchdog-v1"
EXTERNAL_WATCHDOG_DRILL_MARKER_NAME = (
    "EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json"
)
EXTERNAL_WATCHDOG_DRILL_PROTOCOL = (
    "schema5-v1.2-r7-external-watchdog-drill-v1"
)
THROUGHPUT_QUALIFICATION_ROOT_NAME = (
    "schema5_throughput_qualification_v1"
)
THROUGHPUT_QUALIFICATION_MARKER_NAME = (
    "THROUGHPUT_QUALIFICATION_COMPLETE.json"
)
THROUGHPUT_QUALIFICATION_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-v3"
)
THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION = 3
THROUGHPUT_QUALIFICATION_PLAN_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-load-plan-v3"
)
THROUGHPUT_QUALIFICATION_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-evidence-v3"
)
THROUGHPUT_QUALIFICATION_SCHEDULER_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-scheduler-evidence-v3"
)
THROUGHPUT_QUALIFICATION_SEMANTIC_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-semantic-evidence-v3"
)
THROUGHPUT_QUALIFICATION_OBSERVATION_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-observation-v3"
)
THROUGHPUT_QUALIFICATION_WINDOW_INTENT_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-load-window-intent-v3"
)
THROUGHPUT_QUALIFICATION_CYCLE_INTENT_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-cycle-intent-v3"
)
THROUGHPUT_QUALIFICATION_ATTEMPT_POINTER_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-attempt-pointer-v1"
)
THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-current-attempt-v1"
)
THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-failure-v4"
)
THROUGHPUT_QUALIFICATION_LEGACY_FAILURE_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-failure-v1"
)
THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_NAME = "CURRENT_ATTEMPT.json"
THROUGHPUT_QUALIFICATION_ATTEMPT_DIRECTORY = "attempts"
THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY = "attempt-pointers"
THROUGHPUT_QUALIFICATION_RUN_DIRECTORY = (
    "throughput-qualification-attempts"
)
THROUGHPUT_QUALIFICATION_FAILURE_NAME = "QUALIFICATION_FAILURE.json"
THROUGHPUT_QUALIFICATION_TRANSITION_DIRECTORY = "capacity-transitions"
THROUGHPUT_QUALIFICATION_CURRENT_TRANSITION_NAME = (
    "CURRENT_CAPACITY_TRANSITION.json"
)
THROUGHPUT_QUALIFICATION_TRANSITION_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-capacity-transition-v2"
)
THROUGHPUT_QUALIFICATION_CURRENT_TRANSITION_PROTOCOL = (
    "schema5-v1.2-r7-throughput-qualification-current-capacity-transition-v1"
)
THROUGHPUT_QUALIFICATION_CELLS = 768
THROUGHPUT_QUALIFICATION_QIDS = 15_360
THROUGHPUT_QUALIFICATION_CEILINGS = (24, 96, 192, 384)
THROUGHPUT_QUALIFICATION_HEALTH_SOAK_SECONDS = 7_200
THROUGHPUT_QUALIFICATION_MIN_LOADED_OBSERVATIONS = 2
THROUGHPUT_QUALIFICATION_MAX_OBSERVATION_GAP_SECONDS = 660
THROUGHPUT_QUALIFICATION_MIN_QIDS_PER_DAY = 201_994
THROUGHPUT_QUALIFICATION_MIN_EXECUTION_EVENTS = 16_833
QUARANTINE_ROOT_NAME = "quarantine"
QUARANTINE_EVIDENCE_ROOT_NAME = "materialization_quarantines"
RENDER_LOCK_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_2_R7.render.lock"
SUBMISSION_LOCK_NAME = ".RECOVERY_CHAIN_SCHEMA5_V1_2_R7.submit.lock"
SENTINEL_TOOL_FILENAME = "schema5_recovery_sentinel.py"
SENTINEL_BOOTSTRAP_ROOT_NAME = "sentinel_bootstraps"
SENTINEL_BOOTSTRAP_MARKER_NAME = "BOOTSTRAP_COMPLETE.json"
SENTINEL_BOOTSTRAP_INVENTORY_NAME = "BOOTSTRAP_PAYLOAD.sha256"
SENTINEL_BOOTSTRAP_PROTOCOL = "schema5-v1.2-r7-sentinel-bootstrap"
SOURCE_CHECKOUT_SEAL_NAME = "SOURCE_CHECKOUT_SCHEMA5_V1_2_R7_COMPLETE.json"
SOURCE_CHECKOUT_SEAL_PROTOCOL = "schema5-v1.2-r7-source-checkout-seal-v1"
BUNDLED_TOOL_GIT_PATHS = (
    "scripts/schema5_recovery_sentinel.py",
    "scripts/verify_schema5_recovery_evidence.py",
    "scripts/render_schema5_recovery_chain.py",
    "scripts/render_schema5_recovery_chain_v12.py",
    "scripts/schema5_conda_runtime_identity.py",
    "scripts/provision_schema5_conda_toolchain.py",
    "scripts/seal_recovery_evidence.py",
    "scripts/seal_schema5_r4_toolchain_failure.py",
    "scripts/seal_schema5_r5_prelaunch_failure.py",
    "scripts/seal_schema5_r6_prelaunch_failure.py",
)
MATERIALIZATION_PILOT_GIT_PATH = "scripts/run_schema5_materialization_pilot.py"
OWNERSHIP_POLICY_GIT_PATH = "configs/environment_ownership_policy.v1.json"
INTEGRITY_NORMALIZATION_POLICY_GIT_PATH = (
    "configs/environment_integrity_normalization_policy.v1.json"
)
SLURM_CANARY_GIT_PATH = "scripts/run_schema5_slurm_fleet_canary.py"
DURABLE_GIT_RELEASE_GIT_PATH = (
    "scripts/publish_schema5_durable_git_release.py"
)
FLEET_TRANSACTIONS_GIT_PATH = (
    "src/agents_scaling/serving/fleet_transactions.py"
)
PREREQUISITE_CODE_GIT_PATHS = (
    MATERIALIZATION_PILOT_GIT_PATH,
    "scripts/capture_schema5_environments.py",
    "scripts/materialize_schema5_release.py",
    "scripts/freeze_schema5_release.py",
    "scripts/schema5_conda_runtime_identity.py",
    "scripts/provision_schema5_conda_toolchain.py",
    "scripts/seal_recovery_evidence.py",
    "scripts/seal_schema5_r4_toolchain_failure.py",
    "scripts/seal_schema5_r5_prelaunch_failure.py",
    "scripts/seal_schema5_r6_prelaunch_failure.py",
    OWNERSHIP_POLICY_GIT_PATH,
    INTEGRITY_NORMALIZATION_POLICY_GIT_PATH,
    SLURM_CANARY_GIT_PATH,
    DURABLE_GIT_RELEASE_GIT_PATH,
    FLEET_TRANSACTIONS_GIT_PATH,
)
MATERIALIZATION_PILOT_MARKER = "PILOT_COMPLETE.json"
SLURM_CANARY_MARKER = "CANARY_COMPLETE.json"
PREREQUISITE_PROTOCOL = "schema5-v1.2-r7-prerequisite-evidence-v10"
PRODUCTION_CONDA_RECONCILIATION_INCIDENT_SHA256 = (
    "9f588ce4ffc4aeb5a3ac494a35244604e4eb100a7190b18d9eb3c568468fdb09"
)
PRODUCTION_CONDA_RECONCILIATION_INCIDENT_ID = (
    "2abfc4fab5828cd1e965ff822e54e55462b1a6a2b280b7f47eebd90cadcfd872"
)
R1_PROTOCOL_VERIFICATION_PROTOCOL = (
    "schema5-v1.2-r7-native-r1-evidence-verification-v1"
)
R1_CHAIN_PROTOCOL = "schema5-v1.1-r1-recovery-chain"
R1_EVIDENCE_DISPATCH_PROTOCOL = "schema5-recovery-evidence-protocol-dispatch"
R1_EVIDENCE_VERIFIER_GIT_PATH = "scripts/verify_schema5_recovery_evidence.py"
R1_NATIVE_RENDERER_GIT_PATH = "scripts/render_schema5_recovery_chain.py"
R1_PROTOCOL_TOOL_GIT_PATHS = (
    R1_EVIDENCE_VERIFIER_GIT_PATH,
    R1_NATIVE_RENDERER_GIT_PATH,
    "scripts/render_schema5_recovery_chain_v12.py",
)
CHAIN_SCHEMA_VERSION = 11
SUBMISSION_SCHEMA_VERSION = 4
VISIBILITY_GRACE_SECONDS = 300.0
LAUNCH_GATE_TIMEOUT_SECONDS = 1_200
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
SMOKE_ATTEMPT_BASE_NAME = "schema5-smoke-readiness-v1"
SMOKE_ATTEMPT_RUNS_NAME = "schema5-smoke-attempt-runs-v1"
SMOKE_ATTEMPT_POINTER_PROTOCOL = (
    "schema5-v1.2-r7-smoke-attempt-pointer-v1"
)
SMOKE_CURRENT_SELECTOR_PROTOCOL = (
    "schema5-v1.2-r7-smoke-current-selector-v1"
)
SMOKE_ATTEMPT_COMPLETE_PROTOCOL = (
    "schema5-v1.2-r7-smoke-attempt-complete-v1"
)
SMOKE_ATTEMPT_FAILURE_PROTOCOL = (
    "schema5-v1.2-r7-smoke-attempt-failure-v1"
)
SMOKE_ATTEMPT_BINDING_PROTOCOL = (
    "schema5-v1.2-r7-smoke-attempt-binding-v1"
)
SMOKE_ATTEMPT_BINDING_FIELDS = frozenset(
    {
        "protocol",
        "attempt_id",
        "attempt_ordinal",
        "immutable_sha256",
        "capacity_generation",
        "rollout_generation",
        "fleet_contract_sha256",
        "release_fleet_contract_sha256",
        "trusted_catalog_id",
    }
)
SMOKE_CURRENT_SELECTOR_NAME = "CURRENT.json"
SMOKE_ATTEMPT_COMPLETE_NAME = "SMOKE_ATTEMPT_COMPLETE.json"
SMOKE_ATTEMPT_FAILURE_NAME = "SMOKE_ATTEMPT_FAILURE.json"
SMOKE_EVIDENCE_NAME = "smoke_runs.json"
SMOKE_TRUSTED_CATALOG_FIELDS = frozenset(
    {
        "catalog_id",
        "marker_path",
        "marker_sha256",
        "inventory_sha256",
        "catalog_payload_sha256",
        "allowed_generation_tuple_count",
    }
)
HEAVY_SERIAL_ORDER = (
    "snapshot_adopt_verify",
    "environment_capture",
    "release_materialize",
    "release_freeze",
    "legacy_consolidate",
    "legacy_consolidated_snapshot",
    "legacy_consolidated_verify",
)
FIRST_LEGACY_MUTATION = "legacy_consolidate"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
STAGE_SENTINEL_PREFIX = "stage_failure_sentinel_"
PRODUCTION_STAGE_NAMES = (
    "source_checkout",
    "maintenance_preflight",
    "snapshot_adopt_verify",
    "environment_capture",
    "release_materialize",
    "release_freeze",
    "legacy_consolidate",
    "legacy_consolidated_snapshot",
    "legacy_consolidated_verify",
    "legacy_retire",
    "schema5_initialize",
    "static_readiness",
    "context_readiness",
    "email_readiness",
    "supplementary_cache",
    "fleet_bootstrap",
    "fleet_readiness",
    "smoke_readiness",
    "throughput_qualification",
    "controller_drill",
    "production_resume",
)
STAGE_SENTINEL_NAMES = tuple(
    f"{STAGE_SENTINEL_PREFIX}{stage}" for stage in PRODUCTION_STAGE_NAMES
)
STAGE_SENTINEL_CONTRACT = tuple(
    (
        f"{STAGE_SENTINEL_PREFIX}{stage}",
        f"{21 + index:02d}_stage_sentinel_{stage}.sbatch",
        (stage,),
        "01:00:00",
        "2G",
        1,
        "afterany",
    )
    for index, stage in enumerate(PRODUCTION_STAGE_NAMES)
)
SEALED_SNAPSHOT_CONTRACT = {
    "file_count": 201_528,
    "total_bytes": 48_293_388_049,
    "snapshot_inventory_sha256": (
        "9e14555fa6b8ead630d61f02a57c11e12aff8e11f297cbe7eba9fb34fb3a938f"
    ),
    "completion_sha256": (
        "8acd280037833b6df660046d4c89c7e5945d59ead59e2f019552f1243c0780f2"
    ),
    "attestation_sha256": (
        "e6fd20818800ba6b16041b919d9ee288cba35849ee10134a2bdb300ba092df7d"
    ),
}
R1_EVIDENCE_RELATIVE_PATHS = {
    "chain_manifest": "RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json",
    "submission_receipt": "RECOVERY_CHAIN_SCHEMA5_V1_1_R1_SUBMISSION.json",
    "failure_envelope": "FAILED_RECOVERY_CHAIN_SCHEMA5_V1_1_R1.json",
    "quarantine_seal": "materialization_quarantines/partial-job-18555913.sealed.json",
    "quarantine_inventory": (
        "materialization_quarantines/"
        "partial-job-18555913.sealed-inventory.txt"
    ),
    "quarantine_completion": (
        "materialization_quarantines/partial-job-18555913.complete.json"
    ),
    "quarantine_idempotency_receipt": (
        "r1_acceptance/QUARANTINE_IDEMPOTENCY_RECEIPT.json"
    ),
    "zero_result_mutation_receipt": (
        "r1_acceptance/ZERO_RESULT_MUTATION_RECEIPT.json"
    ),
    "failure_evidence_snapshot_completion": (
        "r1_failure_evidence/SNAPSHOT_COMPLETE.json"
    ),
    "failure_evidence_snapshot_attestation": (
        "r1_failure_evidence.attestation.json"
    ),
    "conda_incident": (
        "post_snapshot_incidents/"
        "2026-07-23_conda_pip_interop_source_metadata_reconciliation.json"
    ),
}

# This deliberately duplicates the operational contract encoded by ``job_specs``.
# Verification must compare a rendered manifest against an independent, fixed DAG
# contract rather than merely proving that the manifest is internally consistent.
EXPECTED_JOB_CONTRACT: tuple[
    tuple[str, str, tuple[str, ...], str, str, int, str], ...
] = (
    ("source_checkout", "00_source_checkout.sbatch", (), "01:00:00", "4G", 1, "afterok"),
    ("maintenance_preflight", "01_maintenance_preflight.sbatch", ("source_checkout",), "02:00:00", "8G", 1, "afterok"),
    ("snapshot_adopt_verify", "02_snapshot_adopt_verify.sbatch", ("maintenance_preflight",), "11:30:00", "8G", 1, "afterok"),
    ("environment_capture", "03_environment_capture.sbatch", ("snapshot_adopt_verify",), "11:30:00", "12G", 2, "afterok"),
    ("release_materialize", "04_release_materialize.sbatch", ("environment_capture",), "11:30:00", "12G", 2, "afterok"),
    ("release_freeze", "05_release_freeze.sbatch", ("release_materialize",), "11:30:00", "12G", 2, "afterok"),
    ("legacy_consolidate", "06_legacy_consolidate.sbatch", ("release_freeze", "snapshot_adopt_verify"), "11:30:00", "16G", 2, "afterok"),
    ("legacy_consolidated_snapshot", "07_legacy_consolidated_snapshot.sbatch", ("legacy_consolidate",), "11:30:00", "8G", 1, "afterok"),
    ("legacy_consolidated_verify", "08_legacy_consolidated_verify.sbatch", ("legacy_consolidated_snapshot",), "11:30:00", "8G", 1, "afterok"),
    ("legacy_retire", "09_legacy_retire.sbatch", ("legacy_consolidated_verify",), "06:00:00", "8G", 1, "afterok"),
    ("schema5_initialize", "10_schema5_initialize.sbatch", ("legacy_retire",), "11:30:00", "16G", 2, "afterok"),
    ("static_readiness", "11_static_readiness.sbatch", ("schema5_initialize",), "11:30:00", "8G", 1, "afterok"),
    ("context_readiness", "13_context_readiness.sbatch", ("static_readiness",), "11:30:00", "32G", 4, "afterok"),
    ("email_readiness", "14_email_readiness.sbatch", ("static_readiness",), "08:00:00", "2G", 1, "afterok"),
    ("supplementary_cache", "15_supplementary_cache.sbatch", ("context_readiness",), "11:30:00", "64G", 4, "afterok"),
    ("fleet_bootstrap", "12_fleet_bootstrap.sbatch", ("supplementary_cache",), "02:00:00", "4G", 1, "afterok"),
    ("fleet_readiness", "16_fleet_readiness.sbatch", ("fleet_bootstrap",), "11:00:00", "8G", 1, "afterok"),
    ("smoke_readiness", "17_smoke_readiness.sbatch", ("fleet_readiness",), "11:30:00", "8G", 1, "afterok"),
    ("throughput_qualification", "18_throughput_qualification.sbatch", ("smoke_readiness",), "11:30:00", "8G", 1, "afterok"),
    ("controller_drill", "19_controller_drill.sbatch", ("throughput_qualification", "email_readiness"), "02:00:00", "8G", 1, "afterok"),
    ("production_resume", "20_production_resume.sbatch", ("controller_drill", "throughput_qualification"), "11:30:00", "8G", 1, "afterok"),
    *STAGE_SENTINEL_CONTRACT,
    (
        "failure_sentinel",
        "42_failure_sentinel.sbatch",
        (*PRODUCTION_STAGE_NAMES, *STAGE_SENTINEL_NAMES),
        "01:00:00",
        "2G",
        1,
        "afterany",
    ),
)
EXPECTED_JOB_ORDER = tuple(row[0] for row in EXPECTED_JOB_CONTRACT)
MAINTENANCE_PREFLIGHT_PYTHON = r"""import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

RUN_IDS = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
)
results = Path(sys.argv[1])
recovery = Path(sys.argv[2])
expected_user = sys.argv[3]
if os.environ.get("USER") != expected_user:
    raise SystemExit(
        f"scheduler user drift: {os.environ.get('USER')!r} != {expected_user!r}"
    )
interlock_path = recovery / "MAINTENANCE_INTERLOCK.json"
try:
    interlock = json.loads(interlock_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"maintenance interlock is unavailable: {exc}") from exc
if (
    interlock.get("desired_state") != "maintenance"
    or interlock.get("admission_enabled") is not False
    or tuple(interlock.get("retired_run_ids", ())) != RUN_IDS
):
    raise SystemExit("maintenance interlock does not freeze the exact legacy runs")
proc = subprocess.run(
    ["squeue", "-u", expected_user, "-h", "-r", "-o", "%i|%j|%T|%o"],
    capture_output=True,
    text=True,
    check=False,
)
if proc.returncode != 0:
    raise SystemExit(
        f"squeue failed during maintenance check: {proc.stderr[:500]}"
    )
unsafe_rows = [
    line
    for line in proc.stdout.splitlines()
    if any(run_id in line for run_id in RUN_IDS)
    or any(
        prefix in line
        for prefix in ("asys-cells", "asys-dispatch", "asys-driver", "asys-serve")
    )
]
if unsafe_rows:
    raise SystemExit(
        "legacy jobs remain live during cleanup: " + "; ".join(unsafe_rows[:10])
    )
active_locks = []
inspected_cells = 0
for run_id in RUN_IDS:
    cells_root = results / run_id / "cells"
    if not cells_root.is_dir():
        continue
    for cell_dir in cells_root.iterdir():
        if not cell_dir.is_dir() or cell_dir.is_symlink():
            continue
        inspected_cells += 1
        lock_path = cell_dir / ".cell.lock"
        if not lock_path.exists():
            continue
        try:
            descriptor = os.open(lock_path, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    active_locks.append(f"{run_id}/{cell_dir.name}")
                    continue
                raise
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
if active_locks:
    raise SystemExit(f"held legacy cell locks remain: {active_locks[:10]}")
report = {
    "maintenance_interlock": {
        "name": "maintenance_interlock",
        "path": str(interlock_path.resolve()),
        "sha256": hashlib.sha256(interlock_path.read_bytes()).hexdigest(),
    },
    "scheduler_rows": len(proc.stdout.splitlines()),
    "legacy_jobs": 0,
    "inspected_cell_directories": inspected_cells,
    "held_cell_locks": 0,
}
print(json.dumps(report, indent=2, sort_keys=True))
"""


class ChainError(RuntimeError):
    """The recovery chain cannot be rendered, verified, or safely submitted."""


def _verified_conda_toolchain_binding(
    toolchain_root: Path,
    *,
    exercise: bool = True,
) -> dict[str, Any]:
    """Return the complete portable binding for the canonical sealed toolchain."""

    try:
        binding = conda_toolchain.verified_conda_toolchain_binding(
            toolchain_root,
            exercise=exercise,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        conda_toolchain.CondaToolchainProvisionError,
    ) as exc:
        raise ChainError(f"sealed Conda toolchain is invalid: {exc}") from exc
    if not isinstance(binding, dict):
        raise ChainError("sealed Conda toolchain verifier returned invalid evidence")
    return json.loads(json.dumps(binding, sort_keys=True))


def _validate_package_cache_seed_input(
    binding: Any,
    *,
    source_package_cache: Path,
) -> dict[str, Any]:
    required = {
        "source_package_cache",
        "inventory_sha256",
        "inventory_entry_count",
        "inventory_file_count",
        "inventory_total_file_bytes",
        "requirements_sha256",
        "required_package_count",
        "archive_count",
        "selected_top_level_entries",
        "input_id",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ChainError("pilot selected package-cache binding fields drifted")
    identity = dict(binding)
    input_id = identity.pop("input_id")
    integer_fields = (
        "inventory_entry_count",
        "inventory_file_count",
        "inventory_total_file_bytes",
        "required_package_count",
        "archive_count",
    )
    selected = binding.get("selected_top_level_entries")
    if (
        binding.get("source_package_cache") != str(source_package_cache)
        or _SHA256.fullmatch(str(binding.get("inventory_sha256", "")))
        is None
        or _SHA256.fullmatch(str(binding.get("requirements_sha256", "")))
        is None
        or any(
            not isinstance(binding.get(field), int)
            or isinstance(binding.get(field), bool)
            or binding[field] < 0
            for field in integer_fields
        )
        or binding["archive_count"] > binding["required_package_count"]
        or not isinstance(selected, list)
        or selected != sorted(set(selected))
        or any(
            not isinstance(value, str)
            or not value
            or "/" in value
            or "\n" in value
            or "\r" in value
            for value in selected
        )
        or _SHA256.fullmatch(str(input_id)) is None
        or input_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("pilot selected package-cache binding is invalid")
    return json.loads(json.dumps(binding, sort_keys=True))


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
    conda_toolchain_root: Path
    source_package_cache: Path
    environment_capture_root: Path
    captured_harness: Path
    captured_serving: Path
    jobs_root: Path
    logs_root: Path
    chain_manifest: Path
    immutable_pins: Path
    readiness: Path
    materialization_pilot_root: Path
    slurm_canary_root: Path

    @property
    def sentinel_bootstrap_root(self) -> Path:
        return (
            self.recovery_root
            / SENTINEL_BOOTSTRAP_ROOT_NAME
            / CHAIN_NAMESPACE
        )

    @property
    def sentinel_bootstrap_python(self) -> Path:
        return self.sentinel_bootstrap_root / "bin" / "python"

    @property
    def sentinel_bootstrap_inventory(self) -> Path:
        return (
            self.sentinel_bootstrap_root
            / SENTINEL_BOOTSTRAP_INVENTORY_NAME
        )

    @property
    def sentinel_bootstrap_marker(self) -> Path:
        return self.sentinel_bootstrap_root / SENTINEL_BOOTSTRAP_MARKER_NAME

    @property
    def capacity_transient_root(self) -> Path:
        return (
            self.recovery_root
            / CAPACITY_TRANSIENT_ROOT_NAME
            / CHAIN_NAMESPACE
        )

    @property
    def protected_capacity_marker(self) -> Path:
        return self.recovery_root / PROTECTED_CAPACITY_MARKER_NAME

    @property
    def durable_git_release_marker(self) -> Path:
        return self.recovery_root / DURABLE_GIT_RELEASE_MARKER_NAME

    @property
    def superseded_r2_canary_failure_marker(self) -> Path:
        return (
            self.recovery_root
            / SUPERSEDED_R2_CANARY_FAILURE_RELATIVE_PATH
        )

    @property
    def superseded_r3_prelaunch_failure_root(self) -> Path:
        return (
            self.recovery_root
            / "prelaunch_failures"
            / "schema5-v1.2-r3"
        )

    @property
    def superseded_r3_prelaunch_failure_marker(self) -> Path:
        return (
            self.superseded_r3_prelaunch_failure_root
            / "PRELAUNCH_FAILURE_SEALED.json"
        )

    @property
    def superseded_r4_toolchain_failure_root(self) -> Path:
        return (
            self.recovery_root
            / SUPERSEDED_R4_TOOLCHAIN_FAILURE_RELATIVE_ROOT
        )

    @property
    def superseded_r5_prelaunch_failure_root(self) -> Path:
        return (
            self.recovery_root
            / SUPERSEDED_R5_PRELAUNCH_FAILURE_RELATIVE_ROOT
        )

    @property
    def superseded_r6_prelaunch_failure_root(self) -> Path:
        return (
            self.recovery_root
            / SUPERSEDED_R6_PRELAUNCH_FAILURE_RELATIVE_ROOT
        )

    @property
    def external_watchdog_drill_marker(self) -> Path:
        return self.recovery_root / EXTERNAL_WATCHDOG_DRILL_MARKER_NAME

    @property
    def watchdog_ready_marker(self) -> Path:
        return self.recovery_root / WATCHDOG_READY_MARKER_NAME

    @property
    def throughput_qualification_root(self) -> Path:
        return self.readiness / THROUGHPUT_QUALIFICATION_ROOT_NAME

    @property
    def throughput_qualification_marker(self) -> Path:
        return (
            self.throughput_qualification_root
            / THROUGHPUT_QUALIFICATION_MARKER_NAME
        )


@dataclass(frozen=True)
class JobSpec:
    name: str
    filename: str
    dependencies: tuple[str, ...]
    time_limit: str
    memory: str
    cpus: int
    body: str
    dependency_type: str = "afterok"


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

TRUSTED_SYSTEM_PATH = "/usr/bin:/bin"
_TRUSTED_EXECUTABLES = {
    "bash": "/bin/bash",
    "git": "/usr/bin/git",
    "sacct": "/usr/bin/sacct",
    "sbatch": "/usr/bin/sbatch",
    "scontrol": "/usr/bin/scontrol",
    "squeue": "/usr/bin/squeue",
}
_BLOCKED_PROCESS_ENVIRONMENT = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_ATTR_NOSYSTEM",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_TEMPLATE_DIR",
        "GIT_WORK_TREE",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_EXIT_ERROR",
        "SLURM_TIME_FORMAT",
    }
)
_BLOCKED_PROCESS_ENVIRONMENT_PREFIXES = (
    "BASH_FUNC_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "GIT_TRACE",
    "SACCT_",
    "SBATCH_",
    "SCONTROL_",
    "SQUEUE_",
)


def _sanitized_process_environment() -> dict[str, str]:
    """Return the fixed command trust boundary used by Git and Slurm clients."""

    environment = dict(os.environ)
    for name in tuple(environment):
        if name in _BLOCKED_PROCESS_ENVIRONMENT or name.startswith(
            _BLOCKED_PROCESS_ENVIRONMENT_PREFIXES
        ):
            environment.pop(name, None)
    environment.update(
        {
            "PATH": TRUSTED_SYSTEM_PATH,
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "PAGER": "cat",
        }
    )
    return environment


def _trusted_command_argv(argv: Sequence[str]) -> list[str]:
    """Resolve security-sensitive system commands independently of ambient PATH."""

    command = list(argv)
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ChainError("trusted subprocess argv is empty or invalid")
    trusted = _TRUSTED_EXECUTABLES.get(command[0])
    if trusted is not None:
        command[0] = trusted
    return command


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run an internal scheduler command under the fixed process environment."""

    return subprocess.run(
        _trusted_command_argv(argv),
        text=True,
        capture_output=True,
        check=False,
        env=_sanitized_process_environment(),
    )


def _job_comment(chain_id: str, name: str, generation: int) -> str:
    if generation < 0:
        raise ChainError("recovery-chain submission generation cannot be negative")
    return f"asys:s5-recovery-v1.2-r7:{chain_id}:g{generation:04d}:{name}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slurm_timestamp(timestamp: float) -> str:
    """Return the timezone-free ISO form accepted by this cluster's ``sacct -S``."""

    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _compact_canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
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


def _reject_live_prefix_execution(
    *,
    source_harness: Path,
    source_serving: Path,
) -> None:
    """Prevent renderer imports from changing either pilot-bound live prefix."""

    runtimes = {
        _lexical_absolute(Path(sys.prefix)),
        _lexical_absolute(Path(sys.executable)).parent.parent,
    }
    forbidden = {
        _lexical_absolute(source_harness),
        _lexical_absolute(source_serving),
    }
    if runtimes & forbidden:
        raise ChainError(
            "recovery renderer must run from the sealed pilot harness or another "
            "non-live runtime; --dev-python is bootstrap-copy provenance only"
        )


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
    conda_toolchain_root: Path,
    source_package_cache: Path,
    materialization_pilot_root: Path,
    slurm_canary_root: Path,
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
    materialization_pilot_root = _absolute(
        materialization_pilot_root,
        description="materialization pilot root",
    )
    slurm_canary_root = _absolute(
        slurm_canary_root,
        description="Slurm fleet canary root",
    )
    expected_pilot_root = (
        recovery_root / "materialization_pilots" / CHAIN_NAMESPACE
    )
    expected_canary_root = recovery_root / "slurm_canaries" / CHAIN_NAMESPACE
    if materialization_pilot_root != expected_pilot_root:
        raise ChainError(
            "materialization pilot root must be the canonical "
            f"{expected_pilot_root}, got {materialization_pilot_root}"
        )
    if slurm_canary_root != expected_canary_root:
        raise ChainError(
            f"Slurm fleet canary root must be the canonical {expected_canary_root}, "
            f"got {slurm_canary_root}"
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
        ("materialization pilot root", materialization_pilot_root),
        ("Slurm fleet canary root", slurm_canary_root),
    ):
        if not directory.is_dir():
            raise ChainError(f"{description} is not an existing directory: {directory}")
    dev_python = _environment_executable(
        dev_python,
        environment_prefix=source_harness,
        description="development Python",
    )
    conda_toolchain_root = _absolute(
        conda_toolchain_root, description="sealed Conda toolchain root"
    )
    expected_toolchain_root = (
        recovery_root
        / "toolchains"
        / conda_toolchain.TOOLCHAIN_NAMESPACE_DIRECTORY
        / conda_toolchain.TOOLCHAIN_DIRECTORY_NAME
    )
    if conda_toolchain_root != expected_toolchain_root:
        raise ChainError(
            "Conda toolchain root must be the canonical "
            f"{expected_toolchain_root}, got {conda_toolchain_root}"
        )
    source_package_cache = _absolute(
        source_package_cache, description="source Conda package cache"
    )
    if not source_package_cache.is_dir():
        raise ChainError(
            "source Conda package cache is not an existing directory: "
            f"{source_package_cache}"
        )
    _verified_conda_toolchain_binding(conda_toolchain_root)
    return RecoveryPaths(
        repository=repository,
        results_root=results_root,
        recovery_root=recovery_root,
        source_checkout=recovery_root / "release_source_checkout_v1_2_r7",
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
        conda_toolchain_root=conda_toolchain_root,
        source_package_cache=source_package_cache,
        environment_capture_root=(
            recovery_root / "environment_captures" / RELEASE_ID
        ),
        captured_harness=(
            recovery_root
            / "environment_captures"
            / RELEASE_ID
            / "seeds"
            / "harness"
        ),
        captured_serving=(
            recovery_root
            / "environment_captures"
            / RELEASE_ID
            / "seeds"
            / "serving"
        ),
        jobs_root=recovery_root / "jobs" / CHAIN_NAMESPACE,
        logs_root=recovery_root / "logs" / CHAIN_NAMESPACE,
        chain_manifest=recovery_root / CHAIN_MANIFEST_NAME,
        immutable_pins=recovery_root / "immutable_pins.schema5-v1.json",
        readiness=recovery_root / "readiness",
        materialization_pilot_root=materialization_pilot_root,
        slurm_canary_root=slurm_canary_root,
    )


_BOOTSTRAP_RELATIVE_PATH = re.compile(r"[A-Za-z0-9._+@/-]+\Z")
_PYTHON_STDLIB_DIRECTORY = re.compile(r"python([0-9]+)\.([0-9]+)\Z")
_BOOTSTRAP_PROBE_TOKEN = "schema5_v1_2_r7_sentinel_bootstrap_ok"


def _bootstrap_stdlib_root(paths: RecoveryPaths) -> Path:
    library = _require_canonical_path(
        paths.source_harness / "lib",
        description="sentinel bootstrap source library",
        kind="directory",
    )
    candidates = sorted(
        item
        for item in library.iterdir()
        if (
            not item.is_symlink()
            and item.is_dir()
            and _PYTHON_STDLIB_DIRECTORY.fullmatch(item.name)
            and (item / "os.py").is_file()
            and (item / "lib-dynload").is_dir()
        )
    )
    executable_match = re.fullmatch(
        r"python([0-9]+)\.([0-9]+)", paths.dev_python.name
    )
    if executable_match is not None:
        preferred = library / (
            f"python{executable_match.group(1)}.{executable_match.group(2)}"
        )
        if preferred in candidates:
            return preferred
    if len(candidates) != 1:
        raise ChainError(
            "sentinel bootstrap requires exactly one unambiguous Python stdlib: "
            f"{[str(item) for item in candidates]}"
        )
    return candidates[0]


def _bootstrap_source_records(
    paths: RecoveryPaths,
) -> tuple[list[dict[str, Any]], str]:
    """Inventory the small stdlib-only runtime without invoking Conda."""

    prefix = _require_canonical_path(
        paths.source_harness,
        description="sentinel bootstrap source prefix",
        kind="directory",
    )
    interpreter = _require_canonical_path(
        paths.dev_python,
        description="sentinel bootstrap source interpreter",
        kind="file",
    )
    try:
        interpreter.relative_to(prefix)
    except ValueError as exc:
        raise ChainError(
            "sentinel bootstrap interpreter is outside its source prefix"
        ) from exc
    if not os.access(interpreter, os.X_OK):
        raise ChainError("sentinel bootstrap source interpreter is not executable")
    stdlib = _bootstrap_stdlib_root(paths)

    selected: dict[str, Path] = {"bin/python": interpreter}
    for directory, directory_names, filenames in os.walk(
        stdlib, topdown=True, followlinks=False
    ):
        current = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            if name in {"site-packages", "__pycache__"}:
                continue
            if child.is_symlink():
                raise ChainError(
                    f"sentinel bootstrap stdlib contains a directory symlink: {child}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            source = current / name
            relative = source.relative_to(stdlib)
            selected[f"lib/{stdlib.name}/{relative.as_posix()}"] = source

    # Extension modules use $ORIGIN/../.. for their release-local dependencies.
    # Copy every top-level regular library (and dereference file aliases) so the
    # bootstrap never falls back to a mutable live-prefix library.
    for item in sorted((prefix / "lib").iterdir(), key=lambda value: value.name):
        if item == stdlib or (item.is_dir() and not item.is_symlink()):
            continue
        try:
            resolved = item.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ChainError(
                f"sentinel bootstrap source library is unsafe: {item}: {exc}"
            ) from exc
        if resolved.is_dir():
            # Directory aliases such as terminfo are not used by the sentinel.
            continue
        if not resolved.is_file():
            raise ChainError(
                f"sentinel bootstrap source library is not regular: {item}"
            )
        try:
            resolved.relative_to(prefix)
        except ValueError as exc:
            raise ChainError(
                f"sentinel bootstrap library alias escapes its prefix: {item}"
            ) from exc
        selected[f"lib/{item.name}"] = item

    records: list[dict[str, Any]] = []
    for relative, lexical in sorted(selected.items()):
        if not _BOOTSTRAP_RELATIVE_PATH.fullmatch(relative) or relative.startswith("/"):
            raise ChainError(
                f"sentinel bootstrap selected an unsafe relative path: {relative!r}"
            )
        try:
            source = lexical.resolve(strict=True)
            source.relative_to(prefix)
            metadata = source.stat()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ChainError(
                f"sentinel bootstrap source is unsafe: {lexical}: {exc}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ChainError(
                f"sentinel bootstrap source is not a regular file: {source}"
            )
        records.append(
            {
                "relative_path": relative,
                "source_path": str(source),
                "sha256": _sha256(source),
                "size": metadata.st_size,
                "executable": bool(stat.S_IMODE(metadata.st_mode) & 0o111),
            }
        )
    if not records or records[0]["relative_path"] != "bin/python":
        raise ChainError("sentinel bootstrap source inventory is incomplete")
    return records, stdlib.name


def _bootstrap_inventory_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        f"{record['sha256']}  {record['relative_path']}\n"
        for record in records
    ]
    return "".join(lines).encode("utf-8")


def _bootstrap_marker(
    paths: RecoveryPaths,
    records: Sequence[Mapping[str, Any]],
    *,
    stdlib_name: str,
) -> dict[str, Any]:
    inventory = _bootstrap_inventory_bytes(records)
    interpreter = next(
        record for record in records if record["relative_path"] == "bin/python"
    )
    source_contract = [
        {
            "relative_path": record["relative_path"],
            "source_path": record["source_path"],
            "sha256": record["sha256"],
            "size": record["size"],
            "executable": record["executable"],
        }
        for record in records
    ]
    return {
        "schema_version": 1,
        "protocol": SENTINEL_BOOTSTRAP_PROTOCOL,
        "release_id": RELEASE_ID,
        "chain_namespace": CHAIN_NAMESPACE,
        "bootstrap_root": str(paths.sentinel_bootstrap_root),
        "bootstrap_python": str(paths.sentinel_bootstrap_python),
        "payload_inventory": str(paths.sentinel_bootstrap_inventory),
        "payload_inventory_sha256": _sha256_bytes(inventory),
        "payload_file_count": len(records),
        "payload_total_bytes": sum(int(record["size"]) for record in records),
        "source_prefix": str(paths.source_harness),
        "source_interpreter": str(paths.dev_python),
        "source_interpreter_sha256": interpreter["sha256"],
        "source_stdlib": str(paths.source_harness / "lib" / stdlib_name),
        "source_inventory_sha256": _sha256_bytes(
            _canonical_json(source_contract)
        ),
        "copy_policy": (
            "buffered-read-write;stdlib-without-site-packages-or-bytecode;"
            "top-level-runtime-libraries-dereferenced;no-shared-inodes"
        ),
    }


def _prospective_sentinel_bootstrap(
    paths: RecoveryPaths,
) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    records, stdlib_name = _bootstrap_source_records(paths)
    return (
        _bootstrap_marker(paths, records, stdlib_name=stdlib_name),
        _bootstrap_inventory_bytes(records),
        records,
    )


def _copy_bootstrap_file(source: Path, destination: Path, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, mode)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb", closefd=False
        ) as destination_handle:
            while chunk := source_handle.read(1024 * 1024):
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(descriptor)


def _discard_stale_bootstrap_temporary(path: Path, *, parent: Path) -> None:
    pattern = re.compile(
        rf"\.{re.escape(CHAIN_NAMESPACE)}\.bootstrap\.[0-9]+\.[0-9a-f]{{32}}\Z"
    )
    if (
        path.parent != parent
        or pattern.fullmatch(path.name) is None
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise ChainError(f"unsafe sentinel bootstrap temporary: {path}")
    for directory, directory_names, filenames in os.walk(
        path, topdown=False, followlinks=False
    ):
        current = Path(directory)
        if any((current / name).is_symlink() for name in directory_names + filenames):
            raise ChainError(
                f"sentinel bootstrap temporary contains a symlink: {current}"
            )
        os.chmod(current, 0o700)
    shutil.rmtree(path)
    _fsync_directory(parent)


def _parse_bootstrap_inventory(payload: bytes) -> dict[str, str]:
    records: dict[str, str] = {}
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ChainError("sentinel bootstrap inventory is not UTF-8") from exc
    for line in lines:
        match = re.fullmatch(
            r"([0-9a-f]{64})  ([A-Za-z0-9._+@/-]+)", line
        )
        if match is None:
            raise ChainError("sentinel bootstrap inventory is malformed")
        digest, relative = match.groups()
        if relative.startswith("/") or relative in records:
            raise ChainError("sentinel bootstrap inventory paths are invalid")
        records[relative] = digest
    if not records:
        raise ChainError("sentinel bootstrap inventory is empty")
    return records


def _verify_sentinel_bootstrap(
    paths: RecoveryPaths,
    *,
    expected_marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _require_canonical_path(
        paths.sentinel_bootstrap_root,
        description="sealed sentinel bootstrap",
        kind="directory",
    )
    marker_path = _require_canonical_path(
        paths.sentinel_bootstrap_marker,
        description="sentinel bootstrap completion marker",
        kind="file",
    )
    inventory_path = _require_canonical_path(
        paths.sentinel_bootstrap_inventory,
        description="sentinel bootstrap payload inventory",
        kind="file",
    )
    marker = _read_json(
        marker_path, description="sentinel bootstrap completion marker"
    )
    if expected_marker is not None and marker != dict(expected_marker):
        raise ChainError(
            "sealed sentinel bootstrap does not match the current source capture"
        )
    expected_fields = {
        "schema_version",
        "protocol",
        "release_id",
        "chain_namespace",
        "bootstrap_root",
        "bootstrap_python",
        "payload_inventory",
        "payload_inventory_sha256",
        "payload_file_count",
        "payload_total_bytes",
        "source_prefix",
        "source_interpreter",
        "source_interpreter_sha256",
        "source_stdlib",
        "source_inventory_sha256",
        "copy_policy",
    }
    if (
        set(marker) != expected_fields
        or marker["schema_version"] != 1
        or marker["protocol"] != SENTINEL_BOOTSTRAP_PROTOCOL
        or marker["release_id"] != RELEASE_ID
        or marker["chain_namespace"] != CHAIN_NAMESPACE
        or marker["bootstrap_root"] != str(root)
        or marker["bootstrap_python"] != str(paths.sentinel_bootstrap_python)
        or marker["payload_inventory"] != str(inventory_path)
        or not isinstance(marker["payload_file_count"], int)
        or isinstance(marker["payload_file_count"], bool)
        or marker["payload_file_count"] <= 0
        or not isinstance(marker["payload_total_bytes"], int)
        or isinstance(marker["payload_total_bytes"], bool)
        or marker["payload_total_bytes"] <= 0
        or not _SHA256.fullmatch(str(marker["payload_inventory_sha256"]))
        or not _SHA256.fullmatch(str(marker["source_interpreter_sha256"]))
        or not _SHA256.fullmatch(str(marker["source_inventory_sha256"]))
    ):
        raise ChainError("sentinel bootstrap completion marker is invalid")
    if any(
        stat.S_IMODE(path.stat().st_mode) & 0o222
        for path in (root, marker_path, inventory_path)
    ):
        raise ChainError("sentinel bootstrap is not sealed read-only")
    inventory_payload = inventory_path.read_bytes()
    if _sha256_bytes(inventory_payload) != marker["payload_inventory_sha256"]:
        raise ChainError("sentinel bootstrap inventory hash drifted")
    inventory = _parse_bootstrap_inventory(inventory_payload)
    if len(inventory) != marker["payload_file_count"]:
        raise ChainError("sentinel bootstrap payload count drifted")

    observed: set[str] = set()
    observed_directories: set[str] = set()
    total_bytes = 0
    inodes: set[tuple[int, int]] = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ChainError(f"sentinel bootstrap contains a symlink: {item}")
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise ChainError(f"sentinel bootstrap contains a writable entry: {item}")
        if item.is_dir():
            observed_directories.add(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ChainError(
                f"sentinel bootstrap contains a non-unique regular file: {item}"
            )
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes:
            raise ChainError("sentinel bootstrap contains shared payload inodes")
        inodes.add(inode)
        observed.add(relative)
        if relative in inventory:
            total_bytes += metadata.st_size
            if _sha256(item) != inventory[relative]:
                raise ChainError(
                    f"sentinel bootstrap payload hash drifted: {relative}"
                )
    expected_files = set(inventory) | {
        SENTINEL_BOOTSTRAP_INVENTORY_NAME,
        SENTINEL_BOOTSTRAP_MARKER_NAME,
    }
    if observed != expected_files:
        raise ChainError(
            "sentinel bootstrap tree drifted: "
            f"missing={sorted(expected_files - observed)}, "
            f"unexpected={sorted(observed - expected_files)}"
        )
    expected_directories = {
        parent.as_posix()
        for relative in inventory
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if observed_directories != expected_directories:
        raise ChainError(
            "sentinel bootstrap directory tree drifted: "
            f"missing={sorted(expected_directories - observed_directories)}, "
            f"unexpected={sorted(observed_directories - expected_directories)}"
        )
    if total_bytes != marker["payload_total_bytes"]:
        raise ChainError("sentinel bootstrap payload byte count drifted")
    python = _require_canonical_path(
        paths.sentinel_bootstrap_python,
        description="sentinel bootstrap Python",
        kind="file",
    )
    if not os.access(python, os.X_OK):
        raise ChainError("sentinel bootstrap Python is not executable")
    probe = (
        "import argparse,contextlib,datetime,fcntl,hashlib,importlib,json,math,"
        "os,pathlib,re,shlex,stat,subprocess,tempfile,time,typing,uuid;"
        f"print('{_BOOTSTRAP_PROBE_TOKEN}')"
    )
    probe_environment = _sanitized_process_environment()
    for name in tuple(probe_environment):
        if name.startswith(("PIP_", "CONDA_")) or name in {
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }:
            probe_environment.pop(name, None)
    probe_environment.update(
        {
            "LD_LIBRARY_PATH": str(root / "lib"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-S", "-c", probe],
            text=True,
            capture_output=True,
            check=False,
            timeout=120.0,
            env=probe_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ChainError(f"sentinel bootstrap Python probe failed: {exc}") from exc
    if result.returncode != 0 or result.stdout.strip() != _BOOTSTRAP_PROBE_TOKEN:
        raise ChainError(
            "sentinel bootstrap Python probe failed: "
            f"exit={result.returncode}, stderr={result.stderr.strip()[:500]!r}"
        )
    return marker


def _publish_sentinel_bootstrap(paths: RecoveryPaths) -> dict[str, Any]:
    """Publish a real-copy, marker-last runtime independent of the live prefix."""

    expected_marker, inventory_payload, source_before = (
        _prospective_sentinel_bootstrap(paths)
    )
    parent = (
        paths.recovery_root / SENTINEL_BOOTSTRAP_ROOT_NAME
    )
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    _require_canonical_path(
        parent, description="sentinel bootstrap parent", kind="directory"
    )
    lock = parent / f".{CHAIN_NAMESPACE}.capture.lock"
    with _exclusive_lock(lock, description="sentinel bootstrap capture"):
        if (
            paths.sentinel_bootstrap_root.exists()
            or paths.sentinel_bootstrap_root.is_symlink()
        ):
            return _verify_sentinel_bootstrap(
                paths, expected_marker=expected_marker
            )
        temporary_prefix = f".{CHAIN_NAMESPACE}.bootstrap."
        for stale in sorted(parent.iterdir()):
            if stale.name.startswith(temporary_prefix):
                _discard_stale_bootstrap_temporary(stale, parent=parent)
        temporary = parent / (
            f".{CHAIN_NAMESPACE}.bootstrap.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            temporary.mkdir(mode=0o750)
            source_inodes = {
                (metadata.st_dev, metadata.st_ino)
                for metadata in (
                    Path(str(record["source_path"])).stat()
                    for record in source_before
                )
            }
            destination_inodes: set[tuple[int, int]] = set()
            for record in source_before:
                source = Path(str(record["source_path"]))
                destination = temporary / str(record["relative_path"])
                _copy_bootstrap_file(
                    source,
                    destination,
                    mode=0o550 if record["executable"] else 0o440,
                )
                destination_metadata = destination.stat()
                destination_inode = (
                    destination_metadata.st_dev,
                    destination_metadata.st_ino,
                )
                if (
                    destination_inode in destination_inodes
                    or destination_inode in source_inodes
                    or destination_metadata.st_nlink != 1
                ):
                    raise ChainError(
                        "sentinel bootstrap copy shares an inode with another file"
                    )
                destination_inodes.add(destination_inode)
                if (
                    destination_metadata.st_size != record["size"]
                    or _sha256(destination) != record["sha256"]
                ):
                    raise ChainError(
                        f"sentinel bootstrap copy verification failed: {destination}"
                    )
            source_after, stdlib_name_after = _bootstrap_source_records(paths)
            if source_after != source_before:
                raise ChainError(
                    "sentinel bootstrap source changed during capture"
                )
            if (
                expected_marker
                != _bootstrap_marker(
                    paths, source_after, stdlib_name=stdlib_name_after
                )
            ):
                raise ChainError(
                    "sentinel bootstrap source contract changed during capture"
                )

            inventory = temporary / SENTINEL_BOOTSTRAP_INVENTORY_NAME
            inventory.write_bytes(inventory_payload)
            os.chmod(inventory, 0o440)
            with inventory.open("rb") as handle:
                os.fsync(handle.fileno())
            # The completion record is deliberately the final content write.
            marker = temporary / SENTINEL_BOOTSTRAP_MARKER_NAME
            descriptor = os.open(
                marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical_json(expected_marker))
                handle.flush()
                os.fsync(handle.fileno())
            for directory, _, _ in os.walk(temporary, topdown=False):
                directory_path = Path(directory)
                os.chmod(directory_path, 0o550)
                _fsync_directory(directory_path)
            _fsync_directory(temporary)
            os.rename(temporary, paths.sentinel_bootstrap_root)
            _fsync_directory(parent)
        finally:
            if temporary.exists():
                _discard_stale_bootstrap_temporary(temporary, parent=parent)
        return _verify_sentinel_bootstrap(
            paths, expected_marker=expected_marker
        )


def _sentinel_bootstrap_contract(paths: RecoveryPaths) -> dict[str, Any]:
    if paths.sentinel_bootstrap_marker.exists():
        return _verify_sentinel_bootstrap(paths)
    marker, _, _ = _prospective_sentinel_bootstrap(paths)
    return marker


def _run_checked(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            _trusted_command_argv(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            env=_sanitized_process_environment(),
        )
    except OSError as exc:
        raise ChainError(f"cannot execute {argv[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise ChainError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}: "
            f"{proc.stderr.strip()[:1000]}"
        )
    return proc.stdout.strip()


def _tagged_file_bytes(repository: Path, commit: str, relative: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ChainError(f"invalid tagged Git commit: {commit!r}")
    try:
        proc = subprocess.run(
            ["/usr/bin/git", "show", f"{commit}:{relative}"],
            cwd=repository,
            capture_output=True,
            check=False,
            env=_sanitized_process_environment(),
        )
    except OSError as exc:
        raise ChainError(f"cannot read tagged recovery tool {relative}: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise ChainError(
            f"tagged recovery tool is missing: {relative}: "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )
    return bytes(proc.stdout)


def _run_checked_bytes(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
) -> bytes:
    """Run one read-only command without decoding or stripping its output."""

    try:
        proc = subprocess.run(
            _trusted_command_argv(argv),
            cwd=cwd,
            capture_output=True,
            check=False,
            env=_sanitized_process_environment(),
        )
    except OSError as exc:
        raise ChainError(f"cannot execute {argv[0]}: {exc}") from exc
    if proc.returncode != 0:
        raise ChainError(
            f"command failed ({proc.returncode}): {shlex.join(argv)}: "
            f"{proc.stderr.decode(errors='replace').strip()[:1000]}"
        )
    return bytes(proc.stdout)


def _tagged_source_tree_sha256(repository: Path, commit: str) -> str:
    """Reproduce ``schema5_control.sha256_tree`` from the exact Git tree."""

    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ChainError(f"invalid tagged Git commit: {commit!r}")
    tree = _run_checked_bytes(
        [
            "git",
            "ls-tree",
            "-rz",
            "--full-tree",
            "-r",
            commit,
        ],
        cwd=repository,
    )
    ignored_names = {".git", ".pytest_cache", "__pycache__"}
    entries: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for raw_record in tree.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_relative = raw_record.split(b"\t", 1)
            mode, object_type, raw_object = metadata.split(b" ")
            relative = raw_relative.decode("utf-8")
            object_id = raw_object.decode("ascii")
        except (UnicodeError, ValueError) as exc:
            raise ChainError(
                "tagged source tree contains a malformed Git entry"
            ) from exc
        relative_path = Path(relative)
        if (
            not relative
            or relative.startswith("/")
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise ChainError(
                "tagged source tree contains a duplicate or unsafe path"
            )
        seen.add(relative)
        if (
            any(part in ignored_names for part in relative_path.parts)
            or relative_path.suffix == ".pyc"
        ):
            continue
        if (
            object_type != b"blob"
            or re.fullmatch(r"[0-9a-f]{40}", object_id) is None
            or mode not in {b"100644", b"100755", b"120000"}
        ):
            raise ChainError(
                f"tagged source tree has unsupported entry {relative!r}"
            )
        blob = _run_checked_bytes(
            ["git", "cat-file", "blob", object_id],
            cwd=repository,
        )
        if mode == b"120000":
            try:
                target = blob.decode("utf-8")
            except UnicodeError as exc:
                raise ChainError(
                    f"tagged source symlink target is not UTF-8: {relative}"
                ) from exc
            payload = ("SYMLINK\0" + target).encode("utf-8")
        else:
            payload = blob
        entries.append((relative, payload))
    entries.sort(key=lambda row: row[0])
    digest = hashlib.sha256()
    for relative, payload in entries:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _protected_capacity_release_source_binding(
    paths: RecoveryPaths,
    *,
    git_identity: Mapping[str, str],
) -> dict[str, str]:
    """Derive capacity provenance independently from exact tagged bytes."""

    try:
        commit = str(git_identity["git_commit"])
        tag_object = str(git_identity["tag_object"])
    except KeyError as exc:
        raise ChainError(
            "release Git identity lacks commit/tag-object authority"
        ) from exc
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", tag_object) is None
    ):
        raise ChainError("release Git identity is malformed")
    dispatcher = _tagged_file_bytes(
        paths.repository,
        commit,
        "slurm/dispatch_sweeps.py",
    )
    qualification_runner = _tagged_file_bytes(
        paths.repository,
        commit,
        "scripts/run_schema5_throughput_qualification.py",
    )
    return {
        "source_tree_sha256": _tagged_source_tree_sha256(
            paths.repository, commit
        ),
        "dispatcher_source_sha256": hashlib.sha256(
            dispatcher
        ).hexdigest(),
        "qualification_runner_source_sha256": hashlib.sha256(
            qualification_runner
        ).hexdigest(),
    }


def _reject_git_replace_refs(repository: Path) -> None:
    """Reject local object substitution even though replacement use is disabled."""

    replacement_refs = _run_checked(
        ["git", "for-each-ref", "--format=%(refname)", "refs/replace"],
        cwd=repository,
    ).splitlines()
    if replacement_refs:
        raise ChainError(
            "exact release checkout contains forbidden Git replacement refs: "
            + ", ".join(replacement_refs[:10])
        )


def verify_release_tag(repository: Path) -> dict[str, str]:
    if not repository.is_dir() or not (repository / ".git").exists():
        raise ChainError(f"source repository is not a Git checkout: {repository}")
    _reject_git_replace_refs(repository)
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
    tag_object = _run_checked(
        ["git", "rev-parse", "--verify", f"refs/tags/{RELEASE_TAG}"],
        cwd=repository,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", tag_object):
        raise ChainError("release annotated-tag object identity is invalid")
    return {
        "release_tag": RELEASE_TAG,
        "git_commit": commit,
        "tag_object": tag_object,
    }


def _verify_exact_tag_checkout(
    checkout: Path,
    *,
    expected_commit: str,
    expected_tag_object: str,
) -> None:
    checkout = _require_canonical_path(
        checkout, description="prerequisite verifier checkout", kind="directory"
    )
    if not (checkout / ".git").exists():
        raise ChainError(f"prerequisite verifier checkout is not Git-backed: {checkout}")
    _reject_git_replace_refs(checkout)
    tag_ref = f"refs/tags/{RELEASE_TAG}"
    if (
        _run_checked(["git", "cat-file", "-t", tag_ref], cwd=checkout) != "tag"
        or _run_checked(["git", "rev-parse", "--verify", tag_ref], cwd=checkout)
        != expected_tag_object
        or _run_checked(
            ["git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
            cwd=checkout,
        )
        != expected_commit
        or _run_checked(["git", "rev-parse", "--verify", "HEAD"], cwd=checkout)
        != expected_commit
        or _run_checked(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=checkout,
        )
    ):
        raise ChainError(
            "prerequisite verification requires the exact clean release-tagged checkout"
        )


def _prerequisite_code_records(
    repository: Path, *, commit: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for git_path in PREREQUISITE_CODE_GIT_PATHS:
        payload = _tagged_file_bytes(repository, commit, git_path)
        records.append(
            {
                "git_path": git_path,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
        )
    return records


def _run_json_verifier(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout: float,
    description: str,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            env=dict(environment),
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ChainError(f"{description} could not run: {exc}") from exc
    if completed.returncode != 0:
        raise ChainError(
            f"{description} failed ({completed.returncode}): "
            f"{completed.stderr.strip()[:1000]}"
        )
    try:
        report = json.loads(completed.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ChainError(f"{description} returned invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ChainError(f"{description} did not return one JSON object")
    return report


def _sanitized_python_environment(*, library: Path) -> dict[str, str]:
    """Return a deterministic, offline Python/pip/Conda subprocess environment."""

    environment = _sanitized_process_environment()
    for name in tuple(environment):
        if name.startswith(("PIP_", "CONDA_")) or name in {
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }:
            environment.pop(name, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "LD_LIBRARY_PATH": str(library),
            # These explicit null/off switches make user and site configuration
            # irrelevant even when a child process invokes pip or Conda.
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CONDARC": os.devnull,
            "CONDA_NO_PLUGINS": "true",
        }
    )
    return environment


def _invoke_tagged_prerequisite_verifiers(
    paths: RecoveryPaths,
    *,
    checkout: Path,
    expected_commit: str,
    expected_tag_object: str,
    verifier_python: Path | None = None,
    verifier_library: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run both full sealed verifiers from one exact clean tagged checkout."""

    _verify_exact_tag_checkout(
        checkout,
        expected_commit=expected_commit,
        expected_tag_object=expected_tag_object,
    )
    code_records = {
        record["git_path"]: record
        for record in _prerequisite_code_records(
            checkout, commit=expected_commit
        )
    }
    for git_path, record in code_records.items():
        source = _require_canonical_path(
            checkout / git_path,
            description=f"tagged prerequisite code {git_path}",
            kind="file",
        )
        if (
            _sha256(source) != record["sha256"]
            or source.stat().st_size != record["size"]
        ):
            raise ChainError(f"tagged prerequisite code drifted: {git_path}")
    python = _require_canonical_path(
        paths.dev_python if verifier_python is None else verifier_python,
        description="prerequisite verifier Python",
        kind="file",
    )
    library = _require_canonical_path(
        (
            paths.source_harness / "lib"
            if verifier_library is None
            else verifier_library
        ),
        description="prerequisite verifier library",
        kind="directory",
    )
    if not os.access(python, os.X_OK):
        raise ChainError(f"prerequisite verifier Python is not executable: {python}")
    environment = _sanitized_python_environment(library=library)
    pilot = _run_json_verifier(
        [
            str(python),
            "-I",
            str(checkout / MATERIALIZATION_PILOT_GIT_PATH),
            "verify",
            "--pilot-root",
            str(paths.materialization_pilot_root),
        ],
        environment=environment,
        timeout=18_000.0,
        description="tagged materialization-pilot verifier",
    )
    canary = _run_json_verifier(
        [
            str(python),
            "-I",
            str(checkout / SLURM_CANARY_GIT_PATH),
            "--canary-root",
            str(paths.slurm_canary_root),
            "--verify",
        ],
        environment=environment,
        timeout=600.0,
        description="tagged Slurm-canary verifier",
    )
    return {"materialization_pilot": pilot, "slurm_canary": canary}


def _sealed_prerequisite_marker(
    *,
    root: Path,
    marker_name: str,
    description: str,
) -> dict[str, Any]:
    root = _require_canonical_path(root, description=description, kind="directory")
    marker = _require_canonical_path(
        root / marker_name,
        description=f"{description} completion marker",
        kind="file",
    )
    if stat.S_IMODE(marker.stat().st_mode) & 0o222:
        raise ChainError(f"{description} completion marker remains writable")
    return {
        "root": str(root),
        "marker": str(marker),
        "marker_sha256": _sha256(marker),
        "marker_size": marker.stat().st_size,
    }


def _read_sealed_marker(
    path: Path, *, description: str
) -> tuple[dict[str, Any], bytes]:
    """Read one canonical, immutable marker while rejecting ambiguous JSON."""

    marker = _require_canonical_path(
        path, description=description, kind="file"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker, flags)
    except OSError as exc:
        raise ChainError(f"cannot open sealed {description}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if stat.S_IMODE(before.st_mode) & 0o222:
            raise ChainError(f"{description} remains writable")
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ChainError(
                f"{description} is not one unique regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        first = b"".join(chunks)
        os.lseek(descriptor, 0, os.SEEK_SET)
        repeated_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            repeated_chunks.append(chunk)
        raw = b"".join(repeated_chunks)
        after = os.fstat(descriptor)
        current = marker.stat(follow_symlinks=False)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
            "st_mode",
        )
        if first != raw or any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise ChainError(f"{description} changed during sealed read")
    finally:
        os.close(descriptor)

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ChainError(
                    f"{description} contains duplicate JSON key {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ChainError(
                    f"{description} contains non-finite JSON value {token}"
                )
            ),
        )
    except ChainError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ChainError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ChainError(f"{description} is not one JSON object")
    return value, raw


def _require_marker_identity(
    marker: Mapping[str, Any],
    *,
    identity_field: str,
    description: str,
) -> None:
    identity = dict(marker)
    observed = identity.pop(identity_field, None)
    if (
        not isinstance(observed, str)
        or _SHA256.fullmatch(observed) is None
        or observed != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError(f"{description} self-hash identity is invalid")


def _require_release_marker_binding(
    marker: Mapping[str, Any],
    *,
    git_identity: Mapping[str, str],
    protocol: str,
    description: str,
    schema_version: int = 1,
) -> None:
    if (
        marker.get("schema_version") != schema_version
        or marker.get("protocol") != protocol
        or marker.get("passed") is not True
        or marker.get("release_id") != RELEASE_ID
        or marker.get("release_tag") != RELEASE_TAG
        or marker.get("release_git_commit") != git_identity["git_commit"]
        or marker.get("release_tag_object") != git_identity["tag_object"]
        or marker.get("chain_namespace") != CHAIN_NAMESPACE
    ):
        raise ChainError(
            f"{description} does not bind the exact release and chain"
        )


def _integer_at_least(
    value: object, minimum: int, *, description: str
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ChainError(f"{description} must be at least {minimum}")
    return value


def _validate_protected_placement_rows(
    value: object,
    *,
    role: str,
    preempt_type: str,
) -> dict[str, int]:
    fields = (
        {
            "partition",
            "qos",
            "partition_preempt_mode",
            "qos_preempt_mode",
            "active_serving_gpus",
            "warm_headroom_gpus",
        }
        if role == "server"
        else {
            "partition",
            "qos",
            "partition_preempt_mode",
            "qos_preempt_mode",
            "slots",
            "cpus",
            "memory_mib",
            "reserve_jobs",
            "submit_headroom",
        }
    )
    if not isinstance(value, list) or not value:
        raise ChainError(f"protected scientific {role} placements are absent")
    identities: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != fields:
            raise ChainError(
                f"protected scientific {role} placement {index} fields drifted"
            )
        partition = row.get("partition")
        qos = row.get("qos")
        if (
            not isinstance(partition, str)
            or _SAFE_NAME.fullmatch(partition) is None
            or not isinstance(qos, str)
            or _SAFE_NAME.fullmatch(qos) is None
            or row.get("partition_preempt_mode") != "OFF"
            or (
                row.get("qos_preempt_mode") != "OFF"
                and not (
                    preempt_type == "preempt/partition_prio"
                    and row.get("qos_preempt_mode") == "cluster"
                )
            )
        ):
            raise ChainError(
                f"protected scientific {role} placement identity/policy is invalid"
            )
        identity = (partition, qos)
        if identity in identities:
            raise ChainError(
                f"protected scientific {role} placement is duplicated"
            )
        identities.add(identity)
        normalized.append(dict(row))
    normalized.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["qos"]),
            _canonical_json(row),
        )
    )
    if value != normalized:
        raise ChainError(
            f"protected scientific {role} placements are not canonically sorted"
        )
    capacity_fields = (
        ("active_serving_gpus", "warm_headroom_gpus")
        if role == "server"
        else ("slots", "cpus", "memory_mib", "reserve_jobs", "submit_headroom")
    )
    totals = {field: 0 for field in capacity_fields}
    for row in normalized:
        for field in capacity_fields:
            observed = row.get(field)
            if (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed < 0
            ):
                raise ChainError(
                    f"protected scientific {role} {field} is invalid"
                )
            totals[field] += observed
    if role == "client":
        directly_usable = [
            row
            for row in normalized
            if row["slots"] >= 384
            and row["cpus"] >= 384
            and row["memory_mib"] >= 1_572_864
            and row["reserve_jobs"] >= 64
            and row["submit_headroom"] >= 448
            and row["cpus"] >= row["slots"]
            and row["memory_mib"] >= row["slots"] * 4096
            and row["submit_headroom"]
            >= row["slots"] + row["reserve_jobs"]
        ]
        if len(directly_usable) != 1:
            raise ChainError(
                "protected capacity must have exactly one directly usable "
                "384-cell + 64-reserve client placement"
            )
    return totals


def _validate_scientific_qos_contracts(
    value: object,
    *,
    server_qos: set[str],
    client_qos: set[str],
    expected_running_jobs: int,
) -> None:
    """Prove exact QOS job-count and walltime support for every placement."""

    fields = {
        "qos",
        "max_wall_seconds",
        "max_jobs_per_user",
        "max_submit_jobs_per_user",
        "required_wall_seconds",
        "required_running_jobs",
        "required_submit_jobs",
    }
    if not isinstance(value, list) or not value:
        raise ChainError(
            "protected capacity has no scientific QOS contracts"
        )
    prior = ""
    by_qos: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ChainError(
                f"protected scientific QOS contract {index} fields drifted"
            )
        qos = row.get("qos")
        if (
            not isinstance(qos, str)
            or _SAFE_NAME.fullmatch(qos) is None
            or qos <= prior
        ):
            raise ChainError(
                "protected scientific QOS contracts are duplicated or "
                "not canonically sorted"
            )
        prior = qos
        limits: dict[str, int | None] = {}
        for field in (
            "max_wall_seconds",
            "max_jobs_per_user",
            "max_submit_jobs_per_user",
        ):
            observed = row.get(field)
            if observed is not None and (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed < 1
            ):
                raise ChainError(
                    f"protected scientific QOS {qos} {field} is invalid"
                )
            limits[field] = observed
        required_wall = row.get("required_wall_seconds")
        required_running = row.get("required_running_jobs")
        required_submit = row.get("required_submit_jobs")
        if (
            not isinstance(required_wall, int)
            or isinstance(required_wall, bool)
            or required_wall < PROTECTED_CLIENT_WALL_SECONDS
            or not isinstance(required_running, int)
            or isinstance(required_running, bool)
            or required_running < 0
            or not isinstance(required_submit, int)
            or isinstance(required_submit, bool)
            or required_submit < 1
            or (
                limits["max_wall_seconds"] is not None
                and limits["max_wall_seconds"] < required_wall
            )
            or (
                limits["max_jobs_per_user"] is not None
                and limits["max_jobs_per_user"] < required_running
            )
            or (
                limits["max_submit_jobs_per_user"] is not None
                and limits["max_submit_jobs_per_user"]
                < required_submit
            )
        ):
            raise ChainError(
                f"protected scientific QOS {qos} cannot sustain its "
                "registered load"
            )
        by_qos[qos] = row
    if (
        set(by_qos) != server_qos | client_qos
        or sum(
            int(row["required_running_jobs"])
            for row in by_qos.values()
        )
        != expected_running_jobs
        or sum(
            int(row["required_submit_jobs"])
            for row in by_qos.values()
        )
        != 448
        or max(
            int(row["required_wall_seconds"])
            for row in by_qos.values()
        )
        != PROTECTED_MINIMUM_SCIENTIFIC_WALL_SECONDS
        or any(
            int(by_qos[qos]["required_wall_seconds"])
            != PROTECTED_MINIMUM_SCIENTIFIC_WALL_SECONDS
            for qos in server_qos
        )
        or any(
            int(by_qos[qos]["required_wall_seconds"])
            < PROTECTED_CLIENT_WALL_SECONDS
            for qos in client_qos
        )
    ):
        raise ChainError(
            "protected scientific QOS contracts do not prove the exact running "
            "jobs, 448 submitted jobs, 24-hour serving, and 12-hour cells"
        )


def _validate_protected_capacity_marker(
    paths: RecoveryPaths, *, git_identity: Mapping[str, str]
) -> tuple[dict[str, Any], bytes]:
    source_binding = _protected_capacity_release_source_binding(
        paths,
        git_identity=git_identity,
    )
    try:
        protected_capacity.load_contract(
            paths.protected_capacity_marker,
            expected_release_git_commit=str(git_identity["git_commit"]),
            expected_release_tag_object=str(git_identity["tag_object"]),
            expected_source_tree_sha256=source_binding[
                "source_tree_sha256"
            ],
            expected_dispatcher_source_sha256=source_binding[
                "dispatcher_source_sha256"
            ],
            expected_qualification_runner_source_sha256=source_binding[
                "qualification_runner_source_sha256"
            ],
        )
    except (
        KeyError,
        OSError,
        protected_capacity.ProtectedCapacityError,
    ) as exc:
        raise ChainError(
            f"protected-capacity completion marker is invalid: {exc}"
        ) from exc
    marker, raw = _read_sealed_marker(
        paths.protected_capacity_marker,
        description="protected-capacity completion marker",
    )
    return marker, raw

    # Kept below only as sealed-history context for the superseded v2 protocol.
    # The v4 authority is validated by the shared runtime contract above.
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "active_gpus",
        "warm_headroom_gpus",
        "cell_ceiling",
        "reserve_jobs",
        "submit_headroom",
        "cpu",
        "memory_mib",
        "preempt_type",
        "capacity_source",
        "scheduler_cluster",
        "scheduler_account",
        "scheduler_user",
        "scheduler_max_jobs",
        "scheduler_max_submit_jobs",
        "running_scientific_jobs",
        "minimum_scientific_wall_seconds",
        "scientific_qos_contracts",
        "partition_cpus",
        "partition_memory_mib",
        "partition_gpus",
        "fleet_contract_sha256",
        "active_fleet_topology_sha256",
        "scientific_server_preempt_mode",
        "scientific_client_preempt_mode",
        "scientific_server_placements",
        "scientific_client_placements",
        "scheduler_evidence_id",
        "scheduler_evidence_sha256",
        "canary_id",
        "canary_evidence_sha256",
        "squeue_complete",
        "sacct_complete",
        "marker_id",
    }
    if set(marker) != required:
        raise ChainError("protected-capacity marker fields drifted")
    _require_release_marker_binding(
        marker,
        git_identity=git_identity,
        protocol=PROTECTED_CAPACITY_PROTOCOL,
        description="protected-capacity marker",
        schema_version=2,
    )
    _require_marker_identity(
        marker,
        identity_field="marker_id",
        description="protected-capacity marker",
    )
    active_gpus = _integer_at_least(
        marker.get("active_gpus"),
        24,
        description="protected active GPU capacity",
    )
    warm_headroom_gpus = _integer_at_least(
        marker.get("warm_headroom_gpus"),
        4,
        description="protected warm GPU headroom",
    )
    cell_ceiling = _integer_at_least(
        marker.get("cell_ceiling"),
        384,
        description="protected cell ceiling",
    )
    reserve_jobs = _integer_at_least(
        marker.get("reserve_jobs"),
        64,
        description="protected reserve jobs",
    )
    submit_headroom = _integer_at_least(
        marker.get("submit_headroom"),
        448,
        description="protected submit headroom",
    )
    cpu = _integer_at_least(
        marker.get("cpu"), 384, description="protected client CPUs"
    )
    memory_mib = _integer_at_least(
        marker.get("memory_mib"),
        1_572_864,
        description="protected client memory MiB",
    )
    preempt_type = marker.get("preempt_type")
    if preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
        raise ChainError("protected capacity has unsupported PreemptType")
    servers = _validate_protected_placement_rows(
        marker.get("scientific_server_placements"),
        role="server",
        preempt_type=str(preempt_type),
    )
    clients = _validate_protected_placement_rows(
        marker.get("scientific_client_placements"),
        role="client",
        preempt_type=str(preempt_type),
    )
    server_qos = {
        str(row["qos"])
        for row in marker["scientific_server_placements"]
    }
    client_qos = {
        str(row["qos"])
        for row in marker["scientific_client_placements"]
    }
    _validate_scientific_qos_contracts(
        marker.get("scientific_qos_contracts"),
        server_qos=server_qos,
        client_qos=client_qos,
        expected_running_jobs=PROTECTED_RUNNING_SCIENTIFIC_JOBS,
    )
    scheduler_max_jobs = marker.get("scheduler_max_jobs")
    source_fields = (
        "scheduler_evidence_id",
        "scheduler_evidence_sha256",
        "canary_id",
        "canary_evidence_sha256",
        "fleet_contract_sha256",
        "active_fleet_topology_sha256",
    )
    if (
        servers["active_serving_gpus"] != active_gpus
        or servers["warm_headroom_gpus"] != warm_headroom_gpus
        or clients["slots"] != cell_ceiling
        or clients["reserve_jobs"] != reserve_jobs
        or clients["submit_headroom"] != submit_headroom
        or clients["cpus"] != cpu
        or clients["memory_mib"] != memory_mib
        or marker.get("capacity_source") != PROTECTED_CAPACITY_SOURCE
        or marker.get("running_scientific_jobs")
        != PROTECTED_RUNNING_SCIENTIFIC_JOBS
        or marker.get("minimum_scientific_wall_seconds")
        != PROTECTED_MINIMUM_SCIENTIFIC_WALL_SECONDS
        or (
            scheduler_max_jobs is not None
            and (
                not isinstance(scheduler_max_jobs, int)
                or isinstance(scheduler_max_jobs, bool)
                or scheduler_max_jobs
                < PROTECTED_RUNNING_SCIENTIFIC_JOBS
            )
        )
        or any(
            not isinstance(marker.get(field), str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                str(marker[field]),
            )
            is None
            for field in (
                "scheduler_cluster",
                "scheduler_account",
                "scheduler_user",
            )
        )
        or any(
            not isinstance(marker.get(field), int)
            or isinstance(marker.get(field), bool)
            or int(marker[field]) < minimum
            for field, minimum in (
                ("scheduler_max_submit_jobs", 448),
                ("partition_cpus", 384),
                ("partition_memory_mib", 1_572_864),
                ("partition_gpus", 0),
            )
        )
        or any(
            not isinstance(marker.get(field), str)
            or _SHA256.fullmatch(str(marker[field])) is None
            for field in source_fields
        )
        or marker.get("scientific_server_preempt_mode") != "OFF"
        or marker.get("scientific_client_preempt_mode") != "OFF"
        or marker.get("squeue_complete") is not True
        or marker.get("sacct_complete") is not True
    ):
        raise ChainError(
            "protected capacity does not prove non-preemptible scientific "
            "placement and complete squeue+sacct truth"
        )
    return marker, raw


def _validate_durable_git_release_marker(
    paths: RecoveryPaths, *, git_identity: Mapping[str, str]
) -> tuple[dict[str, Any], bytes]:
    marker, raw = _read_sealed_marker(
        paths.durable_git_release_marker,
        description="durable Git release completion marker",
    )
    try:
        binding = durable_git.marker_binding(
            paths.durable_git_release_marker
        )
    except durable_git.DurableGitReleaseError as exc:
        raise ChainError(str(exc)) from exc
    _require_release_marker_binding(
        marker,
        git_identity=git_identity,
        protocol=DURABLE_GIT_RELEASE_PROTOCOL,
        description="durable Git release marker",
    )
    _require_marker_identity(
        marker,
        identity_field="marker_id",
        description="durable Git release marker",
    )
    if (
        binding["release_git_commit"] != git_identity["git_commit"]
        or binding["release_tag_object"] != git_identity["tag_object"]
        or marker.get("clean_checkout") is not True
        or marker.get("annotated_tag") is not True
        or marker.get("remote_query_read_only") is not True
    ):
        raise ChainError(
            "durable Git release marker does not prove the exact pushed release"
        )
    return marker, raw


def _current_canary_tree_inventory(
    root: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    root = _require_canonical_path(
        root,
        description="superseded r2 partial canary",
        kind="directory",
    )
    rows: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    candidates = [
        root,
        *sorted(
            root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        ),
    ]
    for candidate in candidates:
        relative = (
            "."
            if candidate == root
            else candidate.relative_to(root).as_posix()
        )
        metadata = os.lstat(candidate)
        common = {
            "relative_path": relative,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "nlink": metadata.st_nlink,
        }
        if stat.S_ISLNK(metadata.st_mode):
            raise ChainError(
                f"superseded r2 partial canary contains a symlink: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            rows.append(
                common
                | {"type": "directory", "size": 0, "sha256": None}
            )
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ChainError(
                f"superseded r2 partial canary contains an unsafe entry: {relative}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            raise ChainError(
                f"cannot open superseded r2 canary artifact {relative}: {exc}"
            ) from exc
        digest = hashlib.sha256()
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
            ):
                raise ChainError(
                    f"superseded r2 canary artifact is unsafe: {relative}"
                )
            for block in iter(
                lambda: os.read(descriptor, 1024 * 1024),
                b"",
            ):
                digest.update(block)
            after = os.fstat(descriptor)
            current = os.lstat(candidate)
        finally:
            os.close(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise ChainError(
                f"superseded r2 canary artifact changed while hashing: {relative}"
            )
        rows.append(
            common
            | {
                "type": "file",
                "size": before.st_size,
                "sha256": digest.hexdigest(),
            }
        )
        file_count += 1
        total_bytes += before.st_size
    return rows, file_count, total_bytes


def _read_canary_inventory(path: Path) -> list[dict[str, Any]]:
    path = _require_canonical_path(
        path,
        description="superseded r2 canary sealed inventory",
        kind="file",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise ChainError(
                "superseded r2 canary sealed inventory is unsafe"
            )
        blocks: list[bytes] = []
        for block in iter(
            lambda: os.read(descriptor, 1024 * 1024),
            b"",
        ):
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(descriptor)
        current = os.lstat(path)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(after, field) != getattr(current, field)
        for field in stable_fields
    ):
        raise ChainError(
            "superseded r2 canary sealed inventory changed while reading"
        )
    if not raw or not raw.endswith(b"\n"):
        raise ChainError(
            "superseded r2 canary sealed inventory is incomplete"
        )

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ChainError(
                    "superseded r2 canary inventory has duplicate keys"
                )
            value[key] = item
        return value

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines(keepends=True):
        try:
            value = json.loads(
                line,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ChainError(
                        "superseded r2 canary inventory contains "
                        f"non-finite value {token}"
                    )
                ),
            )
        except ChainError:
            raise
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ChainError(
                f"superseded r2 canary sealed inventory is invalid: {exc}"
            ) from exc
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "relative_path",
                "device",
                "inode",
                "mode",
                "nlink",
                "type",
                "size",
                "sha256",
            }
            or line != _compact_canonical_json(value)
        ):
            raise ChainError(
                "superseded r2 canary sealed inventory row drifted"
            )
        rows.append(value)
    return rows


def _validate_superseded_r2_canary_failure_marker(
    paths: RecoveryPaths,
) -> tuple[dict[str, Any], bytes]:
    marker_path = paths.superseded_r2_canary_failure_marker
    marker, raw = _read_sealed_marker(
        marker_path,
        description="superseded r2 canary failure seal",
    )
    identity = dict(marker)
    seal_id = identity.pop("seal_id", None)
    release = marker.get("release")
    code_identity = (
        release.get("code_identity")
        if isinstance(release, dict)
        else None
    )
    known = marker.get("known_scheduler_identity")
    expected_tree = (
        paths.recovery_root / "slurm_canaries" / "schema5-v1.2-r2"
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("protocol")
        != SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("classification")
        != "deterministic_canary_failure_sealed_fail_closed"
        or marker.get("error_classification")
        != "deterministic_missing_subprocess_capture"
        or marker.get("retry_in_place") is not False
        or marker.get("no_active_matching_jobs_preseal") is not True
        or marker.get("no_active_matching_jobs_postseal") is not True
        or marker.get("mutation_claim")
        != "bound_to_preexisting_evidence_not_inferred_from_absence"
        or marker.get("tree") != str(expected_tree)
        or marker.get("file_count")
        != SUPERSEDED_R2_CANARY_FAILURE_FILE_COUNT
        or marker.get("total_bytes")
        != SUPERSEDED_R2_CANARY_FAILURE_TOTAL_BYTES
        or seal_id != SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID
        or seal_id != _sha256_bytes(_compact_canonical_json(identity))
        or not isinstance(release, dict)
        or release.get("release_tag") != SUPERSEDED_R2_RELEASE_TAG
        or release.get("release_git_commit")
        != SUPERSEDED_R2_RELEASE_COMMIT
        or release.get("release_tag_object")
        != SUPERSEDED_R2_RELEASE_TAG_OBJECT
        or not isinstance(code_identity, dict)
        or code_identity.get("release_tag")
        != SUPERSEDED_R2_RELEASE_TAG
        or code_identity.get("release_git_commit")
        != SUPERSEDED_R2_RELEASE_COMMIT
        or code_identity.get("release_tag_object")
        != SUPERSEDED_R2_RELEASE_TAG_OBJECT
        or code_identity.get("protocol")
        != "schema5-v1.2-r2-slurm-canary-code-v3"
        or not isinstance(known, dict)
        or known.get("job_ids") != [SUPERSEDED_R2_CANARY_JOB_ID]
    ):
        raise ChainError(
            "superseded r2 canary failure does not bind the exact "
            "deterministic prelaunch incident"
        )

    evidence_root = marker_path.parent
    artifact_contract = {
        "intent": (
            "CANARY_FAILURE_SEAL_INTENT.json",
            "intent_sha256",
        ),
        "preseal_inventory": (
            "CANARY_FAILURE_PRESEAL_INVENTORY.jsonl",
            "preseal_inventory_sha256",
        ),
        "sealed_inventory": (
            "CANARY_FAILURE_SEALED_INVENTORY.jsonl",
            "sealed_inventory_sha256",
        ),
        "scheduler_pre": (
            "CANARY_FAILURE_SCHEDULER_PRE.json",
            "scheduler_pre_sha256",
        ),
        "scheduler_post": (
            "CANARY_FAILURE_SCHEDULER_POST.json",
            "scheduler_post_sha256",
        ),
    }
    expected_evidence_paths = {marker_path}
    for field, (filename, digest_field) in artifact_contract.items():
        expected = evidence_root / filename
        expected_evidence_paths.add(expected)
        path = _require_canonical_path(
            expected,
            description=f"superseded r2 canary {field}",
            kind="file",
        )
        metadata = path.stat(follow_symlinks=False)
        if (
            marker.get(field) != str(expected)
            or marker.get(digest_field) != _sha256(path)
            or stat.S_IMODE(metadata.st_mode) & 0o222
            or metadata.st_nlink != 1
        ):
            raise ChainError(
                f"superseded r2 canary {field} evidence drifted"
            )
    observed_evidence_paths = set(evidence_root.iterdir())
    if observed_evidence_paths != expected_evidence_paths:
        raise ChainError(
            "superseded r2 canary failure evidence contains unexpected "
            "or missing artifacts"
        )

    sealed_inventory = _read_canary_inventory(
        Path(str(marker["sealed_inventory"]))
    )
    observed, observed_count, observed_bytes = (
        _current_canary_tree_inventory(expected_tree)
    )
    repeated, repeated_count, repeated_bytes = (
        _current_canary_tree_inventory(expected_tree)
    )
    if (
        observed != repeated
        or observed_count != repeated_count
        or observed_bytes != repeated_bytes
        or observed != sealed_inventory
        or observed_count != marker.get("file_count")
        or observed_bytes != marker.get("total_bytes")
    ):
        raise ChainError(
            "superseded r2 partial canary differs from its sealed inventory"
        )

    for phase in ("scheduler_pre", "scheduler_post"):
        scheduler, _ = _read_sealed_marker(
            Path(str(marker[phase])),
            description=f"superseded r2 canary {phase}",
        )
        rows = scheduler.get("matching_sacct_rows")
        if (
            scheduler.get("protocol")
            != "schema5-v1.2-r2-partial-canary-scheduler-evidence-v1"
            or scheduler.get("complete_squeue_truth") is not True
            or scheduler.get("complete_sacct_truth") is not True
            or scheduler.get("no_active_matching_jobs") is not True
            or scheduler.get("matching_squeue_rows") != []
            or not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("job_id")
            != SUPERSEDED_R2_CANARY_JOB_ID
            or rows[0].get("state") != "CANCELLED"
        ):
            raise ChainError(
                f"superseded r2 canary {phase} scheduler truth drifted"
            )

    for root, description in (
        (expected_tree, "superseded r2 partial canary"),
        (evidence_root, "superseded r2 canary failure evidence"),
    ):
        root = _require_canonical_path(
            root,
            description=description,
            kind="directory",
        )
        for candidate in (root, *root.rglob("*")):
            metadata = candidate.stat(follow_symlinks=False)
            if candidate.is_symlink() or stat.S_IMODE(metadata.st_mode) & 0o222:
                raise ChainError(f"{description} is not recursively sealed")
    if set(evidence_root.iterdir()) != expected_evidence_paths:
        raise ChainError(
            "superseded r2 canary failure evidence changed during validation"
        )
    return marker, raw


def _validate_external_watchdog_drill_marker(
    paths: RecoveryPaths, *, git_identity: Mapping[str, str]
) -> tuple[dict[str, Any], bytes]:
    marker, raw = _read_sealed_marker(
        paths.external_watchdog_drill_marker,
        description="external-watchdog drill completion marker",
    )
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "deployment_id",
        "watchdog_code_sha256",
        "immutable_release_sha256",
        "control_sha256",
        "namespace_cancellation_recovery_seconds",
        "duplicate_jobs",
        "duplicate_admission_intents",
        "fairness_mutations",
        "drill_id",
    }
    if set(marker) != required:
        raise ChainError("external-watchdog drill marker fields drifted")
    _require_release_marker_binding(
        marker,
        git_identity=git_identity,
        protocol=EXTERNAL_WATCHDOG_DRILL_PROTOCOL,
        description="external-watchdog drill marker",
    )
    _require_marker_identity(
        marker,
        identity_field="drill_id",
        description="external-watchdog drill marker",
    )
    if any(
        _SHA256.fullmatch(str(marker.get(field, ""))) is None
        for field in (
            "deployment_id",
            "watchdog_code_sha256",
            "immutable_release_sha256",
            "control_sha256",
        )
    ):
        raise ChainError("external-watchdog drill code binding is malformed")
    recovery_seconds = marker.get(
        "namespace_cancellation_recovery_seconds"
    )
    if (
        not isinstance(recovery_seconds, (int, float))
        or isinstance(recovery_seconds, bool)
        or not math.isfinite(float(recovery_seconds))
        or not 0 <= float(recovery_seconds) <= 900
        or marker.get("duplicate_jobs") != 0
        or marker.get("duplicate_admission_intents") != 0
        or marker.get("fairness_mutations") != 0
    ):
        raise ChainError(
            "external-watchdog drill does not prove bounded, duplicate-free "
            "namespace recovery with preserved fairness"
        )
    return marker, raw


def _validate_watchdog_ready_marker(
    paths: RecoveryPaths,
    *,
    git_identity: Mapping[str, str],
    drill_marker: Mapping[str, Any],
    drill_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    marker, raw = _read_sealed_marker(
        paths.watchdog_ready_marker,
        description="external-watchdog readiness marker",
    )
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "deployment_id",
        "watchdog_code_sha256",
        "immutable_release_sha256",
        "control_sha256",
        "forced_command_only",
        "timer_seconds",
        "scheduler_observations",
        "liveness_email_ack",
        "external_watchdog_drill",
        "namespace_cancellation_recovery_seconds",
        "duplicate_jobs",
        "duplicate_admission_intents",
        "fairness_mutations",
        "marker_id",
    }
    if set(marker) != required:
        raise ChainError("external-watchdog readiness marker fields drifted")
    _require_release_marker_binding(
        marker,
        git_identity=git_identity,
        protocol=WATCHDOG_READY_PROTOCOL,
        description="external-watchdog readiness marker",
    )
    _require_marker_identity(
        marker,
        identity_field="marker_id",
        description="external-watchdog readiness marker",
    )
    observations = marker.get("scheduler_observations")
    if (
        not isinstance(observations, list)
        or len(observations) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in observations
        )
        or float(observations[1]) - float(observations[0]) < 60
    ):
        raise ChainError(
            "external watchdog lacks two scheduler observations at least "
            "60 seconds apart"
        )
    expected_drill = {
        "marker": str(paths.external_watchdog_drill_marker),
        "marker_sha256": _sha256_bytes(drill_raw),
        "drill_id": drill_marker["drill_id"],
    }
    if (
        any(
            marker.get(field) != drill_marker.get(field)
            for field in (
                "deployment_id",
                "watchdog_code_sha256",
                "immutable_release_sha256",
                "control_sha256",
                "namespace_cancellation_recovery_seconds",
                "duplicate_jobs",
                "duplicate_admission_intents",
                "fairness_mutations",
            )
        )
        or marker.get("external_watchdog_drill") != expected_drill
        or marker.get("forced_command_only") is not True
        or marker.get("timer_seconds") != 300
        or marker.get("liveness_email_ack") is not True
    ):
        raise ChainError(
            "external watchdog does not bind its forced-command deployment, "
            "five-minute timer, acknowledged liveness, and completed drill"
        )
    return marker, raw


def verify_post_initialize_watchdog(
    manifest_path: Path,
    *,
    expected_desired_state: str = "paused",
) -> dict[str, Any]:
    """Verify watchdog gates that necessarily bind the initialized control plane."""

    manifest_path = _require_canonical_path(
        _lexical_absolute(manifest_path),
        description="post-initialize watchdog chain manifest",
        kind="file",
    )
    chain_report = verify_chain(manifest_path)
    manifest = _read_json(
        manifest_path, description="post-initialize watchdog chain manifest"
    )
    recovery_root = _manifest_path(
        manifest.get("recovery_root"),
        description="post-initialize watchdog recovery root",
    )
    state_root = _manifest_path(
        manifest.get("state_root"),
        description="post-initialize watchdog state root",
    )

    @dataclass(frozen=True)
    class _MarkerPaths:
        external_watchdog_drill_marker: Path
        watchdog_ready_marker: Path

    paths = _MarkerPaths(
        external_watchdog_drill_marker=(
            recovery_root / EXTERNAL_WATCHDOG_DRILL_MARKER_NAME
        ),
        watchdog_ready_marker=recovery_root / WATCHDOG_READY_MARKER_NAME,
    )
    git_identity = {
        "git_commit": str(manifest.get("release_git_commit", "")),
        "tag_object": str(manifest.get("release_tag_object", "")),
    }
    drill, drill_raw = _validate_external_watchdog_drill_marker(
        paths, git_identity=git_identity
    )
    watchdog, watchdog_raw = _validate_watchdog_ready_marker(
        paths,
        git_identity=git_identity,
        drill_marker=drill,
        drill_raw=drill_raw,
    )
    control_path = _require_canonical_path(
        state_root / "control.json",
        description="initialized schema-5 control",
        kind="file",
    )
    control = _read_json(
        control_path, description="initialized schema-5 control"
    )
    immutable = control.get("immutable")
    control_sha256 = control.get("immutable_sha256")
    if expected_desired_state not in {"paused", "running"}:
        raise ChainError("invalid expected watchdog control state")
    if (
        not isinstance(immutable, Mapping)
        or control.get("desired_state") != expected_desired_state
        or control.get("drain_requested") is not False
        or _SHA256.fullmatch(str(control_sha256)) is None
        or immutable.get("release_id") != RELEASE_ID
        or immutable.get("git_commit") != git_identity["git_commit"]
        or watchdog.get("control_sha256") != control_sha256
        or drill.get("control_sha256") != control_sha256
    ):
        raise ChainError(
            "external watchdog does not bind the exact initialized "
            f"{expected_desired_state} control"
        )
    return {
        "passed": True,
        "chain_id": chain_report["chain_id"],
        "manifest": str(manifest_path),
        "control_sha256": control_sha256,
        "external_watchdog_drill": _launch_prerequisite_binding(
            path=paths.external_watchdog_drill_marker,
            marker=drill,
            raw=drill_raw,
            identity_field="drill_id",
        ),
        "watchdog_ready": _launch_prerequisite_binding(
            path=paths.watchdog_ready_marker,
            marker=watchdog,
            raw=watchdog_raw,
            identity_field="marker_id",
        ),
    }


def _launch_prerequisite_binding(
    *,
    path: Path,
    marker: Mapping[str, Any],
    raw: bytes,
    identity_field: str,
) -> dict[str, Any]:
    return {
        "marker": str(path),
        "marker_sha256": _sha256_bytes(raw),
        "marker_size": len(raw),
        "protocol": marker["protocol"],
        identity_field: marker[identity_field],
    }


def _verified_superseded_r3_prelaunch_failure(
    paths: RecoveryPaths,
) -> dict[str, Any]:
    expected_root = (
        paths.recovery_root
        / recovery_evidence.R3_PRELAUNCH_FAILURE_RELATIVE_ROOT
    )
    if paths.superseded_r3_prelaunch_failure_root != expected_root:
        raise ChainError("superseded r3 prelaunch-failure root is not canonical")
    try:
        binding = recovery_evidence.verify_prelaunch_failure_seal(
            expected_root
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        recovery_evidence.EvidenceError,
    ) as exc:
        raise ChainError(
            f"superseded r3 prelaunch-failure seal is invalid: {exc}"
        ) from exc
    marker = binding.get("marker") if isinstance(binding, dict) else None
    if (
        not isinstance(binding, dict)
        or binding.get("root") != str(expected_root)
        or binding.get("protocol")
        != "schema5-v1.2-r3-prelaunch-failure-seal-binding-v1"
        or binding.get("release_id") != RELEASE_ID
        or binding.get("release_tag") != SUPERSEDED_R3_RELEASE_TAG
        or binding.get("release_git_commit")
        != SUPERSEDED_R3_RELEASE_COMMIT
        or binding.get("release_tag_object")
        != SUPERSEDED_R3_RELEASE_TAG_OBJECT
        or binding.get("chain_namespace")
        != SUPERSEDED_R3_CHAIN_NAMESPACE
        or binding.get("classification")
        != "deterministic_materialization_contract_failure_sealed_fail_closed"
        or binding.get("failure_classifications")
        != list(SUPERSEDED_R3_FAILURE_CLASSIFICATIONS)
        or not isinstance(marker, dict)
        or set(marker) != {"path", "sha256", "size", "seal_id"}
        or marker.get("path")
        != str(paths.superseded_r3_prelaunch_failure_marker)
        or _SHA256.fullmatch(str(marker.get("sha256", ""))) is None
        or _SHA256.fullmatch(str(marker.get("seal_id", ""))) is None
        or not isinstance(marker.get("size"), int)
        or isinstance(marker.get("size"), bool)
        or marker["size"] <= 0
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("scheduler_evidence_required") is not False
        or binding.get("known_scheduler_job_ids") != []
        or _SHA256.fullmatch(
            str(binding.get("archive_inventory_sha256", ""))
        )
        is None
    ):
        raise ChainError(
            "superseded r3 prelaunch-failure seal binding drifted"
        )
    return json.loads(json.dumps(binding, sort_keys=True))


def _verified_superseded_r4_toolchain_failure(
    paths: RecoveryPaths,
) -> dict[str, Any]:
    expected_root = (
        paths.recovery_root / SUPERSEDED_R4_TOOLCHAIN_FAILURE_RELATIVE_ROOT
    )
    if paths.superseded_r4_toolchain_failure_root != expected_root:
        raise ChainError("superseded r4 toolchain-failure root is not canonical")
    try:
        binding = r4_toolchain_failure.verify_failure_seal(
            expected_root,
            recovery_root=paths.recovery_root,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        r4_toolchain_failure.FailureSealError,
        conda_toolchain.CondaToolchainProvisionError,
    ) as exc:
        raise ChainError(
            f"superseded r4 toolchain-failure seal is invalid: {exc}"
        ) from exc
    expected_keys = {
        "passed",
        "root",
        "protocol",
        "release_tag",
        "release_git_commit",
        "chain_namespace",
        "marker",
        "marker_sha256",
        "marker_size",
        "marker_id",
        "classification",
        "retry_in_place",
        "requires_superseding_release",
        "pre_scheduler_submission",
        "known_scheduler_job_ids",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != expected_keys
        or binding.get("passed") is not True
        or binding.get("root") != str(expected_root)
        or binding.get("protocol") != r4_toolchain_failure.PROTOCOL
        or binding.get("release_tag") != r4_toolchain_failure.R4_TAG
        or binding.get("release_git_commit") != r4_toolchain_failure.R4_COMMIT
        or binding.get("chain_namespace") != r4_toolchain_failure.R4_NAMESPACE
        or binding.get("marker")
        != str(expected_root / r4_toolchain_failure.MARKER_NAME)
        or _SHA256.fullmatch(str(binding.get("marker_sha256", ""))) is None
        or _SHA256.fullmatch(str(binding.get("marker_id", ""))) is None
        or not isinstance(binding.get("marker_size"), int)
        or isinstance(binding.get("marker_size"), bool)
        or binding["marker_size"] <= 0
        or binding.get("classification") != r4_toolchain_failure.CLASSIFICATION
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("known_scheduler_job_ids") != []
    ):
        raise ChainError("superseded r4 toolchain-failure binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _verified_superseded_r5_prelaunch_failure(
    paths: RecoveryPaths,
) -> dict[str, Any]:
    expected_root = (
        paths.recovery_root / SUPERSEDED_R5_PRELAUNCH_FAILURE_RELATIVE_ROOT
    )
    if paths.superseded_r5_prelaunch_failure_root != expected_root:
        raise ChainError("superseded r5 prelaunch-failure root is not canonical")
    try:
        binding = r5_prelaunch_failure.verify_failure_seal(
            expected_root,
            recovery_root=paths.recovery_root,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        r5_prelaunch_failure.R5FailureSealError,
    ) as exc:
        raise ChainError(
            f"superseded r5 prelaunch-failure seal is invalid: {exc}"
        ) from exc
    expected_keys = {
        "passed",
        "root",
        "protocol",
        "release_tag",
        "release_git_commit",
        "chain_namespace",
        "marker",
        "marker_sha256",
        "marker_size",
        "marker_id",
        "classification",
        "retry_in_place",
        "requires_superseding_release",
        "pre_scheduler_submission",
        "known_scheduler_job_ids",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != expected_keys
        or binding.get("passed") is not True
        or binding.get("root") != str(expected_root)
        or binding.get("protocol") != r5_prelaunch_failure.PROTOCOL
        or binding.get("release_tag") != r5_prelaunch_failure.R5_TAG
        or binding.get("release_git_commit") != r5_prelaunch_failure.R5_COMMIT
        or binding.get("chain_namespace") != r5_prelaunch_failure.R5_NAMESPACE
        or binding.get("marker")
        != str(expected_root / r5_prelaunch_failure.MARKER_NAME)
        or _SHA256.fullmatch(str(binding.get("marker_sha256", ""))) is None
        or _SHA256.fullmatch(str(binding.get("marker_id", ""))) is None
        or not isinstance(binding.get("marker_size"), int)
        or isinstance(binding.get("marker_size"), bool)
        or binding["marker_size"] <= 0
        or binding.get("classification") != r5_prelaunch_failure.CLASSIFICATION
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("known_scheduler_job_ids") != []
    ):
        raise ChainError("superseded r5 prelaunch-failure binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _verified_superseded_r6_prelaunch_failure(
    paths: RecoveryPaths,
) -> dict[str, Any]:
    expected_root = (
        paths.recovery_root / SUPERSEDED_R6_PRELAUNCH_FAILURE_RELATIVE_ROOT
    )
    if paths.superseded_r6_prelaunch_failure_root != expected_root:
        raise ChainError("superseded r6 prelaunch-failure root is not canonical")
    try:
        binding = r6_prelaunch_failure.verify_failure_seal(
            expected_root,
            recovery_root=paths.recovery_root,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        r6_prelaunch_failure.R6FailureSealError,
        r6_prelaunch_failure.MarkerLastEvidenceError,
        r5_prelaunch_failure.R5FailureSealError,
    ) as exc:
        raise ChainError(
            f"superseded r6 prelaunch-failure seal is invalid: {exc}"
        ) from exc
    expected_keys = {
        "passed",
        "root",
        "protocol",
        "release_tag",
        "release_git_commit",
        "chain_namespace",
        "marker",
        "marker_sha256",
        "marker_size",
        "marker_id",
        "classification",
        "retry_in_place",
        "requires_superseding_release",
        "pre_scheduler_submission",
        "known_scheduler_job_ids",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != expected_keys
        or binding.get("passed") is not True
        or binding.get("root") != str(expected_root)
        or binding.get("protocol") != r6_prelaunch_failure.PROTOCOL
        or binding.get("release_tag") != r6_prelaunch_failure.R6_TAG
        or binding.get("release_git_commit") != r6_prelaunch_failure.R6_COMMIT
        or binding.get("chain_namespace") != r6_prelaunch_failure.R6_NAMESPACE
        or binding.get("marker")
        != str(expected_root / r6_prelaunch_failure.MARKER_NAME)
        or _SHA256.fullmatch(str(binding.get("marker_sha256", ""))) is None
        or _SHA256.fullmatch(str(binding.get("marker_id", ""))) is None
        or not isinstance(binding.get("marker_size"), int)
        or isinstance(binding.get("marker_size"), bool)
        or binding["marker_size"] <= 0
        or binding.get("classification") != r6_prelaunch_failure.CLASSIFICATION
        or binding.get("retry_in_place") is not False
        or binding.get("requires_superseding_release") is not True
        or binding.get("pre_scheduler_submission") is not True
        or binding.get("known_scheduler_job_ids") != []
    ):
        raise ChainError("superseded r6 prelaunch-failure binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _external_launch_prerequisite_contract(
    paths: RecoveryPaths, *, git_identity: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Return only evidence that can exist before the recovery DAG is rendered.

    The external watchdog binds the initialized control hash and exercises the
    controller-drill namespace.  It therefore cannot truthfully be a render-time
    prerequisite; its marker-last gates are verified after ``schema5_initialize`` and
    again immediately before production resume.
    """

    protected, protected_raw = _validate_protected_capacity_marker(
        paths, git_identity=git_identity
    )
    durable, durable_raw = _validate_durable_git_release_marker(
        paths, git_identity=git_identity
    )
    superseded_r2, superseded_r2_raw = (
        _validate_superseded_r2_canary_failure_marker(paths)
    )
    superseded_r3 = _verified_superseded_r3_prelaunch_failure(paths)
    superseded_r4 = _verified_superseded_r4_toolchain_failure(paths)
    superseded_r5 = _verified_superseded_r5_prelaunch_failure(paths)
    superseded_r6 = _verified_superseded_r6_prelaunch_failure(paths)
    return {
        "durable_git_release": {
            **_launch_prerequisite_binding(
                path=paths.durable_git_release_marker,
                marker=durable,
                raw=durable_raw,
                identity_field="marker_id",
            ),
            "release_git_commit": durable["release_git_commit"],
            "release_tag_object": durable["release_tag_object"],
            "bundle_sha256": durable["bundle_sha256"],
        },
        "superseded_r2_canary_failure": {
            **_launch_prerequisite_binding(
                path=paths.superseded_r2_canary_failure_marker,
                marker=superseded_r2,
                raw=superseded_r2_raw,
                identity_field="seal_id",
            ),
            "release_git_commit": SUPERSEDED_R2_RELEASE_COMMIT,
            "release_tag_object": SUPERSEDED_R2_RELEASE_TAG_OBJECT,
            "job_id": SUPERSEDED_R2_CANARY_JOB_ID,
            "retry_in_place": False,
        },
        "superseded_r3_prelaunch_failure": superseded_r3,
        "superseded_r4_toolchain_failure": superseded_r4,
        "superseded_r5_prelaunch_failure": superseded_r5,
        "superseded_r6_prelaunch_failure": superseded_r6,
        "protected_capacity": _launch_prerequisite_binding(
            path=paths.protected_capacity_marker,
            marker=protected,
            raw=protected_raw,
            identity_field="marker_id",
        ),
    }


def _sealed_external_launch_prerequisite_contract(
    paths: RecoveryPaths,
    *,
    git_identity: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Rebind immutable launch bytes without consulting a mutable Git checkout."""

    protected, protected_raw = _read_sealed_marker(
        paths.protected_capacity_marker,
        description="protected-capacity completion marker",
    )
    _require_release_marker_binding(
        protected,
        git_identity=git_identity,
        protocol=PROTECTED_CAPACITY_PROTOCOL,
        description="protected-capacity completion marker",
        schema_version=protected_capacity.SCHEMA_VERSION,
    )
    _require_marker_identity(
        protected,
        identity_field="marker_id",
        description="protected-capacity completion marker",
    )
    if protected.get("passed") is not True:
        raise ChainError("sealed protected-capacity authority is not passing")
    durable, durable_raw = _validate_durable_git_release_marker(
        paths, git_identity=git_identity
    )
    superseded_r2, superseded_r2_raw = (
        _validate_superseded_r2_canary_failure_marker(paths)
    )
    superseded_r3 = _verified_superseded_r3_prelaunch_failure(paths)
    superseded_r4 = _verified_superseded_r4_toolchain_failure(paths)
    superseded_r5 = _verified_superseded_r5_prelaunch_failure(paths)
    superseded_r6 = _verified_superseded_r6_prelaunch_failure(paths)
    return {
        "durable_git_release": {
            **_launch_prerequisite_binding(
                path=paths.durable_git_release_marker,
                marker=durable,
                raw=durable_raw,
                identity_field="marker_id",
            ),
            "release_git_commit": durable["release_git_commit"],
            "release_tag_object": durable["release_tag_object"],
            "bundle_sha256": durable["bundle_sha256"],
        },
        "superseded_r2_canary_failure": {
            **_launch_prerequisite_binding(
                path=paths.superseded_r2_canary_failure_marker,
                marker=superseded_r2,
                raw=superseded_r2_raw,
                identity_field="seal_id",
            ),
            "release_git_commit": SUPERSEDED_R2_RELEASE_COMMIT,
            "release_tag_object": SUPERSEDED_R2_RELEASE_TAG_OBJECT,
            "job_id": SUPERSEDED_R2_CANARY_JOB_ID,
            "retry_in_place": False,
        },
        "superseded_r3_prelaunch_failure": superseded_r3,
        "superseded_r4_toolchain_failure": superseded_r4,
        "superseded_r5_prelaunch_failure": superseded_r5,
        "superseded_r6_prelaunch_failure": superseded_r6,
        "protected_capacity": _launch_prerequisite_binding(
            path=paths.protected_capacity_marker,
            marker=protected,
            raw=protected_raw,
            identity_field="marker_id",
        ),
    }


def _verify_pilot_runtime_binding(
    paths: RecoveryPaths,
    binding: Mapping[str, Any],
) -> tuple[Path, Path]:
    required = {
        "harness_prefix",
        "environment_manifest_path",
        "environment_manifest_sha256",
        "directory_inventory_sha256",
        "python_path",
        "python_sha256",
        "python_size",
        "library_path",
    }
    expected_prefix = (
        paths.materialization_pilot_root
        / "materialization"
        / "harness-environment"
    )
    expected_manifest = (
        paths.materialization_pilot_root
        / "materialization"
        / "release"
        / "harness_environment.schema5-v1.json"
    )
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise ChainError("pilot verifier-runtime binding fields drifted")
    prefix = _require_canonical_path(
        Path(str(binding.get("harness_prefix", ""))),
        description="pilot verifier harness",
        kind="directory",
    )
    manifest = _require_canonical_path(
        Path(str(binding.get("environment_manifest_path", ""))),
        description="pilot verifier environment manifest",
        kind="file",
    )
    python = _require_canonical_path(
        Path(str(binding.get("python_path", ""))),
        description="pilot verifier Python",
        kind="file",
    )
    library = _require_canonical_path(
        Path(str(binding.get("library_path", ""))),
        description="pilot verifier library",
        kind="directory",
    )
    try:
        python.relative_to(prefix)
        library.relative_to(prefix)
    except ValueError as exc:
        raise ChainError("pilot verifier runtime escapes its harness") from exc
    try:
        expected_python = (prefix / "bin" / "python").resolve(strict=True)
        manifest_payload = _read_json(
            manifest,
            description="pilot verifier environment manifest",
        )
    except (OSError, RuntimeError) as exc:
        raise ChainError("pilot verifier runtime aliases are unsafe") from exc
    if (
        prefix != expected_prefix
        or manifest != expected_manifest
        or python != expected_python
        or library != prefix / "lib"
        or binding.get("environment_manifest_sha256") != _sha256(manifest)
        or manifest_payload.get("prefix") != str(prefix)
        or manifest_payload.get("sealed_read_only") is not True
        or manifest_payload.get("directory_inventory", {}).get(
            "inventory_sha256"
        )
        != binding.get("directory_inventory_sha256")
        or not isinstance(binding.get("directory_inventory_sha256"), str)
        or _SHA256.fullmatch(binding["directory_inventory_sha256"]) is None
        or binding.get("python_sha256") != _sha256(python)
        or binding.get("python_size") != python.stat().st_size
        or not os.access(python, os.X_OK)
        or any(
            stat.S_IMODE(path.stat().st_mode) & 0o222
            for path in (prefix, manifest, python, library)
        )
    ):
        raise ChainError("pilot verifier runtime differs from sealed evidence")
    return python, library


def _validate_prerequisite_reports(
    *,
    reports: Mapping[str, Any],
    paths: RecoveryPaths,
    git_identity: Mapping[str, str],
    code_records: Sequence[Mapping[str, Any]],
    verify_live_bindings: bool = True,
) -> None:
    if set(reports) != {"materialization_pilot", "slurm_canary"}:
        raise ChainError("prerequisite verifier reports are incomplete")
    pilot = reports["materialization_pilot"]
    canary = reports["slurm_canary"]
    code_by_path = {record["git_path"]: record for record in code_records}
    canary_code = canary.get("code_identity") if isinstance(canary, dict) else None
    live_inventories = (
        pilot.get("live_source_inventory_sha256")
        if isinstance(pilot, dict)
        else None
    )
    pilot_conda_toolchain = (
        pilot.get("conda_toolchain") if isinstance(pilot, dict) else None
    )
    source_package_cache = (
        pilot.get("source_package_cache") if isinstance(pilot, dict) else None
    )
    package_cache_seed_input = (
        pilot.get("package_cache_seed_input")
        if isinstance(pilot, dict)
        else None
    )
    validated_cache_input = _validate_package_cache_seed_input(
        package_cache_seed_input,
        source_package_cache=paths.source_package_cache,
    )
    verifier_runtime = (
        pilot.get("verifier_runtime") if isinstance(pilot, dict) else None
    )
    scheduler_acceptance = (
        pilot.get("scheduler_acceptance") if isinstance(pilot, dict) else None
    )
    reconciliation_incident = (
        pilot.get("reconciliation_incident")
        if isinstance(pilot, dict)
        else None
    )
    expected_incident = {
        "path": str(
            paths.materialization_pilot_root
            / "environment-capture"
            / "evidence"
            / "CONDA_RECONCILIATION_INCIDENT.json"
        ),
        "sha256": PRODUCTION_CONDA_RECONCILIATION_INCIDENT_SHA256,
        "incident_id": PRODUCTION_CONDA_RECONCILIATION_INCIDENT_ID,
        "harness_stale_conda_record_present": False,
        "serving_stale_conda_record_present": False,
    }
    expected_durable = (
        _external_launch_prerequisite_contract(
            paths, git_identity=git_identity
        )
        if verify_live_bindings
        else _sealed_external_launch_prerequisite_contract(
            paths, git_identity=git_identity
        )
    )["durable_git_release"]
    portable_durable = {
        "path": expected_durable["marker"],
        "sha256": expected_durable["marker_sha256"],
        "marker_id": expected_durable["marker_id"],
        "release_git_commit": expected_durable["release_git_commit"],
        "release_tag_object": expected_durable["release_tag_object"],
        "bundle_sha256": expected_durable["bundle_sha256"],
    }
    ownership_policy_sha256 = (
        pilot.get("ownership_policy_sha256")
        if isinstance(pilot, dict)
        else None
    )
    integrity_policy_sha256 = (
        pilot.get("integrity_normalization_policy_sha256")
        if isinstance(pilot, dict)
        else None
    )
    if (
        not isinstance(pilot, dict)
        or pilot.get("status") != "verified"
        or pilot.get("schema_version") != 5
        or pilot.get("release_id") != RELEASE_ID
        or pilot.get("expected_tag") != RELEASE_TAG
        or pilot.get("expected_commit") != git_identity["git_commit"]
        or pilot.get("durable_git_release") != portable_durable
        or reconciliation_incident != expected_incident
        or not isinstance(pilot.get("pilot_id"), str)
        or _SHA256.fullmatch(pilot["pilot_id"]) is None
        or not isinstance(scheduler_acceptance, dict)
        or set(scheduler_acceptance)
        != {
            "acceptance_id",
            "marker",
            "marker_sha256",
            "job_id",
            "receipt",
            "receipt_sha256",
            "receipt_id",
            "effective_requeue",
            "spooled_script_sha256",
            "terminal_state",
            "exit_code",
            "reason",
        }
        or _SHA256.fullmatch(
            str(scheduler_acceptance.get("acceptance_id", ""))
        )
        is None
        or scheduler_acceptance.get("marker")
        != str(
            paths.materialization_pilot_root
            / "PILOT_SCHEDULER_ACCEPTED.json"
        )
        or _SHA256.fullmatch(
            str(scheduler_acceptance.get("marker_sha256", ""))
        )
        is None
        or not str(scheduler_acceptance.get("job_id", "")).isdigit()
        or _SHA256.fullmatch(
            str(scheduler_acceptance.get("receipt_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(scheduler_acceptance.get("receipt_id", ""))
        )
        is None
        or scheduler_acceptance.get("effective_requeue") != 0
        or _SHA256.fullmatch(
            str(scheduler_acceptance.get("spooled_script_sha256", ""))
        )
        is None
        or scheduler_acceptance.get("terminal_state") != "COMPLETED"
        or scheduler_acceptance.get("exit_code") != "0:0"
        or not isinstance(scheduler_acceptance.get("reason"), str)
        or not scheduler_acceptance["reason"]
        or not isinstance(live_inventories, dict)
        or set(live_inventories) != {"harness", "serving"}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in live_inventories.values()
        )
        or ownership_policy_sha256
        != code_by_path.get(OWNERSHIP_POLICY_GIT_PATH, {}).get("sha256")
        or integrity_policy_sha256
        != code_by_path.get(
            INTEGRITY_NORMALIZATION_POLICY_GIT_PATH, {}
        ).get("sha256")
        or source_package_cache != str(paths.source_package_cache)
        or package_cache_seed_input != validated_cache_input
        or pilot_conda_toolchain
        != _verified_conda_toolchain_binding(paths.conda_toolchain_root)
        or not isinstance(canary, dict)
        or canary.get("schema_version") != 4
        or canary.get("kind")
        != "schema5_slurm_fleet_composite_canary_complete"
        or canary.get("canary_root") != str(paths.slurm_canary_root)
        or canary.get("partition") != "mit_normal"
        or canary.get("qos") != "normal"
        or not isinstance(canary.get("canary_id"), str)
        or _SHA256.fullmatch(canary["canary_id"]) is None
        or canary.get("transaction_root")
        != str(paths.slurm_canary_root / "transaction")
        or canary.get("dependency_root")
        != str(paths.slurm_canary_root / "dependency-cascade")
        or canary.get("turnover_root")
        != str(paths.slurm_canary_root / "turnover")
        or not isinstance(canary.get("transaction_canary_id"), str)
        or _SHA256.fullmatch(canary["transaction_canary_id"]) is None
        or not isinstance(canary.get("dependency_canary_id"), str)
        or _SHA256.fullmatch(canary["dependency_canary_id"]) is None
        or not isinstance(canary.get("dependency_marker_sha256"), str)
        or _SHA256.fullmatch(canary["dependency_marker_sha256"]) is None
        or not isinstance(canary.get("dependency_parameters"), list)
        or any(
            not isinstance(value, str)
            for value in canary.get("dependency_parameters", [])
        )
        or "kill_invalid_depend"
        not in canary.get("dependency_parameters", [])
        or canary.get("dependency_kill_invalid_depend") is not True
        or canary.get("dependency_root_initial_hold") is not True
        or canary.get("dependency_child_never_started") is not True
        or not isinstance(
            canary.get("dependency_alert_latency_seconds"), (int, float)
        )
        or isinstance(
            canary.get("dependency_alert_latency_seconds"), bool
        )
        or not isinstance(
            canary.get("dependency_alert_latency_bound_seconds"),
            (int, float),
        )
        or isinstance(
            canary.get("dependency_alert_latency_bound_seconds"), bool
        )
        or canary.get("dependency_alert_latency_seconds", float("inf"))
        < 0
        or canary.get("dependency_alert_latency_seconds", float("inf"))
        > canary.get("dependency_alert_latency_bound_seconds", -1)
        or canary.get("dependency_alert_latency_bound_seconds")
        != 180.0
        or not isinstance(canary.get("turnover_canary_id"), str)
        or _SHA256.fullmatch(canary["turnover_canary_id"]) is None
        or canary.get("turnover_cycles_completed") != 2
        or canary.get("turnover_allocations_submitted") != 3
        or canary.get("turnover_maximum_physical_allocations_observed") != 2
        or canary.get("turnover_maximum_extra_gpus_observed") != 0
        or canary.get("turnover_all_effective_requeue") != 0
        or canary.get("turnover_continuous_routed_endpoint_evidence") is not True
        or canary.get("production_overlap_gpu_ceiling") != 4
        or not isinstance(canary.get("transaction_marker_sha256"), str)
        or _SHA256.fullmatch(canary["transaction_marker_sha256"]) is None
        or not isinstance(canary.get("turnover_marker_sha256"), str)
        or _SHA256.fullmatch(canary["turnover_marker_sha256"]) is None
        or not isinstance(canary_code, dict)
        or canary_code.get("release_tag") != RELEASE_TAG
        or canary_code.get("release_git_commit") != git_identity["git_commit"]
        or canary_code.get("release_tag_object") != git_identity["tag_object"]
        or canary_code.get("canary_script")
        != code_by_path.get(SLURM_CANARY_GIT_PATH)
        or canary_code.get("fleet_transactions")
        != code_by_path.get(FLEET_TRANSACTIONS_GIT_PATH)
        or canary_code.get("durable_git_publisher")
        != code_by_path.get(DURABLE_GIT_RELEASE_GIT_PATH)
        or canary_code.get("durable_git_release") != portable_durable
    ):
        raise ChainError(
            "sealed prerequisite evidence does not belong to the exact release code"
        )
    if verify_live_bindings:
        _verify_pilot_runtime_binding(paths, verifier_runtime)


def _prerequisite_evidence_contract(
    paths: RecoveryPaths,
    *,
    git_identity: Mapping[str, str],
    verifier_checkout: Path,
    verifier_python: Path | None = None,
    verifier_library: Path | None = None,
) -> dict[str, Any]:
    external_launch = _external_launch_prerequisite_contract(
        paths, git_identity=git_identity
    )
    code_records = _prerequisite_code_records(
        verifier_checkout, commit=git_identity["git_commit"]
    )
    reports = _invoke_tagged_prerequisite_verifiers(
        paths,
        checkout=verifier_checkout,
        expected_commit=git_identity["git_commit"],
        expected_tag_object=git_identity["tag_object"],
        verifier_python=verifier_python,
        verifier_library=verifier_library,
    )
    _validate_prerequisite_reports(
        reports=reports,
        paths=paths,
        git_identity=git_identity,
        code_records=code_records,
    )
    if verifier_python is None:
        sealed_python, sealed_library = _verify_pilot_runtime_binding(
            paths,
            reports["materialization_pilot"]["verifier_runtime"],
        )
        sealed_reports = _invoke_tagged_prerequisite_verifiers(
            paths,
            checkout=verifier_checkout,
            expected_commit=git_identity["git_commit"],
            expected_tag_object=git_identity["tag_object"],
            verifier_python=sealed_python,
            verifier_library=sealed_library,
        )
        _validate_prerequisite_reports(
            reports=sealed_reports,
            paths=paths,
            git_identity=git_identity,
            code_records=code_records,
        )
        if sealed_reports != reports:
            raise ChainError(
                "sealed pilot verifier runtime returned different prerequisite evidence"
            )
        reports = sealed_reports
    else:
        sealed_python = verifier_python
        sealed_library = verifier_library
    if sealed_python is None or sealed_library is None:
        raise ChainError("sealed pilot runtime is unavailable")
    live_conda_toolchain = _verified_conda_toolchain_binding(
        paths.conda_toolchain_root
    )
    if live_conda_toolchain != reports["materialization_pilot"].get(
        "conda_toolchain"
    ):
        raise ChainError(
            "Conda runtime toolchain differs from sealed pilot provenance"
        )
    pilot_marker = _sealed_prerequisite_marker(
        root=paths.materialization_pilot_root,
        marker_name=MATERIALIZATION_PILOT_MARKER,
        description="materialization pilot",
    )
    canary_marker = _sealed_prerequisite_marker(
        root=paths.slurm_canary_root,
        marker_name=SLURM_CANARY_MARKER,
        description="Slurm fleet canary",
    )
    contract = {
        "schema_version": 10,
        "protocol": PREREQUISITE_PROTOCOL,
        "release_tag": RELEASE_TAG,
        "release_git_commit": git_identity["git_commit"],
        "release_tag_object": git_identity["tag_object"],
        "tagged_code": code_records,
        **external_launch,
        "materialization_pilot": {
            **pilot_marker,
            "pilot_id": reports["materialization_pilot"]["pilot_id"],
            "live_source_inventory_sha256": dict(
                reports["materialization_pilot"][
                    "live_source_inventory_sha256"
                ]
            ),
            "ownership_policy_sha256": reports[
                "materialization_pilot"
            ]["ownership_policy_sha256"],
            "integrity_normalization_policy_sha256": reports[
                "materialization_pilot"
            ]["integrity_normalization_policy_sha256"],
            "conda_toolchain": dict(
                reports["materialization_pilot"]["conda_toolchain"]
            ),
            "source_package_cache": reports[
                "materialization_pilot"
            ]["source_package_cache"],
            "package_cache_seed_input": dict(
                reports["materialization_pilot"][
                    "package_cache_seed_input"
                ]
            ),
            "verifier_runtime": dict(
                reports["materialization_pilot"]["verifier_runtime"]
            ),
            "reconciliation_incident": dict(
                reports["materialization_pilot"][
                    "reconciliation_incident"
                ]
            ),
            "verifier_report": reports["materialization_pilot"],
            "verifier_report_sha256": _sha256_bytes(
                _canonical_json(reports["materialization_pilot"])
            ),
        },
        "slurm_canary": {
            **canary_marker,
            "canary_id": reports["slurm_canary"]["canary_id"],
            "transaction_canary_id": reports["slurm_canary"][
                "transaction_canary_id"
            ],
            "transaction_marker_sha256": reports["slurm_canary"][
                "transaction_marker_sha256"
            ],
            "dependency_canary_id": reports["slurm_canary"][
                "dependency_canary_id"
            ],
            "dependency_marker_sha256": reports["slurm_canary"][
                "dependency_marker_sha256"
            ],
            "dependency_parameters": reports["slurm_canary"][
                "dependency_parameters"
            ],
            "dependency_kill_invalid_depend": reports["slurm_canary"][
                "dependency_kill_invalid_depend"
            ],
            "dependency_root_initial_hold": reports["slurm_canary"][
                "dependency_root_initial_hold"
            ],
            "dependency_child_never_started": reports["slurm_canary"][
                "dependency_child_never_started"
            ],
            "dependency_alert_latency_seconds": reports["slurm_canary"][
                "dependency_alert_latency_seconds"
            ],
            "dependency_alert_latency_bound_seconds": reports[
                "slurm_canary"
            ]["dependency_alert_latency_bound_seconds"],
            "turnover_canary_id": reports["slurm_canary"][
                "turnover_canary_id"
            ],
            "turnover_marker_sha256": reports["slurm_canary"][
                "turnover_marker_sha256"
            ],
            "turnover_cycles_completed": reports["slurm_canary"][
                "turnover_cycles_completed"
            ],
            "turnover_allocations_submitted": reports["slurm_canary"][
                "turnover_allocations_submitted"
            ],
            "turnover_maximum_physical_allocations_observed": reports[
                "slurm_canary"
            ]["turnover_maximum_physical_allocations_observed"],
            "turnover_maximum_extra_gpus_observed": reports["slurm_canary"][
                "turnover_maximum_extra_gpus_observed"
            ],
            "turnover_all_effective_requeue": reports["slurm_canary"][
                "turnover_all_effective_requeue"
            ],
            "turnover_continuous_routed_endpoint_evidence": reports[
                "slurm_canary"
            ]["turnover_continuous_routed_endpoint_evidence"],
            "production_overlap_gpu_ceiling": reports["slurm_canary"][
                "production_overlap_gpu_ceiling"
            ],
            "verifier_report": reports["slurm_canary"],
            "verifier_report_sha256": _sha256_bytes(
                _canonical_json(reports["slurm_canary"])
            ),
        },
    }
    contract["evidence_id"] = _sha256_bytes(_canonical_json(contract))
    return contract


def _validate_prerequisite_evidence_contract(
    paths: RecoveryPaths,
    contract: Any,
    *,
    git_identity: Mapping[str, str],
    verifier_checkout: Path | None,
    markers_only: bool,
    verifier_python: Path | None = None,
    verifier_library: Path | None = None,
    sealed_only: bool = False,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "protocol",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "tagged_code",
        "durable_git_release",
        "superseded_r2_canary_failure",
        "superseded_r3_prelaunch_failure",
        "superseded_r4_toolchain_failure",
        "superseded_r5_prelaunch_failure",
        "superseded_r6_prelaunch_failure",
        "protected_capacity",
        "materialization_pilot",
        "slurm_canary",
        "evidence_id",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ChainError("prerequisite evidence contract fields drifted")
    identity = dict(contract)
    evidence_id = identity.pop("evidence_id", None)
    code_records = contract.get("tagged_code")
    expected_paths = list(PREREQUISITE_CODE_GIT_PATHS)
    if (
        contract.get("schema_version") != 10
        or contract.get("protocol") != PREREQUISITE_PROTOCOL
        or contract.get("release_tag") != RELEASE_TAG
        or contract.get("release_git_commit") != git_identity["git_commit"]
        or contract.get("release_tag_object") != git_identity["tag_object"]
        or not isinstance(evidence_id, str)
        or _SHA256.fullmatch(evidence_id) is None
        or evidence_id != _sha256_bytes(_canonical_json(identity))
        or not isinstance(code_records, list)
        or len(code_records) != len(expected_paths)
    ):
        raise ChainError("prerequisite evidence identity is invalid")
    for record, git_path in zip(code_records, expected_paths, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"git_path", "sha256", "size"}
            or record.get("git_path") != git_path
            or not isinstance(record.get("sha256"), str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] <= 0
        ):
            raise ChainError(f"prerequisite tagged-code binding drifted: {git_path}")

    expected_external = (
        _sealed_external_launch_prerequisite_contract(
            paths, git_identity=git_identity
        )
        if sealed_only
        else _external_launch_prerequisite_contract(
            paths, git_identity=git_identity
        )
    )
    if any(
        contract.get(name) != binding
        for name, binding in expected_external.items()
    ):
        raise ChainError("external launch-prerequisite binding drifted")

    records: dict[str, dict[str, Any]] = {}
    for name, root, marker_name, id_field in (
        (
            "materialization_pilot",
            paths.materialization_pilot_root,
            MATERIALIZATION_PILOT_MARKER,
            "pilot_id",
        ),
        (
            "slurm_canary",
            paths.slurm_canary_root,
            SLURM_CANARY_MARKER,
            "canary_id",
        ),
    ):
        record = contract.get(name)
        expected_fields = {
            "root",
            "marker",
            "marker_sha256",
            "marker_size",
            id_field,
            "verifier_report",
            "verifier_report_sha256",
        }
        if name == "materialization_pilot":
            expected_fields |= {
                "live_source_inventory_sha256",
                "ownership_policy_sha256",
                "integrity_normalization_policy_sha256",
                "conda_toolchain",
                "source_package_cache",
                "package_cache_seed_input",
                "verifier_runtime",
                "reconciliation_incident",
            }
        else:
            expected_fields |= {
                "transaction_canary_id",
                "transaction_marker_sha256",
                "dependency_canary_id",
                "dependency_marker_sha256",
                "dependency_parameters",
                "dependency_kill_invalid_depend",
                "dependency_root_initial_hold",
                "dependency_child_never_started",
                "dependency_alert_latency_seconds",
                "dependency_alert_latency_bound_seconds",
                "turnover_canary_id",
                "turnover_marker_sha256",
                "turnover_cycles_completed",
                "turnover_allocations_submitted",
                "turnover_maximum_physical_allocations_observed",
                "turnover_maximum_extra_gpus_observed",
                "turnover_all_effective_requeue",
                "turnover_continuous_routed_endpoint_evidence",
                "production_overlap_gpu_ceiling",
            }
        current = _sealed_prerequisite_marker(
            root=root,
            marker_name=marker_name,
            description=name.replace("_", " "),
        )
        if (
            not isinstance(record, dict)
            or set(record) != expected_fields
            or any(record.get(field) != value for field, value in current.items())
            or not isinstance(record.get(id_field), str)
            or _SHA256.fullmatch(record[id_field]) is None
            or not isinstance(record.get("verifier_report"), dict)
            or record.get("verifier_report_sha256")
            != _sha256_bytes(_canonical_json(record["verifier_report"]))
            or record["verifier_report"].get(id_field) != record[id_field]
            or (
                name == "materialization_pilot"
                and (
                    record.get("live_source_inventory_sha256")
                    != record["verifier_report"].get(
                        "live_source_inventory_sha256"
                    )
                    or record.get("ownership_policy_sha256")
                    != record["verifier_report"].get(
                        "ownership_policy_sha256"
                    )
                    or record.get(
                        "integrity_normalization_policy_sha256"
                    )
                    != record["verifier_report"].get(
                        "integrity_normalization_policy_sha256"
                    )
                    or record.get("conda_toolchain")
                    != record["verifier_report"].get("conda_toolchain")
                    or record.get("source_package_cache")
                    != record["verifier_report"].get(
                        "source_package_cache"
                    )
                    or record.get("package_cache_seed_input")
                    != record["verifier_report"].get(
                        "package_cache_seed_input"
                    )
                    or record.get("verifier_runtime")
                    != record["verifier_report"].get("verifier_runtime")
                    or record.get("reconciliation_incident")
                    != record["verifier_report"].get(
                        "reconciliation_incident"
                    )
                )
            )
            or (
                name == "slurm_canary"
                and any(
                    record.get(field)
                    != record["verifier_report"].get(field)
                    for field in (
                        "transaction_canary_id",
                        "transaction_marker_sha256",
                        "dependency_canary_id",
                        "dependency_marker_sha256",
                        "dependency_parameters",
                        "dependency_kill_invalid_depend",
                        "dependency_root_initial_hold",
                        "dependency_child_never_started",
                        "dependency_alert_latency_seconds",
                        "dependency_alert_latency_bound_seconds",
                        "turnover_canary_id",
                        "turnover_marker_sha256",
                        "turnover_cycles_completed",
                        "turnover_allocations_submitted",
                        "turnover_maximum_physical_allocations_observed",
                        "turnover_maximum_extra_gpus_observed",
                        "turnover_all_effective_requeue",
                        "turnover_continuous_routed_endpoint_evidence",
                        "production_overlap_gpu_ceiling",
                    )
                )
            )
        ):
            raise ChainError(f"{name} prerequisite binding drifted")
        records[name] = record
    reports = {
        name: record["verifier_report"] for name, record in records.items()
    }
    _validate_prerequisite_reports(
        reports=reports,
        paths=paths,
        git_identity=git_identity,
        code_records=code_records,
        verify_live_bindings=not sealed_only,
    )
    if sealed_only:
        _verify_pilot_runtime_binding(
            paths,
            records["materialization_pilot"]["verifier_runtime"],
        )
        return json.loads(json.dumps(contract, sort_keys=True))
    if markers_only:
        return json.loads(json.dumps(contract, sort_keys=True))
    if verifier_checkout is None:
        raise ChainError("full prerequisite verification requires a tagged checkout")
    sealed_python, sealed_library = _verify_pilot_runtime_binding(
        paths,
        records["materialization_pilot"]["verifier_runtime"],
    )
    if (
        verifier_python is not None
        and _lexical_absolute(verifier_python) != sealed_python
    ) or (
        verifier_library is not None
        and _lexical_absolute(verifier_library) != sealed_library
    ):
        raise ChainError(
            "full prerequisite verification must use the sealed pilot harness"
        )
    expected = _prerequisite_evidence_contract(
        paths,
        git_identity=git_identity,
        verifier_checkout=verifier_checkout,
        verifier_python=sealed_python,
        verifier_library=sealed_library,
    )
    if contract != expected:
        raise ChainError(
            "live prerequisite verification differs from the immutable chain binding"
        )
    return expected


def _q(path_or_value: object) -> str:
    return shlex.quote(str(path_or_value))


def _trusted_shell_prelude() -> str:
    """Sanitize every generated job before any launch-authorization command."""

    return f"""\
unset BASH_ENV CDPATH ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD
unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_ATTR_NOSYSTEM GIT_CEILING_DIRECTORIES
unset GIT_COMMON_DIR GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_GLOBAL
unset GIT_CONFIG_NOSYSTEM GIT_CONFIG_PARAMETERS GIT_CONFIG_SYSTEM GIT_DIR
unset GIT_DISCOVERY_ACROSS_FILESYSTEM GIT_EXEC_PATH GIT_INDEX_FILE GIT_NAMESPACE
unset GIT_NO_REPLACE_OBJECTS GIT_OBJECT_DIRECTORY GIT_REPLACE_REF_BASE
unset GIT_SHALLOW_FILE GIT_SSH GIT_SSH_COMMAND GIT_TEMPLATE_DIR GIT_WORK_TREE
unset SLURM_CLUSTERS SLURM_CONF SLURM_EXIT_ERROR SLURM_TIME_FORMAT
while IFS= read -r ambient_name; do
  case "$ambient_name" in
    BASH_FUNC_*|PIP_*|CONDA_*|GIT_CONFIG_KEY_*|GIT_CONFIG_VALUE_*|GIT_TRACE*|SACCT_*|SBATCH_*|SCONTROL_*|SQUEUE_*)
      builtin unset -v "$ambient_name" 2>/dev/null || true
      ;;
  esac
done < <(compgen -e)
export PATH={TRUSTED_SYSTEM_PATH}
readonly PATH
while read -r _ _ ambient_function; do
  builtin unset -f "$ambient_function"
done < <(builtin declare -F)
export LANG=C LC_ALL=C
export GIT_ATTR_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0
export GIT_PAGER=cat PAGER=cat
"""


def _common_exports(paths: RecoveryPaths) -> str:
    return f"""\
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH LD_PRELOAD
while IFS= read -r ambient_name; do
  case "$ambient_name" in
    PIP_*|CONDA_*) unset "$ambient_name" ;;
  esac
done < <(compgen -e)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1
export PIP_CONFIG_FILE=/dev/null PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export CONDARC=/dev/null CONDA_NO_PLUGINS=true
export ASYS_RESULTS_ROOT={_q(paths.results_root)}
export ASYS_RELEASE_WORKTREE={_q(paths.worktree)}
export HF_HOME={_q(paths.hf_home)}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"""


def _immutable_tool_shell_check(
    *,
    variable: str,
    expected_sha256: str,
    expected_size: int,
    description: str,
) -> str:
    """Render a fail-closed shell check for one immutable bundled executable."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
        raise ChainError(f"unsafe shell variable for immutable tool: {variable!r}")
    if _SHA256.fullmatch(expected_sha256) is None or expected_size <= 0:
        raise ChainError(f"invalid immutable {description} hash/size contract")
    return f"""\
[[ -f "${{{variable}}}" && ! -L "${{{variable}}}" ]] || {{
  echo "{description} is missing, non-regular, or a symlink" >&2
  exit 2
}}
[[ "$(realpath -e -- "${{{variable}}}")" == "${{{variable}}}" ]] || {{
  echo "{description} path is not canonical" >&2
  exit 2
}}
tool_mode="$(stat -c '%a' -- "${{{variable}}}")"
[[ "$tool_mode" =~ ^[0-7]+$ ]] && (( (8#$tool_mode & 8#222) == 0 )) || {{
  echo "{description} is writable" >&2
  exit 2
}}
[[ "$(stat -c '%s' -- "${{{variable}}}")" == {_q(expected_size)} ]] || {{
  echo "{description} size drifted" >&2
  exit 2
}}
[[ "$(sha256sum -- "${{{variable}}}" | cut -d' ' -f1)" == {_q(expected_sha256)} ]] || {{
  echo "{description} hash drifted" >&2
  exit 2
}}
"""


def _executable_bytes_shell_check(
    *,
    variable: str,
    expected_sha256: str,
    expected_size: int,
    description: str,
) -> str:
    """Render a byte-identity check for an executable that may be owner-writable."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
        raise ChainError(f"unsafe executable shell variable: {variable!r}")
    if _SHA256.fullmatch(expected_sha256) is None or expected_size <= 0:
        raise ChainError(f"invalid {description} hash/size contract")
    return f"""\
[[ -f "${{{variable}}}" && ! -L "${{{variable}}}" && -x "${{{variable}}}" ]] || {{
  echo "{description} is missing, non-regular, symlinked, or non-executable" >&2
  exit 2
}}
[[ "$(realpath -e -- "${{{variable}}}")" == "${{{variable}}}" ]] || {{
  echo "{description} path is not canonical" >&2
  exit 2
}}
[[ "$(stat -c '%s' -- "${{{variable}}}")" == {_q(expected_size)} ]] || {{
  echo "{description} size drifted" >&2
  exit 2
}}
[[ "$(sha256sum -- "${{{variable}}}" | cut -d' ' -f1)" == {_q(expected_sha256)} ]] || {{
  echo "{description} hash drifted" >&2
  exit 2
}}
"""


def _sentinel_bootstrap_runtime_shell(paths: RecoveryPaths) -> str:
    """Render an independent runtime verification of the sealed Python bootstrap."""

    bootstrap = _sentinel_bootstrap_contract(paths)
    marker_sha256 = _sha256_bytes(_canonical_json(bootstrap))
    return f"""\
bootstrap_root={_q(paths.sentinel_bootstrap_root)}
bootstrap_python={_q(paths.sentinel_bootstrap_python)}
bootstrap_inventory={_q(paths.sentinel_bootstrap_inventory)}
bootstrap_marker={_q(paths.sentinel_bootstrap_marker)}
[[ -d "$bootstrap_root" && ! -L "$bootstrap_root" && "$(realpath -e -- "$bootstrap_root")" == "$bootstrap_root" ]] || {{
  echo "sentinel bootstrap root is missing or unsafe" >&2
  exit 2
}}
[[ "$(sha256sum -- "$bootstrap_marker" | cut -d' ' -f1)" == {_q(marker_sha256)} ]] || {{
  echo "sentinel bootstrap completion marker drifted" >&2
  exit 2
}}
[[ "$(sha256sum -- "$bootstrap_inventory" | cut -d' ' -f1)" == {_q(bootstrap['payload_inventory_sha256'])} ]] || {{
  echo "sentinel bootstrap inventory drifted" >&2
  exit 2
}}
[[ -z "$(find "$bootstrap_root" -type l -print -quit)" ]] || {{
  echo "sentinel bootstrap contains a symlink" >&2
  exit 2
}}
[[ -z "$(find "$bootstrap_root" -perm /222 -print -quit)" ]] || {{
  echo "sentinel bootstrap contains a writable entry" >&2
  exit 2
}}
[[ -z "$(find "$bootstrap_root" -type f ! -links 1 -print -quit)" ]] || {{
  echo "sentinel bootstrap contains a shared payload inode" >&2
  exit 2
}}
cmp -s \\
  <(find "$bootstrap_root" -type f -printf '%P\\n' | LC_ALL=C sort) \\
  <({{ awk '{{print $2}}' "$bootstrap_inventory"; printf '%s\\n' {_q(SENTINEL_BOOTSTRAP_INVENTORY_NAME)} {_q(SENTINEL_BOOTSTRAP_MARKER_NAME)}; }} | LC_ALL=C sort) || {{
  echo "sentinel bootstrap file set drifted" >&2
  exit 2
}}
cmp -s \\
  <(find "$bootstrap_root" -mindepth 1 -type d -printf '%P\\n' | LC_ALL=C sort) \\
  <(awk '{{path=$2; while (sub("/[^/]+$", "", path)) print path}}' "$bootstrap_inventory" | LC_ALL=C sort -u) || {{
  echo "sentinel bootstrap directory set drifted" >&2
  exit 2
}}
(cd "$bootstrap_root" && sha256sum --check --strict "$bootstrap_inventory") >/dev/null || {{
  echo "sentinel bootstrap payload verification failed" >&2
  exit 2
}}
"""


def _launch_authorization_gate_shell(
    paths: RecoveryPaths, *, spec_name: str
) -> str:
    """Fence production stages behind the marker-last launch transaction.

    ``scontrol release`` and filesystem publication cannot be atomic together.
    A released root can therefore receive a CPU before the submitter publishes
    the immutable launch marker.  The allocation may wait at this gate, but its
    stage body cannot mutate state until the receipt, exact-root release, and
    launch marker independently validate inside the sealed bootstrap.

    The failure sentinel is intentionally not wrapped by this gate.  If the
    submitter dies at this boundary, the root fails closed after the bounded wait
    and the sentinel must remain able to record and alert on that failure.
    """

    if _SAFE_NAME.fullmatch(spec_name) is None:
        raise ChainError(f"unsafe recovery stage name for launch gate: {spec_name!r}")
    template = r"""\
# No recovery-stage mutation may occur above this launch-authorization gate.
__BOOTSTRAP_RUNTIME__
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "launch gate requires an exact non-array Slurm job ID" >&2
  exit 2
}
launch_gate_job_record="$(scontrol show job -o "$SLURM_JOB_ID")" || {
  echo "launch gate cannot read its exact Slurm job record" >&2
  exit 2
}
launch_gate_comment="$(
  tr ' ' '\n' <<<"$launch_gate_job_record" |
    sed -n 's/^Comment=//p' |
    head -n 1
)"
if [[ ! "$launch_gate_comment" =~ ^asys:s5-recovery-v1\.2-r7:([0-9a-f]{64}):g([0-9]{4}):([A-Za-z0-9._-]+)$ ]]; then
  echo "launch gate scheduler comment is malformed" >&2
  exit 2
fi
launch_gate_chain_id="${BASH_REMATCH[1]}"
launch_gate_generation_digits="${BASH_REMATCH[2]}"
launch_gate_stage="${BASH_REMATCH[3]}"
[[ "$launch_gate_stage" == __SPEC_NAME__ ]] || {
  echo "launch gate scheduler comment names the wrong recovery stage" >&2
  exit 2
}
launch_gate_generation=$((10#$launch_gate_generation_digits))
launch_gate_manifest=__CHAIN_MANIFEST__
launch_gate_evidence_root=__RECOVERY_ROOT__
if (( launch_gate_generation > 0 )); then
  launch_gate_evidence_root=__REPAIR_ROOT__"/g${launch_gate_generation_digits}"
fi
launch_gate_receipt="$launch_gate_evidence_root/__SUBMISSION_RECEIPT_NAME__"
launch_gate_release="$launch_gate_evidence_root/__ROOT_RELEASE_COMPLETE_NAME__"
launch_gate_complete="$launch_gate_evidence_root/__LAUNCH_COMPLETE_NAME__"
launch_gate_deadline=$((SECONDS + __LAUNCH_GATE_TIMEOUT_SECONDS__))
while [[ ! -f "$launch_gate_receipt" || -L "$launch_gate_receipt" ||
         ! -f "$launch_gate_release" || -L "$launch_gate_release" ||
         ! -f "$launch_gate_complete" || -L "$launch_gate_complete" ]]; do
  if (( SECONDS >= launch_gate_deadline )); then
    echo "launch authorization markers were not published within __LAUNCH_GATE_TIMEOUT_SECONDS__ seconds" >&2
    exit 2
  fi
  sleep 1
done
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \
  "$launch_gate_manifest" \
  "$launch_gate_evidence_root" \
  "$launch_gate_receipt" \
  "$launch_gate_release" \
  "$launch_gate_complete" \
  "$launch_gate_chain_id" \
  "$launch_gate_generation" \
  "$launch_gate_stage" \
  "$SLURM_JOB_ID" \
  "$launch_gate_comment" <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import runpy
import stat
import sys
import types
from typing import NoReturn


SCHEMA_VERSION = __SUBMISSION_SCHEMA_VERSION__
DEPENDENCY_POLICY = __DEPENDENCY_POLICY_CONTRACT__
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"launch authorization failed: {message}")


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            fail(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_sealed(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    if not path.is_absolute():
        fail(f"{label} path is not absolute")
    try:
        metadata = os.lstat(path)
    except OSError as error:
        fail(f"{label} is unavailable: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} is symlinked or non-regular")
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        fail(f"{label} is writable")
    try:
        if path.resolve(strict=True) != path:
            fail(f"{label} path is non-canonical")
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: fail(
                f"{label} contains non-finite JSON value {token}"
            ),
        )
    except SystemExit:
        raise
    except Exception as error:
        fail(f"{label} cannot be read as JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object")
    return value, raw


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{label} is not a SHA-256 digest")
    return value


def require_identity(
    value: dict[str, object], field: str, label: str
) -> None:
    identity = dict(value)
    observed = identity.pop(field, None)
    if require_sha256(observed, f"{label} {field}") != digest(
        canonical_json(identity)
    ):
        fail(f"{label} identity drifted")


def require_backing_file(
    raw_path: object,
    expected_hash: object,
    label: str,
    *,
    beneath: Path | None = None,
) -> dict[str, object]:
    if not isinstance(raw_path, str):
        fail(f"{label} path is invalid")
    path = Path(raw_path)
    value, raw = load_sealed(path, label)
    if digest(raw) != require_sha256(expected_hash, f"{label} hash"):
        fail(f"{label} hash drifted")
    if beneath is not None:
        try:
            path.relative_to(beneath)
        except ValueError:
            fail(f"{label} escaped its evidence root")
    return value


def require_canary(value: object, expected: object) -> None:
    fields = {
        "canary_id",
        "marker_sha256",
        "alert_latency_seconds",
        "alert_latency_bound_seconds",
        "kill_invalid_depend",
        "root_initial_hold",
        "child_never_started",
    }
    if not isinstance(value, dict) or set(value) != fields or value != expected:
        fail("dependency canary binding drifted")
    require_sha256(value["canary_id"], "dependency canary ID")
    require_sha256(value["marker_sha256"], "dependency canary marker hash")
    latency = value["alert_latency_seconds"]
    if (
        not isinstance(latency, (int, float))
        or isinstance(latency, bool)
        or not math.isfinite(float(latency))
        or not 0 <= float(latency) <= 180.0
        or value["alert_latency_bound_seconds"] != 180.0
        or value["kill_invalid_depend"] is not True
        or value["root_initial_hold"] is not True
        or value["child_never_started"] is not True
    ):
        fail("dependency canary does not prove bounded fail-closed behavior")


(
    manifest_arg,
    evidence_root_arg,
    receipt_arg,
    release_arg,
    launch_arg,
    chain_id,
    generation_arg,
    stage_name,
    job_id,
    scheduler_comment,
) = sys.argv[1:]
manifest_path = Path(manifest_arg)
evidence_root = Path(evidence_root_arg)
receipt_path = Path(receipt_arg)
release_path = Path(release_arg)
launch_path = Path(launch_arg)
try:
    generation = int(generation_arg)
except ValueError:
    fail("repair generation is not numeric")
if generation < 0 or not job_id.isdigit():
    fail("scheduler generation or job ID is invalid")
expected_comment = (
    f"asys:s5-recovery-v1.2-r7:{chain_id}:g{generation:04d}:{stage_name}"
)
if scheduler_comment != expected_comment:
    fail("scheduler comment does not equal the runtime identity")
if (
    not evidence_root.is_absolute()
    or evidence_root.resolve(strict=True) != evidence_root
    or not evidence_root.is_dir()
    or evidence_root.is_symlink()
):
    fail("launch evidence root is unavailable or unsafe")

manifest, manifest_raw = load_sealed(manifest_path, "chain manifest")
manifest_identity = dict(manifest)
manifest_chain_id = manifest_identity.pop("chain_id", None)
if (
    manifest.get("schema_version") != __CHAIN_SCHEMA_VERSION__
    or manifest.get("protocol") != "schema5-v1.2-r7-recovery-chain"
    or manifest_chain_id != chain_id
    or require_sha256(manifest_chain_id, "manifest chain ID")
    != digest(canonical_json(manifest_identity))
):
    fail("chain manifest identity drifted")
manifest_jobs = manifest.get("jobs")
if not isinstance(manifest_jobs, list) or not manifest_jobs:
    fail("chain manifest has no jobs")
manifest_by_name: dict[str, dict[str, object]] = {}
for row in manifest_jobs:
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("name"), str)
        or row["name"] in manifest_by_name
    ):
        fail("chain manifest job identities are invalid")
    manifest_by_name[row["name"]] = row
if stage_name not in manifest_by_name or stage_name == "failure_sentinel":
    fail("scheduler stage is not an authorized production stage")

canary_source = manifest.get("prerequisite_evidence")
if not isinstance(canary_source, dict):
    fail("manifest prerequisite evidence is invalid")

r3_binding = canary_source.get("superseded_r3_prelaunch_failure")
r3_root = (
    Path(str(manifest.get("recovery_root", "")))
    / "prelaunch_failures"
    / "schema5-v1.2-r3"
)
bundle = manifest.get("recovery_tool_bundle")
sealer_record = next(
    (
        row
        for row in bundle
        if isinstance(row, dict)
        and row.get("git_path") == "scripts/seal_recovery_evidence.py"
    ),
    None,
) if isinstance(bundle, list) else None
if (
    not isinstance(r3_binding, dict)
    or r3_binding.get("root") != str(r3_root)
    or not isinstance(sealer_record, dict)
):
    fail("superseded r3 prelaunch-failure binding is absent")
sealer_path = Path(str(sealer_record.get("path", "")))
try:
    sealer_metadata = os.lstat(sealer_path)
except OSError as error:
    fail(f"bundled r3 seal verifier is unavailable: {error}")
if (
    stat.S_ISLNK(sealer_metadata.st_mode)
    or not stat.S_ISREG(sealer_metadata.st_mode)
    or stat.S_IMODE(sealer_metadata.st_mode) & 0o222
    or sealer_metadata.st_size != sealer_record.get("size")
    or digest(sealer_path.read_bytes()) != sealer_record.get("sha256")
):
    fail("bundled r3 seal verifier drifted")
try:
    observed_r3_binding = runpy.run_path(str(sealer_path))[
        "verify_prelaunch_failure_seal"
    ](r3_root)
except Exception as error:
    fail(f"superseded r3 prelaunch-failure verification failed: {error}")
if observed_r3_binding != r3_binding:
    fail("superseded r3 prelaunch-failure seal differs from manifest")

toolchain_binding = manifest.get("conda_toolchain")
toolchain_root = Path(str(manifest.get("conda_toolchain_root", "")))


def bundled_record(git_path: str) -> tuple[dict[str, object], Path]:
    record = next(
        (
            row
            for row in bundle
            if isinstance(row, dict) and row.get("git_path") == git_path
        ),
        None,
    ) if isinstance(bundle, list) else None
    if not isinstance(record, dict):
        fail(f"immutable tool binding is absent: {git_path}")
    path = Path(str(record.get("path", "")))
    try:
        metadata = os.lstat(path)
    except OSError as error:
        fail(f"immutable tool is unavailable ({git_path}): {error}")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o222
        or metadata.st_size != record.get("size")
        or digest(path.read_bytes()) != record.get("sha256")
    ):
        fail(f"immutable tool drifted: {git_path}")
    return record, path


_, runtime_identity_path = bundled_record(
    "scripts/schema5_conda_runtime_identity.py"
)
_, toolchain_verifier_path = bundled_record(
    "scripts/provision_schema5_conda_toolchain.py"
)
try:
    package = types.ModuleType("scripts")
    package.__path__ = []
    sys.modules["scripts"] = package
    runtime_spec = importlib.util.spec_from_file_location(
        "scripts.schema5_conda_runtime_identity",
        runtime_identity_path,
    )
    if runtime_spec is None or runtime_spec.loader is None:
        fail("cannot load bundled Conda runtime verifier")
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime_module
    runtime_spec.loader.exec_module(runtime_module)
    toolchain_spec = importlib.util.spec_from_file_location(
        "scripts.provision_schema5_conda_toolchain",
        toolchain_verifier_path,
    )
    if toolchain_spec is None or toolchain_spec.loader is None:
        fail("cannot load bundled Conda toolchain verifier")
    toolchain_module = importlib.util.module_from_spec(toolchain_spec)
    sys.modules[toolchain_spec.name] = toolchain_module
    toolchain_spec.loader.exec_module(toolchain_module)
    observed_toolchain = toolchain_module.verified_conda_toolchain_binding(
        toolchain_root,
        exercise=True,
    )
except SystemExit:
    raise
except Exception as error:
    fail(f"sealed Conda toolchain verification failed: {error}")
if observed_toolchain != toolchain_binding:
    fail("sealed Conda toolchain differs from manifest")

r4_binding = canary_source.get("superseded_r4_toolchain_failure")
r4_root = (
    Path(str(manifest.get("recovery_root", "")))
    / "prelaunch_failures"
    / "schema5-v1.2-r4-toolchain"
)
_, r4_sealer_path = bundled_record(
    "scripts/seal_schema5_r4_toolchain_failure.py"
)
try:
    observed_r4_binding = runpy.run_path(str(r4_sealer_path))[
        "verify_failure_seal"
    ](
        r4_root,
        recovery_root=Path(str(manifest.get("recovery_root", ""))),
    )
except Exception as error:
    fail(f"superseded r4 toolchain-failure verification failed: {error}")
if observed_r4_binding != r4_binding:
    fail("superseded r4 toolchain-failure seal differs from manifest")

r5_binding = canary_source.get("superseded_r5_prelaunch_failure")
r5_root = (
    Path(str(manifest.get("recovery_root", "")))
    / "prelaunch_failures"
    / "schema5-v1.2-r5-cli"
)
_, r5_sealer_path = bundled_record(
    "scripts/seal_schema5_r5_prelaunch_failure.py"
)
try:
    observed_r5_binding = runpy.run_path(str(r5_sealer_path))[
        "verify_failure_seal"
    ](
        r5_root,
        recovery_root=Path(str(manifest.get("recovery_root", ""))),
    )
except Exception as error:
    fail(f"superseded r5 prelaunch-failure verification failed: {error}")
if observed_r5_binding != r5_binding:
    fail("superseded r5 prelaunch-failure seal differs from manifest")

r6_binding = canary_source.get("superseded_r6_prelaunch_failure")
r6_root = (
    Path(str(manifest.get("recovery_root", "")))
    / "prelaunch_failures"
    / "schema5-v1.2-r6-toolchain-binding"
)
_, r6_sealer_path = bundled_record(
    "scripts/seal_schema5_r6_prelaunch_failure.py"
)
try:
    observed_r6_binding = runpy.run_path(str(r6_sealer_path))[
        "verify_failure_seal"
    ](
        r6_root,
        recovery_root=Path(str(manifest.get("recovery_root", ""))),
    )
except Exception as error:
    fail(f"superseded r6 prelaunch-failure verification failed: {error}")
if observed_r6_binding != r6_binding:
    fail("superseded r6 prelaunch-failure seal differs from manifest")


def require_launch_prerequisite(
    name: str,
    *,
    marker_name: str,
    protocol: str,
    identity_field: str,
    schema_version: int = 1,
) -> dict[str, object]:
    record = canary_source.get(name)
    fields = {
        "marker", "marker_sha256", "marker_size", "protocol", identity_field
    }
    recovery_root = Path(str(manifest.get("recovery_root", "")))
    expected_path = recovery_root / marker_name
    if (
        not isinstance(record, dict)
        or set(record) != fields
        or record.get("marker") != str(expected_path)
        or record.get("protocol") != protocol
        or not isinstance(record.get("marker_size"), int)
        or isinstance(record.get("marker_size"), bool)
        or record["marker_size"] <= 0
    ):
        fail(f"{name} manifest binding drifted")
    marker = require_backing_file(
        record["marker"],
        record["marker_sha256"],
        name.replace("_", " "),
        beneath=recovery_root,
    )
    if expected_path.stat().st_size != record["marker_size"]:
        fail(f"{name} marker size drifted")
    if marker.get(identity_field) != record.get(identity_field):
        fail(f"{name} marker identity differs from manifest")
    require_identity(marker, identity_field, name.replace("_", " "))
    if (
        marker.get("schema_version") != schema_version
        or marker.get("protocol") != protocol
        or marker.get("passed") is not True
        or marker.get("release_id") != "__RELEASE_ID__"
        or marker.get("release_tag") != "__RELEASE_TAG__"
        or marker.get("release_git_commit")
        != manifest.get("release_git_commit")
        or marker.get("release_tag_object")
        != manifest.get("release_tag_object")
        or marker.get("chain_namespace") != "__CHAIN_NAMESPACE__"
    ):
        fail(f"{name} does not bind the exact release")
    return marker


protected = require_launch_prerequisite(
    "protected_capacity",
    marker_name="__PROTECTED_CAPACITY_MARKER_NAME__",
    protocol="__PROTECTED_CAPACITY_PROTOCOL__",
    identity_field="marker_id",
    schema_version=__PROTECTED_CAPACITY_SCHEMA_VERSION__,
)

superseded_r2_record = canary_source.get(
    "superseded_r2_canary_failure"
)
superseded_r2_fields = {
    "marker", "marker_sha256", "marker_size", "protocol", "seal_id",
    "release_git_commit", "release_tag_object", "job_id",
    "retry_in_place",
}
superseded_r2_path = (
    Path(str(manifest.get("recovery_root", "")))
    / "__SUPERSEDED_R2_CANARY_FAILURE_RELATIVE_PATH__"
)
if (
    not isinstance(superseded_r2_record, dict)
    or set(superseded_r2_record) != superseded_r2_fields
    or superseded_r2_record.get("marker") != str(superseded_r2_path)
    or superseded_r2_record.get("protocol")
    != "__SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL__"
    or superseded_r2_record.get("seal_id")
    != "__SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID__"
    or superseded_r2_record.get("release_git_commit")
    != "__SUPERSEDED_R2_RELEASE_COMMIT__"
    or superseded_r2_record.get("release_tag_object")
    != "__SUPERSEDED_R2_RELEASE_TAG_OBJECT__"
    or superseded_r2_record.get("job_id")
    != "__SUPERSEDED_R2_CANARY_JOB_ID__"
    or superseded_r2_record.get("retry_in_place") is not False
):
    fail("superseded r2 canary failure manifest binding drifted")
superseded_r2 = require_backing_file(
    superseded_r2_record["marker"],
    superseded_r2_record["marker_sha256"],
    "superseded r2 canary failure seal",
    beneath=Path(str(manifest.get("recovery_root", ""))),
)
superseded_r2_release = superseded_r2.get("release")
superseded_r2_known = superseded_r2.get("known_scheduler_identity")
if (
    superseded_r2_path.stat().st_size
    != superseded_r2_record.get("marker_size")
    or superseded_r2.get("schema_version") != 1
    or superseded_r2.get("protocol")
    != "__SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL__"
    or superseded_r2.get("passed") is not True
    or superseded_r2.get("classification")
    != "deterministic_canary_failure_sealed_fail_closed"
    or superseded_r2.get("error_classification")
    != "deterministic_missing_subprocess_capture"
    or superseded_r2.get("retry_in_place") is not False
    or superseded_r2.get("seal_id")
    != "__SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID__"
    or not isinstance(superseded_r2_release, dict)
    or superseded_r2_release.get("release_tag")
    != "__SUPERSEDED_R2_RELEASE_TAG__"
    or superseded_r2_release.get("release_git_commit")
    != "__SUPERSEDED_R2_RELEASE_COMMIT__"
    or superseded_r2_release.get("release_tag_object")
    != "__SUPERSEDED_R2_RELEASE_TAG_OBJECT__"
    or not isinstance(superseded_r2_known, dict)
    or superseded_r2_known.get("job_ids")
    != ["__SUPERSEDED_R2_CANARY_JOB_ID__"]
    or superseded_r2.get("no_active_matching_jobs_preseal") is not True
    or superseded_r2.get("no_active_matching_jobs_postseal") is not True
):
    fail("superseded r2 canary failure seal drifted")


def protected_placement_totals(
    value: object, role: str, preempt_type: str
) -> dict[str, int]:
    fields = (
        {
            "partition", "qos", "partition_preempt_mode", "qos_preempt_mode",
            "base_active_gpus", "reserved_additive_gpus",
            "effective_active_gpus", "retained_warm_turnover_gpus",
            "attested_total_gpus", "partition_cpus",
            "partition_memory_mib", "partition_gpus", "partition_nodes",
        }
        if role == "server"
        else {
            "partition", "qos", "partition_preempt_mode", "qos_preempt_mode",
            "slots", "cpus", "memory_mib", "reserve_jobs", "submit_headroom",
        }
    )
    if not isinstance(value, list) or not value:
        fail(f"protected scientific {role} placements are absent")
    identities: set[tuple[str, str]] = set()
    normalized: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != fields:
            fail(f"protected scientific {role} placement fields drifted")
        partition = row.get("partition")
        qos = row.get("qos")
        if (
            not isinstance(partition, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", partition)
            is None
            or not isinstance(qos, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", qos)
            is None
            or row.get("partition_preempt_mode") != "OFF"
            or (
                row.get("qos_preempt_mode") != "OFF"
                and not (
                    preempt_type == "preempt/partition_prio"
                    and row.get("qos_preempt_mode") == "cluster"
                )
            )
            or (partition, qos) in identities
        ):
            fail(f"protected scientific {role} placement is invalid")
        identities.add((partition, qos))
        normalized.append(dict(row))
    normalized.sort(
        key=lambda row: (
            str(row["partition"]), str(row["qos"]), canonical_json(row)
        )
    )
    if value != normalized:
        fail(f"protected scientific {role} placements are not sorted")
    capacity_fields = (
        (
            "base_active_gpus", "reserved_additive_gpus",
            "effective_active_gpus", "retained_warm_turnover_gpus",
            "attested_total_gpus", "partition_cpus",
            "partition_memory_mib", "partition_gpus", "partition_nodes",
        )
        if role == "server"
        else ("slots", "cpus", "memory_mib", "reserve_jobs", "submit_headroom")
    )
    totals = {field: 0 for field in capacity_fields}
    for row in normalized:
        for field in capacity_fields:
            observed = row.get(field)
            if (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed < 0
            ):
                fail(f"protected scientific {role} capacity is invalid")
            totals[field] += observed
    if role == "client":
        directly_usable = [
            row
            for row in normalized
            if row["slots"] >= 384
            and row["cpus"] >= 384
            and row["memory_mib"] >= 1572864
            and row["reserve_jobs"] >= 64
            and row["submit_headroom"] >= 448
            and row["cpus"] >= row["slots"]
            and row["memory_mib"] >= row["slots"] * 4096
            and row["submit_headroom"]
            >= row["slots"] + row["reserve_jobs"]
        ]
        if len(directly_usable) != 1:
            fail(
                "protected capacity must have exactly one directly usable "
                "384-cell + 64-reserve client placement"
            )
    else:
        if (
            len(normalized) != 1
            or totals["base_active_gpus"] != 24
            or totals["reserved_additive_gpus"] != 0
            or totals["effective_active_gpus"] != 24
            or totals["retained_warm_turnover_gpus"] != 4
            or totals["attested_total_gpus"] != 28
            or totals["partition_cpus"] < 1
            or totals["partition_memory_mib"] < 1
            or totals["partition_gpus"] < 28
            or totals["partition_nodes"] < 1
        ):
            fail(
                "protected scientific server placement does not bind the "
                "exact 24-active plus 4-warm partition inventory"
            )
    return totals


def validate_protected_qos_contracts(
    value: object,
    server_qos: set[str],
    client_qos: set[str],
    expected_running_jobs: int,
) -> None:
    fields = {
        "qos", "max_wall_seconds", "max_jobs_per_user",
        "max_submit_jobs_per_user", "required_wall_seconds",
        "required_running_jobs", "required_submit_jobs",
    }
    if not isinstance(value, list) or not value:
        fail("protected scientific QOS contracts are absent")
    prior = ""
    by_qos: dict[str, dict[str, object]] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != fields:
            fail("protected scientific QOS contract fields drifted")
        qos = row.get("qos")
        if (
            not isinstance(qos, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", qos
            )
            is None
            or qos <= prior
        ):
            fail(
                "protected scientific QOS contracts are duplicated "
                "or not canonical"
            )
        prior = qos
        limits: dict[str, int | None] = {}
        for field in (
            "max_wall_seconds",
            "max_jobs_per_user",
            "max_submit_jobs_per_user",
        ):
            observed = row.get(field)
            if observed is not None and (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed < 1
            ):
                fail(f"protected scientific QOS {qos} limit is invalid")
            limits[field] = observed
        required_wall = row.get("required_wall_seconds")
        required_running = row.get("required_running_jobs")
        required_submit = row.get("required_submit_jobs")
        if (
            not isinstance(required_wall, int)
            or isinstance(required_wall, bool)
            or required_wall < 43200
            or not isinstance(required_running, int)
            or isinstance(required_running, bool)
            or required_running < 0
            or not isinstance(required_submit, int)
            or isinstance(required_submit, bool)
            or required_submit < 1
            or (
                limits["max_wall_seconds"] is not None
                and limits["max_wall_seconds"] < required_wall
            )
            or (
                limits["max_jobs_per_user"] is not None
                and limits["max_jobs_per_user"] < required_running
            )
            or (
                limits["max_submit_jobs_per_user"] is not None
                and limits["max_submit_jobs_per_user"] < required_submit
            )
        ):
            fail(
                f"protected scientific QOS {qos} cannot sustain its load"
            )
        by_qos[qos] = row
    if (
        set(by_qos) != server_qos | client_qos
        or sum(
            int(row["required_running_jobs"])
            for row in by_qos.values()
        )
        != expected_running_jobs
        or sum(
            int(row["required_submit_jobs"])
            for row in by_qos.values()
        )
        != 448
        or max(
            int(row["required_wall_seconds"])
            for row in by_qos.values()
        )
        != 86400
        or any(
            int(by_qos[qos]["required_wall_seconds"]) != 86400
            for qos in server_qos
        )
        or any(
            int(by_qos[qos]["required_wall_seconds"]) < 43200
            for qos in client_qos
        )
    ):
        fail(
            "protected scientific QOS contracts do not prove exact "
            "job and walltime capacity"
        )


protected_fields = {
    "schema_version", "protocol", "passed", "release_id", "release_tag",
    "release_git_commit", "release_tag_object", "chain_namespace",
    "source_tree_sha256", "dispatcher_source_sha256",
    "qualification_runner_source_sha256",
    "capacity_generation",
    "base_fleet_contract_path", "base_fleet_contract_sha256",
    "effective_fleet_contract_path", "effective_fleet_contract_sha256",
    "additive_overlay_contract_path", "additive_overlay_contract_sha256",
    "static_feasibility_certificate",
    "static_feasibility_wave_passed",
    "static_feasibility_configured_client_ceiling",
    "static_feasibility_certified_saturation_target",
    "static_feasibility_selected_cell_count",
    "static_feasibility_target_cell_count",
    "static_feasibility_shortfall_cells",
    "base_active_logical_replicas", "base_active_gpus",
    "base_active_topology", "base_active_topology_sha256",
    "additive_reserved_logical_replicas", "additive_reserved_gpus",
    "additive_reserved_tp1_replicas", "additive_reserved_tp2_replicas",
    "additive_reserved_topology", "additive_reserved_topology_sha256",
    "effective_active_logical_replicas", "effective_active_gpus",
    "effective_active_topology", "effective_active_topology_sha256",
    "retained_warm_turnover_job_elements",
    "retained_warm_turnover_gpus",
    "retained_warm_turnover_tp1_allocations",
    "retained_warm_turnover_tp2_allocations",
    "retained_warm_turnover_topology",
    "retained_warm_turnover_topology_sha256",
    "attested_total_gpus", "job_element_accounting",
    "active_gpus", "warm_headroom_gpus", "cell_ceiling", "reserve_jobs",
    "submit_headroom", "cpu", "memory_mib", "preempt_type",
    "capacity_source", "scheduler_cluster", "scheduler_account",
    "scheduler_user", "scheduler_max_jobs",
    "scheduler_max_submit_jobs", "running_scientific_jobs",
    "minimum_scientific_wall_seconds", "scientific_qos_contracts",
    "partition_cpus", "partition_memory_mib", "partition_gpus",
    "fleet_contract_sha256", "active_fleet_topology_sha256",
    "scientific_server_preempt_mode", "scientific_client_preempt_mode",
    "scientific_server_placements", "scientific_client_placements",
    "scheduler_evidence_id", "scheduler_evidence_sha256",
    "canary_id", "canary_evidence_sha256",
    "squeue_complete", "sacct_complete", "marker_id",
}
if set(protected) != protected_fields:
    fail("protected-capacity marker fields drifted")
integer_minima = {
    "capacity_generation": 1,
    "base_active_logical_replicas": 22,
    "base_active_gpus": 24,
    "effective_active_logical_replicas": 22,
    "effective_active_gpus": 24,
    "retained_warm_turnover_job_elements": 3,
    "retained_warm_turnover_gpus": 4,
    "attested_total_gpus": 28,
    "active_gpus": 24,
    "warm_headroom_gpus": 4,
    "cell_ceiling": 384,
    "reserve_jobs": 64,
    "submit_headroom": 448,
    "cpu": 384,
    "memory_mib": 1572864,
    "scheduler_max_submit_jobs": 448,
    "partition_cpus": 384,
    "partition_memory_mib": 1572864,
    "partition_gpus": 0,
}
protected_preempt_type = protected.get("preempt_type")
if protected_preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
    fail("protected capacity has unsupported PreemptType")
protected_servers = protected_placement_totals(
    protected.get("scientific_server_placements"),
    "server",
    str(protected_preempt_type),
)
protected_clients = protected_placement_totals(
    protected.get("scientific_client_placements"),
    "client",
    str(protected_preempt_type),
)
protected_server_qos = {
    str(row["qos"])
    for row in protected["scientific_server_placements"]
}
protected_client_qos = {
    str(row["qos"])
    for row in protected["scientific_client_placements"]
}
if (
    not isinstance(protected.get("effective_active_logical_replicas"), int)
    or isinstance(protected.get("effective_active_logical_replicas"), bool)
    or not isinstance(
        protected.get("retained_warm_turnover_job_elements"), int
    )
    or isinstance(
        protected.get("retained_warm_turnover_job_elements"), bool
    )
):
    fail("protected dynamic serving-job counts are malformed")
protected_expected_running = (
    384
    + protected["effective_active_logical_replicas"]
    + protected["retained_warm_turnover_job_elements"]
)
validate_protected_qos_contracts(
    protected.get("scientific_qos_contracts"),
    protected_server_qos,
    protected_client_qos,
    protected_expected_running,
)
protected_max_jobs = protected.get("scheduler_max_jobs")
protected_recovery_root = Path(str(manifest.get("recovery_root", "")))
for path_field, hash_field, description in (
    (
        "base_fleet_contract_path",
        "base_fleet_contract_sha256",
        "protected base fleet contract",
    ),
    (
        "effective_fleet_contract_path",
        "effective_fleet_contract_sha256",
        "protected effective fleet contract",
    ),
    (
        "additive_overlay_contract_path",
        "additive_overlay_contract_sha256",
        "protected additive overlay contract",
    ),
):
    require_backing_file(
        protected.get(path_field),
        protected.get(hash_field),
        description,
        beneath=protected_recovery_root,
    )
certificate_binding = protected.get("static_feasibility_certificate")
if (
    not isinstance(certificate_binding, dict)
    or set(certificate_binding) != {"path", "sha256", "certificate_id"}
):
    fail("protected static-feasibility binding is malformed")
protected_certificate = require_backing_file(
    certificate_binding.get("path"),
    certificate_binding.get("sha256"),
    "protected static-feasibility certificate",
    beneath=protected_recovery_root,
)
require_identity(
    protected_certificate,
    "certificate_id",
    "protected static-feasibility certificate",
)
certificate_wave = protected_certificate.get("wave")
certificate_batches = (
    certificate_wave.get("microbatches")
    if isinstance(certificate_wave, dict)
    else None
)
expected_certificate_batch_sizes = [24] * 11 + [14]
if (
    protected_certificate.get("protocol")
    != "schema5-v1.2-r7-throughput-preflight-capacity-certificate-v1"
    or protected_certificate.get("passed") is not True
    or protected_certificate.get("certificate_id")
    != certificate_binding.get("certificate_id")
    or protected_certificate.get("release_git_commit")
    != manifest.get("release_git_commit")
    or protected_certificate.get("source_tree_sha256")
    != protected.get("source_tree_sha256")
    or protected_certificate.get("dispatcher_source_sha256")
    != protected.get("dispatcher_source_sha256")
    or protected_certificate.get("qualification_runner_source_sha256")
    != protected.get("qualification_runner_source_sha256")
    or protected_certificate.get("capacity_generation")
    != protected.get("capacity_generation")
    or protected_certificate.get("base_fleet_contract_sha256")
    != protected.get("base_fleet_contract_sha256")
    or protected_certificate.get(
        "proposed_effective_fleet_contract_sha256"
    )
    != protected.get("effective_fleet_contract_sha256")
    or protected_certificate.get("additive_overlay_contract_sha256")
    != protected.get("additive_overlay_contract_sha256")
    or protected_certificate.get("base_logical_replicas")
    != protected.get("base_active_logical_replicas")
    or protected_certificate.get("base_allocated_gpus")
    != protected.get("base_active_gpus")
    or protected_certificate.get("effective_logical_replicas")
    != protected.get("effective_active_logical_replicas")
    or protected_certificate.get("effective_active_gpus")
    != protected.get("effective_active_gpus")
    or protected_certificate.get("additive_allocated_gpus")
    != protected.get("additive_reserved_gpus")
    or protected_certificate.get("selected_cell_count") != 278
    or not isinstance(certificate_batches, list)
    or [
        batch.get("selected_count")
        if isinstance(batch, dict)
        else None
        for batch in certificate_batches
    ]
    != expected_certificate_batch_sizes
    or not isinstance(certificate_wave, dict)
    or certificate_wave.get("passed") is not False
    or certificate_wave.get("target_active_cells") != 384
    or certificate_wave.get("selected_cell_count") != 278
    or certificate_wave.get("shortfall_cells") != 106
    or certificate_wave.get("microbatch_count") != 12
    or protected.get("static_feasibility_wave_passed") is not False
    or protected.get("static_feasibility_configured_client_ceiling") != 384
    or protected.get("static_feasibility_certified_saturation_target") != 278
    or protected.get("static_feasibility_selected_cell_count") != 278
    or protected.get("static_feasibility_target_cell_count") != 384
    or protected.get("static_feasibility_shortfall_cells") != 106
):
    fail("protected static-feasibility certificate binding drifted")
for value_field, hash_field in (
    ("base_active_topology", "base_active_topology_sha256"),
    ("additive_reserved_topology", "additive_reserved_topology_sha256"),
    ("effective_active_topology", "effective_active_topology_sha256"),
    (
        "retained_warm_turnover_topology",
        "retained_warm_turnover_topology_sha256",
    ),
):
    if digest(canonical_json(protected.get(value_field))) != protected.get(
        hash_field
    ):
        fail(f"protected topology hash drifted: {value_field}")
protected_accounting = protected.get("job_element_accounting")
expected_protected_accounting = {
    "cell_job_elements": 384,
    "active_server_job_elements": protected[
        "effective_active_logical_replicas"
    ],
    "warm_turnover_job_elements": protected[
        "retained_warm_turnover_job_elements"
    ],
    "controller_monitor_other_held_job_elements": (
        64
        - protected["effective_active_logical_replicas"]
        - protected["retained_warm_turnover_job_elements"]
    ),
    "total_non_cell_reserve_job_elements": 64,
    "total_canary_job_elements": 448,
}
if (
    set(protected) != protected_fields
    or any(
        not isinstance(protected.get(field), int)
        or isinstance(protected.get(field), bool)
        or protected[field] < minimum
        for field, minimum in integer_minima.items()
    )
    or protected_servers["effective_active_gpus"]
    != protected.get("active_gpus")
    or protected_servers["retained_warm_turnover_gpus"]
    != protected.get("warm_headroom_gpus")
    or protected_servers["base_active_gpus"]
    != protected.get("base_active_gpus")
    or protected_servers["reserved_additive_gpus"]
    != protected.get("additive_reserved_gpus")
    or protected_servers["attested_total_gpus"]
    != protected.get("attested_total_gpus")
    or protected_clients["slots"] != protected.get("cell_ceiling")
    or protected_clients["reserve_jobs"] != protected.get("reserve_jobs")
    or protected_clients["submit_headroom"] != protected.get("submit_headroom")
    or protected_clients["cpus"] != protected.get("cpu")
    or protected_clients["memory_mib"] != protected.get("memory_mib")
    or protected_accounting != expected_protected_accounting
    or expected_protected_accounting[
        "controller_monitor_other_held_job_elements"
    ]
    != 39
    or protected.get("fleet_contract_sha256")
    != protected.get("effective_fleet_contract_sha256")
    or protected.get("active_fleet_topology_sha256")
    != protected.get("effective_active_topology_sha256")
    or protected.get("base_active_logical_replicas") != 22
    or protected.get("base_active_gpus") != 24
    or protected.get("capacity_generation") != 1
    or protected.get("effective_active_logical_replicas") != 22
    or protected.get("effective_active_gpus") != 24
    or protected.get("additive_reserved_logical_replicas") != 0
    or protected.get("additive_reserved_gpus") != 0
    or protected.get("additive_reserved_tp1_replicas") != 0
    or protected.get("additive_reserved_tp2_replicas") != 0
    or protected.get("additive_reserved_topology") != []
    or protected.get("effective_fleet_contract_sha256")
    != protected.get("base_fleet_contract_sha256")
    or protected.get("additive_overlay_contract_path")
    != protected.get("effective_fleet_contract_path")
    or protected.get("additive_overlay_contract_sha256")
    != protected.get("base_fleet_contract_sha256")
    or protected.get("effective_active_topology")
    != protected.get("base_active_topology")
    or protected.get("attested_total_gpus") != 28
    or protected.get("active_gpus") != 24
    or protected.get("warm_headroom_gpus") != 4
    or protected.get("effective_active_logical_replicas")
    != (
        protected.get("base_active_logical_replicas")
        + protected.get("additive_reserved_logical_replicas")
    )
    or protected.get("effective_active_gpus")
    != (
        protected.get("base_active_gpus")
        + protected.get("additive_reserved_gpus")
    )
    or protected.get("additive_reserved_logical_replicas")
    != (
        protected.get("additive_reserved_tp1_replicas")
        + protected.get("additive_reserved_tp2_replicas")
    )
    or protected.get("additive_reserved_gpus")
    != (
        protected.get("additive_reserved_tp1_replicas")
        + 2 * protected.get("additive_reserved_tp2_replicas")
    )
    or protected.get("running_scientific_jobs")
    != protected_expected_running
    or protected.get("minimum_scientific_wall_seconds") != 86400
    or (
        protected_max_jobs is not None
        and (
            not isinstance(protected_max_jobs, int)
            or isinstance(protected_max_jobs, bool)
            or protected_max_jobs < protected_expected_running
        )
    )
    or protected.get("capacity_source") != "__PROTECTED_CAPACITY_SOURCE__"
    or any(
        not isinstance(protected.get(field), str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            str(protected.get(field)),
        )
        is None
        for field in (
            "scheduler_cluster", "scheduler_account", "scheduler_user",
        )
    )
    or any(
        SHA256.fullmatch(str(protected.get(field, ""))) is None
        for field in (
            "scheduler_evidence_id", "scheduler_evidence_sha256",
            "canary_id", "canary_evidence_sha256",
            "fleet_contract_sha256", "active_fleet_topology_sha256",
            "source_tree_sha256", "dispatcher_source_sha256",
            "qualification_runner_source_sha256",
        )
    )
    or protected.get("scientific_server_preempt_mode") != "OFF"
    or protected.get("scientific_client_preempt_mode") != "OFF"
    or protected.get("squeue_complete") is not True
    or protected.get("sacct_complete") is not True
):
    fail("protected capacity does not prove full non-preemptible placement")

slurm_canary = canary_source.get("slurm_canary")
if not isinstance(slurm_canary, dict):
    fail("manifest Slurm canary evidence is invalid")
manifest_canary = {
    "canary_id": slurm_canary.get("dependency_canary_id"),
    "marker_sha256": slurm_canary.get("dependency_marker_sha256"),
    "alert_latency_seconds": slurm_canary.get(
        "dependency_alert_latency_seconds"
    ),
    "alert_latency_bound_seconds": slurm_canary.get(
        "dependency_alert_latency_bound_seconds"
    ),
    "kill_invalid_depend": slurm_canary.get(
        "dependency_kill_invalid_depend"
    ),
    "root_initial_hold": slurm_canary.get(
        "dependency_root_initial_hold"
    ),
    "child_never_started": slurm_canary.get(
        "dependency_child_never_started"
    ),
}
require_canary(manifest_canary, manifest_canary)

receipt, receipt_raw = load_sealed(receipt_path, "submission receipt")
initial_fields = {
    "schema_version", "protocol", "passed", "chain_id", "manifest",
    "manifest_sha256", "submission_journal", "submission_journal_sha256",
    "submitted_at", "dependency_policy", "dependency_policy_check",
    "dependency_policy_check_sha256", "dependency_canary",
    "root_initial_hold", "no_requeue", "stage_failure_sentinels",
    "jobs", "receipt_id",
}
repair_fields = initial_fields | {
    "repair_generation", "parent_receipt", "parent_receipt_sha256",
    "held_root_names",
}
expected_receipt_fields = initial_fields if generation == 0 else repair_fields
expected_receipt_protocol = (
    "schema5-v1.2-r7-recovery-chain-submission"
    if generation == 0
    else "schema5-v1.2-r7-recovery-chain-repair"
)
if (
    set(receipt) != expected_receipt_fields
    or receipt.get("schema_version") != SCHEMA_VERSION
    or receipt.get("protocol") != expected_receipt_protocol
    or receipt.get("passed") is not True
    or receipt.get("chain_id") != chain_id
    or receipt.get("manifest") != str(manifest_path)
    or receipt.get("manifest_sha256") != digest(manifest_raw)
    or receipt.get("dependency_policy") != DEPENDENCY_POLICY
    or receipt.get("root_initial_hold") is not True
    or receipt.get("no_requeue") is not True
    or receipt.get("stage_failure_sentinels")
    != manifest.get("stage_failure_sentinels")
):
    fail("submission receipt contract drifted")
if generation > 0 and receipt.get("repair_generation") != generation:
    fail("repair receipt generation drifted")
require_identity(receipt, "receipt_id", "submission receipt")
require_canary(receipt.get("dependency_canary"), manifest_canary)
if Path(str(receipt.get("submission_journal"))) != (
    evidence_root / "__SUBMISSION_JOURNAL_NAME__"
):
    fail("submission journal path drifted")
require_backing_file(
    receipt.get("submission_journal"),
    receipt.get("submission_journal_sha256"),
    "submission journal",
    beneath=evidence_root,
)
policy = require_backing_file(
    receipt.get("dependency_policy_check"),
    receipt.get("dependency_policy_check_sha256"),
    "pre-submission dependency policy",
    beneath=evidence_root,
)
if (
    policy.get("protocol") != "schema5-v1.2-r7-live-dependency-policy-v1"
    or policy.get("phase") != "pre_submission"
    or policy.get("kill_invalid_depend") is not True
    or "kill_invalid_depend" not in policy.get("dependency_parameters", [])
):
    fail("pre-submission dependency policy is not fail closed")
require_identity(policy, "evidence_id", "pre-submission dependency policy")
if generation > 0:
    require_backing_file(
        receipt.get("parent_receipt"),
        receipt.get("parent_receipt_sha256"),
        "parent submission receipt",
    )

receipt_jobs = receipt.get("jobs")
if not isinstance(receipt_jobs, list) or len(receipt_jobs) != len(manifest_jobs):
    fail("submission receipt job cardinality drifted")
receipt_by_name: dict[str, dict[str, object]] = {}
submitted_ids: dict[str, str] = {}
initial_job_fields = {
    "name", "job_id", "dependencies", "dependency_job_ids", "comment",
    "script", "script_sha256", "dependency_type",
}
repair_job_fields = initial_job_fields | {"generation", "disposition"}
for manifest_row, record in zip(manifest_jobs, receipt_jobs, strict=True):
    expected_job_fields = (
        initial_job_fields if generation == 0 else repair_job_fields
    )
    if not isinstance(record, dict) or set(record) != expected_job_fields:
        fail("submission receipt job fields drifted")
    name = manifest_row["name"]
    record_job_id = record.get("job_id")
    dependencies = manifest_row.get("dependencies")
    if (
        record.get("name") != name
        or not isinstance(record_job_id, str)
        or not record_job_id.isdigit()
        or record_job_id in submitted_ids.values()
        or not isinstance(dependencies, list)
        or record.get("dependencies") != dependencies
        or record.get("dependency_job_ids")
        != [submitted_ids[item] for item in dependencies]
        or record.get("script") != manifest_row.get("script")
        or record.get("script_sha256") != manifest_row.get("script_sha256")
        or record.get("dependency_type") != manifest_row.get("dependency_type")
    ):
        fail(f"submission receipt job drifted: {name}")
    if generation == 0:
        record_generation = 0
    else:
        record_generation = record.get("generation")
        if (
            not isinstance(record_generation, int)
            or isinstance(record_generation, bool)
            or not 0 <= record_generation <= generation
            or record.get("disposition")
            not in {"reused_completed", "resubmitted"}
            or (
                record.get("disposition") == "resubmitted"
                and record_generation != generation
            )
        ):
            fail(f"repair receipt generation drifted: {name}")
    expected_record_comment = (
        f"asys:s5-recovery-v1.2-r7:{chain_id}:"
        f"g{record_generation:04d}:{name}"
    )
    if record.get("comment") != expected_record_comment:
        fail(f"submission receipt comment drifted: {name}")
    script_path = Path(str(record.get("script")))
    try:
        script_metadata = os.lstat(script_path)
    except OSError as error:
        fail(f"immutable sbatch is unavailable for {name}: {error}")
    if (
        stat.S_ISLNK(script_metadata.st_mode)
        or not stat.S_ISREG(script_metadata.st_mode)
        or stat.S_IMODE(script_metadata.st_mode) & 0o222
        or script_path.resolve(strict=True) != script_path
        or digest(script_path.read_bytes())
        != require_sha256(record.get("script_sha256"), f"{name} script hash")
    ):
        fail(f"immutable sbatch drifted: {name}")
    submitted_ids[name] = record_job_id
    receipt_by_name[name] = record

current = receipt_by_name.get(stage_name)
if (
    current is None
    or current.get("job_id") != job_id
    or current.get("comment") != scheduler_comment
):
    fail("current allocation is not the receipt-bound stage")
if generation > 0 and (
    current.get("generation") != generation
    or current.get("disposition") != "resubmitted"
):
    fail("current allocation is not part of this repair generation")
root_name = "source_checkout"
root_names = ["source_checkout"]
if generation > 0:
    root_names = receipt.get("held_root_names")
    if (
        not isinstance(root_names, list)
        or not root_names
        or len(root_names) != len(set(root_names))
        or any(
            not isinstance(name, str)
            or name not in receipt_by_name
            or receipt_by_name[name].get("generation") != generation
            or receipt_by_name[name].get("disposition") != "resubmitted"
            for name in root_names
        )
    ):
        fail("repair receipt held frontier drifted")
    root_name = root_names[0]
if not root_name or root_name not in receipt_by_name:
    fail("submission receipt has no held generation root")
root_record = receipt_by_name[root_name]
root_records = [receipt_by_name[name] for name in root_names]
root_job_ids = [record["job_id"] for record in root_records]
root_comments = [record["comment"] for record in root_records]

release, release_raw = load_sealed(release_path, "root release completion")
release_fields = {
    "schema_version", "protocol", "completed_at", "receipt",
    "receipt_sha256", "receipt_id", "root_name", "root_job_id",
    "root_comment", "root_names", "root_job_ids", "root_comments",
    "release_intent", "release_intent_sha256",
    "dependency_policy_check", "dependency_policy_check_sha256",
    "dependency_canary", "state_before", "state_after",
    "scheduler_observation", "roots_state_before", "roots_state_after",
    "scheduler_observations", "release_attempts", "release_reconciled",
    "root_no_longer_held", "all_roots_no_longer_held", "release_id",
}
if generation == 0:
    release_fields |= {
        "scheduler_acceptance", "scheduler_acceptance_sha256",
        "scheduler_acceptance_id",
        "bootstrap_watchdog_ready", "bootstrap_watchdog_ready_sha256",
        "bootstrap_watchdog_ready_id",
        "bootstrap_watchdog_armed", "bootstrap_watchdog_armed_sha256",
        "bootstrap_watchdog_armed_id",
    }
else:
    release_fields |= {
        "generation_scheduler_acceptance",
        "generation_scheduler_acceptance_sha256",
        "generation_scheduler_acceptance_id",
    }
    if "bootstrap_descendant_armed" in release:
        release_fields |= {
            "bootstrap_descendant_armed",
            "bootstrap_descendant_armed_sha256",
            "bootstrap_descendant_armed_id",
        }
if (
    set(release) != release_fields
    or release.get("schema_version") != SCHEMA_VERSION
    or release.get("protocol")
    != "schema5-v1.2-r7-recovery-root-release-v1"
    or release.get("receipt") != str(receipt_path)
    or release.get("receipt_sha256") != digest(receipt_raw)
    or release.get("receipt_id") != receipt.get("receipt_id")
    or release.get("root_name") != root_name
    or release.get("root_job_id") != root_record.get("job_id")
    or release.get("root_comment") != root_record.get("comment")
    or release.get("root_names") != root_names
    or release.get("root_job_ids") != root_job_ids
    or release.get("root_comments") != root_comments
    or release.get("dependency_canary") != manifest_canary
    or release.get("root_no_longer_held") is not True
    or release.get("all_roots_no_longer_held") is not True
    or not isinstance(release.get("roots_state_before"), list)
    or len(release["roots_state_before"]) != len(root_names)
    or not isinstance(release.get("roots_state_after"), list)
    or len(release["roots_state_after"]) != len(root_names)
    or not isinstance(release.get("scheduler_observations"), list)
    or len(release["scheduler_observations"]) != len(root_names)
    or release.get("state_before") != release["roots_state_before"][0]
    or release.get("state_after") != release["roots_state_after"][0]
    or release.get("scheduler_observation")
    != release["scheduler_observations"][0]
    or not isinstance(release.get("release_attempts"), list)
    or not release["release_attempts"]
):
    fail("root release completion drifted")
require_identity(release, "release_id", "root release completion")
if generation == 0:
    bootstrap_armed = require_backing_file(
        release.get("bootstrap_watchdog_armed"),
        release.get("bootstrap_watchdog_armed_sha256"),
        "bootstrap watchdog armed marker",
        beneath=evidence_root,
    )
    bootstrap_namespace = bootstrap_armed.get("job_namespace")
    if (
        bootstrap_armed.get("schema_version") != 1
        or bootstrap_armed.get("protocol")
        != "__BOOTSTRAP_WATCHDOG_ARMED_PROTOCOL__"
        or bootstrap_armed.get("passed") is not True
        or bootstrap_armed.get("release_id") != "__RELEASE_ID__"
        or bootstrap_armed.get("release_tag") != "__RELEASE_TAG__"
        or bootstrap_armed.get("release_git_commit")
        != manifest.get("release_git_commit")
        or bootstrap_armed.get("release_tag_object")
        != manifest.get("release_tag_object")
        or bootstrap_armed.get("chain_namespace")
        != "__CHAIN_NAMESPACE__"
        or bootstrap_armed.get("chain_manifest") != str(manifest_path)
        or bootstrap_armed.get("chain_manifest_sha256")
        != digest(manifest_raw)
        or bootstrap_armed.get("chain_id") != chain_id
        or bootstrap_armed.get("submission_receipt") != str(receipt_path)
        or bootstrap_armed.get("submission_receipt_sha256")
        != digest(receipt_raw)
        or bootstrap_armed.get("submission_receipt_id")
        != receipt.get("receipt_id")
        or bootstrap_armed.get("root_name") != root_name
        or bootstrap_armed.get("root_job_id")
        != root_record.get("job_id")
        or bootstrap_armed.get("root_comment")
        != root_record.get("comment")
        or bootstrap_armed.get("job_count") != len(receipt_jobs)
        or not isinstance(bootstrap_namespace, list)
        or bootstrap_namespace
        != [
            {
                "name": row["name"],
                "job_id": row["job_id"],
                "comment": row["comment"],
            }
            for row in receipt_jobs
        ]
        or bootstrap_armed.get("forced_command_only") is not True
        or bootstrap_armed.get("forced_operation") != "bootstrap-repair"
        or bootstrap_armed.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or bootstrap_armed.get("prelaunch_root_release") is not False
        or bootstrap_armed.get("descendant_rearm_authority") is not True
        or bootstrap_armed.get("release_intent_continuation_authority")
        is not True
        or bootstrap_armed.get("scientific_admission_direct") is not False
        or bootstrap_armed.get("production_control_mutation") is not False
        or bootstrap_armed.get("safety_hold_clear_authority") is not False
        or bootstrap_armed.get("handoff_stage") != "watchdog_readiness"
        or bootstrap_armed.get("bootstrap_watchdog_ready")
        != release.get("bootstrap_watchdog_ready")
        or bootstrap_armed.get("bootstrap_watchdog_ready_sha256")
        != release.get("bootstrap_watchdog_ready_sha256")
        or bootstrap_armed.get("bootstrap_watchdog_ready_id")
        != release.get("bootstrap_watchdog_ready_id")
        or bootstrap_armed.get("marker_id")
        != release.get("bootstrap_watchdog_armed_id")
    ):
        fail("bootstrap watchdog armed marker drifted")
    require_identity(
        bootstrap_armed, "marker_id", "bootstrap watchdog armed marker"
    )
    bootstrap_ready = require_backing_file(
        bootstrap_armed.get("bootstrap_watchdog_ready"),
        bootstrap_armed.get("bootstrap_watchdog_ready_sha256"),
        "bootstrap watchdog deployment readiness",
        beneath=evidence_root,
    )
    if (
        bootstrap_ready.get("protocol")
        != "__BOOTSTRAP_WATCHDOG_READY_PROTOCOL__"
        or bootstrap_ready.get("ready_id")
        != bootstrap_armed.get("bootstrap_watchdog_ready_id")
        or bootstrap_ready.get("separate_bootstrap_key") is not True
        or bootstrap_ready.get("forced_command_only") is not True
        or bootstrap_ready.get("allowed_operations")
        != ["bootstrap-status", "bootstrap-repair"]
        or bootstrap_ready.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or bootstrap_ready.get("prelaunch_root_release") is not False
        or bootstrap_ready.get("descendant_rearm_authority") is not True
        or bootstrap_ready.get("release_intent_continuation_authority")
        is not True
        or bootstrap_ready.get("scientific_admission_direct") is not False
        or bootstrap_ready.get("production_control_mutation") is not False
        or bootstrap_ready.get("safety_hold_clear_authority") is not False
    ):
        fail("bootstrap watchdog deployment readiness drifted")
    require_identity(
        bootstrap_ready, "ready_id",
        "bootstrap watchdog deployment readiness",
    )
    scheduler_acceptance = require_backing_file(
        release.get("scheduler_acceptance"),
        release.get("scheduler_acceptance_sha256"),
        "scheduler acceptance evidence",
        beneath=evidence_root,
    )
    if (
        scheduler_acceptance.get("protocol")
        != "__SCHEDULER_ACCEPTANCE_PROTOCOL__"
        or scheduler_acceptance.get("passed") is not True
        or scheduler_acceptance.get("chain_id") != chain_id
        or scheduler_acceptance.get("submission_receipt") != str(receipt_path)
        or scheduler_acceptance.get("submission_receipt_id")
        != receipt.get("receipt_id")
        or scheduler_acceptance.get("job_count") != len(manifest_jobs)
        or scheduler_acceptance.get("acceptance_id")
        != release.get("scheduler_acceptance_id")
    ):
        fail("scheduler acceptance evidence drifted")
    require_identity(
        scheduler_acceptance, "acceptance_id", "scheduler acceptance evidence"
    )
else:
    generation_acceptance = require_backing_file(
        release.get("generation_scheduler_acceptance"),
        release.get("generation_scheduler_acceptance_sha256"),
        "repair generation scheduler acceptance",
        beneath=evidence_root,
    )
    if (
        generation_acceptance.get("protocol")
        != "schema5-v1.2-r7-bootstrap-generation-provenance-v1"
        or generation_acceptance.get("passed") is not True
        or generation_acceptance.get("chain_id") != chain_id
        or generation_acceptance.get("submission_receipt")
        != str(receipt_path)
        or generation_acceptance.get("submission_receipt_id")
        != receipt.get("receipt_id")
        or generation_acceptance.get("repair_generation") != generation
        or generation_acceptance.get("generation_root_names")
        != root_names
        or generation_acceptance.get("scheduler_topology_valid") is not True
        or generation_acceptance.get("provenance_id")
        != release.get("generation_scheduler_acceptance_id")
    ):
        fail("repair generation scheduler acceptance drifted")
    require_identity(
        generation_acceptance,
        "provenance_id",
        "repair generation scheduler acceptance",
    )
    if "bootstrap_descendant_armed" in release:
        descendant_armed = require_backing_file(
            release.get("bootstrap_descendant_armed"),
            release.get("bootstrap_descendant_armed_sha256"),
            "bootstrap descendant armed marker",
            beneath=evidence_root,
        )
        if (
            descendant_armed.get("protocol")
            != "__BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL__"
            or descendant_armed.get("passed") is not True
            or descendant_armed.get("chain_id") != chain_id
            or descendant_armed.get("submission_receipt_id")
            != receipt.get("receipt_id")
            or descendant_armed.get("repair_generation") != generation
            or descendant_armed.get("root_names") != root_names
            or descendant_armed.get("root_job_ids") != root_job_ids
            or descendant_armed.get("marker_id")
            != release.get("bootstrap_descendant_armed_id")
        ):
            fail("bootstrap descendant armed marker drifted")
        require_identity(
            descendant_armed,
            "marker_id",
            "bootstrap descendant armed marker",
        )
require_backing_file(
    release.get("release_intent"),
    release.get("release_intent_sha256"),
    "root release intent",
    beneath=evidence_root,
)
release_policy = require_backing_file(
    release.get("dependency_policy_check"),
    release.get("dependency_policy_check_sha256"),
    "root-release dependency policy",
    beneath=evidence_root,
)
if (
    release_policy.get("protocol")
    != "schema5-v1.2-r7-live-dependency-policy-v1"
    or release_policy.get("phase") != "root_release"
    or release_policy.get("kill_invalid_depend") is not True
    or "kill_invalid_depend"
    not in release_policy.get("dependency_parameters", [])
):
    fail("root-release dependency policy is not fail closed")
require_identity(
    release_policy, "evidence_id", "root-release dependency policy"
)
covered_release_roots = set()
successful_release_roots = set()
for ordinal, attempt_binding in enumerate(
    release["release_attempts"], start=1
):
    if (
        not isinstance(attempt_binding, dict)
        or set(attempt_binding)
        != {
            "attempt", "root_name", "job_id", "intent",
            "intent_sha256", "result", "result_sha256",
            "result_id", "result_kind",
        }
        or attempt_binding.get("attempt") != ordinal
        or attempt_binding.get("root_name") not in root_names
    ):
        fail("root release attempt summary drifted")
    release_root_index = root_names.index(
        attempt_binding["root_name"]
    )
    attempt_intent = require_backing_file(
        attempt_binding.get("intent"),
        attempt_binding.get("intent_sha256"),
        "root release attempt intent",
        beneath=evidence_root,
    )
    attempt_result = require_backing_file(
        attempt_binding.get("result"),
        attempt_binding.get("result_sha256"),
        "root release attempt result",
        beneath=evidence_root,
    )
    if (
        attempt_intent.get("protocol")
        != "schema5-v1.2-r7-root-release-attempt-intent-v1"
        or attempt_intent.get("attempt") != ordinal
        or attempt_intent.get("root_index") != release_root_index
        or attempt_intent.get("root_name")
        != attempt_binding.get("root_name")
        or attempt_intent.get("job_id")
        != root_job_ids[release_root_index]
        or attempt_intent.get("command")
        != [
            "scontrol", "release",
            root_job_ids[release_root_index],
        ]
        or attempt_intent.get("dependency_policy_check")
        != release.get("dependency_policy_check")
        or attempt_intent.get("dependency_policy_check_sha256")
        != release.get("dependency_policy_check_sha256")
        or attempt_result.get("protocol")
        != "schema5-v1.2-r7-root-release-attempt-result-v1"
        or attempt_result.get("attempt") != ordinal
        or attempt_result.get("root_index") != release_root_index
        or attempt_result.get("root_name")
        != attempt_binding.get("root_name")
        or attempt_result.get("job_id")
        != attempt_binding.get("job_id")
        or attempt_result.get("intent")
        != attempt_binding.get("intent")
        or attempt_result.get("intent_sha256")
        != attempt_binding.get("intent_sha256")
        or attempt_result.get("result_id")
        != attempt_binding.get("result_id")
        or attempt_result.get("result_kind")
        != attempt_binding.get("result_kind")
    ):
        fail("root release attempt lineage drifted")
    require_identity(
        attempt_result, "result_id", "root release attempt result"
    )
    result_kind = attempt_result.get("result_kind")
    if result_kind == "scontrol_result":
        if (
            not isinstance(attempt_result.get("returncode"), int)
            or isinstance(attempt_result.get("returncode"), bool)
            or not isinstance(attempt_result.get("stdout"), str)
            or not isinstance(attempt_result.get("stderr"), str)
            or attempt_result.get("stdout_sha256")
            != digest(attempt_result["stdout"].encode())
            or attempt_result.get("stderr_sha256")
            != digest(attempt_result["stderr"].encode())
            or attempt_result.get("scheduler_reconciliation") is not None
        ):
            fail("direct root release attempt result drifted")
        if attempt_result["returncode"] == 0:
            successful_release_roots.add(
                attempt_binding["root_name"]
            )
    elif (
        result_kind
        == "scheduler_reconciled_after_ambiguous_release"
    ):
        if (
            attempt_result.get("returncode") is not None
            or attempt_result.get("stdout") is not None
            or attempt_result.get("stderr") is not None
            or attempt_result.get("stdout_sha256") is not None
            or attempt_result.get("stderr_sha256") is not None
            or attempt_result.get("scheduler_reconciliation")
            != release["roots_state_before"][release_root_index]
        ):
            fail("adopted root release attempt result drifted")
        successful_release_roots.add(attempt_binding["root_name"])
    else:
        fail("root release attempt result kind drifted")
    covered_release_roots.add(attempt_binding["root_name"])
if (
    covered_release_roots != set(root_names)
    or successful_release_roots != set(root_names)
):
    fail("root release attempts do not cover every frontier root")

launch, _ = load_sealed(launch_path, "launch completion")
launch_fields = {
    "schema_version", "protocol", "completed_at", "receipt",
    "receipt_sha256", "receipt_id", "root_release",
    "root_release_sha256", "root_release_id", "dependency_policy",
    "dependency_canary", "root_initial_hold",
    "alert_latency_bound_seconds", "launch_id",
}
if generation == 0:
    launch_fields |= {
        "bootstrap_watchdog_ready", "bootstrap_watchdog_ready_sha256",
        "bootstrap_watchdog_ready_id",
        "bootstrap_watchdog_armed", "bootstrap_watchdog_armed_sha256",
        "bootstrap_watchdog_armed_id",
    }
else:
    launch_fields |= {
        "generation_scheduler_acceptance",
        "generation_scheduler_acceptance_sha256",
        "generation_scheduler_acceptance_id",
    }
    if "bootstrap_descendant_armed" in release:
        launch_fields |= {
            "bootstrap_descendant_armed",
            "bootstrap_descendant_armed_sha256",
            "bootstrap_descendant_armed_id",
        }
if (
    set(launch) != launch_fields
    or launch.get("schema_version") != SCHEMA_VERSION
    or launch.get("protocol")
    != "schema5-v1.2-r7-recovery-chain-launched-v1"
    or launch.get("receipt") != str(receipt_path)
    or launch.get("receipt_sha256") != digest(receipt_raw)
    or launch.get("receipt_id") != receipt.get("receipt_id")
    or launch.get("root_release") != str(release_path)
    or launch.get("root_release_sha256") != digest(release_raw)
    or launch.get("root_release_id") != release.get("release_id")
    or launch.get("dependency_policy") != DEPENDENCY_POLICY
    or launch.get("dependency_canary") != manifest_canary
    or launch.get("root_initial_hold") is not True
    or launch.get("alert_latency_bound_seconds") != 180.0
    or (
        generation == 0
        and (
            launch.get("bootstrap_watchdog_ready")
            != release.get("bootstrap_watchdog_ready")
            or launch.get("bootstrap_watchdog_ready_sha256")
            != release.get("bootstrap_watchdog_ready_sha256")
            or launch.get("bootstrap_watchdog_ready_id")
            != release.get("bootstrap_watchdog_ready_id")
            or launch.get("bootstrap_watchdog_armed")
            != release.get("bootstrap_watchdog_armed")
            or launch.get("bootstrap_watchdog_armed_sha256")
            != release.get("bootstrap_watchdog_armed_sha256")
            or launch.get("bootstrap_watchdog_armed_id")
            != release.get("bootstrap_watchdog_armed_id")
        )
    )
    or (
        generation > 0
        and any(
            launch.get(field) != release.get(field)
            for field in (
                "generation_scheduler_acceptance",
                "generation_scheduler_acceptance_sha256",
                "generation_scheduler_acceptance_id",
                "bootstrap_descendant_armed",
                "bootstrap_descendant_armed_sha256",
                "bootstrap_descendant_armed_id",
            )
            if field in launch_fields
        )
    )
):
    fail("launch completion drifted")
require_identity(launch, "launch_id", "launch completion")
print(
    f"launch authorization passed for generation {generation}, "
    f"stage {stage_name}, job {job_id}"
)
PY
"""
    replacements = {
        "__BOOTSTRAP_RUNTIME__": _sentinel_bootstrap_runtime_shell(paths).rstrip(),
        "__SPEC_NAME__": _q(spec_name),
        "__CHAIN_MANIFEST__": _q(paths.chain_manifest),
        "__RECOVERY_ROOT__": _q(paths.recovery_root),
        "__REPAIR_ROOT__": _q(paths.recovery_root / REPAIR_ROOT_NAME),
        "__SUBMISSION_RECEIPT_NAME__": SUBMISSION_RECEIPT_NAME,
        "__SUBMISSION_JOURNAL_NAME__": SUBMISSION_JOURNAL_NAME,
        "__ROOT_RELEASE_COMPLETE_NAME__": ROOT_RELEASE_COMPLETE_NAME,
        "__LAUNCH_COMPLETE_NAME__": LAUNCH_COMPLETE_NAME,
        "__LAUNCH_GATE_TIMEOUT_SECONDS__": str(LAUNCH_GATE_TIMEOUT_SECONDS),
        "__CHAIN_SCHEMA_VERSION__": str(CHAIN_SCHEMA_VERSION),
        "__SUBMISSION_SCHEMA_VERSION__": str(SUBMISSION_SCHEMA_VERSION),
        "__SCHEDULER_ACCEPTANCE_PROTOCOL__": SCHEDULER_ACCEPTANCE_PROTOCOL,
        "__BOOTSTRAP_WATCHDOG_READY_PROTOCOL__": (
            BOOTSTRAP_WATCHDOG_READY_PROTOCOL
        ),
        "__BOOTSTRAP_WATCHDOG_ARMED_PROTOCOL__": (
            BOOTSTRAP_WATCHDOG_ARMED_PROTOCOL
        ),
        "__BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL__": (
            BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL
        ),
        "__RELEASE_ID__": RELEASE_ID,
        "__RELEASE_TAG__": RELEASE_TAG,
        "__CHAIN_NAMESPACE__": CHAIN_NAMESPACE,
        "__PROTECTED_CAPACITY_MARKER_NAME__": (
            PROTECTED_CAPACITY_MARKER_NAME
        ),
        "__PROTECTED_CAPACITY_PROTOCOL__": PROTECTED_CAPACITY_PROTOCOL,
        "__PROTECTED_CAPACITY_SCHEMA_VERSION__": str(
            protected_capacity.SCHEMA_VERSION
        ),
        "__PROTECTED_CAPACITY_SOURCE__": PROTECTED_CAPACITY_SOURCE,
        "__SUPERSEDED_R2_CANARY_FAILURE_RELATIVE_PATH__": (
            SUPERSEDED_R2_CANARY_FAILURE_RELATIVE_PATH.as_posix()
        ),
        "__SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL__": (
            SUPERSEDED_R2_CANARY_FAILURE_PROTOCOL
        ),
        "__SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID__": (
            SUPERSEDED_R2_CANARY_FAILURE_SEAL_ID
        ),
        "__SUPERSEDED_R2_RELEASE_TAG__": SUPERSEDED_R2_RELEASE_TAG,
        "__SUPERSEDED_R2_RELEASE_COMMIT__": (
            SUPERSEDED_R2_RELEASE_COMMIT
        ),
        "__SUPERSEDED_R2_RELEASE_TAG_OBJECT__": (
            SUPERSEDED_R2_RELEASE_TAG_OBJECT
        ),
        "__SUPERSEDED_R2_CANARY_JOB_ID__": (
            SUPERSEDED_R2_CANARY_JOB_ID
        ),
        "__EXTERNAL_WATCHDOG_DRILL_MARKER_NAME__": (
            EXTERNAL_WATCHDOG_DRILL_MARKER_NAME
        ),
        "__EXTERNAL_WATCHDOG_DRILL_PROTOCOL__": (
            EXTERNAL_WATCHDOG_DRILL_PROTOCOL
        ),
        "__WATCHDOG_READY_MARKER_NAME__": WATCHDOG_READY_MARKER_NAME,
        "__WATCHDOG_READY_PROTOCOL__": WATCHDOG_READY_PROTOCOL,
        "__DEPENDENCY_POLICY_CONTRACT__": json.dumps(
            DEPENDENCY_POLICY_CONTRACT
        ),
    }
    for placeholder, replacement in replacements.items():
        template = template.replace(placeholder, replacement)
    unresolved = [placeholder for placeholder in replacements if placeholder in template]
    if unresolved:
        raise ChainError(
            f"unresolved recovery launch-gate placeholders: {unresolved}"
        )
    return template


def _source_checkout_seal_contract(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SOURCE_CHECKOUT_SEAL_PROTOCOL,
        "source_checkout": str(paths.source_checkout),
        "release_tag": RELEASE_TAG,
        "release_git_commit": commit,
        "release_tag_object": tag_object,
        "sealed_read_only": True,
        "verification": "git-diff-index-status-fsck-at-every-pre-freeze-stage",
    }
    identity["seal_id"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _source_checkout_guard_shell(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
) -> str:
    marker = paths.recovery_root / SOURCE_CHECKOUT_SEAL_NAME
    payload = _canonical_json(
        _source_checkout_seal_contract(
            paths, commit=commit, tag_object=tag_object
        )
    ) + b"\n"
    return f"""\
sealed_source_checkout={_q(paths.source_checkout)}
source_checkout_seal={_q(marker)}
[[ -d "$sealed_source_checkout" && ! -L "$sealed_source_checkout" && \
   "$(realpath -e -- "$sealed_source_checkout")" == "$sealed_source_checkout" ]] || {{
  echo "sealed source checkout is missing or unsafe" >&2
  exit 2
}}
[[ -f "$source_checkout_seal" && ! -L "$source_checkout_seal" ]] || {{
  echo "source checkout seal is missing or unsafe" >&2
  exit 2
}}
[[ "$(stat -c '%s' -- "$source_checkout_seal")" == {_q(len(payload))} && \
   "$(sha256sum -- "$source_checkout_seal" | cut -d' ' -f1)" == {_q(_sha256_bytes(payload))} ]] || {{
  echo "source checkout seal identity drifted" >&2
  exit 2
}}
[[ -z "$(find "$sealed_source_checkout" -perm /222 -print -quit)" ]] || {{
  echo "sealed source checkout contains writable state" >&2
  exit 2
}}
export GIT_OPTIONAL_LOCKS=0
[[ "$(git -C "$sealed_source_checkout" cat-file -t "refs/tags/{RELEASE_TAG}")" == tag ]]
[[ "$(git -C "$sealed_source_checkout" rev-parse "refs/tags/{RELEASE_TAG}")" == {_q(tag_object)} ]]
[[ "$(git -C "$sealed_source_checkout" rev-parse HEAD)" == {_q(commit)} ]]
[[ "$(git -C "$sealed_source_checkout" rev-parse "{RELEASE_TAG}^{{commit}}")" == {_q(commit)} ]]
[[ -z "$(git -C "$sealed_source_checkout" status --porcelain=v1 --untracked-files=all)" ]]
git -C "$sealed_source_checkout" diff --no-ext-diff --quiet HEAD --
git -C "$sealed_source_checkout" diff --cached --no-ext-diff --quiet
git -C "$sealed_source_checkout" fsck --full --strict
"""


def _source_checkout_body(
    paths: RecoveryPaths,
    commit: str,
    *,
    tag_object: str,
    verifier_payload: bytes,
) -> str:
    verifier_sha256 = _sha256_bytes(verifier_payload)
    verifier_size = len(verifier_payload)
    seal = _source_checkout_seal_contract(
        paths, commit=commit, tag_object=tag_object
    )
    seal_payload = _canonical_json(seal) + b"\n"
    return _common_exports(paths) + f"""\
repository={_q(paths.repository)}
target={_q(paths.source_checkout)}
release_tag={_q(RELEASE_TAG)}
expected_commit={_q(commit)}
expected_tag_object={_q(tag_object)}
chain_manifest={_q(paths.chain_manifest)}
prerequisite_verifier={_q(paths.jobs_root / Path(__file__).name)}
target_prerequisite_verifier="$target/scripts/render_schema5_recovery_chain_v12.py"
temporary="${{target}}.clone.${{SLURM_JOB_ID:-manual}}"
[[ "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || {{
  echo "source checkout requires its exact Slurm job identity" >&2
  exit 2
}}
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
comment="$(tr ' ' '\\n' <<<"$job_record" | sed -n 's/^Comment=//p')"
[[ "$(wc -l <<<"$comment")" -eq 1 ]] || {{
  echo "source checkout scheduler comment is ambiguous" >&2
  exit 2
}}
if [[ "$comment" =~ ^asys:s5-recovery-v1\\.2-r7:([0-9a-f]{{64}}):g[0-9]{{4}}:source_checkout$ ]]; then
  scheduler_chain_id="${{BASH_REMATCH[1]}}"
else
  echo "source checkout scheduler identity is outside the immutable r7 chain" >&2
  exit 2
fi
# Authenticate the sealed interpreter and its complete stdlib/native-library
# payload before running either prerequisite verifier.
{_sentinel_bootstrap_runtime_shell(paths)}
# Independently authenticate the bundled verifier before its first execution.  This
# check is spooled into the sbatch file, so it does not trust the manifest parser it
# is about to invoke.
{_immutable_tool_shell_check(
    variable="prerequisite_verifier",
    expected_sha256=verifier_sha256,
    expected_size=verifier_size,
    description="bundled prerequisite verifier",
)}
# This marker/hash check precedes both the idempotent early exit and every checkout
# mutation.
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
  "$prerequisite_verifier" verify-prerequisites \
  --chain-manifest "$chain_manifest" \
  --expected-chain-id "$scheduler_chain_id" --markers-only
verify_target() {{
  [[ -d "$target" && ! -L "$target" ]]
  [[ ! -s "$target/.git/objects/info/alternates" ]]
  [[ "$(git -C "$target" cat-file -t "refs/tags/$release_tag")" == tag ]]
  [[ "$(git -C "$target" rev-parse "refs/tags/$release_tag")" == "$expected_tag_object" ]]
  [[ "$(git -C "$target" rev-parse HEAD)" == "$expected_commit" ]]
  [[ "$(git -C "$target" rev-parse "${{release_tag}}^{{commit}}")" == "$expected_commit" ]]
  [[ -z "$(git -C "$target" status --porcelain=v1 --untracked-files=all)" ]]
  git -C "$target" fsck --full --strict
}}
seal_target() {{
  find "$target" -depth ! -type l -exec chmod a-w -- {{}} +
  [[ -z "$(find "$target" -perm /222 -print -quit)" ]]
  env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \
    {_q(paths.recovery_root / SOURCE_CHECKOUT_SEAL_NAME)} \
    {_q(seal_payload.decode("utf-8"))} <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = sys.argv[2].encode("utf-8")
if path.exists() or path.is_symlink():
    if path.is_symlink() or path.read_bytes() != payload:
        raise SystemExit("source checkout seal conflicts with completed checkout")
else:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
PY
}}
verify_sealed_target() {{
{_source_checkout_guard_shell(
    paths,
    commit=commit,
    tag_object=tag_object,
)}
}}
verify_full_prerequisites() {{
{_immutable_tool_shell_check(
    variable="target_prerequisite_verifier",
    expected_sha256=verifier_sha256,
    expected_size=verifier_size,
    description="tagged-checkout prerequisite verifier",
)}
  env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
    "$target_prerequisite_verifier" \
    verify-prerequisites --chain-manifest "$chain_manifest" \
    --expected-chain-id "$scheduler_chain_id" --verifier-checkout "$target"
}}
if [[ -e "$target" || -L "$target" ]]; then
  verify_target
  if [[ -f {_q(paths.recovery_root / SOURCE_CHECKOUT_SEAL_NAME)} && \
        ! -L {_q(paths.recovery_root / SOURCE_CHECKOUT_SEAL_NAME)} ]]; then
    verify_sealed_target
    verify_full_prerequisites
    exit 0
  fi
  verify_full_prerequisites
  seal_target
  verify_sealed_target
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
verify_full_prerequisites
seal_target
verify_sealed_target
"""


def _maintenance_body(paths: RecoveryPaths, slurm_user: str) -> str:
    return _common_exports(paths) + _sentinel_bootstrap_runtime_shell(paths) + f"""\
exec env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \
  {_q(paths.results_root)} {_q(paths.recovery_root)} {_q(slurm_user)} <<'PY'
{MAINTENANCE_PREFLIGHT_PYTHON.rstrip()}
PY
"""


def _r1_evidence_contract(paths: RecoveryPaths) -> dict[str, dict[str, Any]]:
    """Bind every immutable r1 artifact needed to supersede the failed release."""

    relative_paths = {
        **R1_EVIDENCE_RELATIVE_PATHS,
        "snapshot_completion": "pre_repair/SNAPSHOT_COMPLETE.json",
        "snapshot_attestation": "pre_repair.attestation.json",
    }
    contract: dict[str, dict[str, Any]] = {}
    for name, relative in relative_paths.items():
        path = paths.recovery_root / relative
        if path.is_symlink() or not path.is_file():
            raise ChainError(f"required sealed r1 evidence is missing or unsafe: {path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ChainError(f"required sealed r1 evidence is writable: {path}")
        contract[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    if (
        contract["snapshot_completion"]["sha256"]
        != SEALED_SNAPSHOT_CONTRACT["completion_sha256"]
        or contract["snapshot_attestation"]["sha256"]
        != SEALED_SNAPSHOT_CONTRACT["attestation_sha256"]
    ):
        raise ChainError("sealed pre-repair snapshot identity drifted")
    failure = _read_json(
        Path(contract["failure_envelope"]["path"]),
        description="r1 failure envelope",
    )
    if (
        failure.get("classification") != "requires_superseding_release"
        or failure.get("retry_same_generation") is not False
        or failure.get("superseded_by") != RELEASE_ID
    ):
        raise ChainError("r1 failure envelope does not require this superseding release")
    idempotency = _read_json(
        Path(contract["quarantine_idempotency_receipt"]["path"]),
        description="r1 quarantine idempotency receipt",
    )
    idempotency_identity = dict(idempotency)
    idempotency_id = idempotency_identity.pop("receipt_id", None)
    repeats = (
        idempotency.get("first_repeat"),
        idempotency.get("second_repeat"),
    )
    sealed_quarantine = idempotency.get("sealed_quarantine")
    if (
        idempotency.get("protocol")
        != "schema5-v1.2-r2-r1-quarantine-idempotency-receipt-v1"
        or idempotency.get("passed") is not True
        or idempotency.get("sealed_tree_mutated") is not False
        or not all(isinstance(item, dict) for item in repeats)
        or repeats[0] != repeats[1]
        or repeats[0].get("status") != "already_quarantined"
        or not isinstance(sealed_quarantine, dict)
        or sealed_quarantine.get("seal", {}).get("sha256")
        != contract["quarantine_seal"]["sha256"]
        or sealed_quarantine.get("inventory", {}).get("sha256")
        != contract["quarantine_inventory"]["sha256"]
        or sealed_quarantine.get("completion", {}).get("sha256")
        != contract["quarantine_completion"]["sha256"]
        or idempotency.get("r1_renderer", {}).get("git_commit")
        != "a5cd9305e8fd741ba59bc14d9d296e3ecb5c5f96"
        or idempotency_id != _sha256_bytes(_canonical_json(idempotency_identity))
    ):
        raise ChainError("r1 quarantine idempotency receipt is invalid")
    zero_mutation = _read_json(
        Path(contract["zero_result_mutation_receipt"]["path"]),
        description="r1 zero-result-mutation receipt",
    )
    zero_identity = dict(zero_mutation)
    zero_id = zero_identity.pop("receipt_id", None)
    if (
        zero_mutation.get("protocol")
        != "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1"
        or zero_mutation.get("passed") is not True
        or zero_mutation.get("failed_stage") != "release_materialize"
        or zero_mutation.get("first_result_mutating_stage")
        != "legacy_consolidate"
        or zero_mutation.get("started_result_mutating_stages") != []
        or zero_mutation.get("legacy_result_mutation_count") != 0
        or zero_mutation.get("schema5_result_mutation_count") != 0
        or zero_mutation.get("r1_renderer", {}).get("git_commit")
        != "a5cd9305e8fd741ba59bc14d9d296e3ecb5c5f96"
        or zero_mutation.get("r1_manifest", {}).get("sha256")
        != contract["chain_manifest"]["sha256"]
        or zero_mutation.get("r1_submission_receipt", {}).get("sha256")
        != contract["submission_receipt"]["sha256"]
        or zero_mutation.get("r1_failure_envelope", {}).get("sha256")
        != contract["failure_envelope"]["sha256"]
        or zero_id != _sha256_bytes(_canonical_json(zero_identity))
    ):
        raise ChainError("r1 zero-result-mutation receipt is invalid")
    return contract


def _invoke_protocol_aware_r1_verifier(
    paths: RecoveryPaths,
    *,
    checkout: Path,
    expected_commit: str,
    expected_tag_object: str,
    verifier_python: Path,
    verifier_library: Path,
) -> dict[str, Any]:
    """Run the native r1 verifier from authenticated release-tagged code."""

    _verify_exact_tag_checkout(
        checkout,
        expected_commit=expected_commit,
        expected_tag_object=expected_tag_object,
    )
    for git_path in R1_PROTOCOL_TOOL_GIT_PATHS:
        expected = _tagged_file_bytes(checkout, expected_commit, git_path)
        source = _require_canonical_path(
            checkout / git_path,
            description=f"tagged r1 evidence verifier code {git_path}",
            kind="file",
        )
        if (
            source.stat().st_size != len(expected)
            or _sha256(source) != _sha256_bytes(expected)
        ):
            raise ChainError(f"tagged r1 evidence verifier code drifted: {git_path}")

    python = _require_canonical_path(
        verifier_python,
        description="sealed r1 evidence verifier Python",
        kind="file",
    )
    library = _require_canonical_path(
        verifier_library,
        description="sealed r1 evidence verifier library",
        kind="directory",
    )
    if not os.access(python, os.X_OK):
        raise ChainError(f"sealed r1 evidence verifier Python is not executable: {python}")
    return _run_json_verifier(
        [
            str(python),
            "-I",
            str(checkout / R1_EVIDENCE_VERIFIER_GIT_PATH),
            "--chain-manifest",
            str(paths.recovery_root / R1_EVIDENCE_RELATIVE_PATHS["chain_manifest"]),
            "--submission-receipt",
            str(
                paths.recovery_root
                / R1_EVIDENCE_RELATIVE_PATHS["submission_receipt"]
            ),
        ],
        environment=_sanitized_python_environment(library=library),
        timeout=600.0,
        description="protocol-aware immutable r1 evidence verifier",
    )


def _r1_protocol_verification_contract(
    paths: RecoveryPaths,
    *,
    checkout: Path,
    git_identity: Mapping[str, str],
    verifier_python: Path,
    verifier_library: Path,
) -> dict[str, Any]:
    """Bind native protocol validation, not merely hashes of historical r1 files."""

    r1_evidence = _r1_evidence_contract(paths)
    report = _invoke_protocol_aware_r1_verifier(
        paths,
        checkout=checkout,
        expected_commit=git_identity["git_commit"],
        expected_tag_object=git_identity["tag_object"],
        verifier_python=verifier_python,
        verifier_library=verifier_library,
    )
    expected_manifest = r1_evidence["chain_manifest"]
    expected_receipt = r1_evidence["submission_receipt"]
    chain_report = report.get("chain_report")
    if (
        report.get("passed") is not True
        or report.get("schema_version") != 1
        or report.get("protocol") != R1_EVIDENCE_DISPATCH_PROTOCOL
        or report.get("chain_protocol") != R1_CHAIN_PROTOCOL
        or report.get("renderer") != "render_schema5_recovery_chain"
        or report.get("manifest_path") != expected_manifest["path"]
        or report.get("manifest_sha256") != expected_manifest["sha256"]
        or report.get("submission_receipt_path") != expected_receipt["path"]
        or report.get("submission_receipt_sha256") != expected_receipt["sha256"]
        or not isinstance(chain_report, dict)
        or chain_report.get("passed") is not True
    ):
        raise ChainError(
            "protocol-aware r1 verification does not validate the exact sealed history"
        )

    tool_records = []
    for git_path in R1_PROTOCOL_TOOL_GIT_PATHS:
        payload = _tagged_file_bytes(
            checkout,
            git_identity["git_commit"],
            git_path,
        )
        tool_records.append(
            {
                "git_path": git_path,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
        )
    identity: dict[str, Any] = {
        "schema_version": 1,
        "protocol": R1_PROTOCOL_VERIFICATION_PROTOCOL,
        "release_tag": git_identity["release_tag"],
        "release_git_commit": git_identity["git_commit"],
        "release_tag_object": git_identity["tag_object"],
        "verifier_runtime": {
            "python_path": str(verifier_python),
            "python_sha256": _sha256(verifier_python),
            "python_size": verifier_python.stat().st_size,
            "library_path": str(verifier_library),
        },
        "tagged_tools": tool_records,
        "native_report": report,
        "native_report_sha256": _sha256_bytes(_canonical_json(report)),
    }
    identity["verification_id"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _verify_sealed_r1_protocol_contract(
    paths: RecoveryPaths,
    contract: Any,
    *,
    git_identity: Mapping[str, str],
    bundled_tools: Mapping[str, bytes],
) -> dict[str, Any]:
    """Re-run native r1 validation using only sealed bootstrap/bundle bytes."""

    required = {
        "schema_version",
        "protocol",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "verifier_runtime",
        "tagged_tools",
        "native_report",
        "native_report_sha256",
        "verification_id",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ChainError("native r1 protocol-verification fields drifted")
    identity = dict(contract)
    verification_id = identity.pop("verification_id", None)
    expected_tool_records = []
    for git_path in R1_PROTOCOL_TOOL_GIT_PATHS:
        payload = bundled_tools[Path(git_path).name]
        expected_tool_records.append(
            {
                "git_path": git_path,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
        )
    report = contract.get("native_report")
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != R1_PROTOCOL_VERIFICATION_PROTOCOL
        or contract.get("release_tag") != git_identity["release_tag"]
        or contract.get("release_git_commit") != git_identity["git_commit"]
        or contract.get("release_tag_object") != git_identity["tag_object"]
        or contract.get("tagged_tools") != expected_tool_records
        or not isinstance(contract.get("verifier_runtime"), dict)
        or not isinstance(report, dict)
        or report.get("passed") is not True
        or report.get("chain_protocol") != R1_CHAIN_PROTOCOL
        or report.get("protocol") != R1_EVIDENCE_DISPATCH_PROTOCOL
        or report.get("renderer") != "render_schema5_recovery_chain"
        or contract.get("native_report_sha256")
        != _sha256_bytes(_canonical_json(report))
        or verification_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("native r1 protocol-verification identity is invalid")
    bootstrap = _verify_sentinel_bootstrap(paths)
    verifier = paths.jobs_root / Path(R1_EVIDENCE_VERIFIER_GIT_PATH).name
    observed = _run_json_verifier(
        [
            str(paths.sentinel_bootstrap_python),
            "-I",
            "-S",
            str(verifier),
            "--chain-manifest",
            str(
                paths.recovery_root
                / R1_EVIDENCE_RELATIVE_PATHS["chain_manifest"]
            ),
            "--submission-receipt",
            str(
                paths.recovery_root
                / R1_EVIDENCE_RELATIVE_PATHS["submission_receipt"]
            ),
        ],
        environment=_sanitized_python_environment(
            library=paths.sentinel_bootstrap_root / "lib"
        ),
        timeout=600.0,
        description="sealed bundled native r1 evidence verifier",
    )
    if observed != report:
        raise ChainError("sealed native r1 evidence verification report drifted")
    if (
        bootstrap.get("bootstrap_root") != str(paths.sentinel_bootstrap_root)
    ):
        raise ChainError("sealed r1 verifier bootstrap identity drifted")
    return json.loads(json.dumps(contract, sort_keys=True))


def _snapshot_adopt_body(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
    bundled_tools: Mapping[str, bytes],
    r1_protocol_verification: Mapping[str, Any],
) -> str:
    tool = paths.source_checkout / "scripts" / "create_recovery_snapshot.py"
    evidence_contract = _r1_evidence_contract(paths)
    encoded_contract = json.dumps(evidence_contract, sort_keys=True)
    snapshot_contract = json.dumps(SEALED_SNAPSHOT_CONTRACT, sort_keys=True)
    encoded_native_contract = json.dumps(
        dict(r1_protocol_verification), sort_keys=True
    )
    verifier_name = Path(R1_EVIDENCE_VERIFIER_GIT_PATH).name
    r1_renderer_name = Path(R1_NATIVE_RENDERER_GIT_PATH).name
    v12_renderer_name = Path("scripts/render_schema5_recovery_chain_v12.py").name
    return (
        _common_exports(paths)
        + _sentinel_bootstrap_runtime_shell(paths)
        + _source_checkout_guard_shell(
            paths, commit=commit, tag_object=tag_object
        )
        + f"""\
protocol_verifier={_q(paths.jobs_root / verifier_name)}
r1_native_renderer={_q(paths.jobs_root / r1_renderer_name)}
v12_native_renderer={_q(paths.jobs_root / v12_renderer_name)}
{_immutable_tool_shell_check(
    variable="protocol_verifier",
    expected_sha256=_sha256_bytes(bundled_tools[verifier_name]),
    expected_size=len(bundled_tools[verifier_name]),
    description="bundled protocol-aware r1 evidence verifier",
)}
{_immutable_tool_shell_check(
    variable="r1_native_renderer",
    expected_sha256=_sha256_bytes(bundled_tools[r1_renderer_name]),
    expected_size=len(bundled_tools[r1_renderer_name]),
    description="bundled native r1 recovery renderer",
)}
{_immutable_tool_shell_check(
    variable="v12_native_renderer",
    expected_sha256=_sha256_bytes(bundled_tools[v12_renderer_name]),
    expected_size=len(bundled_tools[v12_renderer_name]),
    description="bundled native v1.2 recovery renderer",
)}
native_report=$(mktemp)
trap 'rm -f -- "$native_report"' EXIT
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
  "$protocol_verifier" \
  --chain-manifest {_q(Path(evidence_contract["chain_manifest"]["path"]))} \
  --submission-receipt {_q(Path(evidence_contract["submission_receipt"]["path"]))} \
  >"$native_report"
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \
  {_q(encoded_native_contract)} "$native_report" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

expected = json.loads(sys.argv[1])
observed = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if observed != expected["native_report"]:
    raise SystemExit("protocol-aware r1 evidence report differs from sealed contract")
canonical = json.dumps(
    observed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode("utf-8")
if hashlib.sha256(canonical).hexdigest() != expected["native_report_sha256"]:
    raise SystemExit("protocol-aware r1 evidence report identity drifted")
PY
ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH="$bootstrap_root/lib" \\
  "$bootstrap_python" -I -S {_q(tool)} \\
  --verify-only \\
  --snapshot-root {_q(paths.recovery_root / 'pre_repair')} \\
  --attestation-path {_q(paths.recovery_root / 'pre_repair.attestation.json')}
ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH="$bootstrap_root/lib" \\
  "$bootstrap_python" -I -S {_q(tool)} \\
  --verify-only \\
  --snapshot-root {_q(paths.recovery_root / 'r1_failure_evidence')} \\
  --attestation-path {_q(paths.recovery_root / 'r1_failure_evidence.attestation.json')}
exec env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \\
  {_q(encoded_contract)} {_q(snapshot_contract)} <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

evidence = json.loads(sys.argv[1])
expected_snapshot = json.loads(sys.argv[2])
for name, record in evidence.items():
    path = Path(record["path"])
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o222:
        raise SystemExit(f"unsealed r1 evidence: {{name}}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"] or info.st_size != record["size"]:
        raise SystemExit(f"r1 evidence drift: {{name}}")
marker = json.loads(Path(evidence["snapshot_completion"]["path"]).read_text())
for key in ("file_count", "total_bytes", "snapshot_inventory_sha256"):
    if marker.get(key) != expected_snapshot[key]:
        raise SystemExit(f"sealed snapshot contract drift: {{key}}")
failure = json.loads(Path(evidence["failure_envelope"]["path"]).read_text())
if (
    failure.get("classification") != "requires_superseding_release"
    or failure.get("retry_same_generation") is not False
    or failure.get("superseded_by") != "sweep-recovery-schema5-v1.2"
):
    raise SystemExit("r1 failure is not fenced to the superseding release")
print(json.dumps({{"passed": True, "adopted_snapshot": marker["snapshot_id"]}}, sort_keys=True))
PY
"""
    )


def _environment_capture_body(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
    expected_source_inventories: Mapping[str, str],
    expected_policy_hashes: Mapping[str, str],
    expected_reconciliation_incident: Mapping[str, Any],
) -> str:
    expected_incident_fields = {
        "path",
        "sha256",
        "incident_id",
        "harness_stale_conda_record_present",
        "serving_stale_conda_record_present",
    }
    if (
        set(expected_source_inventories) != {"harness", "serving"}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in expected_source_inventories.values()
        )
        or set(expected_policy_hashes) != {"ownership", "integrity"}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in expected_policy_hashes.values()
        )
        or set(expected_reconciliation_incident) != expected_incident_fields
        or expected_reconciliation_incident.get("sha256")
        != PRODUCTION_CONDA_RECONCILIATION_INCIDENT_SHA256
        or expected_reconciliation_incident.get("incident_id")
        != PRODUCTION_CONDA_RECONCILIATION_INCIDENT_ID
        or expected_reconciliation_incident.get(
            "harness_stale_conda_record_present"
        )
        is not False
        or expected_reconciliation_incident.get(
            "serving_stale_conda_record_present"
        )
        is not False
    ):
        raise ChainError(
            "pilot source-inventory, environment-policy, or production "
            "reconciliation-incident binding is incomplete"
        )
    tool = paths.source_checkout / "scripts" / "capture_schema5_environments.py"
    ownership_policy = (
        paths.source_checkout / OWNERSHIP_POLICY_GIT_PATH
    )
    integrity_policy = (
        paths.source_checkout / INTEGRITY_NORMALIZATION_POLICY_GIT_PATH
    )
    incident = paths.recovery_root / R1_EVIDENCE_RELATIVE_PATHS["conda_incident"]
    recovered = (
        paths.recovery_root
        / "releases"
        / "quarantine"
        / "sweep-recovery-schema5-v1.1.partial-job-18555913"
        / "environments"
        / "harness"
        / "conda-meta"
        / "setuptools-82.0.1-pyh332efcf_0.json"
    )
    python = 'env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S'
    command = f"""\
{python} {_q(tool)} capture \\
  --output-root {_q(paths.environment_capture_root)} \\
  --harness-source {_q(paths.source_harness)} \\
  --serving-source {_q(paths.source_serving)} \\
  --ownership-policy {_q(ownership_policy)} \\
  --integrity-normalization-policy {_q(integrity_policy)} \\
  --reconciliation-incident {_q(incident)} \\
  --recovered-setuptools-record {_q(recovered)}"""
    harness_inventory = expected_source_inventories["harness"]
    serving_inventory = expected_source_inventories["serving"]
    ownership_policy_sha256 = expected_policy_hashes["ownership"]
    integrity_policy_sha256 = expected_policy_hashes["integrity"]
    incident_sha256 = str(expected_reconciliation_incident["sha256"])
    incident_id = str(expected_reconciliation_incident["incident_id"])
    marker = paths.environment_capture_root / "ENVIRONMENT_CAPTURE_COMPLETE.json"
    return (
        _common_exports(paths)
        + _sentinel_bootstrap_runtime_shell(paths)
        + _source_checkout_guard_shell(
            paths, commit=commit, tag_object=tag_object
        )
        + f"""\
verify_live_sources() {{
  {python} - {_q(tool)} {_q(paths.source_harness)} {_q(paths.source_serving)} \\
    {_q(harness_inventory)} {_q(serving_inventory)} <<'PY'
import runpy
from pathlib import Path
import sys

tool, harness, serving = map(Path, sys.argv[1:4])
expected = {{"harness": sys.argv[4], "serving": sys.argv[5]}}
inventory = runpy.run_path(str(tool))["directory_inventory"]
for role, root in (("harness", harness), ("serving", serving)):
    observed = inventory(root)["inventory_sha256"]
    if observed != expected[role]:
        raise SystemExit(
            f"live {{role}} prefix differs from sealed pilot: "
            f"{{observed}} != {{expected[role]}}"
        )
PY
}}
verify_reconciliation_incident() {{
  {python} - "$1" {_q(incident_sha256)} {_q(incident_id)} <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()
payload = json.loads(raw)
expected = {{
    "sha256": sys.argv[2],
    "incident_id": sys.argv[3],
    "harness_stale_conda_record_present": False,
    "serving_stale_conda_record_present": False,
}}
observed = {{
    "sha256": hashlib.sha256(raw).hexdigest(),
    "incident_id": payload.get("incident_id"),
    "harness_stale_conda_record_present": payload.get(
        "harness_stale_conda_record_present"
    ),
    "serving_stale_conda_record_present": payload.get(
        "serving_stale_conda_record_present"
    ),
}}
if observed != expected:
    raise SystemExit(
        f"production Conda reconciliation incident drifted: "
        f"{{observed}} != {{expected}}"
    )
PY
}}
verify_capture_binding() {{
  ionice -c 2 -n 7 nice -n 10 {python} {_q(tool)} verify \\
    --output-root {_q(paths.environment_capture_root)}
  {python} - {_q(marker)} {_q(harness_inventory)} {_q(serving_inventory)} \\
    {_q(ownership_policy_sha256)} {_q(integrity_policy_sha256)} \\
    {_q(incident_sha256)} <<'PY'
import json
from pathlib import Path
import sys

marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {{"harness": sys.argv[2], "serving": sys.argv[3]}}
expected_policy = {{
    "ownership_policy_sha256": sys.argv[4],
    "integrity_normalization_policy_sha256": sys.argv[5],
}}
expected_incident_sha256 = sys.argv[6]
observed = {{
    role: marker.get("source_inventories", {{}})
    .get(role, {{}})
    .get("inventory_sha256")
    for role in expected
}}
if observed != expected:
    raise SystemExit(
        f"production capture used bytes outside sealed pilot: "
        f"{{observed}} != {{expected}}"
    )
observed_policy = {{
    field: marker.get(field) for field in expected_policy
}}
if observed_policy != expected_policy:
    raise SystemExit(
        f"production capture used policies outside sealed pilot: "
        f"{{observed_policy}} != {{expected_policy}}"
    )
if marker.get("reconciliation_incident_sha256") != expected_incident_sha256:
    raise SystemExit(
        "production capture used a different Conda reconciliation incident"
    )
PY
  captured_incident="$({python} - {_q(marker)} <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text(
    encoding="utf-8"
))["reconciliation_incident_path"])
PY
)"
  verify_reconciliation_incident "$captured_incident"
}}
if [[ -f {_q(paths.environment_capture_root / 'ENVIRONMENT_CAPTURE_COMPLETE.json')} && \
      ! -L {_q(paths.environment_capture_root / 'ENVIRONMENT_CAPTURE_COMPLETE.json')} ]]; then
  verify_capture_binding
  exit 0
fi
verify_live_sources
verify_reconciliation_incident {_q(incident)}
ionice -c 2 -n 7 nice -n 10 {command}
ionice -c 2 -n 7 nice -n 10 {command} --apply
verify_capture_binding
"""
    )


def _materialize_body(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
    expected_source_inventories: Mapping[str, str],
    conda_toolchain_binding: Mapping[str, Any],
    package_cache_seed_input: Mapping[str, Any],
) -> str:
    validated_cache_input = _validate_package_cache_seed_input(
        dict(package_cache_seed_input),
        source_package_cache=paths.source_package_cache,
    )
    if (
        set(expected_source_inventories) != {"harness", "serving"}
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in expected_source_inventories.values()
        )
        or conda_toolchain_binding
        != _verified_conda_toolchain_binding(paths.conda_toolchain_root)
        or dict(package_cache_seed_input) != validated_cache_input
    ):
        raise ChainError("pilot materialization input binding is incomplete")
    tool = paths.source_checkout / "scripts" / "materialize_schema5_release.py"
    capture_tool = (
        paths.source_checkout / "scripts" / "capture_schema5_environments.py"
    )
    captured_harness = paths.captured_harness
    encoded_conda_toolchain = json.dumps(
        dict(conda_toolchain_binding), sort_keys=True
    )
    encoded_cache_input = json.dumps(
        dict(package_cache_seed_input), sort_keys=True
    )
    python = (
        'env LD_LIBRARY_PATH="$captured_harness/lib" '
        '"$captured_python" -I'
    )
    command = f"""\
{python} {_q(tool)} materialize \\
  --output-root {_q(paths.release_root)} \\
  --source-repository {_q(paths.source_checkout)} \\
  --release-worktree {_q(paths.worktree)} \\
  --source-harness-prefix {_q(paths.captured_harness)} \\
  --source-serving-prefix {_q(paths.captured_serving)} \\
  --source-package-cache {_q(paths.source_package_cache)} \\
  --environment-capture-root {_q(paths.environment_capture_root)} \\
  --harness-prefix {_q(paths.harness)} \\
  --serving-prefix {_q(paths.serving)} \\
  --expected-package-cache-seed-input-json {_q(encoded_cache_input)} \\
  --conda-toolchain-root {_q(paths.conda_toolchain_root)}"""
    return (
        _common_exports(paths)
        + _sentinel_bootstrap_runtime_shell(paths)
        + _source_checkout_guard_shell(
            paths, commit=commit, tag_object=tag_object
        )
        + f"""\
capture_marker={_q(paths.environment_capture_root / 'ENVIRONMENT_CAPTURE_COMPLETE.json')}
captured_harness={_q(captured_harness)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \\
  {_q(capture_tool)} verify --output-root {_q(paths.environment_capture_root)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \\
  "$capture_marker" {_q(expected_source_inventories['harness'])} \\
  {_q(expected_source_inventories['serving'])} <<'PY'
import json
from pathlib import Path
import sys

marker = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {{"harness": sys.argv[2], "serving": sys.argv[3]}}
observed = {{
    role: marker.get("source_inventories", {{}})
    .get(role, {{}})
    .get("inventory_sha256")
    for role in expected
}}
if observed != expected:
    raise SystemExit(
        f"captured production seeds are not pilot-bound: {{observed}} != {{expected}}"
    )
PY
captured_python="$(realpath -e -- "$captured_harness/bin/python")"
[[ "$captured_python" == "$captured_harness/"* && -f "$captured_python" && \
   ! -L "$captured_python" && -x "$captured_python" ]] || {{
  echo "captured harness Python is missing or escapes its verified seed" >&2
  exit 2
}}
[[ -z "$(find "$captured_harness" -perm /222 -print -quit)" ]] || {{
  echo "captured harness seed is writable" >&2
  exit 2
}}
verify_conda_runtime_toolchain() {{
  env LD_LIBRARY_PATH="$captured_harness/lib" "$captured_python" -I -S - \
    {_q(paths.source_checkout)} {_q(paths.conda_toolchain_root)} \
    {_q(encoded_conda_toolchain)} <<'PY'
import json
import sys

sys.path.insert(0, sys.argv[1])
from scripts.provision_schema5_conda_toolchain import (
    verified_conda_toolchain_binding,
)
observed = verified_conda_toolchain_binding(sys.argv[2], exercise=True)
expected = json.loads(sys.argv[3])
if observed != expected:
    raise SystemExit(
        "Conda runtime toolchain differs from sealed pilot provenance"
    )
PY
}}
verify_package_cache_input() {{
  env LD_LIBRARY_PATH="$captured_harness/lib" "$captured_python" -I -S - \
    {_q(paths.source_checkout)} {_q(paths.source_package_cache)} \
    {_q(paths.captured_harness)} {_q(paths.captured_serving)} \
    {_q(encoded_cache_input)} <<'PY'
import json
import sys

sys.path.insert(0, sys.argv[1])
from scripts.materialize_schema5_release import (
    selected_package_cache_input_binding,
)
observed = selected_package_cache_input_binding(
    source_package_cache=sys.argv[2],
    harness_seed=sys.argv[3],
    serving_seed=sys.argv[4],
)
expected = json.loads(sys.argv[5])
if observed != expected:
    raise SystemExit(
        "production selected package-cache input differs from sealed pilot provenance"
    )
PY
}}
verify_conda_runtime_toolchain
verify_package_cache_input
if [[ -e {_q(paths.release_root)} || -L {_q(paths.release_root)} ]]; then
  if [[ -f {_q(paths.release_root / 'MATERIALIZATION_COMPLETE.json')} && \
        ! -L {_q(paths.release_root / 'MATERIALIZATION_COMPLETE.json')} ]]; then
    exec ionice -c 2 -n 7 nice -n 10 {python} {_q(tool)} verify \
      --output-root {_q(paths.release_root)}
  fi
  echo "partial materialization is preserved at {_q(paths.release_root)}; run the explicit quarantine-materialization command before repair" >&2
  exit 3
fi
verify_conda_runtime_toolchain
verify_package_cache_input
ionice -c 2 -n 7 nice -n 10 {command}
verify_conda_runtime_toolchain
verify_package_cache_input
ionice -c 2 -n 7 nice -n 10 {command} --apply
verify_conda_runtime_toolchain
exec ionice -c 2 -n 7 nice -n 10 {python} {_q(tool)} verify \\
  --output-root {_q(paths.release_root)}
"""
    )


def _freeze_body(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
) -> str:
    materializer = paths.source_checkout / "scripts" / "materialize_schema5_release.py"
    freezer = paths.worktree / "scripts" / "freeze_schema5_release.py"
    command = f"""\
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(freezer)} create \\
  --output-root {_q(paths.identity)} \\
  --release-worktree {_q(paths.worktree)} \\
  --harness-prefix {_q(paths.harness)} \\
  --serving-prefix {_q(paths.serving)} \\
  --model-contract {_q(paths.worktree / 'configs/model_contracts.v1.json')} \\
  --fleet-contract {_q(paths.worktree / 'configs/schema5_fleet.v1.json')}"""
    return (
        _common_exports(paths)
        + _sentinel_bootstrap_runtime_shell(paths)
        + _source_checkout_guard_shell(
            paths, commit=commit, tag_object=tag_object
        )
        + f"""\
ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I {_q(materializer)} verify \\
  --output-root {_q(paths.release_root)}
ionice -c 2 -n 7 nice -n 10 {command}
ionice -c 2 -n 7 nice -n 10 {command} \\
  --apply --seal-worktree --seal-environments --seal-output-root
exec ionice -c 2 -n 7 nice -n 10 env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \\
  {_q(paths.harness / 'bin/python')} -I {_q(freezer)} verify \\
  --output-root {_q(paths.identity)}
"""
    )


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
# Smoke manifests are materialized only inside a generation/attempt-scoped run
# namespace immediately before their first draw. This dry-run freezes the exact
# 15+20+6 design without creating a mutable fixed root that a killed request or
# later capacity generation could contaminate.

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
    acknowledgement_tool = (
        paths.worktree / "scripts" / "schema5_email_ack.py"
    )
    active_challenge = (
        paths.readiness / "email_challenges" / "CURRENT.json"
    )
    return _common_exports(paths) + f"""\
mkdir -p -- {_q(active_challenge.parent)}
[[ -n "${{SLURM_JOB_ID:-}}" ]] || {{ echo "email readiness requires a Slurm job identity" >&2; exit 2; }}
chain_id="$(jq -er '.chain_id' {_q(paths.chain_manifest)})"
release_tag="$(jq -er '.release_tag' {_q(paths.chain_manifest)})"
release_git_commit="$(jq -er '.release_git_commit' {_q(paths.chain_manifest)})"
[[ "$release_tag" == {_q(RELEASE_TAG)} ]]
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
if [[ "$job_record" =~ (^|[[:space:]])Comment=([^[:space:]]+) ]]; then
  job_comment="${{BASH_REMATCH[2]}}"
else
  echo "email readiness cannot resolve its scheduler comment" >&2
  exit 2
fi
squeue_comment="$(squeue -h -j "$SLURM_JOB_ID" -o '%k')"
[[ -n "$squeue_comment" && "$squeue_comment" != *$'\\n'* && "$squeue_comment" == "$job_comment" ]] || {{
  echo "email readiness scheduler comment sources disagree" >&2
  exit 2
}}
email_comment_re="^asys:s5-recovery-v1\\\\.2-r7:${{chain_id}}:g([0-9]{{4}}):email_readiness$"
[[ "$job_comment" =~ $email_comment_re ]] || {{
  echo "email readiness scheduler comment is not bound to this chain" >&2
  exit 2
}}
challenge_generation=$((10#${{BASH_REMATCH[1]}}))
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(active_challenge)} email-request \\
  --chain-id "$chain_id" --challenge-generation "$challenge_generation" \\
  --release-tag "$release_tag" --release-git-commit "$release_git_commit" \\
  --acknowledgement-script {_q(acknowledgement_tool)} \\
  --acknowledgement-python {_q(paths.harness / 'bin/python')} --apply
ack_deadline=$(( $(date +%s) + 27000 ))
while true; do
  acknowledgement="$(jq -er '.acknowledgement' {_q(active_challenge)})"
  [[ -f "$acknowledgement" ]] && break
  (( $(date +%s) < ack_deadline )) || {{ echo "email acknowledgement was not received within 7.5 hours" >&2; exit 2; }}
  sleep 30
done
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
  --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'email_test.json')} email-test \\
  --active-challenge {_q(active_challenge)}
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
# Close the all-RUNNING race at the deadline before classifying capacity.  This is a
# complete ordinary readiness transaction, not a cached result from the prior poll.
{_fleet_once_python(paths)}
if env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
    --state-dir {_q(paths.state)} --output {_q(paths.readiness / 'fleet.json')} fleet --probe-timeout 10; then
  exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
    --state-dir {_q(paths.state)} attest --gate fleet --evidence {_q(paths.readiness / 'fleet.json')}
fi
echo "fleet did not satisfy readiness within the bounded 10-hour window; checking the exact capacity-only repair contract" >&2
[[ "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || {{ echo "fleet readiness requires its exact Slurm job ID" >&2; exit 2; }}
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
comment="$(tr ' ' '\\n' <<<"$job_record" | sed -n 's/^Comment=//p')"
[[ "$(wc -l <<<"$comment")" -eq 1 ]] || {{ echo "fleet readiness scheduler comment is ambiguous" >&2; exit 2; }}
if [[ "$comment" =~ ^asys:s5-recovery-v1\\.2-r7:[0-9a-f]{{64}}:(g[0-9]{{4}}):fleet_readiness$ ]]; then
  generation="${{BASH_REMATCH[1]}}"
else
  echo "fleet readiness scheduler comment is outside the immutable r7 namespace: $comment" >&2
  exit 2
fi
if [[ "$generation" == g0000 ]]; then
  receipt={_q(paths.recovery_root / SUBMISSION_RECEIPT_NAME)}
else
  receipt={_q(paths.recovery_root / REPAIR_ROOT_NAME)}/"$generation"/{_q(SUBMISSION_RECEIPT_NAME)}
fi
[[ -f "$receipt" && ! -L "$receipt" ]] || {{ echo "fleet readiness generation receipt is unavailable" >&2; exit 2; }}
capacity_root={_q(paths.capacity_transient_root)}/"$generation"/"$SLURM_JOB_ID"
capacity_receipt="$capacity_root"/{_q(CAPACITY_TRANSIENT_MARKER_NAME)}
mkdir -p -- "$capacity_root"
if env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(builder)} \\
    --state-dir {_q(paths.state)} --output "$capacity_receipt" fleet-capacity-transient \\
    --chain-manifest {_q(paths.chain_manifest)} --submission-receipt "$receipt" \\
    --readiness-job-id "$SLURM_JOB_ID" --boundary-seconds 36000 --probe-timeout 10; then
  echo "exact fleet capacity transient sealed; exiting with reserved status 75" >&2
  exit 75
fi
echo "fleet failure did not satisfy the exact capacity-transient contract" >&2
exit 2
"""


def _smoke_body(paths: RecoveryPaths, *, renderer_payload: bytes) -> str:
    runner = paths.worktree / "scripts" / "run_schema5_smokes.py"
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    control = paths.worktree / "slurm" / "schema5_control.py"
    verifier = paths.jobs_root / Path(
        "scripts/render_schema5_recovery_chain_v12.py"
    ).name
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
smoke_verifier={_q(verifier)}
{_immutable_tool_shell_check(
    variable="smoke_verifier",
    expected_sha256=_sha256_bytes(renderer_payload),
    expected_size=len(renderer_payload),
    description="bundled smoke-attempt verifier",
)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I \\
  "$smoke_verifier" verify-smoke-attempt --chain-manifest {_q(paths.chain_manifest)}
smoke_verification="$(
  env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(runner)} \\
    --results-root {_q(paths.results_root)} --server-pool-root {_q(paths.pool)} \\
    --release-worktree {_q(paths.worktree)} --harness-prefix {_q(paths.harness)} \\
    --state-root {_q(paths.state)} --immutable-pins-sha256 "$immutable_sha" \\
    --rollout-generation "$next_generation"
)"
smoke_evidence="$(
  env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I - \\
    "$smoke_verification" <<'PY'
import json
import sys
value = json.loads(sys.argv[1])
if value.get("status") != "verified" or value.get("passed") is not True:
    raise SystemExit("smoke CURRENT verification did not pass")
evidence = value.get("evidence")
if not isinstance(evidence, str) or not evidence.startswith("/"):
    raise SystemExit("smoke CURRENT verification lacks an absolute evidence path")
print(evidence)
PY
)"
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} {_q(paths.harness / 'bin/python')} -I {_q(control)} \\
  --state-dir {_q(paths.state)} attest --gate smoke_runs \\
  --evidence "$smoke_evidence"
"""


def _qualification_verifier_shell(
    paths: RecoveryPaths,
    *,
    renderer_payload: bytes,
    wait_for_marker: bool,
) -> str:
    verifier = paths.jobs_root / Path(
        "scripts/render_schema5_recovery_chain_v12.py"
    ).name
    wait = ""
    if wait_for_marker:
        wait = f"""\
qualification_deadline=$(( $(date +%s) + 36000 ))
while [[ ! -f {_q(paths.throughput_qualification_marker)} || \
         -L {_q(paths.throughput_qualification_marker)} ]]; do
  (( $(date +%s) < qualification_deadline )) || {{
    echo "full-capacity throughput qualification was not published within 10 hours" >&2
    exit 2
  }}
  sleep 30
done
"""
    return f"""\
{wait}qualification_verifier={_q(verifier)}
{_immutable_tool_shell_check(
    variable="qualification_verifier",
    expected_sha256=_sha256_bytes(renderer_payload),
    expected_size=len(renderer_payload),
    description="bundled throughput-qualification verifier",
)}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I "$qualification_verifier" \
  verify-throughput-qualification --chain-manifest {_q(paths.chain_manifest)}
"""


def _throughput_qualification_body(
    paths: RecoveryPaths, *, renderer_payload: bytes
) -> str:
    producer = (
        paths.worktree / "scripts" / "run_schema5_throughput_qualification.py"
    )
    verifier = _qualification_verifier_shell(
        paths,
        renderer_payload=renderer_payload,
        wait_for_marker=False,
    )
    control = paths.worktree / "slurm" / "schema5_control.py"
    return (
        _common_exports(paths)
        + f"mkdir -p -- {_q(paths.throughput_qualification_root)}\n"
        + f"""\
qualification_producer={_q(producer)}
[[ -f "$qualification_producer" && ! -L "$qualification_producer" ]] || {{
  echo "immutable throughput-qualification producer is unavailable" >&2
  exit 2
}}
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I "$qualification_producer" execute \
  --chain-manifest {_q(paths.chain_manifest)} --timeout-seconds 36000
"""
        + verifier
        + f"""\
exec env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I {_q(control)} \
  --state-dir {_q(paths.state)} attest \
  --gate throughput_qualification \
  --chain-manifest {_q(paths.chain_manifest)} \
  --evidence {_q(paths.throughput_qualification_marker)}
"""
    )


def _drill_body(
    paths: RecoveryPaths,
    *,
    renderer_payload: bytes,
    git_commit: str,
    tag_object: str,
) -> str:
    control = paths.worktree / "slurm" / "schema5_control.py"
    publisher = paths.worktree / "scripts" / "publish_schema5_watchdog_ready.py"
    watchdog_root = paths.readiness / "external_watchdog"
    deployment = watchdog_root / "DEPLOYMENT_EVIDENCE.json"
    liveness = watchdog_root / "LIVENESS_EVIDENCE.json"
    drill_evidence = watchdog_root / "DRILL_EVIDENCE.json"
    renderer = paths.jobs_root / "render_schema5_recovery_chain_v12.py"
    prefix = (
        f"env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "
        f"{_q(paths.harness / 'bin/python')} -I {_q(control)} "
        f"--state-dir {_q(paths.state)}"
    )
    return (
        _common_exports(paths)
        + _qualification_verifier_shell(
            paths,
            renderer_payload=renderer_payload,
            wait_for_marker=False,
        )
        + f"""\
control=( {prefix} )
if [[ -f {_q(paths.state / 'CONTROLLER_KILL_DRILL_COMPLETE.json')} && \
      ! -L {_q(paths.state / 'CONTROLLER_KILL_DRILL_COMPLETE.json')} && \
      -f {_q(paths.external_watchdog_drill_marker)} && \
      ! -L {_q(paths.external_watchdog_drill_marker)} && \
      -f {_q(paths.watchdog_ready_marker)} && \
      ! -L {_q(paths.watchdog_ready_marker)} ]]; then
  "${{control[@]}}" drill status --live
  env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
    {_q(paths.harness / 'bin/python')} -I {_q(renderer)} \
    verify-watchdog-readiness --chain-manifest {_q(paths.chain_manifest)}
  "${{control[@]}}" attest --gate external_watchdog \
    --chain-manifest {_q(paths.chain_manifest)} \
    --evidence {_q(paths.watchdog_ready_marker)}
  exit 0
fi
"${{control[@]}}" reconcile --all --no-admit

watchdog_input_deadline=$(( $(date +%s) + 1800 ))
while [[ ! -f {_q(deployment)} || -L {_q(deployment)} || \
         ! -f {_q(liveness)} || -L {_q(liveness)} ]]; do
  (( $(date +%s) < watchdog_input_deadline )) || {{
    echo "sealed external-watchdog deployment/liveness evidence was not available within 30 minutes" >&2
    exit 2
  }}
  sleep 15
done

"${{control[@]}}" drill start
deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" drill status --live; do
  (( $(date +%s) < deadline )) || {{ echo "initial drill readiness exceeded 15 minutes" >&2; exit 2; }}
  sleep 5
done
"${{control[@]}}" watchdog-drill arm \
  --deployment-evidence {_q(deployment)} --expires-seconds 1800

cancel_deadline=$(( $(date +%s) + 300 ))
while true; do
  if ! cancel_json="$("${{control[@]}}" watchdog-drill cancel)"; then
    (( $(date +%s) < cancel_deadline )) || {{
      echo "external-watchdog cancellation lock remained unavailable" >&2
      exit 2
    }}
    sleep 5
    continue
  fi
  cancel_phase="$(
    printf '%s' "$cancel_json" |
      env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
        {_q(paths.harness / 'bin/python')} -I -c \
        'import json,sys; print(json.load(sys.stdin)["state"]["phase"])'
  )"
  [[ "$cancel_phase" == "cancelled" ]] && break
  (( $(date +%s) < cancel_deadline )) || {{
    echo "exact external-watchdog namespace cancellation did not reconcile within five minutes" >&2
    exit 2
  }}
  sleep 5
done

# The institutional watchdog now observes this exact paused namespace twice and its
# forced repair-chain selector is routed solely to the sealed drill reconciliation.
external_recovery_deadline=$(( $(date +%s) + 900 ))
until "${{control[@]}}" watchdog-drill finish --output {_q(drill_evidence)}; do
  (( $(date +%s) < external_recovery_deadline )) || {{
    echo "external watchdog did not restore the isolated namespace within 15 minutes" >&2
    exit 2
  }}
  sleep 5
done

env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I {_q(publisher)} \
  --recovery-root {_q(paths.recovery_root)} \
  --deployment-evidence {_q(deployment)} \
  --drill-evidence {_q(drill_evidence)} \
  --liveness-evidence {_q(liveness)} \
  --git-commit {_q(git_commit)} \
  --tag-object {_q(tag_object)} \
  --apply
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I {_q(renderer)} \
  verify-watchdog-readiness --chain-manifest {_q(paths.chain_manifest)}
"${{control[@]}}" attest --gate external_watchdog \
  --chain-manifest {_q(paths.chain_manifest)} \
  --evidence {_q(paths.watchdog_ready_marker)}

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
"${{control[@]}}" drill status --live
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I {_q(renderer)} \
  verify-watchdog-readiness --chain-manifest {_q(paths.chain_manifest)}
"""
    )


def _resume_body(
    paths: RecoveryPaths, *, renderer_payload: bytes
) -> str:
    control = paths.worktree / "slurm" / "schema5_control.py"
    builder = paths.worktree / "scripts" / "build_schema5_readiness.py"
    prefix = (
        f"env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} "
        f"{_q(paths.harness / 'bin/python')} -I {_q(control)} "
        f"--state-dir {_q(paths.state)}"
    )
    return (
        _common_exports(paths)
        + _qualification_verifier_shell(
            paths,
            renderer_payload=renderer_payload,
            wait_for_marker=False,
        )
        + f"""\
control=( {prefix} )
"${{control[@]}}" drill status --live
env LD_LIBRARY_PATH={_q(paths.harness / 'lib')} \
  {_q(paths.harness / 'bin/python')} -I \
  {_q(paths.jobs_root / 'render_schema5_recovery_chain_v12.py')} \
  verify-watchdog-readiness --chain-manifest {_q(paths.chain_manifest)}
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

# Seal the exact five-field endpoint-generation tuples while the refreshed fleet
# readiness proof and paused scheduler cut are still current.  Production analysis and
# finalization trust only this immutable marker-last authority, never a cross product of
# marginal provenance values or a mutable registry pointer.
"${{control[@]}}" refresh-generation-catalog

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
    )


def _r3_prelaunch_seal_shell_check(
    paths: RecoveryPaths,
    *,
    sealer_payload: bytes,
) -> str:
    sealer = paths.jobs_root / "seal_recovery_evidence.py"
    if not sealer_payload:
        raise ChainError("bundled r3 prelaunch-seal verifier is missing")
    return f"""\
r3_seal_verifier={_q(sealer)}
{_immutable_tool_shell_check(
    variable="r3_seal_verifier",
    expected_sha256=_sha256_bytes(sealer_payload),
    expected_size=len(sealer_payload),
    description="bundled r3 prelaunch-seal verifier",
)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S - \
  "$r3_seal_verifier" \
  {_q(paths.superseded_r3_prelaunch_failure_root)} \
  {_q(paths.chain_manifest)} <<'PY'
import json
from pathlib import Path
import runpy
import sys

verifier, root, manifest_path = sys.argv[1:]
observed = runpy.run_path(verifier)[
    "verify_prelaunch_failure_seal"
](Path(root))
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
expected = manifest.get("prerequisite_evidence", {{}}).get(
    "superseded_r3_prelaunch_failure"
)
if observed != expected:
    raise SystemExit(
        "superseded r3 prelaunch-failure seal differs from chain manifest"
    )
PY
"""


def _r4_toolchain_failure_shell_check(
    paths: RecoveryPaths,
    *,
    verifier_payload: bytes,
) -> str:
    verifier = paths.jobs_root / "seal_schema5_r4_toolchain_failure.py"
    if not verifier_payload:
        raise ChainError("bundled r4 toolchain-failure verifier is missing")
    return f"""\
r4_failure_verifier={_q(verifier)}
{_immutable_tool_shell_check(
    variable="r4_failure_verifier",
    expected_sha256=_sha256_bytes(verifier_payload),
    expected_size=len(verifier_payload),
    description="bundled r4 toolchain-failure verifier",
)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
  "$r4_failure_verifier" verify \
  --evidence-root {_q(paths.superseded_r4_toolchain_failure_root)} \
  --recovery-root {_q(paths.recovery_root)}
"""


def _r5_prelaunch_failure_shell_check(
    paths: RecoveryPaths,
    *,
    verifier_payload: bytes,
) -> str:
    verifier = paths.jobs_root / "seal_schema5_r5_prelaunch_failure.py"
    if not verifier_payload:
        raise ChainError("bundled r5 prelaunch-failure verifier is missing")
    return f"""\
r5_failure_verifier={_q(verifier)}
{_immutable_tool_shell_check(
    variable="r5_failure_verifier",
    expected_sha256=_sha256_bytes(verifier_payload),
    expected_size=len(verifier_payload),
    description="bundled r5 prelaunch-failure verifier",
)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
  "$r5_failure_verifier" verify \
  --evidence-root {_q(paths.superseded_r5_prelaunch_failure_root)} \
  --recovery-root {_q(paths.recovery_root)}
"""


def _r6_prelaunch_failure_shell_check(
    paths: RecoveryPaths,
    *,
    verifier_payload: bytes,
) -> str:
    verifier = paths.jobs_root / "seal_schema5_r6_prelaunch_failure.py"
    if not verifier_payload:
        raise ChainError("bundled r6 prelaunch-failure verifier is missing")
    return f"""\
r6_failure_verifier={_q(verifier)}
{_immutable_tool_shell_check(
    variable="r6_failure_verifier",
    expected_sha256=_sha256_bytes(verifier_payload),
    expected_size=len(verifier_payload),
    description="bundled r6 prelaunch-failure verifier",
)}
env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S \
  "$r6_failure_verifier" verify \
  --evidence-root {_q(paths.superseded_r6_prelaunch_failure_root)} \
  --recovery-root {_q(paths.recovery_root)}
"""


def _failure_sentinel_body(
    paths: RecoveryPaths,
    *,
    sentinel_payload: bytes,
    renderer_payload: bytes,
    sealer_payload: bytes,
    r4_failure_verifier_payload: bytes,
    r5_failure_verifier_payload: bytes,
    r6_failure_verifier_payload: bytes,
) -> str:
    # This tool is copied from the tagged commit into the immutable job namespace.
    # It therefore remains runnable even when the source-checkout stage itself fails.
    tool = paths.jobs_root / SENTINEL_TOOL_FILENAME
    tool_sha256 = _sha256_bytes(sentinel_payload)
    tool_size = len(sentinel_payload)
    renderer = paths.jobs_root / "render_schema5_recovery_chain_v12.py"
    renderer_sha256 = _sha256_bytes(renderer_payload)
    renderer_size = len(renderer_payload)
    return _common_exports(paths) + f"""\
[[ "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || {{ echo "sentinel requires an exact Slurm job ID" >&2; exit 2; }}
sentinel_tool={_q(tool)}
renderer={_q(renderer)}
{_sentinel_bootstrap_runtime_shell(paths)}
{_immutable_tool_shell_check(
    variable="sentinel_tool",
    expected_sha256=tool_sha256,
    expected_size=tool_size,
    description="bundled recovery sentinel",
)}
{_immutable_tool_shell_check(
    variable="renderer",
    expected_sha256=renderer_sha256,
    expected_size=renderer_size,
    description="bundled recovery renderer",
)}
{_r3_prelaunch_seal_shell_check(paths, sealer_payload=sealer_payload)}
{_r4_toolchain_failure_shell_check(
    paths, verifier_payload=r4_failure_verifier_payload
)}
{_r5_prelaunch_failure_shell_check(
    paths, verifier_payload=r5_failure_verifier_payload
)}
{_r6_prelaunch_failure_shell_check(
    paths, verifier_payload=r6_failure_verifier_payload
)}
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
comment="$(tr ' ' '\\n' <<<"$job_record" | sed -n 's/^Comment=//p')"
[[ "$(wc -l <<<"$comment")" -eq 1 ]] || {{ echo "sentinel scheduler comment is ambiguous" >&2; exit 2; }}
if [[ "$comment" =~ ^asys:s5-recovery-v1\\.2-r7:[0-9a-f]{{64}}:(g[0-9]{{4}}):failure_sentinel$ ]]; then
  generation="${{BASH_REMATCH[1]}}"
else
  echo "sentinel scheduler comment is outside the immutable r7 namespace: $comment" >&2
  exit 2
fi
if [[ "$generation" == g0000 ]]; then
  receipt={_q(paths.recovery_root / SUBMISSION_RECEIPT_NAME)}
else
  receipt={_q(paths.recovery_root / REPAIR_ROOT_NAME)}/"$generation"/{_q(SUBMISSION_RECEIPT_NAME)}
fi
receipt_deadline=$(( $(date +%s) + 300 ))
while [[ ! -f "$receipt" ]]; do
  (( $(date +%s) < receipt_deadline )) || {{ echo "generation receipt was not published" >&2; exit 2; }}
  sleep 2
done
fleet_readiness_job_id="$("$bootstrap_python" -I -S - "$receipt" <<'PY'
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1])
payload = json.loads(receipt_path.read_text(encoding="utf-8"))
rows = [
    row for row in payload.get("jobs", [])
    if isinstance(row, dict) and row.get("name") == "fleet_readiness"
]
if len(rows) != 1 or not str(rows[0].get("job_id", "")).isdigit():
    raise SystemExit("generation receipt has ambiguous fleet_readiness identity")
print(rows[0]["job_id"])
PY
)"
capacity_receipt={_q(paths.capacity_transient_root)}/"$generation"/"$fleet_readiness_job_id"/{_q(CAPACITY_TRANSIENT_MARKER_NAME)}
sentinel_report="$(env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S "$sentinel_tool" \\
  --chain-manifest {_q(paths.chain_manifest)} \\
  --submission-receipt "$receipt" \\
  --capacity-transient-receipt "$capacity_receipt" \\
  --output-root {_q(paths.recovery_root / 'recovery_chain_sentinels' / CHAIN_NAMESPACE)}/"$generation" \\
  --recipient mabdel03@mit.edu \\
  --apply)"
printf '%s\\n' "$sentinel_report"
run_succeeded="$(printf '%s\\n' "$sentinel_report" | "$bootstrap_python" -I -S -c '
import json
import sys

value = json.load(sys.stdin).get("run_succeeded")
if not isinstance(value, bool):
    raise SystemExit("aggregate sentinel omitted Boolean run_succeeded")
print("true" if value else "false")
')"
if [[ "$run_succeeded" != true ]]; then
  exit 0
fi
exec env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S "$renderer" \\
  publish-bootstrap-handoff \\
  --chain-manifest {_q(paths.chain_manifest)} \\
  --apply
"""


def _stage_failure_sentinel_body(
    paths: RecoveryPaths,
    *,
    stage_name: str,
    sentinel_payload: bytes,
    sealer_payload: bytes,
    r4_failure_verifier_payload: bytes,
    r5_failure_verifier_payload: bytes,
    r6_failure_verifier_payload: bytes,
) -> str:
    """Render one independent, fail-fast observer for an exact production stage."""

    if stage_name not in PRODUCTION_STAGE_NAMES:
        raise ChainError(f"unknown stage-sentinel target: {stage_name!r}")
    observer_name = f"{STAGE_SENTINEL_PREFIX}{stage_name}"
    tool = paths.jobs_root / SENTINEL_TOOL_FILENAME
    tool_sha256 = _sha256_bytes(sentinel_payload)
    tool_size = len(sentinel_payload)
    return _common_exports(paths) + f"""\
[[ "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || {{ echo "stage sentinel requires an exact Slurm job ID" >&2; exit 2; }}
sentinel_tool={_q(tool)}
{_sentinel_bootstrap_runtime_shell(paths)}
{_immutable_tool_shell_check(
    variable="sentinel_tool",
    expected_sha256=tool_sha256,
    expected_size=tool_size,
    description="bundled recovery sentinel",
)}
{_r3_prelaunch_seal_shell_check(paths, sealer_payload=sealer_payload)}
{_r4_toolchain_failure_shell_check(
    paths, verifier_payload=r4_failure_verifier_payload
)}
{_r5_prelaunch_failure_shell_check(
    paths, verifier_payload=r5_failure_verifier_payload
)}
{_r6_prelaunch_failure_shell_check(
    paths, verifier_payload=r6_failure_verifier_payload
)}
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
comment="$(tr ' ' '\\n' <<<"$job_record" | sed -n 's/^Comment=//p')"
[[ "$(wc -l <<<"$comment")" -eq 1 ]] || {{ echo "stage sentinel scheduler comment is ambiguous" >&2; exit 2; }}
if [[ "$comment" =~ ^asys:s5-recovery-v1\\.2-r7:[0-9a-f]{{64}}:(g[0-9]{{4}}):{re.escape(observer_name)}$ ]]; then
  generation="${{BASH_REMATCH[1]}}"
else
  echo "stage sentinel scheduler comment is outside the immutable r7 namespace: $comment" >&2
  exit 2
fi
if [[ "$generation" == g0000 ]]; then
  receipt={_q(paths.recovery_root / SUBMISSION_RECEIPT_NAME)}
else
  receipt={_q(paths.recovery_root / REPAIR_ROOT_NAME)}/"$generation"/{_q(SUBMISSION_RECEIPT_NAME)}
fi
receipt_deadline=$(( $(date +%s) + 300 ))
while [[ ! -f "$receipt" ]]; do
  (( $(date +%s) < receipt_deadline )) || {{ echo "generation receipt was not published" >&2; exit 2; }}
  sleep 2
done
exec env LD_LIBRARY_PATH="$bootstrap_root/lib" "$bootstrap_python" -I -S "$sentinel_tool" \\
  --chain-manifest {_q(paths.chain_manifest)} \\
  --submission-receipt "$receipt" \\
  --output-root {_q(paths.recovery_root / 'recovery_chain_stage_sentinels' / CHAIN_NAMESPACE)}/"$generation"/{_q(stage_name)} \\
  --recipient mabdel03@mit.edu \\
  --stage-name {_q(stage_name)} \\
  --stage-sentinel-job-id "$SLURM_JOB_ID" \\
  --apply
"""


def job_specs(
    paths: RecoveryPaths,
    *,
    commit: str,
    tag_object: str,
    slurm_user: str,
    bundled_tools: Mapping[str, bytes],
    prerequisite_evidence: Mapping[str, Any],
    r1_protocol_verification: Mapping[str, Any],
) -> tuple[JobSpec, ...]:
    if not _SAFE_NAME.fullmatch(slurm_user):
        raise ChainError(f"unsafe Slurm user name: {slurm_user!r}")
    expected_tool_names = {Path(path).name for path in BUNDLED_TOOL_GIT_PATHS}
    if set(bundled_tools) != expected_tool_names or any(
        not isinstance(payload, bytes) or not payload
        for payload in bundled_tools.values()
    ):
        raise ChainError("immutable recovery tool payload set is incomplete")
    pilot = prerequisite_evidence.get("materialization_pilot")
    if not isinstance(pilot, Mapping):
        raise ChainError("materialization-pilot prerequisite is missing")
    source_inventories = pilot.get("live_source_inventory_sha256")
    policy_hashes = {
        "ownership": pilot.get("ownership_policy_sha256"),
        "integrity": pilot.get(
            "integrity_normalization_policy_sha256"
        ),
    }
    conda_toolchain_binding = pilot.get("conda_toolchain")
    package_cache_seed_input = pilot.get("package_cache_seed_input")
    reconciliation_incident = pilot.get("reconciliation_incident")
    if not isinstance(source_inventories, Mapping) or not isinstance(
        conda_toolchain_binding, Mapping
    ) or not isinstance(package_cache_seed_input, Mapping) or not isinstance(
        reconciliation_incident, Mapping
    ) or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in policy_hashes.values()
    ):
        raise ChainError("materialization-pilot runtime bindings are missing")
    renderer_payload = bundled_tools[
        Path("scripts/render_schema5_recovery_chain_v12.py").name
    ]
    return (
        JobSpec(
            "source_checkout",
            "00_source_checkout.sbatch",
            (),
            "01:00:00",
            "4G",
            1,
            _source_checkout_body(
                paths,
                commit,
                tag_object=tag_object,
                verifier_payload=bundled_tools[
                    Path("scripts/render_schema5_recovery_chain_v12.py").name
                ],
            ),
        ),
        JobSpec("maintenance_preflight", "01_maintenance_preflight.sbatch", ("source_checkout",), "02:00:00", "8G", 1, _maintenance_body(paths, slurm_user)),
        JobSpec(
            "snapshot_adopt_verify",
            "02_snapshot_adopt_verify.sbatch",
            ("maintenance_preflight",),
            "11:30:00",
            "8G",
            1,
            _snapshot_adopt_body(
                paths,
                commit=commit,
                tag_object=tag_object,
                bundled_tools=bundled_tools,
                r1_protocol_verification=r1_protocol_verification,
            ),
        ),
        JobSpec(
            "environment_capture",
            "03_environment_capture.sbatch",
            ("snapshot_adopt_verify",),
            "11:30:00",
            "12G",
            2,
            _environment_capture_body(
                paths,
                commit=commit,
                tag_object=tag_object,
                expected_source_inventories=source_inventories,
                expected_policy_hashes=policy_hashes,
                expected_reconciliation_incident=(
                    reconciliation_incident
                ),
            ),
        ),
        JobSpec(
            "release_materialize",
            "04_release_materialize.sbatch",
            ("environment_capture",),
            "11:30:00",
            "12G",
            2,
            _materialize_body(
                paths,
                commit=commit,
                tag_object=tag_object,
                expected_source_inventories=source_inventories,
                conda_toolchain_binding=conda_toolchain_binding,
                package_cache_seed_input=package_cache_seed_input,
            ),
        ),
        JobSpec(
            "release_freeze",
            "05_release_freeze.sbatch",
            ("release_materialize",),
            "11:30:00",
            "12G",
            2,
            _freeze_body(paths, commit=commit, tag_object=tag_object),
        ),
        JobSpec("legacy_consolidate", "06_legacy_consolidate.sbatch", ("release_freeze", "snapshot_adopt_verify"), "11:30:00", "16G", 2, _consolidate_body(paths)),
        JobSpec("legacy_consolidated_snapshot", "07_legacy_consolidated_snapshot.sbatch", ("legacy_consolidate",), "11:30:00", "8G", 1, _consolidated_snapshot_body(paths)),
        JobSpec("legacy_consolidated_verify", "08_legacy_consolidated_verify.sbatch", ("legacy_consolidated_snapshot",), "11:30:00", "8G", 1, _consolidated_verify_body(paths)),
        JobSpec("legacy_retire", "09_legacy_retire.sbatch", ("legacy_consolidated_verify",), "06:00:00", "8G", 1, _retire_body(paths)),
        JobSpec("schema5_initialize", "10_schema5_initialize.sbatch", ("legacy_retire",), "11:30:00", "16G", 2, _initialize_body(paths)),
        JobSpec("static_readiness", "11_static_readiness.sbatch", ("schema5_initialize",), "11:30:00", "8G", 1, _static_readiness_body(paths)),
        JobSpec("context_readiness", "13_context_readiness.sbatch", ("static_readiness",), "11:30:00", "32G", 4, _context_body(paths)),
        JobSpec("email_readiness", "14_email_readiness.sbatch", ("static_readiness",), "08:00:00", "2G", 1, _email_body(paths)),
        JobSpec("supplementary_cache", "15_supplementary_cache.sbatch", ("context_readiness",), "11:30:00", "64G", 4, _supplementary_body(paths)),
        JobSpec("fleet_bootstrap", "12_fleet_bootstrap.sbatch", ("supplementary_cache",), "02:00:00", "4G", 1, _bootstrap_fleet_body(paths)),
        JobSpec("fleet_readiness", "16_fleet_readiness.sbatch", ("fleet_bootstrap",), "11:00:00", "8G", 1, _fleet_wait_body(paths)),
        JobSpec(
            "smoke_readiness",
            "17_smoke_readiness.sbatch",
            ("fleet_readiness",),
            "11:30:00",
            "8G",
            1,
            _smoke_body(paths, renderer_payload=renderer_payload),
        ),
        JobSpec(
            "throughput_qualification",
            "18_throughput_qualification.sbatch",
            ("smoke_readiness",),
            "11:30:00",
            "8G",
            1,
            _throughput_qualification_body(
                paths, renderer_payload=renderer_payload
            ),
        ),
        JobSpec(
            "controller_drill",
            "19_controller_drill.sbatch",
            ("throughput_qualification", "email_readiness"),
            "02:00:00",
            "8G",
            1,
            _drill_body(
                paths,
                renderer_payload=renderer_payload,
                git_commit=commit,
                tag_object=tag_object,
            ),
        ),
        JobSpec(
            "production_resume",
            "20_production_resume.sbatch",
            ("controller_drill", "throughput_qualification"),
            "11:30:00",
            "8G",
            1,
            _resume_body(paths, renderer_payload=renderer_payload),
        ),
        *(
            JobSpec(
                f"{STAGE_SENTINEL_PREFIX}{stage}",
                f"{21 + index:02d}_stage_sentinel_{stage}.sbatch",
                (stage,),
                "01:00:00",
                "2G",
                1,
                _stage_failure_sentinel_body(
                    paths,
                    stage_name=stage,
                    sentinel_payload=bundled_tools[SENTINEL_TOOL_FILENAME],
                    sealer_payload=bundled_tools[
                        "seal_recovery_evidence.py"
                    ],
                    r4_failure_verifier_payload=bundled_tools[
                        "seal_schema5_r4_toolchain_failure.py"
                    ],
                    r5_failure_verifier_payload=bundled_tools[
                        "seal_schema5_r5_prelaunch_failure.py"
                    ],
                    r6_failure_verifier_payload=bundled_tools[
                        "seal_schema5_r6_prelaunch_failure.py"
                    ],
                ),
                dependency_type="afterany",
            )
            for index, stage in enumerate(PRODUCTION_STAGE_NAMES)
        ),
        JobSpec(
            "failure_sentinel",
            "42_failure_sentinel.sbatch",
            tuple(row[0] for row in EXPECTED_JOB_CONTRACT[:-1]),
            "01:00:00",
            "2G",
            1,
            _failure_sentinel_body(
                paths,
                sentinel_payload=bundled_tools[SENTINEL_TOOL_FILENAME],
                renderer_payload=renderer_payload,
                sealer_payload=bundled_tools[
                    "seal_recovery_evidence.py"
                ],
                r4_failure_verifier_payload=bundled_tools[
                    "seal_schema5_r4_toolchain_failure.py"
                ],
                r5_failure_verifier_payload=bundled_tools[
                    "seal_schema5_r5_prelaunch_failure.py"
                ],
                r6_failure_verifier_payload=bundled_tools[
                    "seal_schema5_r6_prelaunch_failure.py"
                ],
            ),
            dependency_type="afterany",
        ),
    )


def _job_name(name: str) -> str:
    if name.startswith(STAGE_SENTINEL_PREFIX):
        stage = name.removeprefix(STAGE_SENTINEL_PREFIX)
        if stage not in PRODUCTION_STAGE_NAMES:
            raise ChainError(f"unknown stage-sentinel job name: {name!r}")
        return f"asys-s5v12r7-alert-{PRODUCTION_STAGE_NAMES.index(stage):02d}"
    shortened = {
        "snapshot_adopt_verify": "snapshot-adopt",
        "environment_capture": "env-capture",
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
        "throughput_qualification": "qualify",
        "controller_drill": "drill",
        "production_resume": "resume",
        "source_checkout": "checkout",
        "legacy_retire": "retire",
        "failure_sentinel": "sentinel",
    }[name]
    return f"asys-s5v12r7-{shortened}"


def render_sbatch(spec: JobSpec, paths: RecoveryPaths, *, partition: str) -> bytes:
    if not _SAFE_NAME.fullmatch(partition):
        raise ChainError(f"unsafe Slurm partition: {partition!r}")
    log = paths.logs_root / f"{spec.name}_%j.out"
    launch_gate = (
        ""
        if spec.name == "failure_sentinel"
        or spec.name.startswith(STAGE_SENTINEL_PREFIX)
        else _launch_authorization_gate_shell(paths, spec_name=spec.name)
    )
    text = f"""#!/bin/bash
#SBATCH --job-name={_job_name(spec.name)}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={spec.cpus}
#SBATCH --mem={spec.memory}
#SBATCH --time={spec.time_limit}
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --output={log}

set -euo pipefail
umask 027
{_trusted_shell_prelude().rstrip()}
{launch_gate.rstrip()}
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
            spec.dependency_type,
        )
        for spec in ordered
    )
    if observed_contract != EXPECTED_JOB_CONTRACT:
        raise ChainError("recovery DAG differs from the fixed v1.2-r7 job contract")
    by_name = {spec.name: spec for spec in ordered}
    for earlier, later in zip(HEAVY_SERIAL_ORDER, HEAVY_SERIAL_ORDER[1:]):
        if earlier not in _ancestors(later, by_name):
            raise ChainError(f"heavy I/O is not serialized: {earlier} -> {later}")
    mutation_ancestors = _ancestors(FIRST_LEGACY_MUTATION, by_name)
    required = {"snapshot_adopt_verify", "release_freeze"}
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
        "throughput_qualification",
        "controller_drill",
        "supplementary_cache",
    }
    if not required_final <= final_ancestors:
        raise ChainError(
            f"production resume lacks gates: {sorted(required_final - final_ancestors)}"
        )
    sentinel = by_name["failure_sentinel"]
    expected_sentinel_dependencies = set(by_name) - {"failure_sentinel"}
    stage_observers_valid = all(
        (
            by_name[f"{STAGE_SENTINEL_PREFIX}{stage}"].dependency_type
            == "afterany"
            and by_name[f"{STAGE_SENTINEL_PREFIX}{stage}"].dependencies
            == (stage,)
        )
        for stage in PRODUCTION_STAGE_NAMES
    )
    if (
        sentinel.dependency_type != "afterany"
        or set(sentinel.dependencies) != expected_sentinel_dependencies
        or not stage_observers_valid
        or any(
            spec.dependency_type != "afterok"
            for spec in ordered
            if spec.name not in {"failure_sentinel", *STAGE_SENTINEL_NAMES}
        )
    ):
        raise ChainError(
            "per-stage and aggregate failure sentinels do not fence every "
            "afterok stage"
        )


def _preflight_fresh_destinations(paths: RecoveryPaths) -> None:
    for description, path in (
        ("v1.2-r7 source checkout", paths.source_checkout),
        (
            "v1.2-r7 source checkout seal",
            paths.recovery_root / SOURCE_CHECKOUT_SEAL_NAME,
        ),
        ("v1.2 production release root", paths.release_root),
        ("v1.2 environment capture root", paths.environment_capture_root),
        ("schema-5 control state", paths.state),
        ("schema-5 canonical server pool", paths.pool),
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
    _r1_evidence_contract(paths)


def _verify_jobs_namespace(
    jobs_root: Path,
    specs: Sequence[JobSpec],
    scripts: Mapping[str, bytes],
    bundled_tools: Mapping[str, bytes],
) -> None:
    if jobs_root.is_symlink() or not jobs_root.is_dir():
        raise ChainError(f"rendered jobs namespace is missing or unsafe: {jobs_root}")
    if jobs_root.resolve(strict=True) != jobs_root:
        raise ChainError(f"rendered jobs namespace traverses a symlink: {jobs_root}")
    if stat.S_IMODE(jobs_root.stat().st_mode) & 0o222:
        raise ChainError(f"rendered jobs namespace must be read-only: {jobs_root}")
    expected_names = {spec.filename for spec in specs} | set(bundled_tools)
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
    for filename, expected in bundled_tools.items():
        tool = jobs_root / filename
        if (
            tool.is_symlink()
            or not tool.is_file()
            or stat.S_IMODE(tool.stat().st_mode) & 0o222
            or tool.read_bytes() != expected
        ):
            raise ChainError(f"immutable recovery tool drifted: {tool}")


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
    bundled_tools: Mapping[str, bytes],
    prerequisite_evidence: Mapping[str, Any],
    r1_protocol_verification: Mapping[str, Any],
) -> dict[str, Any]:
    sentinel_bootstrap = _sentinel_bootstrap_contract(paths)
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
                "dependency_type": spec.dependency_type,
                "no_requeue": True,
                "time_limit": spec.time_limit,
                "memory": spec.memory,
                "cpus": spec.cpus,
            }
        )
    identity = {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "protocol": "schema5-v1.2-r7-recovery-chain",
        "namespace": CHAIN_NAMESPACE,
        "release_id": RELEASE_ID,
        "release_tag": git_identity["release_tag"],
        "release_git_commit": git_identity["git_commit"],
        "release_tag_object": git_identity["tag_object"],
        "repository": str(paths.repository),
        "results_root": str(paths.results_root),
        "recovery_root": str(paths.recovery_root),
        "source_checkout": str(paths.source_checkout),
        "release_root": str(paths.release_root),
        "state_root": str(paths.state),
        "server_pool_root": str(paths.pool),
        "hf_home": str(paths.hf_home),
        "dev_python": str(paths.dev_python),
        "dev_python_sha256": sentinel_bootstrap[
            "source_interpreter_sha256"
        ],
        "sentinel_bootstrap": sentinel_bootstrap,
        "source_harness_prefix": str(paths.source_harness),
        "source_serving_prefix": str(paths.source_serving),
        "conda_toolchain_root": str(paths.conda_toolchain_root),
        "conda_toolchain": dict(
            prerequisite_evidence["materialization_pilot"][
                "conda_toolchain"
            ]
        ),
        "source_package_cache": str(paths.source_package_cache),
        "environment_capture_root": str(paths.environment_capture_root),
        "captured_harness_prefix": str(paths.captured_harness),
        "captured_serving_prefix": str(paths.captured_serving),
        "prerequisite_evidence": dict(prerequisite_evidence),
        "r1_evidence": _r1_evidence_contract(paths),
        "r1_protocol_verification": dict(r1_protocol_verification),
        "sealed_snapshot_contract": dict(SEALED_SNAPSHOT_CONTRACT),
        "recovery_tool_bundle": [
            {
                "path": str(paths.jobs_root / Path(git_path).name),
                "sha256": _sha256_bytes(bundled_tools[Path(git_path).name]),
                "size": len(bundled_tools[Path(git_path).name]),
                "git_path": git_path,
            }
            for git_path in BUNDLED_TOOL_GIT_PATHS
        ],
        "jobs_root": str(paths.jobs_root),
        "logs_root": str(paths.logs_root),
        "immutable_pins": str(paths.immutable_pins),
        "readiness_root": str(paths.readiness),
        "partition": partition,
        "slurm_user": slurm_user,
        "heavy_io_serial_order": list(HEAVY_SERIAL_ORDER),
        "first_legacy_mutation": FIRST_LEGACY_MUTATION,
        "stage_failure_sentinels": [
            {
                "stage": stage,
                "sentinel": f"{STAGE_SENTINEL_PREFIX}{stage}",
            }
            for stage in PRODUCTION_STAGE_NAMES
        ],
        "jobs": jobs,
    }
    identity["chain_id"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _isolated_bootstrap_drill_location(
    manifest_path: Path,
) -> tuple[Path, Path] | None:
    """Return the canonical manifest/root for the one allowed sibling drill."""

    manifest_path = _lexical_absolute(manifest_path)
    isolated_root = manifest_path.parent
    if (
        manifest_path.name != CHAIN_MANIFEST_NAME
        or isolated_root.name != BOOTSTRAP_ISOLATED_DRILL_DIRECTORY
        or manifest_path != isolated_root / CHAIN_MANIFEST_NAME
    ):
        return None
    canonical_root = isolated_root.parent
    return canonical_root / CHAIN_MANIFEST_NAME, isolated_root


def _isolated_bootstrap_script(
    row: Mapping[str, Any],
    *,
    isolated_root: Path,
    partition: str,
) -> bytes:
    """Render a harmless topology-equivalent script for cancellation testing."""

    name = row.get("name")
    job_name = row.get("job_name")
    cpus = row.get("cpus")
    memory = row.get("memory")
    time_limit = row.get("time_limit")
    if (
        not isinstance(name, str)
        or _SAFE_NAME.fullmatch(name) is None
        or not isinstance(job_name, str)
        or _SAFE_NAME.fullmatch(job_name) is None
        or not isinstance(cpus, int)
        or isinstance(cpus, bool)
        or cpus < 1
        or not isinstance(memory, str)
        or re.fullmatch(r"[1-9][0-9]*[KMGT]", memory) is None
        or not isinstance(time_limit, str)
        or re.fullmatch(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", time_limit) is None
        or _SAFE_NAME.fullmatch(partition) is None
    ):
        raise ChainError(
            "canonical recovery topology cannot produce an isolated drill script"
        )
    log = isolated_root / "logs" / CHAIN_NAMESPACE / f"{name}_%j.out"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --output={log}

set -euo pipefail
umask 077
{_trusted_shell_prelude().rstrip()}
printf '%s\\n' 'schema5 isolated bootstrap cancellation drill: execution is forbidden' >&2
exit {BOOTSTRAP_ISOLATED_SCRIPT_EXIT_CODE}
""".encode("utf-8")


def _isolated_bootstrap_manifest_payload(
    canonical_manifest: Mapping[str, Any],
    *,
    canonical_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Derive the exact inert sibling manifest from the verified canonical DAG."""

    canonical_root = canonical_manifest_path.parent
    isolated_root = canonical_root / BOOTSTRAP_ISOLATED_DRILL_DIRECTORY
    isolated_jobs_root = isolated_root / "jobs" / CHAIN_NAMESPACE
    isolated_logs_root = isolated_root / "logs" / CHAIN_NAMESPACE
    partition = canonical_manifest.get("partition")
    jobs = canonical_manifest.get("jobs")
    tools = canonical_manifest.get("recovery_tool_bundle")
    if (
        not isinstance(partition, str)
        or not isinstance(jobs, list)
        or len(jobs) != len(EXPECTED_JOB_ORDER)
        or not isinstance(tools, list)
    ):
        raise ChainError(
            "canonical recovery manifest cannot seed the isolated drill"
        )

    payload = json.loads(
        json.dumps(canonical_manifest, allow_nan=False)
    )
    payload.pop("chain_id", None)
    payload.update(
        {
            "recovery_root": str(isolated_root),
            "source_checkout": str(
                isolated_root / "release_source_checkout_v1_2_r7"
            ),
            "release_root": str(
                isolated_root / "releases" / RELEASE_ID
            ),
            "environment_capture_root": str(
                isolated_root / "environment_captures" / RELEASE_ID
            ),
            "captured_harness_prefix": str(
                isolated_root
                / "environment_captures"
                / RELEASE_ID
                / "seeds"
                / "harness"
            ),
            "captured_serving_prefix": str(
                isolated_root
                / "environment_captures"
                / RELEASE_ID
                / "seeds"
                / "serving"
            ),
            "jobs_root": str(isolated_jobs_root),
            "logs_root": str(isolated_logs_root),
            "immutable_pins": str(
                isolated_root / "immutable_pins.schema5-v1.json"
            ),
            "readiness_root": str(isolated_root / "readiness"),
        }
    )

    scripts: dict[str, bytes] = {}
    isolated_jobs: list[dict[str, Any]] = []
    for row in jobs:
        if not isinstance(row, Mapping):
            raise ChainError("canonical recovery manifest has a malformed job")
        name = str(row.get("name", ""))
        script = _isolated_bootstrap_script(
            row,
            isolated_root=isolated_root,
            partition=partition,
        )
        scripts[name] = script
        isolated_row = dict(row)
        isolated_row["script"] = str(
            isolated_jobs_root / Path(str(row["script"])).name
        )
        isolated_row["script_sha256"] = _sha256_bytes(script)
        isolated_jobs.append(isolated_row)
    payload["jobs"] = isolated_jobs

    isolated_tools: list[dict[str, Any]] = []
    for record in tools:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
        ):
            raise ChainError(
                "canonical recovery tool binding is malformed"
            )
        isolated_record = dict(record)
        isolated_record["path"] = str(
            isolated_jobs_root / Path(record["path"]).name
        )
        isolated_tools.append(isolated_record)
    payload["recovery_tool_bundle"] = isolated_tools
    payload["chain_id"] = _sha256_bytes(_canonical_json(payload))
    return payload, scripts


def _canonical_prelaunch_receipt(
    *,
    canonical_manifest: Mapping[str, Any],
    canonical_manifest_path: Path,
) -> dict[str, Any]:
    """Require one held canonical g0 receipt and no canonical launch mutation."""

    canonical_root = canonical_manifest_path.parent
    for forbidden in (
        canonical_root / REPAIR_ROOT_NAME,
        canonical_root / ROOT_RELEASE_COMPLETE_NAME,
        canonical_root / LAUNCH_COMPLETE_NAME,
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise ChainError(
                "isolated cancellation drill requires an untouched canonical "
                f"prelaunch namespace: {forbidden}"
            )
    receipt_path = canonical_root / SUBMISSION_RECEIPT_NAME
    return _validate_submission_receipt(
        receipt_path,
        manifest=canonical_manifest,
        manifest_path=canonical_manifest_path,
        comments=_submission_comments(canonical_manifest),
    )


def verify_isolated_bootstrap_drill_chain(
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the exact inert sibling used only for bootstrap cancellation."""

    manifest_path = _lexical_absolute(manifest_path)
    location = _isolated_bootstrap_drill_location(manifest_path)
    if location is None:
        raise ChainError(
            "isolated bootstrap drill manifest is outside its fixed sibling root"
        )
    canonical_manifest_path, isolated_root = location
    manifest_path = _require_canonical_path(
        manifest_path,
        description="isolated bootstrap drill manifest",
        kind="file",
    )
    manifest_metadata = manifest_path.stat(follow_symlinks=False)
    if (
        stat.S_IMODE(manifest_metadata.st_mode) & 0o222
        or manifest_metadata.st_nlink != 1
    ):
        raise ChainError(
            "isolated bootstrap drill manifest must be one read-only file"
        )

    canonical_verification = verify_chain(canonical_manifest_path)
    canonical_manifest = _read_json(
        canonical_manifest_path,
        description="canonical recovery-chain manifest",
    )
    _canonical_prelaunch_receipt(
        canonical_manifest=canonical_manifest,
        canonical_manifest_path=canonical_manifest_path,
    )
    expected, scripts = _isolated_bootstrap_manifest_payload(
        canonical_manifest,
        canonical_manifest_path=canonical_manifest_path,
    )
    manifest = _read_json(
        manifest_path,
        description="isolated bootstrap drill manifest",
    )
    if manifest != expected:
        raise ChainError(
            "isolated bootstrap drill manifest is not the exact inert "
            "canonical-topology derivation"
        )

    root = _require_canonical_path(
        isolated_root,
        description="isolated bootstrap drill root",
        kind="directory",
    )
    jobs_root = _require_canonical_path(
        Path(str(manifest["jobs_root"])),
        description="isolated bootstrap drill jobs",
        kind="directory",
    )
    logs_root = _require_canonical_path(
        Path(str(manifest["logs_root"])),
        description="isolated bootstrap drill logs",
        kind="directory",
    )
    if (
        root != isolated_root
        or jobs_root != isolated_root / "jobs" / CHAIN_NAMESPACE
        or logs_root != isolated_root / "logs" / CHAIN_NAMESPACE
        or stat.S_IMODE(jobs_root.stat().st_mode) & 0o222
    ):
        raise ChainError(
            "isolated bootstrap drill filesystem namespace drifted"
        )
    for forbidden in (
        isolated_root / ROOT_RELEASE_COMPLETE_NAME,
        isolated_root / LAUNCH_COMPLETE_NAME,
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise ChainError(
                "isolated bootstrap drill must never publish a release marker"
            )

    canonical_tools = {
        Path(str(record["path"])).name: record
        for record in canonical_manifest["recovery_tool_bundle"]
    }
    expected_names = {
        Path(str(row["script"])).name for row in manifest["jobs"]
    } | set(canonical_tools)
    if {path.name for path in jobs_root.iterdir()} != expected_names:
        raise ChainError(
            "isolated bootstrap drill jobs namespace contents drifted"
        )
    for row in manifest["jobs"]:
        name = str(row["name"])
        path = _require_canonical_path(
            Path(str(row["script"])),
            description=f"isolated bootstrap script {name}",
            kind="file",
        )
        metadata = path.stat(follow_symlinks=False)
        expected_script = scripts[name]
        if (
            path.parent != jobs_root
            or stat.S_IMODE(metadata.st_mode) & 0o222
            or metadata.st_nlink != 1
            or path.read_bytes() != expected_script
            or row["script_sha256"] != _sha256_bytes(expected_script)
        ):
            raise ChainError(
                f"isolated bootstrap drill script drifted: {name}"
            )
    for name, canonical_record in canonical_tools.items():
        canonical_tool = _require_canonical_path(
            Path(str(canonical_record["path"])),
            description=f"canonical recovery tool {name}",
            kind="file",
        )
        isolated_tool = _require_canonical_path(
            jobs_root / name,
            description=f"isolated recovery tool {name}",
            kind="file",
        )
        metadata = isolated_tool.stat(follow_symlinks=False)
        if (
            stat.S_IMODE(metadata.st_mode) & 0o222
            or metadata.st_nlink != 1
            or isolated_tool.read_bytes() != canonical_tool.read_bytes()
            or _sha256(isolated_tool) != canonical_record["sha256"]
        ):
            raise ChainError(
                f"isolated bootstrap recovery tool drifted: {name}"
            )
    return {
        "passed": True,
        "isolated_bootstrap_drill": True,
        "chain_id": manifest["chain_id"],
        "canonical_chain_id": canonical_manifest["chain_id"],
        "canonical_verification": canonical_verification,
        "job_count": len(manifest["jobs"]),
        "inert_exit_code": BOOTSTRAP_ISOLATED_SCRIPT_EXIT_CODE,
    }


def prepare_isolated_bootstrap_drill_chain(
    canonical_manifest_path: Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Publish a topology-equivalent, inert sibling DAG marker last."""

    canonical_manifest_path = _lexical_absolute(canonical_manifest_path)
    verify_chain(canonical_manifest_path)
    canonical_manifest = _read_json(
        canonical_manifest_path,
        description="canonical recovery-chain manifest",
    )
    canonical_receipt = _canonical_prelaunch_receipt(
        canonical_manifest=canonical_manifest,
        canonical_manifest_path=canonical_manifest_path,
    )
    manifest, scripts = _isolated_bootstrap_manifest_payload(
        canonical_manifest,
        canonical_manifest_path=canonical_manifest_path,
    )
    canonical_root = canonical_manifest_path.parent
    isolated_root = canonical_root / BOOTSTRAP_ISOLATED_DRILL_DIRECTORY
    isolated_manifest_path = isolated_root / CHAIN_MANIFEST_NAME
    report = {
        "status": "dry_run",
        "isolated_bootstrap_drill": True,
        "canonical_chain_id": canonical_manifest["chain_id"],
        "canonical_submission_receipt_id": canonical_receipt["receipt_id"],
        "chain_id": manifest["chain_id"],
        "chain_manifest": str(isolated_manifest_path),
        "job_count": len(manifest["jobs"]),
        "inert_exit_code": BOOTSTRAP_ISOLATED_SCRIPT_EXIT_CODE,
    }
    if isolated_root.exists() or isolated_root.is_symlink():
        verified = verify_isolated_bootstrap_drill_chain(
            isolated_manifest_path
        )
        return report | {
            "status": "already_prepared",
            "verified": verified,
        }
    if not apply:
        return report

    lock_path = canonical_root / BOOTSTRAP_ISOLATED_RENDER_LOCK_NAME
    with _exclusive_lock(
        lock_path,
        description="isolated bootstrap drill renderer",
    ):
        if isolated_root.exists() or isolated_root.is_symlink():
            verified = verify_isolated_bootstrap_drill_chain(
                isolated_manifest_path
            )
            return report | {
                "status": "already_prepared",
                "verified": verified,
            }
        temporary_root = canonical_root / (
            f".{BOOTSTRAP_ISOLATED_DRILL_DIRECTORY}.render."
            f"{os.getpid()}.{uuid.uuid4().hex}"
        )
        temporary_root.mkdir(mode=0o750)
        try:
            temporary_jobs = temporary_root / "jobs" / CHAIN_NAMESPACE
            temporary_logs = temporary_root / "logs" / CHAIN_NAMESPACE
            temporary_jobs.mkdir(parents=True, mode=0o750)
            temporary_logs.mkdir(parents=True, mode=0o750)
            for row in manifest["jobs"]:
                name = str(row["name"])
                target = temporary_jobs / Path(str(row["script"])).name
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o444,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(scripts[name])
                    handle.flush()
                    os.fsync(handle.fileno())
            for record in canonical_manifest["recovery_tool_bundle"]:
                source = _require_canonical_path(
                    Path(str(record["path"])),
                    description="canonical recovery tool",
                    kind="file",
                )
                target = temporary_jobs / source.name
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o444,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(source.read_bytes())
                    handle.flush()
                    os.fsync(handle.fileno())
            os.chmod(temporary_jobs, 0o550)
            _fsync_directory(temporary_jobs)
            _fsync_directory(temporary_logs)
            _atomic_json(
                temporary_root / CHAIN_MANIFEST_NAME,
                manifest,
                mode=0o444,
            )
            _fsync_directory(temporary_root)
            os.rename(temporary_root, isolated_root)
            _fsync_directory(canonical_root)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
    verified = verify_isolated_bootstrap_drill_chain(
        isolated_manifest_path
    )
    return report | {"status": "complete", "verified": verified}


def render_chain(
    paths: RecoveryPaths,
    *,
    partition: str = "mit_normal",
    slurm_user: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    _reject_live_prefix_execution(
        source_harness=paths.source_harness,
        source_serving=paths.source_serving,
    )
    if partition != "mit_normal":
        raise ChainError(
            "durable recovery wrappers require the non-preempting mit_normal partition"
        )
    slurm_user = getpass.getuser() if slurm_user is None else slurm_user
    git_identity = verify_release_tag(paths.repository)
    bundled_tools = {
        Path(git_path).name: _tagged_file_bytes(
            paths.repository,
            git_identity["git_commit"],
            git_path,
        )
        for git_path in BUNDLED_TOOL_GIT_PATHS
    }
    prerequisite_evidence = _prerequisite_evidence_contract(
        paths,
        git_identity=git_identity,
        verifier_checkout=paths.repository,
    )
    r1_verifier_python, r1_verifier_library = _verify_pilot_runtime_binding(
        paths,
        prerequisite_evidence["materialization_pilot"]["verifier_runtime"],
    )
    r1_protocol_verification = _r1_protocol_verification_contract(
        paths,
        checkout=paths.repository,
        git_identity=git_identity,
        verifier_python=r1_verifier_python,
        verifier_library=r1_verifier_library,
    )
    if apply:
        _publish_sentinel_bootstrap(paths)
    else:
        # A dry run hashes the exact prospective payload but performs no capture.
        _prospective_sentinel_bootstrap(paths)
    specs = job_specs(
        paths,
        commit=git_identity["git_commit"],
        tag_object=git_identity["tag_object"],
        slurm_user=slurm_user,
        bundled_tools=bundled_tools,
        prerequisite_evidence=prerequisite_evidence,
        r1_protocol_verification=r1_protocol_verification,
    )
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
        bundled_tools=bundled_tools,
        prerequisite_evidence=prerequisite_evidence,
        r1_protocol_verification=r1_protocol_verification,
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
            ("v1.2-r7 job namespace", paths.jobs_root),
            ("v1.2-r7 log namespace", paths.logs_root),
            ("v1.2-r7 chain manifest", paths.chain_manifest),
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
            _verify_jobs_namespace(paths.jobs_root, specs, scripts, bundled_tools)
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
                for filename, payload in bundled_tools.items():
                    tool_target = temporary_jobs / filename
                    descriptor = os.open(
                        tool_target,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o444,
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(payload)
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

            _verify_jobs_namespace(paths.jobs_root, specs, scripts, bundled_tools)
            _verify_logs_namespace(paths.logs_root, require_empty=True)
            # This immutable manifest is the only successful-render signal.
            _atomic_json(paths.chain_manifest, manifest, mode=0o444)
        finally:
            for temporary in (temporary_jobs, temporary_logs):
                if temporary is not None and temporary.exists():
                    shutil.rmtree(temporary)
        verified = verify_chain(paths.chain_manifest)
        return report | {"status": "complete", "verified": verified}


def verify_bound_prerequisites(
    manifest_path: Path,
    *,
    expected_chain_id: str,
    markers_only: bool = False,
    verifier_checkout: Path | None = None,
    verifier_python: Path | None = None,
    verifier_library: Path | None = None,
) -> dict[str, Any]:
    """Reverify the scheduler-bound prerequisites without mutating any evidence."""

    manifest_path = _require_canonical_path(
        _lexical_absolute(manifest_path),
        description="recovery-chain manifest",
        kind="file",
    )
    if stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
        raise ChainError("recovery-chain manifest must be read-only")
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    identity = dict(manifest)
    chain_id = identity.pop("chain_id", None)
    prerequisite = manifest.get("prerequisite_evidence")
    if (
        manifest.get("schema_version") != CHAIN_SCHEMA_VERSION
        or manifest.get("protocol") != "schema5-v1.2-r7-recovery-chain"
        or chain_id != expected_chain_id
        or not isinstance(chain_id, str)
        or _SHA256.fullmatch(chain_id) is None
        or chain_id != _sha256_bytes(_canonical_json(identity))
        or not isinstance(prerequisite, dict)
        or not isinstance(prerequisite.get("materialization_pilot"), dict)
        or not isinstance(prerequisite.get("slurm_canary"), dict)
    ):
        raise ChainError("scheduler-bound recovery-chain identity is invalid")
    repository = _manifest_path(
        manifest.get("repository"), description="manifest repository"
    )
    results_root = _manifest_path(
        manifest.get("results_root"), description="manifest results root"
    )
    recovery_root = _manifest_path(
        manifest.get("recovery_root"), description="manifest recovery root"
    )
    if (
        manifest_path != recovery_root / CHAIN_MANIFEST_NAME
        or recovery_root != results_root / "recovery" / "schema5-v1"
    ):
        raise ChainError("scheduler-bound recovery root is not canonical")
    pilot_root = _manifest_path(
        prerequisite["materialization_pilot"].get("root"),
        description="manifest materialization pilot root",
    )
    canary_root = _manifest_path(
        prerequisite["slurm_canary"].get("root"),
        description="manifest Slurm canary root",
    )
    if (
        pilot_root != recovery_root / "materialization_pilots" / CHAIN_NAMESPACE
        or canary_root != recovery_root / "slurm_canaries" / CHAIN_NAMESPACE
    ):
        raise ChainError("scheduler-bound prerequisite roots are not canonical")
    release_root = _manifest_path(
        manifest.get("release_root"), description="manifest release root"
    )
    paths = RecoveryPaths(
        repository=repository,
        results_root=results_root,
        recovery_root=recovery_root,
        source_checkout=_manifest_path(
            manifest.get("source_checkout"), description="manifest source checkout"
        ),
        release_root=release_root,
        worktree=release_root / "worktree",
        identity=release_root / "identity",
        harness=release_root / "environments" / "harness",
        serving=release_root / "environments" / "serving",
        state=_manifest_path(
            manifest.get("state_root"), description="manifest state root"
        ),
        pool=_manifest_path(
            manifest.get("server_pool_root"), description="manifest server pool root"
        ),
        hf_home=_manifest_path(
            manifest.get("hf_home"), description="manifest HF home"
        ),
        dev_python=_manifest_path(
            manifest.get("dev_python"), description="manifest development Python"
        ),
        source_harness=_manifest_path(
            manifest.get("source_harness_prefix"),
            description="manifest source harness prefix",
        ),
        source_serving=_manifest_path(
            manifest.get("source_serving_prefix"),
            description="manifest source serving prefix",
        ),
        conda_toolchain_root=_manifest_path(
            manifest.get("conda_toolchain_root"),
            description="manifest Conda toolchain root",
        ),
        source_package_cache=_manifest_path(
            manifest.get("source_package_cache"),
            description="manifest source Conda package cache",
        ),
        environment_capture_root=_manifest_path(
            manifest.get("environment_capture_root"),
            description="manifest environment capture root",
        ),
        captured_harness=_manifest_path(
            manifest.get("captured_harness_prefix"),
            description="manifest captured harness prefix",
        ),
        captured_serving=_manifest_path(
            manifest.get("captured_serving_prefix"),
            description="manifest captured serving prefix",
        ),
        jobs_root=_manifest_path(
            manifest.get("jobs_root"), description="manifest jobs root"
        ),
        logs_root=_manifest_path(
            manifest.get("logs_root"), description="manifest logs root"
        ),
        chain_manifest=manifest_path,
        immutable_pins=_manifest_path(
            manifest.get("immutable_pins"), description="manifest immutable pins"
        ),
        readiness=_manifest_path(
            manifest.get("readiness_root"), description="manifest readiness root"
        ),
        materialization_pilot_root=pilot_root,
        slurm_canary_root=canary_root,
    )
    commit = manifest.get("release_git_commit")
    tag_object = manifest.get("release_tag_object")
    if (
        manifest.get("release_tag") != RELEASE_TAG
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(tag_object, str)
        or re.fullmatch(r"[0-9a-f]{40}", tag_object) is None
    ):
        raise ChainError("scheduler-bound release identity is invalid")
    contract = _validate_prerequisite_evidence_contract(
        paths,
        prerequisite,
        git_identity={
            "release_tag": RELEASE_TAG,
            "git_commit": commit,
            "tag_object": tag_object,
        },
        verifier_checkout=verifier_checkout,
        markers_only=markers_only,
        verifier_python=verifier_python,
        verifier_library=verifier_library,
    )
    if manifest.get("conda_toolchain") != contract[
        "materialization_pilot"
    ].get("conda_toolchain"):
        raise ChainError("scheduler-bound Conda toolchain binding drifted")
    native_r1_id = None
    if not markers_only:
        sealed_python, sealed_library = _verify_pilot_runtime_binding(
            paths,
            contract["materialization_pilot"]["verifier_runtime"],
        )
        native_r1 = _r1_protocol_verification_contract(
            paths,
            checkout=_lexical_absolute(verifier_checkout),
            git_identity={
                "release_tag": RELEASE_TAG,
                "git_commit": commit,
                "tag_object": tag_object,
            },
            verifier_python=sealed_python,
            verifier_library=sealed_library,
        )
        if manifest.get("r1_protocol_verification") != native_r1:
            raise ChainError("native r1 protocol verification binding drifted")
        native_r1_id = native_r1["verification_id"]
    return {
        "passed": True,
        "markers_only": markers_only,
        "chain_id": chain_id,
        "prerequisite_evidence_id": contract["evidence_id"],
        "materialization_pilot_id": contract["materialization_pilot"][
            "pilot_id"
        ],
        "slurm_canary_id": contract["slurm_canary"]["canary_id"],
        "r1_protocol_verification_id": native_r1_id,
        "release_tag": RELEASE_TAG,
        "release_git_commit": commit,
    }


def verify_live_creation_prerequisites(manifest_path: Path) -> dict[str, Any]:
    """Recheck mutable creation inputs without weakening sealed ``verify_chain``."""

    isolated_location = _isolated_bootstrap_drill_location(manifest_path)
    if isolated_location is not None:
        canonical_manifest_path, _isolated_root = isolated_location
        isolated = verify_isolated_bootstrap_drill_chain(manifest_path)
        canonical = verify_live_creation_prerequisites(
            canonical_manifest_path
        )
        return canonical | {
            "passed": True,
            "isolated_bootstrap_drill": True,
            "chain_id": isolated["chain_id"],
            "canonical_chain_id": isolated["canonical_chain_id"],
        }

    manifest_path = _require_canonical_path(
        _lexical_absolute(manifest_path),
        description="recovery-chain manifest",
        kind="file",
    )
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    source_checkout = _manifest_path(
        manifest.get("source_checkout"),
        description="manifest source checkout",
    )
    repository = _manifest_path(
        manifest.get("repository"),
        description="manifest repository",
    )
    checkout = source_checkout if source_checkout.exists() else repository
    return verify_bound_prerequisites(
        manifest_path,
        expected_chain_id=str(manifest.get("chain_id", "")),
        markers_only=False,
        verifier_checkout=checkout,
    )


def verify_chain(manifest_path: Path) -> dict[str, Any]:
    manifest_path = _lexical_absolute(manifest_path)
    if _isolated_bootstrap_drill_location(manifest_path) is not None:
        return verify_isolated_bootstrap_drill_chain(manifest_path)
    if manifest_path.is_symlink():
        raise ChainError(f"recovery-chain manifest cannot be a symlink: {manifest_path}")
    manifest_path = _require_canonical_path(
        manifest_path,
        description="recovery-chain manifest",
        kind="file",
    )
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    _reject_live_prefix_execution(
        source_harness=_manifest_path(
            manifest.get("source_harness_prefix"),
            description="manifest source harness prefix",
        ),
        source_serving=_manifest_path(
            manifest.get("source_serving_prefix"),
            description="manifest source serving prefix",
        ),
    )
    expected_fields = {
        "schema_version",
        "protocol",
        "namespace",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "repository",
        "results_root",
        "recovery_root",
        "source_checkout",
        "release_root",
        "state_root",
        "server_pool_root",
        "hf_home",
        "dev_python",
        "dev_python_sha256",
        "sentinel_bootstrap",
        "source_harness_prefix",
        "source_serving_prefix",
        "conda_toolchain_root",
        "conda_toolchain",
        "source_package_cache",
        "environment_capture_root",
        "captured_harness_prefix",
        "captured_serving_prefix",
        "prerequisite_evidence",
        "r1_evidence",
        "r1_protocol_verification",
        "sealed_snapshot_contract",
        "recovery_tool_bundle",
        "jobs_root",
        "logs_root",
        "immutable_pins",
        "readiness_root",
        "partition",
        "slurm_user",
        "heavy_io_serial_order",
        "first_legacy_mutation",
        "stage_failure_sentinels",
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
        or manifest["protocol"] != "schema5-v1.2-r7-recovery-chain"
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
    prerequisite = manifest["prerequisite_evidence"]
    if (
        not isinstance(prerequisite, dict)
        or not isinstance(prerequisite.get("materialization_pilot"), dict)
        or not isinstance(prerequisite.get("slurm_canary"), dict)
    ):
        raise ChainError("recovery-chain prerequisite evidence is malformed")
    materialization_pilot_root = _manifest_path(
        prerequisite["materialization_pilot"].get("root"),
        description="manifest materialization pilot root",
    )
    slurm_canary_root = _manifest_path(
        prerequisite["slurm_canary"].get("root"),
        description="manifest Slurm canary root",
    )

    bootstrap_marker = _verify_sentinel_bootstrap(
        RecoveryPaths(
            repository=_manifest_path(
                manifest["repository"], description="manifest repository"
            ),
            results_root=_manifest_path(
                manifest["results_root"], description="manifest results root"
            ),
            recovery_root=_manifest_path(
                manifest["recovery_root"], description="manifest recovery root"
            ),
            source_checkout=_manifest_path(
                manifest["source_checkout"], description="manifest source checkout"
            ),
            release_root=_manifest_path(
                manifest["release_root"], description="manifest release root"
            ),
            worktree=_manifest_path(
                manifest["release_root"], description="manifest release root"
            )
            / "worktree",
            identity=_manifest_path(
                manifest["release_root"], description="manifest release root"
            )
            / "identity",
            harness=_manifest_path(
                manifest["release_root"], description="manifest release root"
            )
            / "environments"
            / "harness",
            serving=_manifest_path(
                manifest["release_root"], description="manifest release root"
            )
            / "environments"
            / "serving",
            state=_manifest_path(
                manifest["state_root"], description="manifest state root"
            ),
            pool=_manifest_path(
                manifest["server_pool_root"],
                description="manifest server pool root",
            ),
            hf_home=_manifest_path(
                manifest["hf_home"], description="manifest HF home"
            ),
            dev_python=_manifest_path(
                manifest["dev_python"], description="manifest development Python"
            ),
            source_harness=_manifest_path(
                manifest["source_harness_prefix"],
                description="manifest source harness prefix",
            ),
            source_serving=_manifest_path(
                manifest["source_serving_prefix"],
                description="manifest source serving prefix",
            ),
            conda_toolchain_root=_manifest_path(
                manifest["conda_toolchain_root"],
                description="manifest Conda toolchain root",
            ),
            source_package_cache=_manifest_path(
                manifest["source_package_cache"],
                description="manifest source Conda package cache",
            ),
            environment_capture_root=_manifest_path(
                manifest["environment_capture_root"],
                description="manifest environment capture root",
            ),
            captured_harness=_manifest_path(
                manifest["captured_harness_prefix"],
                description="manifest captured harness prefix",
            ),
            captured_serving=_manifest_path(
                manifest["captured_serving_prefix"],
                description="manifest captured serving prefix",
            ),
            jobs_root=_manifest_path(
                manifest["jobs_root"], description="manifest jobs root"
            ),
            logs_root=_manifest_path(
                manifest["logs_root"], description="manifest logs root"
            ),
            chain_manifest=manifest_path,
            immutable_pins=_manifest_path(
                manifest["immutable_pins"], description="manifest immutable pins"
            ),
            readiness=_manifest_path(
                manifest["readiness_root"], description="manifest readiness root"
            ),
            materialization_pilot_root=materialization_pilot_root,
            slurm_canary_root=slurm_canary_root,
        )
    )
    if manifest["sentinel_bootstrap"] != bootstrap_marker:
        raise ChainError("recovery-chain sentinel bootstrap binding drifted")

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
        "conda_toolchain_root",
        "source_package_cache",
        "environment_capture_root",
        "captured_harness_prefix",
        "captured_serving_prefix",
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
    if (
        materialization_pilot_root
        != recovery_root / "materialization_pilots" / CHAIN_NAMESPACE
        or slurm_canary_root
        != recovery_root / "slurm_canaries" / CHAIN_NAMESPACE
    ):
        raise ChainError("recovery-chain prerequisite roots are not canonical")
    expected_paths = {
        "recovery_root": results_root / "recovery" / "schema5-v1",
        "source_checkout": recovery_root / "release_source_checkout_v1_2_r7",
        "release_root": recovery_root / "releases" / RELEASE_ID,
        "state_root": results_root / ".dispatcher-schema5-v1",
        "server_pool_root": results_root / "server_pools" / "schema5-v1",
        "jobs_root": recovery_root / "jobs" / CHAIN_NAMESPACE,
        "logs_root": recovery_root / "logs" / CHAIN_NAMESPACE,
        "immutable_pins": recovery_root / "immutable_pins.schema5-v1.json",
        "readiness_root": recovery_root / "readiness",
        "conda_toolchain_root": (
            recovery_root
            / "toolchains"
            / conda_toolchain.TOOLCHAIN_NAMESPACE_DIRECTORY
            / conda_toolchain.TOOLCHAIN_DIRECTORY_NAME
        ),
        "environment_capture_root": (
            recovery_root / "environment_captures" / RELEASE_ID
        ),
        "captured_harness_prefix": (
            recovery_root / "environment_captures" / RELEASE_ID / "seeds" / "harness"
        ),
        "captured_serving_prefix": (
            recovery_root / "environment_captures" / RELEASE_ID / "seeds" / "serving"
        ),
    }
    if any(paths_by_name[name] != expected for name, expected in expected_paths.items()):
        raise ChainError("recovery-chain canonical path contract drifted")
    if manifest_path != recovery_root / CHAIN_MANIFEST_NAME:
        raise ChainError("recovery-chain manifest is outside its canonical recovery root")
    if (
        not isinstance(manifest["release_git_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["release_git_commit"])
        or not isinstance(manifest["release_tag_object"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["release_tag_object"])
        or manifest["partition"] != "mit_normal"
        or not isinstance(manifest["slurm_user"], str)
        or not _SAFE_NAME.fullmatch(manifest["slurm_user"])
        or manifest["first_legacy_mutation"] != FIRST_LEGACY_MUTATION
        or manifest["heavy_io_serial_order"] != list(HEAVY_SERIAL_ORDER)
        or manifest["stage_failure_sentinels"]
        != [
            {
                "stage": stage,
                "sentinel": f"{STAGE_SENTINEL_PREFIX}{stage}",
            }
            for stage in PRODUCTION_STAGE_NAMES
        ]
        or manifest["dev_python_sha256"]
        != bootstrap_marker["source_interpreter_sha256"]
        or manifest["sealed_snapshot_contract"] != SEALED_SNAPSHOT_CONTRACT
        or manifest["r1_evidence"] != _r1_evidence_contract(
            RecoveryPaths(
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
                conda_toolchain_root=paths_by_name["conda_toolchain_root"],
                source_package_cache=paths_by_name["source_package_cache"],
                environment_capture_root=paths_by_name["environment_capture_root"],
                captured_harness=paths_by_name["captured_harness_prefix"],
                captured_serving=paths_by_name["captured_serving_prefix"],
                jobs_root=paths_by_name["jobs_root"],
                logs_root=paths_by_name["logs_root"],
                chain_manifest=manifest_path,
                immutable_pins=paths_by_name["immutable_pins"],
                readiness=paths_by_name["readiness_root"],
                materialization_pilot_root=materialization_pilot_root,
                slurm_canary_root=slurm_canary_root,
            )
        )
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
        conda_toolchain_root=paths_by_name["conda_toolchain_root"],
        source_package_cache=paths_by_name["source_package_cache"],
        environment_capture_root=paths_by_name["environment_capture_root"],
        captured_harness=paths_by_name["captured_harness_prefix"],
        captured_serving=paths_by_name["captured_serving_prefix"],
        jobs_root=paths_by_name["jobs_root"],
        logs_root=paths_by_name["logs_root"],
        chain_manifest=manifest_path,
        immutable_pins=paths_by_name["immutable_pins"],
        readiness=paths_by_name["readiness_root"],
        materialization_pilot_root=materialization_pilot_root,
        slurm_canary_root=slurm_canary_root,
    )
    verified_prerequisites = _validate_prerequisite_evidence_contract(
        reconstructed,
        manifest["prerequisite_evidence"],
        git_identity={
            "release_tag": RELEASE_TAG,
            "git_commit": manifest["release_git_commit"],
            "tag_object": manifest["release_tag_object"],
        },
        verifier_checkout=None,
        markers_only=False,
        sealed_only=True,
    )
    if manifest.get("conda_toolchain") != verified_prerequisites[
        "materialization_pilot"
    ].get("conda_toolchain"):
        raise ChainError("recovery-chain Conda toolchain binding drifted")
    bundled_tools: dict[str, bytes] = {}
    bundle_records = manifest.get("recovery_tool_bundle")
    if not isinstance(bundle_records, list):
        raise ChainError("immutable recovery tool bundle is malformed")
    by_git_path = {
        record.get("git_path"): record
        for record in bundle_records
        if isinstance(record, dict)
    }
    for git_path in BUNDLED_TOOL_GIT_PATHS:
        record = by_git_path.get(git_path)
        expected_path = reconstructed.jobs_root / Path(git_path).name
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size", "git_path"}
            or record.get("path") != str(expected_path)
            or _SHA256.fullmatch(str(record.get("sha256", ""))) is None
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] <= 0
        ):
            raise ChainError(f"sealed bundled-tool binding drifted: {git_path}")
        tool = _require_canonical_path(
            expected_path,
            description=f"sealed bundled recovery tool {git_path}",
            kind="file",
        )
        if (
            stat.S_IMODE(tool.stat().st_mode) & 0o222
            or tool.stat().st_size != record["size"]
            or _sha256(tool) != record["sha256"]
        ):
                raise ChainError(
                    f"sealed bundled recovery tool content drifted: {git_path}"
                )
        bundled_tools[tool.name] = tool.read_bytes()
    expected_bundle = [
        {
            "path": str(reconstructed.jobs_root / Path(git_path).name),
            "sha256": _sha256_bytes(bundled_tools[Path(git_path).name]),
            "size": len(bundled_tools[Path(git_path).name]),
            "git_path": git_path,
        }
        for git_path in BUNDLED_TOOL_GIT_PATHS
    ]
    if manifest["recovery_tool_bundle"] != expected_bundle:
        raise ChainError("immutable recovery tool bundle binding drifted")
    verified_r1_protocol = _verify_sealed_r1_protocol_contract(
        reconstructed,
        manifest["r1_protocol_verification"],
        git_identity={
            "release_tag": RELEASE_TAG,
            "git_commit": manifest["release_git_commit"],
            "tag_object": manifest["release_tag_object"],
        },
        bundled_tools=bundled_tools,
    )
    expected_specs = job_specs(
        reconstructed,
        commit=manifest["release_git_commit"],
        tag_object=manifest["release_tag_object"],
        slurm_user=manifest["slurm_user"],
        bundled_tools=bundled_tools,
        prerequisite_evidence=verified_prerequisites,
        r1_protocol_verification=verified_r1_protocol,
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
            "tag_object": manifest["release_tag_object"],
        },
        slurm_user=manifest["slurm_user"],
        bundled_tools=bundled_tools,
        prerequisite_evidence=verified_prerequisites,
        r1_protocol_verification=verified_r1_protocol,
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
        *(spec.filename for spec in expected_specs),
        *bundled_tools,
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
    for record in expected_bundle:
        tool_path = Path(record["path"])
        _require_canonical_path(
            tool_path,
            description="immutable recovery bundled tool",
            kind="file",
        )
        if (
            stat.S_IMODE(tool_path.stat().st_mode) & 0o222
            or tool_path.read_bytes() != bundled_tools[tool_path.name]
            or _sha256(tool_path) != record["sha256"]
        ):
            raise ChainError(f"immutable recovery tool content drifted: {tool_path}")

    fleet_text = (jobs_root / "16_fleet_readiness.sbatch").read_text(encoding="utf-8")
    if "36000" not in fleet_text or "seq 1 144" in fleet_text:
        raise ChainError("fleet readiness does not use the bounded 10-hour deadline")
    if (
        "fleet-capacity-transient" not in fleet_text
        or "--readiness-job-id \"$SLURM_JOB_ID\"" not in fleet_text
        or "--boundary-seconds 36000" not in fleet_text
        or "exit 75" not in fleet_text
        or CAPACITY_TRANSIENT_ROOT_NAME not in fleet_text
        or CAPACITY_TRANSIENT_MARKER_NAME not in fleet_text
    ):
        raise ChainError(
            "fleet readiness lacks the exact generation/job-scoped capacity "
            "transient exit-75 contract"
        )
    smoke_text = (jobs_root / "17_smoke_readiness.sbatch").read_text(encoding="utf-8")
    smoke_apply_offset = smoke_text.find("--rollout-generation \"$next_generation\" --apply")
    smoke_verify_offset = smoke_text.find("verify-smoke-attempt")
    smoke_resolve_offset = smoke_text.find('smoke_verification="$(')
    smoke_attest_offset = smoke_text.find(
        "attest --gate smoke_runs"
    )
    if (
        "generation + 1" not in smoke_text
        or min(
            smoke_apply_offset,
            smoke_verify_offset,
            smoke_resolve_offset,
            smoke_attest_offset,
        )
        < 0
        or not (
            smoke_apply_offset
            < smoke_verify_offset
            < smoke_resolve_offset
            < smoke_attest_offset
        )
        or '--evidence "$smoke_evidence"' not in smoke_text
        or "readiness/smoke_runs.json" in smoke_text
        or "bundled smoke-attempt verifier" not in smoke_text
    ):
        raise ChainError(
            "smoke job does not execute, verify, resolve, and attest the "
            "marker-last generation-scoped CURRENT attempt"
        )
    qualification_text = (
        jobs_root / "18_throughput_qualification.sbatch"
    ).read_text(encoding="utf-8")
    qualification_attestation = list(
        re.finditer(
            (
                rf"--state-dir\s+{re.escape(_q(reconstructed.state))}\s+"
                r"attest\s+--gate\s+throughput_qualification\s+"
                rf"--chain-manifest\s+{re.escape(_q(manifest_path))}\s+"
                rf"--evidence\s+"
                f"{re.escape(_q(reconstructed.throughput_qualification_marker))}"
            ),
            qualification_text,
        )
    )
    qualification_verify_offset = qualification_text.find(
        "verify-throughput-qualification"
    )
    if (
        "run_schema5_throughput_qualification.py" not in qualification_text
        or '"$qualification_producer" execute' not in qualification_text
        or "--timeout-seconds 36000" not in qualification_text
        or qualification_verify_offset < 0
        or len(qualification_attestation) != 1
        or qualification_verify_offset
        >= qualification_attestation[0].start()
    ):
        raise ChainError(
            "throughput qualification does not execute, verify, and attest its "
            "exact chain-bound marker-last full-capacity evidence"
        )
    drill_text = (jobs_root / "19_controller_drill.sbatch").read_text(
        encoding="utf-8"
    )
    watchdog_attestations = list(
        re.finditer(
            (
                r'"\$\{control\[@\]\}"\s+attest\s+--gate\s+'
                r"external_watchdog\s+"
                rf"--chain-manifest\s+{re.escape(_q(manifest_path))}\s+"
                rf"--evidence\s+"
                f"{re.escape(_q(reconstructed.watchdog_ready_marker))}"
            ),
            drill_text,
        )
    )
    watchdog_verifications = [
        match.start()
        for match in re.finditer(
            r"\bverify-watchdog-readiness\b", drill_text
        )
    ]
    early_guard_offset = drill_text.find(
        f"if [[ -f {_q(reconstructed.state / 'CONTROLLER_KILL_DRILL_COMPLETE.json')}"
    )
    early_exit_offset = drill_text.find(
        "  exit 0\nfi\n", early_guard_offset
    )
    reconcile_offset = drill_text.find(
        '"${control[@]}" reconcile --all --no-admit',
        early_guard_offset,
    )
    publisher_offset = drill_text.find(
        f"-I {_q(reconstructed.worktree / 'scripts' / 'publish_schema5_watchdog_ready.py')}",
        reconcile_offset,
    )
    dispatcher_kill_offset = drill_text.find(
        '"${control[@]}" drill kill --role dispatcher'
    )
    if (
        "verify-throughput-qualification" not in drill_text
        or len(watchdog_attestations) != 2
        or len(watchdog_verifications) != 3
        or min(
            early_guard_offset,
            early_exit_offset,
            reconcile_offset,
            publisher_offset,
            dispatcher_kill_offset,
        )
        < 0
        or not (
            early_guard_offset
            < watchdog_verifications[0]
            < watchdog_attestations[0].start()
            < early_exit_offset
            < reconcile_offset
            < publisher_offset
            < watchdog_verifications[1]
            < watchdog_attestations[1].start()
            < dispatcher_kill_offset
            < watchdog_verifications[2]
        )
        or "exec env" in drill_text[
            early_guard_offset : watchdog_attestations[0].end()
        ]
    ):
        raise ChainError(
            "controller drill does not verify and attest the exact chain-bound "
            "external-watchdog evidence on both fresh and idempotent crash paths"
        )
    resume_text = (jobs_root / "20_production_resume.sbatch").read_text(
        encoding="utf-8"
    )
    catalog_offset = resume_text.find(
        '"${control[@]}" refresh-generation-catalog'
    )
    resume_offset = resume_text.find('until "${control[@]}" resume')
    if (
        catalog_offset < 0
        or resume_offset < 0
        or catalog_offset >= resume_offset
        or "verify-throughput-qualification" not in resume_text
    ):
        raise ChainError(
            "production resume does not seal the trusted-generation catalog "
            "before controller admission"
        )
    launch_gate_token = (
        "No recovery-stage mutation may occur above this "
        "launch-authorization gate."
    )
    expected_spec_by_name = {spec.name: spec for spec in expected_specs}
    for record in manifest["jobs"]:
        job_text = Path(record["script"]).read_text(encoding="utf-8")
        if (
            record["name"] == "failure_sentinel"
            or str(record["name"]).startswith(STAGE_SENTINEL_PREFIX)
        ):
            if launch_gate_token in job_text:
                raise ChainError(
                    "failure sentinels must remain able to alert when launch "
                    "marker publication fails"
                )
            continue
        if (
            launch_gate_token not in job_text
            or ROOT_RELEASE_COMPLETE_NAME not in job_text
            or LAUNCH_COMPLETE_NAME not in job_text
            or SUBMISSION_RECEIPT_NAME not in job_text
            or f"SECONDS + {LAUNCH_GATE_TIMEOUT_SECONDS}" not in job_text
            or "launch authorization failed:" not in job_text
            or "current allocation is not the receipt-bound stage" not in job_text
            or "root-release dependency policy is not fail closed" not in job_text
            or job_text.find(launch_gate_token)
            >= job_text.find(
                expected_spec_by_name[record["name"]].body.rstrip()
            )
        ):
            raise ChainError(
                f"production stage lacks the sealed launch-authorization "
                f"gate: {record['name']}"
            )
    checkout_text = (jobs_root / "00_source_checkout.sbatch").read_text(encoding="utf-8")
    bundled_auth_offset = checkout_text.find(
        "bundled prerequisite verifier hash drifted"
    )
    marker_check_offset = checkout_text.find("--markers-only")
    early_exit_offset = checkout_text.find('if [[ -e "$target"')
    clone_offset = checkout_text.find("git clone --no-local --no-checkout")
    target_auth_offset = checkout_text.find(
        "tagged-checkout prerequisite verifier hash drifted"
    )
    target_exec_offset = checkout_text.find(
        '"$target_prerequisite_verifier"',
        target_auth_offset,
    )
    full_check_offset = checkout_text.rfind("verify_full_prerequisites")
    if (
        bundled_auth_offset < 0
        or marker_check_offset < bundled_auth_offset
        or early_exit_offset < marker_check_offset
        or clone_offset < early_exit_offset
        or target_auth_offset < marker_check_offset
        or target_exec_offset < target_auth_offset
        or full_check_offset < clone_offset
        or '[[ -f "${prerequisite_verifier}" && ! -L "${prerequisite_verifier}" ]]'
        not in checkout_text
        or '[[ -f "${target_prerequisite_verifier}" && ! -L "${target_prerequisite_verifier}" ]]'
        not in checkout_text
        or checkout_text.count("tool_mode") < 4
        or checkout_text.count("stat -c '%s'") < 2
        or checkout_text.count("sha256sum --") < 2
        or "git clone --no-local --no-checkout" not in checkout_text
        or "checkout --detach" not in checkout_text
        or SOURCE_CHECKOUT_SEAL_NAME not in checkout_text
        or 'find "$target" -depth ! -type l -exec chmod a-w' not in checkout_text
        or checkout_text.rfind("verify_full_prerequisites")
        > checkout_text.rfind("seal_target")
    ):
        raise ChainError(
            "source checkout does not authenticate and reverify prerequisites "
            "before every exit/mutation"
        )
    for stage_name in (
        "02_snapshot_adopt_verify.sbatch",
        "03_environment_capture.sbatch",
        "04_release_materialize.sbatch",
        "05_release_freeze.sbatch",
    ):
        stage_text = (jobs_root / stage_name).read_text(encoding="utf-8")
        if (
            SOURCE_CHECKOUT_SEAL_NAME not in stage_text
            or "sealed source checkout contains writable state" not in stage_text
            or "diff --no-ext-diff --quiet HEAD --" not in stage_text
            or "diff --cached --no-ext-diff --quiet" not in stage_text
            or "fsck --full --strict" not in stage_text
        ):
            raise ChainError(
                f"pre-freeze stage lacks fresh sealed source authentication: "
                f"{stage_name}"
            )
    materialize_text = (
        jobs_root / "04_release_materialize.sbatch"
    ).read_text(encoding="utf-8")
    conda_tool_auth = materialize_text.find(
        "verified_conda_toolchain_binding"
    )
    first_conda_runtime_gate = materialize_text.find(
        "verify_conda_runtime_toolchain",
        conda_tool_auth,
    )
    if (
        conda_tool_auth < 0
        or first_conda_runtime_gate < conda_tool_auth
        or materialize_text.count("verify_conda_runtime_toolchain") < 3
        or "Conda runtime toolchain differs from sealed pilot provenance"
        not in materialize_text
        or "selected_package_cache_input_binding" not in materialize_text
        or materialize_text.count("verify_package_cache_input") < 3
        or (
            "production selected package-cache input differs from sealed "
            "pilot provenance"
        )
        not in materialize_text
        or "--conda-toolchain-root" not in materialize_text
        or "--source-package-cache" not in materialize_text
        or "--conda-executable" in materialize_text
    ):
        raise ChainError(
            "materialization does not revalidate the complete Conda toolchain "
            "and selected cache input before both invocation boundaries"
        )
    sentinel_text = (jobs_root / "42_failure_sentinel.sbatch").read_text(
        encoding="utf-8"
    )
    sentinel_auth_offset = sentinel_text.find(
        "bundled recovery sentinel hash drifted"
    )
    sentinel_exec_offset = sentinel_text.find(
        'exec env LD_LIBRARY_PATH="$bootstrap_root/lib"'
    )
    if (
        sentinel_auth_offset < 0
        or sentinel_exec_offset < sentinel_auth_offset
        or '[[ -f "${sentinel_tool}" && ! -L "${sentinel_tool}" ]]'
        not in sentinel_text
        or "bundled recovery sentinel size drifted" not in sentinel_text
        or "bundled recovery sentinel is writable" not in sentinel_text
        or "--capacity-transient-receipt \"$capacity_receipt\"" not in sentinel_text
        or CAPACITY_TRANSIENT_ROOT_NAME not in sentinel_text
        or CAPACITY_TRANSIENT_MARKER_NAME not in sentinel_text
    ):
        raise ChainError(
            "failure sentinel does not independently authenticate its bundled tool"
        )
    for index, stage in enumerate(PRODUCTION_STAGE_NAMES):
        observer = f"{STAGE_SENTINEL_PREFIX}{stage}"
        observer_path = (
            jobs_root / f"{21 + index:02d}_stage_sentinel_{stage}.sbatch"
        )
        observer_text = observer_path.read_text(encoding="utf-8")
        if (
            "bundled recovery sentinel hash drifted" not in observer_text
            or "bundled recovery sentinel size drifted" not in observer_text
            or "bundled recovery sentinel is writable" not in observer_text
            or f"--stage-name {shlex.quote(stage)}" not in observer_text
            or '--stage-sentinel-job-id "$SLURM_JOB_ID"' not in observer_text
            or observer not in observer_text
        ):
            raise ChainError(
                f"stage failure sentinel is not independently authenticated: {stage}"
            )
    snapshot_adopt_text = (
        jobs_root / "02_snapshot_adopt_verify.sbatch"
    ).read_text(encoding="utf-8")
    native_auth_offsets = [
        snapshot_adopt_text.find(
            "bundled protocol-aware r1 evidence verifier hash drifted"
        ),
        snapshot_adopt_text.find(
            "bundled native r1 recovery renderer hash drifted"
        ),
        snapshot_adopt_text.find(
            "bundled native v1.2 recovery renderer hash drifted"
        ),
    ]
    native_exec_offset = snapshot_adopt_text.find(
        '"$protocol_verifier"',
        max(native_auth_offsets),
    )
    if (
        any(offset < 0 for offset in native_auth_offsets)
        or native_exec_offset < max(native_auth_offsets)
        or "protocol-aware r1 evidence report differs from sealed contract"
        not in snapshot_adopt_text
        or snapshot_adopt_text.count("stat -c '%s'") < 3
        or snapshot_adopt_text.count("sha256sum --") < 3
    ):
        raise ChainError(
            "snapshot adoption does not authenticate and natively verify r1 history"
        )
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
        "prerequisite_evidence_id": verified_prerequisites["evidence_id"],
        "production_resume_dependencies_verified": True,
    }


def _qualification_prerequisite_binding(
    record: object, *, identity_field: str, description: str
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ChainError(f"{description} manifest binding is missing")
    required = {
        "marker",
        "marker_sha256",
        "marker_size",
        "protocol",
        identity_field,
    }
    if (
        set(record) != required
        or not isinstance(record.get("marker"), str)
        or not Path(str(record["marker"])).is_absolute()
        or _SHA256.fullmatch(str(record.get("marker_sha256", ""))) is None
        or not isinstance(record.get("marker_size"), int)
        or isinstance(record.get("marker_size"), bool)
        or record["marker_size"] <= 0
        or _SHA256.fullmatch(str(record.get(identity_field, ""))) is None
    ):
        raise ChainError(f"{description} manifest binding is malformed")
    return {
        "marker": record["marker"],
        "marker_sha256": record["marker_sha256"],
        identity_field: record[identity_field],
    }


_QUALIFICATION_READINESS_FIELDS = {
    "catalog_id",
    "marker_path",
    "marker_sha256",
    "inventory_sha256",
    "catalog_payload_sha256",
    "allowed_generation_tuple_count",
    "release_fleet_contract_sha256",
    "fleet_contract_sha256",
    "capacity_generation",
    "rollout_generation",
}
_QUALIFICATION_POINTER_FIELDS = {
    "schema_version",
    "protocol",
    "chain_id",
    "attempt_ordinal",
    "attempt_id",
    "attempt_root",
    "run_root",
    "dispatcher_state",
    "readiness_generation",
    "predecessor",
    "additive_retry",
    "created_at",
    "created_timestamp",
    "pointer_id",
}
_QUALIFICATION_CURRENT_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "pointer",
    "pointer_sha256",
    "pointer_id",
    "current_id",
}
_QUALIFICATION_RETRY_FIELDS = {
    "previous_failure_id",
    "serving_profile",
    "additional_replicas",
    "tensor_parallel_size",
    "from_capacity_generation",
    "to_capacity_generation",
    "from_rollout_generation",
    "to_rollout_generation",
    "from_fleet_contract_sha256",
    "to_fleet_contract_sha256",
    "validation",
}
_QUALIFICATION_FAILURE_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "intent_id",
    "attempt",
    "readiness_generation",
    "reason",
    "admission_capacity_certificate",
    "additive_scaling_requirement",
    "scheduler_capacity_mutated",
    "rerun_requirement",
    "failure_drain_intent",
    "cycle_run_roots",
    "refill_reconciliations",
    "failure_id",
}
_QUALIFICATION_LEGACY_FAILURE_FIELDS = (
    _QUALIFICATION_FAILURE_FIELDS
    - {
        "failure_drain_intent",
        "cycle_run_roots",
        "refill_reconciliations",
        "admission_capacity_certificate",
    }
)
_QUALIFICATION_SCALING_FIELDS = {
    "serving_profile",
    "server_pool_root",
    "backlog_fanout_work",
    "live_replicas",
    "backlog_work_per_replica",
    "additional_replicas",
    "tensor_parallel_size",
    "additional_gpus",
    "requirement",
    "capacity_mutated",
}
_QUALIFICATION_SERVING_PROFILES = {
    "0.6B",
    "1.7B",
    "4B",
    "8B",
    "14B",
    "32B",
    "0.6B-long",
    "1.7B-long",
    "4B-long",
    "8B-long",
    "14B-long",
    "32B-long",
}

_SMOKE_POINTER_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "attempt_ordinal",
    "attempt_root",
    "runs_root",
    "execution_root",
    "immutable_sha256",
    "readiness_generation",
    "protected_capacity",
    "predecessor",
    "created_at",
    "created_timestamp",
    "pointer_id",
}
_SMOKE_SELECTOR_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "attempt_pointer",
    "attempt_pointer_sha256",
    "pointer_id",
    "completion_marker",
    "completion_marker_sha256",
    "completion_id",
    "evidence",
    "evidence_sha256",
    "selector_id",
}


def _verify_self_identity(
    value: Mapping[str, Any], field: str, *, description: str
) -> None:
    identity = dict(value)
    observed = identity.pop(field, None)
    if (
        not isinstance(observed, str)
        or _SHA256.fullmatch(observed) is None
        or observed != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError(f"{description} identity is invalid")


def _smoke_assert_sealed_tree(root: Path, *, description: str) -> None:
    root = _require_canonical_path(root, description=description, kind="directory")
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
            or stat.S_IMODE(metadata.st_mode) & 0o222
            or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
        ):
            raise ChainError(f"{description} is not recursively sealed: {path}")


def _smoke_tree_inventory(
    *,
    attempt_root: Path,
    runs_root: Path,
    exclude_attempt_files: frozenset[str],
) -> dict[str, Any]:
    def files(root: Path, *, excluded: frozenset[str]) -> list[dict[str, Any]]:
        return [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(root).as_posix() not in excluded
        ]

    def directories(root: Path) -> list[str]:
        return [
            ".",
            *[
                path.relative_to(root).as_posix()
                for path in sorted(root.rglob("*"))
                if path.is_dir() and not path.is_symlink()
            ],
        ]

    return {
        "attempt_artifacts": files(
            attempt_root, excluded=exclude_attempt_files
        ),
        "attempt_directories": directories(attempt_root),
        "run_artifacts": files(runs_root, excluded=frozenset()),
        "run_directories": directories(runs_root),
    }


def _smoke_catalog_binding(
    catalog: TrustedGenerationCatalog,
) -> dict[str, Any]:
    return {
        "catalog_id": catalog.catalog_id,
        "marker_path": str(catalog.marker_path),
        "marker_sha256": catalog.marker_sha256,
        "inventory_sha256": catalog.inventory_sha256,
        "catalog_payload_sha256": catalog.catalog_sha256,
        "allowed_generation_tuple_count": len(
            catalog.allowed_generation_tuples
        ),
    }


def _read_sealed_smoke_result(path: Path, *, description: str) -> bytes:
    """Read one canonical result inode twice without following a final symlink."""

    canonical = _require_canonical_path(
        path, description=description, kind="file"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise ChainError(f"cannot open {description}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        first_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            first_chunks.append(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            second_chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = canonical.stat(follow_symlinks=False)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    first = b"".join(first_chunks)
    second = b"".join(second_chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
        or first != second
        or any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable_fields
        )
    ):
        raise ChainError(f"{description} is mutable, shared, or unstable")
    return second


def _verify_smoke_result_catalog(
    *,
    suite_artifacts: Mapping[str, tuple[Path, Mapping[str, Any]]],
    results_root: Path,
    attempt_id: str,
    catalog: TrustedGenerationCatalog,
) -> None:
    """Independently compare every retained smoke result row to the sealed catalog."""

    expected_suites = {
        "long_32b_smoke": ("schema5_smoke_32b_long_v1", 15),
        "selective_long_smoke": (
            "schema5_smoke_selective_long_v1",
            20,
        ),
        "standard_canary_smoke": (
            "schema5_smoke_standard_canaries_v1",
            6,
        ),
    }
    if set(suite_artifacts) != set(expected_suites):
        raise ChainError("smoke suite catalog validation has incomplete coverage")
    catalog_binding = _smoke_catalog_binding(catalog)
    total_cells = 0
    selected_prefix: tuple[str, str, int, int] | None = None
    selected_generation_tuples: frozenset[
        tuple[str, str, int, int, str]
    ] | None = None
    for name, (run_id, expected_cells) in expected_suites.items():
        artifact_path, suite = suite_artifacts[name]
        identity = suite.get("suite_identity")
        provenance = suite.get("provenance")
        cells = suite.get("cells")
        attempt = (
            identity.get("smoke_attempt")
            if isinstance(identity, Mapping)
            else None
        )
        attempt_prefix = (
            (
                str(attempt.get("release_fleet_contract_sha256")),
                str(attempt.get("fleet_contract_sha256")),
                int(attempt.get("capacity_generation")),
                int(attempt.get("rollout_generation")),
            )
            if (
                isinstance(attempt, Mapping)
                and all(
                    isinstance(attempt.get(field), int)
                    and not isinstance(attempt[field], bool)
                    and int(attempt[field]) > 0
                    for field in (
                        "capacity_generation",
                        "rollout_generation",
                    )
                )
            )
            else None
        )
        attempt_ordinal = (
            int(attempt["attempt_ordinal"])
            if (
                isinstance(attempt, Mapping)
                and isinstance(attempt.get("attempt_ordinal"), int)
                and not isinstance(attempt["attempt_ordinal"], bool)
                and int(attempt["attempt_ordinal"]) > 0
            )
            else None
        )
        expected_attempt_id = (
            (
                f"a{attempt_ordinal:06d}-"
                f"g{attempt_prefix[3]:06d}-"
                f"c{attempt_prefix[2]:06d}-"
                f"{str(attempt.get('trusted_catalog_id'))[:16]}"
            )
            if attempt_prefix is not None and attempt_ordinal is not None
            else None
        )
        if (
            artifact_path.parent.name != "artifacts"
            or artifact_path.parent.parent.name != attempt_id
            or not isinstance(attempt, Mapping)
            or set(attempt) != SMOKE_ATTEMPT_BINDING_FIELDS
            or attempt.get("protocol") != SMOKE_ATTEMPT_BINDING_PROTOCOL
            or attempt.get("attempt_id") != attempt_id
            or expected_attempt_id != attempt_id
            or _SHA256.fullmatch(
                str(attempt.get("immutable_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(attempt.get("trusted_catalog_id", ""))
            )
            is None
            or attempt.get("trusted_catalog_id") != catalog.catalog_id
            or attempt_prefix is None
            or any(
                _SHA256.fullmatch(str(value)) is None
                for value in attempt_prefix[:2]
            )
            or not isinstance(provenance, Mapping)
            or provenance.get("trusted_generation_catalog") != catalog_binding
            or provenance.get("release_fleet_contract_sha256")
            != attempt_prefix[0]
            or provenance.get("fleet_contract_sha256")
            != attempt_prefix[1]
            or provenance.get("capacity_generation") != attempt_prefix[2]
            or provenance.get("rollout_generation") != attempt_prefix[3]
            or suite.get("trusted_generation_failures") != 0
            or suite.get("run_id") != run_id
            or suite.get("expected_cells") != expected_cells
            or suite.get("schema5_complete_cells") != expected_cells
            or not isinstance(cells, list)
            or len(cells) != expected_cells
        ):
            raise ChainError(
                f"smoke suite {name} lacks exact trusted-generation evidence"
            )
        if selected_prefix is None:
            selected_prefix = attempt_prefix
            selected_generation_tuples = frozenset(
                candidate
                for candidate in catalog.allowed_generation_tuples
                if candidate[:4] == selected_prefix
            )
            if not selected_generation_tuples:
                raise ChainError(
                    "smoke catalog contains no endpoint for its selected "
                    "readiness generation"
                )
        elif attempt_prefix != selected_prefix:
            raise ChainError(
                "smoke suites bind different readiness generations"
            )
        seen: set[str] = set()
        for cell in cells:
            if (
                not isinstance(cell, Mapping)
                or not isinstance(cell.get("cell_id"), str)
                or not cell["cell_id"]
                or cell.get("status") != "complete"
                or not isinstance(cell.get("valid_qids"), int)
                or isinstance(cell.get("valid_qids"), bool)
                or int(cell["valid_qids"]) < 1
                or cell.get("valid_qids") != cell.get("expected_qids")
                or cell.get("trusted_generation_errors") != []
            ):
                raise ChainError(
                    f"smoke suite {name} has an invalid trusted-generation cell"
                )
            cell_id = str(cell["cell_id"])
            seen.add(cell_id)
            result_path = (
                results_root
                / SMOKE_ATTEMPT_RUNS_NAME
                / attempt_id
                / run_id
                / "cells"
                / cell_id
                / "results.jsonl"
            )
            reference = cell.get("results_artifact")
            raw = _read_sealed_smoke_result(
                result_path,
                description=f"smoke suite {name} result artifact",
            )
            if (
                not isinstance(reference, Mapping)
                or set(reference) != {"path", "sha256", "row_count"}
                or reference.get("path") != str(result_path)
                or reference.get("row_count") != cell["valid_qids"]
                or _SHA256.fullmatch(str(reference.get("sha256", ""))) is None
                or hashlib.sha256(raw).hexdigest() != reference["sha256"]
            ):
                raise ChainError(
                    f"smoke suite {name} result-artifact binding is invalid"
                )
            raw_lines = raw.splitlines()
            if len(raw_lines) != int(cell["valid_qids"]) or any(
                not line for line in raw_lines
            ):
                raise ChainError(
                    f"smoke suite {name} result row count is invalid"
                )
            records: list[dict[str, Any]] = []
            for index, line in enumerate(raw_lines):
                def reject_duplicates(
                    pairs: Sequence[tuple[str, Any]],
                ) -> dict[str, Any]:
                    value: dict[str, Any] = {}
                    for key, nested in pairs:
                        if key in value:
                            raise ChainError(
                                f"smoke suite {name} row {index} "
                                f"duplicates key {key!r}"
                            )
                        value[key] = nested
                    return value

                def reject_nonfinite(token: str) -> Any:
                    raise ChainError(
                        f"smoke suite {name} row {index} contains "
                        f"non-finite {token}"
                    )

                try:
                    row = json.loads(
                        line.decode("utf-8"),
                        object_pairs_hook=reject_duplicates,
                        parse_constant=reject_nonfinite,
                    )
                except ChainError:
                    raise
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ChainError(
                        f"smoke suite {name} row {index} is malformed"
                    ) from exc
                if not isinstance(row, dict):
                    raise ChainError(
                        f"smoke suite {name} row {index} is not an object"
                    )
                records.append(row)
            errors = trusted_generation_catalog_errors(
                records, selected_generation_tuples or ()
            )
            if errors:
                raise ChainError(
                    f"smoke suite {name} contains out-of-catalog endpoint "
                    f"provenance: {errors[:3]}"
                )
        if len(seen) != expected_cells:
            raise ChainError(f"smoke suite {name} repeats a cell ID")
        total_cells += len(cells)
    if total_cells != 41:
        raise ChainError("smoke catalog validation does not cover exactly 41 cells")


def verify_smoke_attempt(chain_manifest: Path) -> dict[str, Any]:
    """Verify CURRENT selects the latest immutable 41-cell smoke attempt."""

    verified = verify_chain(chain_manifest)
    manifest_path = Path(str(verified["manifest"]))
    manifest = _read_json(
        manifest_path, description="smoke-bound recovery-chain manifest"
    )
    state_root = Path(str(manifest["state_root"])).resolve()
    results_root = Path(str(manifest["results_root"])).resolve()
    base = (
        Path(str(manifest["readiness_root"])).resolve()
        / SMOKE_ATTEMPT_BASE_NAME
    )
    pointer_root = _require_canonical_path(
        base / "attempt-pointers",
        description="smoke attempt-pointer directory",
        kind="directory",
    )
    pointer_paths = sorted(pointer_root.iterdir())
    if not pointer_paths or any(path.suffix != ".json" for path in pointer_paths):
        raise ChainError("smoke attempt-pointer journal is empty or malformed")
    pointers: list[tuple[Path, dict[str, Any]]] = []
    predecessor: dict[str, str] | None = None
    previous: Mapping[str, Any] | None = None
    for ordinal, path in enumerate(pointer_paths, start=1):
        path = _require_canonical_path(
            path, description=f"smoke attempt pointer {ordinal}", kind="file"
        )
        if stat.S_IMODE(path.stat().st_mode) & 0o222 or path.stat().st_nlink != 1:
            raise ChainError("smoke attempt pointer is not sealed")
        pointer = _read_json(path, description=f"smoke attempt pointer {ordinal}")
        _verify_self_identity(
            pointer, "pointer_id", description="smoke attempt pointer"
        )
        readiness = pointer.get("readiness_generation")
        if (
            not isinstance(readiness, Mapping)
            or any(
                not isinstance(readiness.get(field), int)
                or isinstance(readiness[field], bool)
                or int(readiness[field]) < 1
                for field in ("rollout_generation", "capacity_generation")
            )
            or _SHA256.fullmatch(str(readiness.get("catalog_id", ""))) is None
        ):
            raise ChainError("smoke attempt pointer lacks readiness generation")
        attempt_id = (
            f"a{ordinal:06d}-"
            f"g{int(readiness.get('rollout_generation', -1)):06d}-"
            f"c{int(readiness.get('capacity_generation', -1)):06d}-"
            f"{str(readiness.get('catalog_id', ''))[:16]}"
        )
        attempt_root = base / "attempts" / attempt_id
        runs_root = (
            results_root / SMOKE_ATTEMPT_RUNS_NAME / attempt_id
        )
        expected_path = pointer_root / f"{ordinal:06d}-{attempt_id}.json"
        protected = pointer.get("protected_capacity")
        if (
            set(pointer) != _SMOKE_POINTER_FIELDS
            or pointer.get("schema_version") != 1
            or pointer.get("protocol") != SMOKE_ATTEMPT_POINTER_PROTOCOL
            or pointer.get("attempt_id") != attempt_id
            or pointer.get("attempt_ordinal") != ordinal
            or path != expected_path
            or pointer.get("attempt_root") != str(attempt_root)
            or pointer.get("runs_root") != str(runs_root)
            or pointer.get("execution_root") != str(attempt_root / "execution")
            or pointer.get("predecessor") != predecessor
            or not isinstance(protected, Mapping)
            or protected.get("path")
            != manifest["prerequisite_evidence"]["protected_capacity"]["marker"]
            or protected.get("sha256")
            != manifest["prerequisite_evidence"]["protected_capacity"][
                "marker_sha256"
            ]
            or protected.get("marker_id")
            != manifest["prerequisite_evidence"]["protected_capacity"]["marker_id"]
            or _SHA256.fullmatch(str(pointer.get("immutable_sha256", ""))) is None
        ):
            raise ChainError("smoke attempt pointer identity is invalid")
        complete = attempt_root / SMOKE_ATTEMPT_COMPLETE_NAME
        failure = attempt_root / SMOKE_ATTEMPT_FAILURE_NAME
        if ordinal < len(pointer_paths):
            if complete.exists() == failure.exists():
                raise ChainError("superseded smoke attempt is not uniquely terminal")
            _smoke_assert_sealed_tree(
                attempt_root, description="superseded smoke attempt"
            )
            _smoke_assert_sealed_tree(
                runs_root, description="superseded smoke runs"
            )
        if previous is not None:
            prior_generation = previous["readiness_generation"]
            if readiness == prior_generation:
                prior_failure_path = (
                    Path(str(previous["attempt_root"]))
                    / SMOKE_ATTEMPT_FAILURE_NAME
                )
                prior_failure = _read_json(
                    prior_failure_path,
                    description="smoke retry predecessor failure",
                )
                _verify_self_identity(
                    prior_failure,
                    "failure_id",
                    description="smoke retry predecessor failure",
                )
                prior_pointer_path = pointers[-1][0]
                prior_reference = {
                    "path": str(prior_pointer_path),
                    "sha256": _sha256(prior_pointer_path),
                    "pointer_id": previous["pointer_id"],
                    "attempt_id": previous["attempt_id"],
                }
                if (
                    prior_failure.get("protocol")
                    != SMOKE_ATTEMPT_FAILURE_PROTOCOL
                    or prior_failure.get("retryable") is not True
                    or prior_failure.get("attempt") != prior_reference
                    or prior_failure.get("inventory")
                    != _smoke_tree_inventory(
                        attempt_root=Path(str(previous["attempt_root"])),
                        runs_root=Path(str(previous["runs_root"])),
                        exclude_attempt_files=frozenset(
                            {SMOKE_ATTEMPT_FAILURE_NAME}
                        ),
                    )
                ):
                    raise ChainError(
                        "same-generation smoke retry lacks retryable failure authority"
                    )
            elif (
                int(readiness["capacity_generation"])
                <= int(prior_generation["capacity_generation"])
                or int(readiness["rollout_generation"])
                <= int(prior_generation["rollout_generation"])
                or readiness["catalog_id"] == prior_generation["catalog_id"]
                or readiness["fleet_contract_sha256"]
                == prior_generation["fleet_contract_sha256"]
            ):
                raise ChainError("smoke attempt generations do not strictly advance")
        predecessor = {
            "path": str(path),
            "sha256": _sha256(path),
            "pointer_id": pointer["pointer_id"],
            "attempt_id": pointer["attempt_id"],
        }
        pointers.append((path, pointer))
        previous = pointer

    selector_path = _require_canonical_path(
        base / SMOKE_CURRENT_SELECTOR_NAME,
        description="current smoke selector",
        kind="file",
    )
    if (
        stat.S_IMODE(selector_path.stat().st_mode) & 0o222
        or selector_path.stat().st_nlink != 1
    ):
        raise ChainError("current smoke selector is not sealed")
    selector = _read_json(selector_path, description="current smoke selector")
    _verify_self_identity(
        selector, "selector_id", description="current smoke selector"
    )
    pointer_path, pointer = pointers[-1]
    attempt_root = Path(str(pointer["attempt_root"]))
    runs_root = Path(str(pointer["runs_root"]))
    completion_path = attempt_root / SMOKE_ATTEMPT_COMPLETE_NAME
    evidence_path = attempt_root / SMOKE_EVIDENCE_NAME
    completion = _read_json(
        completion_path, description="smoke attempt completion marker"
    )
    _verify_self_identity(
        completion, "completion_id", description="smoke attempt completion"
    )
    expected_attempt_reference = {
        "path": str(pointer_path),
        "sha256": _sha256(pointer_path),
        "pointer_id": pointer["pointer_id"],
        "attempt_id": pointer["attempt_id"],
    }
    if (
        set(selector) != _SMOKE_SELECTOR_FIELDS
        or selector.get("schema_version") != 1
        or selector.get("protocol") != SMOKE_CURRENT_SELECTOR_PROTOCOL
        or selector.get("attempt_id") != pointer["attempt_id"]
        or selector.get("attempt_pointer") != str(pointer_path)
        or selector.get("attempt_pointer_sha256") != _sha256(pointer_path)
        or selector.get("pointer_id") != pointer["pointer_id"]
        or selector.get("completion_marker") != str(completion_path)
        or selector.get("completion_marker_sha256") != _sha256(completion_path)
        or selector.get("completion_id") != completion.get("completion_id")
        or selector.get("evidence") != str(evidence_path)
        or selector.get("evidence_sha256") != _sha256(evidence_path)
        or completion.get("protocol") != SMOKE_ATTEMPT_COMPLETE_PROTOCOL
        or completion.get("passed") is not True
        or completion.get("attempt") != expected_attempt_reference
        or completion.get("evidence") != str(evidence_path)
        or completion.get("evidence_sha256") != _sha256(evidence_path)
        or completion.get("cells") != 41
        or completion.get("suite_cells")
        != {
            "schema5_smoke_32b_long_v1": 15,
            "schema5_smoke_selective_long_v1": 20,
            "schema5_smoke_standard_canaries_v1": 6,
        }
        or completion.get("inventory")
        != _smoke_tree_inventory(
            attempt_root=attempt_root,
            runs_root=runs_root,
            exclude_attempt_files=frozenset(
                {SMOKE_ATTEMPT_COMPLETE_NAME}
            ),
        )
    ):
        raise ChainError("current smoke selector/completion binding is invalid")
    _smoke_assert_sealed_tree(
        attempt_root, description="successful smoke attempt"
    )
    _smoke_assert_sealed_tree(runs_root, description="successful smoke runs")
    evidence = _read_json(evidence_path, description="selected smoke evidence")
    expected_metrics = {
        "long_32b_cells": 15,
        "selective_long_cells": 20,
        "standard_canary_cells": 6,
        "schema5_complete_cells": 41,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "transport_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
        "trusted_generation_failures": 0,
    }
    artifacts = evidence.get("artifacts")
    if (
        evidence.get("schema_version") != 2
        or evidence.get("gate") != "smoke_runs"
        or evidence.get("passed") is not True
        or evidence.get("immutable_sha256") != pointer["immutable_sha256"]
        or evidence.get("metrics") != expected_metrics
        or not isinstance(artifacts, list)
        or len(artifacts) != 3
    ):
        raise ChainError("selected smoke evidence does not prove exact 41-cell readiness")
    expected_artifacts = {
        "long_32b_smoke",
        "selective_long_smoke",
        "standard_canary_smoke",
    }
    if {
        str(item.get("name")) for item in artifacts if isinstance(item, Mapping)
    } != expected_artifacts:
        raise ChainError("selected smoke evidence has invalid suite coverage")
    suite_artifacts: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for item in artifacts:
        assert isinstance(item, Mapping)
        path = Path(str(item.get("path", "")))
        if (
            path.parent != attempt_root / "artifacts"
            or item.get("sha256") != _sha256(path)
        ):
            raise ChainError("selected smoke suite artifact binding is invalid")
        suite = _read_json(path, description="selected smoke suite evidence")
        suite_artifacts[str(item["name"])] = (path, suite)
    control_path = state_root / "control.json"
    control = _read_json(control_path, description="paused schema-5 control")
    generation = pointer["readiness_generation"]
    provenance = completion.get("provenance")
    immutable = control.get("immutable")
    if (
        not isinstance(immutable, Mapping)
        or control.get("desired_state") != "paused"
        or pointer.get("immutable_sha256") != control.get("immutable_sha256")
        or generation.get("rollout_generation")
        != int(control.get("rollout_generation", -1)) + 1
        or not isinstance(provenance, Mapping)
        or provenance.get("immutable_sha256") != control.get("immutable_sha256")
        or provenance.get("capacity_generation")
        != generation.get("capacity_generation")
        or provenance.get("rollout_generation")
        != generation.get("rollout_generation")
        or provenance.get("fleet_contract_sha256")
        != generation.get("fleet_contract_sha256")
        or provenance.get("release_fleet_contract_sha256")
        != generation.get("release_fleet_contract_sha256")
        or provenance.get("trusted_generation") != generation
        or provenance.get("release_id") != immutable.get("release_id")
        or provenance.get("git_commit") != immutable.get("git_commit")
    ):
        raise ChainError("selected smoke attempt has stale control/fleet provenance")
    trusted_generation = (
        provenance.get("trusted_generation")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(trusted_generation, Mapping)
        or set(trusted_generation)
        != {
            "catalog_id",
            "marker_path",
            "marker_sha256",
            "inventory_sha256",
            "catalog_payload_sha256",
            "allowed_generation_tuple_count",
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
        }
        or trusted_generation != generation
        or not isinstance(immutable, Mapping)
        or not isinstance(immutable.get("server_pool_root"), str)
    ):
        raise ChainError("smoke completion lacks an exact trusted catalog binding")
    try:
        catalog = validate_trusted_generation_catalog(
            Path(str(trusted_generation["marker_path"])),
            server_pool_root=Path(str(immutable["server_pool_root"])).resolve(),
        )
    except (OSError, ValueError, GenerationCatalogError) as exc:
        raise ChainError(
            f"selected smoke trusted-generation catalog is invalid: {exc}"
        ) from exc
    expected_catalog = _smoke_catalog_binding(catalog)
    if (
        expected_catalog
        != {
            field: trusted_generation[field]
            for field in SMOKE_TRUSTED_CATALOG_FIELDS
        }
        or not any(
            identity[:4]
            == (
                generation["release_fleet_contract_sha256"],
                generation["fleet_contract_sha256"],
                generation["capacity_generation"],
                generation["rollout_generation"],
            )
            for identity in catalog.allowed_generation_tuples
        )
    ):
        raise ChainError(
            "selected smoke trusted-generation catalog differs from readiness"
        )
    _verify_smoke_result_catalog(
        suite_artifacts=suite_artifacts,
        results_root=results_root,
        attempt_id=str(pointer["attempt_id"]),
        catalog=catalog,
    )
    for run_id in SMOKE_RUN_IDS:
        fixed = results_root / run_id
        if fixed.exists() or fixed.is_symlink():
            raise ChainError(
                f"legacy fixed smoke root would permit stale reuse: {fixed}"
            )
    return {
        "status": "verified",
        "passed": True,
        "attempt_id": pointer["attempt_id"],
        "pointer_id": pointer["pointer_id"],
        "selector_id": selector["selector_id"],
        "completion_id": completion["completion_id"],
        "evidence": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "capacity_generation": generation["capacity_generation"],
        "rollout_generation": generation["rollout_generation"],
        "trusted_catalog_id": generation["catalog_id"],
    }


def _qualification_attempt_id(
    readiness: Mapping[str, Any],
) -> str:
    return (
        f"g{int(readiness['rollout_generation']):06d}-"
        f"c{int(readiness['capacity_generation']):06d}-"
        f"{readiness['catalog_id']}"
    )


def _qualification_pointer_reference(
    path: Path,
    pointer: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "pointer_id": str(pointer["pointer_id"]),
        "attempt_id": str(pointer["attempt_id"]),
    }


def _qualification_attempt_binding(
    path: Path,
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    readiness = pointer["readiness_generation"]
    return {
        **_qualification_pointer_reference(path, pointer),
        "attempt_root": str(pointer["attempt_root"]),
        "run_root": str(pointer["run_root"]),
        "rollout_generation": int(readiness["rollout_generation"]),
        "capacity_generation": int(readiness["capacity_generation"]),
        "trusted_generation_catalog_id": str(readiness["catalog_id"]),
    }


def _require_recursively_read_only(
    root: Path,
    *,
    description: str,
) -> Path:
    root = _require_canonical_path(
        root,
        description=description,
        kind="directory",
    )
    try:
        descendants = list(root.rglob("*"))
    except OSError as exc:
        raise ChainError(
            f"cannot inventory {description} for recursive sealing: {exc}"
        ) from exc
    for path in [root, *descendants]:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ChainError(
                f"cannot stat {description} member {path}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
            or (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink != 1
            )
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise ChainError(
                f"{description} is not recursively sealed read-only: {path}"
            )
    return root


def _validate_qualification_readiness(
    value: object,
    *,
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _QUALIFICATION_READINESS_FIELDS:
        raise ChainError(f"{description} fields drifted")
    for field in (
        "allowed_generation_tuple_count",
        "capacity_generation",
        "rollout_generation",
    ):
        observed = value.get(field)
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed < 1
        ):
            raise ChainError(f"{description} integer fields are malformed")
    for field in (
        "catalog_id",
        "marker_sha256",
        "inventory_sha256",
        "catalog_payload_sha256",
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
    ):
        if _SHA256.fullmatch(str(value.get(field, ""))) is None:
            raise ChainError(f"{description} hash fields are malformed")
    _manifest_path(
        value.get("marker_path"),
        description=f"{description} marker path",
    )
    return dict(value)


def _load_qualification_attempt_chain(
    *,
    qualification_root: Path,
    results_root: Path,
    chain_id: str,
) -> list[tuple[Path, dict[str, Any]]]:
    pointer_root = _require_canonical_path(
        qualification_root
        / THROUGHPUT_QUALIFICATION_POINTER_DIRECTORY,
        description="throughput-qualification attempt-pointer directory",
        kind="directory",
    )
    try:
        pointer_paths = sorted(pointer_root.iterdir())
    except OSError as exc:
        raise ChainError(
            f"cannot inventory throughput-qualification attempt pointers: {exc}"
        ) from exc
    if not pointer_paths:
        raise ChainError(
            "throughput qualification has no immutable attempt pointers"
        )
    loaded: list[tuple[Path, dict[str, Any]]] = []
    predecessor: dict[str, str] | None = None
    previous_readiness: Mapping[str, Any] | None = None
    for ordinal, path in enumerate(pointer_paths, start=1):
        pointer, _ = _read_sealed_marker(
            path,
            description=(
                f"throughput-qualification attempt pointer {ordinal}"
            ),
        )
        if set(pointer) != _QUALIFICATION_POINTER_FIELDS:
            raise ChainError(
                "throughput-qualification attempt-pointer fields drifted"
            )
        _require_marker_identity(
            pointer,
            identity_field="pointer_id",
            description="throughput-qualification attempt pointer",
        )
        readiness = _validate_qualification_readiness(
            pointer.get("readiness_generation"),
            description=(
                f"throughput-qualification attempt {ordinal} readiness"
            ),
        )
        attempt_id = _qualification_attempt_id(readiness)
        attempt_root = (
            qualification_root
            / THROUGHPUT_QUALIFICATION_ATTEMPT_DIRECTORY
            / attempt_id
        )
        run_root = (
            results_root
            / THROUGHPUT_QUALIFICATION_RUN_DIRECTORY
            / attempt_id
            / THROUGHPUT_QUALIFICATION_ROOT_NAME
        )
        expected_path = (
            pointer_root / f"{ordinal:06d}-{attempt_id}.json"
        )
        timestamp = pointer.get("created_timestamp")
        if (
            pointer.get("schema_version") != 1
            or pointer.get("protocol")
            != THROUGHPUT_QUALIFICATION_ATTEMPT_POINTER_PROTOCOL
            or pointer.get("chain_id") != chain_id
            or pointer.get("attempt_ordinal") != ordinal
            or pointer.get("attempt_id") != attempt_id
            or path != expected_path
            or pointer.get("attempt_root") != str(attempt_root)
            or pointer.get("run_root") != str(run_root)
            or pointer.get("dispatcher_state")
            != str(attempt_root / "dispatcher")
            or pointer.get("predecessor") != predecessor
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or float(timestamp) <= 0
            or pointer.get("created_at")
            != datetime.fromtimestamp(
                float(timestamp), tz=timezone.utc
            ).isoformat()
        ):
            raise ChainError(
                "throughput-qualification attempt-pointer identity is invalid"
            )
        retry = pointer.get("additive_retry")
        if ordinal == 1:
            if retry is not None:
                raise ChainError(
                    "first throughput-qualification attempt cannot be a retry"
                )
        else:
            if (
                not isinstance(retry, Mapping)
                or set(retry) != _QUALIFICATION_RETRY_FIELDS
                or previous_readiness is None
            ):
                raise ChainError(
                    "throughput-qualification additive-retry proof is malformed"
                )
            profile = retry.get("serving_profile")
            expected_tp = 2 if profile == "32B-long" else 1
            if (
                _SHA256.fullmatch(
                    str(retry.get("previous_failure_id", ""))
                )
                is None
                or profile not in _QUALIFICATION_SERVING_PROFILES
                or retry.get("additional_replicas") != 1
                or retry.get("tensor_parallel_size") != expected_tp
                or retry.get("from_capacity_generation")
                != previous_readiness["capacity_generation"]
                or retry.get("to_capacity_generation")
                != readiness["capacity_generation"]
                or retry.get("from_rollout_generation")
                != previous_readiness["rollout_generation"]
                or retry.get("to_rollout_generation")
                != readiness["rollout_generation"]
                or retry.get("from_fleet_contract_sha256")
                != previous_readiness["fleet_contract_sha256"]
                or retry.get("to_fleet_contract_sha256")
                != readiness["fleet_contract_sha256"]
                or retry.get("validation")
                != (
                    "schema5_control._assert_additive_capacity_contract+"
                    "exact-required-profile-delta"
                )
                or readiness["capacity_generation"]
                != previous_readiness["capacity_generation"] + 1
                or readiness["rollout_generation"]
                <= previous_readiness["rollout_generation"]
                or readiness["catalog_id"]
                == previous_readiness["catalog_id"]
                or readiness["release_fleet_contract_sha256"]
                != previous_readiness["release_fleet_contract_sha256"]
            ):
                raise ChainError(
                    "throughput-qualification additive-retry proof does not "
                    "bind an exact successor generation"
                )
        loaded.append((path, pointer))
        predecessor = _qualification_pointer_reference(path, pointer)
        previous_readiness = readiness
    return loaded


def _load_current_qualification_attempt(
    *,
    qualification_root: Path,
    pointers: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[Path, Mapping[str, Any]]:
    current, _ = _read_sealed_marker(
        qualification_root
        / THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_NAME,
        description="current throughput-qualification attempt",
    )
    if set(current) != _QUALIFICATION_CURRENT_FIELDS:
        raise ChainError(
            "current throughput-qualification attempt fields drifted"
        )
    _require_marker_identity(
        current,
        identity_field="current_id",
        description="current throughput-qualification attempt",
    )
    pointer_path, pointer = pointers[-1]
    reference = _qualification_pointer_reference(
        pointer_path,
        pointer,
    )
    expected = {
        "schema_version": 1,
        "protocol": THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL,
        "attempt_id": reference["attempt_id"],
        "pointer": reference["path"],
        "pointer_sha256": reference["sha256"],
        "pointer_id": reference["pointer_id"],
    }
    expected["current_id"] = _sha256_bytes(_canonical_json(expected))
    if current != expected:
        raise ChainError(
            "current throughput-qualification attempt does not select the "
            "latest immutable pointer"
        )
    return pointer_path, pointer


def _verify_superseded_qualification_failure(
    *,
    pointer_path: Path,
    pointer: Mapping[str, Any],
    successor: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = _require_recursively_read_only(
        Path(str(pointer["attempt_root"])),
        description="superseded throughput-qualification attempt",
    )
    _require_recursively_read_only(
        Path(str(pointer["run_root"])),
        description="superseded throughput-qualification run",
    )
    success_path = (
        attempt_root / THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    if success_path.exists() or success_path.is_symlink():
        raise ChainError(
            "a superseded throughput-qualification attempt claims success"
        )
    failure, _ = _read_sealed_marker(
        attempt_root / THROUGHPUT_QUALIFICATION_FAILURE_NAME,
        description="superseded throughput-qualification failure",
    )
    failure_protocol = failure.get("protocol")
    expected_failure_fields = (
        _QUALIFICATION_FAILURE_FIELDS
        if failure_protocol == THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL
        else _QUALIFICATION_LEGACY_FAILURE_FIELDS
    )
    if set(failure) != expected_failure_fields:
        raise ChainError(
            "superseded throughput-qualification failure fields drifted"
        )
    _require_marker_identity(
        failure,
        identity_field="failure_id",
        description="superseded throughput-qualification failure",
    )
    scaling = failure.get("additive_scaling_requirement")
    retry = successor.get("additive_retry")
    if (
        failure.get("schema_version")
        != (
            THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
            if failure_protocol
            == THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL
            else 1
        )
        or failure_protocol
        not in {
            THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL,
            THROUGHPUT_QUALIFICATION_LEGACY_FAILURE_PROTOCOL,
        }
        or failure.get("passed") is not False
        or _SHA256.fullmatch(str(failure.get("intent_id", ""))) is None
        or failure.get("attempt")
        != _qualification_attempt_binding(pointer_path, pointer)
        or failure.get("readiness_generation")
        != pointer["readiness_generation"]
        or failure.get("scheduler_capacity_mutated") is not False
        or not isinstance(failure.get("reason"), str)
        or not str(failure["reason"]).strip()
        or not isinstance(failure.get("rerun_requirement"), str)
        or not isinstance(scaling, Mapping)
        or set(scaling) != _QUALIFICATION_SCALING_FIELDS
        or not isinstance(retry, Mapping)
        or retry.get("previous_failure_id") != failure.get("failure_id")
        or retry.get("serving_profile") != scaling.get("serving_profile")
        or retry.get("additional_replicas")
        != scaling.get("additional_replicas")
        or retry.get("tensor_parallel_size")
        != scaling.get("tensor_parallel_size")
        or scaling.get("capacity_mutated") is not False
    ):
        raise ChainError(
            "superseded throughput-qualification failure does not authorize "
            "its exact additive successor"
        )
    return failure


_THROUGHPUT_MARKER_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "release_id",
    "release_tag",
    "release_git_commit",
    "release_tag_object",
    "chain_namespace",
    "chain_id",
    "manifest",
    "manifest_sha256",
    "protected_capacity",
    "attempt",
    "evidence",
    "cells",
    "qids",
    "unique_design",
    "load_execution",
    "ceilings",
    "health_soak_384_seconds",
    "loaded_384_seconds",
    "loaded_384_useful_qids",
    "loaded_384_observation_count",
    "configured_client_ceiling",
    "certified_saturation_target",
    "certified_saturation_target_cuts",
    "throughput_qids_per_day",
    "throughput_unit",
    "every_stratum_progress",
    "integrity_incidents",
    "transport_censor_incidents",
    "qualification_id",
}
_THROUGHPUT_EVIDENCE_FIELDS = {
    "schema_version",
    "protocol",
    "intent_id",
    "plan_id",
    "run_id",
    "estimand_excluded",
    "primary_analysis_eligible",
    "unique_design",
    "load_execution",
    "load_window_intent",
    "load_window_drain",
    "load_window_end_intent",
    "cycle_run_roots",
    "refill_reconciliations",
    "observations",
    "evaluation",
    "evidence_id",
}
_THROUGHPUT_LOAD_FIELDS = {
    "unit",
    "repeated_coordinates",
    "window_intent_id",
    "window_start_sequence",
    "window_end_sequence",
    "window_start_timestamp",
    "window_end_timestamp",
    "window_duration_seconds",
    "trusted_execution_events",
    "replay_execution_events_total",
    "cycle_count",
    "cycle_inventory",
    "configured_client_ceiling",
    "certified_saturation_target",
    "certified_saturation_target_cuts",
    "work_conserving_refill",
    "sealed_refill_deficit_journal",
    "refill_deficit_scan_count",
    "refill_wall_seconds",
    "rate_denominator_includes_refill_wall_time",
    "minimum_unfinished_assignments",
    "all_strata_progress",
    "stratum_execution_event_deltas",
    "throughput_events_per_day",
}
_THROUGHPUT_CYCLE_FIELDS = {
    "cycle_index",
    "cycle_id",
    "run_id",
    "semantic_reference",
    "status",
    "validated_execution_events",
    "unfinished_assignments",
    "estimand_excluded",
    "primary_analysis_eligible",
}


def _read_bound_qualification_record(
    binding: object,
    *,
    identity_field: str,
    description: str,
) -> tuple[dict[str, Any], Path]:
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", identity_field}
    ):
        raise ChainError(f"{description} binding fields drifted")
    path = _manifest_path(
        binding.get("path"),
        description=f"{description} path",
    )
    value, _ = _read_sealed_marker(path, description=description)
    _require_marker_identity(
        value,
        identity_field=identity_field,
        description=description,
    )
    if (
        binding.get("sha256") != _sha256(path)
        or binding.get(identity_field) != value.get(identity_field)
    ):
        raise ChainError(f"{description} binding hash or identity drifted")
    return value, path


def _qualification_tree_inventory(
    root: Path,
    *,
    description: str,
) -> dict[str, Any]:
    root = _require_recursively_read_only(root, description=description)
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ChainError(f"{description} contains unsafe member: {path}")
        size = metadata.st_size
        total += size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "root": str(root),
        "files": len(files),
        "bytes": total,
        "inventory_sha256": _sha256_bytes(_canonical_json(files)),
    }


def _verify_r7_throughput_evidence(
    *,
    attempt_root: Path,
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the repeated-execution gate from sealed observation cuts."""

    evidence, evidence_path = _read_bound_qualification_record(
        marker.get("evidence"),
        identity_field="evidence_id",
        description="throughput-qualification aggregate evidence",
    )
    if (
        evidence_path != attempt_root / "QUALIFICATION_EVIDENCE.json"
        or set(evidence) != _THROUGHPUT_EVIDENCE_FIELDS
        or evidence.get("schema_version")
        != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
        or evidence.get("protocol")
        != THROUGHPUT_QUALIFICATION_EVIDENCE_PROTOCOL
        or evidence.get("run_id")
        != THROUGHPUT_QUALIFICATION_ROOT_NAME
        or evidence.get("estimand_excluded") is not True
        or evidence.get("primary_analysis_eligible") is not False
        or _SHA256.fullmatch(str(evidence.get("intent_id", ""))) is None
        or _SHA256.fullmatch(str(evidence.get("plan_id", ""))) is None
    ):
        raise ChainError(
            "throughput-qualification aggregate evidence schema drifted"
        )
    unique = evidence.get("unique_design")
    load = evidence.get("load_execution")
    evaluation = evidence.get("evaluation")
    if (
        not isinstance(unique, Mapping)
        or set(unique)
        != {"cells", "qids", "semantic_reference_cycle"}
        or unique.get("cells") != THROUGHPUT_QUALIFICATION_CELLS
        or unique.get("qids") != THROUGHPUT_QUALIFICATION_QIDS
        or _SHA256.fullmatch(
            str(unique.get("semantic_reference_cycle", ""))
        )
        is None
        or not isinstance(load, Mapping)
        or set(load) != _THROUGHPUT_LOAD_FIELDS
        or not isinstance(evaluation, Mapping)
        or evaluation.get("unique_design") != unique
        or evaluation.get("load_execution") != load
    ):
        raise ChainError(
            "throughput qualification does not separate unique design from "
            "load execution"
        )
    observation_bindings = evidence.get("observations")
    if (
        not isinstance(observation_bindings, list)
        or len(observation_bindings)
        < THROUGHPUT_QUALIFICATION_MIN_LOADED_OBSERVATIONS
    ):
        raise ChainError("throughput qualification lacks observation evidence")
    observations: list[dict[str, Any]] = []
    prior_timestamp = -math.inf
    prior_events = -1
    prior_progress: Mapping[str, Any] | None = None
    for sequence, binding in enumerate(observation_bindings):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {
                "sequence",
                "observation",
                "scheduler",
                "semantic",
            }
            or binding.get("sequence") != sequence
        ):
            raise ChainError(
                "throughput observation inventory is not contiguous"
            )
        receipt, receipt_path = _read_bound_qualification_record(
            binding["observation"],
            identity_field="observation_id",
            description=f"throughput observation {sequence}",
        )
        scheduler, scheduler_path = _read_bound_qualification_record(
            binding["scheduler"],
            identity_field="scheduler_id",
            description=f"throughput scheduler evidence {sequence}",
        )
        semantic, semantic_path = _read_bound_qualification_record(
            binding["semantic"],
            identity_field="semantic_id",
            description=f"throughput semantic evidence {sequence}",
        )
        timestamp = receipt.get("captured_timestamp")
        progress = semantic.get("load_strata_progress")
        if (
            receipt.get("schema_version")
            != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
            or receipt.get("protocol")
            != THROUGHPUT_QUALIFICATION_OBSERVATION_PROTOCOL
            or scheduler.get("schema_version")
            != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
            or scheduler.get("protocol")
            != THROUGHPUT_QUALIFICATION_SCHEDULER_PROTOCOL
            or semantic.get("schema_version")
            != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
            or semantic.get("protocol")
            != THROUGHPUT_QUALIFICATION_SEMANTIC_PROTOCOL
            or any(
                record.get("intent_id") != evidence["intent_id"]
                for record in (receipt, scheduler, semantic)
            )
            or any(
                record.get("sequence") != sequence
                for record in (receipt, scheduler, semantic)
            )
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or float(timestamp) <= prior_timestamp
            or scheduler.get("captured_timestamp") != timestamp
            or semantic.get("captured_timestamp") != timestamp
            or receipt.get("ceiling") != scheduler.get("ceiling")
            or receipt.get("scheduler")
            != {
                "path": str(scheduler_path),
                "sha256": _sha256(scheduler_path),
                "scheduler_id": scheduler["scheduler_id"],
            }
            or receipt.get("semantic")
            != {
                "path": str(semantic_path),
                "sha256": _sha256(semantic_path),
                "semantic_id": semantic["semantic_id"],
            }
            or scheduler.get("squeue_complete") is not True
            or scheduler.get("sacct_complete") is not True
            or scheduler.get("errors") != []
            or scheduler.get("qualification_tasks_only") is not True
            or scheduler.get("production_run_ids") != []
            or not isinstance(
                scheduler.get("active_qualification_cells"), int
            )
            or isinstance(
                scheduler.get("active_qualification_cells"), bool
            )
            or not isinstance(
                scheduler.get("unfinished_load_assignments"), int
            )
            or isinstance(
                scheduler.get("unfinished_load_assignments"), bool
            )
            or semantic.get("integrity_incidents") != 0
            or semantic.get("transport_censor_incidents") != 0
            or semantic.get("load_integrity_incidents") != 0
            or semantic.get("load_censor_incidents") != 0
            or not isinstance(
                semantic.get("trusted_qid_execution_events"), int
            )
            or isinstance(
                semantic.get("trusted_qid_execution_events"), bool
            )
            or semantic["trusted_qid_execution_events"] < prior_events
            or not isinstance(progress, Mapping)
            or not progress
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in progress.values()
            )
            or (
                prior_progress is not None
                and (
                    set(progress) != set(prior_progress)
                    or any(
                        int(progress[label])
                        < int(prior_progress[label])
                        for label in progress
                    )
                )
            )
        ):
            raise ChainError(
                f"throughput observation {sequence} is incomplete or drifted"
            )
        prior_timestamp = float(timestamp)
        prior_events = int(
            semantic["trusted_qid_execution_events"]
        )
        prior_progress = progress
        observations.append(
            {
                "receipt": receipt,
                "scheduler": scheduler,
                "semantic": semantic,
                "receipt_path": receipt_path,
            }
        )

    start_sequence = load.get("window_start_sequence")
    end_sequence = load.get("window_end_sequence")
    configured_ceiling = load.get("configured_client_ceiling")
    saturation_target = load.get("certified_saturation_target")
    if (
        load.get("unit") != "trusted_qid_execution_events"
        or load.get("repeated_coordinates") is not True
        or configured_ceiling
        != THROUGHPUT_QUALIFICATION_CEILINGS[-1]
        or not isinstance(saturation_target, int)
        or isinstance(saturation_target, bool)
        or not 0 < saturation_target <= configured_ceiling
        or not isinstance(start_sequence, int)
        or isinstance(start_sequence, bool)
        or not isinstance(end_sequence, int)
        or isinstance(end_sequence, bool)
        or not 0 <= start_sequence < end_sequence < len(observations)
    ):
        raise ChainError("throughput load-window indexes are invalid")
    start = observations[start_sequence]
    end = observations[end_sequence]

    def clean_loaded_cut(observation: Mapping[str, Any]) -> bool:
        scheduler = observation["scheduler"]
        semantic = observation["semantic"]
        return bool(
            scheduler.get("ceiling")
            == configured_ceiling
            and scheduler.get("active_qualification_cells")
            == saturation_target
            and scheduler.get("unfinished_load_assignments")
            >= saturation_target
            and semantic.get("integrity_incidents") == 0
            and semantic.get("transport_censor_incidents") == 0
            and semantic.get("load_integrity_incidents") == 0
            and semantic.get("load_censor_incidents") == 0
        )

    if (
        not clean_loaded_cut(start)
        or any(clean_loaded_cut(row) for row in observations[:start_sequence])
    ):
        raise ChainError(
            "throughput window is not bound to the first signed saturation cut"
        )
    window = observations[start_sequence : end_sequence + 1]
    for prior, current in zip(window, window[1:]):
        gap = (
            float(current["receipt"]["captured_timestamp"])
            - float(prior["receipt"]["captured_timestamp"])
        )
        if (
            gap > THROUGHPUT_QUALIFICATION_MAX_OBSERVATION_GAP_SECONDS
            or not clean_loaded_cut(current)
        ):
            raise ChainError(
                "throughput load window lost cadence or exact saturation"
            )
    start_time = float(start["receipt"]["captured_timestamp"])
    end_time = float(end["receipt"]["captured_timestamp"])
    duration = end_time - start_time
    events = (
        int(end["semantic"]["trusted_qid_execution_events"])
        - int(start["semantic"]["trusted_qid_execution_events"])
    )
    start_progress = start["semantic"]["load_strata_progress"]
    end_progress = end["semantic"]["load_strata_progress"]
    deltas = {
        label: int(end_progress[label]) - int(start_progress[label])
        for label in start_progress
    }
    throughput = math.floor(events * 86_400.0 / duration)
    final = observations[-1]
    final_semantic = final["semantic"]
    final_cycles = final_semantic.get("load_cycle_inventory")
    if (
        duration < THROUGHPUT_QUALIFICATION_HEALTH_SOAK_SECONDS
        or events < THROUGHPUT_QUALIFICATION_MIN_EXECUTION_EVENTS
        or throughput < THROUGHPUT_QUALIFICATION_MIN_QIDS_PER_DAY
        or any(delta <= 0 for delta in deltas.values())
        or load.get("window_intent_id") is None
        or load.get("window_start_timestamp") != start_time
        or load.get("window_end_timestamp") != end_time
        or load.get("window_duration_seconds") != math.floor(duration)
        or load.get("trusted_execution_events") != events
        or load.get("configured_client_ceiling")
        != THROUGHPUT_QUALIFICATION_CEILINGS[-1]
        or load.get("certified_saturation_target_cuts") is not True
        or load.get("work_conserving_refill") is not True
        or load.get("sealed_refill_deficit_journal") is not True
        or not isinstance(load.get("refill_deficit_scan_count"), int)
        or isinstance(load.get("refill_deficit_scan_count"), bool)
        or load["refill_deficit_scan_count"] < 0
        or not isinstance(load.get("refill_wall_seconds"), (int, float))
        or isinstance(load.get("refill_wall_seconds"), bool)
        or not math.isfinite(float(load["refill_wall_seconds"]))
        or float(load["refill_wall_seconds"]) < 0
        or load.get("rate_denominator_includes_refill_wall_time")
        is not True
        or load.get("minimum_unfinished_assignments")
        != saturation_target
        or load.get("all_strata_progress") is not True
        or load.get("stratum_execution_event_deltas") != deltas
        or load.get("throughput_events_per_day") != throughput
        or final["scheduler"].get("active_qualification_cells") != 0
        or final_semantic.get("states")
        != {"complete": THROUGHPUT_QUALIFICATION_CELLS}
        or final_semantic.get("validated_qids")
        != THROUGHPUT_QUALIFICATION_QIDS
        or final_semantic.get("useful_qids")
        != THROUGHPUT_QUALIFICATION_QIDS
        or final_semantic.get("artifact_schema_counts")
        != {"5": THROUGHPUT_QUALIFICATION_QIDS}
        or final_semantic.get("semantic_reference_cycle")
        != unique["semantic_reference_cycle"]
        or not isinstance(final_cycles, list)
        or not final_cycles
        or load.get("cycle_inventory") != final_cycles
        or load.get("cycle_count") != len(final_cycles)
        or load.get("replay_execution_events_total")
        != final_semantic.get("replay_qid_execution_events")
        or any(
            not isinstance(record, Mapping)
            or set(record) != _THROUGHPUT_CYCLE_FIELDS
            or record.get("cycle_index") != index
            or _SHA256.fullmatch(str(record.get("cycle_id", ""))) is None
            or record.get("status")
            not in {"complete", "load_window_drained"}
            or record.get("estimand_excluded") is not True
            or record.get("primary_analysis_eligible") is not False
            for index, record in enumerate(final_cycles)
        )
        or len(
            [
                record
                for record in final_cycles
                if record["semantic_reference"] is True
            ]
        )
        != 1
        or final_cycles[0]["semantic_reference"] is not True
        or final_cycles[0]["cycle_id"]
        != unique["semantic_reference_cycle"]
    ):
        raise ChainError(
            "throughput qualification does not prove its repeated exact-384 "
            "execution contract"
        )
    cycle_roots = evidence.get("cycle_run_roots")
    if (
        not isinstance(cycle_roots, list)
        or len(cycle_roots) != len(final_cycles)
    ):
        raise ChainError("throughput cycle-root inventory is incomplete")
    for cycle, root_binding in zip(final_cycles, cycle_roots):
        if not isinstance(root_binding, Mapping):
            raise ChainError("throughput cycle-root binding is malformed")
        root = Path(str(root_binding.get("root", "")))
        observed_inventory = _qualification_tree_inventory(
            root,
            description=(
                f"throughput load cycle {cycle['cycle_index']} run"
            ),
        )
        if (
            root_binding.get("cycle_index") != cycle["cycle_index"]
            or root_binding.get("cycle_id") != cycle["cycle_id"]
            or root_binding.get("run_id") != cycle["run_id"]
            or root_binding.get("semantic_reference")
            is not cycle["semantic_reference"]
            or root_binding.get("estimand_excluded") is not True
            or root_binding.get("primary_analysis_eligible") is not False
            or {
                field: root_binding.get(field)
                for field in (
                    "root",
                    "files",
                    "bytes",
                    "inventory_sha256",
                )
            }
            != observed_inventory
        ):
            raise ChainError(
                "throughput cycle-root inventory drifted"
            )
    refill_bindings = evidence.get("refill_reconciliations")
    if not isinstance(refill_bindings, list):
        raise ChainError("throughput refill journal inventory is malformed")
    loaded_refills: list[dict[str, Any]] = []
    for index, binding in enumerate(refill_bindings):
        refill, _ = _read_bound_qualification_record(
            (
                {
                    "path": binding.get("path"),
                    "sha256": binding.get("sha256"),
                    "refill_id": binding.get("refill_id"),
                }
                if isinstance(binding, Mapping)
                else binding
            ),
            identity_field="refill_id",
            description=f"throughput refill reconciliation {index}",
        )
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {"refill_index", "path", "sha256", "refill_id"}
            or binding.get("refill_index") != index
            or refill.get("refill_index") != index
            or refill.get("schema_version")
            != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
        ):
            raise ChainError(
                "throughput refill journal binding drifted"
            )
        loaded_refills.append(refill)
    if loaded_refills:
        try:
            from scripts import (
                run_schema5_throughput_qualification as qualification_runtime,
            )
        except (ImportError, OSError) as exc:
            raise ChainError(
                "cannot load the frozen throughput runtime needed to reopen "
                f"refill ledgers: {exc}"
            ) from exc
        try:
            pointer_path = Path(str(marker["attempt"]["path"]))
            pointer = qualification_runtime._read_json(  # noqa: SLF001
                pointer_path,
                description="throughput qualification attempt pointer",
                sealed=True,
            )
            base_context = (
                qualification_runtime.load_qualification_context(
                    Path(str(marker["manifest"])),
                    verify_chain=False,
                )
            )
            attempt_context = (
                qualification_runtime._attempt_context_from_pointer(  # noqa: SLF001
                    base_context,
                    path=pointer_path,
                    pointer=pointer,
                )
            )
            intent = qualification_runtime._read_json(  # noqa: SLF001
                attempt_root / qualification_runtime.INTENT_NAME,
                description="throughput qualification intent",
                sealed=True,
            )
            qualification_runtime._verify_refill_dispatch_ledger_bindings(  # noqa: SLF001
                attempt_context,
                intent=intent,
                require_read_only=True,
            )
        except (
            OSError,
            ValueError,
            qualification_runtime.ThroughputQualificationError,
        ) as exc:
            raise ChainError(
                "throughput refill admissions do not match their final "
                f"accepted ledgers and immutable artifacts: {exc}"
            ) from exc
    window_refills = [
        refill
        for refill in loaded_refills
        if start_time
        <= float(refill.get("captured_timestamp", -1))
        <= end_time
    ]
    deficit_refills = [
        refill
        for refill in window_refills
        if int(refill.get("active_deficit", 0)) > 0
    ]
    refill_wall_seconds = 0.0
    refill_groups: dict[int, list[dict[str, Any]]] = {}
    for refill in window_refills:
        sequence = refill.get("measurement_sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ChainError(
                "throughput refill measurement sequence is malformed"
            )
        refill_groups.setdefault(sequence, []).append(refill)
    for records in refill_groups.values():
        deficits = [
            record
            for record in records
            if int(record.get("active_deficit", 0)) > 0
        ]
        if not deficits:
            continue
        eligible = [
            record
            for record in records
            if record.get("measurement_eligible") is True
        ]
        if not eligible:
            raise ChainError(
                "throughput refill deficit lacks a certified target cut"
            )
        refill_wall_seconds += (
            float(eligible[-1]["captured_timestamp"])
            - float(deficits[0]["captured_timestamp"])
        )
    if (
        load.get("refill_deficit_scan_count")
        != len(deficit_refills)
        or float(load.get("refill_wall_seconds", -1))
        != refill_wall_seconds
    ):
        raise ChainError(
            "throughput refill count or wall-time accounting drifted"
        )
    return {
        "evidence_id": evidence["evidence_id"],
        "unique_design": dict(unique),
        "load_execution": dict(load),
        "duration": math.floor(duration),
        "events": events,
        "observation_count": len(window),
        "throughput": throughput,
    }


def verify_throughput_qualification(
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify the marker-last full-capacity production qualification."""

    manifest_path = _require_canonical_path(
        _lexical_absolute(manifest_path),
        description="throughput-qualification chain manifest",
        kind="file",
    )
    chain_report = verify_chain(manifest_path)
    manifest = _read_json(
        manifest_path, description="throughput-qualification chain manifest"
    )
    prerequisite = manifest.get("prerequisite_evidence")
    if not isinstance(prerequisite, Mapping):
        raise ChainError(
            "throughput qualification lacks launch-prerequisite bindings"
        )
    recovery_root = _manifest_path(
        manifest.get("recovery_root"),
        description="throughput-qualification recovery root",
    )
    results_root = _manifest_path(
        manifest.get("results_root"),
        description="throughput-qualification results root",
    )
    readiness_root = _manifest_path(
        manifest.get("readiness_root"),
        description="throughput-qualification readiness root",
    )
    qualification_root = (
        readiness_root / THROUGHPUT_QUALIFICATION_ROOT_NAME
    )
    expected_marker = (
        qualification_root
        / THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    marker, marker_raw = _read_sealed_marker(
        expected_marker,
        description="throughput-qualification completion marker",
    )
    pointers = _load_qualification_attempt_chain(
        qualification_root=qualification_root,
        results_root=results_root,
        chain_id=str(manifest.get("chain_id", "")),
    )
    pointer_path, pointer = _load_current_qualification_attempt(
        qualification_root=qualification_root,
        pointers=pointers,
    )
    for index, (prior_path, prior_pointer) in enumerate(
        pointers[:-1]
    ):
        _verify_superseded_qualification_failure(
            pointer_path=prior_path,
            pointer=prior_pointer,
            successor=pointers[index + 1][1],
        )
    attempt_root = _require_recursively_read_only(
        Path(str(pointer["attempt_root"])),
        description="current throughput-qualification attempt",
    )
    _require_recursively_read_only(
        Path(str(pointer["run_root"])),
        description="current throughput-qualification run",
    )
    failure_path = (
        attempt_root / THROUGHPUT_QUALIFICATION_FAILURE_NAME
    )
    if failure_path.exists() or failure_path.is_symlink():
        raise ChainError(
            "latest throughput-qualification attempt is terminally failed"
        )
    attempt_marker_path = (
        attempt_root / THROUGHPUT_QUALIFICATION_MARKER_NAME
    )
    attempt_marker, attempt_marker_raw = _read_sealed_marker(
        attempt_marker_path,
        description="current-attempt throughput-qualification marker",
    )
    if (
        attempt_marker_raw != marker_raw
        or attempt_marker != marker
    ):
        raise ChainError(
            "current-attempt and fixed-root throughput-qualification "
            "markers are not byte-identical"
        )
    attempt_binding = _qualification_attempt_binding(
        pointer_path,
        pointer,
    )
    if set(marker) != _THROUGHPUT_MARKER_FIELDS:
        raise ChainError("throughput-qualification marker fields drifted")
    _require_release_marker_binding(
        marker,
        git_identity={
            "git_commit": str(manifest.get("release_git_commit", "")),
            "tag_object": str(manifest.get("release_tag_object", "")),
        },
        protocol=THROUGHPUT_QUALIFICATION_PROTOCOL,
        description="throughput-qualification marker",
        schema_version=THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION,
    )
    _require_marker_identity(
        marker,
        identity_field="qualification_id",
        description="throughput-qualification marker",
    )
    protected = _qualification_prerequisite_binding(
        prerequisite.get("protected_capacity"),
        identity_field="marker_id",
        description="protected capacity",
    )
    if marker.get("attempt") != attempt_binding:
        raise ChainError(
            "throughput-qualification marker does not bind the exact "
            "current attempt"
        )
    evidence_report = _verify_r7_throughput_evidence(
        attempt_root=attempt_root,
        marker=marker,
    )
    unique = evidence_report["unique_design"]
    load = evidence_report["load_execution"]
    if (
        recovery_root
        != _manifest_path(
            prerequisite["protected_capacity"].get("marker"),
            description="protected-capacity marker binding",
        ).parent
        or marker.get("chain_id") != manifest.get("chain_id")
        or marker.get("manifest") != str(manifest_path)
        or marker.get("manifest_sha256") != _sha256(manifest_path)
        or marker.get("protected_capacity") != protected
        or marker.get("cells") != THROUGHPUT_QUALIFICATION_CELLS
        or marker.get("qids") != THROUGHPUT_QUALIFICATION_QIDS
        or marker.get("ceilings")
        != list(THROUGHPUT_QUALIFICATION_CEILINGS)
        or marker.get("unique_design") != unique
        or marker.get("load_execution") != load
        or marker.get("health_soak_384_seconds")
        != evidence_report["duration"]
        or marker.get("loaded_384_seconds")
        != evidence_report["duration"]
        or marker.get("loaded_384_useful_qids")
        != evidence_report["events"]
        or marker.get("loaded_384_observation_count")
        != evidence_report["observation_count"]
        or marker.get("configured_client_ceiling")
        != THROUGHPUT_QUALIFICATION_CEILINGS[-1]
        or marker.get("certified_saturation_target")
        != load.get("certified_saturation_target")
        or marker.get("certified_saturation_target_cuts") is not True
        or marker.get("throughput_qids_per_day")
        != evidence_report["throughput"]
        or marker.get("throughput_unit")
        != "trusted_qid_execution_events"
        or marker.get("every_stratum_progress") is not True
        or marker.get("integrity_incidents") != 0
        or marker.get("transport_censor_incidents") != 0
    ):
        raise ChainError(
            "throughput qualification does not prove the exact 768-cell, "
            "15,360-QID unique design and repeated full-384 execution contract"
        )
    return {
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(manifest_path),
        "qualification_marker": str(expected_marker),
        "attempt_marker": str(attempt_marker_path),
        "attempt_id": pointer["attempt_id"],
        "attempt_pointer": str(pointer_path),
        "qualification_id": marker["qualification_id"],
        "cells": marker["cells"],
        "qids": marker["qids"],
        "health_soak_384_seconds": marker[
            "health_soak_384_seconds"
        ],
        "loaded_384_seconds": marker["loaded_384_seconds"],
        "loaded_384_useful_qids": marker[
            "loaded_384_useful_qids"
        ],
        "loaded_384_observation_count": marker[
            "loaded_384_observation_count"
        ],
        "configured_client_ceiling": marker[
            "configured_client_ceiling"
        ],
        "certified_saturation_target": marker[
            "certified_saturation_target"
        ],
        "certified_saturation_target_cuts": marker[
            "certified_saturation_target_cuts"
        ],
        "unique_design": marker["unique_design"],
        "load_execution": marker["load_execution"],
        "throughput_unit": marker["throughput_unit"],
        "throughput_qids_per_day": marker["throughput_qids_per_day"],
        "evidence_id": evidence_report["evidence_id"],
        "chain_verification": chain_report,
    }


def submission_argv(
    record: Mapping[str, Any],
    *,
    dependency_job_ids: Sequence[str],
    comment: str,
    initial_hold: bool = False,
) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9:._-]{1,256}", comment):
        raise ChainError(f"unsafe Slurm recovery-chain comment: {comment!r}")
    if any(not str(job_id).isdigit() for job_id in dependency_job_ids):
        raise ChainError("dependency job IDs must be numeric")
    argv = [
        "sbatch",
        "--parsable",
        "--no-requeue",
    ]
    if initial_hold:
        argv.append("--hold")
    argv.append(f"--comment={comment}")
    if dependency_job_ids:
        dependency_type = record.get("dependency_type")
        if dependency_type not in {"afterok", "afterany"}:
            raise ChainError(
                f"unsupported recovery dependency type: {dependency_type!r}"
            )
        argv.append(
            f"--dependency={dependency_type}:" + ":".join(dependency_job_ids)
        )
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
_TRANSIENT_REPAIR_ROOT_STATES = {
    "BOOT_FAIL",
    "NODE_FAIL",
    "PREEMPTED",
    "REVOKED",
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

    receipt_jobs = receipt.get("jobs")
    manifest_jobs = manifest.get("jobs")
    if (
        not isinstance(receipt_jobs, list)
        or not isinstance(manifest_jobs, list)
        or len(receipt_jobs) != len(manifest_jobs)
        or any(not isinstance(row, Mapping) for row in receipt_jobs)
        or [str(row.get("name", "")) for row in receipt_jobs]
        != [str(row.get("name", "")) for row in manifest_jobs]
    ):
        raise ChainError(
            "receipt job topology differs from the immutable recovery DAG"
        )
    receipt_by_id = {
        str(row["job_id"]): row for row in receipt_jobs
    }
    manifest_by_name = {
        str(row["name"]): row for row in manifest_jobs
    }
    if (
        len(receipt_by_id) != len(receipt_jobs)
        or any(not job_id.isdigit() for job_id in receipt_by_id)
    ):
        raise ChainError("receipt scheduler job IDs are not unique numeric IDs")
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
            "submit_line": fields[5].strip(),
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
            "submit_line": (
                None if history is None else history.get("submit_line")
            ),
        }
    return result


def _bootstrap_receipt_lineage(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[tuple[Path, Mapping[str, Any]]]:
    """Return the exact root-to-leaf receipt lineage for one bootstrap cut."""

    reversed_lineage: list[tuple[Path, Mapping[str, Any]]] = []
    current = receipt
    current_path = receipt_path
    seen: set[Path] = set()
    protocol = current.get("protocol")
    if protocol == "schema5-v1.2-r7-recovery-chain-submission":
        current_generation = 0
    elif protocol == "schema5-v1.2-r7-recovery-chain-repair":
        current_generation = current.get("repair_generation")
        if (
            not isinstance(current_generation, int)
            or isinstance(current_generation, bool)
            or current_generation <= 0
        ):
            raise ChainError(
                "bootstrap repair receipt lineage is malformed"
            )
    else:
        raise ChainError("bootstrap receipt lineage protocol is unknown")
    recovery_root = manifest_path.parent
    while True:
        current_path = _require_canonical_path(
            current_path,
            description="bootstrap receipt lineage member",
            kind="file",
        )
        expected_path = (
            recovery_root / SUBMISSION_RECEIPT_NAME
            if current_generation == 0
            else (
                recovery_root
                / REPAIR_ROOT_NAME
                / f"g{current_generation:04d}"
                / SUBMISSION_RECEIPT_NAME
            )
        )
        if current_path != expected_path:
            raise ChainError(
                "bootstrap receipt lineage member is outside its exact "
                f"generation path: g{current_generation:04d}"
            )
        if current_path in seen:
            raise ChainError("bootstrap receipt lineage contains a cycle")
        seen.add(current_path)
        if current_generation == 0:
            validated = _validate_submission_receipt(
                current_path,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=_submission_comments(manifest),
            )
            if current != validated:
                raise ChainError(
                    "bootstrap anchor receipt differs from its sealed bytes"
                )
            reversed_lineage.append((current_path, validated))
            break

        expected_parent_path = (
            recovery_root / SUBMISSION_RECEIPT_NAME
            if current_generation == 1
            else (
                recovery_root
                / REPAIR_ROOT_NAME
                / f"g{current_generation - 1:04d}"
                / SUBMISSION_RECEIPT_NAME
            )
        )
        parent_value = current.get("parent_receipt")
        if parent_value != str(expected_parent_path):
            raise ChainError(
                "bootstrap repair receipt skips or reorders its exact "
                "generation ancestry"
            )
        parent_path = _require_canonical_path(
            expected_parent_path,
            description="bootstrap parent receipt",
            kind="file",
        )
        validated = _validate_repair_receipt(
            current_path,
            manifest=manifest,
            manifest_path=manifest_path,
            generation=current_generation,
            parent_path=parent_path,
        )
        if current != validated:
            raise ChainError(
                "bootstrap repair receipt differs from its sealed bytes"
            )
        reversed_lineage.append((current_path, validated))
        current = _read_json(
            parent_path, description="bootstrap parent receipt"
        )
        current_path = parent_path
        current_generation -= 1
    return list(reversed(reversed_lineage))


def _validate_bootstrap_scheduler_namespace(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    runner: Runner,
) -> dict[str, Any]:
    """Join complete user-wide scheduler truth and reject unbound chain jobs.

    Exact-ID receipt queries are insufficient here: a duplicate job carrying a
    valid recovery comment but a foreign job ID would otherwise be invisible.
    The scan therefore reads the complete user namespace from both ``squeue`` and
    ``sacct`` and accepts only comment/job-ID pairs present in the sealed receipt
    ancestry.
    """

    lineage = _bootstrap_receipt_lineage(
        receipt=receipt,
        receipt_path=receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    expected_by_comment: dict[str, str] = {}
    expected_by_id: dict[str, str] = {}
    expected_name_by_id: dict[str, str] = {}
    manifest_by_name = {
        str(row["name"]): row
        for row in manifest.get("jobs", [])
        if isinstance(row, Mapping)
    }
    for _path, lineage_receipt in lineage:
        for row in lineage_receipt.get("jobs", []):
            if not isinstance(row, Mapping):
                raise ChainError("bootstrap receipt lineage job is malformed")
            comment = str(row.get("comment", ""))
            job_id = str(row.get("job_id", ""))
            name = str(row.get("name", ""))
            manifest_row = manifest_by_name.get(name)
            if (
                not job_id.isdigit()
                or not comment
                or not isinstance(manifest_row, Mapping)
                or not isinstance(manifest_row.get("job_name"), str)
            ):
                raise ChainError("bootstrap receipt lineage identity is invalid")
            previous_id = expected_by_comment.get(comment)
            previous_comment = expected_by_id.get(job_id)
            previous_name = expected_name_by_id.get(job_id)
            if (
                previous_id not in {None, job_id}
                or previous_comment not in {None, comment}
                or previous_name
                not in {None, str(manifest_row["job_name"])}
            ):
                raise ChainError(
                    "bootstrap receipt lineage scheduler identity is ambiguous"
                )
            expected_by_comment[comment] = job_id
            expected_by_id[job_id] = comment
            expected_name_by_id[job_id] = str(
                manifest_row["job_name"]
            )

    anchor_journal = _read_json(
        manifest_path.parent / SUBMISSION_JOURNAL_NAME,
        description="bootstrap anchor submission journal",
    )
    scheduler_since = anchor_journal.get("scheduler_since")
    if not isinstance(scheduler_since, str) or not scheduler_since:
        raise ChainError("bootstrap anchor scheduler window is unavailable")
    user = str(manifest["slurm_user"])
    commands = (
        [
            "squeue",
            "-u",
            user,
            "-h",
            "-o",
            "%i|%T|%k|%j",
        ],
        [
            "sacct",
            "-u",
            user,
            "-X",
            "-n",
            "-P",
            "-S",
            scheduler_since,
            "--format=JobIDRaw,State,ExitCode,Comment%256,JobName%64,SubmitLine",
        ],
    )
    observed: dict[str, dict[str, str]] = {}
    prefix = (
        f"asys:s5-recovery-v1.2-r7:{manifest['chain_id']}:g"
    )
    for argv in commands:
        process = runner(argv)
        if process.returncode != 0:
            raise ChainError(
                "bootstrap namespace scan requires complete "
                f"{argv[0]} truth: {process.stderr.strip()[:500]}"
            )
        source = argv[0]
        for raw in process.stdout.splitlines():
            fields = raw.rstrip("\n").split("|")
            minimum = 6 if source == "sacct" else 4
            if len(fields) < minimum:
                if raw.strip():
                    raise ChainError(
                        "bootstrap namespace scan received malformed "
                        f"{source} row: {raw[:300]!r}"
                    )
                continue
            if source == "squeue":
                job_id, state, comment, job_name = (
                    field.strip() for field in fields[:4]
                )
            else:
                job_id, state, _exit_code, stored, job_name = (
                    field.strip() for field in fields[:5]
                )
                if "." in job_id:
                    continue
                comment = _accounting_comment(
                    stored, fields[5], job_id=job_id
                )
            if not comment.startswith(prefix):
                continue
            if (
                not job_id.isdigit()
                or expected_by_comment.get(comment) != job_id
                or expected_by_id.get(job_id) != comment
                or expected_name_by_id.get(job_id) != job_name
            ):
                raise ChainError(
                    f"foreign {source} job collides with the bootstrap "
                    f"recovery namespace: {job_id}|{comment}|{job_name}"
                )
            key = f"{source}:{job_id}"
            if key in observed:
                raise ChainError(
                    "bootstrap namespace scan found a duplicate "
                    f"{source} row for {job_id}"
                )
            observed[key] = {
                "source": source,
                "job_id": job_id,
                "comment": comment,
                "job_name": job_name,
                "state": _normalize_slurm_state(state),
            }
    return {
        "complete": True,
        "lineage_receipt_ids": [
            str(lineage_receipt["receipt_id"])
            for _path, lineage_receipt in lineage
        ],
        "lineage_job_ids": [
            [str(row["job_id"]) for row in lineage_receipt["jobs"]]
            for _path, lineage_receipt in lineage
        ],
        "bound_job_ids": sorted(expected_by_id, key=int),
        "observed_namespace_rows": sorted(
            observed.values(),
            key=lambda row: (int(row["job_id"]), row["source"]),
        ),
    }


def _dependency_config_allows_fail_closed(
    runner: Runner, *, checked_at: float | None = None
) -> dict[str, Any]:
    proc = runner(["scontrol", "show", "config"])
    if proc.returncode != 0:
        raise ChainError(f"cannot read Slurm dependency policy: {proc.stderr[:500]}")
    match = re.search(r"^DependencyParameters\s*=\s*(.*?)\s*$", proc.stdout, re.M)
    values = sorted(
        value
        for value in (
            re.split(r"[,:\s]+", match.group(1).strip()) if match else []
        )
        if value
    )
    if "kill_invalid_depend" not in values:
        raise ChainError(
            "Slurm must enable DependencyParameters=kill_invalid_depend so a failed "
            "ancestor cannot leave held recovery successors indefinitely"
        )
    timestamp = time.time() if checked_at is None else float(checked_at)
    if not math.isfinite(timestamp):
        raise ChainError("dependency-policy check timestamp must be finite")
    return {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r7-live-dependency-policy-v1",
        "checked_at": timestamp,
        "argv": ["scontrol", "show", "config"],
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        "dependency_parameters": values,
        "kill_invalid_depend": True,
    }


def _validate_dependency_policy_check(
    path: Path, *, expected_phase: str | None = None
) -> dict[str, Any]:
    path = _require_canonical_path(
        path, description="dependency-policy evidence", kind="file"
    )
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError("dependency-policy evidence must be read-only")
    payload = _read_json(path, description="dependency-policy evidence")
    expected = {
        "schema_version",
        "protocol",
        "phase",
        "checked_at",
        "argv",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
        "dependency_parameters",
        "kill_invalid_depend",
        "evidence_id",
    }
    identity = dict(payload)
    evidence_id = identity.pop("evidence_id", None)
    timestamp = payload.get("checked_at")
    if (
        set(payload) != expected
        or payload.get("schema_version") != 1
        or payload.get("protocol")
        != "schema5-v1.2-r7-live-dependency-policy-v1"
        or not isinstance(payload.get("phase"), str)
        or not payload["phase"]
        or (
            expected_phase is not None
            and payload.get("phase") != expected_phase
        )
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or payload.get("argv") != ["scontrol", "show", "config"]
        or payload.get("returncode") != 0
        or not isinstance(payload.get("stdout"), str)
        or not isinstance(payload.get("stderr"), str)
        or payload.get("stdout_sha256")
        != _sha256_bytes(str(payload.get("stdout", "")).encode("utf-8"))
        or payload.get("stderr_sha256")
        != _sha256_bytes(str(payload.get("stderr", "")).encode("utf-8"))
        or not isinstance(payload.get("dependency_parameters"), list)
        or any(
            not isinstance(value, str)
            for value in payload.get("dependency_parameters", [])
        )
        or "kill_invalid_depend"
        not in payload.get("dependency_parameters", [])
        or payload.get("kill_invalid_depend") is not True
        or not isinstance(evidence_id, str)
        or _SHA256.fullmatch(evidence_id) is None
        or evidence_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("dependency-policy evidence is invalid")
    return payload


def _persist_dependency_policy_check(
    recovery_root: Path,
    *,
    phase: str,
    runner: Runner,
    checked_at: float,
) -> tuple[dict[str, Any], Path]:
    if not _SAFE_NAME.fullmatch(phase):
        raise ChainError(f"unsafe dependency-policy phase: {phase!r}")
    root = recovery_root / DEPENDENCY_POLICY_CHECKS_ROOT_NAME
    if root.is_symlink():
        raise ChainError("dependency-policy evidence root is symlinked")
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("check-*.json"))
    if any(
        path.name != f"check-{index:04d}.json"
        for index, path in enumerate(existing, start=1)
    ):
        raise ChainError("dependency-policy checks are not contiguous")
    path = root / f"check-{len(existing) + 1:04d}.json"
    payload = _dependency_config_allows_fail_closed(
        runner, checked_at=checked_at
    )
    payload["phase"] = phase
    payload["evidence_id"] = _sha256_bytes(_canonical_json(payload))
    _atomic_json(path, payload, mode=0o444)
    return _validate_dependency_policy_check(
        path, expected_phase=phase
    ), path


def _dependency_canary_binding(
    manifest: Mapping[str, Any]
) -> dict[str, Any]:
    prerequisite = manifest.get("prerequisite_evidence")
    canary = (
        prerequisite.get("slurm_canary")
        if isinstance(prerequisite, Mapping)
        else None
    )
    if (
        not isinstance(canary, Mapping)
        or not isinstance(canary.get("dependency_canary_id"), str)
        or _SHA256.fullmatch(canary["dependency_canary_id"]) is None
        or canary.get("dependency_kill_invalid_depend") is not True
        or canary.get("dependency_root_initial_hold") is not True
        or canary.get("dependency_child_never_started") is not True
        or canary.get("dependency_alert_latency_bound_seconds") != 180.0
        or not isinstance(
            canary.get("dependency_alert_latency_seconds"), (int, float)
        )
        or isinstance(canary.get("dependency_alert_latency_seconds"), bool)
        or canary.get("dependency_alert_latency_seconds", float("inf"))
        < 0
        or canary.get("dependency_alert_latency_seconds", float("inf"))
        > 180.0
    ):
        raise ChainError(
            "manifest lacks the sealed bounded dependency-cascade proof"
        )
    return {
        "canary_id": canary["dependency_canary_id"],
        "marker_sha256": canary["dependency_marker_sha256"],
        "alert_latency_seconds": canary[
            "dependency_alert_latency_seconds"
        ],
        "alert_latency_bound_seconds": canary[
            "dependency_alert_latency_bound_seconds"
        ],
        "kill_invalid_depend": True,
        "root_initial_hold": True,
        "child_never_started": True,
    }


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
        "dependency_policy_check",
        "dependency_policy_check_sha256",
        "dependency_canary",
        "jobs",
    }
    if set(journal) != expected_fields:
        raise ChainError("submission journal fields drifted")
    started_timestamp = journal["started_timestamp"]
    if (
        journal["schema_version"] != SUBMISSION_SCHEMA_VERSION
        or journal["protocol"] != "schema5-v1.2-r7-recovery-chain-submission"
        or journal["chain_id"] != manifest["chain_id"]
        or journal["manifest"] != str(manifest_path)
        or journal["slurm_user"] != manifest["slurm_user"]
        or not isinstance(journal["started_at"], str)
        or not isinstance(started_timestamp, (int, float))
        or isinstance(started_timestamp, bool)
        or not math.isfinite(float(started_timestamp))
        or not isinstance(journal["scheduler_since"], str)
        or journal["scheduler_since"] != _slurm_timestamp(float(started_timestamp))
        or not isinstance(journal["dependency_policy_check"], str)
        or not isinstance(journal["dependency_policy_check_sha256"], str)
        or _SHA256.fullmatch(journal["dependency_policy_check_sha256"])
        is None
        or journal["dependency_canary"]
        != _dependency_canary_binding(manifest)
        or not isinstance(journal["jobs"], dict)
    ):
        raise ChainError("submission journal identity is invalid")
    policy_path = _require_canonical_path(
        Path(journal["dependency_policy_check"]),
        description="submission dependency-policy evidence",
        kind="file",
    )
    if (
        policy_path.parent
        != manifest_path.parent / DEPENDENCY_POLICY_CHECKS_ROOT_NAME
        or _sha256(policy_path) != journal["dependency_policy_check_sha256"]
    ):
        raise ChainError("submission dependency-policy evidence binding drifted")
    _validate_dependency_policy_check(
        policy_path, expected_phase="pre_submission"
    )

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
            initial_hold=name == "source_checkout",
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
        "dependency_policy_check",
        "dependency_policy_check_sha256",
        "dependency_canary",
        "root_initial_hold",
        "no_requeue",
        "stage_failure_sentinels",
        "jobs",
        "receipt_id",
    }
    if set(receipt) != expected_fields:
        raise ChainError("chain submission receipt fields drifted")
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id")
    if (
        receipt["schema_version"] != SUBMISSION_SCHEMA_VERSION
        or receipt["protocol"] != "schema5-v1.2-r7-recovery-chain-submission"
        or receipt["passed"] is not True
        or receipt["chain_id"] != manifest["chain_id"]
        or receipt["manifest"] != str(manifest_path)
        or receipt["manifest_sha256"] != _sha256(manifest_path)
        or receipt["submission_journal"] != str(journal_path)
        or receipt["submission_journal_sha256"] != _sha256(journal_path)
        or not isinstance(receipt["submitted_at"], str)
        or receipt["dependency_policy"] != DEPENDENCY_POLICY_CONTRACT
        or receipt["dependency_policy_check"]
        != journal["dependency_policy_check"]
        or receipt["dependency_policy_check_sha256"]
        != journal["dependency_policy_check_sha256"]
        or receipt["dependency_canary"]
        != _dependency_canary_binding(manifest)
        or receipt["stage_failure_sentinels"]
        != manifest["stage_failure_sentinels"]
        or receipt["root_initial_hold"] is not True
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
        "dependency_type",
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
            or record["dependency_type"] != row["dependency_type"]
            or journal["jobs"].get(row["name"], {}).get("job_id") != job_id
        ):
            raise ChainError(f"chain submission receipt job drifted: {row['name']}")
        observed_ids.add(job_id)
        submitted[row["name"]] = job_id
    return receipt


def _write_immutable_json_once(
    path: Path, payload: Mapping[str, Any], *, description: str
) -> None:
    encoded = _canonical_json(payload)
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != encoded
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise ChainError(f"existing {description} drifted: {path}")
        return
    _atomic_json(path, payload, mode=0o444)


def _show_recovery_job(job_id: str, *, runner: Runner) -> dict[str, Any]:
    if not job_id.isdigit():
        raise ChainError("recovery root job ID must be numeric")
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


def _scheduler_dependency_pairs(value: str) -> list[tuple[str, str]]:
    """Normalize Slurm's dependency display without discarding its exact IDs."""

    normalized = value.strip()
    if normalized.lower() in {"", "(null)", "null", "none", "n/a"}:
        return []
    if "?" in normalized:
        raise ChainError(
            f"scheduler dependency unexpectedly uses OR semantics: {value!r}"
        )
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    result: list[tuple[str, str]] = []
    for group in normalized.split(","):
        pieces = [piece.strip() for piece in group.split(":") if piece.strip()]
        if len(pieces) < 2 or pieces[0] not in {"afterok", "afterany"}:
            raise ChainError(f"unsupported scheduler dependency value: {value!r}")
        dependency_type = pieces[0]
        for job_id in pieces[1:]:
            if not job_id.isdigit():
                raise ChainError(
                    f"scheduler dependency contains a nonnumeric job ID: {value!r}"
                )
            result.append((dependency_type, job_id))
    return result


def _scheduler_acceptance_projection(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the scheduler-stable facts used to reconcile a replay."""

    rows = payload.get("jobs")
    if not isinstance(rows, list):
        raise ChainError("scheduler acceptance has no typed jobs")
    return [
        {
            "name": row["name"],
            "job_id": row["job_id"],
            "job_name": row["job_name"],
            "comment": row["comment"],
            "dependency_type": row["dependency_type"],
            "dependency_job_ids": row["dependency_job_ids"],
            "command": row["command"],
            "requeue": row["requeue"],
            "squeue_state": row["squeue_state"],
            "sacct_state": row["sacct_state"],
            "local_script_sha256": row["local_script_sha256"],
            "spooled_script_sha256": row["spooled_script_sha256"],
            "spooled_script_bytes": row["spooled_script_bytes"],
            "spooled_script_exact_match": row["spooled_script_exact_match"],
        }
        for row in rows
    ]


def _write_immutable_bytes_once(
    path: Path, payload: bytes, *, description: str
) -> None:
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or stat.S_IMODE(path.stat().st_mode) & 0o222
            or path.read_bytes() != payload
        ):
            raise ChainError(f"existing {description} drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_exact_scontrol_job_output(
    raw: str, *, job_id: str
) -> dict[str, str]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ChainError(
            f"scontrol job {job_id} did not return exactly one nonempty line"
        )
    fields: dict[str, str] = {}
    for token in lines[0].split():
        if "=" not in token:
            raise ChainError(
                f"scontrol job {job_id} returned a non-field token"
            )
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ChainError(
                f"scontrol job {job_id} returned a duplicate/empty field"
            )
        fields[key] = value
    return fields


def _capture_scheduler_acceptance(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
    runner: Runner,
    timestamp: float,
    scheduler_since: str,
    artifact_root: Path | None,
    transaction_intent: Mapping[str, Any] | None,
    transaction_intent_path: Path | None,
    spool_staging_root: Path,
) -> dict[str, Any]:
    """Prove every held DAG job's live Slurm and spooled-script identity."""

    persist_artifacts = artifact_root is not None
    if persist_artifacts:
        assert artifact_root is not None
        if (
            transaction_intent is None
            or transaction_intent_path is None
            or artifact_root.is_symlink()
            or not artifact_root.is_dir()
            or artifact_root.resolve() != artifact_root
        ):
            raise ChainError(
                "scheduler acceptance artifact transaction is unavailable or unsafe"
            )
    elif transaction_intent is not None or transaction_intent_path is not None:
        raise ChainError(
            "ephemeral scheduler acceptance cannot claim a transaction intent"
        )
    if (
        spool_staging_root.is_symlink()
        or not spool_staging_root.is_dir()
        or spool_staging_root.resolve() != spool_staging_root
    ):
        raise ChainError("scheduler spool staging root is unavailable or unsafe")
    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs")
    if (
        not isinstance(manifest_jobs, list)
        or not isinstance(receipt_jobs, list)
        or len(manifest_jobs) != len(EXPECTED_JOB_ORDER)
        or len(receipt_jobs) != len(EXPECTED_JOB_ORDER)
        or len(EXPECTED_JOB_ORDER) != 43
    ):
        raise ChainError("scheduler acceptance requires the exact 43-job DAG")
    receipt_by_id = {
        str(row.get("job_id")): row
        for row in receipt_jobs
        if isinstance(row, Mapping)
    }
    if len(receipt_by_id) != len(receipt_jobs) or any(
        not job_id.isdigit() for job_id in receipt_by_id
    ):
        raise ChainError("scheduler acceptance receipt job IDs are ambiguous")
    requested_ids = sorted(receipt_by_id, key=int)
    requested = set(requested_ids)
    expected_comments = {
        str(row["comment"]) for row in receipt_jobs if isinstance(row, Mapping)
    }
    expected_comment_prefix = (
        "asys:s5-recovery-v1.2-r7:"
        f"{manifest['chain_id']}:g0000:"
    )

    squeue_argv = [
        "squeue",
        "-u",
        str(manifest["slurm_user"]),
        "-h",
        "-o",
        "%i|%T|%k|%j",
    ]
    squeue = runner(squeue_argv)
    if squeue.returncode != 0:
        raise ChainError(
            "scheduler acceptance requires complete squeue truth: "
            + squeue.stderr.strip()[:500]
        )
    queue_rows: dict[str, dict[str, str]] = {}
    for raw in squeue.stdout.splitlines():
        fields = [field.strip() for field in raw.split("|")]
        if len(fields) < 4:
            if raw.strip():
                raise ChainError(
                    f"scheduler acceptance received malformed squeue row: {raw[:300]!r}"
                )
            continue
        job_id, state, comment, job_name = fields[:4]
        if job_id not in requested:
            if (
                comment in expected_comments
                or comment.startswith(expected_comment_prefix)
            ):
                raise ChainError(
                    "foreign squeue job collides with the recovery namespace: "
                    f"{job_id}|{comment}|{job_name}"
                )
            continue
        if job_id in queue_rows:
            raise ChainError(
                f"scheduler acceptance found duplicate squeue job {job_id}"
            )
        queue_rows[job_id] = {
            "state": _normalize_slurm_state(state),
            "comment": comment,
            "job_name": job_name,
        }
    if set(queue_rows) != requested:
        raise ChainError(
            "scheduler acceptance squeue set is incomplete: "
            f"missing={sorted(requested - set(queue_rows), key=int)}"
        )

    sacct_argv = [
        "sacct",
        "-u",
        str(manifest["slurm_user"]),
        "-X",
        "-n",
        "-P",
        "-S",
        scheduler_since,
        "--format=JobIDRaw,State,ExitCode,Comment%256,JobName%64,SubmitLine",
    ]
    sacct = runner(sacct_argv)
    if sacct.returncode != 0:
        raise ChainError(
            "scheduler acceptance requires complete sacct truth: "
            + sacct.stderr.strip()[:500]
        )
    accounting_rows: dict[str, dict[str, str]] = {}
    for raw in sacct.stdout.splitlines():
        fields = raw.rstrip("\n").split("|")
        if len(fields) < 6:
            if raw.strip():
                raise ChainError(
                    f"scheduler acceptance received malformed sacct row: {raw[:300]!r}"
                )
            continue
        job_id, state, exit_code, stored_comment, job_name = (
            field.strip() for field in fields[:5]
        )
        if "." in job_id:
            continue
        comment = _accounting_comment(
            stored_comment, fields[5], job_id=job_id
        )
        if job_id not in requested:
            if (
                comment in expected_comments
                or comment.startswith(expected_comment_prefix)
            ):
                raise ChainError(
                    "foreign sacct job collides with the recovery namespace: "
                    f"{job_id}|{comment}|{job_name}"
                )
            continue
        candidate = {
            "state": _normalize_slurm_state(state),
            "exit_code": exit_code,
            "comment": comment,
            "job_name": job_name,
        }
        if job_id in accounting_rows:
            raise ChainError(
                f"scheduler acceptance found duplicate sacct job {job_id}"
            )
        accounting_rows[job_id] = candidate
    if set(accounting_rows) != requested:
        raise ChainError(
            "scheduler acceptance sacct set is incomplete: "
            f"missing={sorted(requested - set(accounting_rows), key=int)}"
        )

    accepted_jobs: list[dict[str, Any]] = []
    artifact_inventory: list[dict[str, Any]] = []
    submitted: dict[str, str] = {}
    for index, (manifest_row, receipt_row) in enumerate(
        zip(manifest_jobs, receipt_jobs, strict=True)
    ):
        if not isinstance(manifest_row, Mapping) or not isinstance(
            receipt_row, Mapping
        ):
            raise ChainError("scheduler acceptance job record is malformed")
        name = str(manifest_row["name"])
        job_id = str(receipt_row["job_id"])
        expected_dependencies = [
            submitted[dependency]
            for dependency in manifest_row["dependencies"]
        ]
        expected_pairs = sorted(
            (
                str(manifest_row["dependency_type"]),
                dependency_job_id,
            )
            for dependency_job_id in expected_dependencies
        )
        queue = queue_rows[job_id]
        accounting = accounting_rows[job_id]
        expected_comment = str(receipt_row["comment"])
        expected_name = str(manifest_row["job_name"])
        if (
            queue["state"] != "PENDING"
            or accounting["state"] != "PENDING"
            or queue["comment"] != expected_comment
            or accounting["comment"] != expected_comment
            or queue["job_name"] != expected_name
            or accounting["job_name"] != expected_name
        ):
            raise ChainError(
                f"squeue/sacct acceptance identity drifted for {name} ({job_id})"
            )

        details = _show_recovery_job(job_id, runner=runner)
        if details["returncode"] != 0:
            raise ChainError(
                f"scontrol cannot inspect recovery job {name} ({job_id})"
            )
        fields = _parse_exact_scontrol_job_output(
            details["stdout"], job_id=job_id
        )
        observed_pairs = sorted(
            _scheduler_dependency_pairs(str(fields.get("Dependency", "")))
        )
        local_path = Path(str(manifest_row["script"]))
        if (
            fields.get("JobId") != job_id
            or fields.get("JobName") != expected_name
            or fields.get("Comment") != expected_comment
            or fields.get("Requeue") != "0"
            or _normalize_slurm_state(str(fields.get("JobState", "")))
            != "PENDING"
            or observed_pairs != expected_pairs
            or fields.get("Command") != str(local_path)
        ):
            raise ChainError(
                f"scontrol acceptance provenance drifted for {name} ({job_id})"
            )
        if name == "source_checkout" and str(
            fields.get("Reason", "")
        ).lower() != "jobhelduser":
            raise ChainError("recovery root is not held during scheduler acceptance")

        if (
            local_path.is_symlink()
            or not local_path.is_file()
            or stat.S_IMODE(local_path.stat().st_mode) & 0o222
        ):
            raise ChainError(f"immutable local sbatch is unsafe for {name}")
        local = local_path.read_bytes()
        local_sha256 = _sha256_bytes(local)
        if (
            local_sha256 != manifest_row["script_sha256"]
            or local_sha256 != receipt_row["script_sha256"]
        ):
            raise ChainError(f"immutable local sbatch hash drifted for {name}")
        safe_name = f"{index:02d}_{name}"
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{safe_name}.spool.",
                dir=spool_staging_root,
            )
        )
        staging_path = staging_directory / "batch_script.sbatch"
        spool_argv = [
            "scontrol",
            "write",
            "batch_script",
            job_id,
            str(staging_path),
        ]
        try:
            if staging_path.exists() or staging_path.is_symlink():
                raise ChainError(
                    f"scheduler spool staging path unexpectedly exists for {name}"
                )
            spool = runner(spool_argv)
            if spool.returncode != 0:
                raise ChainError(
                    f"cannot read exact spooled sbatch for {name} ({job_id}): "
                    + spool.stderr.strip()[:500]
                )
            try:
                spool_metadata = staging_path.lstat()
            except OSError as exc:
                raise ChainError(
                    f"scontrol did not create a spooled sbatch for {name}: {exc}"
                ) from exc
            if (
                not stat.S_ISREG(spool_metadata.st_mode)
                or spool_metadata.st_nlink != 1
                or staging_path.resolve() != staging_path
                or staging_path.parent != staging_directory
            ):
                raise ChainError(
                    f"scontrol created an unsafe spooled sbatch for {name}"
                )
            with staging_path.open("rb") as handle:
                observed_spool = handle.read()
                os.fsync(handle.fileno())
            if observed_spool != local:
                raise ChainError(
                    "Slurm spooled sbatch differs from sealed local bytes "
                    f"for {name}"
                )
            os.chmod(staging_path, 0o444)
            with staging_path.open("rb") as handle:
                os.fsync(handle.fileno())
        except BaseException:
            try:
                staging_path.unlink()
            except FileNotFoundError:
                pass
            try:
                staging_directory.rmdir()
            except OSError:
                pass
            raise
        scontrol_capture_path: str | None = None
        spooled_script_path: str | None = None
        scontrol_artifact_sha256 = details["stdout_sha256"]
        if persist_artifacts:
            assert artifact_root is not None
            scontrol_path = artifact_root / f"{safe_name}.scontrol.txt"
            spool_path = artifact_root / f"{safe_name}.spooled.sbatch"
            if scontrol_path.exists() or scontrol_path.is_symlink():
                if (
                    scontrol_path.is_symlink()
                    or not scontrol_path.is_file()
                    or scontrol_path.stat().st_nlink != 1
                    or stat.S_IMODE(scontrol_path.stat().st_mode) & 0o222
                ):
                    raise ChainError(
                        "existing scheduler acceptance scontrol capture is unsafe: "
                        f"{scontrol_path}"
                    )
                prior_fields = _parse_exact_scontrol_job_output(
                    scontrol_path.read_text(encoding="utf-8"),
                    job_id=job_id,
                )
                stable_keys = {
                    "JobId",
                    "JobName",
                    "JobState",
                    "Reason",
                    "Comment",
                    "Command",
                    "Dependency",
                    "Requeue",
                }
                if {
                    key: prior_fields.get(key) for key in stable_keys
                } != {key: fields.get(key) for key in stable_keys}:
                    raise ChainError(
                        "live scheduler identity drifted from its partial "
                        f"scontrol capture for {name}"
                    )
            else:
                _write_immutable_bytes_once(
                    scontrol_path,
                    details["stdout"].encode("utf-8"),
                    description=(
                        f"scheduler acceptance scontrol capture for {name}"
                    ),
                )
            scontrol_artifact_sha256 = _sha256(scontrol_path)
            if spool_path.exists() or spool_path.is_symlink():
                if (
                    spool_path.is_symlink()
                    or not spool_path.is_file()
                    or spool_path.stat().st_nlink != 1
                    or stat.S_IMODE(spool_path.stat().st_mode) & 0o222
                    or spool_path.read_bytes() != observed_spool
                ):
                    raise ChainError(
                        "existing scheduler acceptance spooled sbatch drifted: "
                        f"{spool_path}"
                    )
                staging_path.unlink()
            else:
                os.replace(staging_path, spool_path)
                _fsync_directory(spool_path.parent)
            staging_directory.rmdir()
            scontrol_capture_path = str(scontrol_path)
            spooled_script_path = str(spool_path)
            artifact_inventory.extend(
                (
                    {
                        "kind": "scontrol",
                        "name": name,
                        "path": str(scontrol_path),
                        "sha256": _sha256(scontrol_path),
                        "bytes": scontrol_path.stat().st_size,
                    },
                    {
                        "kind": "spooled_script",
                        "name": name,
                        "path": str(spool_path),
                        "sha256": _sha256(spool_path),
                        "bytes": spool_path.stat().st_size,
                    },
                )
            )
        else:
            staging_path.unlink()
            staging_directory.rmdir()
        accepted_jobs.append(
            {
                "name": name,
                "job_id": job_id,
                "job_name": expected_name,
                "comment": expected_comment,
                "dependency_type": str(manifest_row["dependency_type"]),
                "dependency_job_ids": expected_dependencies,
                "command": str(local_path),
                "requeue": 0,
                "squeue_state": queue["state"],
                "sacct_state": accounting["state"],
                "scontrol_argv": details["argv"],
                "scontrol_capture_path": scontrol_capture_path,
                "scontrol_stdout_sha256": scontrol_artifact_sha256,
                "scontrol_stderr_sha256": details["stderr_sha256"],
                "spooled_script_argv": spool_argv,
                "spooled_script_path": spooled_script_path,
                "local_script_sha256": local_sha256,
                "spooled_script_sha256": _sha256_bytes(observed_spool),
                "spooled_script_bytes": len(observed_spool),
                "spooled_script_exact_match": True,
            }
        )
        submitted[name] = job_id

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SCHEDULER_ACCEPTANCE_PROTOCOL,
        "passed": True,
        "chain_id": manifest["chain_id"],
        "manifest": str(Path(str(receipt["manifest"]))),
        "manifest_sha256": receipt["manifest_sha256"],
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "captured_at": datetime.fromtimestamp(
            timestamp, timezone.utc
        ).isoformat(),
        "captured_timestamp": timestamp,
        "job_count": len(accepted_jobs),
        "transaction_intent": (
            None if transaction_intent_path is None else str(transaction_intent_path)
        ),
        "transaction_intent_sha256": (
            None
            if transaction_intent_path is None
            else _sha256(transaction_intent_path)
        ),
        "transaction_id": (
            None
            if transaction_intent is None
            else transaction_intent.get("transaction_id")
        ),
        "artifact_root": (
            None if artifact_root is None else str(artifact_root)
        ),
        "spool_staging_root": str(spool_staging_root),
        "artifact_inventory": artifact_inventory,
        "scheduler_queries": {
            "squeue": {
                "argv": squeue_argv,
                "stdout": squeue.stdout,
                "stdout_sha256": _sha256_bytes(
                    squeue.stdout.encode("utf-8")
                ),
                "stderr_sha256": _sha256_bytes(
                    squeue.stderr.encode("utf-8")
                ),
            },
            "sacct": {
                "argv": sacct_argv,
                "stdout": sacct.stdout,
                "stdout_sha256": _sha256_bytes(
                    sacct.stdout.encode("utf-8")
                ),
                "stderr_sha256": _sha256_bytes(
                    sacct.stderr.encode("utf-8")
                ),
            },
        },
        "jobs": accepted_jobs,
    }
    payload["acceptance_id"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _validate_scheduler_acceptance(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    path = _require_canonical_path(
        path, description="scheduler acceptance evidence", kind="file"
    )
    if (
        path.stat().st_nlink != 1
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise ChainError("scheduler acceptance evidence must be read-only")
    payload = _read_json(path, description="scheduler acceptance evidence")
    identity = dict(payload)
    acceptance_id = identity.pop("acceptance_id", None)
    expected_fields = {
        "schema_version",
        "protocol",
        "passed",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "captured_at",
        "captured_timestamp",
        "job_count",
        "transaction_intent",
        "transaction_intent_sha256",
        "transaction_id",
        "artifact_root",
        "spool_staging_root",
        "artifact_inventory",
        "scheduler_queries",
        "jobs",
        "acceptance_id",
    }
    timestamp = payload.get("captured_timestamp")
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != 1
        or payload.get("protocol") != SCHEDULER_ACCEPTANCE_PROTOCOL
        or payload.get("passed") is not True
        or payload.get("chain_id") != manifest.get("chain_id")
        or payload.get("manifest") != receipt.get("manifest")
        or payload.get("manifest_sha256") != receipt.get("manifest_sha256")
        or payload.get("submission_receipt") != str(receipt_path)
        or payload.get("submission_receipt_sha256") != _sha256(receipt_path)
        or payload.get("submission_receipt_id") != receipt.get("receipt_id")
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or payload.get("captured_at")
        != datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
        or payload.get("job_count") != len(EXPECTED_JOB_ORDER)
        or acceptance_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("scheduler acceptance evidence identity is invalid")
    intent_path = _require_canonical_path(
        Path(str(payload.get("transaction_intent", ""))),
        description="scheduler acceptance transaction intent",
        kind="file",
    )
    if (
        intent_path.stat().st_nlink != 1
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
        or _sha256(intent_path) != payload.get("transaction_intent_sha256")
    ):
        raise ChainError("scheduler acceptance transaction intent drifted")
    intent = _read_json(
        intent_path, description="scheduler acceptance transaction intent"
    )
    intent_identity = dict(intent)
    transaction_id = intent_identity.pop("transaction_id", None)
    artifact_root = _require_canonical_path(
        Path(str(payload.get("artifact_root", ""))),
        description="scheduler acceptance artifact root",
        kind="directory",
    )
    if (
        set(intent)
        != {
            "schema_version",
            "protocol",
            "chain_id",
            "submission_receipt",
            "submission_receipt_sha256",
            "submission_receipt_id",
            "artifact_root",
            "spool_staging_root",
            "created_at",
            "created_timestamp",
            "transaction_id",
        }
        or intent.get("schema_version") != 1
        or intent.get("protocol")
        != "schema5-v1.2-r7-recovery-scheduler-acceptance-intent-v1"
        or intent.get("chain_id") != manifest.get("chain_id")
        or intent.get("submission_receipt") != str(receipt_path)
        or intent.get("submission_receipt_sha256") != _sha256(receipt_path)
        or intent.get("submission_receipt_id") != receipt.get("receipt_id")
        or intent.get("artifact_root") != str(artifact_root)
        or intent.get("spool_staging_root")
        != payload.get("spool_staging_root")
        or transaction_id != _sha256_bytes(_canonical_json(intent_identity))
        or payload.get("transaction_id") != transaction_id
        or stat.S_IMODE(artifact_root.stat().st_mode) & 0o222
    ):
        raise ChainError("scheduler acceptance transaction identity is invalid")
    spool_staging_root = _require_canonical_path(
        Path(str(payload.get("spool_staging_root", ""))),
        description="scheduler acceptance spool staging root",
        kind="directory",
    )
    if (
        spool_staging_root.parent != artifact_root.parent
        or spool_staging_root.name != SCHEDULER_ACCEPTANCE_STAGING_ROOT_NAME
    ):
        raise ChainError("scheduler acceptance spool staging root drifted")
    inventory = payload.get("artifact_inventory")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 2 * len(EXPECTED_JOB_ORDER)
    ):
        raise ChainError("scheduler acceptance artifact inventory is incomplete")
    inventory_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    inventory_paths: set[Path] = set()
    for record in inventory:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"kind", "name", "path", "sha256", "bytes"}
            or record.get("kind") not in {"scontrol", "spooled_script"}
            or not isinstance(record.get("name"), str)
            or (str(record["kind"]), str(record["name"])) in inventory_by_key
        ):
            raise ChainError("scheduler acceptance artifact record is invalid")
        artifact = _require_canonical_path(
            Path(str(record["path"])),
            description="scheduler acceptance artifact",
            kind="file",
        )
        try:
            artifact.relative_to(artifact_root)
        except ValueError as exc:
            raise ChainError(
                "scheduler acceptance artifact escaped its transaction root"
            ) from exc
        if (
            artifact.parent != artifact_root
            or artifact.stat().st_nlink != 1
            or stat.S_IMODE(artifact.stat().st_mode) & 0o222
            or _sha256(artifact) != record.get("sha256")
            or artifact.stat().st_size != record.get("bytes")
        ):
            raise ChainError("scheduler acceptance artifact drifted")
        inventory_by_key[(str(record["kind"]), str(record["name"]))] = record
        inventory_paths.add(artifact)
    observed_artifacts = {
        member
        for member in artifact_root.iterdir()
        if member.is_file() and not member.is_symlink()
    }
    if (
        observed_artifacts != inventory_paths
        or len(list(artifact_root.iterdir())) != len(inventory_paths)
    ):
        raise ChainError(
            "scheduler acceptance artifact root contains unlisted members"
        )
    queries = payload.get("scheduler_queries")
    if not isinstance(queries, Mapping) or set(queries) != {"squeue", "sacct"}:
        raise ChainError("scheduler acceptance query evidence is incomplete")
    for name in ("squeue", "sacct"):
        record = queries.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"argv", "stdout", "stdout_sha256", "stderr_sha256"}
            or not isinstance(record.get("argv"), list)
            or not isinstance(record.get("stdout"), str)
            or record.get("stdout_sha256")
            != _sha256_bytes(record["stdout"].encode("utf-8"))
            or _SHA256.fullmatch(str(record.get("stderr_sha256", ""))) is None
        ):
            raise ChainError(
                f"scheduler acceptance {name} query evidence is invalid"
            )
    rows = payload.get("jobs")
    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs")
    if (
        not isinstance(rows, list)
        or not isinstance(manifest_jobs, list)
        or not isinstance(receipt_jobs, list)
        or len(rows) != len(EXPECTED_JOB_ORDER)
    ):
        raise ChainError("scheduler acceptance job evidence is incomplete")
    expected_job_fields = {
        "name",
        "job_id",
        "job_name",
        "comment",
        "dependency_type",
        "dependency_job_ids",
        "command",
        "requeue",
        "squeue_state",
        "sacct_state",
        "scontrol_argv",
        "scontrol_capture_path",
        "scontrol_stdout_sha256",
        "scontrol_stderr_sha256",
        "spooled_script_argv",
        "spooled_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_bytes",
        "spooled_script_exact_match",
    }
    submitted: dict[str, str] = {}
    for row_index, (row, manifest_row, receipt_row) in enumerate(
        zip(rows, manifest_jobs, receipt_jobs, strict=True)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_job_fields
            or row.get("name") != manifest_row.get("name")
            or row.get("job_id") != receipt_row.get("job_id")
            or row.get("job_name") != manifest_row.get("job_name")
            or row.get("comment") != receipt_row.get("comment")
            or row.get("dependency_type")
            != manifest_row.get("dependency_type")
            or row.get("dependency_job_ids")
            != [submitted[name] for name in manifest_row.get("dependencies", [])]
            or row.get("command") != manifest_row.get("script")
            or row.get("requeue") != 0
            or row.get("squeue_state") != "PENDING"
            or row.get("sacct_state") != "PENDING"
            or row.get("scontrol_argv")
            != ["scontrol", "show", "job", "-o", row.get("job_id")]
            or not isinstance(row.get("spooled_script_argv"), list)
            or len(row["spooled_script_argv"]) != 5
            or row["spooled_script_argv"][:4]
            != [
                "scontrol",
                "write",
                "batch_script",
                row.get("job_id"),
            ]
            or row.get("spooled_script_exact_match") is not True
            or row.get("local_script_sha256")
            != manifest_row.get("script_sha256")
            or row.get("spooled_script_sha256")
            != manifest_row.get("script_sha256")
            or _SHA256.fullmatch(
                str(row.get("scontrol_stdout_sha256", ""))
            )
            is None
            or _SHA256.fullmatch(
                str(row.get("scontrol_stderr_sha256", ""))
            )
            is None
            or not isinstance(row.get("spooled_script_bytes"), int)
            or isinstance(row.get("spooled_script_bytes"), bool)
            or int(row["spooled_script_bytes"]) <= 0
        ):
            raise ChainError(
                "scheduler acceptance job evidence drifted: "
                f"{manifest_row.get('name')}"
            )
        local_path = Path(str(row["command"]))
        spool_staging_path = Path(str(row["spooled_script_argv"][4]))
        try:
            spool_staging_path.relative_to(spool_staging_root)
        except ValueError as exc:
            raise ChainError(
                "scheduler acceptance spool staging path escaped its transaction root"
            ) from exc
        if (
            not spool_staging_path.is_absolute()
            or spool_staging_path.name != "batch_script.sbatch"
            or not spool_staging_path.parent.name.startswith(
                f".{row_index:02d}_{row['name']}.spool."
            )
        ):
            raise ChainError(
                f"scheduler acceptance spool command drifted: {row['name']}"
            )
        scontrol_record = inventory_by_key.get(("scontrol", str(row["name"])))
        spool_record = inventory_by_key.get(
            ("spooled_script", str(row["name"]))
        )
        if (
            scontrol_record is None
            or spool_record is None
            or row.get("scontrol_capture_path") != scontrol_record.get("path")
            or row.get("spooled_script_path") != spool_record.get("path")
            or row.get("scontrol_stdout_sha256")
            != scontrol_record.get("sha256")
            or row.get("spooled_script_sha256") != spool_record.get("sha256")
        ):
            raise ChainError(
                f"scheduler acceptance artifact binding drifted: {row['name']}"
            )
        captured_fields = _parse_exact_scontrol_job_output(
            Path(str(row["scontrol_capture_path"])).read_text(
                encoding="utf-8"
            ),
            job_id=str(row["job_id"]),
        )
        expected_pairs = sorted(
            (
                str(row["dependency_type"]),
                dependency_job_id,
            )
            for dependency_job_id in row["dependency_job_ids"]
        )
        if (
            local_path.is_symlink()
            or not local_path.is_file()
            or stat.S_IMODE(local_path.stat().st_mode) & 0o222
            or _sha256(local_path) != row["local_script_sha256"]
            or local_path.stat().st_size != row["spooled_script_bytes"]
            or Path(str(row["spooled_script_path"])).read_bytes()
            != local_path.read_bytes()
            or captured_fields.get("JobId") != row["job_id"]
            or captured_fields.get("JobName") != row["job_name"]
            or captured_fields.get("Comment") != row["comment"]
            or captured_fields.get("Command") != row["command"]
            or captured_fields.get("Requeue") != "0"
            or _normalize_slurm_state(
                str(captured_fields.get("JobState", ""))
            )
            != "PENDING"
            or sorted(
                _scheduler_dependency_pairs(
                    str(captured_fields.get("Dependency", ""))
                )
            )
            != expected_pairs
            or (
                row["name"] == "source_checkout"
                and str(captured_fields.get("Reason", "")).lower()
                != "jobhelduser"
            )
        ):
            raise ChainError(
                f"scheduler acceptance local script drifted: {row['name']}"
            )
        submitted[str(row["name"])] = str(row["job_id"])
    return payload


def _ensure_scheduler_acceptance(
    *,
    evidence_root: Path,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: Path,
    runner: Runner,
    timestamp: float,
    scheduler_since: str,
) -> tuple[dict[str, Any], Path]:
    """Publish scheduler/spool acceptance marker-last and re-prove it on replay."""

    path = evidence_root / SCHEDULER_ACCEPTANCE_COMPLETE_NAME
    if path.exists() or path.is_symlink():
        persisted = _validate_scheduler_acceptance(
            path,
            manifest=manifest,
            receipt=receipt,
            receipt_path=receipt_path,
        )
        roots = [
            row
            for row in receipt["jobs"]
            if isinstance(row, Mapping)
            and row.get("name") == "source_checkout"
        ]
        if len(roots) != 1:
            raise ChainError("scheduler acceptance has no exact held root")
        root_job_id = str(roots[0]["job_id"])
        root_details = _show_recovery_job(root_job_id, runner=runner)
        root_fields = (
            _parse_exact_scontrol_job_output(
                root_details["stdout"], job_id=root_job_id
            )
            if root_details["returncode"] == 0
            else {}
        )
        root_held = (
            _normalize_slurm_state(str(root_fields.get("JobState", "")))
            == "PENDING"
            and str(root_fields.get("Reason", "")).lower() == "jobhelduser"
        )
        if root_held:
            staging_root = _require_canonical_path(
                Path(str(persisted["spool_staging_root"])),
                description="scheduler acceptance spool staging root",
                kind="directory",
            )
            fresh = _capture_scheduler_acceptance(
                manifest=manifest,
                receipt=receipt,
                receipt_path=receipt_path,
                runner=runner,
                timestamp=timestamp,
                scheduler_since=scheduler_since,
                artifact_root=None,
                transaction_intent=None,
                transaction_intent_path=None,
                spool_staging_root=staging_root,
            )
            if _scheduler_acceptance_projection(
                persisted
            ) != _scheduler_acceptance_projection(fresh):
                raise ChainError(
                    "live scheduler/spool acceptance drifted on transaction replay"
                )
            return persisted, path

        # A submitter can die immediately after Slurm accepts ``scontrol release``.
        # In that state a fresh all-PENDING capture is impossible.  Reconcile only
        # through the sealed marker-first release intent and an exact immutable
        # external-boundary attempt; never infer authority from RUNNING alone.
        release_intent_path = evidence_root / ROOT_RELEASE_INTENT_NAME
        release_intent = _read_json(
            _require_canonical_path(
                release_intent_path,
                description="recovery root release intent",
                kind="file",
            ),
            description="recovery root release intent",
        )
        if (
            release_intent_path.stat().st_nlink != 1
            or stat.S_IMODE(release_intent_path.stat().st_mode) & 0o222
            or release_intent.get("root_job_id") != root_job_id
            or release_intent.get("command")
            != ["scontrol", "release", root_job_id]
            or release_intent.get("scheduler_acceptance") != str(path)
            or release_intent.get("scheduler_acceptance_sha256")
            != _sha256(path)
            or release_intent.get("scheduler_acceptance_id")
            != persisted["acceptance_id"]
        ):
            raise ChainError(
                "non-held recovery root lacks its exact acceptance-bound release intent"
            )
        attempts_root = evidence_root / "root_release_attempts"
        attempts = sorted(attempts_root.glob("attempt-*.intent.json"))
        if not attempts:
            raise ChainError(
                "non-held recovery root lacks a marker-first release attempt"
            )
        for index, attempt_path in enumerate(attempts, start=1):
            attempt = _read_json(
                _require_canonical_path(
                    attempt_path,
                    description="recovery root release attempt intent",
                    kind="file",
                ),
                description="recovery root release attempt intent",
            )
            if (
                attempt_path.name != f"attempt-{index:04d}.intent.json"
                or attempt_path.stat().st_nlink != 1
                or stat.S_IMODE(attempt_path.stat().st_mode) & 0o222
                or attempt.get("attempt") != index
                or attempt.get("job_id") != root_job_id
                or attempt.get("command")
                != ["scontrol", "release", root_job_id]
            ):
                raise ChainError(
                    "recovery root release attempt intent drifted"
                )
        return persisted, path

    artifact_root = evidence_root / SCHEDULER_ACCEPTANCE_ARTIFACT_ROOT_NAME
    if artifact_root.is_symlink() or (
        artifact_root.exists() and not artifact_root.is_dir()
    ):
        raise ChainError("scheduler acceptance artifact root is unsafe")
    artifact_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve()
    staging_root = evidence_root / SCHEDULER_ACCEPTANCE_STAGING_ROOT_NAME
    if staging_root.is_symlink() or (
        staging_root.exists() and not staging_root.is_dir()
    ):
        raise ChainError("scheduler acceptance spool staging root is unsafe")
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging_root = staging_root.resolve()
    intent_path = evidence_root / SCHEDULER_ACCEPTANCE_INTENT_NAME
    intent = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r7-recovery-scheduler-acceptance-intent-v1"
        ),
        "chain_id": manifest["chain_id"],
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "artifact_root": str(artifact_root),
        "spool_staging_root": str(staging_root),
        "created_at": datetime.fromtimestamp(
            timestamp, timezone.utc
        ).isoformat(),
        "created_timestamp": timestamp,
    }
    intent["transaction_id"] = _sha256_bytes(_canonical_json(intent))
    if intent_path.exists() or intent_path.is_symlink():
        persisted_intent = _read_json(
            intent_path, description="scheduler acceptance transaction intent"
        )
        comparable = dict(intent)
        comparable["created_at"] = persisted_intent.get("created_at")
        comparable["created_timestamp"] = persisted_intent.get(
            "created_timestamp"
        )
        comparable_without_id = dict(comparable)
        comparable_without_id.pop("transaction_id", None)
        comparable["transaction_id"] = _sha256_bytes(
            _canonical_json(comparable_without_id)
        )
        if (
            persisted_intent != comparable
            or intent_path.stat().st_nlink != 1
            or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
        ):
            raise ChainError("scheduler acceptance transaction intent drifted")
        intent = persisted_intent
    else:
        _write_immutable_json_once(
            intent_path,
            intent,
            description="marker-first scheduler acceptance transaction intent",
        )
    fresh = _capture_scheduler_acceptance(
        manifest=manifest,
        receipt=receipt,
        receipt_path=receipt_path,
        runner=runner,
        timestamp=timestamp,
        scheduler_since=scheduler_since,
        artifact_root=artifact_root,
        transaction_intent=intent,
        transaction_intent_path=intent_path,
        spool_staging_root=staging_root,
    )
    expected_artifacts = {
        Path(str(record["path"])) for record in fresh["artifact_inventory"]
    }
    if (
        len(expected_artifacts) != 2 * len(EXPECTED_JOB_ORDER)
        or set(artifact_root.iterdir()) != expected_artifacts
    ):
        raise ChainError(
            "scheduler acceptance artifact transaction is incomplete"
        )
    artifact_root.chmod(0o555)
    _fsync_directory(artifact_root.parent)
    _write_immutable_json_once(
        path,
        fresh,
        description="marker-last scheduler acceptance evidence",
    )
    return (
        _validate_scheduler_acceptance(
            path,
            manifest=manifest,
            receipt=receipt,
            receipt_path=receipt_path,
        ),
        path,
    )


def _validate_bootstrap_watchdog_deployment_ready(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    path = _require_canonical_path(
        path,
        description="bootstrap watchdog deployment readiness",
        kind="file",
    )
    if path.stat().st_nlink != 1 or stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError(
            "bootstrap watchdog deployment readiness must be sealed"
        )
    payload = _read_json(
        path, description="bootstrap watchdog deployment readiness"
    )
    identity = dict(payload)
    ready_id = identity.pop("ready_id", None)
    if (
        set(payload)
        != {
            "schema_version",
            "protocol",
            "passed",
            "release_id",
            "release_tag",
            "release_git_commit",
            "release_tag_object",
            "chain_namespace",
            "chain_manifest",
            "chain_manifest_sha256",
            "chain_id",
            "deployment_id",
            "watchdog_code_sha256",
            "bootstrap_attestation",
            "bootstrap_attestation_sha256",
            "bootstrap_attestation_id",
            "bundle_manifest",
            "bundle_manifest_sha256",
            "bundle_id",
            "ssh_public_key_sha256",
            "separate_bootstrap_key",
            "forced_command_only",
            "allowed_operations",
            "timer_seconds",
            "root_release_authority",
            "prelaunch_root_release",
            "descendant_rearm_authority",
            "release_intent_continuation_authority",
            "scientific_admission_direct",
            "production_control_mutation",
            "safety_hold_clear_authority",
            "ready_id",
        }
        or payload.get("schema_version") != 1
        or payload.get("protocol") != BOOTSTRAP_WATCHDOG_READY_PROTOCOL
        or payload.get("passed") is not True
        or payload.get("release_id") != RELEASE_ID
        or payload.get("release_tag") != RELEASE_TAG
        or payload.get("release_git_commit")
        != manifest.get("release_git_commit")
        or payload.get("release_tag_object")
        != manifest.get("release_tag_object")
        or payload.get("chain_namespace") != CHAIN_NAMESPACE
        or payload.get("chain_manifest") != str(manifest_path)
        or payload.get("chain_manifest_sha256") != _sha256(manifest_path)
        or payload.get("chain_id") != manifest.get("chain_id")
        or _SHA256.fullmatch(str(payload.get("deployment_id", ""))) is None
        or _SHA256.fullmatch(
            str(payload.get("watchdog_code_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(payload.get("bootstrap_attestation_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(payload.get("bootstrap_attestation_id", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(payload.get("bundle_manifest_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(str(payload.get("bundle_id", ""))) is None
        or _SHA256.fullmatch(
            str(payload.get("ssh_public_key_sha256", ""))
        )
        is None
        or payload.get("separate_bootstrap_key") is not True
        or payload.get("forced_command_only") is not True
        or payload.get("allowed_operations")
        != ["bootstrap-status", "bootstrap-repair"]
        or payload.get("timer_seconds") != 300
        or payload.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or payload.get("prelaunch_root_release") is not False
        or payload.get("descendant_rearm_authority") is not True
        or payload.get("release_intent_continuation_authority")
        is not True
        or payload.get("scientific_admission_direct") is not False
        or payload.get("production_control_mutation") is not False
        or payload.get("safety_hold_clear_authority") is not False
        or not isinstance(ready_id, str)
        or _SHA256.fullmatch(ready_id) is None
        or ready_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError(
            "bootstrap watchdog deployment readiness contract is invalid"
        )
    attestation_path = _require_canonical_path(
        Path(str(payload["bootstrap_attestation"])),
        description="bootstrap watchdog external attestation",
        kind="file",
    )
    bundle_path = _require_canonical_path(
        Path(str(payload["bundle_manifest"])),
        description="bootstrap watchdog bundle manifest",
        kind="file",
    )
    attestation = _read_json(
        attestation_path,
        description="bootstrap watchdog external attestation",
    )
    bundle = _read_json(
        bundle_path, description="bootstrap watchdog bundle manifest"
    )
    attestation_identity = dict(attestation)
    attestation_id = attestation_identity.pop("evidence_id", None)
    bundle_identity = dict(bundle)
    bundle_id = bundle_identity.pop("bundle_id", None)
    try:
        deployment_path = _require_canonical_path(
            Path(str(attestation["deployment_evidence"])),
            description="bootstrap watchdog deployment evidence",
            kind="file",
        )
        drill_path = _require_canonical_path(
            Path(str(attestation["cancellation_drill_evidence"])),
            description="bootstrap watchdog cancellation drill evidence",
            kind="file",
        )
        deployment = _read_json(
            deployment_path,
            description="bootstrap watchdog deployment evidence",
        )
        drill = _read_json(
            drill_path,
            description="bootstrap watchdog cancellation drill evidence",
        )
    except KeyError as exc:
        raise ChainError(
            "bootstrap watchdog attestation lacks backing evidence"
        ) from exc
    deployment_identity = dict(deployment)
    deployment_id = deployment_identity.pop("evidence_id", None)
    drill_identity = dict(drill)
    drill_id = drill_identity.pop("evidence_id", None)
    bundle_harness = bundle.get("harness_environment_binding")
    bundle_pilot = bundle.get("materialization_pilot")
    bundle_forced = bundle.get("forced_command_argv")

    def bundle_forced_value(flag: str) -> str | None:
        if not isinstance(bundle_forced, list):
            return None
        try:
            index = bundle_forced.index(flag)
        except ValueError:
            return None
        if index + 1 >= len(bundle_forced):
            return None
        value = bundle_forced[index + 1]
        return value if isinstance(value, str) else None

    if (
        attestation_path.stat().st_nlink != 1
        or stat.S_IMODE(attestation_path.stat().st_mode) & 0o222
        or bundle_path.stat().st_nlink != 1
        or stat.S_IMODE(bundle_path.stat().st_mode) & 0o222
        or _sha256(attestation_path)
        != payload["bootstrap_attestation_sha256"]
        or attestation_id != payload["bootstrap_attestation_id"]
        or attestation_id
        != _sha256_bytes(_canonical_json(attestation_identity))
        or _sha256(bundle_path) != payload["bundle_manifest_sha256"]
        or bundle_id != payload["bundle_id"]
        or bundle_id != _sha256_bytes(_canonical_json(bundle_identity))
        or bundle.get("protocol")
        != "schema5-v1.2-r7-bootstrap-watchdog-bundle-v1"
        or bundle.get("release_git_commit")
        != manifest.get("release_git_commit")
        or bundle.get("release_tag_object")
        != manifest.get("release_tag_object")
        or bundle.get("chain_manifest_sha256") != _sha256(manifest_path)
        or bundle.get("watchdog_code_sha256")
        != payload["watchdog_code_sha256"]
        or bundle.get("ssh_public_key_sha256")
        != payload["ssh_public_key_sha256"]
        or bundle.get("separate_bootstrap_key") is not True
        or bundle.get("allowed_operations")
        != ["bootstrap-status", "bootstrap-repair"]
        or bundle.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or bundle.get("prelaunch_root_release") is not False
        or bundle.get("descendant_rearm_authority") is not True
        or bundle.get("release_intent_continuation_authority")
        is not True
        or bundle.get("scientific_admission_direct") is not False
        or bundle.get("production_control_mutation") is not False
        or bundle.get("safety_hold_clear_authority") is not False
        or not isinstance(bundle_harness, dict)
        or _SHA256.fullmatch(
            str(bundle_harness.get("manifest_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(bundle_harness.get("inventory_sha256", ""))
        )
        is None
        or not isinstance(bundle_pilot, dict)
        or set(bundle_pilot)
        != {
            "marker",
            "marker_sha256",
            "pilot_id",
            "pilot_root",
            "release_worktree",
            "release_bundle",
            "source_tree_sha256",
            "release_bundle_id",
            "release_identity_sha256",
            "release_completion_sha256",
        }
        or _SHA256.fullmatch(
            str(bundle_pilot.get("marker_sha256", ""))
        )
        is None
        or _SHA256.fullmatch(
            str(bundle_pilot.get("pilot_id", ""))
        )
        is None
        or any(
            _SHA256.fullmatch(str(bundle_pilot.get(field, ""))) is None
            for field in (
                "source_tree_sha256",
                "release_bundle_id",
                "release_identity_sha256",
                "release_completion_sha256",
            )
        )
        or bundle_forced_value("--harness-python")
        != bundle_harness.get("lexical_path")
        or bundle_forced_value("--resolved-harness-python")
        != bundle_harness.get("resolved_path")
        or bundle_forced_value("--harness-environment-manifest")
        != bundle_harness.get("manifest_path")
        or bundle_forced_value("--harness-environment-sha256")
        != bundle_harness.get("manifest_sha256")
        or bundle_forced_value("--materialization-pilot-marker")
        != bundle_pilot.get("marker")
        or bundle_forced_value("--materialization-pilot-sha256")
        != bundle_pilot.get("marker_sha256")
        or bundle_forced_value("--materialization-pilot-id")
        != bundle_pilot.get("pilot_id")
        or attestation.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or attestation.get("prelaunch_root_release") is not False
        or attestation.get("descendant_rearm_authority") is not True
        or attestation.get("release_intent_continuation_authority")
        is not True
        or attestation.get("scientific_admission_direct") is not False
        or attestation.get("production_control_mutation") is not False
        or attestation.get("safety_hold_clear_authority") is not False
        or deployment_path.stat().st_nlink != 1
        or stat.S_IMODE(deployment_path.stat().st_mode) & 0o222
        or _sha256(deployment_path)
        != attestation.get("deployment_evidence_sha256")
        or deployment_id != attestation.get("deployment_evidence_id")
        or deployment_id
        != _sha256_bytes(_canonical_json(deployment_identity))
        or deployment.get("protocol")
        != "schema5-v1.2-r7-bootstrap-watchdog-deployment-evidence-v1"
        or deployment.get("passed") is not True
        or deployment.get("bundle_id") != bundle_id
        or deployment.get("deployment_id")
        != attestation.get("deployment_id")
        or deployment.get("watchdog_code_sha256")
        != bundle.get("watchdog_code_sha256")
        or deployment.get("release_git_commit")
        != manifest.get("release_git_commit")
        or deployment.get("release_tag_object")
        != manifest.get("release_tag_object")
        or deployment.get("chain_id") != manifest.get("chain_id")
        or deployment.get("chain_manifest_sha256")
        != _sha256(manifest_path)
        or deployment.get("submission_receipt_id")
        != bundle.get("submission_receipt_id")
        or deployment.get("submission_receipt_sha256")
        != bundle.get("submission_receipt_sha256")
        or deployment.get("runtime_inventory_sha256")
        != bundle.get("runtime_inventory_sha256")
        or deployment.get("runtime_file_count")
        != bundle.get("runtime_file_count")
        or deployment.get("runtime_total_bytes")
        != bundle.get("runtime_total_bytes")
        or deployment.get("vm_python_path") != bundle.get("vm_python")
        or _SHA256.fullmatch(
            str(deployment.get("vm_python_sha256", ""))
        )
        is None
        or deployment.get("vm_python_immutable") is not True
        or not isinstance(
            deployment.get("heartbeat_freshness_seconds"), (int, float)
        )
        or isinstance(
            deployment.get("heartbeat_freshness_seconds"), bool
        )
        or not math.isfinite(
            float(deployment.get("heartbeat_freshness_seconds", -1))
        )
        or not 0
        <= float(deployment["heartbeat_freshness_seconds"])
        <= BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("heartbeat_max_age_seconds")
        != BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("separate_bootstrap_key") is not True
        or deployment.get("forced_command_only") is not True
        or deployment.get("systemd_service_loaded") is not True
        or deployment.get("systemd_timer_active") is not True
        or drill_path.stat().st_nlink != 1
        or stat.S_IMODE(drill_path.stat().st_mode) & 0o222
        or _sha256(drill_path)
        != attestation.get("cancellation_drill_evidence_sha256")
        or drill_id != attestation.get("cancellation_drill_evidence_id")
        or drill_id != _sha256_bytes(_canonical_json(drill_identity))
        or drill.get("protocol")
        != "schema5-v1.2-r7-bootstrap-watchdog-drill-evidence-v1"
        or drill.get("passed") is not True
        or drill.get("bundle_id") != bundle_id
        or drill.get("deployment_id")
        != attestation.get("deployment_id")
        or drill.get("root_remained_held") is not True
        or drill.get("scientific_jobs_started") != 0
        or drill.get("duplicate_jobs") != 0
        or drill.get("duplicate_submission_intents") != 0
        or drill.get("isolated_cancellation_drill") is not True
        or not isinstance(
            bundle.get("isolated_cancellation_drill"), Mapping
        )
        or drill.get("isolation_id")
        != bundle["isolated_cancellation_drill"].get("isolation_id")
        or drill.get("canonical_job_id_overlap") != 0
        or drill.get("canonical_comment_overlap") != 0
        or drill.get("canonical_control_paths_absent") is not True
        or drill.get("squeue_complete") is not True
        or drill.get("sacct_complete") is not True
    ):
        raise ChainError(
            "bootstrap watchdog deployment backing evidence drifted"
        )
    return payload


def _validate_bootstrap_watchdog_armed(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    root_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the external, pre-control resurrection authority.

    This is deliberately a stronger contract than the production watchdog
    readiness marker.  It exists before ``schema5_initialize`` and therefore
    binds the immutable recovery transaction itself: release identity, exact
    manifest and receipt bytes, every scheduler comment/job ID, and the sole
    narrowly forced operation that may reconstruct a cancelled *held* DAG.
    It never authorizes releasing the root or changing scientific control state.
    """

    path = _require_canonical_path(
        path,
        description="bootstrap watchdog armed marker",
        kind="file",
    )
    if path.stat().st_nlink != 1 or stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError("bootstrap watchdog armed marker must be sealed")
    payload = _read_json(path, description="bootstrap watchdog armed marker")
    identity = dict(payload)
    marker_id = identity.pop("marker_id", None)
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "bootstrap_watchdog_ready",
        "bootstrap_watchdog_ready_sha256",
        "bootstrap_watchdog_ready_id",
        "arm_intent",
        "arm_intent_sha256",
        "arm_intent_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "chain_id",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "root_name",
        "root_job_id",
        "root_comment",
        "job_namespace",
        "job_count",
        "forced_command_only",
        "forced_operation",
        "scheduler_observations",
        "scheduler_observation_artifacts",
        "cancellation_drill",
        "root_release_authority",
        "prelaunch_root_release",
        "descendant_rearm_authority",
        "release_intent_continuation_authority",
        "scientific_admission_direct",
        "production_control_mutation",
        "safety_hold_clear_authority",
        "handoff_stage",
        "marker_id",
    }
    observations = payload.get("scheduler_observations")
    observation_artifacts = payload.get(
        "scheduler_observation_artifacts"
    )
    namespace = payload.get("job_namespace")
    drill = payload.get("cancellation_drill")
    expected_namespace = [
        {
            "name": record.get("name"),
            "job_id": record.get("job_id"),
            "comment": record.get("comment"),
        }
        for record in receipt.get("jobs", [])
        if isinstance(record, Mapping)
    ]
    observation_fields = {
        "observed_at_timestamp",
        "squeue_complete",
        "sacct_complete",
        "submission_receipt_id",
        "job_count",
        "ambiguous_jobs",
        "root_held",
    }
    valid_observations = (
        isinstance(observations, list)
        and len(observations) == 2
        and all(
            isinstance(item, Mapping)
            and set(item) == observation_fields
            and isinstance(item.get("observed_at_timestamp"), (int, float))
            and not isinstance(item.get("observed_at_timestamp"), bool)
            and math.isfinite(float(item["observed_at_timestamp"]))
            and item.get("squeue_complete") is True
            and item.get("sacct_complete") is True
            and item.get("submission_receipt_id") == receipt.get("receipt_id")
            and item.get("job_count") == len(expected_namespace)
            and item.get("ambiguous_jobs") == 0
            and item.get("root_held") is True
            for item in observations
        )
        and float(observations[1]["observed_at_timestamp"])
        - float(observations[0]["observed_at_timestamp"])
        >= 60.0
    )
    valid_observation_artifacts = (
        isinstance(observation_artifacts, list)
        and len(observation_artifacts) == 2
    )
    if valid_observation_artifacts:
        for binding, summary in zip(
            observation_artifacts, observations, strict=True
        ):
            if (
                not isinstance(binding, Mapping)
                or set(binding)
                != {"artifact", "artifact_sha256", "observation_id"}
            ):
                valid_observation_artifacts = False
                break
            try:
                artifact_path = _require_canonical_path(
                    Path(str(binding["artifact"])),
                    description="bootstrap scheduler observation",
                    kind="file",
                )
                artifact = _read_json(
                    artifact_path,
                    description="bootstrap scheduler observation",
                )
            except (OSError, ChainError):
                valid_observation_artifacts = False
                break
            artifact_identity = dict(artifact)
            observation_id = artifact_identity.pop(
                "observation_id", None
            )
            artifact_jobs = artifact.get("jobs")
            if (
                artifact_path.stat().st_nlink != 1
                or stat.S_IMODE(artifact_path.stat().st_mode) & 0o222
                or binding["artifact_sha256"] != _sha256(artifact_path)
                or binding["observation_id"] != observation_id
                or observation_id
                != _sha256_bytes(_canonical_json(artifact_identity))
                or artifact.get("protocol")
                != "schema5-v1.2-r7-bootstrap-status-v1"
                or artifact.get("submission_receipt_id")
                != receipt.get("receipt_id")
                or artifact.get("submission_receipt_sha256")
                != _sha256(receipt_path)
                or artifact.get("anchor_submission_receipt")
                != str(receipt_path)
                or artifact.get("anchor_submission_receipt_sha256")
                != _sha256(receipt_path)
                or artifact.get("anchor_submission_receipt_id")
                != receipt.get("receipt_id")
                or artifact.get("descendant_chain_validated") is not True
                or artifact.get("root_job_id")
                != root_record.get("job_id")
                or artifact.get("root_held") is not True
                or artifact.get("squeue_complete") is not True
                or artifact.get("sacct_complete") is not True
                or artifact.get("ambiguous_jobs") != 0
                or artifact.get("handoff_complete") is not False
                or artifact.get(
                    "watchdog_scientific_jobs_submitted"
                )
                != 0
                or not isinstance(artifact_jobs, list)
                or artifact_jobs
                != [
                    {
                        "name": manifest_row["name"],
                        "job_id": receipt_row["job_id"],
                        "comment": receipt_row["comment"],
                        "job_name": manifest_row["job_name"],
                        "state": "PENDING",
                        "active": True,
                        "script_sha256": manifest_row[
                            "script_sha256"
                        ],
                        "submit_line_sha256": job["submit_line_sha256"],
                        "submit_line_exact": True,
                        "scontrol_command": receipt_row["script"],
                        "scontrol_requeue": 0,
                        "spooled_script_sha256": manifest_row[
                            "script_sha256"
                        ],
                        "spooled_script_exact_match": True,
                    }
                    for job, manifest_row, receipt_row in zip(
                        artifact_jobs,
                        manifest["jobs"],
                        receipt["jobs"],
                        strict=True,
                    )
                ]
                or [
                    {
                        "name": item.get("name"),
                        "job_id": item.get("job_id"),
                        "comment": item.get("comment"),
                    }
                    for item in artifact_jobs
                    if isinstance(item, Mapping)
                ]
                != expected_namespace
                or any(
                    _SHA256.fullmatch(
                        str(item.get("submit_line_sha256", ""))
                    )
                    is None
                    for item in artifact_jobs
                )
                or summary.get("observed_at_timestamp")
                != artifact.get("observed_at_timestamp")
            ):
                valid_observation_artifacts = False
                break
    valid_drill = (
        isinstance(drill, Mapping)
        and set(drill)
        == {
            "recovery_namespace_cancellation_recovery_seconds",
            "isolated_cancellation_drill",
            "isolation_id",
            "isolated_drill_root",
            "isolated_chain_id",
            "isolated_anchor_submission_receipt_id",
            "canonical_submission_receipt_id",
            "canonical_job_id_overlap",
            "canonical_comment_overlap",
            "canonical_control_paths_absent",
            "squeue_complete",
            "sacct_complete",
            "duplicate_jobs",
            "duplicate_submission_intents",
            "root_remained_held",
            "scientific_jobs_started",
        }
        and isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"),
            (int, float),
        )
        and not isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"), bool
        )
        and math.isfinite(
            float(drill["recovery_namespace_cancellation_recovery_seconds"])
        )
        and 0.0
        <= float(drill["recovery_namespace_cancellation_recovery_seconds"])
        <= 900.0
        and drill.get("isolated_cancellation_drill") is True
        and _SHA256.fullmatch(str(drill.get("isolation_id", ""))) is not None
        and isinstance(drill.get("isolated_drill_root"), str)
        and bool(drill["isolated_drill_root"])
        and _SHA256.fullmatch(str(drill.get("isolated_chain_id", "")))
        is not None
        and _SHA256.fullmatch(
            str(drill.get("isolated_anchor_submission_receipt_id", ""))
        )
        is not None
        and _SHA256.fullmatch(
            str(drill.get("canonical_submission_receipt_id", ""))
        )
        is not None
        and drill.get("isolated_chain_id") != manifest.get("chain_id")
        and drill.get("isolated_anchor_submission_receipt_id")
        != receipt.get("receipt_id")
        and drill.get("canonical_submission_receipt_id")
        == receipt.get("receipt_id")
        and drill.get("canonical_job_id_overlap") == 0
        and drill.get("canonical_comment_overlap") == 0
        and drill.get("canonical_control_paths_absent") is True
        and drill.get("squeue_complete") is True
        and drill.get("sacct_complete") is True
        and drill.get("duplicate_jobs") == 0
        and drill.get("duplicate_submission_intents") == 0
        and drill.get("root_remained_held") is True
        and drill.get("scientific_jobs_started") == 0
    )
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("protocol") != BOOTSTRAP_WATCHDOG_ARMED_PROTOCOL
        or payload.get("passed") is not True
        or payload.get("release_id") != RELEASE_ID
        or payload.get("release_tag") != RELEASE_TAG
        or payload.get("release_git_commit")
        != manifest.get("release_git_commit")
        or payload.get("release_tag_object")
        != manifest.get("release_tag_object")
        or payload.get("chain_namespace") != CHAIN_NAMESPACE
        or payload.get("chain_manifest") != str(manifest_path)
        or payload.get("chain_manifest_sha256") != _sha256(manifest_path)
        or payload.get("chain_id") != manifest.get("chain_id")
        or payload.get("submission_receipt") != str(receipt_path)
        or payload.get("submission_receipt_sha256") != _sha256(receipt_path)
        or payload.get("submission_receipt_id") != receipt.get("receipt_id")
        or payload.get("root_name") != root_record.get("name")
        or payload.get("root_job_id") != root_record.get("job_id")
        or payload.get("root_comment") != root_record.get("comment")
        or namespace != expected_namespace
        or payload.get("job_count") != len(expected_namespace)
        or len(expected_namespace) != len(manifest.get("jobs", []))
        or payload.get("forced_command_only") is not True
        or payload.get("forced_operation") != "bootstrap-repair"
        or payload.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or payload.get("prelaunch_root_release") is not False
        or payload.get("descendant_rearm_authority") is not True
        or payload.get("release_intent_continuation_authority")
        is not True
        or payload.get("scientific_admission_direct") is not False
        or payload.get("production_control_mutation") is not False
        or payload.get("safety_hold_clear_authority") is not False
        or payload.get("handoff_stage") != "watchdog_readiness"
        or not valid_observations
        or not valid_observation_artifacts
        or not valid_drill
        or not isinstance(marker_id, str)
        or _SHA256.fullmatch(marker_id) is None
        or marker_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("bootstrap watchdog armed contract is invalid")
    ready_path = _require_canonical_path(
        Path(str(payload["bootstrap_watchdog_ready"])),
        description="bootstrap watchdog deployment readiness",
        kind="file",
    )
    expected_ready_path = receipt_path.parent / BOOTSTRAP_WATCHDOG_READY_NAME
    ready = _validate_bootstrap_watchdog_deployment_ready(
        ready_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    intent_path = _require_canonical_path(
        Path(str(payload["arm_intent"])),
        description="bootstrap watchdog arm intent",
        kind="file",
    )
    intent = _read_json(
        intent_path, description="bootstrap watchdog arm intent"
    )
    intent_identity = dict(intent)
    arm_intent_id = intent_identity.pop("arm_intent_id", None)
    if (
        ready_path != expected_ready_path
        or payload["bootstrap_watchdog_ready_sha256"]
        != _sha256(ready_path)
        or payload["bootstrap_watchdog_ready_id"] != ready["ready_id"]
        or intent_path
        != receipt_path.parent / BOOTSTRAP_WATCHDOG_ARM_INTENT_NAME
        or payload["arm_intent_sha256"] != _sha256(intent_path)
        or payload["arm_intent_id"] != arm_intent_id
        or set(intent)
        != {
            "schema_version",
            "protocol",
            "chain_id",
            "submission_receipt",
            "submission_receipt_sha256",
            "submission_receipt_id",
            "root_job_id",
            "bootstrap_watchdog_ready",
            "bootstrap_watchdog_ready_sha256",
            "bootstrap_watchdog_ready_id",
            "arm_intent_id",
        }
        or intent.get("schema_version") != 1
        or intent.get("protocol")
        != "schema5-v1.2-r7-bootstrap-watchdog-arm-intent-v1"
        or intent.get("chain_id") != manifest.get("chain_id")
        or intent.get("submission_receipt") != str(receipt_path)
        or intent.get("submission_receipt_sha256") != _sha256(receipt_path)
        or intent.get("submission_receipt_id") != receipt.get("receipt_id")
        or intent.get("root_job_id") != root_record.get("job_id")
        or intent.get("bootstrap_watchdog_ready") != str(ready_path)
        or intent.get("bootstrap_watchdog_ready_sha256")
        != _sha256(ready_path)
        or intent.get("bootstrap_watchdog_ready_id") != ready["ready_id"]
        or not isinstance(arm_intent_id, str)
        or _SHA256.fullmatch(arm_intent_id) is None
        or arm_intent_id != _sha256_bytes(_canonical_json(intent_identity))
    ):
        raise ChainError("bootstrap watchdog arm transaction drifted")
    return payload


def _bootstrap_anchor_armed_authorization(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Return the sealed generation-zero ARMED authority, if fully published."""

    receipt_path = manifest_path.parent / SUBMISSION_RECEIPT_NAME
    armed_path = manifest_path.parent / BOOTSTRAP_WATCHDOG_ARMED_NAME
    if not (armed_path.exists() or armed_path.is_symlink()):
        return None
    receipt = _validate_submission_receipt(
        receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
        comments=_submission_comments(manifest),
    )
    root_record = next(
        row for row in receipt["jobs"] if row["name"] == "source_checkout"
    )
    armed = _validate_bootstrap_watchdog_armed(
        armed_path,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        root_record=root_record,
    )
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "armed": str(armed_path),
        "armed_sha256": _sha256(armed_path),
        "armed_id": armed["marker_id"],
        "ready": armed["bootstrap_watchdog_ready"],
        "ready_sha256": armed["bootstrap_watchdog_ready_sha256"],
        "ready_id": armed["bootstrap_watchdog_ready_id"],
    }


def _bootstrap_anchor_release_intent_authorization(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Authenticate a marker-first g0 release decision without inferring it."""

    intent_path = manifest_path.parent / ROOT_RELEASE_INTENT_NAME
    if not (intent_path.exists() or intent_path.is_symlink()):
        return None
    armed = _bootstrap_anchor_armed_authorization(
        manifest=manifest, manifest_path=manifest_path
    )
    if armed is None:
        raise ChainError(
            "bootstrap root release intent exists without g0 ARMED authority"
        )
    receipt_path = Path(str(armed["receipt"]))
    receipt = _read_json(
        receipt_path, description="bootstrap anchor submission receipt"
    )
    root = next(
        row for row in receipt["jobs"] if row["name"] == "source_checkout"
    )
    intent_path = _require_canonical_path(
        intent_path,
        description="bootstrap anchor root release intent",
        kind="file",
    )
    intent = _read_json(
        intent_path, description="bootstrap anchor root release intent"
    )
    acceptance_path = _require_canonical_path(
        Path(str(intent.get("scheduler_acceptance", ""))),
        description="bootstrap anchor scheduler acceptance",
        kind="file",
    )
    acceptance = _validate_scheduler_acceptance(
        acceptance_path,
        manifest=manifest,
        receipt=receipt,
        receipt_path=receipt_path,
    )
    if (
        intent_path.stat().st_nlink != 1
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
        or intent.get("receipt") != str(receipt_path)
        or intent.get("receipt_sha256") != _sha256(receipt_path)
        or intent.get("receipt_id") != receipt["receipt_id"]
        or intent.get("root_name") != "source_checkout"
        or intent.get("root_job_id") != root["job_id"]
        or intent.get("root_comment") != root["comment"]
        or intent.get("command")
        != ["scontrol", "release", str(root["job_id"])]
        or intent.get("bootstrap_watchdog_armed") != armed["armed"]
        or intent.get("bootstrap_watchdog_armed_sha256")
        != armed["armed_sha256"]
        or intent.get("bootstrap_watchdog_armed_id") != armed["armed_id"]
        or intent.get("scheduler_acceptance") != str(acceptance_path)
        or intent.get("scheduler_acceptance_sha256")
        != _sha256(acceptance_path)
        or intent.get("scheduler_acceptance_id")
        != acceptance["acceptance_id"]
    ):
        raise ChainError(
            "bootstrap anchor root release intent is not an exact "
            "acceptance-bound decision"
        )
    return {
        "intent": str(intent_path),
        "intent_sha256": _sha256(intent_path),
        "receipt_id": receipt["receipt_id"],
        "root_job_id": root["job_id"],
        "scheduler_acceptance_id": acceptance["acceptance_id"],
    }


def _generation_root_records(
    receipt: Mapping[str, Any], *, generation: int
) -> list[Mapping[str, Any]]:
    """Return the exact minimal held frontier recorded for one generation."""

    names = (
        ["source_checkout"]
        if generation == 0
        else list(receipt.get("held_root_names", []))
    )
    if (
        not names
        or len(names) != len(set(names))
        or any(not isinstance(name, str) for name in names)
    ):
        raise ChainError("recovery receipt has no exact held frontier")
    by_name = {
        str(record.get("name")): record
        for record in receipt.get("jobs", [])
        if isinstance(record, Mapping)
    }
    records: list[Mapping[str, Any]] = []
    for name in names:
        record = by_name.get(name)
        if (
            not isinstance(record, Mapping)
            or not str(record.get("job_id", "")).isdigit()
            or not isinstance(record.get("comment"), str)
            or (
                generation > 0
                and (
                    record.get("generation") != generation
                    or record.get("disposition") != "resubmitted"
                )
            )
        ):
            raise ChainError(
                f"recovery receipt held frontier drifted for {name}"
            )
        records.append(record)
    return records


def _validate_bootstrap_descendant_armed(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
    root_record: Mapping[str, Any],
) -> dict[str, Any]:
    path = _require_canonical_path(
        path,
        description="bootstrap descendant armed marker",
        kind="file",
    )
    payload = _read_json(
        path, description="bootstrap descendant armed marker"
    )
    identity = dict(payload)
    marker_id = identity.pop("marker_id", None)
    anchor = _bootstrap_anchor_armed_authorization(
        manifest=manifest, manifest_path=manifest_path
    )
    provenance_path = _bootstrap_generation_provenance_path(
        manifest_path=manifest_path, generation=generation
    )
    provenance = _validate_bootstrap_generation_provenance(
        provenance_path,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
    )
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    if (
        not root_records
        or root_records[0]["name"] != root_record.get("name")
        or root_records[0]["job_id"] != root_record.get("job_id")
    ):
        raise ChainError(
            "bootstrap descendant arm was given the wrong frontier root"
        )
    root_names = [str(record["name"]) for record in root_records]
    root_job_ids = [str(record["job_id"]) for record in root_records]
    root_comments = [str(record["comment"]) for record in root_records]
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "repair_generation",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "parent_submission_receipt",
        "parent_submission_receipt_sha256",
        "root_name",
        "root_job_id",
        "root_comment",
        "root_names",
        "root_job_ids",
        "root_comments",
        "anchor_bootstrap_armed",
        "anchor_bootstrap_armed_sha256",
        "anchor_bootstrap_armed_id",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "scheduler_observation_artifacts",
        "arm_intent",
        "arm_intent_sha256",
        "arm_intent_id",
        "release_authority",
        "marker_id",
    }
    parent_path = _require_canonical_path(
        Path(str(receipt["parent_receipt"])),
        description="bootstrap descendant parent receipt",
        kind="file",
    )
    observation_paths = payload.get("scheduler_observation_artifacts")
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("protocol") != BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL
        or payload.get("passed") is not True
        or payload.get("chain_id") != manifest["chain_id"]
        or payload.get("chain_manifest") != str(manifest_path)
        or payload.get("chain_manifest_sha256") != _sha256(manifest_path)
        or payload.get("repair_generation") != generation
        or payload.get("submission_receipt") != str(receipt_path)
        or payload.get("submission_receipt_sha256") != _sha256(receipt_path)
        or payload.get("submission_receipt_id") != receipt["receipt_id"]
        or payload.get("parent_submission_receipt") != str(parent_path)
        or payload.get("parent_submission_receipt_sha256")
        != _sha256(parent_path)
        or payload.get("root_name") != root_record["name"]
        or payload.get("root_job_id") != root_record["job_id"]
        or payload.get("root_comment") != root_record["comment"]
        or payload.get("root_names") != root_names
        or payload.get("root_job_ids") != root_job_ids
        or payload.get("root_comments") != root_comments
        or anchor is None
        or payload.get("anchor_bootstrap_armed") != anchor["armed"]
        or payload.get("anchor_bootstrap_armed_sha256")
        != anchor["armed_sha256"]
        or payload.get("anchor_bootstrap_armed_id") != anchor["armed_id"]
        or payload.get("generation_provenance") != str(provenance_path)
        or payload.get("generation_provenance_sha256")
        != _sha256(provenance_path)
        or payload.get("generation_provenance_id")
        != provenance["provenance_id"]
        or not isinstance(observation_paths, list)
        or len(observation_paths) != 2
        or payload.get("release_authority")
        != "rearm-only-without-release-intent"
        or not isinstance(marker_id, str)
        or _SHA256.fullmatch(marker_id) is None
        or marker_id != _sha256_bytes(_canonical_json(identity))
        or path.stat().st_nlink != 1
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise ChainError("bootstrap descendant armed contract is invalid")
    intent_path = _require_canonical_path(
        Path(str(payload["arm_intent"])),
        description="bootstrap descendant arm intent",
        kind="file",
    )
    intent = _read_json(
        intent_path, description="bootstrap descendant arm intent"
    )
    intent_identity = dict(intent)
    intent_id = intent_identity.pop("arm_intent_id", None)
    if (
        intent_path.parent != receipt_path.parent
        or intent_path.name != BOOTSTRAP_DESCENDANT_ARM_INTENT_NAME
        or payload["arm_intent_sha256"] != _sha256(intent_path)
        or payload["arm_intent_id"] != intent_id
        or intent.get("submission_receipt_id") != receipt["receipt_id"]
        or intent.get("generation_provenance_id")
        != provenance["provenance_id"]
        or intent.get("anchor_bootstrap_armed_id") != anchor["armed_id"]
        or intent.get("root_job_id") != root_record["job_id"]
        or intent.get("root_names") != root_names
        or intent.get("root_job_ids") != root_job_ids
        or intent.get("root_comments") != root_comments
        or intent.get("scheduler_observation_artifacts")
        != observation_paths
        or intent_id != _sha256_bytes(_canonical_json(intent_identity))
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
    ):
        raise ChainError("bootstrap descendant arm transaction drifted")
    for artifact in observation_paths:
        artifact_path = _require_canonical_path(
            Path(str(artifact)),
            description="bootstrap descendant arm observation",
            kind="file",
        )
        observation = _read_json(
            artifact_path,
            description="bootstrap descendant arm observation",
        )
        if (
            artifact_path.parent
            != receipt_path.parent / BOOTSTRAP_OBSERVATIONS_ROOT_NAME
            or observation.get("submission_receipt_id")
            != receipt["receipt_id"]
            or observation.get("generation_provenance_id")
            != provenance["provenance_id"]
            or observation.get("root_job_id") != root_record["job_id"]
            or observation.get("root_names") != root_names
            or observation.get("root_job_ids") != root_job_ids
            or observation.get("roots_held") is not True
            or observation.get("root_held") is not True
            or observation.get("descendant_rearm_required") is not True
            or observation.get("observation_id")
            != _sha256_bytes(
                _canonical_json(
                    {
                        key: value
                        for key, value in observation.items()
                        if key != "observation_id"
                    }
                )
            )
            or stat.S_IMODE(artifact_path.stat().st_mode) & 0o222
        ):
            raise ChainError(
                "bootstrap descendant arm observation drifted"
            )
    return payload


def _ensure_bootstrap_descendant_armed(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
    root_record: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path]:
    if generation <= 0:
        raise ChainError("only a repair generation can be descendant-armed")
    path = receipt_path.parent / BOOTSTRAP_DESCENDANT_ARMED_NAME
    if path.exists() or path.is_symlink():
        return (
            _validate_bootstrap_descendant_armed(
                path,
                manifest=manifest,
                manifest_path=manifest_path,
                receipt=receipt,
                receipt_path=receipt_path,
                generation=generation,
                root_record=root_record,
            ),
            path,
        )
    anchor = _bootstrap_anchor_armed_authorization(
        manifest=manifest, manifest_path=manifest_path
    )
    if anchor is None:
        raise ChainError(
            "bootstrap descendant re-arm lacks sealed g0 ARMED authority"
        )
    provenance_path = _bootstrap_generation_provenance_path(
        manifest_path=manifest_path, generation=generation
    )
    provenance = _validate_bootstrap_generation_provenance(
        provenance_path,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
    )
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    if (
        root_records[0]["name"] != root_record.get("name")
        or root_records[0]["job_id"] != root_record.get("job_id")
    ):
        raise ChainError(
            "bootstrap descendant arm was given the wrong frontier root"
        )
    root_names = [str(record["name"]) for record in root_records]
    root_job_ids = [str(record["job_id"]) for record in root_records]
    root_comments = [str(record["comment"]) for record in root_records]
    artifacts = [str(row["observation_artifact"]) for row in observations]
    parent_path = _require_canonical_path(
        Path(str(receipt["parent_receipt"])),
        description="bootstrap descendant parent receipt",
        kind="file",
    )
    intent: dict[str, Any] = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r7-bootstrap-descendant-arm-intent-v1"
        ),
        "chain_id": manifest["chain_id"],
        "repair_generation": generation,
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "root_job_id": root_record["job_id"],
        "root_names": root_names,
        "root_job_ids": root_job_ids,
        "root_comments": root_comments,
        "anchor_bootstrap_armed": anchor["armed"],
        "anchor_bootstrap_armed_sha256": anchor["armed_sha256"],
        "anchor_bootstrap_armed_id": anchor["armed_id"],
        "generation_provenance": str(provenance_path),
        "generation_provenance_sha256": _sha256(provenance_path),
        "generation_provenance_id": provenance["provenance_id"],
        "scheduler_observation_artifacts": artifacts,
        "release_authority": "rearm-only-without-release-intent",
    }
    intent["arm_intent_id"] = _sha256_bytes(_canonical_json(intent))
    intent_path = receipt_path.parent / BOOTSTRAP_DESCENDANT_ARM_INTENT_NAME
    _write_immutable_json_once(
        intent_path,
        intent,
        description="marker-first bootstrap descendant arm intent",
    )
    armed: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_DESCENDANT_ARMED_PROTOCOL,
        "passed": True,
        "chain_id": manifest["chain_id"],
        "chain_manifest": str(manifest_path),
        "chain_manifest_sha256": _sha256(manifest_path),
        "repair_generation": generation,
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "parent_submission_receipt": str(parent_path),
        "parent_submission_receipt_sha256": _sha256(parent_path),
        "root_name": root_record["name"],
        "root_job_id": root_record["job_id"],
        "root_comment": root_record["comment"],
        "root_names": root_names,
        "root_job_ids": root_job_ids,
        "root_comments": root_comments,
        "anchor_bootstrap_armed": anchor["armed"],
        "anchor_bootstrap_armed_sha256": anchor["armed_sha256"],
        "anchor_bootstrap_armed_id": anchor["armed_id"],
        "generation_provenance": str(provenance_path),
        "generation_provenance_sha256": _sha256(provenance_path),
        "generation_provenance_id": provenance["provenance_id"],
        "scheduler_observation_artifacts": artifacts,
        "arm_intent": str(intent_path),
        "arm_intent_sha256": _sha256(intent_path),
        "arm_intent_id": intent["arm_intent_id"],
        "release_authority": "rearm-only-without-release-intent",
    }
    armed["marker_id"] = _sha256_bytes(_canonical_json(armed))
    _write_immutable_json_once(
        path,
        armed,
        description="marker-last bootstrap descendant armed marker",
    )
    return (
        _validate_bootstrap_descendant_armed(
            path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            root_record=root_record,
        ),
        path,
    )


def _validate_root_release_attempt_result(
    path: Path,
    *,
    intent_path: Path,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable result for one marker-first release attempt."""

    path = _require_canonical_path(
        path,
        description="recovery root release attempt result",
        kind="file",
    )
    metadata = path.stat()
    if metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise ChainError(
            "recovery root release attempt result must be sealed"
        )
    payload = _read_json(
        path, description="recovery root release attempt result"
    )
    identity = dict(payload)
    result_id = identity.pop("result_id", None)
    required = {
        "schema_version",
        "protocol",
        "attempt",
        "completed_at",
        "root_index",
        "root_name",
        "job_id",
        "intent",
        "intent_sha256",
        "result_kind",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
        "scheduler_reconciliation",
        "result_id",
    }
    result_kind = payload.get("result_kind")
    reconciliation = payload.get("scheduler_reconciliation")
    valid_direct = (
        result_kind == "scontrol_result"
        and isinstance(payload.get("returncode"), int)
        and not isinstance(payload.get("returncode"), bool)
        and isinstance(payload.get("stdout"), str)
        and isinstance(payload.get("stderr"), str)
        and payload.get("stdout_sha256")
        == _sha256_bytes(payload["stdout"].encode("utf-8"))
        and payload.get("stderr_sha256")
        == _sha256_bytes(payload["stderr"].encode("utf-8"))
        and reconciliation is None
    )
    valid_reconciliation = (
        result_kind
        in {
            "scheduler_reconciled_after_ambiguous_release",
            "scheduler_reconciled_no_effect",
        }
        and payload.get("returncode") is None
        and payload.get("stdout") is None
        and payload.get("stderr") is None
        and payload.get("stdout_sha256") is None
        and payload.get("stderr_sha256") is None
        and isinstance(reconciliation, Mapping)
        and reconciliation.get("returncode") == 0
        and str(reconciliation.get("fields", {}).get("JobId", ""))
        == str(intent.get("job_id", ""))
        and (
            (
                result_kind
                == "scheduler_reconciled_after_ambiguous_release"
                and not (
                    _normalize_slurm_state(
                        str(
                            reconciliation.get("fields", {}).get(
                                "JobState", ""
                            )
                        )
                    )
                    == "PENDING"
                    and str(
                        reconciliation.get("fields", {}).get(
                            "Reason", ""
                        )
                    ).lower()
                    == "jobhelduser"
                )
            )
            or (
                result_kind == "scheduler_reconciled_no_effect"
                and _normalize_slurm_state(
                    str(
                        reconciliation.get("fields", {}).get(
                            "JobState", ""
                        )
                    )
                )
                == "PENDING"
                and str(
                    reconciliation.get("fields", {}).get("Reason", "")
                ).lower()
                == "jobhelduser"
            )
        )
    )
    if (
        set(payload) != required
        or payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or payload.get("protocol")
        != "schema5-v1.2-r7-root-release-attempt-result-v1"
        or payload.get("attempt") != intent.get("attempt")
        or payload.get("root_index") != intent.get("root_index")
        or payload.get("root_name") != intent.get("root_name")
        or payload.get("job_id") != intent.get("job_id")
        or payload.get("intent") != str(intent_path)
        or payload.get("intent_sha256") != _sha256(intent_path)
        or not isinstance(payload.get("completed_at"), str)
        or not (valid_direct or valid_reconciliation)
        or not isinstance(result_id, str)
        or _SHA256.fullmatch(result_id) is None
        or result_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError(
            "recovery root release attempt result drifted"
        )
    return payload


def _validate_root_release_complete(
    path: Path,
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    root_record: Mapping[str, Any],
) -> dict[str, Any]:
    path = _require_canonical_path(
        path, description="recovery root release completion", kind="file"
    )
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError("recovery root release completion must be read-only")
    payload = _read_json(path, description="recovery root release completion")
    identity = dict(payload)
    release_id = identity.pop("release_id", None)
    generation = int(receipt.get("repair_generation", 0))
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    if (
        root_records[0]["name"] != root_record.get("name")
        or root_records[0]["job_id"] != root_record.get("job_id")
    ):
        raise ChainError(
            "recovery root release was given the wrong frontier root"
        )
    root_names = [str(record["name"]) for record in root_records]
    root_job_ids = [str(record["job_id"]) for record in root_records]
    root_comments = [str(record["comment"]) for record in root_records]
    required = {
        "schema_version",
        "protocol",
        "completed_at",
        "receipt",
        "receipt_sha256",
        "receipt_id",
        "root_name",
        "root_job_id",
        "root_comment",
        "root_names",
        "root_job_ids",
        "root_comments",
        "release_intent",
        "release_intent_sha256",
        "dependency_policy_check",
        "dependency_policy_check_sha256",
        "dependency_canary",
        "state_before",
        "state_after",
        "scheduler_observation",
        "roots_state_before",
        "roots_state_after",
        "scheduler_observations",
        "release_attempts",
        "release_reconciled",
        "root_no_longer_held",
        "all_roots_no_longer_held",
        "release_id",
    }
    initial_submission = (
        receipt.get("protocol")
        == "schema5-v1.2-r7-recovery-chain-submission"
    )
    if initial_submission:
        required |= {
            "scheduler_acceptance",
            "scheduler_acceptance_sha256",
            "scheduler_acceptance_id",
            "bootstrap_watchdog_ready",
            "bootstrap_watchdog_ready_sha256",
            "bootstrap_watchdog_ready_id",
            "bootstrap_watchdog_armed",
            "bootstrap_watchdog_armed_sha256",
            "bootstrap_watchdog_armed_id",
        }
    else:
        required |= {
            "generation_scheduler_acceptance",
            "generation_scheduler_acceptance_sha256",
            "generation_scheduler_acceptance_id",
        }
        if "bootstrap_descendant_armed" in payload:
            required |= {
                "bootstrap_descendant_armed",
                "bootstrap_descendant_armed_sha256",
                "bootstrap_descendant_armed_id",
            }
    if (
        set(payload) != required
        or payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or payload.get("protocol")
        != "schema5-v1.2-r7-recovery-root-release-v1"
        or not isinstance(payload.get("completed_at"), str)
        or payload.get("receipt") != str(receipt_path)
        or payload.get("receipt_sha256") != _sha256(receipt_path)
        or payload.get("receipt_id") != receipt.get("receipt_id")
        or payload.get("root_name") != root_record.get("name")
        or payload.get("root_job_id") != root_record.get("job_id")
        or payload.get("root_comment") != root_record.get("comment")
        or payload.get("root_names") != root_names
        or payload.get("root_job_ids") != root_job_ids
        or payload.get("root_comments") != root_comments
        or payload.get("dependency_canary")
        != receipt.get("dependency_canary")
        or payload.get("root_no_longer_held") is not True
        or payload.get("all_roots_no_longer_held") is not True
        or not isinstance(payload.get("roots_state_before"), list)
        or len(payload["roots_state_before"]) != len(root_records)
        or not isinstance(payload.get("roots_state_after"), list)
        or len(payload["roots_state_after"]) != len(root_records)
        or not isinstance(payload.get("scheduler_observations"), list)
        or len(payload["scheduler_observations"]) != len(root_records)
        or payload.get("state_before") != payload["roots_state_before"][0]
        or payload.get("state_after") != payload["roots_state_after"][0]
        or payload.get("scheduler_observation")
        != payload["scheduler_observations"][0]
        or not isinstance(payload.get("release_attempts"), list)
        or not payload["release_attempts"]
        or not isinstance(release_id, str)
        or _SHA256.fullmatch(release_id) is None
        or release_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("recovery root release completion is invalid")
    intent_path = _require_canonical_path(
        Path(payload["release_intent"]),
        description="recovery root release intent",
        kind="file",
    )
    policy_path = _require_canonical_path(
        Path(payload["dependency_policy_check"]),
        description="root-release dependency-policy evidence",
        kind="file",
    )
    if (
        _sha256(intent_path) != payload["release_intent_sha256"]
        or _sha256(policy_path)
        != payload["dependency_policy_check_sha256"]
    ):
        raise ChainError("recovery root release evidence hashes drifted")
    intent = _read_json(
        intent_path, description="recovery root release intent"
    )
    if (
        intent.get("receipt_id") != receipt.get("receipt_id")
        or intent.get("root_name") != root_names[0]
        or intent.get("root_job_id") != root_job_ids[0]
        or intent.get("root_comment") != root_comments[0]
        or intent.get("root_names") != root_names
        or intent.get("root_job_ids") != root_job_ids
        or intent.get("root_comments") != root_comments
        or intent.get("command")
        != ["scontrol", "release", root_job_ids[0]]
        or intent.get("commands")
        != [
            ["scontrol", "release", job_id]
            for job_id in root_job_ids
        ]
    ):
        raise ChainError(
            "recovery root release intent frontier drifted"
        )
    covered_roots: set[str] = set()
    successful_roots: set[str] = set()
    seen_attempt_paths: set[Path] = set()
    reconciled_observations: dict[str, Mapping[str, Any]] = {}
    successful_attempt_by_root: dict[str, int] = {}
    prior_root_index = -1
    attempts_root = path.parent / "root_release_attempts"
    attempts_root = _require_canonical_path(
        attempts_root,
        description="recovery root release attempts",
        kind="directory",
    )
    expected_intent_paths: list[Path] = []
    expected_result_paths: list[Path] = []
    for ordinal, binding in enumerate(
        payload["release_attempts"], start=1
    ):
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "attempt",
                "root_name",
                "job_id",
                "intent",
                "intent_sha256",
                "result",
                "result_sha256",
                "result_id",
                "result_kind",
            }
        ):
            raise ChainError(
                "recovery root release attempt binding drifted"
            )
        attempt_path = _require_canonical_path(
            Path(str(binding["intent"])),
            description="recovery root release attempt intent",
            kind="file",
        )
        result_path = _require_canonical_path(
            Path(str(binding["result"])),
            description="recovery root release attempt result",
            kind="file",
        )
        attempt_metadata = attempt_path.stat()
        attempt = _read_json(
            attempt_path,
            description="recovery root release attempt intent",
        )
        name = str(attempt.get("root_name", ""))
        if name not in root_names:
            raise ChainError(
                "recovery root release attempt names a foreign root"
            )
        root_index = root_names.index(name)
        expected_result_path = attempts_root / (
            f"attempt-{ordinal:04d}.result.json"
        )
        if (
            attempt_path in seen_attempt_paths
            or attempt_path.parent != attempts_root
            or result_path != expected_result_path
            or attempt_metadata.st_nlink != 1
            or stat.S_IMODE(attempt_metadata.st_mode) & 0o222
            or attempt_path.name
            != f"attempt-{ordinal:04d}.intent.json"
            or attempt.get("schema_version")
            != SUBMISSION_SCHEMA_VERSION
            or attempt.get("protocol")
            != "schema5-v1.2-r7-root-release-attempt-intent-v1"
            or attempt.get("attempt") != ordinal
            or attempt.get("root_index") != root_index
            or attempt.get("job_id") != root_job_ids[root_index]
            or attempt.get("command")
            != ["scontrol", "release", root_job_ids[root_index]]
            or attempt.get("dependency_policy_check")
            != str(policy_path)
            or attempt.get("dependency_policy_check_sha256")
            != _sha256(policy_path)
            or binding.get("attempt") != ordinal
            or binding.get("root_name") != name
            or binding.get("job_id") != attempt.get("job_id")
            or binding.get("intent_sha256") != _sha256(attempt_path)
            or binding.get("result_sha256") != _sha256(result_path)
        ):
            raise ChainError(
                "recovery root release attempt intent drifted"
            )
        if (
            root_index < prior_root_index
            or name in successful_attempt_by_root
        ):
            raise ChainError(
                "recovery root release attempts are not an exact "
                "frontier-ordered transaction"
            )
        prior_root_index = root_index
        seen_attempt_paths.add(attempt_path)
        expected_intent_paths.append(attempt_path)
        expected_result_paths.append(result_path)
        result = _validate_root_release_attempt_result(
            result_path,
            intent_path=attempt_path,
            intent=attempt,
        )
        if (
            binding.get("result_id") != result["result_id"]
            or binding.get("result_kind") != result["result_kind"]
        ):
            raise ChainError(
                "recovery root release attempt result binding drifted"
            )
        covered_roots.add(name)
        if (
            result["result_kind"]
            == "scheduler_reconciled_after_ambiguous_release"
        ):
            if name in reconciled_observations:
                raise ChainError(
                    "a frontier root has multiple ambiguous adoptions"
                )
            reconciled_observations[name] = result[
                "scheduler_reconciliation"
            ]
            successful_roots.add(name)
            successful_attempt_by_root[name] = ordinal
        elif result["returncode"] == 0:
            successful_roots.add(name)
            successful_attempt_by_root[name] = ordinal
    actual_intent_paths = sorted(
        attempts_root.glob("attempt-*.intent.json")
    )
    actual_result_paths = sorted(
        attempts_root.glob("attempt-*.result.json")
    )
    actual_attempt_members = sorted(attempts_root.iterdir())
    if (
        covered_roots != set(root_names)
        or successful_roots != set(root_names)
        or set(successful_attempt_by_root) != set(root_names)
        or actual_intent_paths != expected_intent_paths
        or actual_result_paths != expected_result_paths
        or actual_attempt_members
        != sorted(expected_intent_paths + expected_result_paths)
    ):
        raise ChainError(
            "recovery root release attempts do not exactly cover and "
            "successfully resolve every frontier root"
        )
    for index, (record, before, after, observation) in enumerate(
        zip(
            root_records,
            payload["roots_state_before"],
            payload["roots_state_after"],
            payload["scheduler_observations"],
            strict=True,
        )
    ):
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or not isinstance(observation, Mapping)
            or observation.get("job_id") != record["job_id"]
            or (
                str(record["name"]) in reconciled_observations
                and reconciled_observations[str(record["name"])]
                != before
            )
            or (
                after.get("returncode") == 0
                and _normalize_slurm_state(
                    str(after.get("fields", {}).get("JobState", ""))
                )
                == "PENDING"
                and str(
                    after.get("fields", {}).get("Reason", "")
                ).lower()
                == "jobhelduser"
            )
            or index >= len(root_names)
        ):
            raise ChainError(
                "recovery root release frontier evidence drifted"
            )
    _validate_dependency_policy_check(
        policy_path, expected_phase="root_release"
    )
    if initial_submission:
        manifest_path = Path(str(receipt["manifest"]))
        manifest = _read_json(
            manifest_path,
            description="bootstrap-watchdog chain manifest",
        )
        bootstrap_path = _require_canonical_path(
            Path(str(payload["bootstrap_watchdog_armed"])),
            description="bootstrap watchdog armed marker",
            kind="file",
        )
        bootstrap = _validate_bootstrap_watchdog_armed(
            bootstrap_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            root_record=root_record,
        )
        if (
            _sha256(bootstrap_path)
            != payload["bootstrap_watchdog_armed_sha256"]
            or bootstrap["marker_id"]
            != payload["bootstrap_watchdog_armed_id"]
            or bootstrap["bootstrap_watchdog_ready"]
            != payload["bootstrap_watchdog_ready"]
            or bootstrap["bootstrap_watchdog_ready_sha256"]
            != payload["bootstrap_watchdog_ready_sha256"]
            or bootstrap["bootstrap_watchdog_ready_id"]
            != payload["bootstrap_watchdog_ready_id"]
        ):
            raise ChainError(
                "recovery root release bootstrap-watchdog binding drifted"
            )
        acceptance_path = _require_canonical_path(
            Path(str(payload["scheduler_acceptance"])),
            description="scheduler acceptance evidence",
            kind="file",
        )
        acceptance = _validate_scheduler_acceptance(
            acceptance_path,
            manifest=manifest,
            receipt=receipt,
            receipt_path=receipt_path,
        )
        if (
            _sha256(acceptance_path)
            != payload["scheduler_acceptance_sha256"]
            or acceptance["acceptance_id"]
            != payload["scheduler_acceptance_id"]
        ):
            raise ChainError(
                "recovery root release scheduler acceptance binding drifted"
            )
    else:
        generation = int(receipt.get("repair_generation", 0))
        manifest_path = Path(str(receipt["manifest"]))
        provenance_path = _require_canonical_path(
            Path(str(payload["generation_scheduler_acceptance"])),
            description="repair generation scheduler acceptance",
            kind="file",
        )
        provenance = _validate_bootstrap_generation_provenance(
            provenance_path,
            manifest=_read_json(
                manifest_path, description="repair chain manifest"
            ),
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
        )
        if (
            payload["generation_scheduler_acceptance_sha256"]
            != _sha256(provenance_path)
            or payload["generation_scheduler_acceptance_id"]
            != provenance["provenance_id"]
        ):
            raise ChainError(
                "repair root release generation acceptance drifted"
            )
        if "bootstrap_descendant_armed" in payload:
            descendant_path = _require_canonical_path(
                Path(str(payload["bootstrap_descendant_armed"])),
                description="repair bootstrap descendant armed marker",
                kind="file",
            )
            descendant = _validate_bootstrap_descendant_armed(
                descendant_path,
                manifest=_read_json(
                    manifest_path, description="repair chain manifest"
                ),
                manifest_path=manifest_path,
                receipt=receipt,
                receipt_path=receipt_path,
                generation=generation,
                root_record=root_record,
            )
            if (
                payload["bootstrap_descendant_armed_sha256"]
                != _sha256(descendant_path)
                or payload["bootstrap_descendant_armed_id"]
                != descendant["marker_id"]
            ):
                raise ChainError(
                    "repair root release descendant arm binding drifted"
                )
    return payload


def _validate_launch_complete(
    path: Path,
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    release_path: Path,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    path = _require_canonical_path(
        path, description="recovery-chain launch completion", kind="file"
    )
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise ChainError("recovery-chain launch completion must be read-only")
    payload = _read_json(path, description="recovery-chain launch completion")
    identity = dict(payload)
    launch_id = identity.pop("launch_id", None)
    required = {
            "schema_version",
            "protocol",
            "completed_at",
            "receipt",
            "receipt_sha256",
            "receipt_id",
            "root_release",
            "root_release_sha256",
            "root_release_id",
            "dependency_policy",
            "dependency_canary",
            "root_initial_hold",
            "alert_latency_bound_seconds",
            "launch_id",
        }
    if receipt.get("protocol") == "schema5-v1.2-r7-recovery-chain-submission":
        required |= {
            "bootstrap_watchdog_ready",
            "bootstrap_watchdog_ready_sha256",
            "bootstrap_watchdog_ready_id",
            "bootstrap_watchdog_armed",
            "bootstrap_watchdog_armed_sha256",
            "bootstrap_watchdog_armed_id",
        }
    else:
        required |= {
            "generation_scheduler_acceptance",
            "generation_scheduler_acceptance_sha256",
            "generation_scheduler_acceptance_id",
        }
        if "bootstrap_descendant_armed" in release:
            required |= {
                "bootstrap_descendant_armed",
                "bootstrap_descendant_armed_sha256",
                "bootstrap_descendant_armed_id",
            }
    if (
        set(payload)
        != required
        or payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or payload.get("protocol")
        != "schema5-v1.2-r7-recovery-chain-launched-v1"
        or payload.get("receipt") != str(receipt_path)
        or payload.get("receipt_sha256") != _sha256(receipt_path)
        or payload.get("receipt_id") != receipt.get("receipt_id")
        or payload.get("root_release") != str(release_path)
        or payload.get("root_release_sha256") != _sha256(release_path)
        or payload.get("root_release_id") != release.get("release_id")
        or payload.get("dependency_policy") != DEPENDENCY_POLICY_CONTRACT
        or payload.get("dependency_canary")
        != receipt.get("dependency_canary")
        or payload.get("root_initial_hold") is not True
        or payload.get("alert_latency_bound_seconds") != 180.0
        or (
            "bootstrap_watchdog_ready" in required
            and (
                payload.get("bootstrap_watchdog_ready")
                != release.get("bootstrap_watchdog_ready")
                or payload.get("bootstrap_watchdog_ready_sha256")
                != release.get("bootstrap_watchdog_ready_sha256")
                or payload.get("bootstrap_watchdog_ready_id")
                != release.get("bootstrap_watchdog_ready_id")
                or payload.get("bootstrap_watchdog_armed")
                != release.get("bootstrap_watchdog_armed")
                or payload.get("bootstrap_watchdog_armed_sha256")
                != release.get("bootstrap_watchdog_armed_sha256")
                or payload.get("bootstrap_watchdog_armed_id")
                != release.get("bootstrap_watchdog_armed_id")
            )
        )
        or any(
            payload.get(field) != release.get(field)
            for field in (
                "generation_scheduler_acceptance",
                "generation_scheduler_acceptance_sha256",
                "generation_scheduler_acceptance_id",
                "bootstrap_descendant_armed",
                "bootstrap_descendant_armed_sha256",
                "bootstrap_descendant_armed_id",
            )
            if field in required
        )
        or not isinstance(launch_id, str)
        or _SHA256.fullmatch(launch_id) is None
        or launch_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError("recovery-chain launch completion is invalid")
    return payload


def _ensure_recovery_root_released(
    *,
    evidence_root: Path,
    manifest: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
    journal: Mapping[str, Any],
    root_name: str,
    runner: Runner,
    timestamp: float,
    bootstrap_watchdog_armed_path: Path | None = None,
    bootstrap_descendant_armed_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = int(receipt.get("repair_generation", 0))
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    root_record = root_records[0]
    if (
        root_name != root_record["name"]
        or receipt.get("root_initial_hold") is not True
    ):
        raise ChainError(
            "recovery release target is not the receipt's exact held frontier"
        )
    root_names = [str(record["name"]) for record in root_records]
    root_job_ids = [str(record["job_id"]) for record in root_records]
    root_comments = [str(record["comment"]) for record in root_records]
    release_path = evidence_root / ROOT_RELEASE_COMPLETE_NAME
    launch_path = evidence_root / LAUNCH_COMPLETE_NAME
    initial_submission = (
        receipt.get("protocol")
        == "schema5-v1.2-r7-recovery-chain-submission"
    )
    bootstrap_binding: dict[str, Any] = {}
    if initial_submission:
        expected_bootstrap_path = (
            evidence_root / BOOTSTRAP_WATCHDOG_ARMED_NAME
        )
        if bootstrap_watchdog_armed_path is None:
            bootstrap_watchdog_armed_path = expected_bootstrap_path
        bootstrap_watchdog_armed_path = _lexical_absolute(
            bootstrap_watchdog_armed_path
        )
        if bootstrap_watchdog_armed_path != expected_bootstrap_path:
            raise ChainError(
                "bootstrap watchdog armed marker must be receipt-local"
            )
        bootstrap = _validate_bootstrap_watchdog_armed(
            bootstrap_watchdog_armed_path,
            manifest=manifest,
            manifest_path=Path(str(receipt["manifest"])),
            receipt=receipt,
            receipt_path=receipt_path,
            root_record=root_record,
        )
        bootstrap_binding = {
            "bootstrap_watchdog_ready": bootstrap[
                "bootstrap_watchdog_ready"
            ],
            "bootstrap_watchdog_ready_sha256": bootstrap[
                "bootstrap_watchdog_ready_sha256"
            ],
            "bootstrap_watchdog_ready_id": bootstrap[
                "bootstrap_watchdog_ready_id"
            ],
            "bootstrap_watchdog_armed": str(
                bootstrap_watchdog_armed_path
            ),
            "bootstrap_watchdog_armed_sha256": _sha256(
                bootstrap_watchdog_armed_path
            ),
            "bootstrap_watchdog_armed_id": bootstrap["marker_id"],
        }
    elif bootstrap_descendant_armed_path is not None:
        if generation <= 0:
            raise ChainError("repair release lacks its exact generation")
        expected_descendant_path = (
            evidence_root / BOOTSTRAP_DESCENDANT_ARMED_NAME
        )
        if bootstrap_descendant_armed_path is None:
            bootstrap_descendant_armed_path = expected_descendant_path
        bootstrap_descendant_armed_path = _lexical_absolute(
            bootstrap_descendant_armed_path
        )
        if bootstrap_descendant_armed_path != expected_descendant_path:
            raise ChainError(
                "bootstrap descendant armed marker must be receipt-local"
            )
        descendant = _validate_bootstrap_descendant_armed(
            bootstrap_descendant_armed_path,
            manifest=manifest,
            manifest_path=Path(str(receipt["manifest"])),
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            root_record=root_record,
        )
        provenance_path = _bootstrap_generation_provenance_path(
            manifest_path=Path(str(receipt["manifest"])),
            generation=generation,
        )
        bootstrap_binding = {
            "bootstrap_descendant_armed": str(
                bootstrap_descendant_armed_path
            ),
            "bootstrap_descendant_armed_sha256": _sha256(
                bootstrap_descendant_armed_path
            ),
            "bootstrap_descendant_armed_id": descendant["marker_id"],
            "generation_scheduler_acceptance": str(provenance_path),
            "generation_scheduler_acceptance_sha256": _sha256(
                provenance_path
            ),
            "generation_scheduler_acceptance_id": descendant[
                "generation_provenance_id"
            ],
        }
    else:
        if generation <= 0:
            raise ChainError("repair release lacks its exact generation")
        provenance_path = _bootstrap_generation_provenance_path(
            manifest_path=Path(str(receipt["manifest"])),
            generation=generation,
        )
        provenance = _validate_bootstrap_generation_provenance(
            provenance_path,
            manifest=manifest,
            manifest_path=Path(str(receipt["manifest"])),
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
        )
        bootstrap_binding = {
            "generation_scheduler_acceptance": str(provenance_path),
            "generation_scheduler_acceptance_sha256": _sha256(
                provenance_path
            ),
            "generation_scheduler_acceptance_id": provenance[
                "provenance_id"
            ],
        }

    def publish_launch(release: Mapping[str, Any]) -> dict[str, Any]:
        launch = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "protocol": "schema5-v1.2-r7-recovery-chain-launched-v1",
            "completed_at": _utc_now(),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "receipt_id": receipt["receipt_id"],
            "root_release": str(release_path),
            "root_release_sha256": _sha256(release_path),
            "root_release_id": release["release_id"],
            "dependency_policy": DEPENDENCY_POLICY_CONTRACT,
            "dependency_canary": receipt["dependency_canary"],
            "root_initial_hold": True,
            "alert_latency_bound_seconds": receipt["dependency_canary"][
                "alert_latency_bound_seconds"
            ],
            **bootstrap_binding,
        }
        if launch_path.exists() or launch_path.is_symlink():
            return _validate_launch_complete(
                launch_path,
                receipt_path=receipt_path,
                receipt=receipt,
                release_path=release_path,
                release=release,
            )
        launch["launch_id"] = _sha256_bytes(_canonical_json(launch))
        _write_immutable_json_once(
            launch_path,
            launch,
            description="marker-last recovery-chain launch completion",
        )
        return _validate_launch_complete(
            launch_path,
            receipt_path=receipt_path,
            receipt=receipt,
            release_path=release_path,
            release=release,
        )

    if release_path.exists() or release_path.is_symlink():
        release = _validate_root_release_complete(
            release_path,
            receipt_path=receipt_path,
            receipt=receipt,
            root_record=root_record,
        )
        return release, publish_launch(release)

    acceptance: dict[str, Any] | None = None
    acceptance_path: Path | None = None
    if initial_submission:
        acceptance, acceptance_path = _ensure_scheduler_acceptance(
            evidence_root=evidence_root,
            manifest=manifest,
            receipt=receipt,
            receipt_path=receipt_path,
            runner=runner,
            timestamp=timestamp,
            scheduler_since=str(journal["scheduler_since"]),
        )
    acceptance_binding = (
        {}
        if acceptance is None or acceptance_path is None
        else {
            "scheduler_acceptance": str(acceptance_path),
            "scheduler_acceptance_sha256": _sha256(acceptance_path),
            "scheduler_acceptance_id": acceptance["acceptance_id"],
        }
    )
    intent_path = evidence_root / ROOT_RELEASE_INTENT_NAME
    release_intent = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "protocol": "schema5-v1.2-r7-recovery-root-release-intent-v1",
        "created_at": _utc_now(),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "root_name": root_name,
        "root_job_id": root_job_ids[0],
        "root_comment": root_record["comment"],
        "root_names": root_names,
        "root_job_ids": root_job_ids,
        "root_comments": root_comments,
        "command": ["scontrol", "release", root_job_ids[0]],
        "commands": [
            ["scontrol", "release", job_id]
            for job_id in root_job_ids
        ],
        **bootstrap_binding,
        **acceptance_binding,
    }
    if intent_path.exists() or intent_path.is_symlink():
        persisted_intent = _read_json(
            intent_path, description="recovery root release intent"
        )
        comparable = dict(release_intent)
        comparable["created_at"] = persisted_intent.get("created_at")
        if (
            persisted_intent != comparable
            or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
        ):
            raise ChainError("recovery root release intent drifted")
        release_intent = persisted_intent
    else:
        _write_immutable_json_once(
            intent_path,
            release_intent,
            description="marker-first recovery root release intent",
        )

    attempts_root = evidence_root / "root_release_attempts"
    if attempts_root.is_symlink():
        raise ChainError("recovery root release attempts are symlinked")
    attempts_root.mkdir(parents=True, mode=0o750, exist_ok=True)
    attempts_root = _require_canonical_path(
        attempts_root,
        description="recovery root release attempts",
        kind="directory",
    )
    existing_intents = sorted(attempts_root.glob("attempt-*.intent.json"))
    if any(
        path.name != f"attempt-{index:04d}.intent.json"
        for index, path in enumerate(existing_intents, start=1)
    ):
        raise ChainError("recovery root release attempts are not contiguous")
    if existing_intents:
        first_attempt = _read_json(
            _require_canonical_path(
                existing_intents[0],
                description="recovery root release attempt intent",
                kind="file",
            ),
            description="recovery root release attempt intent",
        )
        policy_path = _require_canonical_path(
            Path(str(first_attempt.get("dependency_policy_check", ""))),
            description="root-release dependency-policy evidence",
            kind="file",
        )
        policy = _validate_dependency_policy_check(
            policy_path, expected_phase="root_release"
        )
        if first_attempt.get(
            "dependency_policy_check_sha256"
        ) != _sha256(policy_path):
            raise ChainError(
                "root-release attempt lost its dependency policy binding"
            )
    else:
        policy, policy_path = _persist_dependency_policy_check(
            evidence_root,
            phase="root_release",
            runner=runner,
            checked_at=timestamp,
        )
    attempts_by_root: dict[str, list[Path]] = {
        name: [] for name in root_names
    }
    for index, attempt_path in enumerate(existing_intents, start=1):
        attempt = _read_json(
            _require_canonical_path(
                attempt_path,
                description="recovery root release attempt intent",
                kind="file",
            ),
            description="recovery root release attempt intent",
        )
        name = str(attempt.get("root_name", ""))
        if (
            name not in attempts_by_root
            or attempt.get("attempt") != index
            or attempt.get("job_id")
            != root_job_ids[root_names.index(name)]
            or attempt.get("command")
            != release_intent["commands"][root_names.index(name)]
            or attempt.get("dependency_policy_check") != str(policy_path)
            or attempt.get("dependency_policy_check_sha256")
            != _sha256(policy_path)
        ):
            raise ChainError(
                "recovery root release attempt intent drifted"
            )
        attempts_by_root[name].append(attempt_path)

    manifest_by_name = {
        str(row["name"]): row for row in manifest["jobs"]
    }
    roots_state_before: list[dict[str, Any]] = []
    roots_state_after: list[dict[str, Any]] = []
    scheduler_observations: list[dict[str, Any]] = []
    performed = False
    for root_index, (name, job_id, comment) in enumerate(
        zip(root_names, root_job_ids, root_comments, strict=True)
    ):
        matches = _query_comment_jobs(
            comment,
            slurm_user=str(manifest["slurm_user"]),
            since=str(journal["scheduler_since"]),
            runner=runner,
        )
        if (
            len(matches) != 1
            or matches[0]["job_id"] != job_id
            or matches[0]["job_name"]
            != manifest_by_name[name]["job_name"]
        ):
            raise ChainError(
                f"held recovery root scheduler identity is ambiguous: {name}"
            )
        before = _show_recovery_job(job_id, runner=runner)
        if before["returncode"] != 0:
            raise ChainError(
                f"cannot reconcile exact recovery root {name}/{job_id}"
            )
        roots_state_before.append(before)
        state = _normalize_slurm_state(
            str(before["fields"].get("JobState", ""))
        )
        held = (
            state == "PENDING"
            and str(before["fields"].get("Reason", "")).lower()
            == "jobhelduser"
        )
        root_attempts = attempts_by_root[name]
        missing_results = [
            attempt_path
            for attempt_path in root_attempts
            if not attempt_path.with_name(
                attempt_path.name.replace(
                    ".intent.json", ".result.json"
                )
            ).exists()
        ]
        if missing_results:
            if (
                len(missing_results) != 1
                or missing_results[0] != existing_intents[-1]
            ):
                raise ChainError(
                    "an ambiguous root release attempt cannot be "
                    "reconciled from exact scheduler state"
                )
            ambiguous_intent_path = missing_results[0]
            ambiguous_intent = _read_json(
                ambiguous_intent_path,
                description="ambiguous root release attempt intent",
            )
            adoption_result = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": (
                    "schema5-v1.2-r7-root-release-attempt-result-v1"
                ),
                "attempt": ambiguous_intent["attempt"],
                "completed_at": _utc_now(),
                "root_index": root_index,
                "root_name": name,
                "job_id": job_id,
                "intent": str(ambiguous_intent_path),
                "intent_sha256": _sha256(ambiguous_intent_path),
                "result_kind": (
                    "scheduler_reconciled_no_effect"
                    if held
                    else (
                        "scheduler_reconciled_after_ambiguous_release"
                    )
                ),
                "returncode": None,
                "stdout": None,
                "stderr": None,
                "stdout_sha256": None,
                "stderr_sha256": None,
                "scheduler_reconciliation": before,
            }
            adoption_result["result_id"] = _sha256_bytes(
                _canonical_json(adoption_result)
            )
            adoption_path = ambiguous_intent_path.with_name(
                ambiguous_intent_path.name.replace(
                    ".intent.json", ".result.json"
                )
            )
            _write_immutable_json_once(
                adoption_path,
                adoption_result,
                description=(
                    "scheduler-reconciled root release attempt result"
                ),
            )
            _validate_root_release_attempt_result(
                adoption_path,
                intent_path=ambiguous_intent_path,
                intent=ambiguous_intent,
            )
        if held:
            attempt_number = len(existing_intents) + 1
            attempt_intent_path = (
                attempts_root
                / f"attempt-{attempt_number:04d}.intent.json"
            )
            attempt_intent = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": (
                    "schema5-v1.2-r7-root-release-attempt-intent-v1"
                ),
                "attempt": attempt_number,
                "created_at": _utc_now(),
                "root_index": root_index,
                "root_name": name,
                "job_id": job_id,
                "command": release_intent["commands"][root_index],
                "dependency_policy_check": str(policy_path),
                "dependency_policy_check_sha256": _sha256(policy_path),
            }
            _write_immutable_json_once(
                attempt_intent_path,
                attempt_intent,
                description="recovery root release attempt intent",
            )
            existing_intents.append(attempt_intent_path)
            attempts_by_root[name].append(attempt_intent_path)
            proc = runner(release_intent["commands"][root_index])
            result = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": (
                    "schema5-v1.2-r7-root-release-attempt-result-v1"
                ),
                "attempt": attempt_number,
                "completed_at": _utc_now(),
                "root_index": root_index,
                "root_name": name,
                "job_id": job_id,
                "intent": str(attempt_intent_path),
                "intent_sha256": _sha256(attempt_intent_path),
                "result_kind": "scontrol_result",
                "returncode": int(proc.returncode),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "stdout_sha256": _sha256_bytes(
                    proc.stdout.encode("utf-8")
                ),
                "stderr_sha256": _sha256_bytes(
                    proc.stderr.encode("utf-8")
                ),
                "scheduler_reconciliation": None,
            }
            result["result_id"] = _sha256_bytes(
                _canonical_json(result)
            )
            result_path = (
                attempts_root
                / f"attempt-{attempt_number:04d}.result.json"
            )
            _write_immutable_json_once(
                result_path,
                result,
                description="recovery root release attempt result",
            )
            _validate_root_release_attempt_result(
                result_path,
                intent_path=attempt_intent_path,
                intent=attempt_intent,
            )
            if proc.returncode != 0:
                raise ChainError(
                    f"cannot release exact recovery root {name}/{job_id}: "
                    f"{proc.stderr.strip()[:500]}"
                )
            performed = True
        elif not attempts_by_root[name]:
            raise ChainError(
                "non-held recovery frontier root lacks its exact "
                f"marker-first release attempt: {name}/{job_id}"
            )
        after = _show_recovery_job(job_id, runner=runner)
        if (
            after["returncode"] != 0
            or (
                _normalize_slurm_state(
                    str(after["fields"].get("JobState", ""))
                )
                == "PENDING"
                and str(after["fields"].get("Reason", "")).lower()
                == "jobhelduser"
            )
        ):
            raise ChainError(
                f"recovery root remains held after exact release: {name}"
            )
        roots_state_after.append(after)
        observed = _query_comment_jobs(
            comment,
            slurm_user=str(manifest["slurm_user"]),
            since=str(journal["scheduler_since"]),
            runner=runner,
        )
        if (
            len(observed) != 1
            or observed[0]["job_id"] != job_id
            or observed[0]["job_name"]
            != manifest_by_name[name]["job_name"]
        ):
            raise ChainError(
                f"released recovery root scheduler identity is ambiguous: {name}"
            )
        scheduler_observations.append(observed[0])
    attempts = []
    for ordinal, attempt_intent_path in enumerate(
        existing_intents, start=1
    ):
        attempt_intent = _read_json(
            attempt_intent_path,
            description="recovery root release attempt intent",
        )
        result_path = attempt_intent_path.with_name(
            attempt_intent_path.name.replace(".intent.json", ".result.json")
        )
        if not result_path.exists():
            raise ChainError(
                "recovery root release completion requires every attempt "
                "result"
            )
        attempt_result = _validate_root_release_attempt_result(
            result_path,
            intent_path=attempt_intent_path,
            intent=attempt_intent,
        )
        attempts.append(
            {
                "attempt": ordinal,
                "intent": str(attempt_intent_path),
                "intent_sha256": _sha256(attempt_intent_path),
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "result_id": attempt_result["result_id"],
                "result_kind": attempt_result["result_kind"],
                "root_name": attempt_intent["root_name"],
                "job_id": attempt_intent["job_id"],
            }
        )
    release = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "protocol": "schema5-v1.2-r7-recovery-root-release-v1",
        "completed_at": _utc_now(),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "root_name": root_name,
        "root_job_id": root_job_ids[0],
        "root_comment": root_record["comment"],
        "root_names": root_names,
        "root_job_ids": root_job_ids,
        "root_comments": root_comments,
        "release_intent": str(intent_path),
        "release_intent_sha256": _sha256(intent_path),
        "dependency_policy_check": str(policy_path),
        "dependency_policy_check_sha256": _sha256(policy_path),
        "dependency_canary": receipt["dependency_canary"],
        "state_before": roots_state_before[0],
        "state_after": roots_state_after[0],
        "scheduler_observation": scheduler_observations[0],
        "roots_state_before": roots_state_before,
        "roots_state_after": roots_state_after,
        "scheduler_observations": scheduler_observations,
        "release_attempts": attempts,
        "release_reconciled": not performed,
        "root_no_longer_held": True,
        "all_roots_no_longer_held": True,
        **bootstrap_binding,
        **acceptance_binding,
    }
    release["release_id"] = _sha256_bytes(_canonical_json(release))
    _write_immutable_json_once(
        release_path,
        release,
        description="marker-last recovery root release completion",
    )
    release = _validate_root_release_complete(
        release_path,
        receipt_path=receipt_path,
        receipt=receipt,
        root_record=root_record,
    )
    return release, publish_launch(release)


def submit_chain(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    manifest_path = _lexical_absolute(manifest_path)
    verified = verify_chain(manifest_path)
    live_prerequisites = verify_live_creation_prerequisites(manifest_path)
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
                    row,
                    dependency_job_ids=(),
                    comment=comments[row["name"]],
                    initial_hold=row["name"] == "source_checkout",
                )[:-1]
                + (
                    [
                        f"--dependency={row['dependency_type']}:"
                        + ":".join(dependencies)
                    ]
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
    runner = _default_runner if runner is None else runner
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
        # Close the render/verify -> scheduler TOCTOU window while the singleton
        # submission boundary is held.  This reruns both full tagged verifiers.
        verified = verify_chain(manifest_path)
        live_prerequisites = verify_live_creation_prerequisites(manifest_path)
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = _validate_submission_receipt(
                receipt_path,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=comments,
            )
            provenance_path = _bootstrap_generation_provenance_path(
                manifest_path=manifest_path,
                generation=0,
            )
            provenance = _ensure_bootstrap_generation_provenance(
                manifest=manifest,
                manifest_path=manifest_path,
                receipt=receipt,
                receipt_path=receipt_path,
                generation=0,
                states=_query_receipt_job_states(
                    receipt=receipt,
                    manifest=manifest,
                    runner=runner,
                ),
                runner=runner,
            )
            return receipt | {
                "status": (
                    "already_launched"
                    if (
                        recovery_root / LAUNCH_COMPLETE_NAME
                    ).is_file()
                    else "awaiting_bootstrap_watchdog"
                ),
                "verified": verified,
                "live_prerequisites": live_prerequisites,
                "launch_marker_complete": (
                    recovery_root / LAUNCH_COMPLETE_NAME
                ).is_file(),
                "release_reconciliation_required": not (
                    recovery_root / LAUNCH_COMPLETE_NAME
                ).is_file(),
                "bootstrap_watchdog_ready": str(
                    recovery_root / BOOTSTRAP_WATCHDOG_READY_NAME
                ),
                "bootstrap_watchdog_armed": str(
                    recovery_root / BOOTSTRAP_WATCHDOG_ARMED_NAME
                ),
                "bootstrap_generation_provenance": str(provenance_path),
                "bootstrap_generation_provenance_sha256": _sha256(
                    provenance_path
                ),
                "bootstrap_generation_provenance_id": provenance[
                    "provenance_id"
                ],
            }

        if journal_path.exists() or journal_path.is_symlink():
            # A resumed submitter reparses current live policy.  The root remains
            # held until a separately persisted release-time check is complete.
            _dependency_config_allows_fail_closed(
                runner, checked_at=boundary_timestamp()
            )
            journal = _read_json(journal_path, description="chain submission journal")
            _validate_submission_journal(
                journal,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=comments,
            )
        else:
            policy, policy_path = _persist_dependency_policy_check(
                recovery_root,
                phase="pre_submission",
                runner=runner,
                checked_at=boundary_timestamp(),
            )
            journal = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": "schema5-v1.2-r7-recovery-chain-submission",
                "chain_id": manifest["chain_id"],
                "manifest": str(manifest_path),
                "started_at": _utc_now(),
                "started_timestamp": timestamp,
                "scheduler_since": _slurm_timestamp(timestamp),
                "slurm_user": manifest["slurm_user"],
                "dependency_policy_check": str(policy_path),
                "dependency_policy_check_sha256": _sha256(policy_path),
                "dependency_canary": _dependency_canary_binding(manifest),
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
                row,
                dependency_job_ids=dependency_ids,
                comment=comment,
                initial_hold=name == "source_checkout",
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
            "protocol": "schema5-v1.2-r7-recovery-chain-submission",
            "passed": True,
            "chain_id": manifest["chain_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "submission_journal": str(journal_path),
            "submission_journal_sha256": _sha256(journal_path),
            "submitted_at": _utc_now(),
            "dependency_policy": DEPENDENCY_POLICY_CONTRACT,
            "dependency_policy_check": journal[
                "dependency_policy_check"
            ],
            "dependency_policy_check_sha256": journal[
                "dependency_policy_check_sha256"
            ],
            "dependency_canary": journal["dependency_canary"],
            "root_initial_hold": True,
            "no_requeue": True,
            "stage_failure_sentinels": manifest[
                "stage_failure_sentinels"
            ],
            "jobs": [
                {
                    "name": row["name"],
                    "job_id": submitted[row["name"]],
                    "dependencies": list(row["dependencies"]),
                    "dependency_job_ids": [submitted[item] for item in row["dependencies"]],
                    "comment": comments[row["name"]],
                    "script": row["script"],
                    "script_sha256": row["script_sha256"],
                    "dependency_type": row["dependency_type"],
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
        provenance_path = _bootstrap_generation_provenance_path(
            manifest_path=manifest_path,
            generation=0,
        )
        provenance = _ensure_bootstrap_generation_provenance(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=validated_receipt,
            receipt_path=receipt_path,
            generation=0,
            states=_query_receipt_job_states(
                receipt=validated_receipt,
                manifest=manifest,
                runner=runner,
            ),
            runner=runner,
        )
        return validated_receipt | {
            "status": "awaiting_bootstrap_watchdog",
            "verified": verified,
            "live_prerequisites": live_prerequisites,
            "launch_marker_complete": False,
            "release_reconciliation_required": False,
            "bootstrap_watchdog_ready": str(
                recovery_root / BOOTSTRAP_WATCHDOG_READY_NAME
            ),
            "bootstrap_watchdog_armed": str(
                recovery_root / BOOTSTRAP_WATCHDOG_ARMED_NAME
            ),
            "bootstrap_generation_provenance": str(provenance_path),
            "bootstrap_generation_provenance_sha256": _sha256(
                provenance_path
            ),
            "bootstrap_generation_provenance_id": provenance[
                "provenance_id"
            ],
        }


def release_recovery_root(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Release the latest exact held generation after its sealed arm decision.

    Submission and release are intentionally separate transactions.  This command
    has no path that creates scheduler jobs, clears a scientific control-plane
    hold, or changes admission; it can release only the exact receipt-bound
    recovery root of the latest complete generation.
    """

    manifest_path = _lexical_absolute(manifest_path)
    verified = verify_chain(manifest_path)
    live_prerequisites = verify_live_creation_prerequisites(manifest_path)
    manifest = _read_json(
        manifest_path, description="recovery-chain manifest"
    )
    recovery_root = manifest_path.parent
    receipt, receipt_path, generation, pending = _latest_chain_receipt(
        manifest=manifest, manifest_path=manifest_path
    )
    if pending is not None:
        raise ChainError(
            "cannot release while the latest repair transaction is incomplete"
        )
    journal_path = receipt_path.parent / SUBMISSION_JOURNAL_NAME
    journal = _read_json(
        journal_path, description="sealed chain submission journal"
    )
    if generation == 0:
        _validate_submission_journal(
            journal,
            manifest=manifest,
            manifest_path=manifest_path,
            comments=_submission_comments(manifest),
        )
    else:
        parent_path = _require_canonical_path(
            Path(str(receipt["parent_receipt"])),
            description="release generation parent receipt",
            kind="file",
        )
        parent = _read_json(
            parent_path, description="release generation parent receipt"
        )
        _validate_repair_journal(
            journal,
            manifest=manifest,
            base=parent,
            base_path=parent_path,
            generation=generation,
        )
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    root_record = root_records[0]
    root_name = str(root_record["name"])
    root_names = [str(record["name"]) for record in root_records]
    root_job_ids = [str(record["job_id"]) for record in root_records]
    anchor_launch: Mapping[str, Any] | None = None
    anchor_release_intent: Mapping[str, Any] | None = None
    if generation > 0:
        anchor_launch = _bootstrap_anchor_launch_authorization(
            manifest=manifest,
            manifest_path=manifest_path,
        )
        anchor_release_intent = (
            _bootstrap_anchor_release_intent_authorization(
                manifest=manifest,
                manifest_path=manifest_path,
            )
        )
        if anchor_launch is None and anchor_release_intent is None:
            raise ChainError(
                "a descendant bootstrap arm authorizes rearm only; "
                "release requires the sealed g0 launch or root-release intent"
            )
    armed_path: Path
    if generation == 0:
        armed_path = recovery_root / BOOTSTRAP_WATCHDOG_ARMED_NAME
        armed = _validate_bootstrap_watchdog_armed(
            armed_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            root_record=root_record,
        )
        armed_result = {
            "bootstrap_watchdog_ready": armed[
                "bootstrap_watchdog_ready"
            ],
            "bootstrap_watchdog_ready_id": armed[
                "bootstrap_watchdog_ready_id"
            ],
            "bootstrap_watchdog_armed": str(armed_path),
            "bootstrap_watchdog_armed_id": armed["marker_id"],
        }
    else:
        armed_path = receipt_path.parent / BOOTSTRAP_DESCENDANT_ARMED_NAME
        armed = _validate_bootstrap_descendant_armed(
            armed_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            root_record=root_record,
        )
        armed_result = {
            "bootstrap_descendant_armed": str(armed_path),
            "bootstrap_descendant_armed_id": armed["marker_id"],
            "anchor_bootstrap_armed_id": armed[
                "anchor_bootstrap_armed_id"
            ],
            "anchor_launch_authorized": anchor_launch is not None,
            "anchor_launch_id": (
                None
                if anchor_launch is None
                else anchor_launch["launch_id"]
            ),
            "anchor_release_intent_authorized": (
                anchor_release_intent is not None
            ),
            "anchor_release_intent_sha256": (
                None
                if anchor_release_intent is None
                else anchor_release_intent["intent_sha256"]
            ),
        }
    if not apply:
        return {
            "status": "ready_to_release",
            "passed": True,
            "chain_id": manifest["chain_id"],
            "receipt": str(receipt_path),
            "receipt_id": receipt["receipt_id"],
            "repair_generation": generation,
            "root_job_id": root_record["job_id"],
            "root_names": root_names,
            "root_job_ids": root_job_ids,
            **armed_result,
            "verified": verified,
            "live_prerequisites": live_prerequisites,
        }
    runner = _default_runner if runner is None else runner
    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp):
        raise ChainError("root-release timestamp must be finite")
    lock_path = recovery_root / SUBMISSION_LOCK_NAME
    with _exclusive_lock(
        lock_path, description="recovery-chain root releaser"
    ):
        # Revalidate every immutable input while holding the same transaction
        # lock used by submit and repair.
        current, current_path, current_generation, current_pending = (
            _latest_chain_receipt(
                manifest=manifest, manifest_path=manifest_path
            )
        )
        if (
            current_pending is not None
            or current_path != receipt_path
            or current_generation != generation
            or current["receipt_id"] != receipt["receipt_id"]
        ):
            raise ChainError(
                "latest recovery generation changed before root release"
            )
        receipt = current
        verify_live_creation_prerequisites(manifest_path)
        states = _query_receipt_job_states(
            receipt=receipt, manifest=manifest, runner=runner
        )
        _ensure_bootstrap_generation_provenance(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            states=states,
            runner=runner,
        )
        if generation == 0:
            _validate_bootstrap_watchdog_armed(
                armed_path,
                manifest=manifest,
                manifest_path=manifest_path,
                receipt=receipt,
                receipt_path=receipt_path,
                root_record=root_record,
            )
        else:
            _validate_bootstrap_descendant_armed(
                armed_path,
                manifest=manifest,
                manifest_path=manifest_path,
                receipt=receipt,
                receipt_path=receipt_path,
                generation=generation,
                root_record=root_record,
            )
            current_anchor_launch = (
                _bootstrap_anchor_launch_authorization(
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            )
            current_anchor_release_intent = (
                _bootstrap_anchor_release_intent_authorization(
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            )
            if (
                current_anchor_launch is None
                and current_anchor_release_intent is None
            ):
                raise ChainError(
                    "a descendant bootstrap arm authorizes rearm only; "
                    "release requires the sealed g0 launch or "
                    "root-release intent"
                )
        release, launch = _ensure_recovery_root_released(
            evidence_root=receipt_path.parent,
            manifest=manifest,
            receipt_path=receipt_path,
            receipt=receipt,
            journal=journal,
            root_name=root_name,
            runner=runner,
            timestamp=timestamp,
            bootstrap_watchdog_armed_path=(
                armed_path if generation == 0 else None
            ),
            bootstrap_descendant_armed_path=(
                armed_path if generation > 0 else None
            ),
        )
    return {
        "status": "launched",
        "passed": True,
        "chain_id": manifest["chain_id"],
        "receipt": str(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "repair_generation": generation,
        "root_names": root_names,
        "root_job_ids": root_job_ids,
        **armed_result,
        "root_release": release,
        "launch_complete": launch,
        "verified": verified,
        "live_prerequisites": live_prerequisites,
    }


def publish_bootstrap_watchdog_handoff(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """End bootstrap authority only from the successful aggregate sentinel."""

    manifest_path = _lexical_absolute(manifest_path)
    verify_chain(manifest_path)
    manifest = _read_json(
        manifest_path, description="recovery-chain manifest"
    )
    recovery_root = manifest_path.parent
    path = recovery_root / BOOTSTRAP_WATCHDOG_HANDOFF_NAME
    if path.exists() or path.is_symlink():
        path = _require_canonical_path(
            path,
            description="bootstrap watchdog handoff",
            kind="file",
        )
        if path.stat().st_nlink != 1 or stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ChainError("bootstrap watchdog handoff must be sealed")
        persisted = _read_json(
            path, description="bootstrap watchdog handoff"
        )
        identity = dict(persisted)
        handoff_id = identity.pop("handoff_id", None)
        receipt_path = _require_canonical_path(
            Path(str(persisted.get("submission_receipt", ""))),
            description="handoff submission receipt",
            kind="file",
        )
        receipt, latest_path, generation, pending = _latest_chain_receipt(
            manifest=manifest, manifest_path=manifest_path
        )
        root_record = _generation_root_records(
            receipt, generation=generation
        )[0]
        release_path = _require_canonical_path(
            Path(str(persisted.get("root_release", ""))),
            description="handoff root release",
            kind="file",
        )
        release = _validate_root_release_complete(
            release_path,
            receipt_path=latest_path,
            receipt=receipt,
            root_record=root_record,
        )
        launch_path = _require_canonical_path(
            Path(str(persisted.get("launch_complete", ""))),
            description="handoff launch completion",
            kind="file",
        )
        launch = _validate_launch_complete(
            launch_path,
            receipt_path=latest_path,
            receipt=receipt,
            release_path=release_path,
            release=release,
        )
        ready_path = _require_canonical_path(
            Path(str(persisted.get("production_watchdog_ready", ""))),
            description="handoff production watchdog readiness",
            kind="file",
        )
        ready = _read_json(
            ready_path, description="handoff production watchdog readiness"
        )
        control = _read_json(
            _require_canonical_path(
                Path(str(manifest["state_root"])) / "control.json",
                description="handoff production control",
                kind="file",
            ),
            description="handoff production control",
        )
        if (
            pending is not None
            or receipt_path != latest_path
            or persisted.get("schema_version") != 2
            or persisted.get("protocol")
            != BOOTSTRAP_WATCHDOG_HANDOFF_PROTOCOL
            or persisted.get("passed") is not True
            or persisted.get("release_id") != RELEASE_ID
            or persisted.get("release_tag") != RELEASE_TAG
            or persisted.get("release_git_commit")
            != manifest["release_git_commit"]
            or persisted.get("release_tag_object")
            != manifest["release_tag_object"]
            or persisted.get("chain_namespace") != CHAIN_NAMESPACE
            or persisted.get("chain_id") != manifest["chain_id"]
            or persisted.get("repair_generation") != generation
            or persisted.get("submission_receipt_sha256")
            != _sha256(latest_path)
            or persisted.get("submission_receipt_id")
            != receipt["receipt_id"]
            or persisted.get("root_release_sha256")
            != _sha256(release_path)
            or persisted.get("root_release_id")
            != release["release_id"]
            or persisted.get("launch_complete_sha256")
            != _sha256(launch_path)
            or persisted.get("launch_id") != launch["launch_id"]
            or persisted.get("production_watchdog_ready_sha256")
            != _sha256(ready_path)
            or persisted.get("production_watchdog_ready_id")
            != ready.get("marker_id")
            or persisted.get("control_sha256")
            != control.get("immutable_sha256")
            or persisted.get("all_predecessors_terminal_success")
            is not True
            or persisted.get("aggregate_sentinel_running")
            is not True
            or persisted.get("aggregate_sentinel_classification_success")
            is not True
            or persisted.get("production_resume_terminal_success")
            is not True
            or persisted.get("production_control_healthy") is not True
            or persisted.get("bootstrap_repair_disabled") is not True
            or persisted.get("production_watchdog_authoritative") is not True
            or not isinstance(persisted.get("handoff_at"), str)
            or not isinstance(handoff_id, str)
            or _SHA256.fullmatch(handoff_id) is None
            or handoff_id
            != _sha256_bytes(_canonical_json(identity))
        ):
            raise ChainError("bootstrap watchdog handoff binding drifted")
        return {
            "passed": True,
            "apply": apply,
            "handoff": str(path),
            "handoff_id": handoff_id,
            "bootstrap_repair_disabled": True,
            "production_watchdog_authoritative": True,
        }

    if not apply:
        raise ChainError("bootstrap watchdog handoff is not published")
    runner = _default_runner if runner is None else runner
    with _exclusive_lock(
        recovery_root / SUBMISSION_LOCK_NAME,
        description="bootstrap watchdog handoff publisher",
    ):
        if path.exists() or path.is_symlink():
            raise ChainError(
                "bootstrap watchdog handoff appeared while acquiring its "
                "transaction lock; replay verification is required"
            )
        report = verify_post_initialize_watchdog(
            manifest_path, expected_desired_state="running"
        )
        receipt, receipt_path, generation, pending = _latest_chain_receipt(
            manifest=manifest, manifest_path=manifest_path
        )
        if pending is not None:
            raise ChainError(
                "bootstrap handoff refuses an incomplete repair transaction"
            )
        root_record = _generation_root_records(
            receipt, generation=generation
        )[0]
        release_path = receipt_path.parent / ROOT_RELEASE_COMPLETE_NAME
        release = _validate_root_release_complete(
            release_path,
            receipt_path=receipt_path,
            receipt=receipt,
            root_record=root_record,
        )
        launch_path = receipt_path.parent / LAUNCH_COMPLETE_NAME
        launch = _validate_launch_complete(
            launch_path,
            receipt_path=receipt_path,
            receipt=receipt,
            release_path=release_path,
            release=release,
        )
        states = _query_receipt_job_states(
            receipt=receipt, manifest=manifest, runner=runner
        )
        sentinel = states.get("failure_sentinel", {})
        current_job_id = os.environ.get("SLURM_JOB_ID")
        if (
            current_job_id != sentinel.get("job_id")
            or sentinel.get("state") != "RUNNING"
            or sentinel.get("active") is not True
        ):
            raise ChainError(
                "bootstrap handoff is not running in the exact aggregate sentinel"
            )
        failures = [
            name
            for name, state in states.items()
            if name != "failure_sentinel"
            and (
                state.get("state") != "COMPLETED"
                or state.get("active") is not False
                or state.get("exit_code") != "0:0"
            )
        ]
        production_resume = states.get("production_resume", {})
        if failures or production_resume.get("state") != "COMPLETED":
            raise ChainError(
                "bootstrap handoff requires every stage and observer to "
                f"finish successfully: {failures}"
            )
        harness_python = (
            Path(str(manifest["release_root"]))
            / "environments"
            / "harness"
            / "bin"
            / "python"
        )
        control_tool = (
            Path(str(manifest["release_root"]))
            / "worktree"
            / "slurm"
            / "schema5_control.py"
        )
        status_proc = runner(
            [
                str(harness_python),
                "-I",
                str(control_tool),
                "--state-dir",
                str(manifest["state_root"]),
                "status",
                "--live",
            ]
        )
        try:
            live = json.loads(status_proc.stdout)
        except json.JSONDecodeError as exc:
            raise ChainError(
                "production control status returned invalid JSON"
            ) from exc
        controllers = live.get("controllers")
        if (
            status_proc.returncode != 0
            or live.get("healthy") is not True
            or live.get("desired_state") != "running"
            or live.get("drain_requested") is not False
            or live.get("immutable_sha256") != report["control_sha256"]
            or not isinstance(controllers, Mapping)
            or set(controllers) != {"dispatcher", "fleet_supervisor"}
            or any(
                not isinstance(value, Mapping)
                or value.get("live_active") is not True
                or value.get("heartbeat_stale") is not False
                or value.get("active_identity_mismatch") is not False
                for value in controllers.values()
            )
            or live.get("active_alerts") != []
            or live.get("admission_safety_hold", {}).get("active") is not False
        ):
            raise ChainError(
                "production controllers are not live and hold-free"
            )
        anchor = _bootstrap_anchor_armed_authorization(
            manifest=manifest, manifest_path=manifest_path
        )
        if anchor is None:
            raise ChainError("bootstrap handoff lacks its sealed g0 arm")
        production = report["watchdog_ready"]
        handoff: dict[str, Any] = {
            "schema_version": 2,
            "protocol": BOOTSTRAP_WATCHDOG_HANDOFF_PROTOCOL,
            "passed": True,
            "release_id": RELEASE_ID,
            "release_tag": RELEASE_TAG,
            "release_git_commit": manifest["release_git_commit"],
            "release_tag_object": manifest["release_tag_object"],
            "chain_namespace": CHAIN_NAMESPACE,
            "chain_id": manifest["chain_id"],
            "repair_generation": generation,
            "submission_receipt": str(receipt_path),
            "submission_receipt_sha256": _sha256(receipt_path),
            "submission_receipt_id": receipt["receipt_id"],
            "root_release": str(release_path),
            "root_release_sha256": _sha256(release_path),
            "root_release_id": release["release_id"],
            "launch_complete": str(launch_path),
            "launch_complete_sha256": _sha256(launch_path),
            "launch_id": launch["launch_id"],
            "bootstrap_watchdog_armed": anchor["armed"],
            "bootstrap_watchdog_armed_sha256": anchor["armed_sha256"],
            "bootstrap_watchdog_armed_id": anchor["armed_id"],
            "production_watchdog_ready": production["marker"],
            "production_watchdog_ready_sha256": production["marker_sha256"],
            "production_watchdog_ready_id": production["marker_id"],
            "control_sha256": report["control_sha256"],
            "production_control_status_sha256": _sha256_bytes(
                _canonical_json(live)
            ),
            "aggregate_sentinel_job_id": sentinel["job_id"],
            "aggregate_sentinel_comment": sentinel["comment"],
            "all_predecessors_terminal_success": True,
            "aggregate_sentinel_running": True,
            "aggregate_sentinel_classification_success": True,
            "production_resume_job_id": production_resume["job_id"],
            "production_resume_comment": production_resume["comment"],
            "production_resume_terminal_success": True,
            "production_control_healthy": True,
            "handoff_at": _utc_now(),
            "bootstrap_repair_disabled": True,
            "production_watchdog_authoritative": True,
        }
        handoff["handoff_id"] = _sha256_bytes(_canonical_json(handoff))
        _write_immutable_json_once(
            path,
            handoff,
            description="marker-last bootstrap watchdog handoff",
        )
    return {
        "passed": True,
        "apply": apply,
        "handoff": str(path),
        "handoff_id": handoff["handoff_id"],
        "bootstrap_repair_disabled": True,
        "production_watchdog_authoritative": True,
    }


def verify_bootstrap_watchdog_handoff(
    manifest_path: Path,
) -> dict[str, Any]:
    path = _lexical_absolute(manifest_path).parent / (
        BOOTSTRAP_WATCHDOG_HANDOFF_NAME
    )
    if not path.exists() or path.is_symlink():
        raise ChainError("bootstrap watchdog handoff is not published")
    return publish_bootstrap_watchdog_handoff(
        manifest_path, apply=False
    )


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
        "submitted_at", "dependency_policy", "dependency_policy_check",
        "dependency_policy_check_sha256", "dependency_canary",
        "root_initial_hold", "held_root_names", "no_requeue",
        "stage_failure_sentinels",
        "jobs", "receipt_id",
    }
    parent_by_name = {
        str(record.get("name")): record
        for record in parent["jobs"]
        if isinstance(record, dict)
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or receipt.get("protocol") != "schema5-v1.2-r7-recovery-chain-repair"
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
        or receipt.get("dependency_policy") != DEPENDENCY_POLICY_CONTRACT
        or receipt.get("dependency_policy_check")
        != journal["dependency_policy_check"]
        or receipt.get("dependency_policy_check_sha256")
        != journal["dependency_policy_check_sha256"]
        or receipt.get("dependency_canary")
        != _dependency_canary_binding(manifest)
        or receipt.get("stage_failure_sentinels")
        != manifest["stage_failure_sentinels"]
        or receipt.get("root_initial_hold") is not True
        or receipt.get("held_root_names")
        != _repair_frontier_names(manifest, repair_names)
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
        "script", "script_sha256", "dependency_type", "generation", "disposition",
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
            or record["dependency_type"] != row["dependency_type"]
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
        "slurm_user",
        "dependency_policy_check",
        "dependency_policy_check_sha256",
        "dependency_canary",
        "held_root_names",
        "jobs",
    }
    started = journal.get("started_timestamp")
    repair_names = journal.get("repair_jobs")
    if (
        set(journal) != expected_fields
        or journal.get("schema_version") != SUBMISSION_SCHEMA_VERSION
        or journal.get("protocol") != "schema5-v1.2-r7-recovery-chain-repair-journal"
        or journal.get("chain_id") != manifest["chain_id"]
        or journal.get("repair_generation") != generation
        or journal.get("base_receipt") != str(base_path)
        or journal.get("base_receipt_sha256") != _sha256(base_path)
        or not isinstance(journal.get("started_at"), str)
        or not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not math.isfinite(float(started))
        or journal.get("scheduler_since") != _slurm_timestamp(float(started))
        or journal.get("slurm_user") != manifest["slurm_user"]
        or not isinstance(journal.get("dependency_policy_check"), str)
        or not isinstance(
            journal.get("dependency_policy_check_sha256"), str
        )
        or _SHA256.fullmatch(journal["dependency_policy_check_sha256"])
        is None
        or journal.get("dependency_canary")
        != _dependency_canary_binding(manifest)
        or not isinstance(repair_names, list)
        or not repair_names
        or len(repair_names) != len(set(repair_names))
        or journal.get("held_root_names")
        != _repair_frontier_names(manifest, repair_names)
        or not isinstance(journal.get("jobs"), dict)
    ):
        raise ChainError("repair journal identity is invalid")
    policy_path = _require_canonical_path(
        Path(journal["dependency_policy_check"]),
        description="repair dependency-policy evidence",
        kind="file",
    )
    expected_policy_parent = (
        Path(manifest["recovery_root"])
        / REPAIR_ROOT_NAME
        / f"g{generation:04d}"
        / DEPENDENCY_POLICY_CHECKS_ROOT_NAME
    )
    if (
        policy_path.parent != expected_policy_parent
        or _sha256(policy_path)
        != journal["dependency_policy_check_sha256"]
    ):
        raise ChainError("repair dependency-policy evidence path drifted")
    _validate_dependency_policy_check(
        policy_path, expected_phase="pre_submission"
    )
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
                row,
                dependency_job_ids=dependency_ids,
                comment=comment,
                initial_hold=name
                in set(_repair_frontier_names(manifest, repair_names)),
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


def _repair_frontier_names(
    manifest: Mapping[str, Any], repair_names: Sequence[str]
) -> list[str]:
    """Return every selected job with no selected ancestor, in DAG order."""

    by_name = {str(row["name"]): row for row in manifest["jobs"]}
    if (
        not repair_names
        or any(not isinstance(name, str) for name in repair_names)
        or len(repair_names) != len(set(repair_names))
        or not set(repair_names) <= set(by_name)
    ):
        raise ChainError(
            "repair set must name unique jobs in the recovery manifest"
        )
    selected = set(repair_names)
    roots = [
        str(row["name"])
        for row in manifest["jobs"]
        if row["name"] in selected
        and not (_manifest_ancestors(str(row["name"]), by_name) & selected)
    ]
    if not roots:
        raise ChainError("repair set has no admission frontier")
    return roots


def _is_exact_completed_stage_observer(
    *,
    name: str,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    failed_set: set[str],
) -> bool:
    """Recognize only the paired after-any observer of its failed stage."""

    pairs = {
        str(binding["sentinel"]): str(binding["stage"])
        for binding in manifest.get("stage_failure_sentinels", [])
        if isinstance(binding, Mapping)
        and isinstance(binding.get("sentinel"), str)
        and isinstance(binding.get("stage"), str)
    }
    stage = pairs.get(name)
    if stage is None or stage not in failed_set:
        return False
    manifest_by_name = {
        str(row["name"]): row
        for row in manifest.get("jobs", [])
        if isinstance(row, Mapping)
    }
    receipt_by_name = {
        str(row["name"]): row
        for row in receipt.get("jobs", [])
        if isinstance(row, Mapping)
    }
    observer_spec = manifest_by_name.get(name)
    stage_spec = manifest_by_name.get(stage)
    observer_receipt = receipt_by_name.get(name)
    stage_receipt = receipt_by_name.get(stage)
    observer_state = states.get(name)
    stage_state = states.get(stage)
    return bool(
        isinstance(observer_spec, Mapping)
        and isinstance(stage_spec, Mapping)
        and isinstance(observer_receipt, Mapping)
        and isinstance(stage_receipt, Mapping)
        and isinstance(observer_state, Mapping)
        and isinstance(stage_state, Mapping)
        and observer_spec.get("dependencies") == [stage]
        and observer_spec.get("dependency_type") == "afterany"
        and observer_receipt.get("dependencies") == [stage]
        and observer_receipt.get("dependency_type") == "afterany"
        and observer_receipt.get("dependency_job_ids")
        == [stage_receipt.get("job_id")]
        and observer_receipt.get("script") == observer_spec.get("script")
        and observer_receipt.get("script_sha256")
        == observer_spec.get("script_sha256")
        and observer_state.get("job_id")
        == observer_receipt.get("job_id")
        and observer_state.get("comment")
        == observer_receipt.get("comment")
        and observer_state.get("state") == "COMPLETED"
        and observer_state.get("active") is False
        and observer_state.get("exit_code") == "0:0"
        and stage_state.get("job_id") == stage_receipt.get("job_id")
        and stage_state.get("state") in _REPAIRABLE_SLURM_STATES
        and stage_state.get("active") is False
    )


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


def _validate_capacity_transient_repair_binding(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    generation: int,
    scheduler_evidence: Mapping[str, Any],
    outcome: Mapping[str, Any],
    fresh_states: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Authenticate the sole FAILED 75:0 root before suffix repair consumes it."""

    roots = outcome.get("capacity_transient_roots", [])
    binding = scheduler_evidence.get("capacity_transient_receipt")
    if roots == []:
        if binding is not None:
            raise ChainError(
                "sentinel bound a capacity receipt without a capacity root"
            )
        return []
    if roots != ["fleet_readiness"] or not isinstance(binding, Mapping):
        raise ChainError("sentinel capacity-transient root/binding is malformed")
    dispositions = outcome.get("dispositions")
    if (
        not isinstance(dispositions, Mapping)
        or dispositions.get("fleet_readiness") != "capacity_transient_root"
    ):
        raise ChainError("sentinel capacity root lacks its exact disposition")
    receipt = _read_json(receipt_path, description="capacity root parent receipt")
    receipt_rows = [
        row
        for row in receipt.get("jobs", [])
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    scheduler_rows = [
        row
        for row in scheduler_evidence.get("jobs", [])
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    fresh = fresh_states.get("fleet_readiness")
    if (
        len(receipt_rows) != 1
        or len(scheduler_rows) != 1
        or scheduler_rows[0].get("job_id") != receipt_rows[0].get("job_id")
        or scheduler_rows[0].get("comment") != receipt_rows[0].get("comment")
        or scheduler_rows[0].get("state") != "FAILED"
        or scheduler_rows[0].get("exit_code") != "75:0"
        or not isinstance(fresh, Mapping)
        or fresh.get("job_id") != receipt_rows[0].get("job_id")
        or fresh.get("state") != "FAILED"
        or fresh.get("exit_code") != "75:0"
        or fresh.get("active") is not False
        or fresh.get("comment") != receipt_rows[0].get("comment")
        or fresh.get("job_name") != scheduler_rows[0].get("job_name")
    ):
        raise ChainError(
            "capacity root is not the exact parent fleet_readiness FAILED 75:0 job"
        )
    job_id = str(receipt_rows[0]["job_id"])
    expected_marker = (
        manifest_path.parent
        / CAPACITY_TRANSIENT_ROOT_NAME
        / CHAIN_NAMESPACE
        / f"g{generation:04d}"
        / job_id
        / CAPACITY_TRANSIENT_MARKER_NAME
    )
    marker_path = _require_canonical_path(
        expected_marker,
        description="fleet capacity-transient receipt",
        kind="file",
    )
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o222:
        raise ChainError("fleet capacity-transient receipt must be read-only")
    marker = _read_json(
        marker_path, description="fleet capacity-transient receipt"
    )
    try:
        from scripts import schema5_recovery_sentinel as capacity_validator
    except ModuleNotFoundError:
        import schema5_recovery_sentinel as capacity_validator  # type: ignore[no-redef]
    try:
        validated_capacity_binding = (
            capacity_validator._validate_capacity_transient_receipt(  # noqa: SLF001
            marker_path,
            verified={
                "manifest": manifest,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "submission_receipt": receipt,
                "submission_receipt_path": str(receipt_path),
                "submission_receipt_sha256": _sha256(receipt_path),
            },
            jobs=scheduler_evidence.get("jobs", []),
        )
        )
    except (capacity_validator.SentinelError, OSError, ValueError) as exc:
        raise ChainError(
            f"strict fleet capacity-transient repair binding failed: {exc}"
        ) from exc
    marker_identity = dict(marker)
    marker_id = marker_identity.pop("receipt_id", None)
    evidence_path = marker_path.parent / "FLEET_CAPACITY_TRANSIENT_EVIDENCE.json"
    evidence_path = _require_canonical_path(
        evidence_path,
        description="fleet capacity-transient evidence",
        kind="file",
    )
    if stat.S_IMODE(evidence_path.stat().st_mode) & 0o222:
        raise ChainError("fleet capacity-transient evidence must be read-only")
    capacity_evidence = _read_json(
        evidence_path, description="fleet capacity-transient evidence"
    )
    capacity_identity = dict(capacity_evidence)
    capacity_evidence_id = capacity_identity.pop("evidence_id", None)
    fleet = capacity_evidence.get("fleet")
    pending = fleet.get("pending") if isinstance(fleet, Mapping) else None
    running = fleet.get("running") if isinstance(fleet, Mapping) else None
    expected_binding = dict(validated_capacity_binding)
    if (
        dict(binding) != expected_binding
        or marker.get("schema_version") != 1
        or marker.get("protocol")
        != "schema5-v1.2-r7-fleet-capacity-transient-receipt"
        or marker.get("passed") is not True
        or marker.get("capacity_transient_root") is not True
        or marker.get("chain_id") != manifest["chain_id"]
        or marker.get("chain_generation") != generation
        or marker.get("fleet_readiness_job_id") != job_id
        or marker.get("fleet_readiness_comment")
        != receipt_rows[0].get("comment")
        or marker.get("evidence") != str(evidence_path)
        or marker.get("evidence_sha256") != _sha256(evidence_path)
        or marker.get("evidence_id") != capacity_evidence_id
        or not isinstance(marker_id, str)
        or marker_id != _sha256_bytes(_canonical_json(marker_identity))
        or capacity_evidence.get("schema_version") != 1
        or capacity_evidence.get("protocol")
        != "schema5-v1.2-r7-fleet-capacity-transient-evidence"
        or capacity_evidence.get("passed") is not True
        or capacity_evidence.get("chain_id") != manifest["chain_id"]
        or capacity_evidence.get("chain_generation") != generation
        or capacity_evidence.get("manifest") != str(manifest_path)
        or capacity_evidence.get("manifest_sha256") != _sha256(manifest_path)
        or capacity_evidence.get("submission_receipt") != str(receipt_path)
        or capacity_evidence.get("submission_receipt_sha256")
        != _sha256(receipt_path)
        or capacity_evidence.get("fleet_readiness_job_id") != job_id
        or capacity_evidence.get("fleet_readiness_comment")
        != receipt_rows[0].get("comment")
        or capacity_evidence.get("boundary_seconds") != 36_000
        or not isinstance(capacity_evidence_id, str)
        or capacity_evidence_id
        != _sha256_bytes(_canonical_json(capacity_identity))
        or not isinstance(fleet, Mapping)
        or fleet.get("logical_replicas") != 22
        or fleet.get("allocated_gpus") != 24
        or not isinstance(pending, list)
        or not pending
        or not isinstance(running, list)
        or not running
        or len(pending) + len(running) != 22
        or fleet.get("ignored_current_terminal_job_ids") != []
        or fleet.get("overlap_replicas") != []
        or any(
            not isinstance(row, Mapping)
            or row.get("state") != "PENDING"
            or row.get("reason") not in {"Priority", "Resources"}
            for row in pending
        )
        or any(
            not isinstance(row, Mapping)
            or row.get("state") != "RUNNING"
            or not isinstance(row.get("http"), Mapping)
            or row["http"].get("healthy") is not True
            for row in running
        )
    ):
        raise ChainError("fleet capacity-transient repair binding is invalid")
    return ["fleet_readiness"]


def _load_qualification_capacity_transition_journal(
    qualification_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    root = (
        qualification_root
        / THROUGHPUT_QUALIFICATION_TRANSITION_DIRECTORY
    )
    if root.is_symlink() or not root.is_dir():
        raise ChainError(
            "qualification capacity-transition journal is missing or unsafe"
        )
    records: list[tuple[Path, dict[str, Any]]] = []
    expected_from = 1
    for path in sorted(root.iterdir()):
        match = re.fullmatch(
            r"c([0-9]{6})-to-c([0-9]{6})[.]json",
            path.name,
        )
        marker, _ = _read_sealed_marker(
            path,
            description="qualification capacity-transition journal record",
        )
        _require_marker_identity(
            marker,
            identity_field="transition_id",
            description="qualification capacity-transition journal record",
        )
        additive = marker.get("additive_transition")
        if (
            match is None
            or not isinstance(additive, Mapping)
            or marker.get("schema_version") != 1
            or marker.get("protocol")
            != THROUGHPUT_QUALIFICATION_TRANSITION_PROTOCOL
            or marker.get("passed") is not True
            or int(match.group(1)) != expected_from
            or int(match.group(2)) != expected_from + 1
            or additive.get("from_capacity_generation")
            != expected_from
            or additive.get("to_capacity_generation")
            != expected_from + 1
        ):
            raise ChainError(
                "qualification capacity-transition journal skips, duplicates, "
                "or drifts from its generation identity"
            )
        records.append((path, marker))
        expected_from += 1
    if not records:
        raise ChainError(
            "qualification capacity-transition journal is empty"
        )
    pointer, _ = _read_sealed_marker(
        qualification_root
        / THROUGHPUT_QUALIFICATION_CURRENT_TRANSITION_NAME,
        description="current qualification capacity-transition pointer",
    )
    required_pointer = {
        "schema_version",
        "protocol",
        "path",
        "sha256",
        "transition_id",
        "from_capacity_generation",
        "to_capacity_generation",
        "failed_attempt_id",
        "pointer_id",
    }
    _require_marker_identity(
        pointer,
        identity_field="pointer_id",
        description="current qualification capacity-transition pointer",
    )
    latest_path, latest = records[-1]
    additive = latest["additive_transition"]
    failed_attempt = latest.get("failed_attempt")
    expected_pointer = {
        "schema_version": 1,
        "protocol": THROUGHPUT_QUALIFICATION_CURRENT_TRANSITION_PROTOCOL,
        "path": str(latest_path),
        "sha256": _sha256(latest_path),
        "transition_id": latest["transition_id"],
        "from_capacity_generation": additive[
            "from_capacity_generation"
        ],
        "to_capacity_generation": additive["to_capacity_generation"],
        "failed_attempt_id": (
            failed_attempt.get("attempt_id")
            if isinstance(failed_attempt, Mapping)
            else None
        ),
    }
    expected_pointer["pointer_id"] = _sha256_bytes(
        _canonical_json(expected_pointer)
    )
    if set(pointer) != required_pointer or pointer != expected_pointer:
        raise ChainError(
            "current qualification capacity-transition pointer does not bind "
            "the immutable journal head"
        )
    return records


def _validate_qualification_capacity_transition_binding(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    scheduler_evidence: Mapping[str, Any],
    outcome: Mapping[str, Any],
    fresh_states: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    roots = outcome.get("qualification_capacity_transition_roots", [])
    failure_binding = scheduler_evidence.get(
        "qualification_capacity_failure"
    )
    if not roots:
        if failure_binding is not None:
            raise ChainError(
                "sentinel bound a qualification-capacity failure without "
                "its dedicated root"
            )
        return []
    dispositions = outcome.get("dispositions")
    if (
        roots != ["throughput_qualification"]
        or not isinstance(failure_binding, Mapping)
        or not isinstance(dispositions, Mapping)
        or dispositions.get("throughput_qualification")
        != "qualification_capacity_transition_root"
    ):
        raise ChainError(
            "qualification capacity-transition root/binding is malformed"
        )
    receipt_value = _read_json(
        receipt_path,
        description="qualification transition parent receipt",
    )
    receipt_identity = dict(receipt_value)
    receipt_id = receipt_identity.pop("receipt_id", None)
    receipt_rows = [
        row
        for row in receipt_value.get("jobs", [])
        if isinstance(row, Mapping)
        and row.get("name") == "throughput_qualification"
    ]
    scheduler_rows = [
        row
        for row in scheduler_evidence.get("jobs", [])
        if isinstance(row, Mapping)
        and row.get("name") == "throughput_qualification"
    ]
    state = fresh_states.get("throughput_qualification")
    if (
        not isinstance(receipt_id, str)
        or _SHA256.fullmatch(receipt_id) is None
        or receipt_id
        != _sha256_bytes(_canonical_json(receipt_identity))
        or len(receipt_rows) != 1
        or len(scheduler_rows) != 1
        or scheduler_rows[0].get("job_id")
        != receipt_rows[0].get("job_id")
        or scheduler_rows[0].get("comment")
        != receipt_rows[0].get("comment")
        or scheduler_rows[0].get("state") != "FAILED"
        or scheduler_rows[0].get("exit_code") != "76:0"
        or scheduler_rows[0].get("active") is not False
        or not isinstance(state, Mapping)
        or state.get("job_id") != receipt_rows[0].get("job_id")
        or state.get("comment") != receipt_rows[0].get("comment")
        or state.get("job_name") != scheduler_rows[0].get("job_name")
        or state.get("state") != "FAILED"
        or state.get("exit_code") != "76:0"
        or state.get("active") is not False
    ):
        raise ChainError(
            "qualification root is not the exact parent stage-18 FAILED "
            "76:0 job"
        )
    try:
        from scripts import schema5_recovery_sentinel as sentinel_validator
    except ModuleNotFoundError:
        import schema5_recovery_sentinel as sentinel_validator  # type: ignore[no-redef]
    verified = {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "submission_receipt": receipt_value,
        "submission_receipt_path": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
    }
    try:
        strict_failure_binding = (
            sentinel_validator._qualification_capacity_failure_binding(  # noqa: SLF001
                verified=verified,
                jobs=scheduler_evidence.get("jobs", []),
            )
        )
        expected_outcome = sentinel_validator.classify_recovery_jobs(
            scheduler_evidence.get("jobs", []),
            capacity_transient_receipt=None,
            qualification_capacity_failure=strict_failure_binding,
        )
    except (sentinel_validator.SentinelError, OSError, ValueError) as exc:
        raise ChainError(
            f"strict qualification capacity-transition binding failed: {exc}"
        ) from exc
    if (
        strict_failure_binding is None
        or dict(failure_binding) != strict_failure_binding
        or scheduler_evidence.get("capacity_transient_receipt") is not None
        or outcome != expected_outcome
    ):
        raise ChainError(
            "qualification capacity-transition sentinel derivation is invalid"
        )
    readiness_root = _manifest_path(
        manifest.get("readiness_root"),
        description="qualification transition readiness root",
    )
    results_root = _manifest_path(
        manifest.get("results_root"),
        description="qualification transition results root",
    )
    qualification_root = (
        readiness_root / THROUGHPUT_QUALIFICATION_ROOT_NAME
    )
    pointers = _load_qualification_attempt_chain(
        qualification_root=qualification_root,
        results_root=results_root,
        chain_id=str(manifest.get("chain_id", "")),
    )
    pointer_path, pointer = _load_current_qualification_attempt(
        qualification_root=qualification_root,
        pointers=pointers,
    )
    for index, (prior_path, prior_pointer) in enumerate(
        pointers[:-1]
    ):
        _verify_superseded_qualification_failure(
            pointer_path=prior_path,
            pointer=prior_pointer,
            successor=pointers[index + 1][1],
        )
    attempt_root = _require_recursively_read_only(
        Path(str(pointer["attempt_root"])),
        description="failed current throughput-qualification attempt",
    )
    _require_recursively_read_only(
        Path(str(pointer["run_root"])),
        description="failed current throughput-qualification run",
    )
    if (
        qualification_root / THROUGHPUT_QUALIFICATION_MARKER_NAME
    ).exists() or (
        qualification_root / THROUGHPUT_QUALIFICATION_MARKER_NAME
    ).is_symlink():
        raise ChainError(
            "successful qualification cannot authorize a capacity repair"
        )
    failure_path = (
        attempt_root / THROUGHPUT_QUALIFICATION_FAILURE_NAME
    )
    failure_marker, _ = _read_sealed_marker(
        failure_path,
        description="current throughput-qualification failure",
    )
    _require_marker_identity(
        failure_marker,
        identity_field="failure_id",
        description="current throughput-qualification failure",
    )
    scaling = failure_marker.get("additive_scaling_requirement")
    profile = (
        scaling.get("serving_profile")
        if isinstance(scaling, Mapping)
        else None
    )
    expected_tp = 2 if profile == "32B-long" else 1
    attempt_binding = _qualification_attempt_binding(
        pointer_path,
        pointer,
    )
    if (
        set(failure_marker) != _QUALIFICATION_FAILURE_FIELDS
        or failure_marker.get("schema_version")
        != THROUGHPUT_QUALIFICATION_ACCOUNTING_SCHEMA_VERSION
        or failure_marker.get("protocol")
        != THROUGHPUT_QUALIFICATION_FAILURE_PROTOCOL
        or failure_marker.get("passed") is not False
        or failure_marker.get("attempt") != attempt_binding
        or failure_marker.get("readiness_generation")
        != pointer["readiness_generation"]
        or failure_marker.get("scheduler_capacity_mutated") is not False
        or failure_binding.get("admission_certificate_id")
        != (
            failure_marker.get("admission_capacity_certificate", {})
            .get("certificate_id")
            if isinstance(
                failure_marker.get("admission_capacity_certificate"),
                Mapping,
            )
            else None
        )
        or failure_binding.get("failure_kind")
        != "measured_throughput_shortfall"
        or not isinstance(scaling, Mapping)
        or set(scaling) != _QUALIFICATION_SCALING_FIELDS
        or profile not in _QUALIFICATION_SERVING_PROFILES
        or scaling.get("additional_replicas") != 1
        or scaling.get("tensor_parallel_size") != expected_tp
        or scaling.get("additional_gpus") != expected_tp
        or scaling.get("capacity_mutated") is not False
    ):
        raise ChainError(
            "current throughput-qualification failure is not one exact "
            "throughput-only additive request"
        )
    transition_journal = (
        _load_qualification_capacity_transition_journal(
            qualification_root
        )
    )
    if len(transition_journal) != len(pointers):
        raise ChainError(
            "qualification capacity-transition journal and failed-attempt "
            "sequence differ in length"
        )
    for index, ((journal_path, journal_record), (failed_pointer_path, failed_pointer)) in enumerate(
        zip(transition_journal, pointers, strict=True)
    ):
        failed_root = Path(str(failed_pointer["attempt_root"]))
        journal_failure, _ = _read_sealed_marker(
            failed_root / THROUGHPUT_QUALIFICATION_FAILURE_NAME,
            description="journal-bound qualification failure",
        )
        journal_additive = journal_record.get("additive_transition")
        target_readiness = journal_record.get(
            "to_readiness_generation"
        )
        expected_target = (
            pointers[index + 1][1]["readiness_generation"]
            if index + 1 < len(pointers)
            else target_readiness
        )
        if (
            journal_path
            != (
                qualification_root
                / THROUGHPUT_QUALIFICATION_TRANSITION_DIRECTORY
                / (
                    f"c{index + 1:06d}-to-"
                    f"c{index + 2:06d}.json"
                )
            )
            or not isinstance(journal_additive, Mapping)
            or journal_record.get("failed_attempt")
            != _qualification_attempt_binding(
                failed_pointer_path,
                failed_pointer,
            )
            or journal_record.get("failure", {}).get("failure_id")
            != journal_failure.get("failure_id")
            or journal_record.get("from_readiness_generation")
            != failed_pointer["readiness_generation"]
            or target_readiness != expected_target
            or journal_additive.get("from_capacity_generation")
            != failed_pointer["readiness_generation"][
                "capacity_generation"
            ]
            or journal_additive.get("to_capacity_generation")
            != failed_pointer["readiness_generation"][
                "capacity_generation"
            ]
            + 1
        ):
            raise ChainError(
                "qualification capacity-transition journal is not an exact "
                "attempt-to-generation chain"
            )
    marker_path, marker = transition_journal[-1]
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "from_protected_capacity",
        "to_protected_capacity",
        "from_admission_capacity_certificate",
        "to_admission_capacity_certificate",
        "failed_attempt",
        "failure",
        "submission_receipt",
        "failed_stage",
        "paused_control",
        "additive_transition",
        "from_readiness_generation",
        "to_readiness_generation",
        "created_at",
        "created_timestamp",
        "transition_id",
    }
    _require_marker_identity(
        marker,
        identity_field="transition_id",
        description="qualification capacity-transition authority",
    )
    additive = marker.get("additive_transition")
    from_readiness = marker.get("from_readiness_generation")
    readiness = marker.get("to_readiness_generation")
    failed_stage = marker.get("failed_stage")
    receipt_binding = marker.get("submission_receipt")
    failure = marker.get("failure")
    paused = marker.get("paused_control")
    predecessor_readiness = pointer["readiness_generation"]
    readiness = _validate_qualification_readiness(
        readiness,
        description="qualification transition readiness generation",
    )
    readiness_marker_path = _require_canonical_path(
        _manifest_path(
            readiness["marker_path"],
            description="qualification transition trusted-generation marker",
        ),
        description="qualification transition trusted-generation marker",
        kind="file",
    )
    _, readiness_raw = _read_sealed_marker(
        readiness_marker_path,
        description="qualification transition trusted-generation marker",
    )
    intent, _ = _read_sealed_marker(
        attempt_root / "QUALIFICATION_INTENT.json",
        description="failed qualification intent",
    )
    expected_from_protected = intent.get("protected_capacity")
    to_protected = marker.get("to_protected_capacity")
    if (
        not isinstance(to_protected, Mapping)
        or set(to_protected)
        != {"marker", "marker_sha256", "marker_id"}
    ):
        raise ChainError(
            "qualification transition target protected capacity is malformed"
        )
    prerequisites = manifest.get("prerequisite_evidence")
    initial_protected_record = (
        prerequisites.get("protected_capacity")
        if isinstance(prerequisites, Mapping)
        else None
    )
    initial_protected_binding = _qualification_prerequisite_binding(
        initial_protected_record,
        identity_field="marker_id",
        description="initial protected capacity",
    )
    initial_protected_marker, _ = _read_sealed_marker(
        Path(initial_protected_binding["marker"]),
        description="initial protected capacity source authority",
    )
    try:
        to_capacity_contract = protected_capacity.load_contract(
            to_protected["marker"],
            expected_release_git_commit=str(
                manifest.get("release_git_commit", "")
            ),
            expected_release_tag_object=str(
                manifest.get("release_tag_object", "")
            ),
            expected_marker_id=str(to_protected["marker_id"]),
            expected_sha256=str(to_protected["marker_sha256"]),
            expected_source_tree_sha256=str(
                initial_protected_marker["source_tree_sha256"]
            ),
            expected_dispatcher_source_sha256=str(
                initial_protected_marker["dispatcher_source_sha256"]
            ),
            expected_qualification_runner_source_sha256=str(
                initial_protected_marker[
                    "qualification_runner_source_sha256"
                ]
            ),
        )
    except protected_capacity.ProtectedCapacityError as exc:
        raise ChainError(
            f"qualification transition target protected capacity failed: {exc}"
        ) from exc
    from_certificate = failure_marker.get(
        "admission_capacity_certificate"
    )
    to_certificate = marker.get(
        "to_admission_capacity_certificate"
    )
    timestamp = marker.get("created_timestamp")
    paused_identity = dict(paused) if isinstance(paused, Mapping) else {}
    paused_hash = paused_identity.pop("guard_sha256", None)
    if (
        set(marker) != required
        or marker.get("schema_version") != 1
        or marker.get("protocol")
        != THROUGHPUT_QUALIFICATION_TRANSITION_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("release_id") != RELEASE_ID
        or marker.get("release_tag") != RELEASE_TAG
        or marker.get("release_git_commit")
        != manifest.get("release_git_commit")
        or marker.get("release_tag_object")
        != manifest.get("release_tag_object")
        or marker.get("chain_namespace") != CHAIN_NAMESPACE
        or marker.get("chain_id") != manifest.get("chain_id")
        or marker.get("manifest") != str(manifest_path)
        or marker.get("manifest_sha256") != _sha256(manifest_path)
        or marker.get("from_protected_capacity")
        != expected_from_protected
        or marker.get("to_protected_capacity") != dict(to_protected)
        or marker.get("from_admission_capacity_certificate")
        != from_certificate
        or not isinstance(from_certificate, Mapping)
        or not isinstance(to_certificate, Mapping)
        or marker.get("from_readiness_generation")
        != predecessor_readiness
        or from_readiness != predecessor_readiness
        or to_capacity_contract.capacity_generation
        != readiness["capacity_generation"]
        or to_capacity_contract.effective_fleet_contract_sha256
        != readiness["fleet_contract_sha256"]
        or to_certificate.get("path")
        != str(
            to_capacity_contract.static_feasibility_certificate_path
        )
        or to_certificate.get("sha256")
        != to_capacity_contract.static_feasibility_certificate_sha256
        or to_certificate.get("certificate_id")
        != to_capacity_contract.static_feasibility_certificate_id
        or to_certificate.get("capacity_generation")
        != readiness["capacity_generation"]
        or to_certificate.get("effective_fleet_contract_sha256")
        != readiness["fleet_contract_sha256"]
        or not isinstance(failure, Mapping)
        or set(failure) != {"path", "sha256", "failure_id"}
        or failure.get("path") != str(failure_path)
        or failure.get("failure_id")
        != failure_marker.get("failure_id")
        or failure.get("sha256") != _sha256(failure_path)
        or marker.get("failed_attempt")
        != attempt_binding
        or not isinstance(receipt_binding, Mapping)
        or set(receipt_binding) != {"path", "sha256", "receipt_id"}
        or receipt_binding.get("path") != str(receipt_path)
        or receipt_binding.get("sha256") != _sha256(receipt_path)
        or receipt_binding.get("receipt_id") != receipt_id
        or not isinstance(failed_stage, Mapping)
        or set(failed_stage) != {"name", "job_id", "comment"}
        or failed_stage.get("name") != "throughput_qualification"
        or failed_stage.get("job_id") != receipt_rows[0].get("job_id")
        or failed_stage.get("comment") != receipt_rows[0].get("comment")
        or not isinstance(additive, Mapping)
        or set(additive) != _QUALIFICATION_RETRY_FIELDS
        or additive.get("previous_failure_id")
        != failure_marker.get("failure_id")
        or additive.get("serving_profile")
        != profile
        or additive.get("tensor_parallel_size") != expected_tp
        or additive.get("additional_replicas") != 1
        or additive.get("from_capacity_generation")
        != predecessor_readiness["capacity_generation"]
        or additive.get("to_capacity_generation")
        != readiness["capacity_generation"]
        or readiness["capacity_generation"]
        != predecessor_readiness["capacity_generation"] + 1
        or additive.get("from_rollout_generation")
        != predecessor_readiness["rollout_generation"]
        or additive.get("to_rollout_generation")
        != readiness["rollout_generation"]
        or readiness["rollout_generation"]
        <= predecessor_readiness["rollout_generation"]
        or readiness["catalog_id"]
        == predecessor_readiness["catalog_id"]
        or readiness["release_fleet_contract_sha256"]
        != predecessor_readiness["release_fleet_contract_sha256"]
        or additive.get("from_fleet_contract_sha256")
        != predecessor_readiness["fleet_contract_sha256"]
        or additive.get("to_fleet_contract_sha256")
        != readiness["fleet_contract_sha256"]
        or readiness["fleet_contract_sha256"]
        == predecessor_readiness["fleet_contract_sha256"]
        or additive.get("validation")
        != (
            "schema5_control._assert_additive_capacity_contract+"
            "exact-required-profile-delta"
        )
        or _sha256_bytes(readiness_raw) != readiness["marker_sha256"]
        or not isinstance(paused, Mapping)
        or set(paused)
        != {
            "immutable_sha256",
            "desired_state",
            "drain_requested",
            "rollout_generation",
            "production_run_ids",
            "admission",
            "admission_ramp",
            "admission_safety_hold",
            "guard_sha256",
        }
        or paused_hash
        != _sha256_bytes(_canonical_json(paused_identity))
        or paused.get("desired_state") != "paused"
        or paused.get("drain_requested") is not False
        or set(paused.get("production_run_ids", []))
        != set(PRODUCTION_RUN_IDS)
        or paused.get("rollout_generation")
        != readiness["rollout_generation"] - 1
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or marker.get("created_at")
        != datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).isoformat()
    ):
        raise ChainError(
            "qualification capacity-transition repair binding is invalid"
        )
    # The additive transition invalidates the control plane's fleet, smoke, and
    # scheduler-readiness gates. Re-enter at fleet readiness so the new endpoint
    # generation is probed, then execute a fresh generation-scoped 41-cell smoke
    # attempt before qualification can consume the new capacity.
    return [
        "fleet_readiness",
        "smoke_readiness",
        "throughput_qualification",
    ]


def _sentinel_repair_jobs(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    generation: int,
    states: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Consume the generation-scoped sentinel's durable causal classification."""

    generation_name = f"g{generation:04d}"
    root = (
        manifest_path.parent
        / "recovery_chain_sentinels"
        / CHAIN_NAMESPACE
        / generation_name
    )
    marker_path = root / "RECOVERY_SENTINEL_COMPLETE.json"
    evidence_path = root / "SCHEDULER_EVIDENCE.json"
    if not marker_path.exists() or not evidence_path.exists():
        # A node transient can prevent the sentinel itself from publishing.  Repair
        # only that exact job when every production stage proved successful.
        sentinel_state = str(states["failure_sentinel"]["state"])
        production_states = {
            name: str(record["state"])
            for name, record in states.items()
            if name != "failure_sentinel"
        }
        if (
            sentinel_state in _TRANSIENT_REPAIR_ROOT_STATES
            and set(production_states.values()) == {"COMPLETED"}
        ):
            return ["failure_sentinel"]
        raise ChainError(
            "generation-scoped sentinel evidence is absent; recovery outcome is "
            "ambiguous and requires a superseding release"
        )
    for description, path in (
        ("sentinel marker", marker_path),
        ("sentinel scheduler evidence", evidence_path),
    ):
        path = _require_canonical_path(path, description=description, kind="file")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ChainError(f"{description} must be read-only")
    marker = _read_json(marker_path, description="sentinel completion marker")
    evidence = _read_json(evidence_path, description="sentinel scheduler evidence")
    marker_identity = dict(marker)
    marker_id = marker_identity.pop("marker_id", None)
    evidence_identity = dict(evidence)
    evidence_id = evidence_identity.pop("evidence_id", None)
    outcome = evidence.get("outcome")
    if (
        marker.get("schema_version") != 1
        or marker.get("protocol")
        != "schema5-v1.2-r7-recovery-sentinel-outcome"
        or marker.get("passed") is not True
        or marker.get("chain_id") != manifest["chain_id"]
        or marker.get("manifest") != str(manifest_path)
        or marker.get("manifest_sha256") != _sha256(manifest_path)
        or marker.get("submission_receipt") != str(receipt_path)
        or marker.get("submission_receipt_sha256") != _sha256(receipt_path)
        or marker.get("scheduler_evidence") != str(evidence_path)
        or marker.get("scheduler_evidence_sha256") != _sha256(evidence_path)
        or marker.get("scheduler_evidence_id") != evidence_id
        or not isinstance(marker_id, str)
        or marker_id != _sha256_bytes(_canonical_json(marker_identity))
        or evidence.get("schema_version") != 1
        or evidence.get("protocol")
        != "schema5-v1.2-r7-recovery-scheduler-evidence"
        or evidence.get("passed") is not True
        or evidence.get("chain_id") != manifest["chain_id"]
        or evidence.get("manifest") != str(manifest_path)
        or evidence.get("manifest_sha256") != _sha256(manifest_path)
        or evidence.get("submission_receipt") != str(receipt_path)
        or evidence.get("submission_receipt_sha256") != _sha256(receipt_path)
        or not isinstance(evidence_id, str)
        or evidence_id != _sha256_bytes(_canonical_json(evidence_identity))
        or not isinstance(outcome, dict)
        or marker.get("classification") != outcome.get("classification")
        or marker.get("same_generation_repair_allowed")
        is not outcome.get("same_generation_repair_allowed")
        or marker.get("requires_superseding_release")
        is not outcome.get("requires_superseding_release")
    ):
        raise ChainError("generation-scoped sentinel evidence identity is invalid")
    capacity_roots = _validate_capacity_transient_repair_binding(
        manifest=manifest,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        generation=generation,
        scheduler_evidence=evidence,
        outcome=outcome,
        fresh_states=states,
    )
    qualification_roots = (
        _validate_qualification_capacity_transition_binding(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            scheduler_evidence=evidence,
            outcome=outcome,
            fresh_states=states,
        )
    )
    classification = outcome.get("classification")
    if classification == "complete":
        return []
    transition_ready = (
        classification == "qualification_capacity_transition_required"
        and qualification_roots
        == [
            "fleet_readiness",
            "smoke_readiness",
            "throughput_qualification",
        ]
        and outcome.get("same_generation_repair_allowed") is False
        and outcome.get("requires_superseding_release") is False
    )
    transient_ready = (
        classification == "transient_repairable"
        and outcome.get("same_generation_repair_allowed") is True
        and outcome.get("requires_superseding_release") is False
    )
    if not (transition_ready or transient_ready):
        raise ChainError(
            "sentinel classified this generation as requiring a superseding release"
        )
    selected_roots = [
        *outcome.get("transient_roots", []),
        *capacity_roots,
        *qualification_roots,
        *outcome.get("dependency_cancelled_suffix", []),
    ]
    order = [str(row["name"]) for row in manifest["jobs"]]
    selected = [*selected_roots]
    for stage in PRODUCTION_STAGE_NAMES:
        observer = f"{STAGE_SENTINEL_PREFIX}{stage}"
        if stage in selected_roots and observer in order:
            selected.append(observer)
    selected.append("failure_sentinel")
    # A repaired observer can itself be a scheduler-transient root.  Production
    # stages are deliberately not redrawn in that case.
    selected = list(dict.fromkeys(selected))
    if (
        any(not isinstance(name, str) for name in selected)
        or not set(selected) <= set(order)
    ):
        raise ChainError("sentinel repair suffix is malformed")
    return [name for name in order if name in set(selected)]


_BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "NODE_FAIL",
    "PREEMPTED",
    "REVOKED",
}


def _bootstrap_anchor_launch_authorization(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Return the immutable initial launch authorization, if it exists.

    A repair generation may inherit only the original marker-last decision to
    launch this exact chain.  An absent pair means the chain is still in its
    prelaunch held phase; a partial or invalid pair is an integrity failure.
    """

    receipt_path = manifest_path.parent / SUBMISSION_RECEIPT_NAME
    receipt = _validate_submission_receipt(
        receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
        comments=_submission_comments(manifest),
    )
    release_path = manifest_path.parent / ROOT_RELEASE_COMPLETE_NAME
    launch_path = manifest_path.parent / LAUNCH_COMPLETE_NAME
    release_exists = release_path.exists() or release_path.is_symlink()
    launch_exists = launch_path.exists() or launch_path.is_symlink()
    if not release_exists and not launch_exists:
        return None
    if release_exists != launch_exists:
        raise ChainError(
            "bootstrap anchor launch authorization is only partially published"
        )
    root_record = next(
        row for row in receipt["jobs"] if row["name"] == "source_checkout"
    )
    release = _validate_root_release_complete(
        release_path,
        receipt_path=receipt_path,
        receipt=receipt,
        root_record=root_record,
    )
    launch = _validate_launch_complete(
        launch_path,
        receipt_path=receipt_path,
        receipt=receipt,
        release_path=release_path,
        release=release,
    )
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "root_release": str(release_path),
        "root_release_sha256": _sha256(release_path),
        "root_release_id": release["release_id"],
        "launch_complete": str(launch_path),
        "launch_complete_sha256": _sha256(launch_path),
        "launch_id": launch["launch_id"],
    }


def _bootstrap_generation_provenance_path(
    *, manifest_path: Path, generation: int
) -> Path:
    if generation == 0:
        return manifest_path.parent / BOOTSTRAP_GENERATION_PROVENANCE_NAME
    return (
        manifest_path.parent
        / REPAIR_ROOT_NAME
        / f"g{generation:04d}"
        / BOOTSTRAP_GENERATION_PROVENANCE_NAME
    )


def _bootstrap_live_job_provenance(
    *,
    manifest_row: Mapping[str, Any],
    receipt_row: Mapping[str, Any],
    state: Mapping[str, Any],
    runner: Runner,
    spool_path: Path,
    origin_generation: int,
    generation_root_names: Sequence[str],
) -> dict[str, Any]:
    name = str(manifest_row["name"])
    submit_line = state.get("submit_line")
    try:
        submit_argv = (
            shlex.split(submit_line)
            if isinstance(submit_line, str)
            else []
        )
    except ValueError as exc:
        raise ChainError(
            f"bootstrap SubmitLine is malformed for {name}: {exc}"
        ) from exc
    expected_submit_argv = submission_argv(
        manifest_row,
        dependency_job_ids=[
            str(value)
            for value in receipt_row.get("dependency_job_ids", [])
        ],
        comment=str(receipt_row["comment"]),
        initial_hold=name in set(generation_root_names),
    )
    if submit_argv != expected_submit_argv:
        raise ChainError(f"bootstrap SubmitLine drifted for {name}")
    details = _show_recovery_job(str(state["job_id"]), runner=runner)
    fields = details["fields"]
    expected_dependency_pairs = sorted(
        (
            str(receipt_row["dependency_type"]),
            str(dependency_id),
        )
        for dependency_id in receipt_row["dependency_job_ids"]
    )
    observed_dependency_pairs = sorted(
        _scheduler_dependency_pairs(str(fields.get("Dependency", "")))
    )
    if (
        details["returncode"] != 0
        or fields.get("JobId") != state["job_id"]
        or fields.get("JobName") != state["job_name"]
        or fields.get("Comment") != state["comment"]
        or fields.get("Command") != receipt_row["script"]
        or fields.get("Requeue") != "0"
        or observed_dependency_pairs != expected_dependency_pairs
        or (
            name in set(generation_root_names)
            and str(fields.get("Reason", "")).lower() != "jobhelduser"
        )
        or (
            name not in set(generation_root_names)
            and str(fields.get("Reason", "")).lower()
            not in {"dependency", "dependencyneverSatisfied".lower()}
        )
    ):
        raise ChainError(
            f"scontrol acceptance provenance drifted for {name}"
        )
    spool = runner(
        [
            "scontrol",
            "write",
            "batch_script",
            str(state["job_id"]),
            str(spool_path),
        ]
    )
    if (
        spool.returncode != 0
        or not spool_path.is_file()
        or spool_path.is_symlink()
    ):
        raise ChainError(
            f"cannot capture bootstrap spooled script for {name}"
        )
    local_script = _require_canonical_path(
        Path(str(receipt_row["script"])),
        description=f"bootstrap local sbatch {name}",
        kind="file",
    )
    spooled_sha256 = _sha256(spool_path)
    local_sha256 = _sha256(local_script)
    if (
        local_sha256 != receipt_row["script_sha256"]
        or spooled_sha256 != local_sha256
        or spool_path.read_bytes() != local_script.read_bytes()
    ):
        raise ChainError(f"spooled sbatch differs for {name}")
    return {
        "name": name,
        "job_id": state["job_id"],
        "comment": state["comment"],
        "job_name": state["job_name"],
        "script_sha256": local_sha256,
        "submit_line_sha256": _sha256_bytes(
            str(submit_line).encode("utf-8")
        ),
        "submit_argv": submit_argv,
        "submit_line_exact": True,
        "scontrol_command": fields["Command"],
        "scontrol_requeue": 0,
        "scontrol_dependency_pairs": [
            {"type": dependency_type, "job_id": dependency_id}
            for dependency_type, dependency_id in observed_dependency_pairs
        ],
        "scontrol_reason": str(fields.get("Reason", "")),
        "dependency_type": receipt_row["dependency_type"],
        "dependency_job_ids": list(receipt_row["dependency_job_ids"]),
        "initial_hold": name in set(generation_root_names),
        "spooled_script_sha256": spooled_sha256,
        "spooled_script_exact_match": True,
        "origin_generation": origin_generation,
    }


def _validate_bootstrap_generation_provenance(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
) -> dict[str, Any]:
    path = _require_canonical_path(
        path,
        description="bootstrap generation scheduler provenance",
        kind="file",
    )
    payload = _read_json(
        path, description="bootstrap generation scheduler provenance"
    )
    identity = dict(payload)
    provenance_id = identity.pop("provenance_id", None)
    parent_fields = (
        {
            "parent_provenance": None,
            "parent_provenance_sha256": None,
            "parent_provenance_id": None,
        }
        if generation == 0
        else None
    )
    if generation > 0:
        parent_receipt_path = _require_canonical_path(
            Path(str(receipt["parent_receipt"])),
            description="bootstrap parent receipt",
            kind="file",
        )
        parent_receipt = _read_json(
            parent_receipt_path,
            description="bootstrap parent receipt",
        )
        parent_generation = int(parent_receipt.get("repair_generation", 0))
        parent_path = _bootstrap_generation_provenance_path(
            manifest_path=manifest_path,
            generation=parent_generation,
        )
        parent = _validate_bootstrap_generation_provenance(
            parent_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=parent_receipt,
            receipt_path=parent_receipt_path,
            generation=parent_generation,
        )
        parent_fields = {
            "parent_provenance": str(parent_path),
            "parent_provenance_sha256": _sha256(parent_path),
            "parent_provenance_id": parent["provenance_id"],
        }
    lineage = _bootstrap_receipt_lineage(
        receipt=receipt,
        receipt_path=receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    expected_lineage_receipt_ids = [
        str(lineage_receipt["receipt_id"])
        for _path, lineage_receipt in lineage
    ]
    expected_lineage_job_ids = sorted(
        {
            str(row["job_id"])
            for _path, lineage_receipt in lineage
            for row in lineage_receipt["jobs"]
        },
        key=int,
    )
    origin_roots: dict[int, set[str]] = {}
    for lineage_path, lineage_receipt in lineage:
        lineage_generation = int(
            lineage_receipt.get("repair_generation", 0)
        )
        origin_roots[lineage_generation] = set(
            ["source_checkout"]
            if lineage_generation == 0
            else lineage_receipt.get("held_root_names", [])
        )
        if (
            not origin_roots[lineage_generation]
            or any(
                root not in {str(row["name"]) for row in manifest["jobs"]}
                for root in origin_roots[lineage_generation]
            )
        ):
            raise ChainError(
                "bootstrap provenance lineage has no generation root: "
                f"{lineage_path}"
            )
    generation_root_names = (
        ["source_checkout"]
        if generation == 0
        else list(receipt.get("held_root_names", []))
    )
    if not generation_root_names:
        raise ChainError(
            "bootstrap generation provenance has no held roots"
        )
    expected_fields = {
        "schema_version",
        "protocol",
        "passed",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "repair_generation",
        "parent_provenance",
        "parent_provenance_sha256",
        "parent_provenance_id",
        "namespace_lineage_receipt_ids",
        "namespace_bound_job_ids",
        "namespace_scan_complete",
        "generation_root_names",
        "current_generation_names",
        "scheduler_topology_valid",
        "jobs",
        "provenance_id",
    }
    jobs = payload.get("jobs")
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != 1
        or payload.get("protocol")
        != "schema5-v1.2-r7-bootstrap-generation-provenance-v1"
        or payload.get("passed") is not True
        or payload.get("release_git_commit")
        != manifest["release_git_commit"]
        or payload.get("release_tag_object")
        != manifest["release_tag_object"]
        or payload.get("chain_id") != manifest["chain_id"]
        or payload.get("chain_manifest") != str(manifest_path)
        or payload.get("chain_manifest_sha256") != _sha256(manifest_path)
        or payload.get("submission_receipt") != str(receipt_path)
        or payload.get("submission_receipt_sha256") != _sha256(receipt_path)
        or payload.get("submission_receipt_id") != receipt["receipt_id"]
        or payload.get("repair_generation") != generation
        or any(
            payload.get(field) != value
            for field, value in parent_fields.items()
        )
        or payload.get("namespace_scan_complete") is not True
        or payload.get("namespace_lineage_receipt_ids")
        != expected_lineage_receipt_ids
        or payload.get("namespace_bound_job_ids")
        != expected_lineage_job_ids
        or payload.get("generation_root_names") != generation_root_names
        or payload.get("current_generation_names")
        != [
            str(row["name"])
            for row in receipt["jobs"]
            if generation == 0
            or int(row.get("generation", 0)) == generation
        ]
        or payload.get("scheduler_topology_valid") is not True
        or not isinstance(jobs, list)
        or len(jobs) != len(manifest["jobs"])
        or not isinstance(provenance_id, str)
        or _SHA256.fullmatch(provenance_id) is None
        or provenance_id != _sha256_bytes(_canonical_json(identity))
        or stat.S_IMODE(path.stat().st_mode) & 0o222
        or path.stat().st_nlink != 1
    ):
        raise ChainError(
            "bootstrap generation scheduler provenance is invalid"
        )
    for job, manifest_row, receipt_row in zip(
        jobs, manifest["jobs"], receipt["jobs"], strict=True
    ):
        origin_generation = int(receipt_row.get("generation", 0))
        if (
            not isinstance(job, dict)
            or set(job)
            != {
                "name",
                "job_id",
                "comment",
                "job_name",
                "script_sha256",
                "submit_line_sha256",
                "submit_argv",
                "submit_line_exact",
                "scontrol_command",
                "scontrol_requeue",
                "scontrol_dependency_pairs",
                "scontrol_reason",
                "dependency_type",
                "dependency_job_ids",
                "initial_hold",
                "spooled_script_sha256",
                "spooled_script_exact_match",
                "origin_generation",
            }
            or job.get("name") != manifest_row["name"]
            or job.get("job_id") != receipt_row["job_id"]
            or job.get("comment") != receipt_row["comment"]
            or job.get("job_name") != manifest_row["job_name"]
            or job.get("script_sha256")
            != manifest_row["script_sha256"]
            or job.get("scontrol_command") != receipt_row["script"]
            or job.get("scontrol_requeue") != 0
            or job.get("dependency_type")
            != receipt_row["dependency_type"]
            or job.get("dependency_job_ids")
            != receipt_row["dependency_job_ids"]
            or job.get("initial_hold")
            is not (
                manifest_row["name"]
                in origin_roots.get(origin_generation, set())
            )
            or job.get("submit_argv")
            != submission_argv(
                manifest_row,
                dependency_job_ids=receipt_row[
                    "dependency_job_ids"
                ],
                comment=receipt_row["comment"],
                initial_hold=bool(job.get("initial_hold")),
            )
            or job.get("scontrol_dependency_pairs")
            != [
                {"type": dependency_type, "job_id": dependency_id}
                for dependency_type, dependency_id in sorted(
                    (
                        receipt_row["dependency_type"],
                        str(dependency_id),
                    )
                    for dependency_id in receipt_row[
                        "dependency_job_ids"
                    ]
                )
            ]
            or not isinstance(job.get("scontrol_reason"), str)
            or (
                job.get("initial_hold") is True
                and str(job.get("scontrol_reason", "")).lower()
                != "jobhelduser"
            )
            or job.get("spooled_script_sha256")
            != manifest_row["script_sha256"]
            or job.get("submit_line_exact") is not True
            or job.get("spooled_script_exact_match") is not True
            or _SHA256.fullmatch(
                str(job.get("submit_line_sha256", ""))
            )
            is None
            or job.get("origin_generation") != origin_generation
            or not 0 <= origin_generation <= generation
        ):
            raise ChainError(
                "bootstrap generation job provenance drifted"
            )
    return payload


def _ensure_bootstrap_generation_provenance(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
    states: Mapping[str, Mapping[str, Any]],
    runner: Runner,
) -> dict[str, Any]:
    path = _bootstrap_generation_provenance_path(
        manifest_path=manifest_path, generation=generation
    )
    if path.exists() or path.is_symlink():
        validated = _validate_bootstrap_generation_provenance(
            path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
        )
        _validate_bootstrap_scheduler_namespace(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            runner=runner,
        )
        release_path = receipt_path.parent / ROOT_RELEASE_COMPLETE_NAME
        launch_path = receipt_path.parent / LAUNCH_COMPLETE_NAME
        if (
            release_path.exists()
            or release_path.is_symlink()
            or launch_path.exists()
            or launch_path.is_symlink()
        ):
            if not (
                (release_path.exists() or release_path.is_symlink())
                and (launch_path.exists() or launch_path.is_symlink())
            ):
                raise ChainError(
                    "bootstrap generation launch markers are only partially "
                    "published"
                )
            root_record = _generation_root_records(
                receipt, generation=generation
            )[0]
            release = _validate_root_release_complete(
                release_path,
                receipt_path=receipt_path,
                receipt=receipt,
                root_record=root_record,
            )
            _validate_launch_complete(
                launch_path,
                receipt_path=receipt_path,
                receipt=receipt,
                release_path=release_path,
                release=release,
            )
            # Scheduler acceptance is deliberately frozen while every frontier
            # root is held.  Once the marker-last launch transaction exists,
            # replay authenticates that sealed cut plus complete namespace
            # truth; it must not demand that a correctly released root return
            # to JobHeldUser or recapture a mutable post-release spool state.
            return validated
        release_intent_path = (
            receipt_path.parent / ROOT_RELEASE_INTENT_NAME
        )
        if (
            release_intent_path.exists()
            or release_intent_path.is_symlink()
        ):
            release_intent_path = _require_canonical_path(
                release_intent_path,
                description="bootstrap generation release intent",
                kind="file",
            )
            release_intent = _read_json(
                release_intent_path,
                description="bootstrap generation release intent",
            )
            root_records = _generation_root_records(
                receipt, generation=generation
            )
            root_names = [
                str(record["name"]) for record in root_records
            ]
            root_job_ids = [
                str(record["job_id"]) for record in root_records
            ]
            root_comments = [
                str(record["comment"]) for record in root_records
            ]
            if (
                release_intent.get("receipt") != str(receipt_path)
                or release_intent.get("receipt_sha256")
                != _sha256(receipt_path)
                or release_intent.get("receipt_id")
                != receipt["receipt_id"]
                or release_intent.get("root_names") != root_names
                or release_intent.get("root_job_ids") != root_job_ids
                or release_intent.get("root_comments") != root_comments
                or release_intent.get("commands")
                != [
                    ["scontrol", "release", job_id]
                    for job_id in root_job_ids
                ]
            ):
                raise ChainError(
                    "bootstrap generation release intent drifted"
                )
            attempts_root = receipt_path.parent / "root_release_attempts"
            attempts = (
                sorted(attempts_root.glob("attempt-*.intent.json"))
                if attempts_root.is_dir()
                and not attempts_root.is_symlink()
                else []
            )
            attempted: set[str] = set()
            for index, attempt_path in enumerate(attempts, start=1):
                attempt = _read_json(
                    _require_canonical_path(
                        attempt_path,
                        description=(
                            "bootstrap generation release attempt"
                        ),
                        kind="file",
                    ),
                    description="bootstrap generation release attempt",
                )
                name = str(attempt.get("root_name", ""))
                if (
                    attempt_path.name
                    != f"attempt-{index:04d}.intent.json"
                    or name not in root_names
                    or attempt.get("attempt") != index
                    or attempt.get("job_id")
                    != root_job_ids[root_names.index(name)]
                    or attempt.get("command")
                    != release_intent["commands"][
                        root_names.index(name)
                    ]
                ):
                    raise ChainError(
                        "bootstrap generation release attempt drifted"
                    )
                attempted.add(name)
            current_names = set(validated["current_generation_names"])
            invalid_transition = []
            for name in current_names:
                state = states[name]
                if name in root_names:
                    if name not in attempted:
                        details = _show_recovery_job(
                            str(state["job_id"]), runner=runner
                        )
                        held = (
                            state.get("state") == "PENDING"
                            and state.get("active") is True
                            and details["returncode"] == 0
                            and str(
                                details["fields"].get("Reason", "")
                            ).lower()
                            == "jobhelduser"
                        )
                        if not held:
                            invalid_transition.append(name)
                elif (
                    state.get("state") != "PENDING"
                    or state.get("active") is not True
                ):
                    invalid_transition.append(name)
            if invalid_transition:
                raise ChainError(
                    "bootstrap release boundary contains an unjournaled "
                    f"scheduler transition: {sorted(invalid_transition)}"
                )
            return validated
        state_values = {
            str(item.get("state")) for item in states.values()
        }
        if (
            len(states) == len(manifest["jobs"])
            and all(item.get("active") is False for item in states.values())
            and state_values
            <= _BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES
            | {"COMPLETED"}
            and bool(
                state_values
                & _BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES
            )
        ):
            # Recovery-namespace cancellation destroys the held topology but not
            # its already sealed acceptance. Two scheduler-complete observations
            # of this exact terminal namespace authorize only a new held
            # descendant generation.
            return validated
        generation_root_names = list(validated["generation_root_names"])
        current_names = set(validated["current_generation_names"])
        missing_active = sorted(
            name
            for name in current_names
            if states[name].get("state") == "PENDING"
            and states[name].get("active") is not True
        )
        if missing_active:
            raise ChainError(
                "squeue set is incomplete for bootstrap generation jobs: "
                f"{missing_active}"
            )
        missing_accounting = sorted(
            name
            for name in current_names
            if states[name].get("active") is True
            and not isinstance(states[name].get("submit_line"), str)
        )
        if missing_accounting:
            raise ChainError(
                "sacct set is incomplete for bootstrap generation jobs: "
                f"{missing_accounting}"
            )
        if any(
            (
                states[name].get("state") != "PENDING"
                or states[name].get("active") is not True
            )
            if name in current_names
            else (
                states[name].get("state") != "COMPLETED"
                or states[name].get("active") is not False
            )
            for name in states
        ):
            raise ChainError(
                "live bootstrap generation topology drifted from its "
                "sealed acceptance"
            )
        sealed_by_name = {
            str(row["name"]): row for row in validated["jobs"]
        }
        receipt_by_name = {
            str(row["name"]): row for row in receipt["jobs"]
        }
        manifest_by_name = {
            str(row["name"]): row for row in manifest["jobs"]
        }
        with tempfile.TemporaryDirectory(
            prefix="schema5-bootstrap-acceptance-replay-"
        ) as spool_directory:
            spool_root = Path(spool_directory)
            for index, name in enumerate(validated["current_generation_names"]):
                current = _bootstrap_live_job_provenance(
                    manifest_row=manifest_by_name[name],
                    receipt_row=receipt_by_name[name],
                    state=states[name],
                    runner=runner,
                    spool_path=spool_root / f"{index:02d}-{name}.sbatch",
                    origin_generation=int(
                        receipt_by_name[name].get("generation", 0)
                    ),
                    generation_root_names=generation_root_names,
                )
                if current != sealed_by_name[name]:
                    raise ChainError(
                        "live bootstrap scheduler acceptance drifted for "
                        f"{name}"
                    )
        return validated
    namespace = _validate_bootstrap_scheduler_namespace(
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        runner=runner,
    )
    parent: dict[str, Any] | None = None
    parent_path: Path | None = None
    if generation > 0:
        parent_receipt_path = _require_canonical_path(
            Path(str(receipt["parent_receipt"])),
            description="bootstrap parent receipt",
            kind="file",
        )
        parent_receipt = _read_json(
            parent_receipt_path,
            description="bootstrap parent receipt",
        )
        parent_generation = int(parent_receipt.get("repair_generation", 0))
        parent_path = _bootstrap_generation_provenance_path(
            manifest_path=manifest_path,
            generation=parent_generation,
        )
        parent = _validate_bootstrap_generation_provenance(
            parent_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=parent_receipt,
            receipt_path=parent_receipt_path,
            generation=parent_generation,
        )
    parent_jobs = (
        {}
        if parent is None
        else {str(row["name"]): row for row in parent["jobs"]}
    )
    generation_root_names = (
        ["source_checkout"]
        if generation == 0
        else list(receipt.get("held_root_names", []))
    )
    if not generation_root_names:
        raise ChainError(
            "bootstrap generation has no exact held frontier"
        )
    current_generation_names = {
        str(row["name"])
        for row in receipt["jobs"]
        if int(row.get("generation", 0)) == generation
    }
    if generation == 0:
        current_generation_names = {
            str(row["name"]) for row in receipt["jobs"]
        }
    if not current_generation_names or any(
        (
            states[name].get("state") != "PENDING"
            or states[name].get("active") is not True
        )
        if name in current_generation_names
        else (
            states[name].get("state") != "COMPLETED"
            or states[name].get("active") is not False
        )
        for name in states
    ):
        raise ChainError(
            "bootstrap generation scheduler topology is not one held "
            "PENDING suffix over completed ancestors"
        )
    jobs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="schema5-bootstrap-provenance-"
    ) as spool_directory:
        spool_root = Path(spool_directory)
        for index, (manifest_row, receipt_row) in enumerate(
            zip(manifest["jobs"], receipt["jobs"], strict=True)
        ):
            name = str(manifest_row["name"])
            origin_generation = int(receipt_row.get("generation", 0))
            if origin_generation < generation:
                inherited = parent_jobs.get(name)
                if (
                    not isinstance(inherited, dict)
                    or inherited.get("job_id") != receipt_row["job_id"]
                    or inherited.get("origin_generation")
                    != origin_generation
                ):
                    raise ChainError(
                        f"bootstrap inherited provenance drifted for {name}"
                    )
                jobs.append(dict(inherited))
            else:
                jobs.append(
                    _bootstrap_live_job_provenance(
                        manifest_row=manifest_row,
                        receipt_row=receipt_row,
                        state=states[name],
                        runner=runner,
                        spool_path=spool_root / f"{index:02d}-{name}.sbatch",
                        origin_generation=origin_generation,
                        generation_root_names=generation_root_names,
                    )
                )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r7-bootstrap-generation-provenance-v1",
        "passed": True,
        "release_git_commit": manifest["release_git_commit"],
        "release_tag_object": manifest["release_tag_object"],
        "chain_id": manifest["chain_id"],
        "chain_manifest": str(manifest_path),
        "chain_manifest_sha256": _sha256(manifest_path),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "repair_generation": generation,
        "parent_provenance": (
            None if parent_path is None else str(parent_path)
        ),
        "parent_provenance_sha256": (
            None if parent_path is None else _sha256(parent_path)
        ),
        "parent_provenance_id": (
            None if parent is None else parent["provenance_id"]
        ),
        "namespace_lineage_receipt_ids": namespace[
            "lineage_receipt_ids"
        ],
        "namespace_bound_job_ids": namespace["bound_job_ids"],
        "namespace_scan_complete": True,
        "generation_root_names": generation_root_names,
        "current_generation_names": sorted(
            current_generation_names,
            key=lambda value: [
                str(row["name"]) for row in manifest["jobs"]
            ].index(value),
        ),
        "scheduler_topology_valid": True,
        "jobs": jobs,
    }
    payload["provenance_id"] = _sha256_bytes(_canonical_json(payload))
    _write_immutable_json_once(
        path,
        payload,
        description="marker-last bootstrap generation scheduler provenance",
    )
    return _validate_bootstrap_generation_provenance(
        path,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
    )


def _validate_bootstrap_status_payload(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
    provenance: Mapping[str, Any],
) -> None:
    """Validate one status cut's exact receipt ancestry and job-ID closure."""

    required = {
        "schema_version",
        "protocol",
        "passed",
        "observed_at_timestamp",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "anchor_submission_receipt",
        "anchor_submission_receipt_sha256",
        "anchor_submission_receipt_id",
        "descendant_chain_validated",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "root_held",
        "squeue_complete",
        "sacct_complete",
        "namespace_scan_complete",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
        "ambiguous_jobs",
        "job_count",
        "jobs",
        "recovery_namespace_cancelled",
        "anchor_launch_authorized",
        "anchor_launch_id",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "descendant_release_required",
        "handoff_complete",
        "watchdog_scientific_jobs_submitted",
        "observation_id",
    }
    identity = dict(payload)
    observation_id = identity.pop("observation_id", None)
    lineage = _bootstrap_receipt_lineage(
        receipt=receipt,
        receipt_path=receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    expected_lineage_ids = [
        str(lineage_receipt["receipt_id"])
        for _path, lineage_receipt in lineage
    ]
    expected_bound_ids = sorted(
        {
            str(row["job_id"])
            for _path, lineage_receipt in lineage
            for row in lineage_receipt["jobs"]
        },
        key=int,
    )
    expected_lineage_job_ids = [
        [str(row["job_id"]) for row in lineage_receipt["jobs"]]
        for _path, lineage_receipt in lineage
    ]
    root_records = _generation_root_records(
        receipt, generation=generation
    )
    expected_root_names = [
        str(record["name"]) for record in root_records
    ]
    expected_root_job_ids = [
        str(record["job_id"]) for record in root_records
    ]
    expected_jobs = receipt.get("jobs")
    status_jobs = payload.get("jobs")
    anchor_path, anchor_receipt = lineage[0]
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("protocol")
        != "schema5-v1.2-r7-bootstrap-status-v1"
        or payload.get("passed") is not True
        or payload.get("release_git_commit")
        != manifest["release_git_commit"]
        or payload.get("release_tag_object")
        != manifest["release_tag_object"]
        or payload.get("chain_id") != manifest["chain_id"]
        or payload.get("chain_manifest") != str(manifest_path)
        or payload.get("chain_manifest_sha256") != _sha256(manifest_path)
        or payload.get("submission_receipt") != str(receipt_path)
        or payload.get("submission_receipt_sha256")
        != _sha256(receipt_path)
        or payload.get("submission_receipt_id")
        != receipt["receipt_id"]
        or payload.get("repair_generation") != generation
        or payload.get("generation_provenance")
        != str(
            _bootstrap_generation_provenance_path(
                manifest_path=manifest_path, generation=generation
            )
        )
        or payload.get("generation_provenance_sha256")
        != _sha256(
            _bootstrap_generation_provenance_path(
                manifest_path=manifest_path, generation=generation
            )
        )
        or payload.get("generation_provenance_id")
        != provenance["provenance_id"]
        or payload.get("anchor_submission_receipt") != str(anchor_path)
        or payload.get("anchor_submission_receipt_sha256")
        != _sha256(anchor_path)
        or payload.get("anchor_submission_receipt_id")
        != anchor_receipt["receipt_id"]
        or payload.get("descendant_chain_validated") is not True
        or payload.get("namespace_scan_complete") is not True
        or payload.get("namespace_lineage_receipt_ids")
        != expected_lineage_ids
        or payload.get("namespace_lineage_job_ids")
        != expected_lineage_job_ids
        or payload.get("namespace_bound_job_ids")
        != expected_bound_ids
        or provenance.get("namespace_lineage_receipt_ids")
        != expected_lineage_ids
        or provenance.get("namespace_bound_job_ids")
        != expected_bound_ids
        or payload.get("root_names") != expected_root_names
        or payload.get("root_job_ids") != expected_root_job_ids
        or payload.get("root_name") != expected_root_names[0]
        or payload.get("root_job_id") != expected_root_job_ids[0]
        or payload.get("roots_held") is not payload.get("root_held")
        or payload.get("job_count") != len(manifest["jobs"])
        or not isinstance(expected_jobs, list)
        or not isinstance(status_jobs, list)
        or len(status_jobs) != len(expected_jobs)
        or not isinstance(observation_id, str)
        or _SHA256.fullmatch(observation_id) is None
        or observation_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise ChainError(
            "bootstrap status receipt ancestry or bound-ID contract drifted"
        )
    provenance_by_name = {
        str(row["name"]): row
        for row in provenance["jobs"]
    }
    status_job_fields = {
        "name",
        "job_id",
        "comment",
        "job_name",
        "state",
        "active",
        "script_sha256",
        "submit_line_sha256",
        "submit_line_exact",
        "scontrol_command",
        "scontrol_requeue",
        "spooled_script_sha256",
        "spooled_script_exact_match",
    }
    observed_current_ids: set[str] = set()
    for status_row, receipt_row, manifest_row in zip(
        status_jobs, expected_jobs, manifest["jobs"], strict=True
    ):
        sealed = provenance_by_name.get(str(receipt_row["name"]))
        job_id = (
            str(status_row.get("job_id", ""))
            if isinstance(status_row, Mapping)
            else ""
        )
        if (
            not isinstance(status_row, Mapping)
            or set(status_row) != status_job_fields
            or not isinstance(sealed, Mapping)
            or status_row.get("name") != receipt_row["name"]
            or job_id != str(receipt_row["job_id"])
            or not job_id.isdigit()
            or job_id in observed_current_ids
            or status_row.get("comment") != receipt_row["comment"]
            or status_row.get("job_name") != manifest_row["job_name"]
            or any(
                status_row.get(field) != sealed.get(field)
                for field in (
                    "name",
                    "job_id",
                    "comment",
                    "job_name",
                    "script_sha256",
                    "submit_line_sha256",
                    "submit_line_exact",
                    "scontrol_command",
                    "scontrol_requeue",
                    "spooled_script_sha256",
                    "spooled_script_exact_match",
                )
            )
            or not isinstance(status_row.get("state"), str)
            or not isinstance(status_row.get("active"), bool)
        ):
            raise ChainError(
                "bootstrap status current-generation job identity drifted"
            )
        observed_current_ids.add(job_id)
    if not observed_current_ids <= set(expected_bound_ids):
        raise ChainError(
            "bootstrap status current job IDs escape the bound ancestry"
        )


def _validate_bootstrap_observation_pair(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    generation: int,
    require_cancelled: bool,
    require_release: bool,
    require_rearm: bool = False,
) -> list[dict[str, Any]]:
    observation_root = (
        receipt_path.parent / BOOTSTRAP_OBSERVATIONS_ROOT_NAME
    )
    observations = (
        sorted(observation_root.glob("observation-*.json"))
        if observation_root.is_dir() and not observation_root.is_symlink()
        else []
    )
    if len(observations) < 2:
        raise ChainError(
            "bootstrap repair requires two persisted observations"
        )
    selected = observations[-2:]
    payloads = [
        _read_json(path, description="bootstrap scheduler observation")
        for path in selected
    ]
    expected_same = (
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "anchor_launch_authorized",
        "anchor_launch_id",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "descendant_release_required",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
    )
    if (
        any(
            stat.S_IMODE(path.stat().st_mode) & 0o222
            or path.stat().st_nlink != 1
            for path in selected
        )
        or any(
            payload.get("protocol")
            != "schema5-v1.2-r7-bootstrap-status-v1"
            or payload.get("passed") is not True
            or payload.get("release_git_commit")
            != manifest["release_git_commit"]
            or payload.get("release_tag_object")
            != manifest["release_tag_object"]
            or payload.get("chain_id") != manifest["chain_id"]
            or payload.get("submission_receipt_id")
            != receipt.get("receipt_id")
            or payload.get("submission_receipt_sha256")
            != _sha256(receipt_path)
            or payload.get("repair_generation") != generation
            or payload.get("squeue_complete") is not True
            or payload.get("sacct_complete") is not True
            or payload.get("ambiguous_jobs") != 0
            or payload.get("handoff_complete") is not False
            or payload.get("watchdog_scientific_jobs_submitted") != 0
            or payload.get("observation_id")
            != _sha256_bytes(
                _canonical_json(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "observation_id"
                    }
                )
            )
            or (
                require_cancelled
                and payload.get("recovery_namespace_cancelled") is not True
            )
            or (
                require_release
                and payload.get("descendant_release_required") is not True
            )
            or (
                require_rearm
                and payload.get("descendant_rearm_required") is not True
            )
            for payload in payloads
        )
        or any(
            payloads[0].get(field) != payloads[1].get(field)
            for field in expected_same
        )
        or not isinstance(
            payloads[0].get("observed_at_timestamp"), (int, float)
        )
        or isinstance(
            payloads[0].get("observed_at_timestamp"), bool
        )
        or not isinstance(
            payloads[1].get("observed_at_timestamp"), (int, float)
        )
        or isinstance(
            payloads[1].get("observed_at_timestamp"), bool
        )
        or not math.isfinite(
            float(payloads[0]["observed_at_timestamp"])
        )
        or not math.isfinite(
            float(payloads[1]["observed_at_timestamp"])
        )
        or float(payloads[1]["observed_at_timestamp"])
        - float(payloads[0]["observed_at_timestamp"])
        < 60.0
        or payloads[0]["observation_id"]
        == payloads[1]["observation_id"]
    ):
        raise ChainError(
            "bootstrap scheduler observations do not authorize repair"
        )
    provenance_path = _bootstrap_generation_provenance_path(
        manifest_path=manifest_path,
        generation=generation,
    )
    provenance = _validate_bootstrap_generation_provenance(
        provenance_path,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
    )
    if (
        payloads[0].get("generation_provenance") != str(provenance_path)
        or payloads[0].get("generation_provenance_sha256")
        != _sha256(provenance_path)
        or payloads[0].get("generation_provenance_id")
        != provenance["provenance_id"]
    ):
        raise ChainError(
            "bootstrap scheduler observations lost generation provenance"
        )
    for payload in payloads:
        _validate_bootstrap_status_payload(
            payload,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            provenance=provenance,
        )
    return [
        dict(payload)
        | {
            "observation_artifact": str(path),
            "observation_artifact_sha256": _sha256(path),
        }
        for payload, path in zip(payloads, selected, strict=True)
    ]


def bootstrap_status(
    manifest_path: Path,
    *,
    runner: Runner | None = None,
    now: float | None = None,
    record: bool = False,
) -> dict[str, Any]:
    """Observe the exact pre-handoff recovery namespace without mutation."""

    manifest_path = _lexical_absolute(manifest_path)
    verify_chain(manifest_path)
    manifest = _read_json(
        manifest_path, description="recovery-chain manifest"
    )
    runner = _default_runner if runner is None else runner
    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp):
        raise ChainError("bootstrap observation timestamp must be finite")
    receipt, receipt_path, generation, pending = _latest_chain_receipt(
        manifest=manifest, manifest_path=manifest_path
    )
    anchor_receipt_path = manifest_path.parent / SUBMISSION_RECEIPT_NAME
    anchor_receipt = _validate_submission_receipt(
        anchor_receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
        comments=_submission_comments(manifest),
    )
    if pending is not None:
        raise ChainError(
            "bootstrap status refuses an incomplete repair transaction"
        )
    anchor_launch = _bootstrap_anchor_launch_authorization(
        manifest=manifest,
        manifest_path=manifest_path,
    )
    anchor_armed = _bootstrap_anchor_armed_authorization(
        manifest=manifest,
        manifest_path=manifest_path,
    )
    anchor_release_intent = (
        _bootstrap_anchor_release_intent_authorization(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    )
    states = _query_receipt_job_states(
        receipt=receipt, manifest=manifest, runner=runner
    )
    namespace = _validate_bootstrap_scheduler_namespace(
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        runner=runner,
    )
    if generation == 0:
        root_names = ["source_checkout"]
    else:
        root_names = list(receipt.get("held_root_names", []))
    root_name = root_names[0] if root_names else ""
    if not root_name or root_name not in states:
        raise ChainError("bootstrap receipt has no exact generation root")
    root = states[root_name]
    root_record = next(
        row for row in receipt["jobs"] if row["name"] == root_name
    )
    root_details_by_name = {
        name: _show_recovery_job(
            str(states[name]["job_id"]), runner=runner
        )
        for name in root_names
    }
    root_held = bool(root_names) and all(
        details["returncode"] == 0
        and _normalize_slurm_state(
            str(details["fields"].get("JobState", ""))
        )
        == "PENDING"
        and str(details["fields"].get("Reason", "")).lower()
        == "jobhelduser"
        for details in root_details_by_name.values()
    )
    state_values = {str(item["state"]) for item in states.values()}
    recovery_namespace_cancelled = (
        len(states) == len(manifest["jobs"])
        and not any(item["active"] for item in states.values())
        and state_values
        <= _BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES | {"COMPLETED"}
        and bool(
            state_values & _BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES
        )
    )
    handoff_path = (
        manifest_path.parent / BOOTSTRAP_WATCHDOG_HANDOFF_NAME
    )
    handoff_complete = False
    if handoff_path.exists() or handoff_path.is_symlink():
        verify_bootstrap_watchdog_handoff(manifest_path)
        handoff_complete = True
    descendant_armed_path = (
        receipt_path.parent / BOOTSTRAP_DESCENDANT_ARMED_NAME
    )
    descendant_armed: Mapping[str, Any] | None = None
    if generation > 0 and (
        descendant_armed_path.exists() or descendant_armed_path.is_symlink()
    ):
        descendant_armed = _validate_bootstrap_descendant_armed(
            descendant_armed_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            root_record=root_record,
        )
    provenance_path = _bootstrap_generation_provenance_path(
        manifest_path=manifest_path, generation=generation
    )
    if (
        record
        and (provenance_path.exists() or provenance_path.is_symlink())
    ):
        provenance = _ensure_bootstrap_generation_provenance(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            states=states,
            runner=runner,
        )
    elif provenance_path.exists() or provenance_path.is_symlink():
        provenance = _validate_bootstrap_generation_provenance(
            provenance_path,
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
        )
    elif record:
        provenance = _ensure_bootstrap_generation_provenance(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=receipt,
            receipt_path=receipt_path,
            generation=generation,
            states=states,
            runner=runner,
        )
    else:
        provenance = None
    provenance_jobs = (
        {}
        if provenance is None
        else {str(row["name"]): row for row in provenance["jobs"]}
    )
    jobs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="schema5-bootstrap-spool-"
    ) as spool_directory:
        spool_root = Path(spool_directory)
        for index, manifest_row in enumerate(manifest["jobs"]):
            name = str(manifest_row["name"])
            item = states[name]
            receipt_row = next(
                row for row in receipt["jobs"] if row["name"] == name
            )
            sealed = provenance_jobs.get(name)
            if item["active"] or sealed is None:
                current = _bootstrap_live_job_provenance(
                    manifest_row=manifest_row,
                    receipt_row=receipt_row,
                    state=item,
                    runner=runner,
                    spool_path=spool_root / f"{index:02d}-{name}.sbatch",
                    origin_generation=int(
                        receipt_row.get("generation", 0)
                    ),
                    generation_root_names=(
                        ["source_checkout"]
                        if generation == 0
                        else list(receipt["held_root_names"])
                    ),
                )
                if sealed is not None and current != sealed:
                    raise ChainError(
                        f"live bootstrap provenance drifted for {name}"
                    )
                sealed = current
            if not isinstance(sealed, Mapping):
                raise ChainError(
                    f"bootstrap provenance is unavailable for {name}"
                )
            jobs.append(
                {
                    "name": sealed["name"],
                    "job_id": sealed["job_id"],
                    "comment": sealed["comment"],
                    "job_name": sealed["job_name"],
                    "state": item["state"],
                    "active": item["active"],
                    "script_sha256": sealed["script_sha256"],
                    "submit_line_sha256": sealed[
                        "submit_line_sha256"
                    ],
                    "submit_line_exact": sealed["submit_line_exact"],
                    "scontrol_command": sealed["scontrol_command"],
                    "scontrol_requeue": sealed["scontrol_requeue"],
                    "spooled_script_sha256": sealed[
                        "spooled_script_sha256"
                    ],
                    "spooled_script_exact_match": sealed[
                        "spooled_script_exact_match"
                    ],
                }
            )
    observation: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r7-bootstrap-status-v1",
        "passed": True,
        "observed_at_timestamp": timestamp,
        "release_git_commit": manifest["release_git_commit"],
        "release_tag_object": manifest["release_tag_object"],
        "chain_id": manifest["chain_id"],
        "chain_manifest": str(manifest_path),
        "chain_manifest_sha256": _sha256(manifest_path),
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": _sha256(receipt_path),
        "submission_receipt_id": receipt["receipt_id"],
        "generation_provenance": (
            None if provenance is None else str(provenance_path)
        ),
        "generation_provenance_sha256": (
            None if provenance is None else _sha256(provenance_path)
        ),
        "generation_provenance_id": (
            None if provenance is None else provenance["provenance_id"]
        ),
        "anchor_submission_receipt": str(anchor_receipt_path),
        "anchor_submission_receipt_sha256": _sha256(
            anchor_receipt_path
        ),
        "anchor_submission_receipt_id": anchor_receipt["receipt_id"],
        "descendant_chain_validated": True,
        "repair_generation": generation,
        "root_name": root_name,
        "root_job_id": root["job_id"],
        "root_names": root_names,
        "root_job_ids": [states[name]["job_id"] for name in root_names],
        "roots_held": root_held,
        "root_held": root_held,
        "squeue_complete": True,
        "sacct_complete": True,
        "namespace_scan_complete": namespace["complete"],
        "namespace_lineage_receipt_ids": namespace[
            "lineage_receipt_ids"
        ],
        "namespace_lineage_job_ids": namespace["lineage_job_ids"],
        "namespace_bound_job_ids": namespace["bound_job_ids"],
        "ambiguous_jobs": 0,
        "job_count": len(jobs),
        "jobs": jobs,
        "recovery_namespace_cancelled": recovery_namespace_cancelled,
        "anchor_launch_authorized": anchor_launch is not None,
        "anchor_launch_id": (
            None if anchor_launch is None else anchor_launch["launch_id"]
        ),
        "anchor_armed_authorized": anchor_armed is not None,
        "anchor_armed_id": (
            None if anchor_armed is None else anchor_armed["armed_id"]
        ),
        "anchor_release_intent_authorized": (
            anchor_release_intent is not None
        ),
        "anchor_release_intent_sha256": (
            None
            if anchor_release_intent is None
            else anchor_release_intent["intent_sha256"]
        ),
        "descendant_armed": (
            None
            if descendant_armed is None
            else str(descendant_armed_path)
        ),
        "descendant_armed_id": (
            None
            if descendant_armed is None
            else descendant_armed["marker_id"]
        ),
        "descendant_rearm_required": (
            generation > 0
            and root_held
            and anchor_armed is not None
            and descendant_armed is None
        ),
        "descendant_release_required": (
            generation > 0
            and root_held
            and (
                anchor_launch is not None
                or (
                    anchor_release_intent is not None
                    and descendant_armed is not None
                )
            )
        ),
        "handoff_complete": handoff_complete,
        "watchdog_scientific_jobs_submitted": 0,
    }
    observation["observation_id"] = _sha256_bytes(
        _canonical_json(observation)
    )
    if provenance is None:
        raise ChainError(
            "bootstrap status requires sealed generation provenance"
        )
    _validate_bootstrap_status_payload(
        observation,
        manifest=manifest,
        manifest_path=manifest_path,
        receipt=receipt,
        receipt_path=receipt_path,
        generation=generation,
        provenance=provenance,
    )
    if record:
        observation_root = (
            receipt_path.parent / BOOTSTRAP_OBSERVATIONS_ROOT_NAME
        )
        if observation_root.is_symlink():
            raise ChainError("bootstrap observation root is symlinked")
        observation_root.mkdir(parents=True, mode=0o750, exist_ok=True)
        observation_path = observation_root / (
            f"observation-{int(timestamp * 1_000_000):020d}-"
            f"{observation['observation_id']}.json"
        )
        _write_immutable_json_once(
            observation_path,
            observation,
            description="append-only bootstrap scheduler observation",
        )
        observation = observation | {
            "observation_artifact": str(observation_path),
            "observation_artifact_sha256": _sha256(observation_path),
        }
    return observation


def repair_chain(
    manifest_path: Path,
    *,
    apply: bool = False,
    runner: Runner | None = None,
    now: float | None = None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    """Resubmit only the non-completed DAG suffix under a new durable generation."""

    manifest_path = _lexical_absolute(manifest_path)
    verify_chain(manifest_path)
    manifest = _read_json(manifest_path, description="recovery-chain manifest")
    runner = _default_runner if runner is None else runner
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
        base_journal_path = base_path.parent / SUBMISSION_JOURNAL_NAME
        base_journal = _read_json(
            base_journal_path,
            description="latest recovery submission journal",
        )
        launched_root_records = _generation_root_records(
            base, generation=base_generation
        )
        launched_roots = [
            str(record["name"]) for record in launched_root_records
        ]
        launched_root = launched_roots[0]
        if bootstrap:
            anchor_receipt_path = (
                manifest_path.parent / SUBMISSION_RECEIPT_NAME
            )
            anchor_receipt = _validate_submission_receipt(
                anchor_receipt_path,
                manifest=manifest,
                manifest_path=manifest_path,
                comments=_submission_comments(manifest),
            )
            anchor_launch = _bootstrap_anchor_launch_authorization(
                manifest=manifest,
                manifest_path=manifest_path,
            )
            anchor_armed = _bootstrap_anchor_armed_authorization(
                manifest=manifest,
                manifest_path=manifest_path,
            )
            anchor_release_intent = (
                _bootstrap_anchor_release_intent_authorization(
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            )
        else:
            anchor_receipt_path = None
            anchor_receipt = None
            anchor_launch = None
            anchor_armed = None
            anchor_release_intent = None
        if bootstrap:
            handoff_path = (
                manifest_path.parent / BOOTSTRAP_WATCHDOG_HANDOFF_NAME
            )
            if handoff_path.exists() or handoff_path.is_symlink():
                raise ChainError(
                    "bootstrap repair authority ended at watchdog handoff"
                )
        else:
            # Repair must never implicitly turn a held, externally
            # unprotected DAG into a running one.  Initial release is the
            # separate ``release-root`` transaction.
            release_path = base_path.parent / ROOT_RELEASE_COMPLETE_NAME
            launch_path = base_path.parent / LAUNCH_COMPLETE_NAME
            root_record = next(
                record
                for record in base["jobs"]
                if record["name"] == launched_root
            )
            release = _validate_root_release_complete(
                release_path,
                receipt_path=base_path,
                receipt=base,
                root_record=root_record,
            )
            _validate_launch_complete(
                launch_path,
                receipt_path=base_path,
                receipt=base,
                release_path=release_path,
                release=release,
            )
        states = _query_receipt_job_states(
            receipt=base, manifest=manifest, runner=runner
        )
        active = sorted(name for name, item in states.items() if item["active"])
        if active:
            if bootstrap and base_generation > 0:
                provenance_path = _bootstrap_generation_provenance_path(
                    manifest_path=manifest_path,
                    generation=base_generation,
                )
                provenance = _ensure_bootstrap_generation_provenance(
                    manifest=manifest,
                    manifest_path=manifest_path,
                    receipt=base,
                    receipt_path=base_path,
                    generation=base_generation,
                    states=states,
                    runner=runner,
                )
                root_record = next(
                    row
                    for row in base["jobs"]
                    if row["name"] == launched_root
                )
                descendant_armed_path = (
                    base_path.parent / BOOTSTRAP_DESCENDANT_ARMED_NAME
                )
                descendant_armed: Mapping[str, Any] | None = None
                if (
                    descendant_armed_path.exists()
                    or descendant_armed_path.is_symlink()
                ):
                    descendant_armed = _validate_bootstrap_descendant_armed(
                        descendant_armed_path,
                        manifest=manifest,
                        manifest_path=manifest_path,
                        receipt=base,
                        receipt_path=base_path,
                        generation=base_generation,
                        root_record=root_record,
                    )
                if anchor_armed is not None and descendant_armed is None:
                    observations = _validate_bootstrap_observation_pair(
                        manifest=manifest,
                        manifest_path=manifest_path,
                        receipt=base,
                        receipt_path=base_path,
                        generation=base_generation,
                        require_cancelled=False,
                        require_release=False,
                        require_rearm=True,
                    )
                    descendant_armed, descendant_armed_path = (
                        _ensure_bootstrap_descendant_armed(
                            manifest=manifest,
                            manifest_path=manifest_path,
                            receipt=base,
                            receipt_path=base_path,
                            generation=base_generation,
                            root_record=root_record,
                            observations=observations,
                        )
                    )
                release_authorized = (
                    anchor_launch is not None
                    or (
                        anchor_release_intent is not None
                        and descendant_armed is not None
                    )
                )
                common = {
                    "passed": True,
                    "repair_generation": base_generation,
                    "submission_receipt": str(base_path),
                    "submission_receipt_sha256": _sha256(base_path),
                    "submission_receipt_id": base["receipt_id"],
                    "generation_provenance": str(provenance_path),
                    "generation_provenance_sha256": _sha256(
                        provenance_path
                    ),
                    "generation_provenance_id": provenance[
                        "provenance_id"
                    ],
                    "parent_submission_receipt_id": (
                        _read_json(
                            Path(str(base["parent_receipt"])),
                            description="bootstrap repair parent receipt",
                        )["receipt_id"]
                    ),
                    "anchor_submission_receipt_id": anchor_receipt[
                        "receipt_id"
                    ],
                    "anchor_submission_receipt_sha256": _sha256(
                        anchor_receipt_path
                    ),
                    "anchor_launch_authorized": anchor_launch is not None,
                    "anchor_launch_id": (
                        None
                        if anchor_launch is None
                        else anchor_launch["launch_id"]
                    ),
                    "anchor_armed_authorized": anchor_armed is not None,
                    "anchor_armed_id": (
                        None
                        if anchor_armed is None
                        else anchor_armed["armed_id"]
                    ),
                    "anchor_release_intent_authorized": (
                        anchor_release_intent is not None
                    ),
                    "descendant_armed": (
                        None
                        if descendant_armed is None
                        else str(descendant_armed_path)
                    ),
                    "descendant_armed_id": (
                        None
                        if descendant_armed is None
                        else descendant_armed["marker_id"]
                    ),
                    "root_name": launched_root,
                    "root_job_id": root_record["job_id"],
                    "root_names": launched_roots,
                    "root_job_ids": [
                        str(record["job_id"])
                        for record in launched_root_records
                    ],
                    "watchdog_scientific_jobs_submitted": 0,
                }
                if not release_authorized:
                    return base | common | {
                        "status": (
                            "bootstrap_descendant_rearmed_held"
                            if descendant_armed is not None
                            else "bootstrap_repair_reconciled"
                        ),
                        "roots_held": True,
                        "root_held": True,
                        "root_released": False,
                        "root_release_id": None,
                        "launch_id": None,
                    }
                if anchor_launch is not None:
                    _validate_bootstrap_observation_pair(
                        manifest=manifest,
                        manifest_path=manifest_path,
                        receipt=base,
                        receipt_path=base_path,
                        generation=base_generation,
                        require_cancelled=False,
                        require_release=True,
                    )
                release, launch = _ensure_recovery_root_released(
                    evidence_root=base_path.parent,
                    manifest=manifest,
                    receipt_path=base_path,
                    receipt=base,
                    journal=base_journal,
                    root_name=launched_root,
                    runner=runner,
                    timestamp=boundary_timestamp(),
                    bootstrap_descendant_armed_path=(
                        descendant_armed_path
                    ),
                )
                return base | common | {
                    "status": (
                        "bootstrap_descendant_rearmed_launched"
                        if anchor_launch is None
                        else "bootstrap_repair_reconciled"
                    ),
                    "roots_held": False,
                    "root_held": False,
                    "root_released": True,
                    "root_release_id": release["release_id"],
                    "launch_id": launch["launch_id"],
                    "watchdog_scientific_jobs_submitted": 0,
                }
            report = {"status": "in_progress", "active_jobs": active, "repair_jobs": []}
            if apply:
                raise ChainError(f"cannot repair while receipt jobs are active: {active}")
            return report
        if bootstrap:
            invalid_states = sorted(
                name
                for name, item in states.items()
                if item["state"]
                not in _BOOTSTRAP_INFRASTRUCTURE_TERMINAL_STATES
                | {"COMPLETED"}
            )
            repair_names = [
                str(row["name"])
                for row in manifest["jobs"]
                if states[str(row["name"])]["state"] != "COMPLETED"
            ]
            if invalid_states or not repair_names:
                raise ChainError(
                    "bootstrap repair requires only completed or "
                    "infrastructure-terminal recovery jobs"
                )
            _validate_bootstrap_observation_pair(
                manifest=manifest,
                manifest_path=manifest_path,
                receipt=base,
                receipt_path=base_path,
                generation=base_generation,
                require_cancelled=True,
                require_release=False,
            )
        else:
            repair_names = _sentinel_repair_jobs(
                manifest=manifest,
                manifest_path=manifest_path,
                receipt_path=base_path,
                generation=base_generation,
                states=states,
            )
        if not repair_names:
            if pending is not None:
                raise ChainError("pending repair exists although its parent DAG is complete")
            return {"status": "complete", "repair_jobs": [], "receipt": str(base_path)}
        failed_set = {
            name
            for name, item in states.items()
            if item["state"] in _REPAIRABLE_SLURM_STATES
        }
        by_name = {row["name"]: row for row in manifest["jobs"]}
        for name, item in states.items():
            if name != "failure_sentinel" and item["state"] == "COMPLETED" and (
                _manifest_ancestors(name, by_name) & failed_set
            ) and not _is_exact_completed_stage_observer(
                name=name,
                manifest=manifest,
                receipt=base,
                states=states,
                failed_set=failed_set,
            ):
                raise ChainError(f"completed job {name} has a failed ancestor")

        if pending is None:
            generation = base_generation + 1
            repair_root = manifest_path.parent / REPAIR_ROOT_NAME
            generation_root = repair_root / f"g{generation:04d}"
            if not apply:
                return {
                    "status": "dry_run", "repair_generation": generation,
                    "repair_jobs": repair_names,
                    "states": {name: states[name]["state"] for name in repair_names},
                    "would_write": str(generation_root),
                }
            policy = _dependency_config_allows_fail_closed(
                runner, checked_at=boundary_timestamp()
            )
            policy["phase"] = "pre_submission"
            policy["evidence_id"] = _sha256_bytes(
                _canonical_json(policy)
            )
            _preflight_repair_stage_artifacts(
                manifest, repair_names, manifest_path
            )
            repair_root.mkdir(parents=True, exist_ok=True)
            _require_canonical_path(
                repair_root, description="repair root", kind="directory"
            )
            journal = {
                "schema_version": SUBMISSION_SCHEMA_VERSION,
                "protocol": "schema5-v1.2-r7-recovery-chain-repair-journal",
                "chain_id": manifest["chain_id"], "repair_generation": generation,
                "base_receipt": str(base_path), "base_receipt_sha256": _sha256(base_path),
                "repair_jobs": repair_names, "started_at": _utc_now(),
                "started_timestamp": timestamp, "scheduler_since": _slurm_timestamp(timestamp),
                "slurm_user": manifest["slurm_user"],
                "dependency_policy_check": str(
                    generation_root
                    / DEPENDENCY_POLICY_CHECKS_ROOT_NAME
                    / "check-0001.json"
                ),
                "dependency_policy_check_sha256": _sha256_bytes(
                    _canonical_json(policy)
                ),
                "dependency_canary": _dependency_canary_binding(manifest),
                "held_root_names": _repair_frontier_names(
                    manifest, repair_names
                ),
                "jobs": {},
            }
            temporary_generation = manifest_path.parent / (
                f".{REPAIR_ROOT_NAME}.g{generation:04d}.{os.getpid()}.{uuid.uuid4().hex}"
            )
            temporary_generation.mkdir(mode=0o750)
            try:
                _atomic_json(
                    temporary_generation
                    / DEPENDENCY_POLICY_CHECKS_ROOT_NAME
                    / "check-0001.json",
                    policy,
                    mode=0o444,
                )
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
            expected_repair_names = (
                [
                    str(row["name"])
                    for row in manifest["jobs"]
                    if states[str(row["name"])]["state"] != "COMPLETED"
                ]
                if bootstrap
                else _sentinel_repair_jobs(
                    manifest=manifest,
                    manifest_path=manifest_path,
                    receipt_path=base_path,
                    generation=base_generation,
                    states=states,
                )
            )
            if repair_names != expected_repair_names:
                raise ChainError(
                    "pending repair set no longer equals its parent's failed job set"
                )
            if not apply:
                return {"status": "pending_repair", "repair_generation": generation, "repair_jobs": repair_names}
            _dependency_config_allows_fail_closed(
                runner, checked_at=boundary_timestamp()
            )
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
            argv = submission_argv(
                row,
                dependency_job_ids=dependency_ids,
                comment=comment,
                initial_hold=name
                in set(journal["held_root_names"]),
            )
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
                "script_sha256": row["script_sha256"],
                "dependency_type": row["dependency_type"],
                "generation": job_generation,
                "disposition": disposition,
            })
        receipt = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "protocol": "schema5-v1.2-r7-recovery-chain-repair", "passed": True,
            "chain_id": manifest["chain_id"], "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "submission_journal": str(journal_path),
            "submission_journal_sha256": _sha256(journal_path),
            "repair_generation": generation,
            "parent_receipt": str(base_path), "parent_receipt_sha256": _sha256(base_path),
            "submitted_at": _utc_now(),
            "dependency_policy": DEPENDENCY_POLICY_CONTRACT,
            "dependency_policy_check": journal[
                "dependency_policy_check"
            ],
            "dependency_policy_check_sha256": journal[
                "dependency_policy_check_sha256"
            ],
            "dependency_canary": journal["dependency_canary"],
            "root_initial_hold": True,
            "held_root_names": list(journal["held_root_names"]),
            "no_requeue": True,
            "stage_failure_sentinels": manifest[
                "stage_failure_sentinels"
            ],
            "jobs": receipt_jobs,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        receipt_path = generation_root / SUBMISSION_RECEIPT_NAME
        _atomic_json(receipt_path, receipt, mode=0o444)
        validated = _validate_repair_receipt(
            receipt_path, manifest=manifest, manifest_path=manifest_path,
            generation=generation, parent_path=base_path,
        )
        repaired_states = _query_receipt_job_states(
            receipt=validated,
            manifest=manifest,
            runner=runner,
        )
        provenance_path = _bootstrap_generation_provenance_path(
            manifest_path=manifest_path,
            generation=generation,
        )
        provenance = _ensure_bootstrap_generation_provenance(
            manifest=manifest,
            manifest_path=manifest_path,
            receipt=validated,
            receipt_path=receipt_path,
            generation=generation,
            states=repaired_states,
            runner=runner,
        )
        if bootstrap:
            # A reconstructed generation is always left held.  Even a durable
            # g0 launch or release intent is reconciled only after two fresh,
            # exact observations of this generation's acceptance-bound root.
            status = "bootstrap_repaired_held"
            root_held = True
            root_released = False
            release_fields = {
                "root_release_id": None,
                "launch_id": None,
            }
            return validated | {
                "status": status,
                "passed": True,
                "repair_jobs": repair_names,
                "submission_receipt": str(receipt_path),
                "submission_receipt_sha256": _sha256(receipt_path),
                "submission_receipt_id": validated["receipt_id"],
                "generation_provenance": str(provenance_path),
                "generation_provenance_sha256": _sha256(
                    provenance_path
                ),
                "generation_provenance_id": provenance["provenance_id"],
                "parent_submission_receipt_id": base["receipt_id"],
                "anchor_submission_receipt_id": anchor_receipt[
                    "receipt_id"
                ],
                "anchor_submission_receipt_sha256": _sha256(
                    anchor_receipt_path
                ),
                "anchor_launch_authorized": anchor_launch is not None,
                "anchor_launch_id": (
                    None
                    if anchor_launch is None
                    else anchor_launch["launch_id"]
                ),
                "anchor_armed_authorized": anchor_armed is not None,
                "anchor_armed_id": (
                    None
                    if anchor_armed is None
                    else anchor_armed["armed_id"]
                ),
                "anchor_release_intent_authorized": (
                    anchor_release_intent is not None
                ),
                "descendant_armed": None,
                "descendant_armed_id": None,
                "root_name": validated["held_root_names"][0],
                "root_job_id": next(
                    str(row["job_id"])
                    for row in validated["jobs"]
                    if row["name"] == validated["held_root_names"][0]
                ),
                "root_names": list(validated["held_root_names"]),
                "root_job_ids": [
                    str(
                        next(
                            row["job_id"]
                            for row in validated["jobs"]
                            if row["name"] == name
                        )
                    )
                    for name in validated["held_root_names"]
                ],
                "root_held": root_held,
                "roots_held": root_held,
                "root_released": root_released,
                **release_fields,
                "watchdog_scientific_jobs_submitted": 0,
                "release_required": not root_released,
            }
        release, launch = _ensure_recovery_root_released(
            evidence_root=generation_root,
            manifest=manifest,
            receipt_path=receipt_path,
            receipt=validated,
            journal=journal,
            root_name=validated["held_root_names"][0],
            runner=runner,
            timestamp=boundary_timestamp(),
        )
        return validated | {
            "status": "repair_launched",
            "repair_jobs": repair_names,
            "root_release": release,
            "launch_complete": launch,
        }


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
    runner = _default_runner if runner is None else runner
    recovery_root = _require_canonical_path(
        manifest_path.parent,
        description="recovery root",
        kind="directory",
    )
    release_root = _lexical_absolute(Path(manifest["release_root"]))
    expected_release_root = recovery_root / "releases" / RELEASE_ID
    if release_root != expected_release_root:
        raise ChainError(
            f"release root is not the pinned v1.2 destination: {release_root}"
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
            "protocol": "schema5-v1.2-r7-partial-materialization-quarantine-intent",
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
                "protocol": "schema5-v1.2-r7-partial-materialization-quarantine",
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
            try:
                from scripts.seal_recovery_evidence import seal_quarantine
            except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
                from seal_recovery_evidence import seal_quarantine

            seal = seal_quarantine(
                tree=destination,
                evidence_root=evidence_root,
                release_id=RELEASE_ID,
                failed_job_id=materialize_job_id,
                quarantine_completion=completion_path,
                apply=apply,
            )
            if apply and seal.get("passed") is not True:
                raise ChainError("quarantined materialization did not seal")
            return completion | {
                "status": (
                    "already_quarantined_and_sealed"
                    if apply
                    else "already_quarantined_would_seal"
                ),
                "quarantine_seal": seal,
            }

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
            "protocol": "schema5-v1.2-r7-partial-materialization-quarantine",
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
        try:
            from scripts.seal_recovery_evidence import seal_quarantine
        except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
            from seal_recovery_evidence import seal_quarantine

        seal = seal_quarantine(
            tree=destination,
            evidence_root=evidence_root,
            release_id=RELEASE_ID,
            failed_job_id=materialize_job_id,
            quarantine_completion=completion_path,
            apply=True,
        )
        if seal.get("passed") is not True:
            raise ChainError("quarantined materialization did not seal")
        return completion | {
            "status": "quarantined_and_sealed",
            "quarantine_seal": seal,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="dry-run or publish the v1.2-r7 chain")
    render.add_argument("--repository", required=True, type=Path)
    render.add_argument("--results-root", required=True, type=Path)
    render.add_argument("--recovery-root", required=True, type=Path)
    render.add_argument("--hf-home", required=True, type=Path)
    render.add_argument("--dev-python", required=True, type=Path)
    render.add_argument("--source-harness-prefix", required=True, type=Path)
    render.add_argument("--source-serving-prefix", required=True, type=Path)
    render.add_argument("--conda-toolchain-root", required=True, type=Path)
    render.add_argument("--source-package-cache", required=True, type=Path)
    render.add_argument(
        "--materialization-pilot-root", required=True, type=Path
    )
    render.add_argument("--slurm-canary-root", required=True, type=Path)
    render.add_argument("--partition", choices=("mit_normal",), default="mit_normal")
    render.add_argument("--slurm-user", default=getpass.getuser())
    render.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser("verify", help="verify an immutable rendered chain")
    verify.add_argument("--chain-manifest", required=True, type=Path)
    qualification = subparsers.add_parser(
        "verify-throughput-qualification",
        help="verify the marker-last full-capacity throughput qualification",
    )
    qualification.add_argument("--chain-manifest", required=True, type=Path)
    smoke = subparsers.add_parser(
        "verify-smoke-attempt",
        help="verify CURRENT selects the latest sealed 41-cell smoke attempt",
    )
    smoke.add_argument("--chain-manifest", required=True, type=Path)
    watchdog = subparsers.add_parser(
        "verify-watchdog-readiness",
        help="verify post-initialize watchdog deployment, liveness, and drill gates",
    )
    watchdog.add_argument("--chain-manifest", required=True, type=Path)
    prerequisites = subparsers.add_parser(
        "verify-prerequisites",
        help="reverify scheduler-bound pilot/canary evidence",
    )
    prerequisites.add_argument("--chain-manifest", required=True, type=Path)
    prerequisites.add_argument("--expected-chain-id", required=True)
    prerequisites.add_argument("--markers-only", action="store_true")
    prerequisites.add_argument("--verifier-checkout", type=Path)
    prerequisites.add_argument("--verifier-python", type=Path)
    prerequisites.add_argument("--verifier-library", type=Path)
    submit = subparsers.add_parser("submit", help="dry-run or transactionally submit")
    submit.add_argument("--chain-manifest", required=True, type=Path)
    submit.add_argument("--apply", action="store_true")
    isolated_drill = subparsers.add_parser(
        "prepare-isolated-bootstrap-drill",
        help=(
            "derive the exact inert sibling DAG used for bootstrap "
            "cancellation recovery"
        ),
    )
    isolated_drill.add_argument(
        "--chain-manifest",
        required=True,
        type=Path,
        help="canonical submitted r7 recovery-chain manifest",
    )
    isolated_drill.add_argument("--apply", action="store_true")
    release_root = subparsers.add_parser(
        "release-root",
        help=(
            "release the exact held root after bootstrap-watchdog "
            "readiness is sealed"
        ),
    )
    release_root.add_argument("--chain-manifest", required=True, type=Path)
    release_root.add_argument("--apply", action="store_true")
    bootstrap_status_parser = subparsers.add_parser(
        "bootstrap-status",
        help="record one exact pre-handoff scheduler observation",
    )
    bootstrap_status_parser.add_argument(
        "--chain-manifest", required=True, type=Path
    )
    bootstrap_status_parser.add_argument(
        "--no-record", action="store_true"
    )
    bootstrap_repair_parser = subparsers.add_parser(
        "bootstrap-repair",
        help=(
            "reconstruct an infrastructure-cancelled recovery suffix "
            "under one held generation"
        ),
    )
    bootstrap_repair_parser.add_argument(
        "--chain-manifest", required=True, type=Path
    )
    bootstrap_repair_parser.add_argument("--apply", action="store_true")
    bootstrap_repair_parser.add_argument(
        "--result-output",
        type=Path,
        help=(
            "publish the exact immutable bootstrap repair result at "
            "BOOTSTRAP_REPAIR_RESULT.json"
        ),
    )
    handoff = subparsers.add_parser(
        "publish-bootstrap-handoff",
        help=(
            "publish marker-last handoff to the ready control-bound watchdog"
        ),
    )
    handoff.add_argument("--chain-manifest", required=True, type=Path)
    handoff.add_argument("--apply", action="store_true")
    verify_handoff = subparsers.add_parser(
        "verify-bootstrap-handoff",
        help="require the immutable aggregate-sentinel watchdog handoff",
    )
    verify_handoff.add_argument(
        "--chain-manifest", required=True, type=Path
    )
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
                conda_toolchain_root=args.conda_toolchain_root,
                source_package_cache=args.source_package_cache,
                materialization_pilot_root=args.materialization_pilot_root,
                slurm_canary_root=args.slurm_canary_root,
            )
            result = render_chain(
                paths,
                partition=args.partition,
                slurm_user=args.slurm_user,
                apply=args.apply,
            )
        elif args.command == "verify":
            result = verify_chain(args.chain_manifest)
        elif args.command == "verify-throughput-qualification":
            result = verify_throughput_qualification(args.chain_manifest)
        elif args.command == "verify-smoke-attempt":
            result = verify_smoke_attempt(args.chain_manifest)
        elif args.command == "verify-watchdog-readiness":
            result = verify_post_initialize_watchdog(args.chain_manifest)
        elif args.command == "verify-prerequisites":
            if args.markers_only and args.verifier_checkout is not None:
                raise ChainError(
                    "--markers-only and --verifier-checkout are mutually exclusive"
                )
            if not args.markers_only and args.verifier_checkout is None:
                raise ChainError(
                    "full prerequisite verification requires --verifier-checkout"
                )
            if (args.verifier_python is None) != (args.verifier_library is None):
                raise ChainError(
                    "--verifier-python and --verifier-library must be supplied together"
                )
            result = verify_bound_prerequisites(
                args.chain_manifest,
                expected_chain_id=args.expected_chain_id,
                markers_only=args.markers_only,
                verifier_checkout=args.verifier_checkout,
                verifier_python=args.verifier_python,
                verifier_library=args.verifier_library,
            )
        elif args.command == "submit":
            result = submit_chain(args.chain_manifest, apply=args.apply)
        elif args.command == "prepare-isolated-bootstrap-drill":
            result = prepare_isolated_bootstrap_drill_chain(
                args.chain_manifest,
                apply=args.apply,
            )
        elif args.command == "release-root":
            result = release_recovery_root(
                args.chain_manifest, apply=args.apply
            )
        elif args.command == "bootstrap-status":
            result = bootstrap_status(
                args.chain_manifest, record=not args.no_record
            )
        elif args.command == "bootstrap-repair":
            result = repair_chain(
                args.chain_manifest,
                apply=args.apply,
                bootstrap=True,
            )
            if args.result_output is not None:
                if not args.apply:
                    raise ChainError(
                        "--result-output requires bootstrap-repair --apply"
                    )
                result_output = _lexical_absolute(args.result_output)
                expected_output = (
                    _lexical_absolute(args.chain_manifest).parent
                    / BOOTSTRAP_REPAIR_RESULT_NAME
                )
                if result_output != expected_output:
                    raise ChainError(
                        "bootstrap repair result must use its fixed "
                        f"namespace path: {expected_output}"
                    )
                _write_immutable_json_once(
                    result_output,
                    result,
                    description="bootstrap repair result",
                )
                result = result | {
                    "repair_result": str(result_output),
                    "repair_result_sha256": _sha256(result_output),
                }
        elif args.command == "publish-bootstrap-handoff":
            result = publish_bootstrap_watchdog_handoff(
                args.chain_manifest, apply=args.apply
            )
        elif args.command == "verify-bootstrap-handoff":
            result = verify_bootstrap_watchdog_handoff(
                args.chain_manifest
            )
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
