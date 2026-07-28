from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from agents_scaling.serving import (
    external_watchdog,
    protected_capacity,
    scheduler_safety,
)
from scripts import build_schema5_readiness as readiness
from scripts import build_schema5_watchdog_deployment as watchdog_deployment
from scripts import init_schema5_smokes as smoke_init
from scripts import publish_schema5_durable_git_release as durable_git
from scripts import publish_schema5_watchdog_ready as watchdog_ready
from scripts import materialize_schema5_release as materializer
from scripts import provision_schema5_conda_toolchain as conda_toolchain
from scripts import render_schema5_recovery_chain_v12 as renderer
from scripts import run_schema5_materialization_pilot as materialization_pilot
from scripts import run_schema5_slurm_fleet_canary as fleet_canary
from scripts import run_schema5_smokes as smokes
from scripts import run_schema5_throughput_qualification as qualification
from scripts import schema5_bootstrap_watchdog as bootstrap_watchdog
from scripts import schema5_recovery_sentinel as sentinel
from scripts import seal_recovery_evidence as recovery_sealer
from scripts import verify_schema5_recovery_evidence as recovery_verifier
from slurm import schema5_control as control


def test_throughput_producer_and_renderer_protocols_are_identical() -> None:
    assert (
        renderer.THROUGHPUT_QUALIFICATION_SCHEDULER_PROTOCOL
        == qualification.SCHEDULER_PROTOCOL
    )
    assert (
        renderer.THROUGHPUT_QUALIFICATION_SEMANTIC_PROTOCOL
        == qualification.SEMANTIC_PROTOCOL
    )


ACTIVE_SOURCE_PATHS = (
    "slurm/schema5_control.py",
    "src/agents_scaling/serving/protected_capacity.py",
    "src/agents_scaling/serving/scheduler_safety.py",
    "scripts/build_schema5_readiness.py",
    "scripts/init_schema5_smokes.py",
    "scripts/run_schema5_smokes.py",
    "scripts/run_schema5_throughput_qualification.py",
)

HISTORICAL_R2_LITERALS = Counter(
    {
        (
            "scripts/publish_schema5_r1_acceptance_receipts.py",
            "schema5-v1.2-r2-r1-quarantine-idempotency-receipt-v1",
        ): 1,
        (
            "scripts/publish_schema5_r1_acceptance_receipts.py",
            "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2-partial-canary-failure-seal-v1",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "sweep-recovery-schema5-v1.2-r2",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2",
        ): 2,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2-slurm-canary-code-v3",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2-r1-quarantine-idempotency-receipt-v1",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1",
        ): 1,
        (
            "scripts/render_schema5_recovery_chain_v12.py",
            "schema5-v1.2-r2-partial-canary-scheduler-evidence-v1",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "schema5-v1.2-r2-partial-canary-failure-seal-v1",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "schema5-v1.2-r2-partial-canary-failure-seal-intent-v1",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "schema5-v1.2-r2-partial-canary-scheduler-evidence-v1",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "schema5-v1.2-r2-slurm-canary-code-v3",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "sweep-recovery-schema5-v1.2-r2",
        ): 1,
        (
            "scripts/seal_recovery_evidence.py",
            "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1",
        ): 2,
        (
            "scripts/verify_schema5_recovery_evidence.py",
            "schema5-v1.2-r2-recovery-chain",
        ): 1,
        (
            "scripts/verify_schema5_recovery_evidence.py",
            "schema5-v1.2-r2",
        ): 1,
        (
            "scripts/verify_schema5_recovery_evidence.py",
            "sweep-recovery-schema5-v1.2-r2",
        ): 1,
    }
)


def test_fresh_schema5_artifact_protocols_identify_r5() -> None:
    protocols = {
        control.PRODUCTION_AUTHORIZATION_PROTOCOL,
        control.CLIENT_CAPACITY_AUTHORIZATION_PROTOCOL,
        control.CLIENT_CAPACITY_BUILD_PROTOCOL,
        control.CLIENT_CAPACITY_SUBMISSION_PROTOCOL,
        control.SMOKE_ATTEMPT_BINDING_PROTOCOL,
        protected_capacity.LIVE_CLIENT_CAPACITY_PROTOCOL,
        protected_capacity.TRUSTED_SCIENTIFIC_PROVENANCE_PROTOCOL,
        scheduler_safety.SCHEDULER_SAFETY_PROTOCOL,
        scheduler_safety.CLIENT_CAPACITY_PROTOCOL,
        readiness.CAPACITY_TRANSIENT_PROTOCOL,
        readiness.CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL,
        readiness.CAPACITY_PREIMAGE_PROTOCOL,
        qualification.ATTEMPT_POINTER_PROTOCOL,
        qualification.CURRENT_ATTEMPT_PROTOCOL,
        qualification.CAPACITY_TRANSITION_PROTOCOL,
        smoke_init.ATTEMPT_BINDING_PROTOCOL,
        smokes.ATTEMPT_POINTER_PROTOCOL,
        smokes.CURRENT_SELECTOR_PROTOCOL,
        smokes.ATTEMPT_COMPLETE_PROTOCOL,
        smokes.ATTEMPT_FAILURE_PROTOCOL,
        materialization_pilot.PILOT_QUARANTINE_PROTOCOL,
        materialization_pilot.PILOT_QUARANTINE_INTENT_PROTOCOL,
        materialization_pilot.SCHEDULER_INTENT_PROTOCOL,
        materialization_pilot.SCHEDULER_ACTIVE_PROTOCOL,
        materialization_pilot.SCHEDULER_ACCEPTANCE_PROTOCOL,
        materialization_pilot.SUBMISSION_INTENT_PROTOCOL,
        materialization_pilot.SUBMISSION_ATTEMPT_PROTOCOL,
        materialization_pilot.SUBMISSION_RESULT_PROTOCOL,
        materialization_pilot.SUBMISSION_ABSENT_PROTOCOL,
        materialization_pilot.SUBMISSION_ACCEPTED_PROTOCOL,
        fleet_canary.TURNOVER_PORT_DERIVATION_PROTOCOL,
        fleet_canary.CODE_IDENTITY_PROTOCOL,
        durable_git.PROTOCOL,
        external_watchdog.WATCHDOG_PROTOCOL,
        watchdog_deployment.HEARTBEAT_PROTOCOL,
        watchdog_ready.DRILL_PROTOCOL,
        watchdog_ready.READY_PROTOCOL,
        bootstrap_watchdog.STATUS_PROTOCOL,
        renderer.SCHEDULER_ACCEPTANCE_PROTOCOL,
        renderer.SENTINEL_BOOTSTRAP_PROTOCOL,
        renderer.SOURCE_CHECKOUT_SEAL_PROTOCOL,
        renderer.PREREQUISITE_PROTOCOL,
        conda_toolchain.PROTOCOL,
        materializer.PACKAGE_CACHE_SEED_INTENT_PROTOCOL,
        materializer.PACKAGE_CACHE_SEED_PROTOCOL,
        sentinel.SCHEDULER_EVIDENCE_PROTOCOL,
        sentinel.MAIL_PROTOCOL,
        sentinel.MARKER_PROTOCOL,
        sentinel.STAGE_SCHEDULER_EVIDENCE_PROTOCOL,
        sentinel.STAGE_MARKER_PROTOCOL,
    }
    assert protocols
    assert all("schema5-v1.2-r5-" in protocol for protocol in protocols)
    assert all("schema5-v1.2-r2-" not in protocol for protocol in protocols)


def test_active_protocol_producers_and_consumers_are_atomic() -> None:
    assert (
        qualification.RECOVERY_CHAIN_PROTOCOL
        == readiness.R5_PROTOCOL
        == recovery_verifier.R5_PROTOCOL
        == "schema5-v1.2-r5-recovery-chain"
    )
    assert (
        control.SMOKE_ATTEMPT_BINDING_PROTOCOL
        == smoke_init.ATTEMPT_BINDING_PROTOCOL
        == renderer.SMOKE_ATTEMPT_BINDING_PROTOCOL
    )
    assert (
        smokes.ATTEMPT_POINTER_PROTOCOL
        == renderer.SMOKE_ATTEMPT_POINTER_PROTOCOL
    )
    assert (
        smokes.CURRENT_SELECTOR_PROTOCOL
        == renderer.SMOKE_CURRENT_SELECTOR_PROTOCOL
    )
    assert (
        smokes.ATTEMPT_COMPLETE_PROTOCOL
        == renderer.SMOKE_ATTEMPT_COMPLETE_PROTOCOL
    )
    assert (
        smokes.ATTEMPT_FAILURE_PROTOCOL
        == renderer.SMOKE_ATTEMPT_FAILURE_PROTOCOL
    )
    assert (
        qualification.ATTEMPT_POINTER_PROTOCOL
        == renderer.THROUGHPUT_QUALIFICATION_ATTEMPT_POINTER_PROTOCOL
        == sentinel.QUALIFICATION_POINTER_PROTOCOL
    )
    assert (
        qualification.CURRENT_ATTEMPT_PROTOCOL
        == renderer.THROUGHPUT_QUALIFICATION_CURRENT_ATTEMPT_PROTOCOL
        == sentinel.QUALIFICATION_CURRENT_PROTOCOL
    )
    assert (
        readiness.CAPACITY_TRANSIENT_PROTOCOL
        == sentinel.CAPACITY_TRANSIENT_PROTOCOL
    )
    assert (
        readiness.CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL
        == sentinel.CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL
    )
    assert readiness.CAPACITY_PREIMAGE_PROTOCOL == sentinel.CAPACITY_PREIMAGE_PROTOCOL
    assert (
        external_watchdog.WATCHDOG_PROTOCOL
        == watchdog_deployment.HEARTBEAT_PROTOCOL
        == watchdog_ready.READY_PROTOCOL
        == renderer.WATCHDOG_READY_PROTOCOL
    )
    assert (
        watchdog_ready.DRILL_PROTOCOL
        == renderer.EXTERNAL_WATCHDOG_DRILL_PROTOCOL
    )


def test_owned_active_sources_contain_no_r2_protocol_literal() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ACTIVE_SOURCE_PATHS:
        source = (root / relative).read_text(encoding="utf-8")
        assert "schema5-v1.2-r2" not in source, relative


def test_all_remaining_r2_literals_are_explicit_immutable_history() -> None:
    root = Path(__file__).resolve().parents[1]
    observed: Counter[tuple[str, str]] = Counter()
    for directory in ("scripts", "slurm", "src"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = str(path.relative_to(root))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            for node in ast.walk(tree):
                value = node.value if isinstance(node, ast.Constant) else None
                if (
                    isinstance(value, str)
                    and (
                        "schema5-v1.2-r2" in value
                        or "sweep-recovery-schema5-v1.2-r2" in value
                    )
                ):
                    observed[(relative, value)] += 1
    assert observed == HISTORICAL_R2_LITERALS


def test_r3_literals_are_confined_to_immutable_failure_history() -> None:
    root = Path(__file__).resolve().parents[1]
    observed_sources: set[str] = set()
    for directory in ("scripts", "slurm", "src"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = str(path.relative_to(root))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            for node in ast.walk(tree):
                value = node.value if isinstance(node, ast.Constant) else None
                if isinstance(value, str) and (
                    "schema5-v1.2-r3" in value
                    or "sweep-recovery-schema5-v1.2-r3" in value
                ):
                    observed_sources.add(relative)
    assert observed_sources == {
        "scripts/render_schema5_recovery_chain_v12.py",
        "scripts/seal_recovery_evidence.py",
    }
    assert (
        recovery_sealer._R3_PRELAUNCH_FAILURE_SEAL_PROTOCOL
        == "schema5-v1.2-r3-prelaunch-failure-seal-v1"
    )
