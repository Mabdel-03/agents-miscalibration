"""Safety and identity tests for the offline QID-checkpoint migration."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from agents_scaling.agents.base_agent import AgentOutput
from agents_scaling.agents.message_builder import PeerContextRender
from agents_scaling.agents.topologies.base import TopologyResult
from agents_scaling.benchmarks.contracts import freeze_benchmark_contracts
from agents_scaling.benchmarks.schema import AnswerType, Question
from agents_scaling.config import (
    ContextShareLevel,
    ExperimentCell,
    ReasoningLevel,
    Topology,
)
from agents_scaling.experiment import io
from agents_scaling.experiment.completion import cell_lock
from agents_scaling.experiment.manifest import freeze_manifest, load_manifest
from agents_scaling.experiment.qid_checkpoint import (
    CHECKPOINT_DIRECTORY,
    CheckpointCorruptionError,
    CheckpointingAgent,
    QIDCheckpoint,
)
from agents_scaling.serving.profiles import serving_profile_for_cell
from scripts import migrate_qid_checkpoints as migration

TARGET_CODE_VERSION = "target-commit+source.0123456789abcdef"


def _cell() -> ExperimentCell:
    return ExperimentCell(
        model_size="0.6B",
        context_share_level=ContextShareLevel.ARTIFACT_ONLY,
        prompt_complexity_level=0,
        reasoning_level=ReasoningLevel.OFF,
        topology=Topology.SINGLE_AGENT,
        benchmark="gpqa",
        n_agents=1,
        rounds=1,
        n_samples=1,
        temperature=0.0,
        n_questions=1,
        seed=17,
    )


def _question() -> Question:
    return Question(
        qid="gpqa-0",
        benchmark="gpqa",
        prompt_stem="Which option is correct?",
        options=["correct", "incorrect"],
        answer_key="A",
        answer_type=AnswerType.MCQ,
    )


def _loader(name: str, n: int | None = None, seed: int = 0) -> list[Question]:
    del seed
    assert name == "gpqa"
    assert n in (None, 1)
    return [_question()]


def _output() -> AgentOutput:
    return AgentOutput(
        agent_id="agent0",
        round=0,
        answer_choice="A",
        raw_text="reasoning; answer A",
        cot_text="reasoning",
        intermediate_results="answer A",
        option_logprobs={"A": 0.9, "B": 0.1},
        verbalized_conf=0.8,
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
        generation_phase_finish_reasons=["stop"],
        generation_phase_seeds=[17],
        generation_phase_prompt_tokens=[10],
        generation_phase_completion_tokens=[5],
        generation_phase_requested_max_tokens=[4096],
        generation_phase_prompt_token_id_hashes=["a" * 64],
        generation_phase_completion_token_id_hashes=["b" * 64],
        endpoint_generation="fixture:8000:1",
    )


def _frozen_run_for_cell(
    tmp_path: Path,
    cell: ExperimentCell,
) -> tuple[Path, ExperimentCell, str]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "cells.json").write_text(
        json.dumps([cell.to_dict()], indent=2) + "\n",
        encoding="utf-8",
    )
    freeze_manifest(run_root)
    snapshot = load_manifest(run_root)
    frozen = freeze_benchmark_contracts(
        run_root,
        snapshot=snapshot,
        benchmark_loader=_loader,
    )
    contract_hash = frozen.contract_for_cell(cell)["question_contract_sha256"]
    return run_root, cell, contract_hash


def _frozen_run(tmp_path: Path) -> tuple[Path, ExperimentCell, str]:
    return _frozen_run_for_cell(tmp_path, _cell())


def _write_schema1_checkpoint(
    run_root: Path,
    cell: ExperimentCell,
    contract_hash: str,
) -> tuple[Path, dict, str]:
    cell_directory = run_root / "cells" / cell.cell_id
    checkpoint = QIDCheckpoint(
        cell_directory,
        cell,
        _question(),
        code_version=TARGET_CODE_VERSION,
        serving_profile=serving_profile_for_cell(cell),
        benchmark_contract_sha256=contract_hash,
    )
    request = {
        "generation_role": "topology",
        "qid": _question().qid,
        "agent_id": "agent0",
        "round": 0,
        "seed": cell.seed,
        "sample_index": None,
        "peer_context": {
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "utf8_bytes": 0,
        },
        "max_tokens": 4096,
        "elicit_cot": True,
    }
    output = _output()
    checkpoint.execute_coordinate(
        "topology:agent0:0",
        request,
        lambda: {
            "termination_status": "completed",
            "agent_output": asdict(output),
            "censored_generation": None,
        },
    )
    checkpoint.record_topology_result(
        TopologyResult(
            final_answer="A",
            per_agent=[output],
            n_turns=1,
            n_messages=0,
            n_rounds=1,
            n_agents=1,
            system_conf={"final_producer_logprob": 0.9},
        ),
        wall_ms=123.5,
    )
    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload.pop("migration_history")
    payload["identity"]["code_version"] = next(
        iter(migration.ALLOWED_SOURCE_CODE_VERSIONS)
    )
    payload["identity"]["protocols"] = copy.deepcopy(migration.LEGACY_PROTOCOL_IDENTITY)
    payload["integrity_sha256"] = migration._integrity_sha256(payload)
    io.write_json(checkpoint.path, payload)
    return (
        checkpoint.path,
        copy.deepcopy(payload),
        checkpoint.path.read_text(encoding="utf-8"),
    )


def _write_schema1_peer_checkpoint(
    run_root: Path,
    cell: ExperimentCell,
    contract_hash: str,
) -> tuple[Path, str, PeerContextRender]:
    cell_directory = run_root / "cells" / cell.cell_id
    checkpoint = QIDCheckpoint(
        cell_directory,
        cell,
        _question(),
        code_version=TARGET_CODE_VERSION,
        serving_profile=serving_profile_for_cell(cell),
        benchmark_contract_sha256=contract_hash,
    )
    peer_text = "one exact legacy peer block"
    rendered = PeerContextRender(
        text=peer_text,
        token_count=5,
        sha256=hashlib.sha256(peer_text.encode("utf-8")).hexdigest(),
        block_token_counts=(5,),
        truncation_marker_count=0,
    )
    request = {
        "generation_role": "topology",
        "qid": _question().qid,
        "agent_id": "agent0",
        "round": 0,
        "seed": cell.seed + 999,
        "sample_index": None,
        "peer_context": {
            "sha256": rendered.sha256,
            "utf8_bytes": len(peer_text.encode("utf-8")),
        },
        "max_tokens": 4096,
        "elicit_cot": True,
    }
    output = _output()
    output.peer_context_tokens = rendered.token_count
    output.peer_context_sha256 = rendered.sha256
    output.peer_context_block_token_counts = list(rendered.block_token_counts)
    checkpoint.execute_coordinate(
        "topology:agent0:0",
        request,
        lambda: {
            "termination_status": "completed",
            "agent_output": asdict(output),
            "censored_generation": None,
        },
    )
    payload = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    durable_output = payload["coordinates"]["topology:agent0:0"]["outcome"][
        "agent_output"
    ]
    durable_output["peer_context_tokens"] = 0
    durable_output["peer_context_sha256"] = hashlib.sha256(b"").hexdigest()
    durable_output["peer_context_block_token_counts"] = []
    durable_output["peer_context_truncation_marker_count"] = 0
    payload["schema_version"] = 1
    payload.pop("migration_history")
    payload["identity"]["code_version"] = next(
        iter(migration.ALLOWED_SOURCE_CODE_VERSIONS)
    )
    payload["identity"]["protocols"] = copy.deepcopy(
        migration.LEGACY_PROTOCOL_IDENTITY
    )
    payload["integrity_sha256"] = migration._integrity_sha256(payload)
    io.write_json(checkpoint.path, payload)
    return (
        checkpoint.path,
        checkpoint.path.read_text(encoding="utf-8"),
        rendered,
    )


def _patch_code_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migration.io, "git_commit", lambda: TARGET_CODE_VERSION)


def test_dry_run_is_read_only_and_manifest_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    stale = run_root / "cells" / "unmanifested-cell" / CHECKPOINT_DIRECTORY
    stale.mkdir(parents=True)
    (stale / checkpoint_path.name).write_text(source_text, encoding="utf-8")
    _patch_code_version(monkeypatch)

    report = migration.migrate_run(
        run_root,
        benchmark_loader=_loader,
    )

    assert report["applied"] is False
    assert report["schema1_candidates"] == 1
    assert report["stale_unmanifested_dirs"] == 1
    assert report["errors"] == []
    assert report["would_change_cells"] == [cell.cell_id]
    assert len(report["would_change_checkpoints"]) == 1
    planned = report["would_change_checkpoints"][0]
    assert planned["cell_id"] == cell.cell_id
    assert planned["qid"] == "gpqa-0"
    assert planned["before_sha256"] == migration._bytes_sha256(
        source_text.encode("utf-8")
    )
    assert planned["after_sha256"] != planned["before_sha256"]
    assert planned["after_schema_version"] == migration.CHECKPOINT_SCHEMA_VERSION
    assert planned["complete_preimage_embedded_in_incident"] is True
    assert checkpoint_path.read_text(encoding="utf-8") == source_text
    assert not (run_root / migration.INCIDENT_FILENAME).exists()
    assert not (run_root / migration.RUN_LOCK_FILENAME).exists()


def test_dry_run_can_project_an_incident_reset_without_touching_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    _patch_code_version(monkeypatch)

    report = migration.migrate_run(
        run_root,
        benchmark_loader=_loader,
        excluded_cell_ids={cell.cell_id},
    )

    assert report["schema1_candidates"] == 0
    assert report["would_change_checkpoints"] == []
    assert report["explicitly_excluded_cells"] == [cell.cell_id]
    assert checkpoint_path.read_text(encoding="utf-8") == source_text
    assert not (run_root / migration.INCIDENT_FILENAME).exists()
    assert not (run_root / migration.RUN_LOCK_FILENAME).exists()


def test_checkpoint_projection_rejects_unknown_excluded_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _frozen_run(tmp_path)
    _patch_code_version(monkeypatch)

    with pytest.raises(migration.MigrationError, match="absent from the frozen manifest"):
        migration.migrate_run(
            run_root,
            benchmark_loader=_loader,
            excluded_cell_ids={"not-manifested"},
        )


def test_dry_run_timestamp_replays_exact_before_after_hashes_on_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    _patch_code_version(monkeypatch)

    dry = migration.migrate_run(run_root, benchmark_loader=_loader)
    assert dry["errors"] == []
    timestamp = dry["planned_migration_timestamp"]
    planned = dry["would_change_checkpoints"]

    applied = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
        migration_timestamp=timestamp,
    )

    assert applied["errors"] == []
    assert applied["would_change_cells"] == [cell.cell_id]
    assert applied["would_change_checkpoints"] == planned
    assert planned[0]["before_sha256"] == hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()
    assert planned[0]["after_sha256"] == hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()


def test_explicit_migration_timestamp_cannot_predate_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    source = json.loads(source_text)
    _patch_code_version(monkeypatch)

    report = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
        migration_timestamp=float(source["updated_at"]) - 1.0,
    )

    assert report["checkpoints_migrated"] == 0
    assert report["unsafe_or_invalid_cells_skipped"] == 1
    assert any("timestamp predates its source" in error for error in report["errors"])
    assert checkpoint_path.read_text(encoding="utf-8") == source_text
    assert not (run_root / migration.INCIDENT_FILENAME).exists()


def test_apply_preserves_observations_records_evidence_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, source_payload, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    preserved = {
        field: copy.deepcopy(source_payload[field])
        for field in (
            "coordinates",
            "topology_terminal",
            "created_at",
            "updated_at",
        )
    }
    _patch_code_version(monkeypatch)

    first = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert first["checkpoints_migrated"] == 1
    assert first["errors"] == []
    migrated_text = checkpoint_path.read_text(encoding="utf-8")
    migrated = json.loads(migrated_text)
    assert migrated["schema_version"] == 2
    assert {field: migrated[field] for field in preserved} == preserved
    assert len(migrated["migration_history"]) == 1
    history = migrated["migration_history"][0]
    assert history["source_file_sha256"] == migration._bytes_sha256(
        source_text.encode("utf-8")
    )
    assert history["source_identity_sha256"] == migration._canonical_sha256(
        source_payload["identity"]
    )
    assert history["target_identity_sha256"] == migration._canonical_sha256(
        migrated["identity"]
    )

    # The production loader independently validates the migrated checkpoint, including
    # its exact migration-history schema and all preserved coordinate outcomes.
    reopened = QIDCheckpoint(
        checkpoint_path.parent.parent,
        cell,
        _question(),
        code_version=TARGET_CODE_VERSION,
        serving_profile=serving_profile_for_cell(cell),
        benchmark_contract_sha256=contract_hash,
    )
    assert len(reopened.observed_topology_coordinates()) == 1
    terminal = reopened.topology_terminal()
    assert terminal is not None and terminal.wall_ms == pytest.approx(123.5)

    incident_path = run_root / migration.INCIDENT_FILENAME
    incident_text = incident_path.read_text(encoding="utf-8")
    incident = json.loads(incident_text)
    assert len(incident["checkpoints"]) == 1
    evidence = next(iter(incident["checkpoints"].values()))
    assert evidence["source_checkpoint_text"] == source_text
    assert evidence["source_checkpoint_sha256"] == history["source_file_sha256"]
    assert evidence["source_identity"] == source_payload["identity"]
    assert evidence["target_identity"] == migrated["identity"]

    second = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert second["schema1_candidates"] == 0
    assert second["schema2_already_migrated"] == 1
    assert second["checkpoints_migrated"] == 0
    assert second["errors"] == []
    assert checkpoint_path.read_text(encoding="utf-8") == migrated_text
    assert incident_path.read_text(encoding="utf-8") == incident_text


def test_migrated_peer_coordinate_replay_durably_upgrades_without_redraw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = replace(
        _cell(),
        topology=Topology.CENTRALIZED,
        n_agents=2,
        rounds=1,
    )
    run_root, cell, contract_hash = _frozen_run_for_cell(tmp_path, cell)
    checkpoint_path, source_text, rendered = _write_schema1_peer_checkpoint(
        run_root,
        cell,
        contract_hash,
    )
    _patch_code_version(monkeypatch)
    applied = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert applied["checkpoints_migrated"] == 1
    assert applied["errors"] == []

    class NoRedrawAgent:
        agent_id = "agent0"

        def __init__(self) -> None:
            self.answer_calls = 0

        def answer(self, *args, **kwargs):
            self.answer_calls += 1
            raise AssertionError("migrated observation was redrawn")

        def prepare_calibration(self, question):
            return None

    checkpoint = QIDCheckpoint(
        checkpoint_path.parent.parent,
        cell,
        _question(),
        code_version=TARGET_CODE_VERSION,
        serving_profile=serving_profile_for_cell(cell),
        benchmark_contract_sha256=contract_hash,
    )
    with pytest.raises(CheckpointCorruptionError, match="peer block count"):
        checkpoint.observed_topology_coordinates()
    replacement = NoRedrawAgent()
    output = CheckpointingAgent(replacement, checkpoint).answer_with_peer_context_audit(
        _question(),
        round_idx=0,
        peer_context_render=rendered,
        max_tokens=4096,
        seed=cell.seed + 999,
    )
    assert replacement.answer_calls == 0
    assert output.peer_context_tokens == rendered.token_count
    assert output.peer_context_sha256 == rendered.sha256

    upgraded_text = checkpoint_path.read_text(encoding="utf-8")
    upgraded = json.loads(upgraded_text)
    coordinate = upgraded["coordinates"]["topology:agent0:0"]
    durable_output = coordinate["outcome"]["agent_output"]
    assert durable_output["peer_context_block_token_counts"] == [5]
    assert durable_output["peer_context_sha256"] == rendered.sha256
    # Reopening and strict outcome validation prove censor snapshots can consume this
    # coordinate without the schema-1 peer-audit exemption.
    reopened = QIDCheckpoint(
        checkpoint_path.parent.parent,
        cell,
        _question(),
        code_version=TARGET_CODE_VERSION,
        serving_profile=serving_profile_for_cell(cell),
        benchmark_contract_sha256=contract_hash,
    )
    observed = reopened.observed_topology_coordinates()
    reopened._validate_outcome(observed[0]["request"], observed[0]["outcome"])

    incident = json.loads(
        (run_root / migration.INCIDENT_FILENAME).read_text(encoding="utf-8")
    )
    evidence = next(iter(incident["checkpoints"].values()))
    assert evidence["source_checkpoint_text"] == source_text
    assert hashlib.sha256(source_text.encode("utf-8")).hexdigest() == evidence[
        "source_checkpoint_sha256"
    ]
    reaudit = migration.migrate_run(run_root, benchmark_loader=_loader)
    assert reaudit["schema2_already_migrated"] == 1
    assert reaudit["errors"] == []
    assert checkpoint_path.read_text(encoding="utf-8") == upgraded_text


def test_interruption_after_incident_publication_resumes_without_new_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    _patch_code_version(monkeypatch)
    original_atomic_write = migration.io.atomic_write_text
    failed = False

    def fail_checkpoint_once(path, payload):
        nonlocal failed
        target = Path(path)
        if (
            not failed
            and run_root in target.parents
            and target.parent.name == CHECKPOINT_DIRECTORY
        ):
            failed = True
            raise OSError("simulated preemption before checkpoint replace")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(migration.io, "atomic_write_text", fail_checkpoint_once)
    interrupted = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert interrupted["checkpoints_migrated"] == 0
    assert len(interrupted["errors"]) == 1
    assert checkpoint_path.read_text(encoding="utf-8") == source_text
    incident_path = run_root / migration.INCIDENT_FILENAME
    incident_before = json.loads(incident_path.read_text(encoding="utf-8"))
    record_before = next(iter(incident_before["checkpoints"].values()))[
        "migration_history_record"
    ]

    monkeypatch.setattr(migration.io, "atomic_write_text", original_atomic_write)
    resumed = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert resumed["checkpoints_migrated"] == 1
    assert resumed["errors"] == []
    migrated = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert migrated["migration_history"] == [record_before]


def test_active_cell_is_skipped_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, source_text = _write_schema1_checkpoint(
        run_root, cell, contract_hash
    )
    _patch_code_version(monkeypatch)

    with cell_lock(checkpoint_path.parent.parent, blocking=False):
        report = migration.migrate_run(
            run_root,
            apply=True,
            benchmark_loader=_loader,
        )

    assert report["active_or_locked_cells_skipped"] == 1
    assert report["checkpoints_migrated"] == 0
    assert report["errors"] == []
    assert checkpoint_path.read_text(encoding="utf-8") == source_text


@pytest.mark.parametrize(
    "mutation",
    [
        "integrity",
        "source_code",
        "protocols",
        "question",
        "profile",
        "contract",
        "root_fields",
    ],
)
def test_schema1_identity_or_integrity_drift_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, _ = _write_schema1_checkpoint(run_root, cell, contract_hash)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if mutation == "integrity":
        payload["integrity_sha256"] = "0" * 64
    elif mutation == "source_code":
        payload["identity"]["code_version"] = "unreviewed"
    elif mutation == "protocols":
        payload["identity"]["protocols"]["artifact_schema_version"] = 3
    elif mutation == "question":
        payload["identity"]["question"]["prompt_stem"] = "drifted"
    elif mutation == "profile":
        payload["identity"]["serving_profile"]["max_model_len"] += 1
    elif mutation == "contract":
        payload["identity"]["benchmark_contract_sha256"] = "0" * 64
    elif mutation == "root_fields":
        payload["unexpected"] = True
    if mutation != "integrity":
        payload["integrity_sha256"] = migration._integrity_sha256(payload)
    io.write_json(checkpoint_path, payload)
    drifted_text = checkpoint_path.read_text(encoding="utf-8")
    _patch_code_version(monkeypatch)

    report = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )

    assert report["checkpoints_migrated"] == 0
    assert report["unsafe_or_invalid_cells_skipped"] == 1
    assert len(report["errors"]) == 1
    assert checkpoint_path.read_text(encoding="utf-8") == drifted_text
    assert not (run_root / migration.INCIDENT_FILENAME).exists()


def test_migration_history_rejects_tampering_and_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, _ = _write_schema1_checkpoint(run_root, cell, contract_hash)
    _patch_code_version(monkeypatch)
    report = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert report["errors"] == []
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["migration_history"].append(copy.deepcopy(payload["migration_history"][0]))
    payload["integrity_sha256"] = migration._integrity_sha256(payload)
    io.write_json(checkpoint_path, payload)

    with pytest.raises(CheckpointCorruptionError, match="exactly one"):
        QIDCheckpoint(
            checkpoint_path.parent.parent,
            cell,
            _question(),
            code_version=TARGET_CODE_VERSION,
            serving_profile=serving_profile_for_cell(cell),
            benchmark_contract_sha256=contract_hash,
        )


def test_idempotency_rejects_changed_checkpoint_bytes_with_recomputed_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    checkpoint_path, _, _ = _write_schema1_checkpoint(run_root, cell, contract_hash)
    _patch_code_version(monkeypatch)
    applied = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert applied["errors"] == []
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    coordinate = next(iter(payload["coordinates"].values()))
    coordinate["outcome"]["agent_output"]["raw_text"] = "valid but replaced outcome"
    payload["integrity_sha256"] = migration._integrity_sha256(payload)
    io.write_json(checkpoint_path, payload)

    report = migration.migrate_run(run_root, benchmark_loader=_loader)

    assert report["schema2_already_migrated"] == 0
    assert report["unsafe_or_invalid_cells_skipped"] == 1
    assert any(
        "changed a preserved observation" in error for error in report["errors"]
    )


def test_incident_target_hash_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cell, contract_hash = _frozen_run(tmp_path)
    _write_schema1_checkpoint(run_root, cell, contract_hash)
    _patch_code_version(monkeypatch)
    applied = migration.migrate_run(
        run_root,
        apply=True,
        benchmark_loader=_loader,
    )
    assert applied["errors"] == []
    incident_path = run_root / migration.INCIDENT_FILENAME
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    evidence = next(iter(incident["checkpoints"].values()))
    evidence["target_checkpoint_sha256"] = "0" * 64
    io.write_json(incident_path, incident)

    with pytest.raises(migration.MigrationError, match="target file checksum failed"):
        migration.migrate_run(run_root, benchmark_loader=_loader)


def test_requires_complete_checksum_frozen_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, _, _ = _frozen_run(tmp_path)
    (run_root / "cells.sha256").unlink()
    _patch_code_version(monkeypatch)

    with pytest.raises(migration.MigrationError, match="frozen run"):
        migration.migrate_run(run_root, benchmark_loader=_loader)
