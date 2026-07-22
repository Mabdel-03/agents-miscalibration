"""Supplementary ingestion preserves progress without promoting partial estimands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.nb_lib import ingest
from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.completion import CanonicalResults


def _cell() -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": "0.6B",
            "context_share_level": "artifact_only",
            "prompt_complexity_level": 0,
            "reasoning_level": "off",
            "topology": "single_agent",
            "benchmark": "gpqa",
            "n_agents": 1,
            "rounds": 1,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 1,
            "seed": 0,
        }
    )


def _sealed_incident(tmp_path: Path, monkeypatch):
    run_id = "full_sweep_v1"
    run_root = tmp_path / run_id
    cell = _cell()
    snapshot = SimpleNamespace(sha256="a" * 64, cells=(cell,))
    archive = (
        run_root
        / ingest.DISCARDED_RESPONSE_INCIDENT_ROOT
        / cell.cell_id
    )
    artifacts = archive / "artifacts"
    artifacts.mkdir(parents=True)
    result_record = {"qid": "gpqa-0", "answer_key": "A"}
    results_bytes = (json.dumps(result_record) + "\n").encode("utf-8")
    results_path = artifacts / "results.jsonl"
    results_path.write_bytes(results_bytes)
    results_sha = hashlib.sha256(results_bytes).hexdigest()
    incident = {
        "incident_schema_version": 1,
        "incident_type": ingest.DISCARDED_RESPONSE_INCIDENT_KIND,
        "source_protocol_error_type": "ServerResponseProtocolError",
        "target_artifact_schema_version": 5,
        "run_id": run_id,
        "manifest": {
            "path": "cells.json",
            "sha256": snapshot.sha256,
            "cell_count": 1,
        },
        "cell": {
            "cell_id": cell.cell_id,
            "manifest_index": 0,
            "config_hash": cell.config_hash(),
            "config": cell.to_dict(),
        },
        "discovery": {"failure": None, "dispatcher_events": []},
        "active_artifacts": [
            {
                "source_relative_path": "results.jsonl",
                "archived_path": "artifacts/results.jsonl",
                "sha256": results_sha,
                "size": len(results_bytes),
            }
        ],
        "reset_contract": {
            "scope": "entire_active_cell_state",
            "active_files": ["results.jsonl", "meta.json", "failure.json"],
            "checkpoint_directory": ".qid_checkpoints",
            "rerun_from_empty_cell_under_schema": 5,
        },
    }
    incident_bytes = (
        json.dumps(incident, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (archive / "incident.json").write_bytes(incident_bytes)
    marker = {
        "reset_marker_schema_version": 1,
        "incident_sha256": hashlib.sha256(incident_bytes).hexdigest(),
        "target_artifact_schema_version": 5,
        "active_cell_state_removed": True,
    }
    (archive / "reset_complete.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalog = SimpleNamespace(
        frozen=object(),
        snapshot=snapshot,
        questions_for=lambda _cell: (SimpleNamespace(qid="gpqa-0"),),
    )
    monkeypatch.setattr(
        ingest,
        "read_canonical_results",
        lambda *_a, **_kw: CanonicalResults(
            records=(result_record,), raw_rows=1
        ),
    )
    return run_id, run_root, snapshot, catalog, archive, results_path


@pytest.mark.parametrize("state", ["partial", "active", "retryable"])
def test_supplementary_preserves_partial_states_but_excludes_cell_aggregate(state):
    assert ingest._manifested_ingest_decision(
        ingest.SUPPLEMENTARY_MODE, state
    ) == (True, False)
    assert ingest._manifested_ingest_decision(
        ingest.PRIMARY_MODE, state
    ) == (False, False)


def test_only_complete_cells_are_aggregate_eligible_in_both_modes():
    for mode in (ingest.PRIMARY_MODE, ingest.SUPPLEMENTARY_MODE):
        assert ingest._manifested_ingest_decision(mode, "complete") == (True, True)
    with pytest.raises(ingest.SupplementaryIntegrityError, match="zero unresolved corrupt"):
        ingest._manifested_ingest_decision(ingest.SUPPLEMENTARY_MODE, "corrupt")


def test_qid_and_protocol_annotations_are_explicit_on_every_row():
    coverage = ingest._qid_contract_annotation(
        completion_state="partial",
        expected_qids=("q0", "q1", "q2"),
        valid_qids=("q0", "q2"),
        semantically_complete=False,
        manifested=True,
    )
    assert coverage["cell_completion_state"] == "partial"
    assert coverage["cell_expected_qid_count"] == 3
    assert coverage["cell_valid_qid_count"] == 2
    assert coverage["cell_missing_qid_count"] == 1
    assert json.loads(coverage["cell_expected_qids_json"]) == ["q0", "q1", "q2"]
    assert json.loads(coverage["cell_valid_qids_json"]) == ["q0", "q2"]
    assert json.loads(coverage["cell_missing_qids_json"]) == ["q1"]
    provenance = ingest._analysis_provenance_annotation(
        mode=ingest.SUPPLEMENTARY_MODE,
        row={"per_agent": []},
        scientifically_excluded=False,
    )
    assert provenance["mixed_protocol"] is True
    assert provenance["protocol_provenance"] == "legacy-mixed-protocol"
    assert provenance["token_provenance"] == "legacy-nonexact-word-count"
    assert provenance["authoritative_exact_token_claim"] is False


def test_sealed_incident_requires_marker_and_artifact_hashes(tmp_path, monkeypatch):
    run_id, run_root, snapshot, catalog, archive, results_path = _sealed_incident(
        tmp_path, monkeypatch
    )
    verified = ingest.verify_discarded_response_incident(
        run_id=run_id,
        run_root=run_root,
        snapshot=snapshot,
        catalog=catalog,
        archive_path=archive,
    )
    assert verified.cell_id == _cell().cell_id
    assert verified.valid_qids == ("gpqa-0",)
    assert verified.missing_qids == ()
    assert verified.results_sha256 == hashlib.sha256(results_path.read_bytes()).hexdigest()

    results_path.write_bytes(results_path.read_bytes() + b"tampered\n")
    with pytest.raises(ingest.SupplementaryIntegrityError, match="cryptographic"):
        ingest.verify_discarded_response_incident(
            run_id=run_id,
            run_root=run_root,
            snapshot=snapshot,
            catalog=catalog,
            archive_path=archive,
        )


def test_sealed_incident_rejects_incident_marker_drift(tmp_path, monkeypatch):
    run_id, run_root, snapshot, catalog, archive, _ = _sealed_incident(
        tmp_path, monkeypatch
    )
    incident_path = archive / "incident.json"
    incident_path.write_bytes(incident_path.read_bytes() + b" ")
    with pytest.raises(
        ingest.SupplementaryIntegrityError,
        match="reset marker does not authenticate",
    ):
        ingest.verify_discarded_response_incident(
            run_id=run_id,
            run_root=run_root,
            snapshot=snapshot,
            catalog=catalog,
            archive_path=archive,
        )


def test_exact_legacy_preservation_acceptance_is_machine_checkable():
    accepted = ingest.legacy_consolidated_acceptance(
        active_validated_qids=888_068,
        sealed_scientifically_excluded_qids=1_064,
        include_unmanifested=False,
        newly_sealed_incidents=8,
        preexisting_historical_incidents=14,
        preexisting_historical_qids=355,
    )
    assert accepted["observed_total_preserved_qids"] == 889_132
    assert accepted["passed"] is True
    rejected = ingest.legacy_consolidated_acceptance(
        active_validated_qids=888_067,
        sealed_scientifically_excluded_qids=1_064,
        include_unmanifested=False,
    )
    assert rejected["passed"] is False
    wrong_history = ingest.legacy_consolidated_acceptance(
        active_validated_qids=888_068,
        sealed_scientifically_excluded_qids=1_064,
        include_unmanifested=False,
        newly_sealed_incidents=7,
        preexisting_historical_incidents=15,
        preexisting_historical_qids=355,
    )
    assert wrong_history["baseline_counts_passed"] is True
    assert wrong_history["incident_partition_passed"] is False
    assert wrong_history["passed"] is False
    assert ingest.legacy_consolidated_acceptance(
        active_validated_qids=888_068,
        sealed_scientifically_excluded_qids=1_064,
        include_unmanifested=True,
    )["applicable"] is False


def _snapshot_with_incident_markers(
    root: Path,
    memberships: list[tuple[str, str]],
) -> Path:
    records: list[tuple[str, str]] = []
    for run_id, cell_id in memberships:
        relative = (
            f"{run_id}/incidents/{ingest.DISCARDED_RESPONSE_INCIDENT_KIND}/"
            f"{cell_id}/reset_complete.json"
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sealed":true}\n', encoding="utf-8")
        records.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    inventory = "".join(
        f"{digest}  {relative}\n"
        for relative, digest in sorted(records)
    ).encode("utf-8")
    (root / ingest.PRE_REPAIR_SNAPSHOT_INVENTORY).write_bytes(inventory)
    (root / ingest.PRE_REPAIR_SNAPSHOT_COMPLETE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "snapshot_id": "snapshot-test",
                "snapshot_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
                "verified": True,
                "read_only": True,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_pre_repair_membership_is_authenticated_by_snapshot_inventory(tmp_path):
    root = _snapshot_with_incident_markers(
        tmp_path / "pre_repair",
        [
            ("full_sweep_v1", "cell-a"),
            ("full_sweep_agent_counts_v1", "cell-b"),
        ],
    )
    membership = ingest.load_pre_repair_incident_membership(root)
    assert membership.snapshot_id == "snapshot-test"
    assert membership.is_preexisting("full_sweep_v1", "cell-a")
    assert not membership.is_preexisting("full_sweep_v1", "new-cell")

    marker = (
        root
        / "full_sweep_v1"
        / "incidents"
        / ingest.DISCARDED_RESPONSE_INCIDENT_KIND
        / "cell-a"
        / "reset_complete.json"
    )
    marker.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(
        ingest.SupplementaryIntegrityError,
        match="failed inventory verification",
    ):
        ingest.load_pre_repair_incident_membership(root)


def test_cache_generation_withdraws_old_manifest_and_publishes_hashes_last(tmp_path):
    out = tmp_path / "cache"
    out.mkdir()
    (out / ingest.CACHE_MANIFEST_FILENAME).write_text(
        '{"generation":"old"}\n', encoding="utf-8"
    )
    generation = ingest.CacheGeneration(
        out,
        mode=ingest.SUPPLEMENTARY_MODE,
        run_ids=ingest.LEGACY_RUN_IDS,
    )
    generation.begin()
    assert not (out / ingest.CACHE_MANIFEST_FILENAME).exists()
    assert (out / ingest.CACHE_BUILD_MARKER_FILENAME).is_file()
    assert json.loads(
        (out / "ingest_manifest_v1.previous.json").read_text(encoding="utf-8")
    ) == {"generation": "old"}

    for index, filename in enumerate(ingest.CACHE_ARTIFACT_FILENAMES):
        (out / filename).write_bytes(f"artifact-{index}".encode("ascii"))
    published = generation.publish(
        {
            "analysis_mode": ingest.SUPPLEMENTARY_MODE,
            "run_ids": list(ingest.LEGACY_RUN_IDS),
            "cache_contract_sha256": "a" * 64,
        }
    )
    assert not (out / ingest.CACHE_BUILD_MARKER_FILENAME).exists()
    observed = json.loads(
        (out / ingest.CACHE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert observed == published
    for filename, contract in observed["cache_artifacts"].items():
        payload = (out / filename).read_bytes()
        assert contract == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
