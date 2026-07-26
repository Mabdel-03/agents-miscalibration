"""Primary schema-5 and supplementary legacy caches cannot be confused."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from analysis.nb_lib import ingest
from analysis.nb_lib import data as analysis_data


def test_primary_runtime_requires_frozen_release_authority(monkeypatch):
    for field in (
        "ASYS_RELEASE_WORKTREE",
        "ASYS_HARNESS_ENVIRONMENT_PREFIX",
        "ASYS_MODEL_CONTRACT",
    ):
        monkeypatch.delenv(field, raising=False)
    with pytest.raises(RuntimeError, match="run analysis/refresh.sh"):
        ingest._analysis_runtime_authority(ingest.PRIMARY_MODE)
    with pytest.raises(RuntimeError, match="run analysis/refresh.sh"):
        ingest._analysis_runtime_authority(ingest.INTERIM_SCHEMA5_MODE)
    assert ingest._analysis_runtime_authority(ingest.SUPPLEMENTARY_MODE) is None


def test_refresh_uses_control_pinned_release_and_immutable_harness():
    refresh = (
        Path(__file__).resolve().parents[1] / "analysis" / "refresh.sh"
    ).read_text(encoding="utf-8")
    assert "conda_envs/asys_analysis" not in refresh
    assert 'immutable["release_worktree"]' in refresh
    assert 'immutable["harness_environment_prefix"]' in refresh
    assert 'load_control(state_dir, verify_files=True)' in refresh
    assert 'ASYS_RELEASE_WORKTREE="$RELEASE_WORKTREE"' in refresh
    assert '"$PY" -I "$INGEST"' in refresh
    assert 'analysis/nb_lib/ingest.py "${INGEST_ARGS[@]}"' not in refresh


def test_primary_scope_is_exact_and_canonical():
    expected = ingest.PRIMARY_SCHEMA5_RUN_IDS
    assert ingest.resolve_ingest_scope(
        ingest.PRIMARY_MODE, None, include_unmanifested=False
    ) == expected
    assert ingest.resolve_ingest_scope(
        ingest.PRIMARY_MODE,
        list(reversed(expected)),
        include_unmanifested=False,
    ) == expected

    with pytest.raises(ValueError, match="requires exactly"):
        ingest.resolve_ingest_scope(
            ingest.PRIMARY_MODE, list(expected[:-1]), include_unmanifested=False
        )
    with pytest.raises(ValueError, match="cannot include unmanifested"):
        ingest.resolve_ingest_scope(
            ingest.PRIMARY_MODE, None, include_unmanifested=True
        )
    assert ingest.resolve_ingest_scope(
        ingest.INTERIM_SCHEMA5_MODE,
        None,
        include_unmanifested=False,
    ) == expected
    with pytest.raises(ValueError, match="cannot include unmanifested"):
        ingest.resolve_ingest_scope(
            ingest.INTERIM_SCHEMA5_MODE,
            None,
            include_unmanifested=True,
        )


def test_primary_acceptance_requires_exact_full_sweep_and_schema5_qids():
    acceptance = ingest.primary_schema5_acceptance(
        manifest_cells_by_run=ingest.PRIMARY_EXPECTED_CELLS_BY_RUN,
        expected_qids_by_run=ingest.PRIMARY_EXPECTED_QIDS_BY_RUN,
        completion_states={"complete": 22_680, "missing": 0},
        validated_qids_by_completion_state={"complete": 4_524_660},
        artifact_schema_counts={"5": 4_524_660},
    )
    assert acceptance["passed"] is True
    assert acceptance["expected_total_cells"] == 22_680
    assert acceptance["expected_total_qids"] == 4_524_660

    with pytest.raises(
        ingest.PrimarySchema5IncompleteError,
        match="22,680 semantically complete",
    ):
        ingest.primary_schema5_acceptance(
            manifest_cells_by_run=ingest.PRIMARY_EXPECTED_CELLS_BY_RUN,
            expected_qids_by_run=ingest.PRIMARY_EXPECTED_QIDS_BY_RUN,
            completion_states={"complete": 22_679, "partial": 1},
            validated_qids_by_completion_state={
                "complete": 4_524_659,
                "partial": 1,
            },
            artifact_schema_counts={"5": 4_524_660},
        )


def test_primary_catalog_authority_is_required_and_pinned(monkeypatch, tmp_path):
    with pytest.raises(
        ingest.PrimarySchema5IncompleteError,
        match="trusted-generation catalog",
    ):
        ingest._trusted_generation_catalog_authority(
            mode=ingest.PRIMARY_MODE,
            marker_path=None,
            server_pool_root=None,
        )

    marker = tmp_path / "TRUSTED_GENERATION_CATALOG_COMPLETE.json"
    pool = tmp_path / "pool"
    catalog = SimpleNamespace(
        marker_path=marker.resolve(),
        marker_sha256="a" * 64,
        inventory_sha256="b" * 64,
        catalog_sha256="c" * 64,
        catalog_id="catalog-1",
        allowed_generation_tuples=frozenset(
            {("d" * 64, "e" * 64, 1, 1, "endpoint-1")}
        ),
    )
    monkeypatch.setattr(
        ingest,
        "validate_trusted_generation_catalog",
        lambda *_args, **_kwargs: catalog,
    )
    observed, authority = ingest._trusted_generation_catalog_authority(
        mode=ingest.PRIMARY_MODE,
        marker_path=marker,
        server_pool_root=pool,
    )

    assert observed is catalog
    assert authority == {
        "marker_path": str(marker.resolve()),
        "marker_sha256": "a" * 64,
        "inventory_sha256": "b" * 64,
        "catalog_payload_sha256": "c" * 64,
        "catalog_id": "catalog-1",
        "server_pool_root": str(pool.resolve()),
        "allowed_generation_tuple_count": 1,
    }


def test_interim_schema5_rows_are_explicitly_nonauthoritative():
    provenance = ingest._analysis_provenance_annotation(
        mode=ingest.INTERIM_SCHEMA5_MODE,
        row={
            "schema_version": 5,
            "per_agent": [
                {
                    "reasoning_token_source": "vllm_native_token_ids",
                }
            ],
        },
        scientifically_excluded=False,
    )
    assert provenance["mixed_protocol"] is False
    assert provenance["protocol_provenance"] == "schema5-homogeneous"
    assert provenance["authoritative_exact_token_claim"] is False


def test_legacy_scope_is_opt_in_and_only_allows_old_runs():
    assert ingest.resolve_ingest_scope(
        ingest.SUPPLEMENTARY_MODE, None, include_unmanifested=True
    ) == ingest.LEGACY_RUN_IDS
    with pytest.raises(ValueError, match="requires exactly"):
        ingest.resolve_ingest_scope(
            ingest.SUPPLEMENTARY_MODE,
            list(ingest.PRIMARY_SCHEMA5_RUN_IDS),
            include_unmanifested=False,
        )


def test_primary_run_contract_records_all_frozen_hashes(monkeypatch, tmp_path):
    policy = SimpleNamespace(
        accepted_manifest_sha256="a" * 64,
        accepted_benchmark_contracts_sha256="b" * 64,
        file_sha256="c" * 64,
        policy_id="d" * 64,
        required_artifact_schema_version=5,
        authoritative=True,
        accepted_model_contract_sha256="e" * 64,
        release=SimpleNamespace(
            release_id="release",
            git_commit="f" * 40,
            source_tree_sha256="1" * 64,
        ),
        environment=SimpleNamespace(
            harness_sha256="2" * 64,
            serving_sha256="3" * 64,
        ),
    )
    monkeypatch.setattr(ingest, "load_artifact_policy", lambda *_a, **_kw: policy)
    record = ingest._run_contract_record(
        run_id="full_sweep_schema5_v1",
        run_root=tmp_path,
        snapshot=SimpleNamespace(sha256="a" * 64),
        catalog=SimpleNamespace(sidecar_sha256="b" * 64),
        mode=ingest.PRIMARY_MODE,
    )
    assert record == {
        "run_id": "full_sweep_schema5_v1",
        "manifest_sha256": "a" * 64,
        "benchmark_contracts_sha256": "b" * 64,
        "artifact_policy_sha256": "c" * 64,
        "artifact_policy_id": "d" * 64,
        "model_contract_sha256": "e" * 64,
        "release_id": "release",
        "git_commit": "f" * 40,
        "source_tree_sha256": "1" * 64,
        "harness_environment_sha256": "2" * 64,
        "serving_environment_sha256": "3" * 64,
        "required_artifact_schema_version": 5,
        "authoritative": True,
    }


def test_primary_run_contract_fails_closed_on_manifest_drift(monkeypatch, tmp_path):
    policy = SimpleNamespace(
        accepted_manifest_sha256="a" * 64,
        accepted_benchmark_contracts_sha256="b" * 64,
    )
    monkeypatch.setattr(ingest, "load_artifact_policy", lambda *_a, **_kw: policy)
    with pytest.raises(RuntimeError, match="policy/manifest mismatch"):
        ingest._run_contract_record(
            run_id="full_sweep_schema5_v1",
            run_root=tmp_path,
            snapshot=SimpleNamespace(sha256="f" * 64),
            catalog=SimpleNamespace(sidecar_sha256="b" * 64),
            mode=ingest.PRIMARY_MODE,
        )


def test_empty_chunk_writer_atomically_replaces_stale_cache(tmp_path):
    target = tmp_path / "items_v1.parquet"
    pd.DataFrame({"stale": [1]}).to_parquet(target)
    writer = ingest.ChunkWriter(target)
    writer.close()
    refreshed = pd.read_parquet(target)
    assert refreshed.empty
    assert "stale" not in refreshed.columns


def test_cache_identity_binds_mode_and_contracts():
    base = {
        "analysis_mode": ingest.PRIMARY_MODE,
        "mixed_protocol": False,
        "run_ids": list(ingest.PRIMARY_SCHEMA5_RUN_IDS),
        "run_contracts": {"run": {"manifest_sha256": "a" * 64}},
        "manifest_scoped": True,
    }
    assert ingest._sha256_json(base) == ingest._sha256_json(dict(reversed(list(base.items()))))
    changed = dict(base, analysis_mode=ingest.SUPPLEMENTARY_MODE)
    assert ingest._sha256_json(base) != ingest._sha256_json(changed)


def test_analysis_loader_rejects_in_progress_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_data, "CACHE", tmp_path)
    (tmp_path / analysis_data.CACHE_MANIFEST_FILENAME).write_text(
        '{"cache_artifacts":{}}\n', encoding="utf-8"
    )
    (tmp_path / analysis_data.CACHE_BUILD_MARKER_FILENAME).write_text(
        '{"status":"in_progress"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="generation is incomplete"):
        analysis_data.manifest()


def test_legacy_notebook_loader_rejects_primary_schema5_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_data, "CACHE", tmp_path)
    (tmp_path / analysis_data.CACHE_MANIFEST_FILENAME).write_text(
        json.dumps({"analysis_mode": ingest.PRIMARY_MODE, "cache_artifacts": {}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refuse a primary schema-5 cache"):
        analysis_data.manifest()


def test_analysis_cache_hash_verifier_fails_on_drift(tmp_path):
    artifact = tmp_path / "items_v1.parquet"
    artifact.write_bytes(b"canonical")
    manifest = {
        "cache_artifacts": {
            artifact.name: {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "size": artifact.stat().st_size,
            }
        }
    }
    (tmp_path / analysis_data.CACHE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert analysis_data.verify_cache_integrity(tmp_path) == manifest
    artifact.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="contract failed|hash drift"):
        analysis_data.verify_cache_integrity(tmp_path)
