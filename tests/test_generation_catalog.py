"""Adversarial tests for the immutable schema-5 trusted-generation authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agents_scaling.serving import generation_catalog as catalog
from agents_scaling.serving import registry


def _endpoint_history(
    tmp_path: Path,
    *,
    capacity: int = 1,
    rollout: int = 1,
    job_id: str = "700",
    release_fleet: str = "a" * 64,
    fleet: str = "b" * 64,
) -> tuple[Path, registry.EndpointHistoryRecord]:
    pool = tmp_path / "server_pools" / "schema5-v1"
    pool.mkdir(parents=True, exist_ok=True)
    entry = registry.ServerEntry(
        model_size="4B",
        hf_id="model",
        host="node001",
        port=18000 + int(job_id),
        slurm_job_id=job_id,
        started_at=100.0 + int(job_id),
        serving_profile="4B",
        served_model_name="served-model",
        max_model_len=16_384,
        tp_size=1,
        release_id="sweep-recovery-schema5-v1.2",
        environment_hash="c" * 64,
        model_revision="model-revision",
        tokenizer_id="tokenizer",
        tokenizer_revision="tokenizer-revision",
        model_contract_sha256="d" * 64,
        fleet_contract_sha256=fleet,
        server_pool_id="schema5-v1",
        replica_id="schema5-v1--4b--standard--r00",
        replica_index=0,
        release_fleet_contract_sha256=release_fleet,
        capacity_generation=capacity,
        rollout_generation=rollout,
    )
    registry.write_standby_entry(pool, entry)
    intent_token = f"{int(job_id):032x}"[-32:]
    script = (
        pool
        / ".fleet-transactions-v1"
        / "sbatch"
        / f"g{rollout:06d}"
        / f"{entry.replica_id}.{intent_token}.sbatch"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    script.chmod(0o444)
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    provenance = {
        "run_root": str(pool.resolve()),
        "server_pool_id": "schema5-v1",
        "replica_id": entry.replica_id,
        "replica_index": 0,
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": release_fleet,
        "fleet_contract_sha256": fleet,
        "capacity_generation": capacity,
        "rollout_generation": rollout,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": script_sha,
    }
    history = registry.seal_endpoint_history(
        pool,
        entry,
        release_fleet_contract_sha256=release_fleet,
        capacity_generation=capacity,
        rollout_generation=rollout,
        ledger_generation=rollout,
        intent_token=intent_token,
        intent_state="committed",
        committed_at=99.0 + int(job_id),
        local_script_path=script,
        local_script_sha256=script_sha,
        spooled_script=script.read_bytes(),
        spooled_provenance=provenance,
        scheduler_job_name="asys-s5-serve-4b-s-r00",
        scheduler_comment=(
            "asys-s5-fleet:"
            f"pool=schema5-v1;profile=4B;replica={entry.replica_id};"
            f"generation={rollout};intent={intent_token};fleet={fleet}"
        ),
        sealed_at=101.0 + int(job_id),
    )
    return pool, history


def _evidence(tmp_path: Path, name: str = "fleet-readiness") -> catalog.GenerationEvidence:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"kind": name}, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return catalog.GenerationEvidence(name, path, digest)


def _publish(
    state: Path,
    pool: Path,
    records,
    evidence,
    *,
    capacity: int = 1,
    rollout: int = 1,
    release_fleet: str = "a" * 64,
    fleet: str = "b" * 64,
):
    return catalog.publish_trusted_generation_catalog(
        state,
        server_pool_root=pool,
        server_pool_id="schema5-v1",
        endpoint_records=records,
        release_fleet_contract_sha256=release_fleet,
        fleet_contract_sha256=fleet,
        capacity_generation=capacity,
        rollout_generation=rollout,
        generation_evidence=evidence,
        now=200.0 + rollout,
    )


def test_catalog_is_marker_last_readonly_exact_and_idempotent(tmp_path):
    pool, history = _endpoint_history(tmp_path)
    state = tmp_path / "state"
    evidence = [_evidence(tmp_path)]
    first = _publish(state, pool, [history], evidence)
    second = _publish(state, pool, [history], evidence)
    assert second.marker_path == first.marker_path
    assert second.allowed_generation_tuples == {history.allowed_generation_tuple}
    assert first.marker_path.stat().st_mode & 0o222 == 0
    assert (first.marker_path.parent / catalog.CATALOG_INVENTORY).stat().st_mode & 0o222 == 0
    sealed_evidence = (
        first.marker_path.parent
        / str(first.generations[0]["evidence"][0]["path"])
    )
    assert sealed_evidence.stat().st_ino != evidence[0].path.stat().st_ino
    evidence[0].path.write_text('{"source":"later drift"}\n', encoding="utf-8")
    loaded = catalog.load_current_trusted_generation_catalog(
        state, server_pool_root=pool
    )
    assert loaded is not None
    assert loaded.marker_sha256 == first.marker_sha256


def test_catalog_rejects_mutable_current_pointer_tamper(tmp_path):
    pool, history = _endpoint_history(tmp_path)
    state = tmp_path / "state"
    _publish(state, pool, [history], [_evidence(tmp_path)])
    pointer = catalog.catalog_root(state) / catalog.CATALOG_CURRENT
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["marker_sha256"] = "f" * 64
    pointer.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(catalog.GenerationCatalogError, match="pointer drifted"):
        catalog.load_current_trusted_generation_catalog(
            state, server_pool_root=pool
        )


def test_catalog_rejects_unknown_cross_product_and_endpoint_archive_tamper(tmp_path):
    pool, history = _endpoint_history(tmp_path)
    state = tmp_path / "state"
    trusted = _publish(state, pool, [history], [_evidence(tmp_path)])
    release, fleet, capacity, rollout, endpoint = history.allowed_generation_tuple
    assert (release, fleet, capacity, rollout, endpoint) in trusted.allowed_generation_tuples
    assert (
        release,
        fleet,
        capacity + 1,
        rollout,
        endpoint,
    ) not in trusted.allowed_generation_tuples

    binding = history.marker_path.parent / "BINDING.json"
    binding.chmod(0o644)
    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["endpoint_generation"] = "forged"
    binding.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    binding.chmod(0o444)
    with pytest.raises(catalog.GenerationCatalogError, match="endpoint-history"):
        catalog.validate_trusted_generation_catalog(
            trusted.marker_path, server_pool_root=pool
        )


def test_catalog_rejects_revision_tamper_and_discontinuous_capacity(tmp_path):
    pool, history = _endpoint_history(tmp_path)
    state = tmp_path / "state"
    trusted = _publish(state, pool, [history], [_evidence(tmp_path)])
    payload = trusted.marker_path.parent / catalog.CATALOG_PAYLOAD
    payload.chmod(0o644)
    value = json.loads(payload.read_text(encoding="utf-8"))
    value["entries"][0]["capacity_generation"] = 2
    payload.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    payload.chmod(0o444)
    with pytest.raises(catalog.GenerationCatalogError, match="inventory"):
        catalog.validate_trusted_generation_catalog(
            trusted.marker_path, server_pool_root=pool
        )

    other = tmp_path / "discontinuous"
    pool3, history3 = _endpoint_history(
        other, capacity=3, rollout=2, job_id="701", fleet="e" * 64
    )
    with pytest.raises(catalog.GenerationCatalogError, match="genesis"):
        _publish(
            other / "state",
            pool3,
            [history3],
            [_evidence(other)],
            capacity=3,
            rollout=2,
            fleet="e" * 64,
        )


def test_catalog_crash_archives_partial_and_retry_succeeds(tmp_path, monkeypatch):
    pool, history = _endpoint_history(tmp_path)
    state = tmp_path / "state"
    evidence = [_evidence(tmp_path)]
    original = catalog._write_new_file
    injected = {"done": False}

    def crash(path, payload, *, mode=0o444):
        if path.name == catalog.CATALOG_MARKER and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("catalog crash boundary")
        return original(path, payload, mode=mode)

    monkeypatch.setattr(catalog, "_write_new_file", crash)
    with pytest.raises(RuntimeError, match="crash boundary"):
        _publish(state, pool, [history], evidence)
    monkeypatch.setattr(catalog, "_write_new_file", original)
    recovered = _publish(state, pool, [history], evidence)
    stale = list(
        (catalog.catalog_root(state) / "stale-partials").glob(
            f"*/{catalog.CATALOG_MARKER}"
        )
    )
    assert len(stale) == 1
    assert recovered.allowed_generation_tuples == {history.allowed_generation_tuple}


def test_catalog_capacity_and_rollout_chain_are_exact(tmp_path):
    pool, g1 = _endpoint_history(tmp_path, capacity=1, rollout=1, job_id="700")
    state = tmp_path / "state"
    _publish(state, pool, [g1], [_evidence(tmp_path, "g1")])

    _, g2 = _endpoint_history(
        tmp_path,
        capacity=1,
        rollout=2,
        job_id="701",
    )
    second = _publish(
        state,
        pool,
        [g1, g2],
        [_evidence(tmp_path, "g2")],
        capacity=1,
        rollout=2,
    )
    assert [row["transition_kind"] for row in second.generations] == [
        "initial_readiness",
        "rollout_transition",
    ]

    _, g3 = _endpoint_history(
        tmp_path,
        capacity=2,
        rollout=3,
        job_id="702",
        fleet="e" * 64,
    )
    third = _publish(
        state,
        pool,
        [g1, g2, g3],
        [_evidence(tmp_path, "g3")],
        capacity=2,
        rollout=3,
        fleet="e" * 64,
    )
    assert [row["transition_kind"] for row in third.generations] == [
        "initial_readiness",
        "rollout_transition",
        "capacity_transition",
    ]
    assert len(third.allowed_generation_tuples) == 3
