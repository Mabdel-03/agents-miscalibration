"""Replica-aware serving: distinct ports per replica + keepalive spec parsing."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_scaling.serving.launch_server import _port_for, _register_role, render_sbatch

# keepalive.py lives in slurm/ (not the package); add it to the path for import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))


def test_port_distinct_per_replica():
    base = _port_for("8B", 0)
    assert _port_for("8B", 1) == base + 1
    assert _port_for("8B", 5) == base + 5
    # distinct replicas of the same size never collide
    ports = {_port_for("8B", r) for r in range(6)}
    assert len(ports) == 6


def test_port_distinct_across_sizes():
    # different sizes already differ via crc32 (replica 0)
    sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
    ports = {_port_for(s, 0) for s in sizes}
    assert len(ports) == len(sizes)


def test_long_profile_has_distinct_port_namespace():
    assert _port_for("32B-long", 0) != _port_for("32B", 0)


def test_port_in_range():
    for s in ["0.6B", "32B"]:
        for r in range(8):
            p = _port_for(s, r)
            assert 8000 <= p < 9100  # base 8000-8999 + small replica offset


def test_keepalive_spec_parse():
    from keepalive import parse_spec

    targets = parse_spec(
        "0.6B:1:pi_tpoggio:7-00:00:00,32B:6:ou_bcs_low:1-00:00:00", default_gpu="a100"
    )
    assert len(targets) == 2
    assert targets[0].size == "0.6B" and targets[0].count == 1
    assert targets[0].partition == "pi_tpoggio" and targets[0].time_limit == "7-00:00:00"
    assert targets[1].size == "32B" and targets[1].count == 6
    # time field with colons survives (split max 4)
    assert targets[1].time_limit == "1-00:00:00"

    long = parse_spec("32B-long:2:pi_tpoggio:7-00:00:00", default_gpu="a100")
    assert long[0].size == "32B-long" and long[0].count == 2


def test_keepalive_spec_rejects_malformed():
    from keepalive import parse_spec

    with pytest.raises(ValueError):
        parse_spec("8B:2:ou_bcs_low", default_gpu="a100")  # missing time field


def test_render_long_profile_uses_tp2_40k_but_serves_model_identity(tmp_path):
    from agents_scaling.serving import registry

    text = render_sbatch(
        "32B",
        str(tmp_path),
        "pi_tpoggio",
        "a100",
        "2-00:00:00",
        str(tmp_path / "logs"),
        serving_profile="32B-long",
    )
    assert (
        f"#SBATCH --job-name={registry.serving_job_name(tmp_path, '32B-long')}"
        in text
    )
    assert f"--server-pool-id \"{registry.server_pool_id(tmp_path)}\"" in text
    assert f'--run-root "{tmp_path.resolve()}"' in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --export=NONE" in text
    assert "#SBATCH --export=ALL" not in text
    assert "unset BASH_ENV CDPATH ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD" in text
    assert "export PATH=/usr/bin:/bin" in text
    assert "readonly PATH" in text
    assert "export GIT_NO_REPLACE_OBJECTS=1" in text
    assert text.index("#SBATCH --export=NONE") < text.index("set -euo pipefail")
    assert text.index("set -euo pipefail") < text.index("export PATH=/usr/bin:/bin")
    assert "#SBATCH --gres=gpu:a100:2" in text
    assert "--tensor-parallel-size 2" in text
    assert "--max-model-len 40960" in text
    assert '--served-model-name "32B"' in text
    assert '--model-size "32B"' in text
    assert '--profile "32B-long"' in text
    assert '--served-model-name "32B-long"' not in text
    # Registration receives the exact values rendered for vLLM instead of looking them
    # up from mutable shared profile code when a pending job eventually starts.
    assert text.count('--served-model-name "32B"') == 2
    assert text.count("--max-model-len 40960") == 2
    assert "--tp-size 2" in text


def test_render_long_profile_shorthand_keeps_model_identity(tmp_path):
    text = render_sbatch(
        "32B-long",
        str(tmp_path),
        "pi_tpoggio",
        "a100",
        "2-00:00:00",
        str(tmp_path / "logs"),
    )
    assert '--served-model-name "32B"' in text
    assert '--model-size "32B"' in text


def test_serving_job_names_and_comments_are_isolated_by_canonical_pool(tmp_path):
    from agents_scaling.serving import registry

    pool_a = tmp_path / "server_pools" / "schema5-v1-a"
    pool_b = tmp_path / "server_pools" / "schema5-v1-b"
    text_a = render_sbatch(
        "8B", str(pool_a), "ou_bcs_low", "a100", "1-00:00:00", "/logs"
    )
    text_b = render_sbatch(
        "8B", str(pool_b), "ou_bcs_low", "a100", "1-00:00:00", "/logs"
    )
    name_a = registry.serving_job_name(pool_a, "8B")
    name_b = registry.serving_job_name(pool_b, "8B")
    assert name_a != name_b
    assert f"#SBATCH --job-name={name_a}" in text_a
    assert f"#SBATCH --job-name={name_b}" in text_b
    assert (
        f"#SBATCH --comment=asys-schema5-pool:{registry.server_pool_id(pool_a)};"
        "profile=8B;replica=0"
    ) in text_a


# --- Registry garbage-collection (Change A) ---------------------------------------

def _register_fake(run_root, size, host, port, *, job_id=None, **runtime):
    """Drop a registry file by hand so we don't need a real vLLM server."""
    import json
    from agents_scaling.serving.registry import _size_dir, ServerEntry
    d = _size_dir(run_root, size)
    e = ServerEntry(
        model_size=size,
        hf_id=f"Qwen/Qwen3-{size}",
        host=host,
        port=port,
        slurm_job_id=job_id,
        **runtime,
    )
    (d / f"{host}_{port}.json").write_text(json.dumps({
        "model_size": e.model_size, "hf_id": e.hf_id, "host": e.host, "port": e.port,
        "slurm_job_id": e.slurm_job_id, "started_at": 0.0,
        "serving_profile": e.serving_profile,
        "served_model_name": e.served_model_name,
        "max_model_len": e.max_model_len,
        "tp_size": e.tp_size,
    }))
    return e


def test_list_live_servers_tolerantly_filters_without_pruning(tmp_path, monkeypatch):
    """One transient miss is tolerated and no client observation deletes shared state."""
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)
    _register_fake(tmp_path, "8B", "nodeC", 8003)

    monkeypatch.setattr(registry, "active_slurm_allocations", lambda _ids: {})
    attempts = {"nodeA": 0, "nodeB": 0, "nodeC": 0}

    def is_alive(host, _port, timeout=3.0):
        attempts[host] += 1
        return host == "nodeA" or (host == "nodeC" and attempts[host] == 3)

    monkeypatch.setattr("agents_scaling.serving.healthcheck.is_alive", is_alive)

    live = registry.list_live_servers(tmp_path, "8B")
    live_hp = {(e.host, e.port) for e in live}
    assert live_hp == {("nodeA", 8001), ("nodeC", 8003)}
    assert attempts == {"nodeA": 1, "nodeB": 3, "nodeC": 3}

    assert len(list(registry._size_dir(tmp_path, "8B").glob("*.json"))) == 3


def test_lookup_server_live_only_skips_dead(tmp_path, monkeypatch):
    """lookup_server(live_only=True) round-robins only across live endpoints; the
    default path (live_only=False) preserves the raw filesystem read used by keepalive."""
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)

    monkeypatch.setattr(registry, "active_slurm_allocations", lambda _ids: {})
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda host, port, timeout=3.0: host == "nodeA",
    )

    # raw path still sees nodeB
    raw_hp = {(registry.lookup_server(tmp_path, "8B", shard=s).host,
               registry.lookup_server(tmp_path, "8B", shard=s).port) for s in range(4)}
    # live-only path only ever returns nodeA
    live = registry.lookup_server(tmp_path, "8B", shard=0, live_only=True)
    assert (live.host, live.port) == ("nodeA", 8001)
    # raw set should include nodeB (proves the default path is unfiltered and that the
    # live-only observation did not mutate the registry).
    raw_hp = set()
    for s in range(4):
        e = registry.lookup_server(tmp_path, "8B", shard=s)
        raw_hp.add((e.host, e.port))
    assert ("nodeB", 8002) in raw_hp


def test_list_live_servers_empty_when_all_dead_but_records_survive(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)
    monkeypatch.setattr(registry, "active_slurm_allocations", lambda _ids: {})
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda host, port, timeout=3.0: False,
    )
    assert registry.list_live_servers(tmp_path, "8B") == []
    # Health is observational: keepalive/server lifecycle owns cleanup.
    sd = registry._size_dir(tmp_path, "8B")
    assert len(list(sd.glob("*.json"))) == 2


def test_running_slurm_allocation_is_authoritative_without_http_probe(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001, job_id="123")
    _register_fake(tmp_path, "8B", "nodeB", 8002, job_id="999")
    monkeypatch.setattr(
        registry, "active_slurm_allocations", lambda _ids: {"123": frozenset({"nodeA"})}
    )
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda *_args, **_kwargs: pytest.fail("job-linked endpoints should not be probed"),
    )

    live = registry.list_live_servers(tmp_path, "8B")
    assert [(entry.host, entry.port) for entry in live] == [("nodeA", 8001)]
    assert len(list(registry._size_dir(tmp_path, "8B").glob("*.json"))) == 2


def test_current_provenance_live_selection_excludes_incomplete_records(
    tmp_path, monkeypatch
):
    import json
    from agents_scaling.serving import registry

    directory = registry._size_dir(tmp_path, "8B")
    common = {
        "model_size": "8B",
        "hf_id": "Qwen/Qwen3-8B",
        "served_model_name": "8B",
        "max_model_len": 32768,
        "tp_size": 1,
    }
    records = [
        common
        | {
            "host": "complete",
            "port": 8001,
            "slurm_job_id": "100",
            "started_at": 12.0,
            "serving_profile": "8B",
        },
        common
        | {
            "host": "zero-time",
            "port": 8002,
            "slurm_job_id": "101",
            "started_at": 0.0,
            "serving_profile": "8B",
        },
        {
            "model_size": "8B",
            "hf_id": "Qwen/Qwen3-8B",
            "host": "legacy",
            "port": 8003,
            "slurm_job_id": None,
            "started_at": 0.0,
        },
    ]
    for record in records:
        (directory / f"{record['host']}_{record['port']}.json").write_text(
            json.dumps(record)
        )
    monkeypatch.setattr(
        registry,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"complete"})},
    )
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda *_a, **_k: pytest.fail("incomplete records must be filtered before probe"),
    )

    live = registry.list_live_servers(
        tmp_path, "8B", require_current_provenance=True
    )
    assert [(entry.host, entry.port) for entry in live] == [("complete", 8001)]


def test_slurm_query_failure_falls_back_to_tolerant_probe(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001, job_id="123")
    monkeypatch.setattr(registry, "active_slurm_allocations", lambda _ids: None)
    attempts = 0

    def is_alive(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        return attempts == 2

    monkeypatch.setattr("agents_scaling.serving.healthcheck.is_alive", is_alive)
    assert len(registry.list_live_servers(tmp_path, "8B")) == 1
    assert attempts == 2


def test_requeued_job_id_does_not_authorize_stale_old_node(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "old-node", 8001, job_id="123")
    _register_fake(tmp_path, "8B", "new-node", 8001, job_id="123")
    monkeypatch.setattr(
        registry,
        "active_slurm_allocations",
        lambda _ids: {"123": frozenset({"new-node"})},
    )
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda *_args, **_kwargs: pytest.fail("known moved allocations should not be probed"),
    )

    live = registry.list_live_servers(tmp_path, "8B")
    assert [(entry.host, entry.port) for entry in live] == [("new-node", 8001)]
    assert len(list(registry._size_dir(tmp_path, "8B").glob("*.json"))) == 2


def test_profile_validation_rejects_standard_or_partial_entry_from_long_pool(
    tmp_path, monkeypatch
):
    import json
    from agents_scaling.serving import registry
    from agents_scaling.serving.registry import ServerEntry

    directory = registry._size_dir(tmp_path, "32B-long")
    standard = ServerEntry(
        model_size="32B",
        hf_id="Qwen/Qwen3-32B",
        host="standard",
        port=8001,
    )
    partial = ServerEntry(
        model_size="32B",
        hf_id="Qwen/Qwen3-32B",
        host="partial",
        port=8002,
        serving_profile="32B-long",
        served_model_name="32B",
        max_model_len=16384,
        tp_size=1,
    )
    valid = ServerEntry(
        model_size="32B",
        hf_id="Qwen/Qwen3-32B",
        host="valid",
        port=8003,
        serving_profile="32B-long",
        served_model_name="32B",
        max_model_len=40960,
        tp_size=2,
    )
    for entry in (standard, partial, valid):
        (directory / f"{entry.host}_{entry.port}.json").write_text(
            json.dumps(entry.__dict__)
        )
    monkeypatch.setattr(registry, "active_slurm_allocations", lambda _ids: {})
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive", lambda *_args, **_kwargs: True
    )

    live = registry.list_live_servers(tmp_path, "32B-long")
    assert [(entry.host, entry.port) for entry in live] == [("valid", 8003)]
    assert len(list(directory.glob("*.json"))) == 3


def test_current_provenance_requires_explicit_layout_allocation_and_timestamp():
    from agents_scaling.serving import registry
    from agents_scaling.serving.registry import ServerEntry

    base = dict(
        model_size="8B",
        hf_id="Qwen/Qwen3-8B",
        host="nodeA",
        port=8001,
        serving_profile="8B",
        served_model_name="8B",
        max_model_len=32768,
        tp_size=1,
    )
    assert not registry.entry_has_current_provenance(ServerEntry(**base), "8B")
    assert not registry.entry_has_current_provenance(
        ServerEntry(**base, slurm_job_id="123", started_at=0.0), "8B"
    )
    assert registry.entry_has_current_provenance(
        ServerEntry(**base, slurm_job_id="123", started_at=10.25), "8B"
    )

    # A legacy standard entry can still be read for historical audit, but cannot be
    # selected to produce a current endpoint generation.
    legacy = ServerEntry(
        model_size="8B",
        hf_id="Qwen/Qwen3-8B",
        host="legacy",
        port=8002,
        slurm_job_id="124",
        started_at=10.0,
    )
    assert registry.entry_matches_profile(legacy, "8B")
    assert not registry.entry_has_current_provenance(legacy, "8B")


def test_long_profile_registry_is_separate_from_normal_model_pool(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    monkeypatch.setattr("socket.gethostname", lambda: "node-long")
    entry = registry.register_server(
        tmp_path,
        "32B",
        "Qwen/Qwen3-32B",
        8123,
        serving_profile="32B-long",
        served_model_name="32B",
        max_model_len=40960,
        tp_size=2,
    )

    assert entry.model_size == "32B"
    assert entry.registry_key == "32B-long"
    assert registry.list_servers(tmp_path, "32B") == []
    found = registry.list_servers(tmp_path, "32B-long")
    assert len(found) == 1
    assert found[0].served_model_name == "32B"
    assert found[0].max_model_len == 40960


def test_registry_rejects_cross_pool_identity_and_invalid_replica(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    monkeypatch.setattr("socket.gethostname", lambda: "node")
    with pytest.raises(ValueError, match="run-root/pool identity mismatch"):
        registry.register_server(
            tmp_path,
            "8B",
            "Qwen/Qwen3-8B",
            _port_for("8B", 0),
            expected_server_pool_id=registry.server_pool_id(tmp_path / "other"),
            replica_id=0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        registry.register_server(
            tmp_path,
            "8B",
            "Qwen/Qwen3-8B",
            _port_for("8B", 0),
            replica_id=-1,
        )


def test_complete_schema5_registration_is_staged_and_not_routable(
    tmp_path, monkeypatch
):
    from agents_scaling.serving import registry
    from agents_scaling.serving.model_contracts import load_model_contracts

    root = tmp_path / "server_pools" / "schema5-v1"
    contracts = load_model_contracts()
    identity = contracts.for_size("0.6B")
    fleet_sha256 = hashlib.sha256(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "schema5_fleet.v1.json"
        ).read_bytes()
    ).hexdigest()
    replica_id = "schema5-v1--0p6b--standard--r00"
    monkeypatch.setattr("socket.gethostname", lambda: "node-primary")
    monkeypatch.setenv("SLURM_JOB_ID", "700")
    entry = registry.register_server(
        root,
        "0.6B",
        identity.hf_id,
        _port_for("0.6B", 0),
        serving_profile="0.6B",
        served_model_name="0.6B",
        max_model_len=32768,
        tp_size=1,
        release_id="sweep-recovery-schema5-v1.2",
        environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=fleet_sha256,
        expected_server_pool_id="schema5-v1",
        replica_id=replica_id,
        replica_index=0,
        release_fleet_contract_sha256=fleet_sha256,
        capacity_generation=1,
        rollout_generation=1,
    )
    assert registry.read_standby_entry(root, "0.6B", replica_id, "700") == entry
    assert registry.read_promoted_entry(root, "0.6B", replica_id) is None
    assert registry.collect_endpoint_history_catalog(root).records == ()


def test_register_role_records_literal_spooled_layout_not_current_profile(
    tmp_path, monkeypatch
):
    from agents_scaling.serving import launch_server

    captured = {}
    monkeypatch.setattr(launch_server.healthcheck, "wait_until_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(
        launch_server.registry,
        "register_server",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs)
        or type("Entry", (), {"base_url": "http://test/v1"})(),
    )

    # Simulate a 32K job starting after the shared 32B-long profile moved to 40K.  Its
    # registry facts must remain 32K so entry_matches_profile rejects it.
    _register_role(
        str(tmp_path),
        "32B",
        "Qwen/Qwen3-32B",
        8826,
        serving_profile="32B-long",
        served_model_name="32B",
        max_model_len=32768,
        tp_size=2,
    )
    assert captured["args"][:3] == (
        str(tmp_path),
        "32B",
        "Qwen/Qwen3-32B",
    )
    assert captured["kwargs"]["serving_profile"] == "32B-long"
    assert captured["kwargs"]["max_model_len"] == 32768
    assert captured["kwargs"]["tp_size"] == 2


def _spooled_provenance(keepalive, run_root, profile_name, *, replica_id=0):
    from agents_scaling.serving import registry
    from agents_scaling.serving.model_contracts import load_model_contracts
    from agents_scaling.serving.profiles import get_serving_profile

    contracts = load_model_contracts()
    profile = get_serving_profile(profile_name)
    identity = contracts.for_size(profile.model_size)
    return keepalive.SpooledServingProvenance(
        run_root=str(Path(run_root).resolve()),
        server_pool_id=registry.server_pool_id(run_root),
        replica_id=replica_id,
        replica_index=int(replica_id),
        release_id="sweep-recovery-schema5-v1.2",
        environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=None,
    )


def test_keepalive_reregisters_verified_long_profile_with_runtime_metadata(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import registry

    port = _port_for("32B-long", 0)
    monkeypatch.setattr(
        keepalive,
        "_running_serve_endpoints",
        lambda name, **_kwargs: {("node-long", port): "123"},
    )
    provenance = _spooled_provenance(keepalive, tmp_path, "32B-long")
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda job_id, name, **_kwargs: (
            provenance if (job_id, name) == ("123", "32B-long") else None
        ),
    )
    monkeypatch.setattr(
        keepalive.healthcheck,
        "is_alive",
        lambda host, candidate_port, timeout=3.0: candidate_port == port,
    )

    assert keepalive._reregister_running(str(tmp_path), "32B-long") == 1
    entries = registry.list_servers(tmp_path, "32B-long")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.model_size == "32B"
    assert entry.serving_profile == "32B-long"
    assert entry.served_model_name == "32B"
    assert entry.tp_size == 2
    assert entry.max_model_len == 40960
    assert entry.slurm_job_id == "123"
    assert entry.started_at > 0
    assert entry.release_id == "sweep-recovery-schema5-v1.2"
    assert entry.environment_hash == "d" * 64
    assert entry.server_pool_id == provenance.server_pool_id
    assert entry.replica_id == 0
    assert registry.entry_has_current_provenance(entry, "32B-long")


def test_keepalive_atomically_upgrades_known_legacy_address(tmp_path, monkeypatch):
    import json
    import keepalive
    from agents_scaling.serving import registry
    from agents_scaling.serving.registry import ServerEntry

    port = _port_for("14B", 0)
    legacy = ServerEntry(
        model_size="14B",
        hf_id="Qwen/Qwen3-14B",
        host="node-standard",
        port=port,
    )
    destination = registry._size_dir(tmp_path, "14B") / f"{legacy.host}_{port}.json"
    destination.write_text(json.dumps(legacy.__dict__))

    monkeypatch.setattr(
        keepalive,
        "_running_serve_endpoints",
        lambda name, **_kwargs: {(legacy.host, port): "123"},
    )
    provenance = _spooled_provenance(keepalive, tmp_path, "14B")
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda job_id, name, **_kwargs: (
            provenance if (job_id, name) == ("123", "14B") else None
        ),
    )
    monkeypatch.setattr(keepalive.healthcheck, "is_alive", lambda *_a, **_k: True)
    monkeypatch.setattr(keepalive.time, "time", lambda: 456.25)
    atomic_calls = []
    real_atomic_write = keepalive.io.atomic_write_text

    def recording_atomic_write(path, payload):
        atomic_calls.append((Path(path), payload))
        real_atomic_write(path, payload)

    monkeypatch.setattr(keepalive.io, "atomic_write_text", recording_atomic_write)

    assert keepalive._reregister_running(str(tmp_path), "14B") == 1
    assert [path for path, _payload in atomic_calls] == [destination]
    upgraded = registry.list_servers(tmp_path, "14B")
    assert len(upgraded) == 1
    entry = upgraded[0]
    assert entry.slurm_job_id == "123"
    assert entry.started_at == 456.25
    assert entry.serving_profile == "14B"
    assert entry.served_model_name == "14B"
    assert entry.max_model_len == 32768
    assert entry.tp_size == 1
    assert registry.entry_has_current_provenance(entry, "14B")


def test_keepalive_does_not_skip_incomplete_but_refuses_unverified_upgrade(
    tmp_path, monkeypatch
):
    import json
    import keepalive
    from agents_scaling.serving import registry
    from agents_scaling.serving.registry import ServerEntry

    port = _port_for("8B", 0)
    incomplete = ServerEntry(
        model_size="8B",
        hf_id="Qwen/Qwen3-8B",
        host="node-standard",
        port=port,
        slurm_job_id="123",
        started_at=0.0,
        serving_profile="8B",
        served_model_name="8B",
        max_model_len=32768,
        tp_size=1,
    )
    destination = registry._size_dir(tmp_path, "8B") / f"{incomplete.host}_{port}.json"
    original = json.dumps(incomplete.__dict__)
    destination.write_text(original)
    monkeypatch.setattr(
        keepalive,
        "_running_serve_endpoints",
        lambda _name, **_kwargs: {(incomplete.host, port): "123"},
    )
    monkeypatch.setattr(
        keepalive, "_spooled_job_provenance", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        keepalive.healthcheck,
        "is_alive",
        lambda *_a, **_k: pytest.fail("unverified engine must not be probed"),
    )

    assert keepalive._reregister_running(str(tmp_path), "8B") == 0
    assert destination.read_text() == original


def test_keepalive_leaves_complete_current_record_untouched(tmp_path, monkeypatch):
    import json
    import keepalive
    from agents_scaling.serving import registry
    from agents_scaling.serving.registry import ServerEntry

    port = _port_for("8B", 0)
    current = ServerEntry(
        model_size="8B",
        hf_id="Qwen/Qwen3-8B",
        host="node-standard",
        port=port,
        slurm_job_id="123",
        started_at=100.0,
        serving_profile="8B",
        served_model_name="8B",
        max_model_len=32768,
        tp_size=1,
    )
    provenance = _spooled_provenance(keepalive, tmp_path, "8B")
    for field in (
        "release_id",
        "environment_hash",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_contract_sha256",
        "fleet_contract_sha256",
        "server_pool_id",
        "replica_id",
        "replica_index",
    ):
        setattr(current, field, getattr(provenance, field))
    destination = registry._size_dir(tmp_path, "8B") / f"{current.host}_{port}.json"
    original = json.dumps(current.__dict__)
    destination.write_text(original)
    monkeypatch.setattr(
        keepalive,
        "_running_serve_endpoints",
        lambda _name, **_kwargs: {(current.host, port): "123"},
    )
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_a, **_k: provenance,
    )
    monkeypatch.setattr(
        keepalive.healthcheck,
        "is_alive",
        lambda *_a, **_k: pytest.fail("complete record must not be reprobed every tick"),
    )

    assert keepalive._reregister_running(str(tmp_path), "8B") == 0
    assert destination.read_text() == original


def test_keepalive_repair_only_cli_never_enters_submission_tick(
    tmp_path, monkeypatch, capsys
):
    import keepalive

    repaired = []
    monkeypatch.setattr(
        keepalive,
        "_reregister_running",
        lambda run_root, profile: repaired.append((run_root, profile)) or 1,
    )
    monkeypatch.setattr(
        keepalive,
        "tick",
        lambda *_a, **_k: pytest.fail("repair-only mode must never enter submission tick"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "keepalive.py",
            "--run-id",
            "production",
            "--run-root",
            str(tmp_path),
            "--repair-profile",
            "14B",
            "--repair-profile",
            "14B",
            "--repair-profile",
            "32B",
            "--allow-legacy-fleet",
        ],
    )

    keepalive.main()

    assert repaired == [(str(tmp_path), "14B"), (str(tmp_path), "32B")]
    assert "repair-only complete: 2 registry records updated" in capsys.readouterr().out


def test_keepalive_refuses_unverifiable_nonstandard_profile(tmp_path, monkeypatch):
    import keepalive
    from agents_scaling.serving import registry

    port = _port_for("32B-long", 0)
    monkeypatch.setattr(
        keepalive,
        "_running_serve_endpoints",
        lambda name, **_kwargs: {("node-old", port): "old-job"},
    )
    monkeypatch.setattr(
        keepalive, "_spooled_job_provenance", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        keepalive.healthcheck,
        "is_alive",
        lambda *_args, **_kwargs: pytest.fail("unverified engine must not be probed"),
    )

    assert keepalive._reregister_running(str(tmp_path), "32B-long") == 0
    assert registry.list_servers(tmp_path, "32B-long") == []


def test_spooled_profile_verifier_rejects_old_layout_and_accepts_exact(monkeypatch):
    import keepalive

    exact = render_sbatch(
        "32B",
        "/results",
        "ou_bcs_low",
        "a100",
        "1-00:00:00",
        "/logs",
        serving_profile="32B-long",
    )

    class Proc:
        returncode = 0
        stdout = exact

    monkeypatch.setattr(keepalive.subprocess, "run", lambda *_a, **_k: Proc())
    assert keepalive._spooled_job_matches_profile("123", "32B-long")

    Proc.stdout = exact.replace("--max-model-len 40960", "--max-model-len 32768")
    assert not keepalive._spooled_job_matches_profile("123", "32B-long")

    Proc.stdout = exact.replace(
        "--reasoning-parser qwen3", "--reasoning-parser deepseek_r1"
    )
    assert not keepalive._spooled_job_matches_profile("123", "32B-long")

    # Standard profiles are verified just as strictly before an incomplete cross-pool
    # record is upgraded; compatibility inference is never sufficient for new traffic.
    Proc.stdout = render_sbatch(
        "8B", "/results", "ou_bcs_low", "a100", "1-00:00:00", "/logs"
    )
    assert keepalive._spooled_job_matches_profile("124", "8B")


def _frozen_environment_render_kwargs(tmp_path, *, generation=1):
    import hashlib
    import json
    import time
    from agents_scaling import runtime_integrity

    release_id = "sweep-recovery-schema5-v1.2"
    records = {}
    for role in ("harness", "serving"):
        prefix = tmp_path / f"{role}-env"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").write_text("#!/bin/sh\n")
        if role == "serving":
            (prefix / "bin" / "vllm").write_text(
                f"#!{prefix / 'bin' / 'python'}\n"
            )
            (prefix / "bin" / "vllm").chmod(0o555)
        (prefix / "bin" / "python").chmod(0o555)
        runtime = {
            "python_version": "3.11.15",
            "python_implementation": "CPython",
            "cuda_version": "13.0" if role == "serving" else None,
            "packages": {
                "torch": "2.11.0" if role == "serving" else None,
                "vllm": "0.21.0" if role == "serving" else None,
                "transformers": "5.9.0",
                "tokenizers": "0.22.2",
            },
        }
        (prefix / "bin").chmod(0o555)
        prefix.chmod(0o555)
        inventory = runtime_integrity.directory_inventory(prefix)
        locks = {
            "conda_explicit": [
                "@EXPLICIT",
                f"https://repo.example.invalid/{role}-runtime.conda#{'1' * 64}",
            ],
            "pip_freeze_all": [f"{role}-runtime==1.0.0"],
        }
        provenance = {
            "conda_creation_tool": {
                "path": f"/sealed/build-tools/{role}/conda",
                "sha256": "2" * 64,
            },
            "conda_toolchain": {
                "protocol": "schema5-v1.2-r4-offline-conda-toolchain-v1",
                "binding_id": "b" * 64,
            },
            "environment_seed": {
                "capture_id": "3" * 64,
                "capture_marker_sha256": "4" * 64,
                "prefix": f"/retired/build-inputs/{role}-seed",
                "normalized_content_inventory_sha256": "5" * 64,
            },
            "ownership_policy": {
                "path": "/retired/build-inputs/environment_ownership_policy.v1.json",
                "sha256": "6" * 64,
            },
            "integrity_normalization_policy": {
                "path": (
                    "/retired/build-inputs/"
                    "environment_integrity_normalization_policy.v1.json"
                ),
                "sha256": "9" * 64,
            },
            "normalization_receipt": {"id": "7" * 64},
            "conda_package_cache_sha256": "8" * 64,
            "conda_package_cache_seed_sha256": "a" * 64,
        }
        payload = {
            "schema_version": 4,
            "release_id": release_id,
            "role": role,
            "prefix": str(prefix.resolve()),
            "sealed_read_only": True,
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            **provenance,
            "runtime": runtime,
            "locks": locks,
            "release_package": None,
            "installed_files": {
                "inventory_sha256": inventory["inventory_sha256"],
                "entry_count": inventory["entry_count"],
                "file_count": inventory["file_count"],
                "total_file_bytes": inventory["total_file_bytes"],
            },
            "directory_inventory": inventory,
        }
        payload["environment_content_sha256"] = hashlib.sha256(
            runtime_integrity.canonical_bytes(
                {
                    "runtime": runtime,
                    "locks": locks,
                    "release_package": None,
                    "environment_seed": provenance["environment_seed"],
                    "ownership_policy": provenance["ownership_policy"],
                    "integrity_normalization_policy": provenance[
                        "integrity_normalization_policy"
                    ],
                    "normalization_receipt": provenance[
                        "normalization_receipt"
                    ],
                    "conda_creation_tool": provenance["conda_creation_tool"],
                    "conda_toolchain": provenance["conda_toolchain"],
                    "conda_package_cache_sha256": provenance[
                        "conda_package_cache_sha256"
                    ],
                    "conda_package_cache_seed_sha256": provenance[
                        "conda_package_cache_seed_sha256"
                    ],
                    "inventory_sha256": inventory["inventory_sha256"],
                }
            )
        ).hexdigest()
        manifest = tmp_path / f"{role}-environment.json"
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        manifest.write_bytes(raw)
        records[role] = (prefix, manifest, hashlib.sha256(raw).hexdigest())
    harness, serving = records["harness"], records["serving"]
    release_worktree = Path(__file__).resolve().parents[1]
    model_contract_path = release_worktree / "configs" / "model_contracts.v1.json"
    fleet_path = release_worktree / "configs" / "schema5_fleet.v1.json"
    fleet_sha256 = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
    environment_pins = {
        role: {
            "prefix": str(records[role][0]),
            "manifest_path": str(records[role][1]),
            "manifest_sha256": records[role][2],
        }
        for role in ("harness", "serving")
    }
    immutable_sha = "9" * 64
    state_dir = tmp_path / "runtime-state"
    state_dir.mkdir()
    attestation = runtime_integrity.ensure_generation_attestation(
        state_dir=state_dir,
        generation=generation,
        release_id=release_id,
        release_bundle_id="8" * 64,
        immutable_pins_sha256=immutable_sha,
        environment_pins=environment_pins,
    )
    lease = runtime_integrity.refresh_generation_lease(
        state_dir=state_dir,
        attestation_path=Path(attestation["path"]),
        attestation_sha256=attestation["sha256"],
        generation=generation,
        release_id=release_id,
        immutable_pins_sha256=immutable_sha,
        expected_environment_hashes={
            role: environment_pins[role]["manifest_sha256"]
            for role in ("harness", "serving")
        },
        expected_prefixes={
            role: environment_pins[role]["prefix"]
            for role in ("harness", "serving")
        },
        now=time.time(),
    )
    return {
        "release_worktree": str(release_worktree),
        "release_id": release_id,
        "environment_hash": serving[2],
        "model_contract_path": str(model_contract_path),
        "harness_environment_prefix": str(harness[0]),
        "serving_environment_prefix": str(serving[0]),
        "harness_environment_manifest_path": str(harness[1]),
        "serving_environment_manifest_path": str(serving[1]),
        "harness_environment_hash": harness[2],
        "fleet_contract_path": str(fleet_path),
        "fleet_contract_sha256": fleet_sha256,
        "release_fleet_contract_sha256": fleet_sha256,
        "capacity_generation": 1,
        "runtime_attestation": attestation["path"],
        "runtime_attestation_sha256": attestation["sha256"],
        "runtime_integrity_lease": lease["path"],
        "immutable_pins_sha256": immutable_sha,
        "rollout_generation": generation,
    }


def _prepare_transactional_fleet_attempt(
    root, fleet, replica, sbatch_text, *, token="1" * 32, now=10.0
):
    from agents_scaling.serving import fleet_transactions

    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[item.replica_id for item in fleet.replicas],
            now=now,
        )
        attempt = fleet_transactions.prepare_attempt(
            directory,
            ledger,
            replica_id=replica.replica_id,
            profile=replica.serving_profile,
            pool_id=replica.pool_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            sbatch_text=sbatch_text,
            now=now,
            token_factory=lambda: token,
        )
    return attempt


def _one_replica_fleet(fleet):
    from types import MappingProxyType

    replica = fleet.replicas[0]
    return replace(
        fleet,
        replicas=(replica,),
        by_profile=MappingProxyType({replica.serving_profile: (replica,)}),
    )


def _fleet_submission_command(attempt):
    from agents_scaling.serving import fleet_transactions

    return " ".join(
        fleet_transactions.submission_argv(attempt["scheduler_comment"])
    )


def _accepted_fleet_runner(job_id, calls=None):
    submitted = {}

    def run(argv, **kwargs):
        if calls is not None:
            calls.append(list(argv))
        if argv[0] == "sbatch":
            submitted["script"] = kwargs["input"]
            return subprocess.CompletedProcess(argv, 0, f"{job_id}\n", "")
        assert argv[:3] == ["scontrol", "write", "batch_script"]
        return subprocess.CompletedProcess(argv, 0, submitted["script"], "")

    return run


def _production_spooled_provenance(keepalive, root, fleet, replica):
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    identity = contracts.for_size(replica.model_size)
    return keepalive.SpooledServingProvenance(
        run_root=str(root.resolve()),
        server_pool_id=replica.pool_id,
        replica_id=replica.replica_id,
        replica_index=replica.replica_index,
        release_id="sweep-recovery-schema5-v1.2",
        environment_hash="d" * 64,
        model_revision=identity.model_revision,
        tokenizer_id=identity.tokenizer_id,
        tokenizer_revision=identity.tokenizer_revision,
        model_contract_sha256=contracts.sha256,
        fleet_contract_sha256=fleet.sha256,
    )


def _seed_committed_handoff(
    tmp_path,
    *,
    predecessor_job="700",
    successor_job="701",
):
    import keepalive
    from agents_scaling.serving import fleet_transactions, registry
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts
    from agents_scaling.serving.profiles import get_serving_profile

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    options = _frozen_environment_render_kwargs(tmp_path / "runtime")
    options["model_contract_sha256"] = contracts.sha256
    primary_script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **options,
    )
    standby_script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        standby=True,
        **options,
    )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=10.0,
        )
        primary = fleet_transactions.prepare_attempt(
            directory,
            ledger,
            replica_id=replica.replica_id,
            profile=replica.serving_profile,
            pool_id=replica.pool_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            sbatch_text=primary_script,
            now=10.0,
            token_factory=lambda: "1" * 32,
            allocated_gpus=replica.gpus_per_replica,
        )
        primary.update(
            {
                "state": "committed",
                "job_id": predecessor_job,
                "submitted_at": 10.0,
                "committed_at": 11.0,
                "last_seen_at": 20.0,
                "scheduler_start_at": 0.0,
                "scheduler_end_at": 86_400.0,
                "scheduler_time_limit_seconds": 86_400,
            }
        )
        fleet_transactions.save_ledger(directory, ledger, now=20.0)
        successor = fleet_transactions.prepare_attempt(
            directory,
            ledger,
            replica_id=replica.replica_id,
            profile=replica.serving_profile,
            pool_id=replica.pool_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            sbatch_text=standby_script,
            now=20.0,
            token_factory=lambda: "2" * 32,
            launch_kind="handoff",
            predecessor_job_id=predecessor_job,
            predecessor_end_at=86_400.0,
            predecessor_attempt=primary,
            allocated_gpus=replica.gpus_per_replica,
        )
        successor.update(
            {
                "state": "committed",
                "job_id": successor_job,
                "submitted_at": 20.0,
                "committed_at": 21.0,
                "last_seen_at": 30.0,
                "scheduler_start_at": 20.0,
                "scheduler_end_at": 86_420.0,
                "scheduler_time_limit_seconds": 86_400,
            }
        )
        fleet_transactions.save_ledger(directory, ledger, now=30.0)
    provenance = _production_spooled_provenance(
        keepalive, root, fleet, replica
    )
    profile = get_serving_profile(replica.serving_profile)
    primary_entry = registry.ServerEntry(
        model_size=profile.model_size,
        hf_id=profile.hf_id,
        host="node-old",
        port=_port_for(replica.serving_profile, replica.replica_index),
        slurm_job_id=predecessor_job,
        started_at=11.0,
        serving_profile=profile.name,
        served_model_name=profile.served_model_name,
        max_model_len=profile.max_model_len,
        tp_size=profile.tp_size,
        release_id=provenance.release_id,
        environment_hash=provenance.environment_hash,
        model_revision=provenance.model_revision,
        tokenizer_id=provenance.tokenizer_id,
        tokenizer_revision=provenance.tokenizer_revision,
        model_contract_sha256=provenance.model_contract_sha256,
        fleet_contract_sha256=provenance.fleet_contract_sha256,
        server_pool_id=provenance.server_pool_id,
        replica_id=provenance.replica_id,
        replica_index=provenance.replica_index,
    )
    successor_entry = replace(
        primary_entry,
        host="node-new",
        slurm_job_id=successor_job,
        started_at=21.0,
    )
    registry.promoted_entry_path(
        root, replica.serving_profile, replica.replica_id
    ).write_text(
        json.dumps(primary_entry.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        keepalive,
        fleet_transactions,
        registry,
        root,
        fleet,
        replica,
        options,
        provenance,
        primary_entry,
        successor_entry,
    )


def _strict_lineage(
    provenance,
    entry,
    *,
    fleet_sha256,
    script_path,
):
    script_sha256 = hashlib.sha256(Path(script_path).read_bytes()).hexdigest()
    exact_provenance = replace(
        provenance,
        release_fleet_contract_sha256=fleet_sha256,
        capacity_generation=1,
        rollout_generation=1,
        spooled_script_sha256=script_sha256,
    )
    exact_entry = replace(
        entry,
        release_fleet_contract_sha256=fleet_sha256,
        capacity_generation=1,
        rollout_generation=1,
    )
    return exact_provenance, exact_entry


def _seal_committed_fixture_endpoint(
    *,
    transactions,
    registry,
    root,
    fleet,
    replica,
    provenance,
    entry,
    now=32.0,
):
    """Upgrade one older fixture entry through the real immutable history gate."""

    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=now,
        )
        matches = [
            attempt
            for attempt in ledger["replicas"][replica.replica_id]["attempts"]
            if str(attempt.get("job_id") or "") == str(entry.slurm_job_id)
        ]
        assert len(matches) == 1
        attempt = matches[0]
        exact_provenance, exact_entry = _strict_lineage(
            provenance,
            entry,
            fleet_sha256=fleet.sha256,
            script_path=attempt["sbatch_path"],
        )
        registry.write_standby_entry(root, exact_entry)
        admission = transactions.committed_endpoint_admission(
            directory,
            ledger,
            replica_id=replica.replica_id,
            attempt=attempt,
            slurm_job_id=str(exact_entry.slurm_job_id),
        )
        script = admission.sbatch_path.read_bytes()
        script_sha256 = hashlib.sha256(script).hexdigest()
        spooled = {
            "run_root": str(root.resolve()),
            "server_pool_id": exact_provenance.server_pool_id,
            "replica_id": exact_provenance.replica_id,
            "replica_index": exact_provenance.replica_index,
            "release_id": exact_provenance.release_id,
            "environment_hash": exact_provenance.environment_hash,
            "model_revision": exact_provenance.model_revision,
            "tokenizer_id": exact_provenance.tokenizer_id,
            "tokenizer_revision": exact_provenance.tokenizer_revision,
            "model_contract_sha256": exact_provenance.model_contract_sha256,
            "release_fleet_contract_sha256": fleet.sha256,
            "fleet_contract_sha256": fleet.sha256,
            "capacity_generation": 1,
            "rollout_generation": 1,
            "effective_context_limit": exact_entry.max_model_len,
            "tp_size": exact_entry.tp_size,
            "spooled_script_sha256": script_sha256,
        }
        registry.seal_endpoint_history(
            root,
            exact_entry,
            release_fleet_contract_sha256=fleet.sha256,
            capacity_generation=1,
            rollout_generation=1,
            ledger_generation=1,
            intent_token=admission.intent_token,
            intent_state="committed",
            committed_at=admission.committed_at,
            local_script_path=admission.sbatch_path,
            local_script_sha256=admission.sbatch_sha256,
            spooled_script=script,
            spooled_provenance=spooled,
            scheduler_job_name=replica.scheduler_job_name,
            scheduler_comment=admission.scheduler_comment,
            sealed_at=now,
        )
    return exact_provenance, exact_entry


def test_initial_and_handoff_endpoints_seal_before_routing_and_survive_rotation(
    tmp_path,
):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        primary_entry,
        successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    pointer = registry.promoted_entry_path(
        root, replica.serving_profile, replica.replica_id
    )
    pointer.unlink()
    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=31.0,
        )
        primary, successor = ledger["replicas"][replica.replica_id]["attempts"]
        primary_provenance, primary_entry = _strict_lineage(
            provenance,
            primary_entry,
            fleet_sha256=fleet.sha256,
            script_path=primary["sbatch_path"],
        )
        registry.write_standby_entry(root, primary_entry)
        primary_row = keepalive.FleetQueueRow(
            "700",
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            "node-old",
            _fleet_submission_command(primary),
            primary["scheduler_comment"],
            "squeue",
            0.0,
            86_400.0,
            86_400,
            "",
            replica.qos,
        )

        assert not keepalive._advance_primary_registration(
            directory=directory,
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=primary,
            row=primary_row,
            provenance=primary_provenance,
            now=32.0,
            probe=lambda *_a: (True, True),
        )
        assert not pointer.exists()
        assert registry.collect_endpoint_history_catalog(root).records == ()

        assert keepalive._advance_primary_registration(
            directory=directory,
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=primary,
            row=primary_row,
            provenance=primary_provenance,
            now=33.0,
            probe=lambda *_a: (True, True),
        )
        assert registry.read_promoted_entry(
            root, replica.serving_profile, replica.replica_id
        ) == primary_entry
        # Promotion never consumes the only job-specific registration preimage.
        assert registry.standby_entry_path(
            root, replica.serving_profile, replica.replica_id, "700"
        ).is_file()

        successor_provenance, successor_entry = _strict_lineage(
            provenance,
            successor_entry,
            fleet_sha256=fleet.sha256,
            script_path=successor["sbatch_path"],
        )
        registry.write_standby_entry(root, successor_entry)
        successor_row = keepalive.FleetQueueRow(
            "701",
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            "node-new",
            _fleet_submission_command(successor),
            successor["scheduler_comment"],
            "squeue",
            20.0,
            86_420.0,
            86_400,
            "",
            replica.qos,
        )
        assert not keepalive._advance_handoff(
            directory=directory,
            ledgers=[ledger],
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=successor,
            row=successor_row,
            provenance=successor_provenance,
            now=34.0,
            probe=lambda *_a: (True, True),
        )
        assert registry.read_promoted_entry(
            root, replica.serving_profile, replica.replica_id
        ) == primary_entry
        assert keepalive._advance_handoff(
            directory=directory,
            ledgers=[ledger],
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=successor,
            row=successor_row,
            provenance=successor_provenance,
            now=35.0,
            probe=lambda *_a: (True, True),
        )
        assert registry.read_promoted_entry(
            root, replica.serving_profile, replica.replica_id
        ) == successor_entry
        assert registry.standby_entry_path(
            root, replica.serving_profile, replica.replica_id, "701"
        ).is_file()

    catalog = registry.collect_endpoint_history_catalog(root)
    assert [
        str(record.server_entry.slurm_job_id) for record in catalog.records
    ] == ["700", "701"]
    expected_tuples = {
        (
            fleet.sha256,
            fleet.sha256,
            1,
            1,
            registry.endpoint_instance_id(entry),
        )
        for entry in (primary_entry, successor_entry)
    }
    assert catalog.allowed_generation_tuples == expected_tuples
    # The live pointer rotated, but both immutable archives still verify in isolation.
    assert all(
        registry.validate_endpoint_history_marker(record.marker_path).server_entry
        == record.server_entry
        for record in catalog.records
    )


def test_endpoint_history_publication_is_crash_safe_idempotent_and_tamper_evident(
    tmp_path, monkeypatch
):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        primary_entry,
        _successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    registry.promoted_entry_path(
        root, replica.serving_profile, replica.replica_id
    ).unlink()
    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=31.0,
        )
        primary = ledger["replicas"][replica.replica_id]["attempts"][0]
        exact_provenance, exact_entry = _strict_lineage(
            provenance,
            primary_entry,
            fleet_sha256=fleet.sha256,
            script_path=primary["sbatch_path"],
        )
        registry.write_standby_entry(root, exact_entry)
        admission = transactions.committed_endpoint_admission(
            directory,
            ledger,
            replica_id=replica.replica_id,
            attempt=primary,
            slurm_job_id="700",
        )
        script = admission.sbatch_path.read_bytes()
        spooled = {
            "run_root": str(root.resolve()),
            "server_pool_id": exact_provenance.server_pool_id,
            "replica_id": exact_provenance.replica_id,
            "replica_index": exact_provenance.replica_index,
            "release_id": exact_provenance.release_id,
            "environment_hash": exact_provenance.environment_hash,
            "model_revision": exact_provenance.model_revision,
            "tokenizer_id": exact_provenance.tokenizer_id,
            "tokenizer_revision": exact_provenance.tokenizer_revision,
            "model_contract_sha256": exact_provenance.model_contract_sha256,
            "release_fleet_contract_sha256": fleet.sha256,
            "fleet_contract_sha256": fleet.sha256,
            "capacity_generation": 1,
            "rollout_generation": 1,
            "effective_context_limit": exact_entry.max_model_len,
            "tp_size": exact_entry.tp_size,
            "spooled_script_sha256": hashlib.sha256(script).hexdigest(),
        }
        kwargs = {
            "release_fleet_contract_sha256": fleet.sha256,
            "capacity_generation": 1,
            "rollout_generation": 1,
            "ledger_generation": 1,
            "intent_token": admission.intent_token,
            "intent_state": "committed",
            "committed_at": admission.committed_at,
            "local_script_path": admission.sbatch_path,
            "local_script_sha256": admission.sbatch_sha256,
            "spooled_script": script,
            "spooled_provenance": spooled,
            "scheduler_job_name": replica.scheduler_job_name,
            "scheduler_comment": admission.scheduler_comment,
            "sealed_at": 32.0,
        }

        original_write = registry._write_private_copy
        calls = 0

        def crash_once(path, payload):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated hard boundary")
            return original_write(path, payload)

        monkeypatch.setattr(registry, "_write_private_copy", crash_once)
        with pytest.raises(RuntimeError, match="simulated hard boundary"):
            registry.seal_endpoint_history(root, exact_entry, **kwargs)
        assert registry.collect_endpoint_history_catalog(root).records == ()

        monkeypatch.setattr(registry, "_write_private_copy", original_write)
        original_rename = registry.os.rename

        def publish_then_die(source, destination):
            original_rename(source, destination)
            raise KeyboardInterrupt("killed after atomic publication")

        monkeypatch.setattr(registry.os, "rename", publish_then_die)
        with pytest.raises(KeyboardInterrupt, match="atomic publication"):
            registry.seal_endpoint_history(root, exact_entry, **kwargs)
        with pytest.raises(ValueError, match="remains writable"):
            registry.collect_endpoint_history_catalog(root)

        monkeypatch.setattr(registry.os, "rename", original_rename)
        first = registry.seal_endpoint_history(root, exact_entry, **kwargs)
        second = registry.seal_endpoint_history(root, exact_entry, **kwargs)
        assert second.marker_sha256 == first.marker_sha256
        assert second.marker_path == first.marker_path
        with pytest.raises(ValueError, match="endpoint-history collision"):
            registry.seal_endpoint_history(
                root,
                exact_entry,
                **{**kwargs, "committed_at": admission.committed_at + 1.0},
            )

    binding = first.marker_path.parent / "BINDING.json"
    binding.chmod(0o600)
    raw = bytearray(binding.read_bytes())
    raw[0] = ord("[")
    binding.write_bytes(raw)
    binding.chmod(0o400)
    with pytest.raises(ValueError):
        registry.validate_endpoint_history_marker(first.marker_path)


def test_endpoint_history_rejects_symlink_shared_inode_and_unlisted_file(tmp_path):
    # The preceding integration test covers content-hash tampering.  Exercise the
    # three structural attacks independently on fresh sealed archives.
    for attack in ("symlink", "shared_inode", "unlisted"):
        case = tmp_path / attack
        (
            keepalive,
            transactions,
            registry,
            root,
            fleet,
            replica,
            _options,
            provenance,
            primary_entry,
            _successor_entry,
        ) = _seed_committed_handoff(case)
        pointer = registry.promoted_entry_path(
            root, replica.serving_profile, replica.replica_id
        )
        pointer.unlink()
        with transactions.transaction_lock(root) as directory:
            ledger = transactions.load_or_create_ledger(
                directory,
                pool_root=root,
                pool_id=fleet.fleet_id,
                fleet_sha256=fleet.sha256,
                rollout_generation=1,
                replica_ids=[replica.replica_id],
                now=31.0,
            )
            primary = ledger["replicas"][replica.replica_id]["attempts"][0]
            exact_provenance, exact_entry = _strict_lineage(
                provenance,
                primary_entry,
                fleet_sha256=fleet.sha256,
                script_path=primary["sbatch_path"],
            )
            registry.write_standby_entry(root, exact_entry)
            row = keepalive.FleetQueueRow(
                "700",
                replica.scheduler_job_name,
                "RUNNING",
                replica.partition,
                "node-old",
                _fleet_submission_command(primary),
                primary["scheduler_comment"],
                "squeue",
                0.0,
                86_400.0,
                86_400,
                "",
                replica.qos,
            )
            primary["ready_probe_count"] = 1
            transactions.save_ledger(directory, ledger, now=31.0)
            assert keepalive._advance_primary_registration(
                directory=directory,
                run_root=str(root),
                replica=replica,
                owner_ledger=ledger,
                attempt=primary,
                row=row,
                provenance=exact_provenance,
                now=32.0,
                probe=lambda *_a: (True, True),
            )
        marker = registry.endpoint_history_for_entry(root, exact_entry).marker_path
        directory = marker.parent
        directory.chmod(0o755)
        target = directory / "SPOOLED_SCRIPT.sbatch"
        target.chmod(0o600)
        target.unlink()
        if attack == "symlink":
            target.symlink_to(directory / "LOCAL_SCRIPT.sbatch")
        elif attack == "shared_inode":
            os.link(directory / "LOCAL_SCRIPT.sbatch", target)
        else:
            target.write_bytes((directory / "LOCAL_SCRIPT.sbatch").read_bytes())
            target.chmod(0o400)
            (directory / "UNLISTED").write_text("unexpected\n", encoding="utf-8")
        directory.chmod(0o555)
        with pytest.raises(ValueError):
            registry.validate_endpoint_history_marker(marker)


def test_standby_promotion_is_idempotent_after_pointer_replace_crash(tmp_path):
    (
        _keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        _primary,
        successor,
    ) = _seed_committed_handoff(tmp_path)
    _strict_provenance, successor = _seal_committed_fixture_endpoint(
        transactions=transactions,
        registry=registry,
        root=root,
        fleet=fleet,
        replica=replica,
        provenance=provenance,
        entry=successor,
    )
    staged = registry.standby_entry_path(
        root, replica.serving_profile, replica.replica_id, "701"
    )
    registry.promote_standby_entry(root, successor)
    before = registry.promoted_entry_path(
        root, replica.serving_profile, replica.replica_id
    ).read_bytes()
    assert staged.exists()

    # Simulates a controller dying after atomic pointer replacement, before the
    # successor lifecycle save. Production retains both immutable history and standby.
    registry.promote_standby_entry(root, successor)
    assert registry.read_promoted_entry(
        root, replica.serving_profile, replica.replica_id
    ) == successor
    assert registry.promoted_entry_path(
        root, replica.serving_profile, replica.replica_id
    ).read_bytes() == before


def test_nonproduction_legacy_standby_promotion_remains_compatible(tmp_path):
    (
        _keepalive,
        _transactions,
        registry,
        root,
        _fleet,
        replica,
        _options,
        _provenance,
        _primary,
        successor,
    ) = _seed_committed_handoff(tmp_path)
    legacy = replace(
        successor,
        release_id=None,
        environment_hash=None,
        model_revision=None,
        tokenizer_id=None,
        tokenizer_revision=None,
        model_contract_sha256=None,
        fleet_contract_sha256=None,
        release_fleet_contract_sha256=None,
        capacity_generation=None,
        rollout_generation=None,
    )
    staged = registry.write_standby_entry(root, legacy)
    registry.promote_standby_entry(root, legacy)
    assert not staged.exists()
    assert registry.read_promoted_entry(
        root, replica.serving_profile, replica.replica_id
    ) == legacy


def test_handoff_recovers_pointer_ahead_of_lifecycle_without_resubmission(
    tmp_path,
):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        _primary,
        successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    provenance, successor_entry = _seal_committed_fixture_endpoint(
        transactions=transactions,
        registry=registry,
        root=root,
        fleet=fleet,
        replica=replica,
        provenance=provenance,
        entry=successor_entry,
    )
    registry.promote_standby_entry(root, successor_entry)
    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id="schema5-v1",
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=40.0,
        )
        primary, successor = ledger["replicas"][replica.replica_id]["attempts"]
        primary["lifecycle"] = "retiring"
        successor["ready_probe_count"] = 2
        transactions.save_ledger(directory, ledger, now=40.0)
        row = keepalive.FleetQueueRow(
            "701",
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            "node-new",
            _fleet_submission_command(successor),
            successor["scheduler_comment"],
            "squeue",
            20.0,
            86_420.0,
            86_400,
            "",
            replica.qos,
        )
        assert keepalive._advance_handoff(
            directory=directory,
            ledgers=[ledger],
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=successor,
            row=row,
            provenance=provenance,
            now=50.0,
            probe=lambda *_a: (True, True),
        )
        assert successor["lifecycle"] == "promoted"
        assert primary["lifecycle"] == "retiring"
        assert successor["promoted_at"] == 50.0


def test_missing_standby_registry_is_rebuilt_only_after_clean_probe(tmp_path):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        primary_entry,
        _successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=40.0,
        )
        successor = ledger["replicas"][replica.replica_id]["attempts"][1]
        row = keepalive.FleetQueueRow(
            "701",
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            "node-new",
            _fleet_submission_command(successor),
            successor["scheduler_comment"],
            "squeue",
            20.0,
            86_420.0,
            86_400,
            "",
            replica.qos,
        )
        assert not keepalive._advance_handoff(
            directory=directory,
            ledgers=[ledger],
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=successor,
            row=row,
            provenance=provenance,
            now=50.0,
            probe=lambda *_a: (False, False),
        )
        staged = registry.standby_entry_path(
            root, replica.serving_profile, replica.replica_id, "701"
        )
        assert not staged.exists()
        assert registry.read_promoted_entry(
            root, replica.serving_profile, replica.replica_id
        ) == primary_entry

        assert not keepalive._advance_handoff(
            directory=directory,
            ledgers=[ledger],
            run_root=str(root),
            replica=replica,
            owner_ledger=ledger,
            attempt=successor,
            row=row,
            provenance=provenance,
            now=60.0,
            probe=lambda *_a: (True, True),
        )
        assert staged.is_file()
        assert successor["ready_probe_count"] == 1


def test_tick_launches_one_typed_standby_at_twelve_hour_lead(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    options = _frozen_environment_render_kwargs(tmp_path / "runtime")
    options["model_contract_sha256"] = contracts.sha256
    primary_script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **options,
    )
    prepared = _prepare_transactional_fleet_attempt(
        root, fleet, replica, primary_script, now=10.0
    )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=11.0,
        )
        primary = ledger["replicas"][replica.replica_id]["attempts"][0]
        primary.update(
            {
                "state": "committed",
                "job_id": "700",
                "submitted_at": 10.0,
                "committed_at": 11.0,
                "last_seen_at": 11.0,
            }
        )
        fleet_transactions.save_ledger(directory, ledger, now=11.0)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node-old",
        _fleet_submission_command(prepared),
        prepared["scheduler_comment"],
        "squeue",
        0.0,
        86_400.0,
        86_400,
        "",
        replica.qos,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (row,))
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_a, **_k: _production_spooled_provenance(
            keepalive, root, fleet, replica
        ),
    )
    monkeypatch.setattr(keepalive, "_reregister_running", lambda *_a, **_k: 0)
    submitted = []

    submit = _accepted_fleet_runner("701", submitted)

    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=options,
        now=50_000.0,
        submission_runner=submit,
        health_probe=lambda *_a: (True, True),
        cancellation_runner=lambda *_a, **_k: pytest.fail(
            "predecessor must drain only after promotion"
        ),
    )
    assert len(submitted) == 2
    persisted = json.loads(
        fleet_transactions.ledger_path(
            root / fleet_transactions.STATE_DIRECTORY, 1
        ).read_text(encoding="utf-8")
    )
    primary, standby = persisted["replicas"][replica.replica_id]["attempts"]
    assert primary["job_id"] == "700"
    assert standby["state"] == "submitted"
    assert standby["job_id"] == "701"
    assert standby["launch_kind"] == "handoff"
    assert standby["lifecycle"] == "standby"
    assert standby["predecessor_job_id"] == "700"
    assert standby["predecessor_end_at"] == 86_400.0
    assert "--standby" in Path(standby["sbatch_path"]).read_text(
        encoding="utf-8"
    )


def test_handoff_retirement_crash_retries_only_exact_predecessor(tmp_path):
    (
        keepalive,
        transactions,
        _registry,
        root,
        fleet,
        replica,
        _options,
        _provenance,
        _primary_entry,
        _successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    calls = []
    with transactions.transaction_lock(root) as directory:
        ledger = transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=100.0,
        )
        predecessor, successor = ledger["replicas"][replica.replica_id]["attempts"]
        predecessor["lifecycle"] = "retiring"
        successor.update(
            {
                "lifecycle": "promoted",
                "ready_probe_count": 2,
                "promoted_at": 100.0,
            }
        )
        transactions.save_ledger(directory, ledger, now=100.0)
        old_row = keepalive.FleetQueueRow(
            "700",
            replica.scheduler_job_name,
            "RUNNING",
            replica.partition,
            "node-old",
            _fleet_submission_command(predecessor),
            predecessor["scheduler_comment"],
            "squeue",
            0.0,
            86_400.0,
            86_400,
            "",
            replica.qos,
        )

        def die_after_scancel(argv, **_kwargs):
            calls.append(argv)
            raise KeyboardInterrupt("controller died after exact-ID scancel")

        with pytest.raises(KeyboardInterrupt, match="after exact-ID"):
            keepalive._retire_handoff_predecessor(
                directory=directory,
                ledgers=[ledger],
                replica=replica,
                promoted_attempt=successor,
                scheduler_rows_by_job={"700": old_row},
                now=800.0,
                cancellation_runner=die_after_scancel,
            )
        assert predecessor["retire_attempts"] == 1
        assert predecessor["last_retire_attempt_at"] == 800.0

        keepalive._retire_handoff_predecessor(
            directory=directory,
            ledgers=[ledger],
            replica=replica,
            promoted_attempt=successor,
            scheduler_rows_by_job={"700": old_row},
            now=1_101.0,
            cancellation_runner=lambda argv, **_kwargs: calls.append(argv)
            or subprocess.CompletedProcess(argv, 0, "", ""),
        )
        assert calls == [["scancel", "700"], ["scancel", "700"]]
        assert predecessor["retire_attempts"] == 2
        assert predecessor["retire_error"] is None


def test_readiness_rejects_overlap_then_accepts_sole_promoted_handoff(
    tmp_path, monkeypatch
):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        _primary_entry,
        successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    provenance, successor_entry = _seal_committed_fixture_endpoint(
        transactions=transactions,
        registry=registry,
        root=root,
        fleet=fleet,
        replica=replica,
        provenance=provenance,
        entry=successor_entry,
    )
    directory = root / transactions.STATE_DIRECTORY
    ledger = json.loads(
        transactions.ledger_path(directory, 1).read_text(encoding="utf-8")
    )
    primary, successor = ledger["replicas"][replica.replica_id]["attempts"]
    old_row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node-old",
        _fleet_submission_command(primary),
        primary["scheduler_comment"],
        "squeue",
        0.0,
        86_400.0,
        86_400,
        "",
        replica.qos,
    )
    new_row = keepalive.FleetQueueRow(
        "701",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node-new",
        _fleet_submission_command(successor),
        successor["scheduler_comment"],
        "squeue",
        20.0,
        86_420.0,
        86_400,
        "",
        replica.qos,
    )
    observed = transactions.SchedulerSnapshot(
        rows=(old_row, new_row),
        captured_at=40.0,
        squeue_ok=True,
        sacct_ok=True,
    )
    monkeypatch.setattr(transactions, "query_scheduler", lambda **_k: observed)
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_a, **_k: provenance,
    )
    with pytest.raises(
        keepalive.FleetContractError, match="forbids an in-progress handoff"
    ):
        keepalive.reconcile_fleet_read_only(
            str(root), fleet, current_generation=1
        )

    registry.promote_standby_entry(root, successor_entry)
    with transactions.transaction_lock(root) as locked:
        material = transactions.load_or_create_ledger(
            locked,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=50.0,
        )
        predecessor, promoted = material["replicas"][replica.replica_id]["attempts"]
        predecessor.update(
                {
                    "state": "terminal",
                    "terminal_at": 50.0,
                    "lifecycle": "retiring",
                    "retire_requested_at": 49.0,
                    "retire_attempts": 1,
                    "last_retire_attempt_at": 49.0,
                    "retire_error": None,
                }
        )
        promoted.update(
            {
                "lifecycle": "promoted",
                "ready_probe_count": 2,
                "last_ready_probe_at": 50.0,
                "promoted_at": 50.0,
            }
        )
        endpoint = (
            f"node-new:{_port_for(replica.serving_profile, replica.replica_index)}"
        )
        material["replicas"][replica.replica_id]["health"] = (
            keepalive._new_health_record(
                "701", endpoint, observer_generation=1
            )
        )
        transactions.save_ledger(locked, material, now=50.0)
    terminal_old = replace(
        old_row,
        state="CANCELLED by 1",
        source="sacct",
        start_timestamp=None,
        end_timestamp=None,
        time_limit_seconds=None,
        dependency=None,
    )
    observed = transactions.SchedulerSnapshot(
        rows=(terminal_old, new_row),
        captured_at=51.0,
        squeue_ok=True,
        sacct_ok=True,
    )
    snapshot = keepalive.reconcile_fleet_read_only(
        str(root), fleet, current_generation=1
    )
    assert [item.row.job_id for item in snapshot.allocations] == ["701"]


def test_self_heal_never_overwrites_promoted_pointer_with_predecessor(
    tmp_path, monkeypatch
):
    (
        keepalive,
        transactions,
        registry,
        root,
        fleet,
        replica,
        _options,
        provenance,
        _primary_entry,
        successor_entry,
    ) = _seed_committed_handoff(tmp_path)
    provenance, successor_entry = _seal_committed_fixture_endpoint(
        transactions=transactions,
        registry=registry,
        root=root,
        fleet=fleet,
        replica=replica,
        provenance=provenance,
        entry=successor_entry,
    )
    registry.promote_standby_entry(root, successor_entry)
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_a, **_k: provenance,
    )
    monkeypatch.setattr(
        keepalive.healthcheck,
        "is_alive",
        lambda *_a, **_k: True,
    )

    restored = keepalive._reregister_running(
        str(root),
        replica.serving_profile,
        running_override={
            (
                "node-old",
                _port_for(replica.serving_profile, replica.replica_index),
            ): "700"
        },
    )
    assert restored == 0
    assert registry.read_promoted_entry(
        root, replica.serving_profile, replica.replica_id
    ) == successor_entry


def test_schema5_server_rejects_expired_lease_after_nested_vllm_drift(
    tmp_path, monkeypatch
):
    import json
    from agents_scaling import runtime_integrity
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    root = tmp_path / "server_pools" / "schema5-v1"
    options = _frozen_environment_render_kwargs(tmp_path)
    vllm = Path(options["serving_environment_prefix"]) / "bin" / "vllm"
    vllm.chmod(0o755)
    vllm.write_text("# corrupted nested vllm executable\n", encoding="utf-8")
    vllm.chmod(0o555)

    with pytest.raises(
        runtime_integrity.RuntimeIntegrityError,
        match="serving runtime metadata drifted",
    ):
        runtime_integrity.refresh_generation_lease(
            state_dir=Path(options["runtime_integrity_lease"]).parent.parent,
            attestation_path=Path(options["runtime_attestation"]),
            attestation_sha256=options["runtime_attestation_sha256"],
            generation=1,
            release_id=options["release_id"],
            immutable_pins_sha256=options["immutable_pins_sha256"],
            expected_environment_hashes={
                "harness": options["harness_environment_hash"],
                "serving": options["environment_hash"],
            },
            expected_prefixes={
                "harness": options["harness_environment_prefix"],
                "serving": options["serving_environment_prefix"],
            },
            force=True,
        )
    lease = json.loads(Path(options["runtime_integrity_lease"]).read_text())
    monkeypatch.setattr(
        runtime_integrity.time,
        "time",
        lambda: float(lease["expires_timestamp"]) + 0.001,
    )
    with pytest.raises(ValueError, match="runtime integrity lease expired"):
        render_sbatch(
            "8B",
            str(root),
            "ou_bcs_low",
            "a100",
            "1-00:00:00",
            "/logs",
            replica=2,
            model_contract_sha256=contracts.sha256,
            **options,
        )


def test_spooled_production_recovery_requires_exact_pool_root_and_full_pins(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import registry
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    root = tmp_path / "server_pools" / "schema5-v1"
    environment = _frozen_environment_render_kwargs(tmp_path)
    exact = render_sbatch(
        "8B",
        str(root),
        "ou_bcs_normal",
        "a100",
        "1-00:00:00",
        "/logs",
        replica=2,
        model_contract_sha256=contracts.sha256,
        **environment,
    )

    class Proc:
        returncode = 0
        stdout = exact

    monkeypatch.setattr(keepalive.subprocess, "run", lambda *_a, **_k: Proc())
    recovered = keepalive._spooled_job_provenance(
        "123", "8B", run_root=str(root), expected_script=exact
    )
    assert recovered is not None
    assert recovered.run_root == str(root.resolve())
    assert recovered.server_pool_id == registry.server_pool_id(root)
    assert recovered.replica_id == "schema5-v1--8b--standard--r02"
    assert recovered.replica_index == 2
    assert recovered.release_id == "sweep-recovery-schema5-v1.2"
    assert recovered.environment_hash == environment["environment_hash"]
    assert recovered.model_contract_sha256 == contracts.sha256
    assert (
        recovered.release_fleet_contract_sha256
        == environment["release_fleet_contract_sha256"]
    )
    assert recovered.capacity_generation == 1
    assert recovered.rollout_generation == 1
    assert recovered.spooled_script_sha256 == hashlib.sha256(
        exact.encode("utf-8")
    ).hexdigest()

    assert (
        keepalive._spooled_job_provenance(
            "123", "8B", run_root=str(tmp_path / "other-pool")
        )
        is None
    )
    serving_hash = environment["environment_hash"]
    Proc.stdout = exact.replace(serving_hash, serving_hash[:-1])
    assert (
        keepalive._spooled_job_provenance("123", "8B", run_root=str(root))
        is None
    )
    # Even a semantically harmless byte change is a different immutable launch record.
    Proc.stdout = exact + "\n"
    assert (
        keepalive._spooled_job_provenance(
            "123", "8B", run_root=str(root), expected_script=exact
        )
        is None
    )


def test_keepalive_counts_only_the_requested_pool(tmp_path, monkeypatch):
    import keepalive
    from agents_scaling.serving import registry

    pool_a = tmp_path / "pool-a"
    pool_b = tmp_path / "pool-b"
    output = "\n".join(
        [
            f"{registry.serving_job_name(pool_a, '8B')}|R|ou_bcs_low",
            f"{registry.serving_job_name(pool_a, '8B')}|PD|ou_bcs_low",
            f"{registry.serving_job_name(pool_b, '8B')}|R|ou_bcs_low",
            "asys-serve-8B|R|ou_bcs_low",
        ]
    )
    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": output})(),
    )
    assert keepalive._serve_jobs_in_flight(
        "8B", "ou_bcs_low", run_root=str(pool_a)
    ) == 2
    assert keepalive._serve_jobs_in_flight(
        "8B", "ou_bcs_low", run_root=str(pool_b)
    ) == 1


def test_frozen_fleet_contract_names_and_places_all_22_replicas():
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = load_fleet_contract(None, model_contracts=load_model_contracts())
    assert fleet.fleet_id == "schema5-v1"
    assert len(fleet.replicas) == 22
    assert sum(replica.gpus_per_replica for replica in fleet.replicas) == 24
    assert len({replica.replica_id for replica in fleet.replicas}) == 22
    assert len({replica.scheduler_job_name for replica in fleet.replicas}) == 22
    assert fleet.for_replica("14B-long", 0).partition == "ou_bcs_normal"
    assert fleet.for_replica("32B-long", 0).partition == "ou_bcs_normal"
    assert fleet.for_replica("32B-long", 1).partition == "ou_bcs_normal"
    assert all(replica.gpu_type == "a100" for replica in fleet.replicas)


def test_runtime_fleet_loader_rejects_resigned_replica_identity_drift(tmp_path):
    import hashlib
    import json

    from agents_scaling.serving.fleet_contract import (
        FleetContractError,
        load_fleet_contract,
    )
    from agents_scaling.serving.model_contracts import load_model_contracts

    source = Path(__file__).resolve().parents[1] / "configs" / "schema5_fleet.v1.json"
    target = tmp_path / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["profiles"][0]["replicas"][0]["scheduler_job_name"] = (
        "asys-s5-serve-renamed-r00"
    )
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    target.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(FleetContractError, match="invalid replica placement"):
        load_fleet_contract(
            target,
            model_contracts=load_model_contracts(),
            allow_capacity_layout=True,
        )


def test_capacity_fleet_loader_allows_only_append_only_exact_replica_layout(tmp_path):
    import hashlib
    import json

    from agents_scaling.serving.fleet_contract import (
        expected_replica_id,
        expected_scheduler_job_name,
        load_fleet_contract,
    )
    from agents_scaling.serving.model_contracts import load_model_contracts

    source = Path(__file__).resolve().parents[1] / "configs" / "schema5_fleet.v1.json"
    target = tmp_path / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    profile = next(
        row for row in payload["profiles"] if row["serving_profile"] == "32B-long"
    )
    added = dict(profile["replicas"][-1])
    added.update(
        {
            "replica_index": 2,
            "replica_id": expected_replica_id("32B-long", 2),
            "scheduler_job_name": expected_scheduler_job_name("32B-long", 2),
            "partition": "ou_bcs_low",
        }
    )
    profile["replicas"].append(added)
    payload["logical_replica_count"] = 23
    payload["allocated_gpu_count"] = 26
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )

    fleet = load_fleet_contract(
        target,
        model_contracts=load_model_contracts(),
        expected_sha256=digest,
        allow_capacity_layout=True,
    )
    assert len(fleet.replicas) == 23
    assert sum(row.gpus_per_replica for row in fleet.replicas) == 26
    assert fleet.for_replica("32B-long", 2).scheduler_job_name == (
        expected_scheduler_job_name("32B-long", 2)
    )


def test_production_fleet_scheduler_query_fails_closed(monkeypatch):
    import keepalive

    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *_a, **_k: type(
            "Proc", (), {"returncode": 1, "stdout": "", "stderr": "slurm down"}
        )(),
    )
    with pytest.raises(
        keepalive.FleetContractError, match=r"complete squeue\+sacct truth"
    ):
        keepalive._query_fleet_queue()


def test_production_fleet_empty_scheduler_launches_exact_contract(tmp_path, monkeypatch):
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = load_fleet_contract(None, model_contracts=load_model_contracts())
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    calls = []
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = load_model_contracts().sha256
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())

    def submit(directory, ledger, *, replica_id, attempt, now, **_kwargs):
        calls.append((replica_id, dict(attempt)))
        job_id = str(len(calls))
        attempt.update(
            {
                "state": "submitted",
                "job_id": job_id,
                "submitted_at": now,
            }
        )
        fleet_transactions.save_ledger(directory, ledger, now=now)
        return job_id

    monkeypatch.setattr(
        keepalive.fleet_tx,
        "submit_attempt",
        submit,
    )

    keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)

    assert len(calls) == 22
    assert [replica_id for replica_id, _attempt in calls] == [
        replica.replica_id for replica in fleet.replicas
    ]
    assert all(attempt["state"] == "prepared" for _replica, attempt in calls)
    assert all(
        attempt["scheduler_comment"].startswith("asys-s5-fleet:")
        for _replica, attempt in calls
    )


def test_accepted_fleet_id_absence_never_retries_duplicate(tmp_path, monkeypatch):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = contracts.sha256
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())

    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=launch_options,
        now=10.0,
        submission_runner=_accepted_fleet_runner("321"),
    )

    with pytest.raises(keepalive.FleetContractError, match="duplicate replacement"):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options=launch_options,
            now=400.0,
            submission_runner=lambda *_args, **_kwargs: pytest.fail(
                "accepted fleet job must never be resubmitted while ambiguous"
            ),
        )


def test_keepalive_cannot_retry_ambiguous_submit_from_raw_rows(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = contracts.sha256
    script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **launch_options,
    )
    _prepare_transactional_fleet_attempt(
        root, fleet, replica, script, now=10.0
    )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=11.0,
        )
        attempt = ledger["replicas"][replica.replica_id]["attempts"][0]
        attempt.update(
            {
                "state": "submitting",
                "submit_started_at": 10.0,
                "submission_attempts": 1,
            }
        )
        fleet_transactions.save_ledger(directory, ledger, now=11.0)
    # A raw row tuple has no independently retained squeue+sacct completion facts.
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())
    with pytest.raises(
        keepalive.FleetContractError,
        match="exact complete scheduler snapshot",
    ):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options=launch_options,
            now=400.0,
            submission_runner=lambda *_args, **_kwargs: pytest.fail(
                "unproven ambiguous submit must not be retried"
            ),
        )
    persisted = json.loads(
        fleet_transactions.ledger_path(
            root / fleet_transactions.STATE_DIRECTORY, 1
        ).read_text(encoding="utf-8")
    )
    assert (
        persisted["replicas"][replica.replica_id]["attempts"][0]["state"]
        == "submitting"
    )


def test_production_fleet_adopts_only_exact_rendered_runtime_and_scheduler_comment(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = load_fleet_contract(None, model_contracts=contracts)
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = contracts.sha256
    exact = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **launch_options,
    )
    attempt = _prepare_transactional_fleet_attempt(
        root, fleet, replica, exact
    )
    comment = attempt["scheduler_comment"]
    row = keepalive.FleetQueueRow(
        "123",
        replica.scheduler_job_name,
        "PENDING",
        replica.partition,
        "(null)",
        _fleet_submission_command(attempt),
        comment,
        "squeue",
        None,
        None,
        86_400,
        "",
        replica.qos,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (row,))
    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *_a, **_k: type(
            "Proc", (), {"returncode": 0, "stdout": exact, "stderr": ""}
        )(),
    )
    launches = []

    def submit(_directory, _ledger, *, replica_id, attempt, **_kwargs):
        launches.append((replica_id, dict(attempt)))
        return str(1000 + len(launches))

    monkeypatch.setattr(
        keepalive.fleet_tx,
        "submit_attempt",
        submit,
    )

    keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)
    assert len(launches) == 21

    drifted = replace(row, comment="operator-mutated-comment")
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (drifted,))
    with pytest.raises(keepalive.FleetContractError, match="intent provenance"):
        keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)


def test_production_fleet_rejects_unknown_and_duplicate_job_names(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = load_fleet_contract(None, model_contracts=load_model_contracts())
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = load_model_contracts().sha256
    unknown = keepalive.FleetQueueRow(
        "1", "asys-s5-serve-unknown", "PENDING", "ou_bcs_low", "(null)", "x", "x"
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (unknown,))
    with pytest.raises(keepalive.FleetContractError, match="unmappable"):
        keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)

    first = fleet.replicas[0]
    exact = render_sbatch(
        first.model_size,
        str(root),
        first.partition,
        first.gpu_type,
        first.time_limit,
        str(root / "logs"),
        replica=first.replica_index,
        serving_profile=first.serving_profile,
        **launch_options,
    )
    attempt = _prepare_transactional_fleet_attempt(root, fleet, first, exact)
    duplicate_one = keepalive.FleetQueueRow(
        "2",
        first.scheduler_job_name,
        "PENDING",
        first.partition,
        "(null)",
        _fleet_submission_command(attempt),
        attempt["scheduler_comment"],
        qos=first.qos,
    )
    duplicate_two = replace(duplicate_one, job_id="3")
    monkeypatch.setattr(
        keepalive,
        "_query_fleet_queue",
        lambda: (duplicate_one, duplicate_two),
    )
    with pytest.raises(keepalive.FleetContractError, match="ambiguous duplicate"):
        keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)


def test_production_fleet_ignores_only_unknown_terminal_accounting_history(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = contracts.sha256
    replica = fleet.replicas[0]
    foreign_comment = (
        f"asys-s5-fleet:pool={fleet.fleet_id};profile={replica.serving_profile};"
        f"replica={replica.replica_id};generation=1;intent={'f' * 32};"
        f"fleet={fleet.sha256}"
    )
    terminal = keepalive.FleetQueueRow(
        "699",
        "asys-srv-retired-test-allocation",
        "CANCELLED by 1",
        replica.partition,
        "None assigned",
        "/retired/test.sbatch",
        foreign_comment,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (terminal,))
    submitted = []

    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=launch_options,
        submission_runner=_accepted_fleet_runner("700", submitted),
    )
    assert len(submitted) == 2

    active = replace(terminal, job_id="701", state="RUNNING", node="node001")
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (active,))
    with pytest.raises(keepalive.FleetContractError, match="unknown fleet intent"):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options=launch_options,
            submission_runner=lambda *_a, **_k: pytest.fail("must fail closed"),
        )


def test_new_rollout_adopts_exact_active_prior_generation_without_relaunch(
    tmp_path, monkeypatch
):
    import json
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    generation_one = _frozen_environment_render_kwargs(
        tmp_path / "generation-one", generation=1
    )
    generation_one["model_contract_sha256"] = contracts.sha256
    old_script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **generation_one,
    )
    attempt = _prepare_transactional_fleet_attempt(
        root, fleet, replica, old_script, now=10.0
    )
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "PENDING",
        replica.partition,
        "(null)",
        _fleet_submission_command(attempt),
        attempt["scheduler_comment"],
        "squeue",
        None,
        None,
        86_400,
        "",
        replica.qos,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (row,))
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_a, **_k: _production_spooled_provenance(
            keepalive, root, fleet, replica
        ),
    )
    generation_two = _frozen_environment_render_kwargs(
        tmp_path / "generation-two", generation=2
    )
    generation_two["model_contract_sha256"] = contracts.sha256

    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=generation_two,
        now=20.0,
        submission_runner=lambda *_a, **_k: pytest.fail(
            "an exact active prior-generation server must be adopted"
        ),
    )

    directory = root / fleet_transactions.STATE_DIRECTORY
    current = json.loads((directory / fleet_transactions.CURRENT_FILENAME).read_text())
    old = json.loads(fleet_transactions.ledger_path(directory, 1).read_text())
    new = json.loads(fleet_transactions.ledger_path(directory, 2).read_text())
    assert current["current_generation"] == 2
    assert old["replicas"][replica.replica_id]["attempts"][0]["state"] == "committed"
    assert old["replicas"][replica.replica_id]["attempts"][0]["job_id"] == "700"
    assert new["replicas"][replica.replica_id]["attempts"] == []
    committed_at = old["replicas"][replica.replica_id]["attempts"][0][
        "committed_at"
    ]
    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=generation_two,
        now=30.0,
        submission_runner=lambda *_a, **_k: pytest.fail(
            "an already committed exact replica must not resubmit"
        ),
    )
    reread = json.loads(fleet_transactions.ledger_path(directory, 1).read_text())
    assert reread["replicas"][replica.replica_id]["attempts"][0][
        "committed_at"
    ] == committed_at


def test_read_only_fleet_readiness_uses_committed_transaction_and_rejects_pending(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    original_query_scheduler = fleet_transactions.query_scheduler
    original_spooled_provenance = keepalive._spooled_job_provenance
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    launch_options = _frozen_environment_render_kwargs(tmp_path)
    launch_options["model_contract_sha256"] = contracts.sha256
    script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **launch_options,
    )
    prepared = _prepare_transactional_fleet_attempt(
        root, fleet, replica, script, now=10.0
    )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=20.0,
        )
        attempt = ledger["replicas"][replica.replica_id]["attempts"][0]
        attempt.update(
            {
                "state": "committed",
                "job_id": "700",
                "submitted_at": 11.0,
                "committed_at": 12.0,
                "last_seen_at": 20.0,
            }
        )
        endpoint = (
            f"node001:{_port_for(replica.serving_profile, replica.replica_index)}"
        )
        ledger["replicas"][replica.replica_id]["health"] = (
            keepalive._new_health_record(
                "700", endpoint, observer_generation=1
            )
        )
        fleet_transactions.save_ledger(directory, ledger, now=20.0)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        _fleet_submission_command(prepared),
        prepared["scheduler_comment"],
        "squeue",
        100.0,
        86_500.0,
        86_400,
        "",
        replica.qos,
    )
    observed = fleet_transactions.SchedulerSnapshot(
        rows=(row,),
        captured_at=21.0,
        squeue_ok=True,
        sacct_ok=True,
    )
    monkeypatch.setattr(
        fleet_transactions,
        "query_scheduler",
        lambda **_kwargs: observed,
    )
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_args, **_kwargs: _production_spooled_provenance(
            keepalive, root, fleet, replica
        ),
    )

    snapshot = keepalive.reconcile_fleet_read_only(
        str(root), fleet, current_generation=1
    )
    assert snapshot.current_generation == 1
    assert snapshot.captured_at == 21.0
    assert [allocation.replica_id for allocation in snapshot.allocations] == [
        replica.replica_id
    ]
    assert snapshot.allocations[0].intent_token == prepared["intent_token"]

    monkeypatch.setattr(
        fleet_transactions, "query_scheduler", original_query_scheduler
    )
    monkeypatch.setattr(
        keepalive, "_spooled_job_provenance", original_spooled_provenance
    )
    submit_line = _fleet_submission_command(prepared)
    start = "1970-01-01T00:01:40Z"
    end = "1970-01-02T00:01:40Z"
    sacct = (
        f"700|{replica.scheduler_job_name}|RUNNING|{replica.partition}|"
        f"{replica.qos}|node001|"
        f"{submit_line}||{start}|{end}|1440\n"
    )
    squeue = (
        f"700|{replica.scheduler_job_name}|RUNNING|{replica.partition}|"
        f"{replica.qos}|node001|"
        f"{submit_line}|{prepared['scheduler_comment']}|{start}|{end}|"
        "1-00:00:00|(null)\n"
    )
    runner_calls = []

    def integrated_runner(argv, **_kwargs):
        runner_calls.append(argv[0])
        if argv[0] == "sacct":
            output = sacct
        elif argv[0] == "squeue":
            output = squeue
        elif argv[:3] == ["scontrol", "write", "batch_script"]:
            output = script
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, output, "")

    integrated = keepalive.reconcile_fleet_read_only(
        str(root),
        fleet,
        current_generation=1,
        scheduler_runner=integrated_runner,
        scheduler_now=200.0,
    )
    assert integrated.allocations[0].row.job_id == "700"
    assert runner_calls == ["sacct", "squeue", "scontrol"]
    monkeypatch.setattr(
        fleet_transactions,
        "query_scheduler",
        lambda **_kwargs: observed,
    )
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_args, **_kwargs: _production_spooled_provenance(
            keepalive, root, fleet, replica
        ),
    )

    foreign_comment = fleet_transactions.intent_comment(
        pool_id="schema5-canary",
        profile="canary-profile",
        replica_id="schema5-canary--r00",
        rollout_generation=1,
        intent_token="f" * 32,
        fleet_sha256="e" * 64,
    )
    foreign_terminal = replace(
        row,
        job_id="999",
        job_name="asys-schema5-canary",
        state="COMPLETED",
        node="None assigned",
        command="/sealed/canary.sbatch",
        comment=foreign_comment,
        source="sacct",
        dependency=None,
    )
    observed = replace(observed, rows=(row, foreign_terminal))
    isolated = keepalive.reconcile_fleet_read_only(
        str(root), fleet, current_generation=1
    )
    assert isolated.isolated_foreign_job_ids == ("999",)

    partial_collision = replace(
        foreign_terminal,
        comment=fleet_transactions.intent_comment(
            pool_id=fleet.fleet_id,
            profile="canary-profile",
            replica_id="schema5-canary--r00",
            rollout_generation=1,
            intent_token="f" * 32,
            fleet_sha256="e" * 64,
        ),
    )
    observed = replace(observed, rows=(row, partial_collision))
    with pytest.raises(keepalive.FleetContractError, match="partially collides"):
        keepalive.reconcile_fleet_read_only(
            str(root), fleet, current_generation=1
        )

    malformed_production_terminal = replace(
        foreign_terminal,
        job_name=replica.scheduler_job_name,
        comment="unparseable-production-history",
    )
    observed = replace(observed, rows=(row, malformed_production_terminal))
    with pytest.raises(keepalive.FleetContractError, match="malformed scheduler row"):
        keepalive.reconcile_fleet_read_only(
            str(root), fleet, current_generation=1
        )
    observed = replace(observed, rows=(row,))
    monkeypatch.setattr(
        fleet_transactions, "query_scheduler", lambda **_kwargs: observed
    )
    monkeypatch.setattr(
        keepalive,
        "_spooled_job_provenance",
        lambda *_args, **_kwargs: _production_spooled_provenance(
            keepalive, root, fleet, replica
        ),
    )

    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=22.0,
        )
        health = ledger["replicas"][replica.replica_id]["health"]
        health["cancel_state"] = "requested"
        health["cancel_requested_at"] = 22.0
        fleet_transactions.save_ledger(directory, ledger, now=22.0)
    with pytest.raises(keepalive.FleetContractError, match="hung/fenced"):
        keepalive.reconcile_fleet_read_only(
            str(root), fleet, current_generation=1
        )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=23.0,
        )
        health = ledger["replicas"][replica.replica_id]["health"]
        health["cancel_state"] = None
        health["cancel_requested_at"] = None
        fleet_transactions.save_ledger(directory, ledger, now=23.0)

    pending = replace(row, state="PENDING", node="(null)")
    monkeypatch.setattr(
        fleet_transactions,
        "query_scheduler",
        lambda **_kwargs: replace(observed, rows=(pending,)),
    )
    with pytest.raises(keepalive.FleetContractError, match="pending/non-running"):
        keepalive.reconcile_fleet_read_only(
            str(root), fleet, current_generation=1
        )


def _open_health_ledger(tmp_path):
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    lock = fleet_transactions.transaction_lock(root)
    directory = lock.__enter__()
    ledger = fleet_transactions.load_or_create_ledger(
        directory,
        pool_root=root,
        pool_id=fleet.fleet_id,
        fleet_sha256=fleet.sha256,
        rollout_generation=1,
        replica_ids=[replica.replica_id],
        now=0.0,
    )
    return fleet, replica, root, lock, directory, ledger


def _successful_process():
    return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def test_hung_replica_requires_three_dual_failures_over_ten_minutes_and_cancels_once(
    tmp_path
):
    import json
    import keepalive
    from agents_scaling.serving import fleet_transactions

    fleet, replica, root, lock, directory, ledger = _open_health_ledger(tmp_path)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        "/immutable.sbatch",
        "intent",
        qos=replica.qos,
    )
    cancelled = []

    def cancel(argv, **_kwargs):
        cancelled.append(argv)
        return _successful_process()

    try:
        for timestamp in (0.0, 300.0, 600.0):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=timestamp,
                probe=lambda *_a: (False, False),
                cancel_runner=cancel,
            )
        assert cancelled == [["scancel", "700"]]
        health = ledger["replicas"][replica.replica_id]["health"]
        assert health["consecutive_failures"] == 3
        assert health["cancel_state"] == "accepted"
        assert health["cancel_attempts"] == 1
        summary = fleet_transactions.read_health_summary(root)
        assert summary["current_generation"] == 1
        assert summary["active_hung_allocations"][0]["job_id"] == "700"
        alerts = [
            json.loads(line)
            for line in (directory / fleet_transactions.ALERTS_FILENAME)
            .read_text()
            .splitlines()
        ]
        assert len(alerts) == 1
        assert alerts[0]["job_id"] == "700"

        # Reobserving the same accepted cancellation cannot issue another mutation.
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=row,
            observer_generation=1,
            now=900.0,
            probe=lambda *_a: (False, False),
            cancel_runner=cancel,
        )
        assert cancelled == [["scancel", "700"]]

        # A replacement allocation has a fresh exact identity and health history.
        replacement = replace(row, job_id="701", node="node002")
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=replacement,
            observer_generation=1,
            now=1200.0,
            probe=lambda *_a: (True, True),
            cancel_runner=cancel,
        )
        reset = ledger["replicas"][replica.replica_id]["health"]
        assert reset["job_id"] == "701"
        assert reset["consecutive_failures"] == 0
        assert reset["cancel_state"] is None
        resolved = fleet_transactions.read_health_summary(root)
        assert resolved["active_hung_allocations"] == []
        assert resolved["historical_alert_count"] == 1
    finally:
        lock.__exit__(None, None, None)


def test_intermittent_probe_success_resets_hung_evidence(tmp_path):
    import keepalive

    _fleet, replica, _root, lock, directory, ledger = _open_health_ledger(tmp_path)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        "/immutable.sbatch",
        "intent",
        qos=replica.qos,
    )
    cancelled = []
    try:
        probes = [(False, False), (False, False), (True, False), (False, False)]
        for timestamp, result in zip((0.0, 300.0, 600.0, 900.0), probes):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=timestamp,
                probe=lambda *_a, result=result: result,
                cancel_runner=lambda argv, **_k: cancelled.append(argv)
                or _successful_process(),
            )
        health = ledger["replicas"][replica.replica_id]["health"]
        assert health["first_failure_at"] == 900.0
        assert health["consecutive_failures"] == 1
        assert cancelled == []
    finally:
        lock.__exit__(None, None, None)


def test_crash_after_cancel_intent_retries_same_exact_job_after_backoff(tmp_path):
    import keepalive

    _fleet, replica, _root, lock, directory, ledger = _open_health_ledger(tmp_path)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        "/immutable.sbatch",
        "intent",
        qos=replica.qos,
    )
    try:
        for timestamp in (0.0, 300.0):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=timestamp,
                probe=lambda *_a: (False, False),
                cancel_runner=lambda *_a, **_k: pytest.fail("not yet eligible"),
            )

        def crash_before_external_acceptance(*_args, **_kwargs):
            raise KeyboardInterrupt("crash at scancel boundary")

        with pytest.raises(KeyboardInterrupt, match="scancel boundary"):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=600.0,
                probe=lambda *_a: (False, False),
                cancel_runner=crash_before_external_acceptance,
            )
        health = ledger["replicas"][replica.replica_id]["health"]
        assert health["cancel_state"] == "retryable"
        assert health["cancel_attempts"] == 1
        accepted = []
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=row,
            observer_generation=1,
            now=900.0,
            probe=lambda *_a: (False, False),
            cancel_runner=lambda argv, **_k: accepted.append(argv)
            or _successful_process(),
        )
        assert accepted == [["scancel", "700"]]
        assert health["cancel_state"] == "accepted"
        assert health["cancel_attempts"] == 2
    finally:
        lock.__exit__(None, None, None)


def test_failed_scancel_is_retryable_and_scheduler_query_failure_never_cancels(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    _fleet, replica, _root, lock, directory, ledger = _open_health_ledger(
        tmp_path / "health"
    )
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        "/immutable.sbatch",
        "intent",
        qos=replica.qos,
    )
    attempts = []
    try:
        for timestamp in (0.0, 300.0):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=timestamp,
                probe=lambda *_a: (False, False),
                cancel_runner=lambda *_a, **_k: pytest.fail("not eligible"),
            )
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=row,
            observer_generation=1,
            now=600.0,
            probe=lambda *_a: (False, False),
            cancel_runner=lambda argv, **_k: attempts.append(argv)
            or type(
                "Proc", (), {"returncode": 1, "stdout": "", "stderr": "busy"}
            )(),
        )
        assert ledger["replicas"][replica.replica_id]["health"]["cancel_state"] == "retryable"
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=row,
            observer_generation=1,
            now=900.0,
            probe=lambda *_a: (False, False),
            cancel_runner=lambda argv, **_k: attempts.append(argv)
            or _successful_process(),
        )
        assert attempts == [["scancel", "700"], ["scancel", "700"]]
    finally:
        lock.__exit__(None, None, None)

    # Admission and health mutation are downstream of complete scheduler truth.
    root = tmp_path / "query" / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    cancellations = []
    monkeypatch.setattr(
        keepalive,
        "_query_fleet_queue",
        lambda: (_ for _ in ()).throw(
            keepalive.FleetContractError("sacct unavailable")
        ),
    )
    with pytest.raises(keepalive.FleetContractError, match="sacct unavailable"):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options={"rollout_generation": 1},
            cancellation_runner=lambda argv, **_k: cancellations.append(argv),
            submission_runner=lambda *_a, **_k: pytest.fail("must not submit"),
        )
    assert cancellations == []


def test_fleet_tick_revalidates_stable_scheduler_policy_under_transaction_lock(
    tmp_path, monkeypatch
):
    import keepalive
    from agents_scaling.serving import scheduler_safety
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    binding_sha256 = (
        scheduler_safety.transport_uncertainty_binding_sha256()
    )

    def runner_for(*, grace, mode="OFF"):
        def run(argv, *, timeout):
            del timeout
            args = list(argv)
            if args == ["scontrol", "show", "config"]:
                stdout = (
                    "KillWait                = 30 sec\n"
                    "PreemptMode             = REQUEUE\n"
                    "PreemptType             = preempt/partition_prio\n"
                )
            elif args == [
                "scontrol",
                "show",
                "partition",
                replica.partition,
                "-o",
            ]:
                stdout = (
                    f"PartitionName={replica.partition} GraceTime={grace} "
                    f"MaxTime={replica.time_limit} PreemptMode={mode} "
                    "State=UP TotalNodes=50\n"
                )
            else:
                raise AssertionError(args)
            return subprocess.CompletedProcess(args, 0, stdout, "")

        return run

    _evidence, attested = keepalive._validate_live_fleet_scheduler_policy(
        fleet,
        expected_policy_contract_id=None,
        expected_transport_binding_sha256=binding_sha256,
        runner=runner_for(grace=0),
        captured_timestamp=90.0,
    )
    monkeypatch.setattr(
        keepalive,
        "_query_fleet_queue",
        lambda: pytest.fail(
            "scheduler/ledger reconciliation must follow safety validation"
        ),
    )
    with pytest.raises(
        keepalive.FleetContractError,
        match="scheduler-safety validation failed",
    ):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options={"rollout_generation": 1},
            now=100.0,
            scheduler_safety_contract={
                "expected_policy_contract_id": attested[
                    "policy_contract_id"
                ],
                "transport_uncertainty_binding_sha256": binding_sha256,
            },
            scheduler_safety_runner=runner_for(grace=1, mode="REQUEUE"),
            protected_capacity_contract=keepalive.protected_capacity.ProtectedCapacityContract(
                path=tmp_path / "PROTECTED_CAPACITY_COMPLETE.json",
                sha256="1" * 64,
                marker_id="2" * 64,
                release_git_commit="3" * 40,
                release_tag_object="4" * 40,
                scheduler_evidence_id="5" * 64,
                scheduler_evidence_sha256="6" * 64,
                canary_id="7" * 64,
                canary_evidence_sha256="8" * 64,
                capacity_generation=1,
                base_fleet_contract_path=tmp_path / "base-fleet.json",
                base_fleet_contract_sha256=fleet.sha256,
                effective_fleet_contract_path=fleet.path,
                effective_fleet_contract_sha256=fleet.sha256,
                additive_overlay_contract_path=tmp_path / "effective-fleet.json",
                additive_overlay_contract_sha256=fleet.sha256,
                static_feasibility_certificate_path=(
                    tmp_path / "STATIC_FEASIBILITY_COMPLETE.json"
                ),
                static_feasibility_certificate_sha256="a" * 64,
                static_feasibility_certificate_id="b" * 64,
                static_feasibility_wave_passed=True,
                static_feasibility_selected_cell_count=384,
                static_feasibility_target_cell_count=384,
                static_feasibility_shortfall_cells=0,
                static_feasibility_configured_client_ceiling=384,
                static_feasibility_certified_saturation_target=384,
                fleet_contract_sha256=fleet.sha256,
                active_fleet_topology_sha256="9" * 64,
                base_active_logical_replicas=len(fleet.replicas),
                base_active_gpus=sum(
                    row.gpus_per_replica for row in fleet.replicas
                ),
                additive_reserved_logical_replicas=0,
                additive_reserved_gpus=0,
                effective_active_logical_replicas=len(fleet.replicas),
                effective_active_gpus=sum(
                    row.gpus_per_replica for row in fleet.replicas
                ),
                retained_warm_turnover_job_elements=3,
                retained_warm_turnover_gpus=4,
                attested_total_gpus=46,
                job_element_accounting={
                    "cell_job_elements": 384,
                    "active_server_job_elements": 40,
                    "warm_turnover_job_elements": 3,
                    "controller_monitor_other_held_job_elements": 21,
                    "total_non_cell_reserve_job_elements": 64,
                    "total_canary_job_elements": 448,
                },
                preempt_type="preempt/qos",
                    capacity_source=(
                        keepalive.protected_capacity.CAPACITY_SOURCE
                    ),
                        scheduler_cluster="test_cluster",
                        scheduler_account="test_account",
                        scheduler_user="test_user",
                        scheduler_max_jobs=427,
                        scheduler_max_submit_jobs=500,
                        running_scientific_jobs=427,
                        scientific_qos_contracts=(
                            {
                                "qos": replica.qos,
                                "max_wall_seconds": 86_400,
                                "max_jobs_per_user": 427,
                                "max_submit_jobs_per_user": 500,
                                "required_wall_seconds": 86_400,
                                "required_running_jobs": 427,
                                "required_submit_jobs": 448,
                            },
                        ),
                        partition_cpus=384,
                    partition_memory_mib=384 * 4096,
                    partition_gpus=46,
                server_placements=(
                    keepalive.protected_capacity.ProtectedPlacement(
                        replica.partition,
                        replica.qos,
                        {
                            "effective_active_gpus": sum(
                                row.gpus_per_replica for row in fleet.replicas
                            ),
                            "warm_headroom_gpus": 4,
                        },
                    ),
                ),
                client_placements=(),
            ),
        )


def test_current_fleet_authority_requires_exact_generations_and_bindings(
    tmp_path, monkeypatch
):
    import keepalive
    from slurm import schema5_control as control_plane
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    marker_path = tmp_path / "PROTECTED_CAPACITY_COMPLETE.json"
    protected = SimpleNamespace(
        path=marker_path,
        sha256="a" * 64,
        marker_id="b" * 64,
        capacity_generation=2,
        effective_fleet_contract_path=fleet.path,
        effective_fleet_contract_sha256=fleet.sha256,
    )
    state = {
        "capacity": {"current_generation": 2},
        "rollout_generation": 7,
    }
    fleet_binding = {
        "capacity_generation": 2,
        "path": str(fleet.path.resolve()),
        "sha256": fleet.sha256,
    }
    protected_binding = {
        "capacity_generation": 2,
        "path": str(marker_path.resolve()),
        "sha256": protected.sha256,
        "marker_id": protected.marker_id,
        "effective_fleet_contract_path": str(fleet.path.resolve()),
        "effective_fleet_contract_sha256": fleet.sha256,
    }
    monkeypatch.setattr(
        control_plane,
        "load_control",
        lambda *_a, **kwargs: (
            state
            if kwargs.get("verify_files") is True
            else pytest.fail("authority load must verify files")
        ),
    )
    monkeypatch.setattr(
        control_plane,
        "effective_fleet_contract_binding",
        lambda *_a, **_kw: fleet_binding,
    )
    monkeypatch.setattr(
        control_plane,
        "effective_protected_capacity_binding",
        lambda *_a, **_kw: protected_binding,
    )
    monkeypatch.setattr(
        control_plane,
        "load_effective_fleet_contract",
        lambda *_a, **_kw: fleet,
    )
    monkeypatch.setattr(
        control_plane,
        "load_effective_protected_capacity_contract",
        lambda *_a, **_kw: protected,
    )
    monkeypatch.setattr(
        keepalive.protected_capacity,
        "authorize_fleet",
        lambda *_a, **_kw: None,
    )
    supplied_marker = {
        "path": str(marker_path.resolve()),
        "sha256": protected.sha256,
        "marker_id": protected.marker_id,
    }

    loaded, current_marker = keepalive._load_current_fleet_authority(
        tmp_path,
        fleet=fleet,
        capacity_generation=2,
        rollout_generation=7,
        expected_protected_binding=supplied_marker,
        expected_protected_capacity_contract=protected,
    )
    assert loaded is state
    assert current_marker is protected

    state["rollout_generation"] = 8
    with pytest.raises(
        keepalive.FleetContractError,
        match="capacity/rollout generation or effective fleet",
    ):
        keepalive._load_current_fleet_authority(
            tmp_path,
            fleet=fleet,
            capacity_generation=2,
            rollout_generation=7,
            expected_protected_binding=supplied_marker,
        )
    state["rollout_generation"] = 7
    protected_binding["sha256"] = "c" * 64
    with pytest.raises(
        keepalive.FleetContractError,
        match="protected-capacity binding differs",
    ):
        keepalive._load_current_fleet_authority(
            tmp_path,
            fleet=fleet,
            capacity_generation=2,
            rollout_generation=7,
            expected_protected_binding=supplied_marker,
        )


@pytest.mark.parametrize(
    ("fail_on_authority_load", "prepared_attempts"),
    ((2, 0), (3, 1)),
)
def test_fleet_tick_reloads_current_authority_before_prepare_and_sbatch(
    tmp_path,
    monkeypatch,
    fail_on_authority_load,
    prepared_attempts,
):
    import keepalive
    from slurm import schema5_control as control_plane
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    options = _frozen_environment_render_kwargs(tmp_path / "runtime")
    options["model_contract_sha256"] = load_model_contracts().sha256
    protected = SimpleNamespace()
    authority_loads = 0

    def reload_authority(*_args, **_kwargs):
        nonlocal authority_loads
        authority_loads += 1
        if authority_loads == fail_on_authority_load:
            raise keepalive.FleetContractError(
                "synthetic current authority drift"
            )
        return {"desired_state": "paused"}, protected

    monkeypatch.setattr(
        keepalive, "_load_current_fleet_authority", reload_authority
    )
    monkeypatch.setattr(
        keepalive,
        "_validate_live_fleet_scheduler_policy",
        lambda *_a, **_kw: ({}, {}),
    )
    monkeypatch.setattr(
        keepalive.protected_capacity,
        "authorize_fleet",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        keepalive.protected_capacity,
        "verify_live_placements",
        lambda *_a, **_kw: {},
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())
    monkeypatch.setattr(
        control_plane,
        "reconcile_trusted_scientific_job_provenance",
        lambda *_a, **_kw: object(),
    )
    original_prepare = keepalive.fleet_tx.prepare_attempt
    prepare_calls = 0

    def observe_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        keepalive.fleet_tx, "prepare_attempt", observe_prepare
    )

    with pytest.raises(
        keepalive.FleetContractError,
        match="synthetic current authority drift",
    ):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options=options,
            now=100.0,
            scheduler_safety_contract={
                "expected_policy_contract_id": "1" * 64,
                "transport_uncertainty_binding_sha256": "2" * 64,
            },
            protected_capacity_contract=protected,
            control_state_dir=tmp_path / "control",
            submission_runner=lambda *_a, **_kw: pytest.fail(
                "authority drift must precede sbatch"
            ),
        )
    assert authority_loads == fail_on_authority_load
    assert prepare_calls == prepared_attempts


@pytest.mark.parametrize(
    "inventory_error",
    (
        "live protected server partition TRES/node inventory shrank",
        "live protected server partition GPU TRES drifted",
    ),
)
def test_fleet_tick_fails_closed_on_live_server_inventory_drift_before_sbatch(
    tmp_path,
    monkeypatch,
    inventory_error,
):
    import keepalive
    from slurm import schema5_control as control_plane
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=load_model_contracts())
    )
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    options = _frozen_environment_render_kwargs(tmp_path / "runtime")
    options["model_contract_sha256"] = load_model_contracts().sha256
    protected = SimpleNamespace()
    monkeypatch.setattr(
        keepalive,
        "_load_current_fleet_authority",
        lambda *_a, **_kw: ({"desired_state": "paused"}, protected),
    )
    monkeypatch.setattr(
        keepalive,
        "_validate_live_fleet_scheduler_policy",
        lambda *_a, **_kw: ({}, {}),
    )
    monkeypatch.setattr(
        keepalive.protected_capacity,
        "authorize_fleet",
        lambda *_a, **_kw: None,
    )

    def reject_inventory(*_args, **_kwargs):
        raise keepalive.protected_capacity.ProtectedCapacityError(
            inventory_error
        )

    monkeypatch.setattr(
        keepalive.protected_capacity,
        "verify_live_placements",
        reject_inventory,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())
    monkeypatch.setattr(
        control_plane,
        "reconcile_trusted_scientific_job_provenance",
        lambda *_a, **_kw: object(),
    )
    submission_calls: list[list[str]] = []

    def submit(argv, **_kwargs):
        submission_calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "999\n", "")

    with pytest.raises(
        keepalive.FleetContractError,
        match="live protected placement rejected.*before sbatch",
    ):
        keepalive.tick_fleet(
            str(root),
            fleet,
            launch_options=options,
            now=100.0,
            scheduler_safety_contract={
                "expected_policy_contract_id": "1" * 64,
                "transport_uncertainty_binding_sha256": "2" * 64,
            },
            protected_capacity_contract=protected,
            control_state_dir=tmp_path / "control",
            submission_runner=submit,
        )
    assert submission_calls == []


def test_failed_scancel_retries_are_bounded_and_end_exhausted(tmp_path):
    import keepalive

    _fleet, replica, _root, lock, directory, ledger = _open_health_ledger(tmp_path)
    row = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "RUNNING",
        replica.partition,
        "node001",
        "/immutable.sbatch",
        "intent",
        qos=replica.qos,
    )
    attempts = []

    def rejected(argv, **_kwargs):
        attempts.append(argv)
        return type(
            "Proc", (), {"returncode": 1, "stdout": "", "stderr": "busy"}
        )()

    try:
        # Establish the required three-failure/ten-minute evidence, then revisit at
        # each persisted exponential-backoff boundary (300, 600, 1200, 1800, 1800).
        for timestamp in (0.0, 300.0, 600.0, 900.0, 1500.0, 2700.0, 4500.0):
            keepalive._monitor_running_replica(
                directory=directory,
                ledger=ledger,
                replica=replica,
                row=row,
                observer_generation=1,
                now=timestamp,
                probe=lambda *_a: (False, False),
                cancel_runner=rejected,
            )
        health = ledger["replicas"][replica.replica_id]["health"]
        assert attempts == [["scancel", "700"]] * keepalive.HUNG_CANCEL_MAX_ATTEMPTS
        assert health["cancel_attempts"] == keepalive.HUNG_CANCEL_MAX_ATTEMPTS
        assert health["cancel_state"] == "exhausted"

        # No later observation is allowed to issue a sixth destructive operation.
        keepalive._monitor_running_replica(
            directory=directory,
            ledger=ledger,
            replica=replica,
            row=row,
            observer_generation=1,
            now=10_000.0,
            probe=lambda *_a: (False, False),
            cancel_runner=rejected,
        )
        assert attempts == [["scancel", "700"]] * keepalive.HUNG_CANCEL_MAX_ATTEMPTS
    finally:
        lock.__exit__(None, None, None)


def test_terminal_fenced_job_gets_one_transactional_replacement(
    tmp_path, monkeypatch
):
    import json
    import keepalive
    from agents_scaling.serving import fleet_transactions
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = _one_replica_fleet(
        load_fleet_contract(None, model_contracts=contracts)
    )
    replica = fleet.replicas[0]
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    options = _frozen_environment_render_kwargs(tmp_path / "runtime")
    options["model_contract_sha256"] = contracts.sha256
    script = render_sbatch(
        replica.model_size,
        str(root),
        replica.partition,
        replica.gpu_type,
        replica.time_limit,
        str(root / "logs"),
        replica=replica.replica_index,
        serving_profile=replica.serving_profile,
        **options,
    )
    attempt = _prepare_transactional_fleet_attempt(
        root, fleet, replica, script, now=10.0
    )
    with fleet_transactions.transaction_lock(root) as directory:
        ledger = fleet_transactions.load_or_create_ledger(
            directory,
            pool_root=root,
            pool_id=fleet.fleet_id,
            fleet_sha256=fleet.sha256,
            rollout_generation=1,
            replica_ids=[replica.replica_id],
            now=11.0,
        )
        stored = ledger["replicas"][replica.replica_id]["attempts"][0]
        stored.update(
            {
                "state": "committed",
                "submit_started_at": 10.0,
                "submission_attempts": 1,
                "job_id": "700",
                "submitted_at": 10.0,
                "committed_at": 11.0,
                "last_seen_at": 11.0,
            }
        )
        ledger["replicas"][replica.replica_id]["health"] = keepalive._new_health_record(
            "700", "node001:8000", observer_generation=1
        )
        fleet_transactions.save_ledger(directory, ledger, now=11.0)
    terminal = keepalive.FleetQueueRow(
        "700",
        replica.scheduler_job_name,
        "CANCELLED by 1",
        replica.partition,
        "None assigned",
        _fleet_submission_command(attempt),
        attempt["scheduler_comment"],
        qos=replica.qos,
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (terminal,))
    submitted = []

    submit = _accepted_fleet_runner("701", submitted)

    keepalive.tick_fleet(
        str(root),
        fleet,
        launch_options=options,
        now=20.0,
        submission_runner=submit,
    )
    assert len(submitted) == 2
    persisted = json.loads(fleet_transactions.ledger_path(
        root / fleet_transactions.STATE_DIRECTORY, 1
    ).read_text())
    attempts = persisted["replicas"][replica.replica_id]["attempts"]
    assert [(row["state"], row["job_id"]) for row in attempts] == [
        ("terminal", "700"),
        ("submitted", "701"),
    ]
    assert persisted["replicas"][replica.replica_id]["health"] is None


def test_keepalive_reserves_pending_long_profile_replica_ids(monkeypatch):
    import keepalive

    output = "\n".join(
        [
            "asys-serve-32B-long|PD|/results/servers/serve_32B-long.sbatch",
            "asys-serve-32B-long|R|/results/servers/serve_32B-long_r3.sbatch",
            "asys-serve-32B|PD|/results/servers/serve_32B_r1.sbatch",
        ]
    )
    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": output})(),
    )
    assert keepalive._replica_ids_in_flight("32B-long") == {0, 3}


def test_keepalive_counts_transitional_requeue_jobs_without_relaunch(monkeypatch):
    import keepalive

    output = "\n".join(
        [
            "asys-serve-32B-long|RQ|ou_bcs_low",
            "asys-serve-32B-long|RH|ou_bcs_low",
            "asys-serve-32B-long|CG|ou_bcs_normal",
            "asys-serve-32B|R|ou_bcs_low",
        ]
    )
    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": output})(),
    )
    assert keepalive._serve_jobs_in_flight("32B-long", "ou_bcs_low") == 2
    assert keepalive._serve_jobs_in_flight("32B-long") == 3


def test_keepalive_reserves_replica_ids_in_transitional_states(monkeypatch):
    import keepalive

    output = "\n".join(
        [
            "asys-serve-32B-long|RQ|/results/servers/serve_32B-long.sbatch",
            "asys-serve-32B-long|RH|/results/servers/serve_32B-long_r3.sbatch",
        ]
    )
    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": output})(),
    )
    assert keepalive._replica_ids_in_flight("32B-long") == {0, 3}
