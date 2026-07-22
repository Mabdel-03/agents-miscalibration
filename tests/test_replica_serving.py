"""Replica-aware serving: distinct ports per replica + keepalive spec parsing."""

import sys
from dataclasses import replace
from pathlib import Path

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
        release_id="sweep-recovery-schema5-v1",
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
    assert entry.release_id == "sweep-recovery-schema5-v1"
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


def _frozen_environment_render_kwargs(tmp_path):
    import hashlib
    import json

    release_id = "sweep-recovery-schema5-v1"
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
        payload = {
            "schema_version": 1,
            "release_id": release_id,
            "role": role,
            "prefix": str(prefix.resolve()),
            "sealed_read_only": True,
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            "runtime": runtime,
        }
        manifest = tmp_path / f"{role}-environment.json"
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        manifest.write_bytes(raw)
        records[role] = (prefix, manifest, hashlib.sha256(raw).hexdigest())
        (prefix / "bin").chmod(0o555)
        prefix.chmod(0o555)
    harness, serving = records["harness"], records["serving"]
    release_worktree = Path(__file__).resolve().parents[1]
    model_contract_path = release_worktree / "configs" / "model_contracts.v1.json"
    fleet_path = release_worktree / "configs" / "schema5_fleet.v1.json"
    fleet_sha256 = hashlib.sha256(fleet_path.read_bytes()).hexdigest()
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
    }


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
        "ou_bcs_low",
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
    assert recovered.release_id == "sweep-recovery-schema5-v1"
    assert recovered.environment_hash == environment["environment_hash"]
    assert recovered.model_contract_sha256 == contracts.sha256

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
    assert fleet.for_replica("32B-long", 1).partition == "ou_bcs_low"
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
        load_fleet_contract(target, model_contracts=load_model_contracts())


def test_production_fleet_scheduler_query_fails_closed(monkeypatch):
    import keepalive

    monkeypatch.setattr(
        keepalive.subprocess,
        "run",
        lambda *_a, **_k: type(
            "Proc", (), {"returncode": 1, "stdout": "", "stderr": "slurm down"}
        )(),
    )
    with pytest.raises(keepalive.FleetContractError, match="query failed"):
        keepalive._query_fleet_queue()


def test_production_fleet_empty_scheduler_launches_exact_contract(tmp_path, monkeypatch):
    import keepalive
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.model_contracts import load_model_contracts

    fleet = load_fleet_contract(None, model_contracts=load_model_contracts())
    root = tmp_path / "server_pools" / "schema5-v1"
    root.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: ())
    monkeypatch.setattr(
        keepalive,
        "submit_server",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(len(calls)),
    )

    keepalive.tick_fleet(str(root), fleet, launch_options={})

    assert len(calls) == 22
    for replica, (args, kwargs) in zip(fleet.replicas, calls):
        assert args[0] == replica.model_size
        assert args[2:5] == (
            replica.partition,
            replica.gpu_type,
            replica.time_limit,
        )
        assert kwargs["replica"] == replica.replica_index
        assert kwargs["serving_profile"] == replica.serving_profile


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
    comment = (
        f"asys-schema5-pool:{replica.pool_id};profile={replica.serving_profile};"
        f"replica={replica.replica_id}"
    )
    row = keepalive.FleetQueueRow(
        "123",
        replica.scheduler_job_name,
        "PENDING",
        replica.partition,
        "(null)",
        "/immutable/intent.sbatch",
        comment,
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
    monkeypatch.setattr(
        keepalive,
        "submit_server",
        lambda *args, **kwargs: launches.append((args, kwargs)) or "new",
    )

    keepalive.tick_fleet(str(root), fleet, launch_options=launch_options)
    assert len(launches) == 21

    drifted = replace(row, comment="operator-mutated-comment")
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (drifted,))
    with pytest.raises(keepalive.FleetContractError, match="comment drift"):
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
    unknown = keepalive.FleetQueueRow(
        "1", "asys-s5-serve-unknown", "PENDING", "ou_bcs_low", "(null)", "x", "x"
    )
    monkeypatch.setattr(keepalive, "_query_fleet_queue", lambda: (unknown,))
    with pytest.raises(keepalive.FleetContractError, match="unmappable"):
        keepalive.tick_fleet(str(root), fleet, launch_options={})

    first = fleet.replicas[0]
    duplicate = keepalive.FleetQueueRow(
        "2",
        first.scheduler_job_name,
        "PENDING",
        first.partition,
        "(null)",
        "x",
        "x",
    )
    monkeypatch.setattr(
        keepalive, "_query_fleet_queue", lambda: (duplicate, duplicate)
    )
    with pytest.raises(keepalive.FleetContractError, match="ambiguous duplicate"):
        keepalive.tick_fleet(str(root), fleet, launch_options={})


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
