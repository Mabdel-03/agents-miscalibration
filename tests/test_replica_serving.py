"""Replica-aware serving: distinct ports per replica + keepalive spec parsing."""

import sys
from pathlib import Path

import pytest

from agents_scaling.serving.launch_server import _port_for

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


def test_keepalive_spec_rejects_malformed():
    from keepalive import parse_spec

    with pytest.raises(ValueError):
        parse_spec("8B:2:ou_bcs_low", default_gpu="a100")  # missing time field


# --- Registry garbage-collection (Change A) ---------------------------------------

def _register_fake(run_root, size, host, port):
    """Drop a registry file by hand so we don't need a real vLLM server."""
    import json
    from agents_scaling.serving.registry import _size_dir, ServerEntry
    d = _size_dir(run_root, size)
    e = ServerEntry(model_size=size, hf_id=f"Qwen/Qwen3-{size}", host=host, port=port)
    (d / f"{host}_{port}.json").write_text(json.dumps({
        "model_size": e.model_size, "hf_id": e.hf_id, "host": e.host, "port": e.port,
        "slurm_job_id": None, "started_at": 0.0,
    }))
    return e


def test_list_live_servers_filters_and_prunes_dead(tmp_path, monkeypatch):
    """list_live_servers returns only alive entries and removes dead files on the spot."""
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)
    _register_fake(tmp_path, "8B", "nodeC", 8003)

    alive_set = {("nodeA", 8001), ("nodeC", 8003)}
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda host, port, timeout=3.0: (host, port) in alive_set,
    )

    live = registry.list_live_servers(tmp_path, "8B")
    live_hp = {(e.host, e.port) for e in live}
    assert live_hp == alive_set

    # dead entry's file was unlinked
    assert not (registry._size_dir(tmp_path, "8B") / "nodeB_8002.json").exists()
    # alive entries' files survived
    assert (registry._size_dir(tmp_path, "8B") / "nodeA_8001.json").exists()
    assert (registry._size_dir(tmp_path, "8B") / "nodeC_8003.json").exists()


def test_lookup_server_live_only_skips_dead(tmp_path, monkeypatch):
    """lookup_server(live_only=True) round-robins only across live endpoints; the
    default path (live_only=False) preserves the raw filesystem read used by keepalive."""
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)

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
    # raw set should include nodeB (proves the default path is unfiltered)
    # NOTE: nodeB's file may have been pruned by the live_only call above (side effect);
    # re-register before asserting raw sees it.
    _register_fake(tmp_path, "8B", "nodeB", 8002)
    raw_hp = set()
    for s in range(4):
        e = registry.lookup_server(tmp_path, "8B", shard=s)
        raw_hp.add((e.host, e.port))
    assert ("nodeB", 8002) in raw_hp


def test_list_live_servers_empty_when_all_dead(tmp_path, monkeypatch):
    from agents_scaling.serving import registry

    _register_fake(tmp_path, "8B", "nodeA", 8001)
    _register_fake(tmp_path, "8B", "nodeB", 8002)
    monkeypatch.setattr(
        "agents_scaling.serving.healthcheck.is_alive",
        lambda host, port, timeout=3.0: False,
    )
    assert registry.list_live_servers(tmp_path, "8B") == []
    # both pruned
    sd = registry._size_dir(tmp_path, "8B")
    assert list(sd.glob("*.json")) == []
