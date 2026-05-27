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
