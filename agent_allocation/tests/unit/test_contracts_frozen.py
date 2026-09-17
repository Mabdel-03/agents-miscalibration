"""The contract file is lead-owned. If this fails in an agent worktree, you edited contracts.py: revert and file a
request in docs/requests/. The lead updates this hash together with any contract change."""
from hashlib import sha256
from pathlib import Path

CONTRACTS_SHA256 = "dbbdccabc01dce1f52fc425e624b43691b96758a394a908a5ad8ccf9d3019609"


def test_contracts_file_is_the_frozen_version():
    path = Path(__file__).resolve().parents[2] / "src" / "agent_allocation" / "contracts.py"
    assert sha256(path.read_bytes()).hexdigest() == CONTRACTS_SHA256
