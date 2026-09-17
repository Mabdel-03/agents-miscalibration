"""The contract file is lead-owned. If this fails in an agent worktree, you edited contracts.py: revert and file a
request in docs/requests/. The lead updates this hash together with any contract change."""
from hashlib import sha256
from pathlib import Path

CONTRACTS_SHA256 = "1698c45934e20a635f5265a4f16faec15d89795c9291843c9b43376e2ef5a340"


def test_contracts_file_is_the_frozen_version():
    path = Path(__file__).resolve().parents[2] / "src" / "agent_allocation" / "contracts.py"
    assert sha256(path.read_bytes()).hexdigest() == CONTRACTS_SHA256
