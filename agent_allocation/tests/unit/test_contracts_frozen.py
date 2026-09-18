"""The contract file is lead-owned. If this fails in an agent worktree, you edited contracts.py: revert and file a
request in docs/requests/. The lead updates this hash together with any contract change."""
from hashlib import sha256
from pathlib import Path

CONTRACTS_SHA256 = "4d539c1e6517770ae68ce047c6b9463b3d6dfefaa7d376b890a411a44fc8d4b1"


def test_contracts_file_is_the_frozen_version():
    path = Path(__file__).resolve().parents[2] / "src" / "agent_allocation" / "contracts.py"
    assert sha256(path.read_bytes()).hexdigest() == CONTRACTS_SHA256
