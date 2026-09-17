"""The vendored OS sandbox runs a trivial program and keeps its isolation properties."""
from __future__ import annotations

import pytest

from agent_allocation.sandbox.sandbox import PROCESS_EXITED, WALL_TIMEOUT, Sandbox, SandboxError, SandboxLimits

pytestmark = pytest.mark.sandbox

LIMITS = SandboxLimits(cpu_seconds=5, wall_seconds=5, memory_bytes=512 * 1024 ** 2)


def test_trivial_program_runs():
    with Sandbox({}, limits=LIMITS) as sandbox:
        result = sandbox.execute("print(6 * 7)")
    assert result["ok"] is True
    assert result["stdout"] == "42\n"


def test_sockets_are_denied():
    with Sandbox({}, limits=LIMITS) as sandbox:
        result = sandbox.execute("import socket\ntry:\n    socket.socket()\n    print('open')\n"
                                 "except OSError as exc:\n    print('denied', exc.errno)\n")
    assert result["stdout"].strip() == "denied 1"


def test_wall_timeout_is_typed():
    with pytest.raises(SandboxError) as info:
        with Sandbox({}, limits=SandboxLimits(cpu_seconds=30, wall_seconds=1.5)) as sandbox:
            sandbox.execute("import time\ntime.sleep(20)")
    assert info.value.kind == WALL_TIMEOUT and info.value.phase == "candidate"


def test_process_exit_is_typed():
    with pytest.raises(SandboxError) as info:
        with Sandbox({}, limits=LIMITS) as sandbox:
            sandbox.execute("import os\nos._exit(3)")
    assert info.value.kind == PROCESS_EXITED and info.value.returncode == 3
