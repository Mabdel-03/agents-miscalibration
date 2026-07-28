from __future__ import annotations

from scripts import schema5_watchdog_forced_command as forced


def test_forced_command_child_environment_is_complete_and_minimal(
    monkeypatch,
) -> None:
    hostile = {
        "PATH": "/tmp/attacker-bin",
        "BASH_ENV": "/tmp/attacker-env",
        "LD_PRELOAD": "/tmp/attacker.so",
        "PYTHONPATH": "/tmp/attacker-python",
        "GIT_DIR": "/tmp/attacker-git",
        "SBATCH_PARTITION": "attacker",
        "SLURM_CONF": "/tmp/attacker-slurm.conf",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)

    assert forced._child_process_environment() == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
