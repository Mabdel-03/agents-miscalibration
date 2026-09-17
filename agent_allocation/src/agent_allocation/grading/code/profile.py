"""Sandbox profile for code grading and the environment digest that keys every code grade/check cache.

Config keys consumed (``grading`` block): ``memory_mib_per_sandbox``, ``pids_limit``, ``stdout_limit_bytes``,
``public_case_wall_seconds``, ``full_suite_wall_seconds``, ``cpu_quota_per_sandbox``, ``untrusted_code_network``
(must be false), ``official_evalplus_timeout_rules`` (must be true), ``offline_workers_max`` (worker pool size),
``grade_cache_by_code_hash``, ``expose_private_tests_to_generation`` (must be false).

``pids_limit``: the seccomp policy denies fork/vfork/clone/clone3, so a candidate runs as exactly one process
(stricter than 64). RLIMIT_NPROC is not used because on this kernel it counts the user's host-wide processes.
``cpu_quota_per_sandbox``: thread creation is denied, so one sandbox uses at most one CPU.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.sandbox.sandbox import CERTIFIED_PYTHON_PREFIX

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "balanced150.json"
SANDBOX_DIR = Path(__file__).resolve().parents[2] / "sandbox"
STACK_BYTES = 64 * 1024 ** 2
FILE_BYTES = 16 * 1024 ** 2


@dataclass(frozen=True)
class GradingProfile:
    memory_bytes: int
    pids_limit: int
    stdout_bytes: int
    public_case_wall_seconds: float
    suite_wall_seconds: float
    cpu_quota: int
    workers_max: int
    stack_bytes: int = STACK_BYTES
    file_bytes: int = FILE_BYTES
    python_prefix: str = CERTIFIED_PYTHON_PREFIX
    cache_by_code_hash: bool = True


def profile_from_config(cfg: Mapping[str, Any]) -> GradingProfile:
    grading = cfg["grading"]
    if grading.get("untrusted_code_network") is not False:
        raise ValueError("grading.untrusted_code_network must be false")
    if grading.get("official_evalplus_timeout_rules") is not True:
        raise ValueError("grading.official_evalplus_timeout_rules must be true")
    if grading.get("expose_private_tests_to_generation") is not False:
        raise ValueError("grading.expose_private_tests_to_generation must be false")
    if int(grading["cpu_quota_per_sandbox"]) < 1 or int(grading["pids_limit"]) < 1:
        raise ValueError("grading cpu quota and pids limit must be positive")
    return GradingProfile(memory_bytes=int(grading["memory_mib_per_sandbox"]) * 1024 ** 2,
                          pids_limit=int(grading["pids_limit"]), stdout_bytes=int(grading["stdout_limit_bytes"]),
                          public_case_wall_seconds=float(grading["public_case_wall_seconds"]),
                          suite_wall_seconds=float(grading["full_suite_wall_seconds"]),
                          cpu_quota=int(grading["cpu_quota_per_sandbox"]),
                          workers_max=int(grading["offline_workers_max"]),
                          cache_by_code_hash=bool(grading["grade_cache_by_code_hash"]))


_ACTIVE: GradingProfile | None = None


def configure(profile: GradingProfile) -> None:
    global _ACTIVE
    _ACTIVE = profile


def active_profile() -> GradingProfile:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = profile_from_config(json.loads(DEFAULT_CONFIG.read_text()))
    return _ACTIVE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _interpreter_identity(prefix: str) -> dict[str, str]:
    resolved = Path(prefix).resolve()
    binary = (resolved / "bin" / "python").resolve()
    return {"prefix": str(resolved), "binary": str(binary), "binary_sha256": _sha256_file(binary)}


@lru_cache(maxsize=1)
def vendored_source_hashes() -> dict[str, str]:
    files = ["sandbox.py", "sandbox_worker.py", "evalplus_scorer.py", "public_runner.py"]
    out = {name: _sha256_file(SANDBOX_DIR / name) for name in files if (SANDBOX_DIR / name).exists()}
    for path in sorted((SANDBOX_DIR / "upstream").rglob("*.py")):
        out[str(path.relative_to(SANDBOX_DIR))] = _sha256_file(path)
    return out


@lru_cache(maxsize=8)
def _digest(profile: GradingProfile) -> str:
    payload = {"interpreter": _interpreter_identity(profile.python_prefix), "sources": vendored_source_hashes(),
               "limits": {k: v for k, v in asdict(profile).items() if k not in {"workers_max", "cache_by_code_hash"}}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment_digest(profile: GradingProfile | None = None) -> str:
    return _digest(profile or active_profile())
