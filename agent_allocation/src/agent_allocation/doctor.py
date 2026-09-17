"""``allocation doctor`` (lead-owned): environment preflight. Spends nothing.

``--online`` additionally calls the free ``GET /api/v1/key`` and ``GET /api/v1/credits`` endpoints and records a
snapshot under ``state/provider_snapshots.jsonl`` (never the key or its label) for later cost reconciliation.
"""
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.config import credit_ceiling, load_config, validate_config
from agent_allocation.paths import RunPaths

PINNED_SNAPSHOTS = {
    "math500": ("datasets--HuggingFaceH4--MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "mmlu_pro": ("datasets--TIGER-Lab--MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "data/test-00000-of-00001.parquet"),
}
OPENROUTER = "https://openrouter.ai/api/v1"


def filesystem_type(path: Path) -> str:
    target = str(path.resolve())
    best, fstype = "", "unknown"
    with open("/proc/mounts") as mounts:
        for line in mounts:
            parts = line.split()
            if len(parts) >= 3 and target.startswith(parts[1]) and len(parts[1]) > len(best):
                best, fstype = parts[1], parts[2]
    return fstype


def home_realpath() -> str:
    return str(Path.home().resolve())


def provider_snapshot(api_key: str) -> dict[str, Any]:
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=20.0) as client:
        key = client.get(f"{OPENROUTER}/key", headers=headers)
        credits = client.get(f"{OPENROUTER}/credits", headers=headers)
    key.raise_for_status()
    credits.raise_for_status()
    key_data = {k: v for k, v in key.json().get("data", {}).items() if k != "label"}
    credit_data = credits.json().get("data", {})
    balance = Decimal(str(credit_data.get("total_credits", 0))) - Decimal(str(credit_data.get("total_usage", 0)))
    return {"recorded_at": datetime.now(timezone.utc).isoformat(), "key": key_data,
            "credits": credit_data, "account_balance_usd": str(balance)}


def write_provider_usage(paths: RunPaths, stage: str, snap: Mapping[str, Any]) -> None:
    """``state/provider_usage.jsonl``: one line per endpoint per snapshot (agent 07's COST_RECONCILIATION schema)."""
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.provider_usage, "a") as handle:
        for endpoint in ("key", "credits"):
            handle.write(json.dumps({"fetched_at": snap["recorded_at"], "stage": stage, "endpoint": endpoint,
                                     "data": snap[endpoint]}, sort_keys=True) + "\n")


def run_checks(cfg: dict[str, Any], paths: RunPaths, *, online: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any, *, blocking: bool = True) -> None:
        checks.append({"check": name, "ok": bool(ok), "blocking": blocking, "detail": detail})

    problems = validate_config(cfg)
    add("config_valid", not problems, problems)

    from agent_allocation.apikey import KeyUnavailable, key_file_path, load_api_key
    try:
        api_key = load_api_key()
        add("api_key_loadable", True, f"from {'env' if os.environ.get('OPENROUTER_API_KEY') else key_file_path()}")
    except KeyUnavailable as exc:
        api_key = None
        add("api_key_loadable", False, str(exc))

    for tool in ("bwrap", "prlimit"):
        add(f"tool_{tool}", shutil.which(tool) is not None, shutil.which(tool))
    add("libseccomp", ctypes.util.find_library("seccomp") is not None, ctypes.util.find_library("seccomp"))
    userns = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "true"], capture_output=True)
    add("user_namespaces", userns.returncode == 0, userns.stderr.decode()[-200:])

    try:
        paths.local_state_dir.mkdir(parents=True, exist_ok=True)
        probe = paths.local_state_dir / ".doctor_probe"
        probe.write_text("ok")
        probe.unlink()
        fstype = filesystem_type(paths.local_state_dir)
        add("local_state_not_network_fs", not fstype.startswith("nfs"), {"dir": str(paths.local_state_dir), "fs": fstype})
    except OSError as exc:
        add("local_state_not_network_fs", False, str(exc))

    home = home_realpath()
    for var in ("UV_CACHE_DIR", "MPLCONFIGDIR", "HF_HOME"):
        value = os.environ.get(var)
        resolved = str(Path(value).resolve()) if value else None
        add(f"cache_{var}_off_home", bool(resolved) and not resolved.startswith(home), resolved)

    hf_home = Path(os.environ.get("HF_HOME", ""))
    for bench, (repo, sha, rel) in PINNED_SNAPSHOTS.items():
        target = hf_home / "hub" / repo / "snapshots" / sha / rel
        add(f"hf_snapshot_{bench}", target.exists(), str(target))

    add("python_in_project_venv", sys.prefix.startswith(str(Path(__file__).resolve().parents[2])), sys.prefix,
        blocking=False)
    affinity = len(os.sched_getaffinity(0))
    add("cpu_affinity", affinity >= 2, {"affinity": affinity, "os_cpu_count": os.cpu_count()}, blocking=False)
    usage = shutil.disk_usage(paths.run_dir if paths.run_dir.exists() else paths.run_dir.parent)
    add("run_dir_free_space_gb", usage.free > 50 * 1024**3, round(usage.free / 1024**3, 1))

    if online:
        if api_key is None:
            add("provider_snapshot", False, "no key")
        else:
            try:
                snap = provider_snapshot(api_key)
                ceiling = credit_ceiling(cfg)
                limit = snap["key"].get("limit")
                add("key_not_free_tier", snap["key"].get("is_free_tier") is False, snap["key"].get("is_free_tier"))
                add("key_limit_at_or_below_study_ceiling", limit is not None and Decimal(str(limit)) <= ceiling,
                    {"key_limit": limit, "study_ceiling": str(ceiling)}, blocking=False)
                add("account_balance_covers_ceiling", Decimal(snap["account_balance_usd"]) >= ceiling,
                    {"account_balance_usd": snap["account_balance_usd"], "study_ceiling": str(ceiling)},
                    blocking=False)
                write_provider_usage(paths, "doctor", snap)
            except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
                add("provider_snapshot", False, type(exc).__name__)
    return checks


def cmd_doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation doctor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--online", action="store_true", help="call the free /key and /credits endpoints")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    checks = run_checks(cfg, paths, online=args.online)
    print(json.dumps(checks, indent=1, sort_keys=True))
    failed = [c["check"] for c in checks if c["blocking"] and not c["ok"]]
    warnings = [c["check"] for c in checks if not c["blocking"] and not c["ok"]]
    print(json.dumps({"blocking_failures": failed, "warnings": warnings}))
    return 1 if failed else 0
