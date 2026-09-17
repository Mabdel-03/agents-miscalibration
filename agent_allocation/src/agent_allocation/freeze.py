"""``allocation freeze`` (lead-owned): the explicit state transition from pilot to main collection.

Requires Gate 2 ``advance``. Writes the write-once prompt, environment and config locks, the full job and policy
manifests for main / high / replication, and ``state/authorization.json``, which ``run`` refuses to start without.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.config import config_hash, credit_ceiling, grids_from_config, load_config, validate_config
from agent_allocation.contracts import canonical_hash
from agent_allocation.manifest import ManifestRefused, load_waves, write_once_json
from agent_allocation.paths import PROJECT_ROOT, RunPaths


class FreezeRefused(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def environment_payload() -> dict[str, Any]:
    packages = {}
    for name in ("httpx", "math-verify", "latex2sympy2_extended", "numpy", "scipy", "pandas", "pyarrow", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sandbox_digest = None
    try:
        private = importlib.import_module("agent_allocation.grading.code.private")
        digest_fn = getattr(private, "environment_digest", None)
        sandbox_digest = digest_fn() if callable(digest_fn) else None
    except ImportError:
        sandbox_digest = None
    vendored = PROJECT_ROOT / "src" / "agent_allocation" / "_vendor" / "PROVENANCE.json"
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages,
            "uv_lock_sha256": file_sha256(PROJECT_ROOT / "uv.lock"),
            "vendored_provenance_sha256": file_sha256(vendored) if vendored.exists() else None,
            "sandbox_environment_digest": sandbox_digest}


def freeze(cfg: Mapping[str, Any], config_path: str, paths: RunPaths) -> dict[str, Any]:
    from agent_allocation.live import (
        graph_for, load_amendments, load_tasks, read_endpoint_locks, write_jobs,
    )
    from agent_allocation.prompts.render import prompt_lock_record
    problems = validate_config(cfg)
    if problems:
        raise FreezeRefused(f"config invalid: {problems}")
    gate2 = paths.gate_result("gate2", "pilot")
    if not gate2.exists() or json.loads(gate2.read_text()).get("decision") != "advance":
        raise FreezeRefused("Gate 2 has not advanced")
    locks = read_endpoint_locks(paths)
    for model in (cfg["primary_model"], cfg["replication_model"]):
        if model not in locks:
            raise FreezeRefused(f"no endpoint lock for {model}")
    if not paths.dataset_lock.exists():
        raise FreezeRefused("no dataset lock")
    forecast = json.loads((paths.state_dir / "pilot_forecast.json").read_text())
    amendments = load_amendments(paths)
    tasks = load_tasks(paths)
    grids = grids_from_config(cfg)
    waves = load_waves(paths)
    primary, replication = locks[cfg["primary_model"]], locks[cfg["replication_model"]]
    main_a, high_a = int(cfg["reasoning"]["main_a"]), int(cfg["reasoning"]["high_a"])

    partitions: dict[str, list] = {"main": [], "high": [], "replication": []}
    for wave in waves:
        for item in wave.item_ids:
            task = tasks[item]
            grid = grids["main_extension"] if "extension" in wave.panels else grids["main"]
            partitions["main"].append(graph_for(grid, cfg=cfg, partition="main", task=task, lock=primary,
                                                base_budget=main_a, amendments=amendments))
            if "high" in wave.panels:
                partitions["high"].append(graph_for(grids["high"], cfg=cfg, partition="high", task=task, lock=primary,
                                                    base_budget=high_a, amendments=amendments))
            if "replication" in wave.panels:
                partitions["replication"].append(graph_for(grids["replication"], cfg=cfg, partition="replication",
                                                           task=task, lock=replication, base_budget=main_a,
                                                           amendments=amendments))
    job_hashes = {name: write_jobs(paths, name, graphs) for name, graphs in partitions.items()}
    counts = {name: sum(len(g.nodes) for g in graphs) for name, graphs in partitions.items()}

    try:
        prompt_digest = write_once_json(paths.prompt_lock, prompt_lock_record())
        env_digest = write_once_json(paths.environment_lock, environment_payload())
        config_digest = write_once_json(paths.config_lock, {
            "config": dict(cfg), "config_path": config_path, "config_sha256": config_hash(cfg),
            "amendments": amendments,
            "amendments_sha256": file_sha256(paths.amendments) if paths.amendments.exists() else None,
            "endpoint_lock_sha256": file_sha256(paths.endpoint_lock),
            "dataset_lock_sha256": file_sha256(paths.dataset_lock), "gate2_sha256": file_sha256(gate2),
            "job_manifest_sha256": job_hashes, "generation_counts": counts, "pilot_forecast": forecast,
        })
    except ManifestRefused as exc:
        raise FreezeRefused(str(exc)) from exc
    authorization = {
        "stage": "main", "study_ceiling_usd": str(credit_ceiling(cfg)),
        "soft_admission_ceiling_usd": str(cfg["budget"]["soft_admission_ceiling_usd"]),
        "config_lock_sha256": config_digest, "prompt_lock_sha256": prompt_digest,
        "environment_lock_sha256": env_digest, "admission": forecast["admission"],
        "authorized_by": "auto-advance on frozen Gate 2 criteria (user decision 2026-09-17)",
        "authorization_sha256": canonical_hash([config_digest, prompt_digest, env_digest]),
    }
    paths.authorization.write_text(json.dumps(authorization, indent=1, sort_keys=True))
    return {"generation_counts": counts, "total": sum(counts.values()), "authorization": authorization}


def cmd_freeze(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation freeze")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    print(json.dumps(freeze(cfg, args.config, paths), indent=1, sort_keys=True, default=str))
    return 0
