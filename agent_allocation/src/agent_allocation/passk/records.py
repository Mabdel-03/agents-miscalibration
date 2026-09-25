"""§15.2 run records: ``records/run_manifest.json`` and ``records/{candidates,pools,selections}.parquet``.

Candidates are ``analysis_inputs/draws.parquet`` as extracted (fenced labels are already absent there); pools and
selections are the evaluated pools and their readouts. The manifest records the study identity, frozen parameters,
code commit and the sha256 of every record file.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

__all__ = ["write_records"]

STUDY = {"study_id": "openrouter_passk_cont_v1", "preset": "passk_cont_v1", "model": "qwen/qwen3.8-27b",
         "permaslug": "qwen/qwen3.8-27b-20260814", "providers": ["mancer/fp8", "parasail/fp8"], "A": 1024,
         "prompt_version": "prompts-v1-e998b556294c", "passk_seed": 20260925, "cfg_seed": 20260917}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_records(run_root: Path, draws: pd.DataFrame, pools: pd.DataFrame, selections: pd.DataFrame) -> dict:
    out = Path(run_root) / "records"
    out.mkdir(parents=True, exist_ok=True)
    files = {"candidates.parquet": draws, "pools.parquet": pools, "selections.parquet": selections}
    hashes = {}
    for name, frame in files.items():
        tmp = out / f".{name}.tmp{os.getpid()}"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, out / name)
        hashes[name] = _sha(out / name)
    manifest = {**STUDY, "version": "passk-records-v1", "code_commit": _commit(),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_candidates": int(len(draws)), "n_pools": int(len(pools)), "n_selections": int(len(selections)),
                "partitions": sorted(map(str, draws["partition"].unique())) if len(draws) else [],
                "configs": sorted(map(str, pools["config_id"].unique())) if len(pools) else [],
                "sha256": hashes}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
