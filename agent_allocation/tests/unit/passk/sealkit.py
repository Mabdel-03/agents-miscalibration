"""Test helpers: fabricate the PREREG seal contents and the Gate P sandbox run records a real
``passk-analyze --stage e2-fit|e5-fit --sandbox`` would write (bwrap itself is exercised in test_seal.py when the
node permits user namespaces)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_allocation.passk import analyze
from agent_allocation.passk.models.predictors import PREDICTORS


def record_sandboxed_run(run_root: Path, stage: str, outputs: list[str], *, commit: str | None = None,
                         rc: int = 0, argv: list[str] | None = None) -> dict:
    run_root = Path(run_root)
    head, _ = analyze._code_commit()
    cmd = argv if argv is not None else analyze.sandbox_argv(run_root, ["--stage", stage])
    rec = {"version": analyze.SANDBOX_RECORD_VERSION, "stage": stage, "sandbox": "bwrap", "argv": cmd,
           "argv_sha256": analyze.argv_sha256(cmd), "code_commit": commit or head, "worktree_dirty": False,
           "rc": rc, "host": "test", "outputs": {rel: analyze._sha256_file(run_root / rel) for rel in outputs}}
    target = analyze.run_record_path(run_root, stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rec))
    return rec


def write_prereg(run_root: Path, *, v48_predictions: pd.DataFrame | None = None,
                 t200_predictions: pd.DataFrame | None = None, predictors=PREDICTORS,
                 skip: tuple[str, ...] = ()) -> None:
    """The complete sealed/ set from sandboxed e2-fit/e5-fit runs (+ e1_selection.json unless present)."""
    run_root = Path(run_root)
    sealed = run_root / "sealed"
    (sealed / "v48" / "models").mkdir(parents=True, exist_ok=True)
    (sealed / "t200" / "models").mkdir(parents=True, exist_ok=True)
    (run_root / "analysis_inputs").mkdir(parents=True, exist_ok=True)
    if not (sealed / "e1_selection.json").exists():
        (sealed / "e1_selection.json").write_text('{"selected": "4x4"}')
    if v48_predictions is None:
        v48_predictions = pd.DataFrame({"item_id": ["c0"], "benchmark": ["aime"], "config_id": ["1x16"],
                                        "model": ["beta"], "pred_coverage": [0.5], "pred_plurality": [0.5]})
    if t200_predictions is None:
        t200_predictions = pd.DataFrame({"item_id": "t0", "benchmark": "aime", "config_id": "1x16",
                                         "predictor": list(predictors), "p_delivered": 0.5, "p_coverage": 0.5})
    e2_files = {"sealed/v48/predictions.parquet": v48_predictions,
                "sealed/v48/crossover.parquet": pd.DataFrame({"item_id": ["c0"], "x": [1.0]}),
                "sealed/v48/manifest.json": {"model": "e2"}, "sealed/v48/seeds.json": {"seed": 1},
                "sealed/v48/models/ridge.json": {"w": [0]}}
    e5_files = {"sealed/t200/predictions.parquet": t200_predictions,
                "sealed/t200/features.parquet": pd.DataFrame({"item_id": ["t0"], "f": [0.1]}),
                "sealed/t200/manifest.json": {"model": "e5"}, "sealed/t200/seeds.json": {"seed": 1},
                "sealed/t200/models/P1.json": {"w": [0]}}
    for stage, files in (("e2-fit", e2_files), ("e5-fit", e5_files)):
        written = []
        for rel, payload in files.items():
            if rel in skip:
                continue
            path = run_root / rel
            if isinstance(payload, pd.DataFrame):
                payload.to_parquet(path, index=False)
            else:
                path.write_text(json.dumps(payload))
            written.append(rel)
        record_sandboxed_run(run_root, stage, written)
