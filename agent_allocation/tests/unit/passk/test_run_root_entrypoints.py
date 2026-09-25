"""File-based entry points: e2/e5 fit_and_predict(run_root) pre-seal, then score(run_root) after the releases.

A synthetic run root with ``analysis_inputs/draws.parquet`` (contract §5.3 columns the models read) and
``artifacts/manifests/splits.json``. Pre-seal the draws hold no bank rows and no t200_diag labels; post-release they do.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent_allocation.passk.models import e2, e5
from tests.unit.passk.test_e5 import _row, synth


def _v48_rows(items, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for item in items:
        b = item.split("/")[0]
        base = rng.normal(0, 1)
        for a, m in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
            p = 1 / (1 + np.exp(-(base + 0.5 * np.log2(a))))
            rows += [_row("v48_bank", item, b, a, r, rng.uniform() < p, rng) for r in range(m)]
    return pd.DataFrame(rows)


def _write_root(root, draws, panels):
    (root / "analysis_inputs").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)
    d = draws.copy()
    d["correct"] = d["correct"].astype(object)
    d.to_parquet(root / "analysis_inputs" / "draws.parquet", index=False)
    full = {"L96": panels["L96"], "C48": panels["C48"], "V48": panels["C48"], "T200": panels["T200"], "D32": [],
            "X80": []}
    (root / "artifacts" / "manifests" / "splits.json").write_text(json.dumps({"panels": full}))


def _lenient(monkeypatch):
    """This synthetic root has no classes.parquet/readouts.parquet (exact-key classes by construction); the
    production strictness (no fallback, no seeded/tie-averaged mix) is tested in test_production_fit_is_strict."""
    for mod in (e2, e5):
        monkeypatch.setattr(mod, "_require_no_fallback", lambda cls: None)
        monkeypatch.setattr(mod, "_require_seeded", lambda observed: None)


def test_production_fit_is_strict(tmp_path):
    labelled, panels = synth(n_dev=4, n_t200=2, t200_labels=True)
    pre = labelled[labelled.partition != "t200_bank"].astype({"correct": object, "grade_state": object})
    pre.loc[pre.partition == "t200_diag", ["correct", "grade_state"]] = None
    root = tmp_path / "run"
    _write_root(root, pre, panels)
    with pytest.raises(RuntimeError, match="exact vote keys"):          # no classes.parquet -> refused
        e5.fit_and_predict(root, n_sim=50)
    with pytest.raises(RuntimeError, match="exact vote keys"):
        e2.fit_and_predict(root, S=20, inner=4)
    assert not (root / "sealed" / "t200" / "predictions.parquet").exists()
    (root / "analysis_inputs" / "manifest.json").write_text(json.dumps({"classes_pending_pools": 3}))
    for fit in (lambda: e5.fit_and_predict(root, n_sim=50), lambda: e2.fit_and_predict(root, S=20, inner=4)):
        with pytest.raises(RuntimeError, match="classes_pending_pools=3"):
            fit()


def test_run_root_fit_then_score(tmp_path, monkeypatch):
    _lenient(monkeypatch)
    labelled, panels = synth(n_dev=4, n_t200=2, t200_labels=True)
    v48 = _v48_rows(panels["C48"])
    pre = labelled[labelled.partition != "t200_bank"].astype({"correct": object, "grade_state": object})
    pre.loc[pre.partition == "t200_diag", ["correct", "grade_state"]] = None
    root = tmp_path / "run"
    _write_root(root, pre, panels)
    e2.fit_and_predict(root, S=60, inner=8)
    e5.fit_and_predict(root, n_sim=100)
    assert (root / "sealed" / "v48" / "predictions.parquet").exists()
    assert (root / "sealed" / "t200" / "predictions.parquet").exists()
    with pytest.raises(FileExistsError):
        e5.fit_and_predict(root)
    # release: labels for the banks and t200_diag appear in analysis_inputs
    post = pd.concat([labelled, v48], ignore_index=True)
    _write_root(root, post, panels)
    s2 = e2.score(root)
    s5 = e5.score(root)
    assert (root / "scores" / "e2" / "overall.parquet").exists()
    assert (root / "scores" / "e5" / "summary.parquet").exists()
    assert set(s2["overall"].model) == set(e2.MODELS)
    assert s5["summary"].primary_brier_mixture.notna().all()
    # re-fitting after the release is refused (held-out labels present)
    with pytest.raises((RuntimeError, FileExistsError)):
        e5.fit_and_predict(root, out_dir=tmp_path / "again")
