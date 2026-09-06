"""N3 analysis CLI tests (CPU): the label firewall (refuse without the evaluation join
table; no protected-path access anywhere in the neural package) and an end-to-end run of
the four stages on a synthetic run root."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents_scaling.study.neural import analysis as A
from agents_scaling.study.neural import calibration as C
from agents_scaling.study.neural import readout as R
from tests.study_neural.n3_support import write_world

NEURAL_PKG = Path(A.__file__).resolve().parent
_PATH_RE = re.compile(r"(^|/)protected(/|$)")
COMMON = ["--blocks", "8,17", "--ranks", "4,8", "--penalties", "1,100", "--text", "tfidf", "--bootstrap", "300", "--panel", "20", "--subsamples", "10"]


def _strings(path: Path) -> list[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    return [s for s in out if s not in docstrings]


def test_neural_package_never_names_a_protected_path_or_reader():
    offenders = []
    for path in sorted(NEURAL_PKG.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "agents_scaling.study.data.protected" in text or "protected_dir(" in text:
            offenders.append(path.name)
        if any(_PATH_RE.search(s) for s in _strings(path)):
            offenders.append(path.name)
    assert offenders == []


def test_stages_refuse_labels_before_aggregate(tmp_path):
    run_root = tmp_path / "study_x"
    write_world(run_root, n_dev=4, n_main=4, seed=1, labels=False, native=False)
    assert not (run_root / "tables" / "selections.parquet").exists()
    for stage in ("G-dev", "G-confirm", "C"):
        assert A.main(["--run-id", "study_x", "--results-root", str(tmp_path), "--stage", stage, *COMMON]) == 4
    with pytest.raises(R.ReadoutError):
        R.load_labels(run_root)
    assert A.main(["--run-id", "missing", "--results-root", str(tmp_path), "--stage", "geometry"]) == 2
    # geometry runs without labels (association skipped) when native rows exist
    write_world(tmp_path / "study_y", n_dev=3, n_main=2, seed=2, labels=False, native=True)
    assert A.main(["--run-id", "study_y", "--results-root", str(tmp_path), "--stage", "geometry", *COMMON]) == 0
    summary = json.loads((tmp_path / "study_y" / "neural" / "geometry_summary.json").read_text())
    assert summary["outcomes_joined"] is False and summary["association"] == {}


def test_end_to_end_geometry_G_and_C(tmp_path):
    run_root = tmp_path / "study_v4"
    world = write_world(run_root, n_dev=12, n_main=20, seed=3, neural_signal=4.0)
    args = ["--run-id", "study_v4", "--results-root", str(tmp_path)]
    # --- geometry -------------------------------------------------------------------
    assert A.main([*args, "--stage", "geometry", *COMMON, "--anchors", "NATIVE_PREFILL,GENERATED_512"]) == 0
    geo = pd.read_parquet(run_root / "neural" / "tables" / "geometry.parquet")
    ind = geo[(geo["anchor"] == "NATIVE_PREFILL") & (geo["block"] == 8)]
    assert len(ind) == len(world["tasks"]) and set(ind["status"]) == {"exact"}
    assert (ind["s"] == 5).all() and (ind["rank_ceiling"] == 4).all() and (ind["n_subsamples"] == 1).all()
    assert ind["participation_ratio"].between(1, 4).all() and ind["standardized"].all()
    missing = geo[geo["anchor"] == "GENERATED_512"]
    assert set(missing["status"]) == {"no_states"} and (missing["n_missing"] == 5).all()
    gsum = json.loads((run_root / "neural" / "geometry_summary.json").read_text())
    assert gsum["outcomes_joined"] and "IND_VOTE|INDEPENDENT_SOLVER|ROOT|8|NATIVE_PREFILL" in gsum["by_config"]
    assert gsum["by_config"]["IND_VOTE|INDEPENDENT_SOLVER|ROOT|8|NATIVE_PREFILL"]["rank_ceiling"] == 4
    assert (run_root / "neural" / "display" / "display_pca.32B.b8.NATIVE_PREFILL.json").is_file()
    assert "outcome_selected_correct" in ind.columns
    # --- G-dev -----------------------------------------------------------------------
    assert A.main([*args, "--stage", "G-dev", *COMMON]) == 0
    dev = json.loads((run_root / "neural" / "G_dev_summary.json").read_text())
    assert dev["assembly"]["rows"] == 3 * 2 * (12 + 20) and dev["assembly"]["rows_in_split"] == 3 * 2 * 12
    assert dev["outer"]["n_items"] == 24 and len(dev["outer"]["folds"]) == 5
    assert dev["frozen"]["selected"]["augmented"]["block"] == 8 and dev["frozen"]["text_kind"] == "tfidf_svd"
    frozen = json.loads((run_root / "neural" / "G_frozen.json").read_text())
    assert frozen["kind"] == "G_FROZEN" and frozen["pca"]["components_hash"] and frozen["sha256"] and frozen["arrays_sha256"]
    assert (run_root / "neural" / "G_frozen.npz").is_file()
    assert set(frozen["feature_lists"]) == set(R.VARIANTS)
    losses = pd.read_parquet(run_root / "neural" / "tables" / "G_dev_losses.parquet")
    assert len(losses) == 72 and np.isfinite(losses["brier_augmented"]).all()
    # --- G-confirm ---------------------------------------------------------------------
    assert A.main([*args, "--stage", "G-confirm", *COMMON]) == 0
    conf = json.loads((run_root / "neural" / "G_confirm_summary.json").read_text())
    assert conf["assembly"]["rows_in_split"] == 3 * 2 * 20
    boot = conf["primary"]["bootstrap"]
    assert boot["n_resamples"] == 300 and boot["n_clusters"] == {"hle": 20, "bcb": 20} and boot["seed"] == conf["bootstrap_seed"]
    assert boot["estimate"] > 0.02 and boot["p_one_sided"] < 0.05
    assert conf["primary"]["losses"]["weighted_brier"]["augmented"] < conf["primary"]["losses"]["weighted_brier"]["baseline"]
    assert "parameter_matched_text_expansion" in conf and conf["parameter_matched_text_expansion"]["bootstrap"]["estimate"] > 0
    # the panel prefix rule: --panel 5 keeps ranks < 5 in each superdomain
    assert A.main([*args, "--stage", "G-confirm", *COMMON[:-4], "--panel", "5", "--subsamples", "10"]) == 0
    small = json.loads((run_root / "neural" / "G_confirm_summary.json").read_text())
    assert small["assembly"]["rows_in_split"] == 3 * 2 * 5 and small["primary"]["bootstrap"]["estimation_only"]
    # frozen artifacts are immutable: G-confirm never rewrote them
    assert json.loads((run_root / "neural" / "G_frozen.json").read_text())["sha256"] == frozen["sha256"]
    # --- C -----------------------------------------------------------------------------
    assert A.main([*args, "--stage", "C", *COMMON]) == 0
    c = json.loads((run_root / "neural" / "C_summary.json").read_text())
    assert set(c["scopes"]) == {C.SCOPE_PERSONAL, C.SCOPE_TEAM}
    assert c["scopes"][C.SCOPE_PERSONAL]["primary"] in ("conditioned", "common")
    assert c["primary"]["bootstrap"]["n_clusters"] == {"hle": 20, "bcb": 20}
    assert (run_root / "neural" / "C_frozen.json").is_file() and (run_root / "neural" / "tables" / "C_losses.parquet").is_file()
    scored = pd.read_parquet(run_root / "neural" / "tables" / "C_losses.parquet")
    assert len(scored) == 120 and {"brier_baseline", "brier_explicit", "q_raw_explicit"} <= set(scored.columns)
    assert c["primary"]["diagnostics"]["explicit"]["recalibrated"]["n"] == 120
    assert c["frozen_source"] == "fitted" and c["dev"]["stage"] == "C-dev" and (run_root / "neural" / "C_dev_summary.json").is_file()
    assert c["primary"]["n_items_complete"] == 40 and c["primary"]["n_items_dropped"] == 0 and c["primary"]["methods"] == ["CEN_FLAT", "DEC", "IND_VOTE"]
    # --- P1-2 regression: the C recalibrators are frozen once; C / C-confirm never refit ---
    frozen_c = json.loads((run_root / "neural" / "C_frozen.json").read_text())
    assert frozen_c["kind"] == "C_FROZEN" and frozen_c["sha256"]
    assert A.main([*args, "--stage", "C", *COMMON]) == 0
    assert json.loads((run_root / "neural" / "C_frozen.json").read_text())["sha256"] == frozen_c["sha256"]
    assert json.loads((run_root / "neural" / "C_summary.json").read_text())["frozen_source"] == "loaded"
    assert A.main([*args, "--stage", "C-confirm", *COMMON]) == 0
    conf_c = json.loads((run_root / "neural" / "C_summary.json").read_text())
    assert conf_c["stage"] == "C-confirm" and conf_c["frozen_sha256"] == frozen_c["sha256"] and conf_c["primary"]["bootstrap"]["n_clusters"] == {"hle": 20, "bcb": 20}
    assert json.loads((run_root / "neural" / "C_frozen.json").read_text())["sha256"] == frozen_c["sha256"]
    # C-dev refuses to overwrite the frozen artifact unless --refreeze is explicit
    assert A.main([*args, "--stage", "C-dev", *COMMON]) == 4
    assert json.loads((run_root / "neural" / "C_frozen.json").read_text())["sha256"] == frozen_c["sha256"]
    assert A.main([*args, "--stage", "C-dev", *COMMON, "--refreeze"]) == 0
    assert json.loads((run_root / "neural" / "C_dev_summary.json").read_text())["refrozen"] is True
    # a tampered / missing frozen artifact is refused by C-confirm and by C (which never silently refits)
    (run_root / "neural" / "C_frozen.json").write_text(json.dumps({**frozen_c, "meta": {"tampered": True}}))
    assert A.main([*args, "--stage", "C-confirm", *COMMON]) == 4
    assert A.main([*args, "--stage", "C", *COMMON]) == 4
    (run_root / "neural" / "C_frozen.json").unlink()
    assert A.main([*args, "--stage", "C-confirm", *COMMON]) == 4
    # frozen G records the registered report anchor (STATE_ANCHOR, not the last prefill token)
    assert frozen["meta"]["report_anchor"] == R.STATE_ANCHOR


def test_outcome_table_uses_the_primary_budget_and_never_mislabels_tied_classes(tmp_path):
    """P1-5 regression: ``duplicate_frequency`` comes from the sealed pool's distinct answer
    classes (seal register ``vote_keys``), not from ``tied_classes``; the budget filter is
    ``cfg.budget.primary``."""
    run_root = tmp_path / "study_o"
    world = write_world(run_root, n_dev=2, n_main=2, seed=5, native=False)
    table = pd.read_parquet(run_root / "tables" / "selections.parquet")
    row = table.iloc[0]
    out = A.outcome_table(run_root)
    assert out is not None and len(out) == len(table)
    rec = out[(str(row["source_id"]), str(row["method"]))]
    assert rec["duplicate_frequency"] is None and rec["tied_winning_classes"] == float(row["tied_classes"]) and rec["all_singleton"] == 0.0
    assert rec["agreement"] == pytest.approx(float(row["winning_count"]) / float(row["valid_count"]))
    assert A.outcome_table(run_root, B=int(row["B"]) + 1) == {}
    # a seal register with vote_keys supplies the distinct classes: 5 valid, 3 classes → 0.4
    from agents_scaling.study.selection import seal as seals

    seal = "ab" * 32
    sel_dir = seals.seals_dir(run_root, seal)
    sel_dir.mkdir(parents=True, exist_ok=True)
    keys = {f"c{i}": k for i, k in enumerate(["B", "B", "A", "C", "B"])}
    register = {"schema_version": 1, "selections": {str(row["selection_id"]): {"selection_id": str(row["selection_id"]), "source_id": str(row["source_id"]),
                                                                                "record": {"vote_keys": {**keys, "c5": None}}}}}
    (sel_dir / seals.SELECTIONS_FILE).write_text(json.dumps(register))
    assert A.sealed_distinct_classes(run_root, seal) == {str(row["selection_id"]): 3}
    out = A.outcome_table(run_root, seal=seal)
    assert out[(str(row["source_id"]), str(row["method"]))]["duplicate_frequency"] == pytest.approx(1.0 - 3.0 / float(row["valid_count"]))
    assert all(v["duplicate_frequency"] is None for k, v in out.items() if k != (str(row["source_id"]), str(row["method"])))
    assert A.sealed_distinct_classes(run_root, "cd" * 32) == {} and A.sealed_distinct_classes(run_root, "nope") == {}
    del world
