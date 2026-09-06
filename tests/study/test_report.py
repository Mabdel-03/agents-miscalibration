"""report.py: panel-aware headline, freshness gate, per-cell totals (regressions from the stats review)."""
from __future__ import annotations

import json
from pathlib import Path

from agents_scaling.study import report as RP


def _stats(panel_used="complete", n_prefix=0, tables=None):
    fam = {f: {"p": 1.0, "executed": False, "note": "not executed", "holm": {"p": 1.0, "p_holm": 1.0, "reject": False}} for f in ("R", "C", "G", "M")}
    fam["A"] = {"executed": True, "estimate": -0.01, "ci95": [-0.05, 0.03], "n": {"hle": 72, "bcb": 56}, "n_prefix": n_prefix,
                "panel_used": panel_used, "panel_note": "matched complete-case panel",
                "prefix": {"estimate": None, "note": "no domain with >= 2 items"},
                "complete": {"estimate": -0.01, "by_domain": {"hle": 0.007, "bcb": -0.027}, "n": {"hle": 72, "bcb": 56}},
                "holm": {"p": 0.6, "p_holm": 1.0, "reject": False}}
    fam["O"] = {"executed": True, "estimate": None, "p": 0.07, "methods": ["S_FRESH", "DEC"], "n_prefix": n_prefix,
                "panel_used": panel_used, "panel_note": "matched complete-case panel",
                "prefix": {"p_global": 1.0, "pairs": [], "methods": []},
                "complete": {"p_global": 0.07, "pairs": [], "methods": [{"method": "S_FRESH", "mean": 0.34}, {"method": "DEC", "mean": 0.37}]},
                "holm": {"p": 0.07, "p_holm": 0.42, "reject": False}}
    return {"manifest": {"analysis_seed": 1, "n_resamples_primary": 20000, "code_version": "abc", "tables": tables or {}},
            "families": fam, "holm": {f: fam[f]["holm"] for f in ("A", "O", "R", "C", "G", "M")}}


def _run_root(tmp_path: Path, *, stats=None, rec=None, cells=None, metas=None) -> Path:
    root = tmp_path / "study_x"
    (root / "tables").mkdir(parents=True)
    (root / "freeze").mkdir(parents=True)
    (root / "freeze" / "amendments.json").write_text(json.dumps([{"id": "F1", "deviation": "d", "spec_clause": "s", "when": "w"}]))
    if stats is not None:
        (root / "tables" / "stats.json").write_text(json.dumps(stats))
        (root / "tables" / "stats.md").write_text("# study_v4 confirmatory statistics\n")
    if rec is not None:
        (root / "reconcile.json").write_text(json.dumps(rec))
    for name, cell_ids in (cells or {}).items():
        (root / name).write_text(json.dumps({"cells": [{"cell_id": c} for c in cell_ids]}))
    for cid, meta in (metas or {}).items():
        d = root / "cells" / cid
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(meta))
    return root


def test_headline_names_the_panel_actually_used_and_prints_its_numbers(tmp_path):
    root = _run_root(tmp_path, stats=_stats(), rec={"manifests": {}, "judge": {}, "vote": {}})
    text = RP.build_report(root, generated_at="now")
    assert "**complete** panel" in text and "prefix panel n" not in text
    assert "{'hle': 0.007, 'bcb': -0.027}" in text or "0.007" in text
    assert "S_FRESH 0.340" in text and "DEC 0.370" in text
    assert "not estimable on this panel" not in text


def test_headline_falls_back_to_a_readable_line_when_a_panel_has_no_methods(tmp_path):
    st = _stats(panel_used="prefix", n_prefix=0)
    root = _run_root(tmp_path, stats=st, rec={"manifests": {}, "judge": {}, "vote": {}})
    text = RP.build_report(root, generated_at="now")
    assert "**prefix** panel" in text and "per-method means not estimable on this panel" in text


def test_freshness_flags_stale_stats_stale_reconcile_and_too_few_resamples(tmp_path):
    st = _stats(tables={"selections": 10, "banks": 2, "episodes": 4})
    st["manifest"]["n_resamples_primary"] = 2000
    root = _run_root(tmp_path, stats=st, rec={"manifests": {"cells_a.json": {}}, "judge": {}, "vote": {}},
                     cells={"cells_a.json": ["c1"], "cells_b.json": ["c2"]})
    warn = RP.freshness(root, st, json.loads((root / "reconcile.json").read_text()))
    assert any("2,000 resamples" in w for w in warn)
    assert any("reconcile.json is STALE" in w and "cells_b.json" in w for w in warn)
    text = RP.build_report(root, generated_at="now")
    assert "INPUT FRESHNESS WARNING" in text


def test_freshness_is_silent_when_inputs_match(tmp_path):
    st = _stats(tables={})
    root = _run_root(tmp_path, stats=st, rec={"manifests": {"cells_a.json": {}}, "judge": {}, "vote": {}}, cells={"cells_a.json": ["c1"]})
    assert RP.freshness(root, st, json.loads((root / "reconcile.json").read_text())) == []
    assert "INPUT FRESHNESS WARNING" not in RP.build_report(root, generated_at="now")


def test_freshness_reports_missing_inputs(tmp_path):
    root = _run_root(tmp_path)
    warn = RP.freshness(root, None, None)
    assert any("stats.json is MISSING" in w for w in warn) and any("reconcile.json is MISSING" in w for w in warn)


def test_run_totals_counts_each_cell_once_across_manifests(tmp_path):
    meta = {"n_generated_requests": 100, "n_aliased_requests": 5, "flops_total": 2_000_000_000_000_000}
    root = _run_root(tmp_path,
                     cells={"cells_1.json": ["c1", "c2"], "cells_1-eval.json": ["c2", "c3"]},
                     metas={"c1": meta, "c2": meta, "c3": meta})
    rec = {"manifests": {"cells_1.json": {"A|32B": {}}, "cells_1-eval.json": {"eval|32B": {}}}}
    tot, dedup = RP.run_totals(rec, root)
    assert tot["cells"] == 3 and tot["generated"] == 300 and tot["aliased"] == 15
    assert tot["flops"] == 6_000_000_000_000_000 and dedup["distinct_cells"] == 3


def test_scope_section_deduplicates_incomplete_items(tmp_path):
    root = _run_root(tmp_path)
    inc = {"cell_id": "A.S_HISTORY.x", "source_id": "hle:1"}
    rec = {"manifests": {"m1.json": {"A|32B": {"incomplete": [inc]}}, "m2.json": {"A|32B": {"incomplete": [dict(inc)]}}}, "judge": {}, "vote": {}}
    text = "\n".join(RP.scope_section(rec, root))
    assert "Infrastructure-incomplete items (distinct, listed and never hidden; §10.4): 1" in text


def test_family_G_caveat_states_the_realised_development_sample():
    g = [c for c in RP.CAVEATS if c.startswith("Family G")][0]
    assert "20 items" in g and "amendment E5" in g and "~60 dev items" not in g
