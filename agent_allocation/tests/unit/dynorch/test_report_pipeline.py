"""WP-C: the report pipeline end to end on SYNTHETIC harness rows joined to the REAL sealed ladder tables
(contract section 7): analysis writes numbers.json and every table and figure, the primary family is Holm over
four with the anchor first, and render resolves every placeholder of report.tmpl.tex (an unresolved one fails)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_allocation.dynorch import tables as T

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs" / "dynorch_report"
CFG = json.loads((ROOT / "configs" / "dynorch_v1.json").read_text())
BASE = Path(CFG["base_run_root"])
pytestmark = pytest.mark.skipif(not (BASE / "merged" / "tables" / "policy_outcomes.parquet").is_file(),
                                reason="sealed forced_budget_v2 tables not mounted")
TABLES = ["headline.tex", "primary.tex", "primary_contrasts.csv", "rung_table.csv", "descriptive.csv",
          "realized.tex", "diagnostics.tex", "provider_sensitivity.tex", "provider_sensitivity.csv", "cost.tex",
          "frontier.csv", "repeat_agreement.csv", "forecast_refit.csv"]
FIGURES = [f"d{i}_{n}.pdf" for i, n in enumerate(("design", "ladder_f1", "realized_f2", "dollars", "per_benchmark",
                                                  "diagnostics", "cap_binding", "provider_sensitivity",
                                                  "repeat_agreement"), 1)]


def _report_modules():
    if str(REPORT) not in sys.path:
        sys.path.insert(0, str(REPORT))
    import analysis, core, primary, render  # noqa: E401,E402
    return analysis, core, primary, render


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dynorch_run")
    panels = json.loads((BASE / "manifests" / "panels.json").read_text())
    assign = json.loads((BASE / "shards" / "assignment.json").read_text())["assignment"]
    items = {it: {"benchmark": b, "shard_id": "parasail", "endpoint_tag": "parasail/fp8"}
             for b in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond") for it in panels[b]["main_item_ids"]}
    plan = T.synthetic_plan(CFG, items, arms=["H1", "H3", "H0"], rungs=[1, 2, 4, 8, 16, 25])
    T.write_plan(T.plan_manifest_path(root, "main"), plan)
    for i, cell in enumerate(plan):
        if cell.arm == "H3" and cell.rung == 25 and (i // 6) % 12 == 0:
            continue                                                        # a few pending cells
        p = {1: 0.3, 2: 0.4, 4: 0.5, 8: 0.6, 16: 0.7, 25: 0.75}[cell.rung]
        h = (i * 2654435761) % 1000 / 1000.0
        kind = ["none", "none", "mid_call", "terminal"][i % 4]
        T.write_run_record(root, T.synthetic_run_record(
            cell, correct=h < p, spawns=(i % 3 if cell.arm != "H0" else 0), valid=(i % 17 != 0),
            realized_share=0.5 + (i % 5) / 10, grace=(200 if kind != "none" else 0), forced_close_kind=kind,
            base_prompt=5500 + (600 if cell.arm != "H0" else 0)))
    T.build_tables(root, CFG, stage="main")
    del assign
    return root


@pytest.fixture(scope="module")
def numbers(synthetic_root, tmp_path_factory):
    analysis, _, _, _ = _report_modules()
    out = tmp_path_factory.mktemp("report_out")
    N = analysis.run_analysis(synthetic_root, BASE / "merged", ROOT / "configs" / "dynorch_v1.json", out,
                              ladder_numbers=ROOT / "docs" / "forced_v2_report" / "numbers.json", replicates=200)
    return N, out


@pytest.mark.timeout(900)
def test_analysis_writes_numbers_tables_and_figures(numbers):
    N, out = numbers
    assert (out / "numbers.json").is_file()
    written = json.loads((out / "numbers.json").read_text())
    assert written["n_items"] == 96 and written["study"]["seed"] == 20260923
    for name in TABLES:
        assert (out / "tables" / name).is_file(), name
    for name in FIGURES:
        assert (out / "figures" / name).stat().st_size > 0, name
    for key in ("acc", "primary", "descriptive", "realized", "diagnostics", "frontier_f2", "frontier_usd",
                "provider_sensitivity", "repeat_agreement", "harness_prompt_tokens", "completeness", "slopes",
                "spawn_split", "refit", "headline", "ladder_check", "oracle", "best_non_deep_at_rung"):
        assert key in N, key


def test_primary_family_is_holm_over_four_with_anchor_first(numbers):
    N, _ = numbers
    assert N["primary_family"] == CFG["statistics"]["primary_family"] and len(N["primary_family"]) == 4
    assert N["anchor"] == "H0/16-deep/16" and N["primary"]["H0/16-deep/16"]["role"] == "descriptive"
    assert N["primary"]["H0/16-deep/16"]["p_holm"] is None
    for name in N["primary_family"]:
        e = N["primary"][name]
        assert e["role"] == "primary" and e["status"] == "complete" and e["n_complete"] == 96
        assert e["p"] is not None and e["p_holm"] is not None and e["p_holm"] >= e["p"]
        assert e["lo_pts"] <= e["est_pts"] <= e["hi_pts"]
        assert e["reading"] in {"loses_to_fixed_rule", "inconclusive", "rule_incomplete", "above_baseline",
                                "below_baseline"}
    assert N["primary_p_reported"] is True
    # the sealed ladder point estimates are reproduced in the merged frame
    assert all(v["equal_3dp"] for v in N["ladder_check"].values())


def test_descriptive_set_covers_every_comparator(numbers):
    N, _ = numbers
    e = N["descriptive"]["H1/16"]
    assert e["vs_deep"]["other"] == "deep/16" and e["vs_sample"]["other"] == "sample/16"
    assert e["vs_private"]["other"] == "private_single/16" and e["vs_H0"]["other"] == "H0/16"
    assert e["vs_best_non_deep"]["other"] == N["best_non_deep_at_rung"]["16"]
    assert not e["vs_best_non_deep"]["other"].startswith("deep/")
    assert e["vs_oracle"]["other"] == "pass@16" and e["vs_oracle"]["n_known"] >= 90
    assert N["descriptive"]["H0/16"]["vs_H0"] is None
    assert N["harness_prompt_tokens"]["H1"]["mean"] == pytest.approx(600.0)
    assert N["completeness"]["H1/16"] == 1.0 and "H3/25" in N["partial_cells"]
    assert N["realized"]["H1/16"]["cap"] == 16384 and N["cap_overrun_runs_total"] == 0


def test_render_resolves_every_placeholder_and_fails_on_a_missing_one(numbers, tmp_path):
    _, out = numbers
    _, _, _, render = _report_modules()
    text = render.render(out / "numbers.json", REPORT / "report.tmpl.tex", tmp_path / "report.tex")
    assert "??" not in text and "{{" not in text
    assert (tmp_path / "report.tex").read_text() == text
    n_placeholders = len(render.PAT.findall((REPORT / "report.tmpl.tex").read_text()))
    assert n_placeholders > 40
    with pytest.raises(render.RenderError, match="no_such_key"):
        render.render_text("x {{no_such_key.deep|3}} y", json.loads((out / "numbers.json").read_text()))
    with pytest.raises(render.RenderError, match="null"):
        render.render_text("{{v|3}}", {"v": None})
    assert render.fmt(-1.25, "+1") == "$-$1.2" or render.fmt(-1.25, "+1") == "$-$1.3"
    assert render.fmt("a_b", "tt") == "\\texttt{a\\_b}" and render.fmt(0.5, "%") == "50.0\\%"
