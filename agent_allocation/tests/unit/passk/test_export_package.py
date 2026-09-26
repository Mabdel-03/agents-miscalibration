"""The paper export package: CSV conventions round-trip, the fixed layout is written, README.md is never touched,
missing sources become gaps, and a re-run is byte-identical."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agent_allocation.passk import export_package as ep


def test_csv_roundtrip_conventions(tmp_path):
    df = pd.DataFrame({
        "i": [1, 2, 3],
        "f": [0.5, None, 1e-9],
        "b": [True, False, True],
        "tri": [True, None, False],
        "lst": [[1, 2], [], ["a"]],
        "obj": [{"b": 1, "a": 2}, None, {}],
        "s": ["x,y", 'q"uote\nline', None],
    })
    path = tmp_path / "t.csv"
    assert ep.write_csv(df, path) == 3
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "i,f,b,tri,lst,obj,s"            # one header row, no index column
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(back["i"]) == ["1", "2", "3"]
    assert list(back["b"]) == ["true", "false", "true"]
    assert list(back["tri"]) == ["true", "", "false"]                 # missing -> empty cell
    assert [json.loads(v) for v in back["lst"]] == [[1, 2], [], ["a"]]
    assert json.loads(back["obj"][0]) == {"a": 2, "b": 1} and back["obj"][1] == "" and back["obj"][2] == "{}"
    assert list(back["s"]) == ["x,y", 'q"uote\nline', ""]
    assert float(back["f"][2]) == 1e-9 and back["f"][1] == ""


def test_count_columns_become_integers_and_types_are_labelled():
    df = pd.DataFrame({"n_items": [48.0, None], "acc_n_items": [3.0, 4.0], "p": [1.0, 0.0], "rate": [2.0, 2.5]})
    out = ep.intify_counts(df)
    assert str(out["n_items"].dtype) == "Int64" and str(out["acc_n_items"].dtype) == "Int64"
    assert out["p"].dtype == float and out["rate"].dtype == float        # not count-like: left alone
    assert ep.infer_type(pd.Series([[1], [2]])) == "JSON list"
    assert ep.infer_type(pd.Series([True, None], dtype=object)) == "boolean"
    assert ep.describe("minus_1x16_acc_ci95_lo", None).startswith("selector minus fixed:1x16, delivered accuracy")
    assert ep.describe("p", None) == "two-sided bootstrap p-value"
    assert ep.describe("no_such_column_xyz", None).startswith("UNDOCUMENTED")


def _fake_study(root):
    (root / "artifacts" / "manifests").mkdir(parents=True)
    (root / "artifacts" / "shards").mkdir(parents=True)
    panels = {"L96": ["aime/2025/1", "gpqa_diamond/recX"], "C48": ["aime/2025/1"], "V48": ["aime/2025/1"],
              "T200": ["hmmt/2025/2"], "D32": [], "X80": []}
    (root / "artifacts" / "manifests" / "splits.json").write_text(json.dumps({
        "panels": panels, "study_id": "openrouter_passk_cont_v1", "passk_seed": 20260925,
        "c48": {"eligible": ["aime/2025/1", "gpqa_diamond/recX"], "cells": {"aime/2025/1": "16-25"},
                "strata": {"aime": {"16-25": {"size": 1, "selected": 1}}}, "inclusion_prob": {"aime/2025/1": 1.0},
                "bank_correct": {"aime/2025/1": 20, "gpqa_diamond/recX": 3}, "target_per_cell": 3}}))
    (root / "artifacts" / "shards" / "assignment.json").write_text(json.dumps({"assignment": {
        "aime/2025/1": "mancer", "gpqa_diamond/recX": "parasail", "hmmt/2025/2": "mancer"}}))
    (root / "analysis_inputs").mkdir()
    rows = []
    for k, (item, bench, part, correct) in enumerate([("aime/2025/1", "aime", "legacy_main", True),
                                                       ("aime/2025/1", "aime", "legacy_main", False),
                                                       ("gpqa_diamond/recX", "gpqa_diamond", "legacy_main", None),
                                                       ("hmmt/2025/2", "hmmt", "t200_bank", True)]):
        rows.append({"candidate_id": f"c{k}", "source": "legacy" if part.startswith("legacy") else "passk",
                     "partition": part, "panel_roles": ["L96"] if part.startswith("legacy") else ["T200"],
                     "draw_role": "pool", "item_id": item, "benchmark": bench, "shard": "mancer", "kind": "root",
                     "allowance_a": 1, "replicate": k, "parse_status": "valid", "final_present": True,
                     "vote_text": "7", "vote_exact_key": "7", "option": None, "answer_len_chars": 10,
                     "answer_finish_reason": "stop", "latency_s": 20.0 + k, "think_in": 400, "think_out": 1024,
                     "answer_in": 1500, "answer_out": 300, "cost_usd": 0.004, "absent": False, "pending": False,
                     "graded": True, "correct": correct,
                     "grade_state": {True: "correct", False: "wrong", None: "unknown"}[correct]})
    draws = pd.DataFrame(rows)
    draws["correct"] = draws["correct"].astype(object)
    draws.to_parquet(root / "analysis_inputs" / "draws.parquet", index=False)
    (root / "e8b").mkdir()
    with (root / "e8b" / "calls.jsonl").open("w") as fh:
        for s in range(5):
            fh.write(json.dumps({"arm": "B0", "panel": "T200", "item_id": "hmmt/2025/2", "benchmark": "hmmt",
                                 "shard": "mancer", "sample": s, "node_id": f"n{s}", "parse_try": 0,
                                 "transport_try": 1, "transport_status": "ok", "choice": "1x16" if s else None,
                                 "valid": bool(s), "final": True, "latency_s": 10.0, "visible": "<final>1x16</final>",
                                 "cost_usd": "0.003", "raw": f"e8b/raw/B0/n{s}_1.json.gz",
                                 "option_order": ["1x1", "1x16"]}) + "\n")
    return draws


def test_export_writes_layout_keeps_readme_and_is_deterministic(tmp_path):
    root, out = tmp_path / "study", tmp_path / "pkg"
    draws = _fake_study(root)
    out.mkdir()
    (out / "README.md").write_text("owner write-up\n")
    (out / "README_methods_draft.md").write_text("a draft another stage keeps here\n")
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n")
    summary = ep.export(root, out, plan_path=plan, load_strata=False)

    for sub in ep.CSV_DIRS:
        assert (out / "csv" / sub).is_dir()
    for sub in ("prompts", "prereg", "figures"):
        assert (out / sub).is_dir()
    for name in ("MANIFEST.csv", "DATA_DICTIONARY.md", "code_manifest.txt", "prereg/plan.md",
                 "prompts/solver_templates/root.txt"):
        assert (out / name).is_file(), name
    assert (out / "README.md").read_text() == "owner write-up\n"
    assert (out / "README_methods_draft.md").read_text() == "a draft another stage keeps here\n"
    assert any("untouched: ok" in c for c in summary["checks"])
    assert any("e8a/calls.jsonl" in g for g in summary["gaps"])           # missing sources are reported

    back = pd.read_csv(out / "csv" / "02_draws" / "draws_all.csv", dtype=str, keep_default_na=False)
    assert list(back["candidate_id"]) == list(draws["candidate_id"])
    assert list(back["correct"]) == ["true", "false", "", "true"]
    assert list(back["panel_roles"].map(json.loads)) == [["L96"], ["L96"], ["L96"], ["T200"]]
    manifest = pd.read_csv(out / "MANIFEST.csv")
    row = manifest[manifest["path"] == "csv/02_draws/draws_all.csv"].iloc[0]
    assert int(row["rows"]) == len(draws)
    assert "README.md" in set(manifest["path"])

    summ = pd.read_csv(out / "csv" / "10_e8b_lex_llm" / "choice_summary.csv")
    assert summ.loc[0, "n_fallback_1x16"] == 1 and summ.loc[0, "modal_choice"] == "1x16"
    assert summ.loc[0, "modal_share"] == 1.0                                 # the invalid decision falls back to 1x16

    first = (out / "MANIFEST.csv").read_bytes()
    ep.export(root, out, plan_path=plan, load_strata=False)
    assert (out / "MANIFEST.csv").read_bytes() == first
    assert (out / "README.md").read_text() == "owner write-up\n"


def test_export_refuses_to_clear_a_directory_that_is_not_a_package(tmp_path):
    out = tmp_path / "somewhere"
    (out / "figures").mkdir(parents=True)                  # an owned name, but no MANIFEST.csv of an earlier export
    (out / "figures" / "important.png").write_bytes(b"keep me")
    with pytest.raises(RuntimeError, match="refusing to clear"):
        ep.export(tmp_path / "missing_study", out, plan_path=None, load_strata=False)
    assert (out / "figures" / "important.png").read_bytes() == b"keep me"
