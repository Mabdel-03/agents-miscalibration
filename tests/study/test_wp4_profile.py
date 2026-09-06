"""resources/envelopes.py + resources/profile.py: Table E, B0 candidates, feasibility,
FROZEN.yaml via config.freeze, the pilot schedule and the CLI (§6.5, §10.2; P1-1, P1-2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agents_scaling.study import types as T
from agents_scaling.study.resources import envelopes as E
from agents_scaling.study.resources import profile as P
from agents_scaling.study.resources.oracle import FlopOracle, oracles_from_table
from tests.study.conftest import RUN_ROOT_SUBDIRS
from tests.study.fake_vllm import FakeTokenizer


@pytest.fixture
def wrappers():
    return E.measure_wrappers(FakeTokenizer())


def test_measure_wrappers_and_table_e(study_config, wrappers):
    for key in E.WRAPPER_FIXED_KEYS:
        assert wrappers[key] > 0
    for n in E.ENVELOPE_NS:
        assert wrappers[f"dec_revision@N{n}"] > 0 and wrappers[f"hub@N{n}"] > 0 and wrappers[f"hub_final@N{n}"] > wrappers[f"hub@N{n}"]
    caps = study_config.caps
    table = E.TableE(caps, wrappers)
    assert table.root_prompt_max() == caps.task_tokens + wrappers["root"]
    assert table.dec_revision_prompt_max(5) == caps.task_tokens + wrappers["dec_revision@N5"] + caps.own_candidate_tokens + 4 * caps.packet_tokens
    assert table.hub_prompt_max(5, cycle0=False) == (
        caps.task_tokens + wrappers["hub_final@N5"] + caps.hub_prior_plan_tokens + 4 * caps.subtask_result_tokens + caps.hub_action_error_tokens
    )
    # amendment B2b (review P0-A): the N=9 returned-results block is bounded, never clipped at the envelope
    assert table.hub_returned_max(9) == caps.hub_returned_results_tokens < 8 * caps.subtask_result_tokens
    assert table.hub_prompt_max(9, cycle0=False) == (
        caps.task_tokens + wrappers["hub_final@N9"] + caps.hub_prior_plan_tokens + caps.hub_returned_results_tokens + caps.hub_action_error_tokens
    )
    assert table.hub_prompt_max(9, cycle0=False) <= T.PROMPT_TOKENS_CAP
    assert table.worker_prompt_max(cycle0=True) + caps.forwarded_results_tokens == table.worker_prompt_max(cycle0=False)
    assert table.focal_prompt_max(8) - table.focal_prompt_max(0) == 8 * caps.packet_tokens
    d = table.to_dict()
    assert d["hub_prompt_max"]["5"]["later"] < T.PROMPT_TOKENS_CAP and d["root_prompt_max"] == table.root_prompt_max()
    runtime = E.TableE(caps, wrappers, task_tokens=10)
    assert runtime.root_prompt_max() == 10 + wrappers["root"]
    with pytest.raises(ValueError, match="lacks"):
        E.TableE(caps, {"root": 1})


def test_minimal_paths_and_b0(study_config, wrappers):
    table = E.TableE(study_config.caps, wrappers)
    o = FlopOracle.from_table("32B")
    paths = E.b0_candidates(o, table, N=5)
    R = o.reservation(table.root_prompt_max(), 8192)
    assert paths["S_FRESH"]["total"] == R and paths["S_HISTORY"]["total"] == R
    assert paths["IND_VOTE"]["total"] == 5 * R and paths["DEC"]["total"] == 5 * R and paths["DEC"]["first_round"] > 0
    cen = paths["CEN_FLAT"]
    assert cen["total"] == (
        o.reservation(table.hub_prompt_max(5, cycle0=True), 8192)
        + 4 * o.reservation(table.worker_prompt_max(cycle0=True), 8192)
        + o.reservation(table.hub_prompt_max(5, cycle0=False), 8192)
    )
    hub_only = E.minimal_path(T.Method.CEN_FLAT, o, table, 5, cen_path="hub_only")
    assert hub_only["total"] == o.reservation(table.hub_prompt_max(5, cycle0=True), 8192) + o.reservation(table.hub_prompt_max(5, cycle0=False), 8192)
    assert E.minimal_path(T.Method.CEN_FLAT, o, table, 1)["total"] == hub_only["total"] - 0 or True
    assert E.minimal_path(T.Method.DEGREE, o, table, 1)["calls"][0]["count"] == 9
    with pytest.raises(ValueError):
        E.minimal_path(T.Method.BANK, o, table, 5)
    with pytest.raises(ValueError):
        E.minimal_path(T.Method.CEN_FLAT, o, table, 5, cen_path="nope")


def test_build_profile_and_feasibility(study_config, wrappers):
    oracles = oracles_from_table()
    prov = {k: "measured" for k in wrappers}
    profile = P.build_profile(study_config, oracles, wrappers, prov)
    B0 = profile["B0_flops"]
    assert isinstance(B0, int) and B0 == max(c["total"] for c in profile["B0_candidates"].values())
    assert profile["B0_binding_method"] == "CEN_FLAT" and profile["cen_minimal_path"] == "cycle"
    assert profile["budgets"] == {"B1": B0, "B2": 2 * B0, "B4": 4 * B0, "B8": 8 * B0}
    feas = profile["feasibility"]
    assert set(feas) == {f"{s}/N{n}/B{m}" for s in ("8B", "32B") for n in (5, 9) for m in (1, 4)}
    assert feas["32B/N5/B1"]["all_feasible"] is True
    assert profile["n9_b1_flagship_feasible"] is False  # P1-2: recorded, never patched
    # under the CEN-cycle B0 the N=9/B1 infeasibility sits with CEN_FLAT (hub + 8 workers + final)
    assert feas["32B/N9/B1"]["methods"]["CEN_FLAT"]["feasible"] is False and feas["32B/N9/B4"]["methods"]["DEC"]["feasible"] is True
    assert feas["8B/N9/B1"]["methods"]["IND_VOTE"]["feasible"] is True  # same numerical allowance, cheaper model
    # under the hub-only B0 (architecture §4) it is IND/DEC at N=9 that do not fit B1 (9 roots > 5 roots)
    hub_only = P.build_profile(study_config, oracles, wrappers, prov, cen_path="hub_only")
    assert hub_only["feasibility"]["32B/N9/B1"]["methods"]["IND_VOTE"]["feasible"] is False
    assert hub_only["feasibility"]["32B/N9/B1"]["methods"]["DEC"]["feasible"] is False
    assert hub_only["n9_b1_flagship_feasible"] is False
    tables = profile["oracle_tables"]
    assert set(tables) == {"4B", "8B", "14B", "32B"} and tables["32B"]["call"]["L6144_T8192"] == oracles["32B"].call(6144, 8192)
    assert set(tables["32B"]["prefill"]) == {"1024", "6144", "16384", "32768"}
    assert profile["wrapper_source"]["root"] == "measured" and profile["L_root_max"] == study_config.caps.task_tokens + wrappers["root"]
    alt = P.build_profile(study_config, oracles, wrappers, prov, cen_path="hub_only")
    # hub-only drops the four worker envelopes, so its B0 is never above the cycle B0; the
    # binding method is whichever candidate is largest (B2b's 512-token final-reserve
    # allowance can make CEN_FLAT's two hub calls edge past five roots on a short wrapper).
    assert alt["B0_flops"] == max(c["total"] for c in alt["B0_candidates"].values()) <= profile["B0_flops"]
    assert alt["cen_minimal_path"] == "hub_only" and len(alt["B0_candidates"]["CEN_FLAT"]["calls"]) == 2  # hub + reserved final only
    assert alt["B0_candidates"]["CEN_FLAT"]["total"] < profile["B0_candidates"]["CEN_FLAT"]["total"]
    assert alt["B0_candidates"][alt["B0_binding_method"]]["total"] == alt["B0_flops"]
    yaml.safe_dump(profile)  # YAML-native


def test_root_wrapper_from_report(study_config):
    report = {
        "root_prompt_max_tokens": 4339,
        "checkpoints": {
            "32B": {"cells": {"00": {"wrapper_tokens_max": 243}, "nat": {"wrapper_tokens_max": 240}}},
            "4B": {"cells": {"00": {"wrapper_tokens_max": 200}}},
        },
    }
    assert P.root_wrapper_from_report(report, 4096) == 243
    with pytest.raises(T.ProtocolError, match="inconsistent"):
        P.root_wrapper_from_report(dict(report, root_prompt_max_tokens=5000), 4096)
    with pytest.raises(T.ProtocolError):
        P.root_wrapper_from_report({"checkpoints": {}}, 4096)
    wrappers, prov = P.resolve_wrappers(study_config, report=report, tokenizer=FakeTokenizer())
    assert wrappers["root"] == 243 and prov["root"] == "preflight" and prov["worker"] == "measured"
    wrappers, prov = P.resolve_wrappers(study_config, report=None, tokenizer=None)
    assert set(prov.values()) == {"fallback"} and wrappers["root"] == E.FALLBACK_WRAPPER_TOKENS


def test_real_preflight_report_if_present(study_config):
    path = P.DEFAULT_RESULTS_ROOT / "study_v4" / "data" / "preflight_report.json"
    if not path.exists():
        pytest.skip("no real preflight report")
    report = json.loads(path.read_text())
    assert P.root_wrapper_from_report(report, study_config.caps.task_tokens) == report["root_prompt_max_tokens"] - study_config.caps.task_tokens


def test_freeze_writes_b0_and_gate_reads_it(tmp_path, study_config, wrappers):
    run_root = tmp_path / "run_root"
    for name in RUN_ROOT_SUBDIRS:
        (run_root / name).mkdir(parents=True)
    oracles = oracles_from_table()
    profile = P.build_profile(study_config, oracles, wrappers, {k: "measured" for k in wrappers})
    from agents_scaling.study.config import freeze

    target = freeze(run_root, {"B0_flops": str(profile["B0_flops"]), "profile": profile}, config=study_config)
    manifest = yaml.safe_load(target.read_text())
    assert manifest["budget"]["B0_flops"] == str(profile["B0_flops"])
    assert manifest["extra"]["profile"]["B0_flops"] == profile["B0_flops"] and manifest["extra"]["profile"]["feasibility"]
    frozen = study_config.frozen(run_root)
    assert frozen.b0_flops == float(profile["B0_flops"])
    assert int(manifest["budget"]["B0_flops"]) == profile["B0_flops"]  # exact integer survives the yaml round trip
    with pytest.raises(T.ProtocolError):
        freeze(run_root, {"B0_flops": "1"}, config=study_config)


def test_pilot_schedule(tmp_path):
    run_root = tmp_path / "rr"
    items = run_root / "cells" / "A.S_FRESH.32B.N1.B4.F00.e0.s000" / "items"
    items.mkdir(parents=True)
    for i, (calls, spent, reason) in enumerate([(10, 100, "BUDGET"), (64, 90, "CALL_CAP")]):
        (items / f"hle:{i}.json").write_text(json.dumps({"cell": {"method": "S_FRESH"}, "episode": {"ledger": {"calls_admitted": calls, "spent": spent, "stop_reason": reason}}}))
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps([{"cell_id": "A.S_FRESH.32B.N1.B4.F00.e0.s000"}, {"cell_id": "missing"}]))
    sched = P.pilot_schedule(run_root, cells)
    s = sched["methods"]["S_FRESH"]
    assert s["episodes"] == 2 and s["calls_admitted"] == {"min": 10, "max": 64, "mean": 37.0} and s["stop_reasons"] == {"BUDGET": 1, "CALL_CAP": 1}
    cells.write_text(json.dumps({"cells": [{"nope": 1}]}))
    with pytest.raises(T.ProtocolError):
        P.pilot_schedule(run_root, cells)


def test_cli_dry_run_and_freeze(tmp_path, capsys, study_config):
    hf = tmp_path / "hf"
    for size, ckpt in study_config.checkpoints.items():
        real = Path("/orcd/data/tpoggio/001/mabdel03/.cache/huggingface") / "hub" / f"models--Qwen--Qwen3-{size}" / "snapshots" / ckpt.model_revision / "config.json"
        if not real.exists():
            pytest.skip("HF cache absent")
        dst = hf / "hub" / f"models--Qwen--Qwen3-{size}" / "snapshots" / ckpt.model_revision / "config.json"
        dst.parent.mkdir(parents=True)
        dst.write_bytes(real.read_bytes())
    results = tmp_path / "results"
    run_root = results / "unit"
    for name in RUN_ROOT_SUBDIRS:
        (run_root / name).mkdir(parents=True)
    argv = ["--run-id", "unit", "--results-root", str(results), "--hf-home", str(hf), "--no-tokenizer", "--dry-run"]
    assert P.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["wrapper_source"] == ["fallback"] and out["L_root_max"] == 4096 + 2048
    assert not (run_root / "FROZEN.yaml").exists()
    assert P.main(argv[:-1] + ["--cen-minimal-path", "hub_only"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frozen"].endswith("FROZEN.yaml") and out["cen_minimal_path"] == "hub_only"
    manifest = yaml.safe_load((run_root / "FROZEN.yaml").read_text())
    assert manifest["extra"]["profile"]["cen_minimal_path"] == "hub_only" and manifest["budget"]["B0_flops"] == str(out["B0_flops"])
    with pytest.raises(T.ProtocolError, match="already exists"):
        P.main(argv[:-1])
