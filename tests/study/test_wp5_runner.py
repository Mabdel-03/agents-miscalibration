"""runner / run_one / seal / aggregate / reconcile end-to-end on the WP2 fake vLLM server.

Fixtures: T9 (kill mid-cell, rerun → zero new HTTP calls), T18 (evaluation refuses unsealed
items), F-4 (pass@K via metrics_reference), P0-3 (aggregate refuses a join when a selection
sealed after the evaluation started), architecture §3 exit codes (0/2/3/4), SIGUSR1 stop
without meta.json, incomplete/ handling, JUDGE_BEST companion scoring outside B.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agents_scaling.study import aggregate as AG
from agents_scaling.study import metrics_reference as MR
from agents_scaling.study import reconcile as RC
from agents_scaling.study import run_one
from agents_scaling.study import types as T
from agents_scaling.study.evaluation import bcb as bcb_eval
from agents_scaling.study.evaluation import hle_judge
from agents_scaling.study.runner import CellPaths
from agents_scaling.study.selection import seal as S
from tests.study import wp5_support as W

POOL = ("x = 1\n", "B", "x = 2\n")


@pytest.fixture
def world(tmp_path: Path, study_config):
    """Export (2 HLE + 2 BCB main items), a registered fake endpoint, a test freeze."""
    run_root = tmp_path / "study_v4"
    for name in ("servers", "cells", "requests", "eval", "seals"):
        (run_root / name).mkdir(parents=True)
    tasks = W.make_tasks(2, "main")
    W.write_export(run_root, tasks)
    with W.fake_server(run_root, answer_pool=POOL) as srv:
        R = W.root_reservation(study_config, srv.tokenizer, tasks[0])
        b0 = (5 * R + R // 4) // 4  # B4 admits a five-root initialization plus a little optional work
        yield {"run_root": run_root, "tasks": tasks, "server": srv, "harness": W.Harness(run_root, study_config, srv), "b0": b0, "R": R}


def _ids(tasks):
    return [t.source_id for t in tasks]


def test_bank_cell_resume_zero_new_calls(world, study_config):
    rr, srv, h, tasks = world["run_root"], world["server"], world["harness"], world["tasks"]
    cell = W.make_cell(T.Method.BANK, _ids(tasks), module="F", N=1, B=0, parallel_items=2, max_inflight=10)
    assert h.run(cell) == T.EXIT_DONE
    paths = CellPaths.of(rr, cell.cell_id)
    assert paths.meta.exists() and srv.chat_count == 40
    item = json.loads(paths.item(tasks[0].source_id).read_text())
    assert item["bank"] is not None and len(item["bank"]) == 10 and item["episode"] is None and item["status"] == "complete"
    assert all(b["vote_key"] is not None for b in item["bank"] if b["valid"])
    assert {b["candidate"]["final_answer"] for b in item["bank"] if b["valid"]} <= set(POOL)
    meta = json.loads(paths.meta.read_text())
    assert meta["n_generated_requests"] == 40 and meta["n_aliased_requests"] == 0 and meta["n_items"] == 4
    # idempotent re-run: nothing to do
    assert h.run(cell) == T.EXIT_DONE and srv.chat_count == 40
    # simulated kill after the requests were committed but before the item files/meta: resume = store hits only (T9)
    for sid in _ids(tasks):
        paths.item(sid).unlink()
    paths.meta.unlink()
    assert h.run(cell) == T.EXIT_DONE
    assert srv.chat_count == 40 and paths.meta.exists()
    meta = json.loads(paths.meta.read_text())
    assert meta["n_aliased_requests"] == 40 and meta["n_generated_requests"] == 0


def test_generate_refuses_without_freeze(world):
    rr, h, tasks = world["run_root"], world["harness"], world["tasks"]
    cell = W.make_cell(T.Method.S_FRESH, _ids(tasks)[:1], N=1)
    assert h.run(cell) == T.EXIT_SUSPENDED
    paths = CellPaths.of(rr, cell.cell_id)
    assert paths.suspended.exists() and not paths.meta.exists()
    assert "FROZEN.yaml" in json.loads(paths.suspended.read_text())["error"]


def test_no_server_exit_2(tmp_path: Path, study_config):
    rr = tmp_path / "empty_run"
    (rr / "servers").mkdir(parents=True)
    tasks = W.make_tasks(1, "main")
    W.write_export(rr, tasks)
    cell = W.make_cell(T.Method.BANK, _ids(tasks), module="F", N=1, B=0)

    class NoServer:
        tokenizer = None

    h = W.Harness(rr, study_config, NoServer())
    assert h.run(cell, wait_s=0.5) == T.EXIT_NO_SERVER
    assert not CellPaths.of(rr, cell.cell_id).meta.exists()


def test_infra_failure_incomplete_then_recovery(world):
    rr, srv, h, tasks = world["run_root"], world["server"], world["harness"], world["tasks"]
    cell = W.make_cell(T.Method.BANK, _ids(tasks)[:2], module="F", N=1, B=0, parallel_items=1, max_inflight=2)
    srv.fail_next(200, status=500)
    assert h.run(cell) == T.EXIT_INCOMPLETE
    paths = CellPaths.of(rr, cell.cell_id)
    assert set(paths.incomplete_items()) == set(_ids(tasks)[:2]) and not paths.meta.exists()
    rec = json.loads(paths.incomplete_item(tasks[0].source_id).read_text())
    assert "InfraFailure" in rec["error"] and rec["attempts"] and len(rec["attempts"]) == 3
    srv.reset_faults()
    assert h.run(cell) == T.EXIT_DONE
    assert paths.incomplete_items() == [] and paths.meta.exists()


def test_stop_event_finishes_inflight_without_meta(world):
    rr, h, tasks = world["run_root"], world["harness"], world["tasks"]
    cell = W.make_cell(T.Method.BANK, _ids(tasks)[:3], module="F", N=1, B=0, parallel_items=1, max_inflight=10)
    stop = threading.Event()
    assert h.run(cell, stop_event=stop, hook=stop.set) == T.EXIT_DONE
    paths = CellPaths.of(rr, cell.cell_id)
    done = [sid for sid in _ids(tasks)[:3] if paths.item(sid).exists()]
    assert len(done) == 1 and not paths.meta.exists()
    events = [json.loads(line)["event"] for line in paths.events.read_text().splitlines()]
    assert "stopped" in events and events.count("item_finished") == 1
    # a later run without the stop finishes the rest
    assert h.run(cell) == T.EXIT_DONE and paths.meta.exists()


def test_run_one_cli(world, monkeypatch):
    rr, tasks = world["run_root"], world["tasks"]
    cell = W.make_cell(T.Method.BANK, _ids(tasks)[:1], module="F", N=1, B=0)
    path, _ = W.write_cells_manifest(rr, "cells_x.json", [cell])
    rc = run_one.main(["--run-id", rr.name, "--results-root", str(rr.parent), "--cells-file", "cells_x.json", "--index", "0", "--dry-run"])
    assert rc == 0
    assert run_one.main(["--run-id", rr.name, "--results-root", str(rr.parent), "--cells-file", "cells_x.json", "--index", "7"]) == T.EXIT_SUSPENDED
    stop = threading.Event()
    run_one.install_stop_handler(stop)
    import os
    import signal

    os.kill(os.getpid(), signal.SIGUSR1)
    assert stop.wait(2.0)


def test_pipeline_seal_select_evaluate_aggregate(world, study_config, tmp_path: Path):
    rr, srv, h, tasks = world["run_root"], world["server"], world["harness"], world["tasks"]
    W.freeze_for_test(rr, study_config, world["b0"])
    ids = _ids(tasks)
    bank = W.make_cell(T.Method.BANK, ids, module="F", N=1, B=0, parallel_items=2, max_inflight=10)
    fresh = W.make_cell(T.Method.S_FRESH, ids, N=1, parallel_items=1, max_inflight=8)
    ind = W.make_cell(T.Method.IND_VOTE, ids, module="N", N=5, framing=T.Framing.F00, parallel_items=2, max_inflight=8)
    dec = W.make_cell(T.Method.DEC, ids, N=2, framing=T.Framing.NATIVE, parallel_items=2, max_inflight=2)
    for cell in (bank, fresh, ind, dec):
        assert h.run(cell) == T.EXIT_DONE, cell.cell_id
    fresh_item = json.loads(CellPaths.of(rr, fresh.cell_id).item(ids[0]).read_text())
    assert fresh_item["episode"]["counters"]["aliased_calls"] >= 1  # S_FRESH draws alias the F00 bank
    cells_path, seal = W.write_cells_manifest(rr, "cells_1_32B.json", [bank, fresh, ind, dec])

    # --- T18: evaluation before any seal refuses ---------------------------------------
    hle_cell = W.make_cell(T.Method.BANK, [t.source_id for t in tasks if t.domain is T.Domain.HLE], kind=T.CellKind.JUDGE_HLE, seal=seal)
    assert h.run(hle_cell) == T.EXIT_SUSPENDED
    assert "not sealed" in json.loads(CellPaths.of(rr, hle_cell.cell_id).suspended.read_text())["error"]
    CellPaths.of(rr, hle_cell.cell_id).suspended.unlink()

    # --- seal pools -> JUDGE_BEST -> seal selections ------------------------------------
    pools_out = S.seal_pools(rr, cells_path, clock=lambda: 1000.0)
    assert pools_out["n_items"] == 4 and pools_out["incomplete"] == 0
    pools = S.load_pools(rr, seal)
    kinds = {(p["method"], p["pool_kind"], p["prefix_k"]) for p in pools["pools"].values()}
    assert {("BANK", "bank_prefix", k) for k in S.PREFIXES} <= kinds
    assert ("S_FRESH", "archive", None) in kinds and ("IND_VOTE", "archive", None) in kinds and ("DEC", "latest_slots", None) in kinds and ("DEC", "archive", None) in kinds
    assert S.seal_pools(rr, cells_path, clock=lambda: 2000.0)["added"] == 0  # append-only, idempotent
    # still refuses evaluation: pools are sealed, selections are not (T18)
    assert h.run(hle_cell) == T.EXIT_SUSPENDED
    CellPaths.of(rr, hle_cell.cell_id).suspended.unlink()

    jb = W.make_cell(T.Method.BANK, ids, kind=T.CellKind.JUDGE_BEST, seal=seal, max_inflight=4)
    calls_before = srv.chat_count
    assert h.run(jb) == T.EXIT_DONE
    jb_item = json.loads(CellPaths.of(rr, jb.cell_id).item(ids[0]).read_text())
    assert jb_item["kind"] == "JUDGE_BEST" and jb_item["companion_cost"]["calls"] >= 1 and all(0 <= s <= 1 for s in jb_item["scores"].values() if s is not None)
    assert srv.chat_count > calls_before
    select_path, _ = W.write_cells_manifest(rr, "cells_1-select_32B.json", [jb])
    sel_out = S.seal_selections(rr, cells_path, cfg=study_config, select_cells_file=select_path, clock=lambda: 3000.0)
    assert sel_out["unscored_pools"] == 0
    selections = S.load_selections(rr, seal)
    selectors = {s["selector_id"] for s in selections["selections"].values()}
    assert selectors == {T.SELECTOR_VOTE, T.SELECTOR_JUDGE_BEST}
    assert all(s["sealed_at"] == 3000.0 for s in selections["selections"].values())

    # --- evaluation wave --------------------------------------------------------------
    bcb_cell = W.make_cell(T.Method.BANK, [t.source_id for t in tasks if t.domain is T.Domain.BCB], kind=T.CellKind.EVAL_BCB, seal=seal, max_inflight=1)
    evaluator = bcb_eval.BcbEvaluator(tmp_path / "none.sif", timeout_s=20.0, container="none", work_root=tmp_path / "bcbwork")
    assert h.run(hle_cell) == T.EXIT_DONE
    assert h.run(bcb_cell, bcb_evaluator=evaluator) == T.EXIT_DONE
    hle_rec = hle_judge.load_hle_eval(rr, ids[0])
    assert hle_rec is not None and hle_rec["answer_format"] == "multipleChoice" and hle_rec["n_distinct_answers"] >= 1
    assert all("letter_correct" in j and j["started_at_by_seal"][seal] > 3000.0 for j in hle_rec["judgements"].values())
    rows = bcb_eval.iter_bcb_eval_rows(rr)
    assert rows and {r["status"] for r in rows} <= {"pass", "fail"} and all(r["seal"] == seal for r in rows)
    passing = [r for r in rows if r["status"] == "pass"]
    assert passing, "the trivial test must pass for the syntactically valid programs"
    # the judge cell is idempotent and re-running an eval cell over a re-sealed item reuses verdicts
    assert h.run(hle_cell) == T.EXIT_DONE and h.run(bcb_cell, bcb_evaluator=evaluator) == T.EXIT_DONE

    # --- aggregate ----------------------------------------------------------------------
    summary = AG.aggregate(rr)
    assert summary["skipped"] == {} and summary["n_rows"]["banks"] == 4
    import pyarrow.parquet as pq

    sel_table = pq.read_table(rr / "tables" / "selections.parquet").to_pylist()
    bank_table = pq.read_table(rr / "tables" / "banks.parquet").to_pylist()
    assert {r["selector_id"] for r in sel_table} == {"VOTE", "JUDGE_BEST"}
    assert {r["method"] for r in sel_table} == {"BANK", "S_FRESH", "IND_VOTE", "DEC"}
    for r in sel_table:
        assert r["selection_gap"] in (0, 1) and 0.0 <= r["candidate_mean"] <= 1.0 and r["stop_reason"] in (None, "BUDGET", "CALL_CAP", "ROUND_CAP", "COMPLETED")
        if r["no_valid_candidate"]:
            assert r["selected_correct"] is False
    # pass@K rows match the reference estimator recomputed from the judged bank (F-4)
    for row in bank_table:
        assert row["n"] == 10
        for k in AG.PASS_KS:
            assert row[f"pass_at_{k}"] == MR.pass_at_k(10, row["c"], k)
    bcb_bank = next(r for r in bank_table if r["domain"] == "bcb")
    assert bcb_bank["c"] == sum(1 for r in rows if r["status"] == "pass" and r["candidate_id"] in _bank_ids(rr, bank, bcb_bank["source_id"]))
    prefix_rows = [r for r in sel_table if r["pool_kind"] == "bank_prefix" and r["prefix_k"] == 1 and r["selector_id"] == "VOTE"]
    assert len(prefix_rows) == 4 and all(r["planned_count"] == 1 for r in prefix_rows)
    assert summary["configs"] and summary["common_prefix_by_module"]["F"] == 0  # 2 items per domain < one 25-block
    assert summary["banks"]["32B|00"]["pass_at_1"]["pooled_equal_weight"] is not None

    # --- P0-3: a selection sealed after the evaluation started is refused ---------------
    reg = json.loads((S.seals_dir(rr, seal) / S.SELECTIONS_FILE).read_text())
    sid = next(iter(reg["selections"]))
    reg["selections"][sid]["sealed_at"] = 10 ** 12
    (S.seals_dir(rr, seal) / S.SELECTIONS_FILE).write_text(json.dumps(reg))
    with pytest.raises(AG.JoinRefused):
        AG.aggregate(rr)
    lenient = AG.aggregate(rr, strict=False)
    assert lenient["skipped"]["join_refused"] == 1

    # --- reconcile ----------------------------------------------------------------------
    report = RC.reconcile(rr, [cells_path, select_path])
    grp = report["manifests"]["cells_1_32B.json"]
    assert grp["F|32B"]["completed_items"] == 4 and grp["F|32B"]["cells_with_meta"] == 1 and grp["A|32B"]["planned_items"] == 8
    assert report["judge"]["judged_answers"] >= 1 and report["vote"]["vote_selections"] >= 24


def _bank_ids(rr: Path, bank: T.CellSpec, sid: str) -> set[str]:
    item = json.loads(CellPaths.of(rr, bank.cell_id).item(sid).read_text())
    return {b["candidate_id"] for b in item["bank"]}


def test_seal_immutability_and_vote_crosscheck(world, study_config):
    rr, h, tasks = world["run_root"], world["harness"], world["tasks"]
    ids = _ids(tasks)[:2]
    bank = W.make_cell(T.Method.BANK, ids, module="F", N=1, B=0)
    assert h.run(bank) == T.EXIT_DONE
    cells_path, seal = W.write_cells_manifest(rr, "cells_b.json", [bank])
    S.seal_pools(rr, cells_path)
    # tampering with an item file after sealing is detected at the next seal
    path = CellPaths.of(rr, bank.cell_id).item(ids[0])
    item = json.loads(path.read_text())
    item["bank"] = list(reversed(item["bank"]))
    path.write_text(json.dumps(item))
    with pytest.raises(T.ProtocolError):
        S.seal_pools(rr, cells_path)
    with pytest.raises(T.ProtocolError):
        S.load_selections(rr, seal)
    with pytest.raises(T.ProtocolError):
        S.assert_item_sealed({"selections": {}}, ids[0])


def test_common_prefix_rule():
    cfgs = {"a": {"hle": set(range(50)), "bcb": set(range(30))}, "b": {"hle": set(range(26)), "bcb": set(range(50))}}
    assert AG.common_prefix(cfgs) == 25
    assert AG.common_prefix({"a": {"hle": {0, 1, 3}, "bcb": set(range(25))}}) == 0
    assert AG.common_prefix({}) == 0
