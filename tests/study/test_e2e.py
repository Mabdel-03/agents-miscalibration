"""End-to-end study pipeline on the fake vLLM server (integration of WP1-WP5).

Drives the real entry points in the order the ops runbook uses them, on a synthetic
four-item dev export (2 HLE + 2 BCB) inside a temporary run root:

1. data export (synthetic rows in WP1's layout: ``data/public/tasks.jsonl`` and the 0700/0600
   protected files);
2. ``python -m agents_scaling.study.resources.profile`` -> ``FROZEN.yaml`` (B0 from the FLOP
   oracle over the pinned ``config.json`` files and Table E; the real preflight report's root
   wrapper when it exists; spec §6.5, P1-1);
3. ``python -m agents_scaling.study.cells --tier pilot --lane 32B`` (the pilot builder narrowed
   to the 4 dev items: F00/F11/F01/F10 banks, then S_HISTORY, DEC, CEN_FLAT, IND_VOTE, S_FRESH
   at N=5 / B4; P0-4 ordering, F before A);
4. ``runner.run_cell`` over every cell against the fake endpoint, with the P0-1 alias table
   checked from each cell's ``meta.json`` (S_FRESH draws 0..9 == F00, IND-11 roots == F11,
   main DEC NATIVE roots alias nothing, S_HISTORY draw 0 == F00[0], CEN never aliases);
5. ``seal --kind pools`` -> ``cells --tier pilot-select`` -> JUDGE_BEST cell (companion cost,
   outside B) -> ``seal --kind selections`` -> ``cells --tier pilot-eval`` (JUDGE_HLE on the 32B
   lane, EVAL_BCB on the eval lane in in-process container mode) -> ``aggregate`` ->
   ``reconcile`` (P0-3 wave order; §3.6 seals before any correctness join);
6. resumability: a cell whose item files vanished after its requests were committed is
   rebuilt from the request store with zero new HTTP calls (§10.4, T9), and a finished cell is
   a no-op.

pass@K columns of ``banks.parquet`` are re-derived with ``metrics_reference.pass_at_k`` from the
evaluator verdicts (§5.6, F-4).  Phase timings are written to ``<run_root>/e2e_timing.json``
and printed (``pytest -s``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from agents_scaling.study import aggregate as AG
from agents_scaling.study import cells as cells_mod
from agents_scaling.study import metrics_reference as MR
from agents_scaling.study import reconcile as RC
from agents_scaling.study import types as T
from agents_scaling.study.config import FROZEN_FILENAME
from agents_scaling.study.evaluation import bcb as bcb_eval
from agents_scaling.study.evaluation import hle_judge
from agents_scaling.study.prompts import load_template
from agents_scaling.study.resources import profile as P
from agents_scaling.study.resources.oracle import arch_config_path
from agents_scaling.study.runner import CellPaths, run_cell
from agents_scaling.study.selection import seal as S
from tests.study import wp5_support as W

RUN_ID = "study_v4_e2e"
N_PER_DOMAIN = 2
#: The fake solver's per-item answer pool: one passing program, one MC letter, one failing program.
POOL: tuple[str, ...] = ("x = 1\n", "B", "x = 2\n")
#: BCB test: the driver runs ``code + "\\n\\n" + test`` as one module, so ``x`` is visible.
X_IS_ONE_TEST = "import unittest\n\n\nclass TestX(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(x, 1)\n"
REAL_PREFLIGHT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/data/preflight_report.json")
GENERATE_MANIFEST = "cells_pilot_32B.json"
SELECT_MANIFEST = "cells_pilot-select_32B.json"
EVAL_32B_MANIFEST = "cells_pilot-eval_32B.json"
EVAL_LANE_MANIFEST = "cells_pilot-eval_eval.json"


def _hub_delegates_until_final(text: str) -> str:
    """CEN_FLAT hub: delegate on every cycle; answer only under the frozen final instruction."""
    return "final" if load_template("cen_final_instruction").strip() in text else "delegate"


class Phases:
    """Wall-clock per phase (reported, never asserted)."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}
        self._start = time.monotonic()
        self._last = self._start

    def mark(self, name: str) -> None:
        now = time.monotonic()
        self.durations[name] = round(now - self._last, 3)
        self._last = now

    @property
    def total(self) -> float:
        return round(time.monotonic() - self._start, 3)


@pytest.fixture
def world(tmp_path: Path, study_config, monkeypatch):
    for ckpt in study_config.checkpoints.values():
        if not arch_config_path(ckpt).exists():
            pytest.skip(f"pinned config.json for {ckpt.size} absent from the HF cache (FLOP oracle / profiler need it)")
    results_root = tmp_path
    run_root = results_root / RUN_ID
    for name in ("servers", "cells", "requests", "eval", "seals", "freeze"):
        (run_root / name).mkdir(parents=True)
    tasks = W.make_tasks(N_PER_DOMAIN, "dev")
    W.write_export(run_root, tasks, test_src=X_IS_ONE_TEST)
    # The pilot tier is defined at 20 dev items (cells.PILOT_ITEMS); the e2e runs the same
    # builder on a 4-item dev split (PILOT_S_HISTORY_ITEMS == 4 keeps S_HISTORY on every item).
    monkeypatch.setattr(cells_mod, "PILOT_ITEMS", 2 * N_PER_DOMAIN)
    with W.fake_server(run_root, answer_pool=POOL, hub_mode=_hub_delegates_until_final, delegate_count=4) as srv:
        yield {"results_root": results_root, "run_root": run_root, "tasks": tasks, "server": srv, "cfg": study_config}


def _cli(world: dict[str, Any]) -> list[str]:
    return ["--run-id", RUN_ID, "--results-root", str(world["results_root"])]


def _run(world: dict[str, Any], cell: T.CellSpec, **kwargs: Any) -> int:
    """``run_cell`` with the fake endpoint's tokenizer; the oracle is the runner's default
    (``FlopOracle.from_checkpoint``), whose hash the runner verifies against FROZEN.yaml."""
    rr = world["run_root"]
    return run_cell(cell, rr, rr, 0, None, cfg=world["cfg"], tokenizer=world["server"].tokenizer, wait_timeout_s=5.0, run_id=RUN_ID, **kwargs)


def _meta(run_root: Path, cell: T.CellSpec) -> dict[str, Any]:
    return json.loads(CellPaths.of(run_root, cell.cell_id).meta.read_text(encoding="utf-8"))


def _item(run_root: Path, cell: T.CellSpec, sid: str) -> dict[str, Any]:
    return json.loads(CellPaths.of(run_root, cell.cell_id).item(sid).read_text(encoding="utf-8"))


def _build(world: dict[str, Any], tier: str, lane: str, out: str, seal: str | None = None) -> tuple[Path, list[T.CellSpec], str]:
    argv = _cli(world) + ["--tier", tier, "--lane", lane, "--out", out]
    if seal is not None:
        argv += ["--seal", seal]
    assert cells_mod.main(argv) == 0
    path = world["run_root"] / out
    assert path.exists() and path.with_name(out + ".sha256").exists()
    return path, cells_mod.load_cells_file(path), cells_mod.cells_file_sha256(path)


def test_pipeline_end_to_end(world, study_config):
    rr: Path = world["run_root"]
    srv = world["server"]
    tasks: list[T.PublicTask] = world["tasks"]
    ids = [t.source_id for t in tasks]
    hle_ids = [t.source_id for t in tasks if t.domain is T.Domain.HLE]
    bcb_ids = [t.source_id for t in tasks if t.domain is T.Domain.BCB]
    n_items = len(ids)
    phases = Phases()

    # ---- 1. freeze: B0 from the FLOP oracle (profile.py CLI) -------------------------------
    argv = _cli(world) + ["--code-version", "e2e"]
    if REAL_PREFLIGHT.exists():
        argv += ["--wrapper-tokens-json", str(REAL_PREFLIGHT)]
    assert P.main(argv) == 0
    frozen = yaml.safe_load((rr / FROZEN_FILENAME).read_text(encoding="utf-8"))
    b0 = int(frozen["budget"]["B0_flops"])
    profile = frozen["extra"]["profile"]
    assert b0 > 0 and profile["B0_flops"] == b0 and profile["B0_binding_method"] in {m.value for m in T.Method}
    assert profile["budgets"] == {f"B{m}": m * b0 for m in study_config.budget.budget_multipliers}
    assert profile["wrapper_source"]["root"] == ("preflight" if REAL_PREFLIGHT.exists() else "measured")
    assert set(profile["oracle_tables"]) == set(study_config.checkpoints)
    assert study_config.frozen(rr).b0_flops == float(b0)
    phases.mark("profile_freeze")

    # ---- 2. cells --tier pilot --lane 32B ---------------------------------------------------
    cells_path, cells, seal = _build(world, "pilot", "32B", GENERATE_MANIFEST)
    f_cells = [c for c in cells if c.module == "F"]
    a_cells = [c for c in cells if c.module == "A"]
    assert len(cells) == len(f_cells) + len(a_cells) == 9
    assert [c.framing for c in f_cells] == list(cells_mod.BANK_FRAMINGS) and all(c.method is T.Method.BANK and c.N == 1 and c.B == 0 for c in f_cells)
    assert [c.method for c in a_cells] == list(cells_mod.A_METHOD_ORDER)
    assert all(c.N == 5 and c.B == study_config.budget.primary and c.split == "dev" and c.checkpoint == "32B" for c in a_cells)
    assert all(set(c.items) == set(ids) for c in cells)
    by_method = {c.method: c for c in a_cells}
    assert by_method[T.Method.DEC].framing is T.Framing.NATIVE and by_method[T.Method.CEN_FLAT].framing is T.Framing.NATIVE
    assert by_method[T.Method.IND_VOTE].framing is T.Framing.F11
    assert by_method[T.Method.S_FRESH].framing is T.Framing.F00 and by_method[T.Method.S_HISTORY].framing is T.Framing.F00
    assert all(set(c.depends_on) == {f.cell_id for f in f_cells} for c in a_cells)
    cells_mod.assert_f_before_a(cells)
    phases.mark("cells_pilot")

    # ---- 3. F banks: 4 framings x 10 draws per item -------------------------------------------
    for cell in f_cells:
        assert _run(world, cell) == T.EXIT_DONE, cell.cell_id
    assert srv.chat_count == len(f_cells) * cells_mod.R_BANK * n_items
    for cell in f_cells:
        meta = _meta(rr, cell)
        assert meta["n_generated_requests"] == cells_mod.R_BANK * n_items and meta["n_aliased_requests"] == 0
        for sid in ids:
            item = _item(rr, cell, sid)
            assert item["status"] == "complete" and item["episode"] is None and len(item["bank"]) == cells_mod.R_BANK
            assert all(b["valid"] and b["vote_key"] is not None for b in item["bank"])
    phases.mark("run_F")

    # ---- 4. A cells at N=5 / B4 in manifest order; alias table (P0-1) -------------------------
    for cell in a_cells:
        before = srv.chat_count
        assert _run(world, cell) == T.EXIT_DONE, cell.cell_id
        assert srv.chat_count > before, f"{cell.cell_id} made no HTTP call"
    aliased = {m: _meta(rr, c)["n_aliased_requests"] for m, c in by_method.items()}
    assert aliased[T.Method.S_FRESH] == cells_mod.R_BANK * n_items  # draws 0..9 == F00[0..9]
    assert aliased[T.Method.IND_VOTE] >= 5 * n_items  # module-A IND-11 roots == F11[0..4] (+ optional draws)
    assert aliased[T.Method.DEC] == 0  # main-tier truthful NATIVE roots alias nothing
    assert aliased[T.Method.CEN_FLAT] == 0
    assert aliased[T.Method.S_HISTORY] == n_items  # draw 0 == F00[0]
    expected_stop = {
        T.Method.S_FRESH: {"CALL_CAP", "BUDGET"},
        T.Method.IND_VOTE: {"CALL_CAP", "BUDGET"},
        T.Method.S_HISTORY: {"CALL_CAP", "BUDGET"},
        T.Method.DEC: {"ROUND_CAP", "BUDGET"},
        T.Method.CEN_FLAT: {"CYCLE_CAP"},
    }
    for cell in a_cells:
        for sid in ids:
            item = _item(rr, cell, sid)
            ep = item["episode"]
            assert item["status"] == "complete" and ep["N"] == 5 and ep["method"] == cell.method.value
            assert ep["stop_reason"] in expected_stop[cell.method], (cell.cell_id, sid, ep["stop_reason"])
            ledger = ep["ledger"]
            assert ledger["B_flops"] == int(round(study_config.budget.primary * float(b0)))
            assert 0 < ledger["spent"] <= ledger["B_flops"] and ledger["slack"] >= 0
            assert ledger["calls_admitted"] <= study_config.caps.solver_calls
            if cell.method is T.Method.CEN_FLAT:
                assert ep["native_final"]["valid"] is True and ep["counters"]["calls_by_role"] == {"hub": 9, "worker": 32}
            else:
                assert ep["selection"] is not None and ep["selection"]["selector_id"] == T.SELECTOR_VOTE
                assert ep["selection"]["selected_candidate_id"] is not None
            if cell.method is T.Method.DEC:
                assert ep["counters"]["calls_by_role"]["root"] == 5 and len(ep["rounds"]) >= 1
                assert len(ep["latest_candidate_ids"]) == 5
    phases.mark("run_A")

    # ---- 5. seal pools (CLI) -> pilot-select -> JUDGE_BEST -> seal selections ------------------
    assert S.main(_cli(world) + ["--cells-file", GENERATE_MANIFEST, "--kind", "pools"]) == 0
    pools = S.load_pools(rr, seal)
    assert pools["n_items"] == n_items and pools["incomplete"] == []
    kinds = {(p["method"], p["pool_kind"], p["prefix_k"]) for p in pools["pools"].values()}
    assert {("BANK", S.POOL_BANK_PREFIX, k) for k in S.PREFIXES} <= kinds
    assert {("S_FRESH", S.POOL_ARCHIVE, None), ("IND_VOTE", S.POOL_ARCHIVE, None), ("S_HISTORY", S.POOL_ARCHIVE, None),
            ("DEC", S.POOL_LATEST, None), ("DEC", S.POOL_ARCHIVE, None), ("CEN_FLAT", S.POOL_NATIVE, None)} <= kinds
    phases.mark("seal_pools")

    select_path, select_cells, _ = _build(world, "pilot-select", "32B", SELECT_MANIFEST, seal=seal)
    assert len(select_cells) == 1 and select_cells[0].kind is T.CellKind.JUDGE_BEST and set(select_cells[0].items) == set(ids)
    assert cells_mod.cell_seal(select_cells[0]) == seal and select_cells[0].lane == study_config.judge_checkpoint
    # T18: evaluation refuses before the selections are sealed
    eval32_path, eval32_cells, _ = _build(world, "pilot-eval", "32B", EVAL_32B_MANIFEST, seal=seal)
    assert [c.kind for c in eval32_cells] == [T.CellKind.JUDGE_HLE] and set(eval32_cells[0].items) == set(hle_ids)
    assert _run(world, eval32_cells[0]) == T.EXIT_SUSPENDED
    suspended = CellPaths.of(rr, eval32_cells[0].cell_id).suspended
    assert "not sealed" in json.loads(suspended.read_text())["error"]
    suspended.unlink()

    before = srv.chat_count
    assert _run(world, select_cells[0]) == T.EXIT_DONE
    jb_calls = srv.chat_count - before
    for sid in ids:
        jb = _item(rr, select_cells[0], sid)
        valid_ids = {cid for p in S.pools_of_item(pools, sid) for cid, ok in p["valid"].items() if ok}
        assert jb["kind"] == "JUDGE_BEST" and set(jb["scores"]) == valid_ids and jb["companion_cost"]["calls"] >= 1
        assert all(s is None or 0.0 <= s <= 1.0 for s in jb["scores"].values())
    assert jb_calls == _meta(rr, select_cells[0])["n_generated_requests"] > 0
    phases.mark("judge_best")

    assert S.main(_cli(world) + ["--cells-file", GENERATE_MANIFEST, "--kind", "selections", "--select-cells-file", SELECT_MANIFEST]) == 0
    selections = S.load_selections(rr, seal)
    assert selections["pools_without_judge_best"] == 0 and selections["n_items"] == n_items
    assert {s["selector_id"] for s in selections["selections"].values()} == {T.SELECTOR_VOTE, T.SELECTOR_JUDGE_BEST}
    assert len(selections["selections"]) == 2 * len(pools["pools"])
    phases.mark("seal_selections")

    # ---- 6. pilot-eval: JUDGE_HLE (32B lane) + EVAL_BCB (eval lane, in-process) ---------------
    evallane_path, evallane_cells, _ = _build(world, "pilot-eval", "eval", EVAL_LANE_MANIFEST, seal=seal)
    assert [c.kind for c in evallane_cells] == [T.CellKind.EVAL_BCB] and set(evallane_cells[0].items) == set(bcb_ids)
    assert _run(world, eval32_cells[0]) == T.EXIT_DONE
    evaluator = bcb_eval.BcbEvaluator(rr / "none.sif", timeout_s=20.0, container="none", work_root=rr / "eval" / "bcb" / "work")
    assert _run(world, evallane_cells[0], bcb_evaluator=evaluator) == T.EXIT_DONE
    for sid in hle_ids:
        rec = hle_judge.load_hle_eval(rr, sid)
        assert rec is not None and rec["judgements"] and all(seal in j["started_at_by_seal"] for j in rec["judgements"].values())
    bcb_rows = bcb_eval.iter_bcb_eval_rows(rr)
    statuses = {r["status"] for r in bcb_rows}
    assert "pass" in statuses and statuses - {"pass"}, f"expected a pass/fail mix, got {statuses}"
    assert all(r["seal"] == seal and r["source_id"] in bcb_ids for r in bcb_rows)
    phases.mark("evaluate")

    # ---- 7. aggregate: tables + pass@K against the reference estimator ----------------------------
    summary = AG.aggregate(rr)
    tables = rr / AG.TABLES_DIR
    for name in ("selections.parquet", "banks.parquet", "episodes.parquet", "summary.json"):
        assert (tables / name).exists(), name
    assert summary["skipped"] == {} and summary["seals"] == [seal]
    import pyarrow.parquet as pq

    sel_table = pq.read_table(tables / "selections.parquet").to_pylist()
    bank_table = pq.read_table(tables / "banks.parquet").to_pylist()
    ep_table = pq.read_table(tables / "episodes.parquet").to_pylist()
    assert summary["n_rows"] == {"selections": len(sel_table), "banks": len(bank_table), "episodes": len(ep_table)}
    assert len(sel_table) == len(selections["selections"]) and len(ep_table) == len(a_cells) * n_items
    assert {r["method"] for r in sel_table} == {"BANK", "S_FRESH", "IND_VOTE", "S_HISTORY", "DEC", "CEN_FLAT"}
    assert {r["selector_id"] for r in sel_table} == {T.SELECTOR_VOTE, T.SELECTOR_JUDGE_BEST}
    for r in sel_table:
        assert r["selection_gap"] in (0, 1) and 0.0 <= r["candidate_mean"] <= 1.0 and r["selected_correct"] in (True, False)
    assert {r["stop_reason"] for r in ep_table} <= {"CALL_CAP", "BUDGET", "ROUND_CAP", "CYCLE_CAP"}
    b4 = int(round(study_config.budget.primary * float(b0)))
    assert all(r["B_flops"] == b4 and 0 < r["spent_flops"] <= b4 and r["slack"] >= 0 for r in ep_table)
    assert all(r["B_flops"] == b4 for r in sel_table if r["method"] != "BANK") and all(r["B_flops"] is None for r in sel_table if r["method"] == "BANK")
    assert all(r["native_final_correct"] in (True, False) for r in ep_table if r["method"] == "CEN_FLAT")

    assert len(bank_table) == len(f_cells) * n_items
    correctness = AG.Correctness(rr, {t.source_id: t for t in tasks})
    for row in bank_table:
        assert row["n"] == cells_mod.R_BANK
        for k in AG.PASS_KS:
            assert row[f"pass_at_{k}"] == MR.pass_at_k(row["n"], row["c"], k), (row["cell_id"], row["source_id"], k)
        assert row["pass_at_1"] == row["c"] / row["n"] and row["pass_at_10"] == (1.0 if row["c"] else 0.0)
        # c re-derived from the evaluator verdicts of that bank's ten draws
        bank_cell = next(c for c in f_cells if c.cell_id == row["cell_id"])
        records = S.candidate_records_of(_item(rr, bank_cell, row["source_id"]))
        verdicts = correctness.of(row["source_id"], records, seal=seal, sealed_at=0.0)
        assert verdicts is not None and sum(verdicts.values()) == row["c"]
    for key in ("32B|00", "32B|11", "32B|01", "32B|10"):
        assert summary["banks"][key]["n_items"] == n_items and summary["banks"][key]["pass_at_1"]["pooled_equal_weight"] is not None
    assert summary["common_prefix_by_module"] == {"F": 0, "A": 0}  # 2 items per domain < one 25-block
    phases.mark("aggregate")

    # ---- 8. reconcile + resumability (§10.4, T9) ---------------------------------------------------
    report = RC.reconcile(rr, [cells_path, select_path, eval32_path, evallane_path])
    grp = report["manifests"][GENERATE_MANIFEST]
    assert grp["F|32B"]["completed_items"] == len(f_cells) * n_items and grp["F|32B"]["cells_with_meta"] == len(f_cells)
    assert grp["A|32B"]["completed_items"] == len(a_cells) * n_items and grp["A|32B"]["cells_with_meta"] == len(a_cells)
    assert report["judge"]["judged_answers"] >= 1 and report["vote"]["vote_selections"] == len(pools["pools"])
    # a finished cell is a no-op
    before = srv.chat_count
    assert _run(world, by_method[T.Method.DEC]) == T.EXIT_DONE and srv.chat_count == before
    # item files lost after the requests were committed: rebuilt from the store, zero new HTTP calls
    fresh = by_method[T.Method.S_FRESH]
    paths = CellPaths.of(rr, fresh.cell_id)
    old_items = {sid: _item(rr, fresh, sid) for sid in ids}
    for sid in ids:
        paths.item(sid).unlink()
    paths.meta.unlink()
    assert _run(world, fresh) == T.EXIT_DONE and srv.chat_count == before
    meta = _meta(rr, fresh)
    assert meta["n_generated_requests"] == 0 and meta["n_aliased_requests"] == sum(len(old_items[sid]["episode"]["candidates"]) for sid in ids)
    for sid in ids:
        new = _item(rr, fresh, sid)
        assert new["episode"]["selection"] == old_items[sid]["episode"]["selection"]
        assert [c["candidate_id"] for c in new["episode"]["candidates"]] == [c["candidate_id"] for c in old_items[sid]["episode"]["candidates"]]
    assert S.seal_pools(rr, cells_path)["added"] == 0  # the rebuilt item seals byte-identically
    phases.mark("reconcile_resume")

    timing = {"phases": phases.durations, "total_s": phases.total, "http_calls": srv.chat_count, "n_items": n_items}
    (rr / "e2e_timing.json").write_text(json.dumps(timing, indent=1) + "\n", encoding="utf-8")
    print("\n[e2e]", json.dumps(timing))
