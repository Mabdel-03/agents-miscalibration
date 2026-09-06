"""N1 capture CLI pieces that need no GPU: the outcome-blind R2 call selection (phases,
hash rule, bookkeeping), panel cell filtering, sharding, items files, the selection file
written by ``--dry-run``, and the Slurm template placeholders."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agents_scaling.study import identity
from agents_scaling.study.neural import capture as C
from agents_scaling.study.types import CellKind, CellSpec, Framing, Method

REPO = Path(__file__).resolve().parents[2]
SEED = bytes.fromhex("fd5f358b38b68a20d74145171aa9ea5f0bd6f4a11d02d297e6160fa8e49c5a3f")


def _call(role, step, rid, *, cf=None, owner=""):
    return {"request_id": rid, "role": role, "actor_slot": 0, "step": step, "owner": owner, "aliased": False,
            "context_failure": cf, "prompt_tokens": 10, "completion_tokens": 5, "finish_reason": "stop"}


def _episode(calls, eid="ep1"):
    return {"episode_id": eid, "calls": calls}


def _rid(i):
    return f"{i:064x}"


def test_phases_per_method():
    calls = [_call("root", 0, _rid(1)), _call("revise", 1, _rid(2)), _call("revise", 2, _rid(3)), _call("revise", 2, _rid(4))]
    assert [C.phase_of("DEC", c, calls) for c in calls] == ["ROOT", "COORDINATION", "TERMINAL", "TERMINAL"]
    calls = [_call("hub", 0, _rid(1)), _call("worker", 0, _rid(2)), _call("hub", 1, _rid(3)), _call("worker", 1, _rid(4)), _call("hub", 2, _rid(5))]
    assert [C.phase_of("CEN_FLAT", c, calls) for c in calls] == ["ROOT", "ROOT", "COORDINATION", "COORDINATION", "TERMINAL"]
    calls = [_call("hub", 0, _rid(1))]
    assert C.phase_of("CEN_FLAT", calls[0], calls) == "ROOT"  # a single hub call is ROOT, never doubly counted
    calls = [_call("root", 0, _rid(1)), _call("root", 5, _rid(2)), _call("judge", 0, _rid(3))]
    assert [C.phase_of("IND_VOTE", c, calls) for c in calls] == ["ROOT", "ROOT", None]


def test_selection_is_hash_min_and_outcome_blind():
    calls = [_call("root", i, _rid(i)) for i in range(5)] + [_call("root", 9, None, cf="CONTEXT")]
    # one call per group (the non-geometry rule) — force it by emptying the geometry groups
    items, book = C.select_native_calls(SEED, "IND_VOTE", "cell", "hle:1", _episode(calls), geometry_groups=frozenset())
    assert len(items) == 1 and len(book) == 1
    expected = min(range(5), key=lambda i: identity.blind_order_key(SEED, "NEURAL_R2", "ep1", "INDEPENDENT_SOLVER", "ROOT", _rid(i)))
    assert items[0].request_id == _rid(expected) and items[0].group_rank == 0
    assert items[0].group_size == 5 and items[0].group_context_failures == 1 and items[0].declared_role == "INDEPENDENT_SOLVER"
    assert book[0]["eligible"] == 5 and book[0]["context_failures"] == 1 and book[0]["selected"] == _rid(expected)
    assert book[0]["geometry_group"] is False and book[0]["captured"] == [_rid(expected)]
    # permuting the calls or changing lengths/finish reasons changes nothing
    shuffled = list(reversed(calls))
    for c in shuffled:
        c["completion_tokens"] = 9999
        c["finish_reason"] = "length"
    again, _ = C.select_native_calls(SEED, "IND_VOTE", "cell", "hle:1", _episode(shuffled), geometry_groups=frozenset())
    assert again[0].request_id == items[0].request_id
    # a different seed gives a (generally) different pick; same seed → deterministic
    other, _ = C.select_native_calls(b"\x01" * 32, "IND_VOTE", "cell", "hle:1", _episode(calls), geometry_groups=frozenset())
    assert other[0].selection_key != items[0].selection_key


def test_geometry_groups_keep_every_eligible_call_in_hmac_order():
    """P1-1 regression: R3 needs s = 5 comparable IND roots / DEC terminal members, so the
    geometry groups capture every eligible call, ranked by the same outcome-blind HMAC."""
    calls = [_call("root", i, _rid(i)) for i in range(5)] + [_call("root", 9, None, cf="CONTEXT")]
    items, book = C.select_native_calls(SEED, "IND_VOTE", "cell", "hle:1", _episode(calls))
    assert ("INDEPENDENT_SOLVER", "ROOT") in C.GEOMETRY_GROUPS and ("DECENTRALIZED_MEMBER", "TERMINAL") in C.GEOMETRY_GROUPS
    assert len(items) == 5 and [w.group_rank for w in items] == [0, 1, 2, 3, 4]
    keys = [identity.blind_order_key(SEED, "NEURAL_R2", "ep1", "INDEPENDENT_SOLVER", "ROOT", w.request_id).hex() for w in items]
    assert keys == sorted(keys) and [w.selection_key for w in items] == keys
    expected = min(range(5), key=lambda i: identity.blind_order_key(SEED, "NEURAL_R2", "ep1", "INDEPENDENT_SOLVER", "ROOT", _rid(i)))
    assert items[0].request_id == _rid(expected) == book[0]["selected"]
    assert book[0]["geometry_group"] is True and sorted(book[0]["captured"]) == sorted(_rid(i) for i in range(5))
    assert all(w.group_size == 5 and w.group_context_failures == 1 for w in items)
    # order is outcome-blind: reversing the calls and changing their lengths reproduces the same ranks
    shuffled = list(reversed(calls))
    for c in shuffled:
        c["completion_tokens"] = 1
    again, _ = C.select_native_calls(SEED, "IND_VOTE", "cell", "hle:1", _episode(shuffled))
    assert [(w.request_id, w.group_rank) for w in again] == [(w.request_id, w.group_rank) for w in items]


def test_dec_and_cen_flat_groups_and_role_labels():
    calls = [_call("root", 0, _rid(i)) for i in range(5)] + [_call("revise", 1, _rid(10 + i)) for i in range(5)] + [_call("revise", 2, _rid(20 + i)) for i in range(5)]
    items, book = C.select_native_calls(SEED, "DEC", "cell", "bcb:1", _episode(calls))
    # COORDINATION and ROOT keep one call; TERMINAL is a geometry group and keeps all five members
    assert [(i.declared_role, i.phase, i.group_size, i.group_rank) for i in items] == [
        ("DECENTRALIZED_MEMBER", "COORDINATION", 5, 0), ("DECENTRALIZED_MEMBER", "ROOT", 5, 0),
        *[("DECENTRALIZED_MEMBER", "TERMINAL", 5, r) for r in range(5)]]
    assert [b["geometry_group"] for b in book] == [False, False, True]
    calls = [_call("hub", 0, _rid(1)), _call("worker", 0, _rid(2)), _call("worker", 0, _rid(3)), _call("hub", 1, _rid(4), owner="final")]
    items, _ = C.select_native_calls(SEED, "CEN_FLAT", "cell", "bcb:2", _episode(calls))
    assert [(i.declared_role, i.phase, i.native_role) for i in items] == [
        ("CENTRAL_HUB", "ROOT", "hub"), ("CENTRAL_HUB", "TERMINAL", "hub"), ("CENTRAL_WORKER", "ROOT", "worker"), ("CENTRAL_WORKER", "ROOT", "worker")]
    # a group whose calls all failed produces bookkeeping but no work item
    calls = [_call("root", 0, None, cf="CONTEXT")]
    items, book = C.select_native_calls(SEED, "IND_VOTE", "cell", "hle:3", _episode(calls))
    assert items == [] and book[0]["selected"] is None and book[0]["context_failures"] == 1


def _cell(module, method, N=5, B=4, checkpoint="32B", shard=0, items=("hle:a", "bcb:a"), split="main", rep=0, degree=None):
    cid = f"{module}.{method.value}.{checkpoint}.N{N}.B{B}.F00.e{rep}{'.d' + str(degree) if degree else ''}.s{shard:03d}"
    return CellSpec(cell_id=cid, kind=CellKind.GENERATE, module=module, method=method, checkpoint=checkpoint, N=N, B=B,
                    framing=Framing.F00, episode_rep=rep, split=split, items=tuple(items), depends_on=(), max_inflight=1,
                    parallel_items=1, lane=checkpoint, degree=degree)


def test_panel_cells_filter_and_order():
    cells = [
        _cell("N", Method.IND_VOTE, shard=1), _cell("A", Method.IND_VOTE, shard=0), _cell("N", Method.IND_VOTE, N=9),
        _cell("A", Method.DEC, B=2), _cell("A", Method.CEN_FLAT, checkpoint="14B"), _cell("A", Method.DEGREE, N=1, degree=3),
        _cell("A", Method.DEC, rep=1), _cell("A", Method.S_FRESH), _cell("A", Method.CEN_FLAT, split="dev"), _cell("A", Method.CEN_FLAT),
    ]
    chosen = C.panel_cells(cells, checkpoint="32B", N=5, B=4, methods=C.R2_METHODS, modules=C.R2_MODULES)
    assert [c.cell_id for c in chosen] == ["A.IND_VOTE.32B.N5.B4.F00.e0.s000", "A.CEN_FLAT.32B.N5.B4.F00.e0.s000", "N.IND_VOTE.32B.N5.B4.F00.e0.s001"]


def test_native_work_list_reads_item_files_and_dedupes(tmp_path: Path):
    from agents_scaling.study.config import load_config

    cfg = load_config()
    run_root = tmp_path
    cells = [_cell("A", Method.IND_VOTE, items=("hle:a", "bcb:a", "hle:b")), _cell("N", Method.IND_VOTE, items=("hle:a",))]
    for cell in cells:
        for sid in cell.items:
            path = run_root / "cells" / cell.cell_id / "items" / f"{sid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            status = "complete" if sid != "hle:b" else "incomplete"
            ep = _episode([_call("root", i, _rid(i)) for i in range(3)], eid=f"ep-{sid}") if status == "complete" else None
            path.write_text(json.dumps({"source_id": sid, "cell": cell.to_dict(), "status": status, "episode": ep}))
    work, meta = C.native_work_list(run_root, cfg, cells, checkpoint="32B", panel_items=300, methods=C.R2_METHODS, modules=C.R2_MODULES)
    # IND roots are a geometry group: all three eligible roots of each episode are captured
    assert [(w.source_id, w.cell_id.split(".")[0]) for w in work] == [("hle:a", "A")] * 3 + [("bcb:a", "A")] * 3
    assert [w.group_rank for w in work] == [0, 1, 2, 0, 1, 2]
    assert meta["counts"] == {"cells": 2, "episodes": 2, "missing_item_files": 0, "incomplete": 1, "skipped_duplicates": 1, "outside_panel": 0}
    assert meta["items_per_method_domain"] == {"IND_VOTE": {"hle": 1, "bcb": 1}}
    assert len(meta["groups"]) == 2 and meta["work_sha256"]
    assert meta["panel_rule"].startswith("cell_order_fallback") and meta["panel"] is None
    assert meta["panel_resolved"] == {"IND_VOTE": {"hle": ["hle:a"], "bcb": ["bcb:a"]}}
    with pytest.raises(FileNotFoundError):
        C.native_work_list(run_root, cfg, cells, checkpoint="32B", panel_items=300, methods=C.R2_METHODS, modules=C.R2_MODULES, allow_panel_fallback=False)
    # per-domain cap (fallback rule)
    work2, _ = C.native_work_list(run_root, cfg, cells, checkpoint="32B", panel_items=2, methods=C.R2_METHODS, modules=C.R2_MODULES)
    assert sorted({w.source_id for w in work2}) == ["bcb:a", "hle:a"]
    work3, meta3 = C.native_work_list(run_root, cfg, cells, checkpoint="32B", panel_items=1, methods=C.R2_METHODS, modules=C.R2_MODULES)
    assert {w.source_id for w in work3} == {"bcb:a", "hle:a"} and meta3["per_domain"] == 1  # half per superdomain, never below one item


def test_native_work_list_panel_is_the_public_rank_prefix(tmp_path: Path):
    """P1-4 regression: with a public export the panel is ``rank < per_domain`` on ``main``
    (the N2/N3 rule), not the first items in cell order."""
    from agents_scaling.study import types as T
    from agents_scaling.study.config import load_config
    from agents_scaling.study.data.public import public_tasks_path
    from agents_scaling.study.neural import panel as P

    cfg = load_config()
    run_root = tmp_path
    # cell order puts the high-rank item first; the rank rule must skip it
    tasks = [T.PublicTask("hle:late", T.Domain.HLE, "main", "q", "exactMatch", "Gold", 7, 1, None, "math"),
             T.PublicTask("hle:early", T.Domain.HLE, "main", "q", "exactMatch", "Gold", 0, 1, None, "math"),
             T.PublicTask("bcb:z", T.Domain.BCB, "main", "q", "code", "bcb", 0, 1, "task_func", None),
             T.PublicTask("bcb:dev", T.Domain.BCB, "dev", "q", "code", "bcb", 0, 1, "task_func", None)]
    pub = public_tasks_path(run_root)
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))
    assert P.panel_source_ids(run_root, 1) == {"bcb": ["bcb:z"], "hle": ["hle:early"]}
    assert P.panel_source_ids(run_root, 8) == {"bcb": ["bcb:z"], "hle": ["hle:early", "hle:late"]}
    assert P.flatten_panel(P.panel_source_ids(run_root, 8)) == ["bcb:z", "hle:early", "hle:late"]
    cells = [_cell("A", Method.IND_VOTE, items=("hle:late", "hle:early", "bcb:z", "bcb:dev"))]
    for cell in cells:
        for sid in cell.items:
            path = run_root / "cells" / cell.cell_id / "items" / f"{sid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"source_id": sid, "cell": cell.to_dict(), "status": "complete", "episode": _episode([_call("root", 0, _rid(1))], eid=f"ep-{sid}")}))
    work, meta = C.native_work_list(run_root, cfg, cells, checkpoint="32B", panel_items=2, methods=C.R2_METHODS, modules=C.R2_MODULES)
    assert [w.source_id for w in work] == ["hle:early", "bcb:z"]
    assert meta["panel_rule"] == P.PANEL_RULE and meta["panel"] == {"bcb": ["bcb:z"], "hle": ["hle:early"]}
    assert meta["counts"]["outside_panel"] == 2 and meta["panel_resolved"] == {"IND_VOTE": {"hle": ["hle:early"], "bcb": ["bcb:z"]}}


def test_shard_and_items_file(tmp_path: Path):
    items = list(range(10))
    parts = [C.shard_of(items, k, 3) for k in range(3)]
    assert sorted(x for p in parts for x in p) == items and [len(p) for p in parts] == [4, 3, 3]
    with pytest.raises(ValueError):
        C.shard_of(items, 3, 3)
    f = tmp_path / "items.jsonl"
    f.write_text(json.dumps({"request_id": _rid(1), "source_id": "hle:x", "native_role": "hub", "method": "CEN_FLAT"}) + "\n" + json.dumps(_rid(2)) + "\n")
    work = C.load_items_file(f, "native")
    assert [w.request_id for w in work] == [_rid(1), _rid(2)] and work[0].native_role == "hub" and work[1].native_role == "root"
    g = tmp_path / "reports.json"
    g.write_text(json.dumps([{"path": "/x/hle_1.IND_VOTE.json", "report_id": "r1", "method": "IND_VOTE"}, "/x/y.json"]))
    reps = C.load_items_file(g, "report")
    assert [(r.report_id, r.method) for r in reps] == [("r1", "IND_VOTE"), ("y", None)]


def test_report_work_list_reads_reports_dir(tmp_path: Path):
    d = tmp_path / "forecast" / "reports"
    d.mkdir(parents=True)
    (d / "hle_1.DEC.json").write_text(json.dumps({"report_id": "rep-1", "source_id": "hle:1", "method": "DEC", "messages": [], "prompt_token_ids": [1]}))
    (d / "bcb_1.DEC.json").write_text(json.dumps({"messages": [], "prompt_token_ids": [1]}))
    work = C.report_work_list(tmp_path)
    assert [(w.report_id, w.source_id) for w in work] == [("bcb_1.DEC", None), ("rep-1", "hle:1")]
    assert C.report_work_list(tmp_path / "nowhere") == []


def test_dry_run_writes_selection_without_a_model(tmp_path: Path):
    run_root = tmp_path / "results" / "study_v4"
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"num_hidden_layers": 64, "hidden_size": 5120}))
    items = tmp_path / "items.json"
    items.write_text(json.dumps([{"request_id": _rid(i), "source_id": f"hle:{i}", "method": "DEC", "native_role": "revise"} for i in range(5)]))
    rc = C.main(["--run-id", "study_v4", "--stage", "native", "--checkpoint", "32B", "--shard", "1", "--num-shards", "2",
                 "--results-root", str(tmp_path / "results"), "--snapshot-path", str(snapshot), "--items-file", str(items), "--dry-run"])
    assert rc == 0
    sel = json.loads((run_root / "neural" / "native" / "selection.native.s001of002.json").read_text())
    assert sel["blocks"] == [15, 31, 47] and sel["generated_ks"] == [32, 128, 512]
    assert sel["total_work_items"] == 5 and [w["request_id"] for w in sel["shard_items"]] == [_rid(1), _rid(3)]
    assert sel["model_revision"] == "9216db5781bf21249d130ec9da846c4624c16137" and sel["hook_convention"]
    rc = C.main(["--run-id", "study_v4", "--stage", "native", "--checkpoint", "32B", "--blocks", "3,1", "--results-root", str(tmp_path / "results"),
                 "--snapshot-path", str(snapshot), "--items-file", str(items), "--dry-run", "--num-shards", "1"])
    assert rc == 0
    sel = json.loads((run_root / "neural" / "native" / "selection.native.s000of001.json").read_text())
    assert sel["blocks"] == [1, 3] and len(sel["shard_items"]) == 5


def test_slurm_template_placeholders():
    text = (REPO / "slurm" / "study_neural.sbatch.tmpl").read_text()
    placeholders = set(re.findall(r"(?<!\$)\{([A-Z_]+)\}", text))  # ${SLURM_JOB_ID} is a shell variable
    assert placeholders == {"RUN_ID", "STAGE", "SHARD", "NUM_SHARDS", "PARTITION", "GPUS", "LOG_DIR", "REPO", "HF_HOME", "PYTHON", "CHECKPOINT", "ARGS"}
    assert "--gres=gpu:a100:{GPUS}" in text and "--cpus-per-task=8" in text and "--mem=120G" in text and "--time=12:00:00" in text
    assert "/tmp/asys_neural_${SLURM_JOB_ID}" in text and "trap 'rm -rf \"$ASYS_LOCAL_CACHE\"' EXIT" in text
    assert "HF_HUB_OFFLINE=1" in text and "agents_scaling.study.neural.capture" in text
    rendered = text
    for key, value in {"RUN_ID": "study_v4", "STAGE": "native", "SHARD": "0", "NUM_SHARDS": "12", "PARTITION": "pi_manoli", "GPUS": "2",
                       "LOG_DIR": "/l", "REPO": "/r", "HF_HOME": "/h", "PYTHON": "/p", "CHECKPOINT": "32B", "ARGS": "--blocks 15,31,47"}.items():
        rendered = rendered.replace("{" + key + "}", value)
    assert not re.search(r"(?<!\$)\{[A-Z_]+\}", rendered)
    assert "${SLURM_JOB_ID}" in rendered  # shell variables survive the placeholder pass
