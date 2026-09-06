"""cells.py: manifest determinism, tier nesting, engine-seed uniqueness (§3.6), F-before-A
ordering (architecture §2.3), the P0-1 alias-table framings, P0-4 sizing, P0-7 lanes,
select/eval tiers, frozen-manifest overwrite refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import cells as C
from agents_scaling.study import types as T
from agents_scaling.study.types import ProtocolError
from tests.study.wp5_support import make_tasks


@pytest.fixture(scope="module")
def tasks():
    return make_tasks(200, "main") + make_tasks(30, "dev")


@pytest.fixture(scope="module")
def index(tasks):
    return {t.source_id: t for t in tasks}


def test_tier1_deterministic_and_ordered(study_config, tasks, index):
    a = C.build_cells(study_config, tasks, "1", "32B")
    b = C.build_cells(study_config, tasks, "1", "32B")
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]
    assert len({c.cell_id for c in a}) == len(a)
    C.assert_tier_nesting(a, index)
    C.assert_f_before_a(a)
    # block 0: the four banks first (F00, F11, F01, F10), then S_HISTORY, DEC, CEN_FLAT, IND, S_FRESH
    heads = [(c.module, c.method, c.framing) for c in a[:12]]
    assert heads[0] == ("F", T.Method.BANK, T.Framing.F00)
    order = []
    for c in a:
        key = (c.module, c.method.value, c.framing.value)
        if key not in order:
            order.append(key)
    assert order[:4] == [("F", "BANK", "00"), ("F", "BANK", "11"), ("F", "BANK", "01"), ("F", "BANK", "10")]
    assert order[4:9] == [("A", "S_HISTORY", "00"), ("A", "DEC", "nat"), ("A", "CEN_FLAT", "nat"), ("A", "IND_VOTE", "11"), ("A", "S_FRESH", "00")]
    # every shard holds both domains (interleaved blocks)
    for c in a:
        domains = {index[s].domain for s in c.items}
        if len(c.items) >= 2:
            assert domains == {T.Domain.HLE, T.Domain.BCB}, c.cell_id


def test_sizing_and_lanes(study_config, tasks):
    cells = C.build_cells(study_config, tasks, "1", "32B")
    by_method = {}
    for c in cells:
        by_method.setdefault(c.method, c)
    f = by_method[T.Method.BANK]
    assert (len(f.items), f.parallel_items, f.max_inflight, f.B, f.N) == (20, 2, 10, 0, 1)
    assert (len(by_method[T.Method.S_FRESH].items), by_method[T.Method.S_FRESH].parallel_items, by_method[T.Method.S_FRESH].max_inflight) == (10, 1, 8)
    assert (len(by_method[T.Method.S_HISTORY].items), by_method[T.Method.S_HISTORY].parallel_items, by_method[T.Method.S_HISTORY].max_inflight) == (4, 4, 1)
    assert (len(by_method[T.Method.DEC].items), by_method[T.Method.DEC].parallel_items, by_method[T.Method.DEC].max_inflight) == (10, 2, 5)
    assert (len(by_method[T.Method.CEN_FLAT].items), by_method[T.Method.CEN_FLAT].parallel_items) == (5, 2)
    assert all(c.lane == "32B" and c.split == "main" and c.kind is T.CellKind.GENERATE for c in cells)
    assert C.build_cells(study_config, tasks, "1", "14B") == []
    a_cells = [c for c in cells if c.module == "A"]
    assert all(c.depends_on and all(d.startswith("F.BANK") for d in c.depends_on) for c in a_cells)
    assert all(c.B == study_config.budget.primary and c.N == 5 for c in a_cells)


def test_tier2_alias_table_framings(study_config, tasks, index):
    cells = C.build_cells(study_config, tasks, "2", "32B")
    C.assert_tier_nesting(cells, index)
    n_cells = [c for c in cells if c.module == "N"]
    assert {c.N for c in n_cells if c.method in (T.Method.IND_VOTE, T.Method.DEC, T.Method.CEN_FLAT)} == {1, 2, 3, 5, 9}
    assert all(c.framing is T.Framing.F00 for c in n_cells if c.method in (T.Method.IND_VOTE, T.Method.DEC, T.Method.DEC_ONE_ROUND, T.Method.IND_PRIVATE_REVISION))
    assert all(c.framing is T.Framing.NATIVE for c in n_cells if c.method is T.Method.CEN_FLAT)
    assert all(c.B == 4 for c in n_cells)  # P1-2: N=9 only at B4
    d_cells = [c for c in cells if c.module == "D"]
    assert {c.degree for c in d_cells} == {0, 1, 2, 4, 8} and all(".d" in c.cell_id for c in d_cells)
    e_cells = [c for c in cells if c.module == "E"]
    assert {c.episode_rep for c in e_cells} == {1, 2, 3, 4, 5}
    assert {len({s for c in e_cells if c.method == m for s in c.items}) for m in C.E_METHODS} == {30}
    assert all(c.framing is T.Framing.NATIVE for c in e_cells if c.method is T.Method.DEC)
    assert all(c.framing is T.Framing.F11 for c in e_cells if c.method is T.Method.IND_VOTE)
    for size in ("4B", "8B", "14B"):
        m_cells = C.build_cells(study_config, tasks, "2", size)
        assert m_cells and all(c.module == "M" and c.lane == size and c.checkpoint == size for c in m_cells)
        assert len({s for c in m_cells for s in c.items}) == 150
        assert all(c.framing is T.Framing.NATIVE for c in m_cells if c.method is T.Method.DEC)


def test_tier3_budget_and_cross(study_config, tasks):
    cells = C.build_cells(study_config, tasks, "3", "32B")
    b_cells = [c for c in cells if c.module == "B"]
    assert {c.B for c in b_cells} == {1, 2} and len({s for c in b_cells for s in c.items}) == 100
    mx = [c for c in cells if c.module == "MX"]
    assert {(c.N, c.B) for c in mx} == {(3, 4), (3, 1), (9, 4)}
    assert all(c.framing is T.Framing.F00 for c in mx if c.method in (T.Method.IND_VOTE, T.Method.DEC))
    mx8 = C.build_cells(study_config, tasks, "3", "8B")
    assert mx8 and all(c.module == "MX" and c.checkpoint == "8B" for c in mx8)


def test_pilot_uses_dev(study_config, tasks):
    cells = C.build_cells(study_config, tasks, "pilot", "32B")
    assert cells and all(c.split == "dev" for c in cells)
    items = {s for c in cells for s in c.items}
    assert len(items) == 20
    history = [c for c in cells if c.method is T.Method.S_HISTORY]
    assert sum(len(c.items) for c in history) == 4


def test_select_and_eval_tiers(study_config, tasks):
    seal = "ab" * 32
    sealed = [t.source_id for t in tasks if t.split == "main"][:30] + [t.source_id for t in tasks if t.split == "main"][200:230]
    with pytest.raises(C.ManifestError):
        C.build_cells(study_config, tasks, "1-select", "32B")
    sel = C.build_cells(study_config, tasks, "1-select", "32B", seal=seal, sealed_items=sealed)
    assert sel and all(c.kind is T.CellKind.JUDGE_BEST and c.method is C.EVAL_METHOD_PLACEHOLDER and C.cell_seal(c) == seal and c.lane == "32B" for c in sel)
    assert all(C.cell_seal(c) is None for c in C.build_cells(study_config, tasks, "1", "32B"))
    assert all(len(c.items) <= 25 and c.max_inflight == 8 for c in sel)
    assert sum(len(c.items) for c in sel) == 60
    ev32 = C.build_cells(study_config, tasks, "1-eval", "32B", seal=seal, sealed_items=sealed)
    assert ev32 and all(c.kind is T.CellKind.JUDGE_HLE for c in ev32) and sum(len(c.items) for c in ev32) == 30
    ev = C.build_cells(study_config, tasks, "1-eval", "eval", seal=seal, sealed_items=sealed)
    assert ev and all(c.kind is T.CellKind.EVAL_BCB and c.max_inflight == 1 for c in ev) and sum(len(c.items) for c in ev) == 30
    assert "x" + seal[:8] in ev[0].cell_id
    with pytest.raises(C.ManifestError):
        C.build_cells(study_config, tasks, "2b", "32B")


def test_engine_seed_uniqueness_and_collision_detection(study_config, tasks, index):
    cells = C.build_cells(study_config, tasks, "1", "32B")[:6]
    assert C.assert_engine_seed_uniqueness(cells, index, study_config) > 0
    # a CEN_FLAT and a DEGREE cell exercise every namespace of the seed table
    cells2 = [c for c in C.build_cells(study_config, tasks, "2", "32B") if c.method in (T.Method.CEN_FLAT, T.Method.DEGREE)][:2]
    assert C.assert_engine_seed_uniqueness(cells2, index, study_config) > 0
    keys = C.episode_seed_keys(cells2[0], index[cells2[0].items[0]], study_config)
    assert len({tuple(k.as_array()[4:]) for k in keys}) == len(keys)


def test_f_before_a_guard(study_config, tasks):
    cells = C.build_cells(study_config, tasks, "1", "32B")
    swapped = list(reversed(cells))
    with pytest.raises(ProtocolError):
        C.assert_f_before_a(swapped)


def test_nesting_guard(study_config, tasks, index):
    cells = C.build_cells(study_config, tasks, "1", "32B")
    bad = [c for c in cells if c.module == "A"][3:4]  # a lone A shard is not a prefix
    with pytest.raises(ProtocolError):
        C.assert_tier_nesting(bad, index)


def test_write_load_refuses_differing_manifest(tmp_path: Path, study_config, tasks):
    cells = C.build_cells(study_config, tasks, "pilot", "32B")
    path = tmp_path / "cells_pilot_32B.json"
    digest = C.write_cells_file(path, cells, {"tier": "pilot"})
    assert digest == C.write_cells_file(path, cells, {"tier": "pilot"})  # identical rewrite is a no-op
    assert C.cells_file_sha256(path) == digest
    assert [c.to_dict() for c in C.load_cells_file(path)] == [c.to_dict() for c in cells]
    with pytest.raises(ProtocolError):
        C.write_cells_file(path, cells[:-1], {"tier": "pilot"})
    data = json.loads(path.read_text())
    data["cells"][0]["items"] = data["cells"][0]["items"][:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ProtocolError):
        C.load_cells_file(path)


def test_cli_dry_run(tmp_path: Path, study_config, tasks, monkeypatch):
    from tests.study.wp5_support import write_export

    run_root = tmp_path / "study_v4"
    write_export(run_root, tasks)
    rc = C.main(["--run-id", "study_v4", "--results-root", str(tmp_path), "--tier", "1", "--lane", "32B", "--out", "cells_1_32B.json"])
    assert rc == 0 and (run_root / "cells_1_32B.json").exists() and (run_root / "cells_1_32B.json.sha256").exists()
    assert C.main(["--run-id", "study_v4", "--results-root", str(tmp_path), "--tier", "1", "--lane", "32B", "--out", "cells_1_32B.json"]) == 0
    with pytest.raises(ProtocolError):
        C.main(["--run-id", "study_v4", "--results-root", str(tmp_path), "--tier", "pilot", "--lane", "32B", "--out", "cells_1_32B.json"])
