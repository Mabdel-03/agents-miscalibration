"""N2 forecast cell CLI: real IND_VOTE/DEC/CEN_FLAT episodes on the fake vLLM server, sealed
with the real seal code, then ``forecast.run`` in --report-only and full mode (resumable,
sharded, content-addressed requests through the study store)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.forecast import manifest as M
from agents_scaling.study.forecast import run as RUN
from agents_scaling.study.forecast import shadow as S
from agents_scaling.study.forecast.report import compile_report
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.selection import seal as seals
from tests.study import wp5_support as W

POOL = ("x = 1\n", "B", "x = 2\n")
RUN_ID = "n2_run"


@pytest.fixture(scope="module")
def study_config():
    from agents_scaling.study.config import load_config

    return load_config()


@pytest.fixture
def world(tmp_path: Path, study_config):
    results_root = tmp_path
    run_root = results_root / RUN_ID
    for name in ("servers", "cells", "requests", "eval", "seals"):
        (run_root / name).mkdir(parents=True)
    tasks = W.make_tasks(1, "main")  # one HLE (MC) + one BCB
    W.write_export(run_root, tasks)
    with W.fake_server(run_root, answer_pool=POOL, delegate_count=2) as srv:
        R = W.root_reservation(study_config, srv.tokenizer, tasks[0])
        b0 = 3 * R  # B4 = 12 root envelopes: CEN_FLAT's final reserve (Table E hub prompt) must fit inside B
        W.freeze_for_test(run_root, study_config, b0)
        ids = [t.source_id for t in tasks]
        cells = [
            W.make_cell(T.Method.IND_VOTE, ids, framing=T.Framing.F11),
            W.make_cell(T.Method.DEC, ids, framing=T.Framing.NATIVE),
            W.make_cell(T.Method.CEN_FLAT, ids, framing=T.Framing.NATIVE),
        ]
        harness = W.Harness(run_root, study_config, srv)
        for cell in cells:
            assert harness.run(cell) == T.EXIT_DONE, cell.cell_id
        path, seal = W.write_cells_manifest(run_root, "cells_n2.json", cells)
        assert seals.seal_pools(run_root, path)["n_items"] == 2
        assert seals.seal_selections(run_root, path, cfg=study_config)["n_items"] == 2
        items_file = run_root / "panel_items.txt"
        items_file.write_text("# panel\n" + "\n".join(ids) + "\n")
        yield {"results_root": results_root, "run_root": run_root, "tasks": tasks, "server": srv, "cfg": study_config,
               "cells": cells, "seal": seal, "items_file": items_file, "harness": harness}


def _argv(world, *extra: str) -> list[str]:
    return ["--run-id", RUN_ID, "--results-root", str(world["results_root"]), "--seal", world["seal"],
            "--items-file", str(world["items_file"]), "--wait-s", "5", *extra]


def _main(world, *extra: str) -> int:
    return RUN.main(_argv(world, *extra), tokenizer=world["server"].tokenizer, client_factory=world["harness"].client_factory())


def test_items_file_and_sharding(tmp_path: Path):
    p = tmp_path / "items.json"
    p.write_text(json.dumps(["hle:a", "bcb:b", "hle:c"]))
    assert RUN.read_items_file(p) == ["hle:a", "bcb:b", "hle:c"]
    p.write_text(json.dumps({"items": [{"source_id": "hle:a"}, {"source_id": "bcb:b"}]}))
    assert RUN.read_items_file(p) == ["hle:a", "bcb:b"]
    p.write_text('{"source_id":"hle:a"}\n{"source_id":"bcb:b"}\n')
    assert RUN.read_items_file(p) == ["hle:a", "bcb:b"]
    p.write_text("hle:a\n\n# c\nbcb:b\n")
    assert RUN.read_items_file(p) == ["hle:a", "bcb:b"]
    p.write_text("hle:a\nhle:a\n")
    with pytest.raises(T.ProtocolError, match="duplicate"):
        RUN.read_items_file(p)
    assert RUN.shard_items(["a", "b", "c", "d", "e"], 1, 2) == ["b", "d"] and RUN.shard_items(["a"], 0, 1) == ["a"]
    with pytest.raises(ValueError):
        RUN.shard_items(["a"], 2, 2)
    assert RUN.parse_methods("IND_VOTE, DEC,CEN_FLAT") == [T.Method.IND_VOTE, T.Method.DEC, T.Method.CEN_FLAT]
    with pytest.raises(T.ProtocolError):
        RUN.parse_methods("S_FRESH")


def test_report_only_then_forecast_resumable(world):
    rr: Path = world["run_root"]
    srv = world["server"]
    ids = [t.source_id for t in world["tasks"]]
    selections = seals.load_selections(rr, world["seal"])
    for method, cell in zip((T.Method.IND_VOTE, T.Method.DEC, T.Method.CEN_FLAT), world["cells"]):
        for sid in ids:
            assert RUN.resolve_cell_id(selections, sid, method, checkpoint="32B", N=5, B=4, module="A", episode_rep=0) == cell.cell_id
    with pytest.raises(T.ProtocolError, match="no sealed"):
        RUN.resolve_cell_id(selections, ids[0], T.Method.DEC, checkpoint="32B", N=3, B=4, module="A", episode_rep=0)

    # --report-only: renders for every (item, method), no HTTP call
    before = srv.chat_count
    assert _main(world, "--report-only") == T.EXIT_DONE
    assert srv.chat_count == before
    renders = {}
    for sid in ids:
        for method in ("IND_VOTE", "DEC", "CEN_FLAT"):
            path = S.report_path(rr, sid, method)
            assert path.exists(), path
            payload = json.loads(path.read_text())
            renders[(sid, method)] = payload
            assert payload["kind"] == "FINAL_HANDOFF_REPORT" and payload["source_id"] == sid and payload["method"] == method
            assert payload["prompt_token_ids"] and payload["anchor_tokens"]["state_anchor"]["index"] < payload["last_prompt_token"]
            assert payload["anchor_tokens"]["task_only_anchor"]["index"] < payload["anchor_tokens"]["state_anchor"]["index"]
            assert payload["manifest"]["information_set"] == {"IND_VOTE": "SELF_ONLY", "DEC": "PEER_EXPOSED", "CEN_FLAT": "HUB_STATE"}[method]
            assert payload["evidence_tokens"] <= payload["evidence_tokens_cap"] == 8192
            assert payload["report"]["selected_candidate"] is not None  # the fake solver always answers
            assert not S.forecast_path(rr, sid, method).exists()
        # the real episodes: IND archive has >= 5 members, DEC latest_slots 5, CEN native 1
        assert renders[(sid, "IND_VOTE")]["report"]["vote_metadata"]["pool_size"] >= 5
        assert renders[(sid, "DEC")]["report"]["vote_metadata"]["pool_size"] == 5 and renders[(sid, "DEC")]["report"]["nonselected_total"] == 4
        assert renders[(sid, "CEN_FLAT")]["report"]["vote_metadata"]["pool_size"] == 1 and renders[(sid, "CEN_FLAT")]["report"]["packets_dropped"] == 0
    # report-only again: everything skipped, nothing rewritten
    mtimes = {k: S.report_path(rr, *k).stat().st_mtime_ns for k in renders}
    assert _main(world, "--report-only") == T.EXIT_DONE
    assert {k: S.report_path(rr, *k).stat().st_mtime_ns for k in renders} == mtimes

    # full mode, sharded 2 ways: 6 forecast requests in total, one per (item, method)
    assert _main(world, "--shard", "0", "--num-shards", "2", "--parallel", "2") == T.EXIT_DONE
    assert srv.chat_count == before + 3
    assert _main(world, "--shard", "1", "--num-shards", "2") == T.EXIT_DONE
    assert srv.chat_count == before + 6
    store = RequestStore(rr / "requests")
    for (sid, method), render in renders.items():
        out = json.loads(S.forecast_path(rr, sid, method).read_text())
        assert out["kind"] == "SHADOW_FORECAST" and out["report_id"] == render["report_id"] and out["request_id"] == render["request_id"]
        assert out["parse_status"] == "ok" and set(out["parsed"]) == set(S.FORECAST_FIELDS) and out["parsed"]["q_child_contract"] is None
        assert 0.0 <= out["parsed"]["q_personal"] <= 1.0 and 0.0 <= out["parsed"]["q_team_now"] <= 1.0
        assert out["cost"]["completion_tokens"] > 0 and out["aliased"] is False
        record = store.get(out["request_id"])
        assert record is not None and record.seed_key.purpose == "forecast" and record.seed_key.namespace == "forecast"
        assert record.sampling["max_tokens"] == 256 and record.chat_template_kwargs == {"enable_thinking": False}
        assert record.sampling["temperature"] == "0.0" and list(record.messages) == render["messages"]
        assert list(record.prompt_token_ids) == render["prompt_token_ids"]
        assert record.producer["cell_id"] == f"forecast.{method}.32B.x{world['seal'][:8]}"
        row = out["confidence_row"]
        assert row["parsing_status"] == "ok" and row["selected_pool_id"] == render["pool_id"] and row["future_label_ids_sealed"] == render["selection_id"]
    # resumable: a second full run issues no request; a lost forecast file is rebuilt from the store (aliased)
    assert _main(world) == T.EXIT_DONE and srv.chat_count == before + 6
    lost = S.forecast_path(rr, ids[0], "DEC")
    lost.unlink()
    assert _main(world, "--methods", "DEC") == T.EXIT_DONE and srv.chat_count == before + 6
    assert json.loads(lost.read_text())["aliased"] is True
    assert not list((rr / "forecast" / "errors").glob("*.json")) if (rr / "forecast" / "errors").exists() else True


def test_unsealed_item_and_bad_forecast_are_recorded(world):
    rr: Path = world["run_root"]
    srv = world["server"]
    ids = [t.source_id for t in world["tasks"]]
    # an item outside the seal → protocol error recorded, exit 4, the other job still runs
    world["items_file"].write_text("\n".join(ids + ["hle:main9999"]) + "\n")
    assert _main(world, "--report-only", "--methods", "IND_VOTE") == T.EXIT_SUSPENDED
    assert S.report_path(rr, ids[0], "IND_VOTE").exists() and S.report_path(rr, ids[1], "IND_VOTE").exists()
    assert not S.report_path(rr, "hle:main9999", "IND_VOTE").exists()
    world["items_file"].write_text("\n".join(ids) + "\n")
    # a malformed forecast completion is recorded with its parse status (never repaired)
    srv.responder = lambda req: "not a forecast" if req["mode"] == "forecast" else None
    assert _main(world, "--methods", "CEN_FLAT") == T.EXIT_DONE
    for sid in ids:
        out = json.loads(S.forecast_path(rr, sid, "CEN_FLAT").read_text())
        assert out["parse_status"] == "NOT_JSON" and out["parsed"] is None and out["raw_content"] == "not a forecast"
    # the compiled report equals the CLI's render (same sealed inputs, same tokenizer)
    cell = world["cells"][2]
    rep = compile_report(rr, ids[0], cell.cell_id, srv.tokenizer, seal=world["seal"], cfg=world["cfg"])
    render = json.loads(S.report_path(rr, ids[0], "CEN_FLAT").read_text())
    assert render["report_id"] == rep.report_id and render["report"]["text"] == rep.text
    assert M.render_forecast_request(rep).messages == render["messages"]
