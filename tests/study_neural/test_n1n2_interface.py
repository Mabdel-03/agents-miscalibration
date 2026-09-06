"""Cross-package contract (P0-1 regression): a real N2 render (``forecast.manifest.report_render``
on a sealed world, pinned Qwen3 tokenizer) resolves in N1 (``neural.anchors.resolve_report_anchors``
+ ``neural.capture.check_report_anchors``) to the task-only and state anchors N2 computed —
never to ``ANCHOR_UNRESOLVED`` / the last-prefill default — and the model-revision guard
reads the render's checkpoint pin.  CPU; skipped when the offline tokenizer is absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.config import load_config
from agents_scaling.study.forecast import manifest as M
from agents_scaling.study.forecast.report import compile_report
from agents_scaling.study.neural import anchors as A
from agents_scaling.study.neural import capture as C
from agents_scaling.study.neural.replay import check_prompt_identity
from agents_scaling.study.prompts import render as PR
from tests.study_neural import n2_support as N
from tests.study_neural import support

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def tok():
    try:
        return support.qwen_tokenizer("8B")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-8B tokenizer unavailable offline: {exc!r}")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _render(tmp_path: Path, cfg, tok, *, words: int = 0) -> tuple[dict, object, N.World]:
    tasks = N.make_tasks(1, "main")[:1]
    w = N.World(tmp_path / "run", cfg, tasks)
    sid = tasks[0].source_id
    w.add(T.Method.IND_VOTE, {sid: (N.ind_pool("real", ["B", "A", "C", "B", "D"], words=words), None, None)})
    w.seal()
    rep = compile_report(w.run_root, sid, w.cells[T.Method.IND_VOTE].cell_id, tok, seal=N.SEAL, cfg=w.cfg)
    payload = M.report_render(rep, tok, cfg, cfg.flagship_checkpoint)
    return payload, rep, w


def test_n2_render_resolves_in_n1_to_the_same_anchors(tmp_path: Path, cfg, tok):
    payload, rep, w = _render(tmp_path, cfg, tok)
    # the flat mirrors equal the nested N2 fields
    assert payload["task_only_anchor_byte"] == payload["byte_anchors"]["task_only_anchor"]
    assert payload["state_anchor_byte"] == payload["byte_anchors"]["state_anchor"]
    assert payload["task_only_anchor_token"] == payload["anchor_tokens"]["task_only_anchor"]["index"]
    assert payload["state_anchor_token"] == payload["anchor_tokens"]["state_anchor"]["index"]
    assert payload["report_sha256"] == rep.text_sha256 == payload["report"]["text_sha256"]
    # write it exactly as forecast.run does and read it back through the capture work list
    path = w.run_root / "forecast" / "reports" / f"{rep.source_id}.IND_VOTE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    [work] = C.report_work_list(w.run_root)
    assert work.report_id == rep.report_id and work.method == "IND_VOTE"
    report = C.load_report(Path(work.path))
    # N1 asserts the exact render identity, then resolves both anchors from the N2 fields
    ids = tuple(int(t) for t in report["prompt_token_ids"])
    check_prompt_identity({"request_id": work.report_id, "prompt_token_ids": ids, "messages": report["messages"],
                           "chat_template_kwargs": report["chat_template_kwargs"], "response": {"token_ids": []}}, tok, strict=True)
    anchors = {a.kind: a for a in A.resolve_report_anchors(report, tok)}
    C.check_report_anchors(work.report_id, anchors.values())
    task, state, last = anchors[A.ANCHOR_TASK_ONLY], anchors[A.ANCHOR_STATE], anchors[A.ANCHOR_LAST_PREFILL]
    assert task.present and task.detail["source"] == "token+byte" and task.token_offset == payload["anchor_tokens"]["task_only_anchor"]["index"]
    assert state.present and state.detail["source"] == "token+byte" and state.token_offset == payload["anchor_tokens"]["state_anchor"]["index"]
    assert task.token_offset < state.token_offset < last.token_offset == len(ids) - 1 and last.detail["equals_state_anchor"] is False
    # the anchors sit where the brief says: end of the task span, closing '===' of the report
    assert tok.decode(list(ids[: task.token_offset + 1])).endswith(rep.task_text[-1]) or payload["anchor_tokens"]["task_only_anchor"]["straddles"]
    assert tok.decode(list(ids[: state.token_offset + 1])).endswith(PR.REPORT_CLOSE)
    assert tok.decode(list(ids[state.token_offset + 1:])).startswith("\n<|im_end|>")
    # the nested shape alone (an N2 render before the flat mirrors existed) resolves identically
    legacy = {k: v for k, v in report.items() if k not in ("task_only_anchor_byte", "state_anchor_byte", "task_only_anchor_token", "state_anchor_token")}
    again = {a.kind: a for a in A.resolve_report_anchors(legacy, tok)}
    assert (again[A.ANCHOR_TASK_ONLY].token_offset, again[A.ANCHOR_STATE].token_offset) == (task.token_offset, state.token_offset)
    # the model-revision guard reads the render's checkpoint pin
    assert C.report_model_revision(report) == cfg.flagship_checkpoint.model_revision
    assert C.report_model_revision({k: v for k, v in report.items() if k != "checkpoint"}) == ""


def test_n1_refuses_a_render_without_the_anchor_interface(tmp_path: Path, cfg, tok):
    payload, rep, w = _render(tmp_path, cfg, tok)
    stripped = {k: v for k, v in payload.items() if k not in ("byte_anchors", "anchor_tokens", "task_only_anchor_byte", "state_anchor_byte",
                                                              "task_only_anchor_token", "state_anchor_token")}
    anchors = A.resolve_report_anchors(stripped, tok)
    with pytest.raises(RuntimeError, match="TASK_ONLY_ANCHOR unresolved"):
        C.check_report_anchors(rep.report_id, anchors)
    only_task = {**stripped, "task_only_anchor_byte": payload["task_only_anchor_byte"]}
    with pytest.raises(RuntimeError, match="STATE_ANCHOR absent"):
        C.check_report_anchors(rep.report_id, A.resolve_report_anchors(only_task, tok))
