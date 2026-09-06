"""N1 end-to-end on CPU: the capture CLI runs a two-layer random Qwen3 (real Qwen3 vocab and
tokenizer files, hidden 32) over synthetic RequestRecords in a real RequestStore and over
one N2-style report, writes ledger rows + vectors, produces the fidelity file, and resumes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from agents_scaling.study import types as T
from agents_scaling.study.inference.store import RequestStore
from agents_scaling.study.neural import capture as C
from agents_scaling.study.neural import storage as S
from tests.study_neural import support

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "study" / "fixtures" / "request_record.example.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


@pytest.fixture(scope="module")
def tok():
    try:
        return support.qwen_tokenizer("8B")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-8B tokenizer unavailable offline: {exc!r}")


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory, tok) -> Path:
    """A tiny random Qwen3 with the real vocabulary, saved next to the real tokenizer files."""
    path = tmp_path_factory.mktemp("snap")
    model = support.tiny_qwen3(seed=0, vocab_size=int(tok.vocab_size) + len(tok.added_tokens_decoder) + 64)
    model.save_pretrained(path, safe_serialization=True)
    src = support.snapshot_path("8B")
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, path / name)
    return path


def _record(tok, rid: str, content: str, *, role: str = "root", think: str = "thinking 一二三", model_revision: str = "9216db5781bf21249d130ec9da846c4624c16137") -> T.RequestRecord:
    base = json.loads(FIXTURE.read_text())
    messages = [{"role": "user", "content": f"Task {rid[:4]}: what is 2+2?"}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
    completion = support.completion_from_text(tok, think, content)
    base.update({
        "request_id": rid, "messages": messages, "prompt_token_ids": list(ids), "prompt_tokens": len(ids),
        "response": {**base["response"], "token_ids": completion, "completion_tokens": len(completion), "content": content, "reasoning": think,
                     "reasoning_tokens": completion.index(tok.convert_tokens_to_ids("</think>")) + 1, "finish_reason": "stop"},
        "identity": {**base["identity"], "model_revision": model_revision},
    })
    base["model"] = {**base["model"], "model_revision": model_revision}
    record = T.RequestRecord.from_dict({k: v for k, v in base.items() if k != "content_sha256"})
    return record.with_content_sha256()


def _run(args: list[str]) -> int:
    return C.main(args)


def test_native_stage_end_to_end_and_resume(tmp_path: Path, snapshot: Path, tok):
    results = tmp_path / "results"
    run_root = results / "study_v4"
    store = RequestStore(run_root / "requests")
    rids = [f"{i + 1:064x}" for i in range(3)]
    contents = ["\n\n" + support.CANDIDATE_JSON, "\n\n```json\n" + support.CANDIDATE_JSON_CJK + "\n```", "\n\nnot json"]
    for rid, content in zip(rids, contents):
        store.publish(_record(tok, rid, content))
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"request_id": rid, "source_id": f"hle:{i}", "method": "IND_VOTE", "native_role": "root",
                                         "declared_role": "INDEPENDENT_SOLVER", "phase": "ROOT", "cell_id": "A.IND_VOTE", "episode_id": f"ep{i}", "group_size": 5}) + "\n"
                          for i, rid in enumerate(rids)))
    common = ["--run-id", "study_v4", "--stage", "native", "--checkpoint", "32B", "--results-root", str(results), "--snapshot-path", str(snapshot),
              "--items-file", str(items), "--device-map", "cpu", "--dtype", "fp32", "--blocks", "0,1", "--generated-ks", "4,32,512",
              "--max-batch-tokens", "256", "--group-size", "2", "--chunk-rows", "4", "--fidelity", "2", "--fidelity-tokens", "6"]
    assert _run(common) == 0
    stage = run_root / "neural" / "native"
    stats = json.loads((stage / "stats.native.s000of001.json").read_text())
    assert stats["captured"] == 3 and stats["missing_records"] == 0 and stats["rows"] == 3 * 5 * 2
    # GENERATED_512 is never reached; the short "not json" completion also misses GENERATED_32
    assert stats["anchors_missing"] == {"NOT_REACHED": 4, "JSON_CLOSE_MISSING": 1}
    fid = json.loads((stage / "fidelity.native.s000of001.json").read_text())
    assert fid["summary"]["records"] == 2 and fid["summary"]["scored_tokens"] == 12 and fid["summary"]["identity_failures"] == 0
    assert fid["summary"]["two_pass_max_abs_logit_diff"] == 0.0
    matrix, frame = S.load_stage(run_root, "native")
    assert len(frame) == 30 and int((frame.vector_index < 0).sum()) == 5 * 2  # 5 missing anchors × 2 blocks, no vectors
    assert matrix.shape == (20, support.TINY_HIDDEN) and matrix.shape[0] == int((frame.vector_index >= 0).sum())
    assert set(frame.anchor_kind) == {"NATIVE_PREFILL", "GENERATED_4", "GENERATED_32", "GENERATED_512", "FINAL_OBJECT_CLOSE"}
    assert set(frame.block) == {0, 1} and set(frame.condition) == {"native:IND_VOTE"} and set(frame.role) == {"INDEPENDENT_SOLVER"}
    assert set(frame.consumer_revision) == {"9216db5781bf21249d130ec9da846c4624c16137"}
    close = frame[(frame.anchor_kind == "FINAL_OBJECT_CLOSE") & (frame.block == 0)].set_index("StateSnapshot_id")
    assert close.loc[rids[0], "vector_index"] >= 0 and close.loc[rids[1], "vector_index"] >= 0
    assert close.loc[rids[2], "missingness"] == "JSON_CLOSE_MISSING" and close.loc[rids[2], "vector_index"] == -1
    assert close.loc[rids[0], "channel"] == "content" and close.loc[rids[0], "generated_token_count"] > 32
    g4 = frame[(frame.anchor_kind == "GENERATED_4") & (frame.block == 1)].iloc[0]
    assert g4.channel == "thinking" and g4.token_offset == g4.extra["prompt_tokens"] + 3 and g4.generated_token_count == 4
    assert all(frame.nonce_hash.str.len() == 64) and all(frame.hook_convention == C.HOOK_CONVENTION)
    present = frame[frame.vector_index >= 0]
    assert all(present.tensor_hash.map(lambda h: isinstance(h, str) and len(h) == 64))
    assert np.isfinite(matrix.astype(np.float32)).all()
    sel = json.loads((stage / "selection.native.s000of001.json").read_text())
    assert sel["shard_work_items"] == 3 and sel["blocks"] == [0, 1]
    # resume: nothing pending, no new chunks
    chunks_before = sorted(p.name for p in stage.glob("*.jsonl"))
    assert _run(common) == 0
    assert sorted(p.name for p in stage.glob("*.jsonl")) == chunks_before
    assert json.loads((stage / "stats.native.s000of001.json").read_text())["pending"] == 0
    # a record from another checkpoint is refused unless explicitly allowed
    other = f"{9:064x}"
    store.publish(_record(tok, other, "\n\n" + support.CANDIDATE_JSON, model_revision="b968826d9c46dd6066d109eabc6255188de91218"))
    items.write_text(json.dumps({"request_id": other}) + "\n")
    with pytest.raises(RuntimeError, match="model_revision"):
        _run(common)
    assert _run(common + ["--allow-model-mismatch"]) == 0
    assert (other, 0, "NATIVE_PREFILL") in S.existing_keys(run_root, "native")


def test_report_stage_end_to_end(tmp_path: Path, snapshot: Path, tok):
    results = tmp_path / "results"
    run_root = results / "study_v4"
    reports = run_root / "forecast" / "reports"
    reports.mkdir(parents=True)
    task = "=== TASK ===\nWhat is 2+2?\n=== END TASK ===\n"
    body = "=== REPORT (observable state) ===\nselected: 4 (confidence 0.9)\n=== END REPORT ===\n"
    content = task + body + "Output contract: {}"
    messages = [{"role": "user", "content": content}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    for i, method in enumerate(("IND_VOTE", "DEC")):
        (reports / f"hle_1.{method}.json").write_text(json.dumps({
            "report_id": f"rep-{method}", "source_id": "hle:1", "method": method, "messages": messages, "prompt_token_ids": list(ids),
            "chat_template_kwargs": {"enable_thinking": False}, "task_only_anchor_byte": len(task.encode()),
            "state_anchor_byte": len((task + body).encode()), "report_sha256": "x" * 64,
        }))
    args = ["--run-id", "study_v4", "--stage", "report", "--checkpoint", "32B", "--results-root", str(results), "--snapshot-path", str(snapshot),
            "--device-map", "cpu", "--dtype", "fp32", "--blocks", "1", "--max-batch-tokens", "256", "--fidelity", "0"]
    assert C.main(args) == 0
    matrix, frame = S.load_stage(run_root, "report")
    assert len(frame) == 6 and matrix.shape == (6, support.TINY_HIDDEN)
    assert set(frame.anchor_kind) == {"TASK_ONLY_ANCHOR", "STATE_ANCHOR", "LAST_PREFILL"} and set(frame.condition) == {"report:IND_VOTE", "report:DEC"}
    by = frame[frame.StateSnapshot_id == "rep-DEC"].set_index("anchor_kind")
    assert by.loc["TASK_ONLY_ANCHOR", "token_offset"] < by.loc["STATE_ANCHOR", "token_offset"] < by.loc["LAST_PREFILL", "token_offset"] == len(ids) - 1
    assert by.loc["STATE_ANCHOR", "role"] == "REPORT_READER" and by.loc["STATE_ANCHOR", "phase"] == "FINAL_HANDOFF_REPORT"
    assert by.loc["STATE_ANCHOR", "extra"]["report_sha256"] == "x" * 64
    # the two reports share the prompt, so the vectors at equal positions are identical
    a = frame[(frame.StateSnapshot_id == "rep-DEC") & (frame.anchor_kind == "STATE_ANCHOR")].vector_index.iloc[0]
    b = frame[(frame.StateSnapshot_id == "rep-IND_VOTE") & (frame.anchor_kind == "STATE_ANCHOR")].vector_index.iloc[0]
    assert np.array_equal(matrix[a], matrix[b])
    # a report whose ids are not the render is refused before any forward pass
    bad = json.loads((reports / "hle_1.DEC.json").read_text())
    bad.update({"report_id": "rep-bad", "prompt_token_ids": list(ids)[:-1]})
    (reports / "hle_1.CEN_FLAT.json").write_text(json.dumps(bad))
    with pytest.raises(Exception, match="identity"):
        C.main(args)
