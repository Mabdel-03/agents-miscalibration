"""evaluation/*: HLE judge parse (yes/no/ambiguous), MC exact scoring (P2-2), the judge
request identity, the BCB driver in-process (pass/fail/timeout), the exact apptainer argv
(06_bcb_container.md), dedupe, and the audit helpers (P1-10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.evaluation import audit, bcb, hle_judge
from agents_scaling.study.evaluation.bcb_container_driver import container_env
from agents_scaling.study.types import ProtocolError
from tests.study.wp5_support import make_tasks

# --------------------------------------------------------------------------- HLE judge parse


@pytest.mark.parametrize(
    "content,expected",
    [
        ("extracted_final_answer: 4\nreasoning: same\ncorrect: yes\nconfidence: 90", "yes"),
        ("extracted_final_answer: 5\nreasoning: differs\ncorrect: no\nconfidence: 100", "no"),
        ("**correct**: Yes.\n", "yes"),
        ('{"extracted_final_answer": "4", "reasoning": "x", "correct": "yes", "confidence": 90}', "yes"),
        ('```json\n{"extracted_final_answer": "4", "reasoning": "x", "correct": "NO", "confidence": 90}\n```', "no"),
        ('{"correct": "maybe"}', "ambiguous"),
        ('{"correct": true}', "ambiguous"),
        ("reasoning: the answer is correct: yes and no", "ambiguous"),
        ("correct: yes\ncorrect: no", "ambiguous"),
        ("", "ambiguous"),
        (None, "ambiguous"),
        ("I think it is right.", "ambiguous"),
    ],
)
def test_parse_hle_judge(content, expected):
    assert hle_judge.parse_hle_judge(content) == expected


def test_parse_hle_judge_truncated_is_ambiguous():
    assert hle_judge.parse_hle_judge("correct: yes", finish_reason="length") == "ambiguous"


# --------------------------------------------------------------------------- MC scoring


@pytest.mark.parametrize(
    "answer,gold,expected",
    [("B", "B", True), ("(b)", "B", True), ("Answer: B.", "B", True), ("A", "B", False), ("B and C", "B", False), ("", "B", False),
     ("two", "B", False), ("42", "42", True), ("41", "42", False), ("$42$", "42", True)],
)
def test_score_mc(answer, gold, expected):
    assert hle_judge.score_mc(answer, gold) is expected


def test_gold_letter_and_types():
    assert hle_judge.gold_mc_letter("(C)") == "C" and hle_judge.gold_mc_letter("forty-two") is None
    with pytest.raises(TypeError):
        hle_judge.score_mc(None, "B")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- judge request


def test_hle_judge_request_identity(study_config):
    task = make_tasks(1, "dev")[0]
    ckpt = study_config.judge
    spec = hle_judge.hle_judge_request(task, "B", "B", ckpt, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)
    assert spec.decoding == T.JUDGE_DECODING and spec.decoding.max_tokens == 1024 and spec.decoding.enable_thinking is False
    assert spec.seed_key.as_array()[4:] == [0, T.PURPOSE_HLE_JUDGE, 0, T.NS_JUDGE]
    assert spec.seed_key.model_cell == ckpt.model_cell and spec.role == hle_judge.ROLE_HLE_JUDGE
    assert spec.messages[0]["role"] == "user" and "[correct_answer]: B" in spec.messages[0]["content"]
    again = hle_judge.hle_judge_request(task, "B", "B", ckpt, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)
    assert again.request_id == spec.request_id
    other = hle_judge.hle_judge_request(task, "C", "B", ckpt, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)
    assert other.request_id != spec.request_id
    guided = hle_judge.hle_judge_request(task, "B", "B", ckpt, True, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)
    assert guided.decoding.guided_json is True and guided.request_id != spec.request_id
    with pytest.raises(ValueError):
        hle_judge.hle_judge_request(task, "  ", "B", ckpt, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)
    with pytest.raises(ProtocolError):
        hle_judge.hle_judge_request(task, "B", "", ckpt, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex)


class _StubStore:
    def __init__(self, content_by_answer):
        self.content_by_answer = content_by_answer
        self.calls = 0

    def get_or_generate(self, spec, client, cell_id, on_event=None):
        self.calls += 1
        answer = spec.messages[0]["content"].split("[response]: ", 1)[1].split("\n", 1)[0]
        content = self.content_by_answer[answer]
        record = type("Rec", (), {})()
        record.request_id = spec.request_id
        record.response = {"content": content, "finish_reason": "stop"}
        return record, False


def _cand(cid: str, answer: str, valid: bool = True) -> T.CandidateRecord:
    cand = None if not valid else T.Candidate("a", (), (), (), answer, 0.5)
    return T.CandidateRecord(cid, "r" + cid, 0, "root", valid, None if valid else "NOT_JSON", cand, "s" + cid, "raw")


def test_judge_item_dedupes_and_scores(study_config):
    task = make_tasks(2, "dev")[1]  # exactMatch
    store = _StubStore({"4": "correct: yes", "5": "correct: no", "?": "hmm"})
    cands = [_cand("c1", "4"), _cand("c2", "4"), _cand("c3", "5"), _cand("c4", "?"), _cand("c5", "", valid=False), _cand("c6", "  ")]
    rec = hle_judge.judge_item(task, "4", cands, checkpoint=study_config.judge, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex,
                               store=store, client=None, cell_id="eval.x", seal="ab" * 32, clock=lambda: 100.0)
    assert store.calls == 3  # c1/c2 share one judgement; c6 (blank) is scored without a judge
    correct = hle_judge.candidate_correctness(rec, cands)
    assert correct == {"c1": True, "c2": True, "c3": False, "c4": False, "c5": False, "c6": False}
    entry = rec["judgements"][hle_judge.answer_sha256("?")]
    assert entry["judge"] == "ambiguous" and entry["flagged"] is True and entry["correct"] is False
    assert rec["judgements"][hle_judge.answer_sha256("  ")]["note"] == hle_judge.EMPTY_ANSWER_NOTE
    assert rec["n_ambiguous"] == 1 and rec["invalid_candidate_ids"] == ["c5"]
    assert rec["judgements"][hle_judge.answer_sha256("4")]["started_at_by_seal"]["ab" * 32] == 100.0
    # a second seal reuses the judgements (deterministic judge) but stamps its own start
    store2 = _StubStore({})
    rec2 = hle_judge.judge_item(task, "4", cands[:2], checkpoint=study_config.judge, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex,
                                store=store2, client=None, cell_id="eval.y", seal="cd" * 32, existing=rec, clock=lambda: 200.0)
    assert store2.calls == 0 and rec2["judgements"][hle_judge.answer_sha256("4")]["started_at_by_seal"]["cd" * 32] == 200.0
    assert set(rec2["judgements"]) == set(rec["judgements"])
    with pytest.raises(ProtocolError):
        hle_judge.candidate_correctness(rec, [_cand("c9", "unjudged")])


def test_judge_item_mc_uses_letter_and_judge(study_config):
    task = make_tasks(1, "dev")[0]  # multipleChoice
    store = _StubStore({"B": "correct: no", "C": "correct: yes"})  # a wrong judge on purpose
    cands = [_cand("c1", "B"), _cand("c2", "C")]
    rec = hle_judge.judge_item(task, "B", cands, checkpoint=study_config.judge, study_id=study_config.study_id, study_seed_hex=study_config.study_seed_hex,
                               store=store, client=None, cell_id="eval.x", seal="ab" * 32)
    correct = hle_judge.candidate_correctness(rec, cands)
    assert correct == {"c1": True, "c2": False}  # letter scoring is authoritative (E1)
    j1 = rec["judgements"][hle_judge.answer_sha256("B")]
    assert j1["letter_correct"] is True and j1["judge"] == "no"


# --------------------------------------------------------------------------- BCB


PASS_CODE = "def task_func():\n    return 1\n"
TEST_SRC = "import unittest\nfrom solution import task_func\n\nclass T(unittest.TestCase):\n    def test_one(self):\n        self.assertEqual(task_func(), 1)\n"
LOOP_CODE = "def task_func():\n    while True:\n        pass\n"


def test_apptainer_argv_exact(tmp_path: Path):
    argv = bcb.apptainer_argv("/x/bcb.sif", tmp_path)
    assert argv == [
        "apptainer", "exec", "--containall", "--cleanenv", "--no-home", "--net", "--network", "none",
        "--bind", f"{tmp_path}:/work", "--pwd", "/work", "/x/bcb.sif",
        "python3", "/work/bcb_container_driver.py", "/work/jobs.jsonl", "/work/results.jsonl",
    ]
    with pytest.raises(T.InfraFailure):
        bcb.BcbEvaluator(tmp_path / "missing.sif")


def test_driver_in_process_pass_fail_timeout(tmp_path: Path):
    ev = bcb.BcbEvaluator(tmp_path / "unused.sif", timeout_s=2.0, container="none", work_root=tmp_path / "work")
    jobs = [
        {"cid": "ok", "code": "```python\n" + PASS_CODE + "```", "test_src": TEST_SRC},
        {"cid": "ok2", "code": bcb.program_of("```\n" + PASS_CODE + "\n```"), "test_src": TEST_SRC},  # byte-identical after the frozen strip
        {"cid": "wrong", "code": "def task_func():\n    return 2\n", "test_src": TEST_SRC},
        {"cid": "syntax", "code": "def task_func(:\n", "test_src": TEST_SRC},
        {"cid": "loop", "code": LOOP_CODE, "test_src": TEST_SRC},
    ]
    jobs[0]["code"] = bcb.program_of(jobs[0]["code"])  # the runner strips one fence (P1-5)
    out = ev.evaluate_many(jobs)
    assert {k: v["status"] for k, v in out.items()} == {"ok": "pass", "ok2": "pass", "wrong": "fail", "syntax": "fail", "loop": "timeout"}
    assert out["ok2"]["deduplicated_from"] == "ok" and "OK" in out["ok"]["stderr_tail"]
    assert out["loop"]["elapsed"] >= 2.0 and out["loop"]["returncode"] is None
    assert not list((tmp_path / "work").iterdir())  # work dir removed


def test_dedupe_and_jobs_for_item():
    unique, reps = bcb.dedupe_jobs([{"cid": "a", "code": "x", "test_src": "t"}, {"cid": "b", "code": "x", "test_src": "t"}, {"cid": "c", "code": "y", "test": "t"}])
    assert [j["cid"] for j in unique] == ["a", "c"] and reps == {"a": "a", "b": "a", "c": "c"}
    jobs, invalid = bcb.jobs_for_item("bcb:1", "t", [_cand("v", "```\nx = 1\n```"), _cand("i", "", valid=False)])
    assert jobs == [{"cid": "v", "code": "x = 1", "test_src": "t"}] and invalid == ["i"]


def test_container_env(tmp_path: Path):
    env = container_env(str(tmp_path))
    assert env["MPLBACKEND"] == "Agg" and env["MPLCONFIGDIR"] == str(tmp_path / ".mpl") and env["HOME"] == str(tmp_path / ".home")
    assert (tmp_path / ".cache").is_dir() and "HF_TOKEN" not in env


# --------------------------------------------------------------------------- audit


def _write_eval(run_root: Path, sid: str, fmt: str, entries: dict) -> None:
    path = hle_judge.hle_eval_path(run_root, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "source_id": sid, "answer_format": fmt, "judgements": entries}))


def test_mc_audit_and_bound(tmp_path: Path):
    def e(judge, letter, sha):
        return {"judge": judge, "letter_correct": letter, "correct": letter, "final_answer_sha256": sha, "request_id": "r" + sha, "candidate_ids": ["c" + sha]}

    _write_eval(tmp_path, "hle:1", "multipleChoice", {"a": e("yes", True, "a"), "b": e("no", False, "b"), "c": e("yes", False, "c"), "d": e("ambiguous", True, "d")})
    _write_eval(tmp_path, "hle:2", "exactMatch", {"x": {"judge": "yes", "letter_correct": None, "correct": True, "final_answer_sha256": "x", "request_id": "rx", "candidate_ids": []}})
    out = audit.mc_audit(tmp_path)
    assert (out["tp"], out["fp"], out["tn"], out["fn"], out["ambiguous"]) == (1, 1, 1, 1, 1)
    assert out["fp_rate"] == 0.5 and out["fn_rate"] == 0.5 and out["n_mc_items"] == 1
    bound = audit.differential_error_bound(0.02, 0.01, 0.5, 0.6)
    assert bound["differential_bound_pp"] == pytest.approx(100 * (max(0.5 * 0.01, 0.5 * 0.02) + max(0.6 * 0.01, 0.4 * 0.02)))
    assert bound["could_exceed_threshold"] is False
    assert audit.differential_error_bound(0.2, 0.2, 0.5, 0.5)["could_exceed_threshold"] is True
    with pytest.raises(ValueError):
        audit.differential_error_bound(1.5, 0, 0, 0)
