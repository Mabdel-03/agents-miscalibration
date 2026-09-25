"""A synthetic passk run root (jobs manifests, parsed candidates, grades, public tasks) for fence tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MATH_TASK = {"__record__": "PublicTask", "allowed_choices": [], "benchmark": "aime", "item_id": "aime/2026/1",
             "output_kind": "math", "prompt": "p", "public_examples": [], "source_revision": "r", "subtask": None}
GPQA_TASK = {"__record__": "PublicTask", "allowed_choices": ["A", "B", "C", "D"], "benchmark": "gpqa_diamond",
             "item_id": "gpqa_diamond/recX", "output_kind": "option", "prompt": "p", "public_examples": [],
             "source_revision": "r", "subtask": None}


def node_id(partition: str, item: str, budget: int, r: int) -> str:
    return hashlib.sha256(f"{partition}|{item}|{budget}|{r}".encode()).hexdigest()


def write_node(shard_dir: Path, partition: str, task: dict, budget: int, r: int, final: str | None,
               correct: bool | None, *, graded: bool = True) -> str:
    nid = node_id(partition, task["item_id"], budget, r)
    jobs = shard_dir / "manifests" / "jobs" / f"{partition}.jsonl"
    jobs.parent.mkdir(parents=True, exist_ok=True)
    spec = {"__record__": "GenerationSpec", "benchmark": task["benchmark"], "item_id": task["item_id"],
            "kind": "root" if budget == 1024 else "deep", "reasoning_budget": budget, "root_slot": r, "draw_id": r,
            "partition_id": partition}
    with jobs.open("a") as handle:
        handle.write(json.dumps({"node_id": nid, "record": spec}) + "\n")
    parsed = shard_dir / "parsed" / nid[:2] / f"{nid}.json"
    parsed.parent.mkdir(parents=True, exist_ok=True)
    candidate = {"__record__": "Candidate", "candidate_id": nid, "final_answer": final, "finish_reason": "stop",
                 "item_id": task["item_id"], "node_id": nid, "output_kind": task["output_kind"],
                 "parse_status": "valid" if final is not None else "invalid", "rationale": "x",
                 "visible_content": f"<final>{final}</final>"}
    result = {"generation_id": f"gen-{nid[:8]}", "latency_seconds": 1.0, "observed_model": "qwen/qwen3.8-27b",
              "observed_provider": "Mancer 2", "raw_response_path": None,
              "usage": {"actual_cost_usd": "0.001", "raw_usage": {"think": {"prompt_tokens": 10,
                                                                            "completion_tokens": budget},
                                                                  "answer": {"prompt_tokens": 20,
                                                                             "completion_tokens": 5}}}}
    parsed.write_text(json.dumps({"candidate": candidate, "result": result}))
    if graded:
        grade = shard_dir / "grades" / nid[:2] / f"{nid}.json"
        grade.parent.mkdir(parents=True, exist_ok=True)
        grade.write_text(json.dumps({
            "__record__": "GradeResult", "candidate_id": nid, "cpu_seconds": 0.0, "detail_code": "x",
            "environment_digest": "e", "grader_version": "g", "operational_success": True,
            "private_eval_hash": "h", "semantic_correct": correct,
            "status": "graded" if final is not None else "invalid_output"}))
    return nid


def build(run_root: Path) -> dict[str, list[str]]:
    art = run_root / "artifacts"
    public = art / "manifests" / "public"
    public.mkdir(parents=True, exist_ok=True)
    (public / "aime.jsonl").write_text(json.dumps(MATH_TASK) + "\n")
    (public / "gpqa_diamond.jsonl").write_text(json.dumps(GPQA_TASK) + "\n")
    shard = art / "shards" / "mancer"
    ids: dict[str, list[str]] = {}
    for partition in ("smoke", "e1_fill", "c48_extra", "t200_diag", "v48_bank", "t200_bank"):
        ids[partition] = [write_node(shard, partition, MATH_TASK, 2048, r, "\\boxed{7}" if r else "\\boxed{3}",
                                     r > 0) for r in range(1, 4)]
    return ids
