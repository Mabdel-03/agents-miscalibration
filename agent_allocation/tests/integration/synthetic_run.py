"""A small, fully known run directory written in the contract's on-disk formats (docs/00_contract.md sections 3 and 6).

Six items in the ``main`` partition with the ``main_extension`` grid (138 nodes each): MATH, MMLU-Pro, two BBH
subtasks (one item on logical_deduction, two on boolean_expressions) and MBPP+. Every node is graded correct except
the pool heads that fix the two primary contrasts; selections pick the first pool member. Hand-computed values:

  item          peer/4/4  pt/4/4  pt/4/1  cp/3  cp/0   D_i   D_ii
  math500          1        0       0      1     0      1     1
  mmlu_pro         0        0       1      1     1      0     1
  bbh logical      1        1       1      0     1      0    -1
  bbh boolean #1   0        1       0      1     0     -1     0
  bbh boolean #2   1        1       0      1     1      0    -1
  mbpp_plus        1        0       1      1     0      1     2

  primary_i  = mean(1, 0, mean(0, mean(-1, 0)), 1)  = 7/16  = 0.4375   (item-weighted BBH would give 5/12)
  primary_ii = mean(1, 1, mean(-1, mean(0, -1)), 2) = 13/16 = 0.8125   (item-weighted BBH would give 5/6)
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

from agent_allocation.config import grids_from_config
from agent_allocation.contracts import (
    STUDY_ID,
    BBH_SUBTASKS,
    Candidate,
    CheckpointView,
    CompletionStatus,
    GenerationResult,
    GradeResult,
    GradeStatus,
    LedgerEvent,
    LedgerEventKind,
    OutputKind,
    ParseStatus,
    PolicyCheckpoint,
    PrivateEvalSpec,
    PublicTask,
    SelectionResult,
    SelectionStatus,
    TransportStatus,
    Usage,
    canonical_hash,
    dump_record,
)
from agent_allocation.paths import RunPaths

from tests.integration import adversarial
from tests.integration import helpers as h
from tests.integration.openrouter_stub import completion_payload

EXPECTED_PRIMARY_I = 7 / 16
EXPECTED_PRIMARY_II = 13 / 16
HEADS = ("peer/4/4", "private_team/4/4", "private_team/4/1", "cp/3", "cp/0")
TABLE = {
    "math500/adv/1": (1, 0, 0, 1, 0),
    "mmlu_pro/adv/1": (0, 0, 1, 1, 1),
    "bbh_logic4/adv/1": (1, 1, 1, 0, 1),
    "bbh_logic4/adv/2": (0, 1, 0, 1, 0),
    "bbh_logic4/syn/3": (1, 1, 0, 1, 1),
    "mbpp_plus/adv/1": (1, 0, 1, 1, 0),
}
INVALID_ROOT_SLOT = 24                                  # one unparseable root per item: INVALID_OUTPUT, never "wrong"


def tasks() -> list[tuple[PublicTask, PrivateEvalSpec]]:
    items = {t.item_id: (t, p) for t, p, _ in adversarial.load_items()}
    extra = PublicTask("bbh_logic4/syn/3", "bbh_logic4", adversarial.REVISION, "True and not False is",
                       OutputKind.BOOLEAN, allowed_choices=("True", "False"), subtask=BBH_SUBTASKS[2])
    extra_private = PrivateEvalSpec(extra.item_id, extra.benchmark, extra.source_revision,
                                    items["bbh_logic4/adv/2"][1].eval_kind, "True", None, canonical_hash("syn3"),
                                    subtask=extra.subtask, allowed_choices=extra.allowed_choices)
    items[extra.item_id] = (extra, extra_private)
    return [items[i] for i in TABLE]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _answer(task: PublicTask, private: PrivateEvalSpec, correct: bool) -> str:
    if task.output_kind is OutputKind.PYTHON:
        return "def add_two(a, b):\n    return a + b" if correct else "def add_two(a, b):\n    return a - b"
    if task.output_kind is OutputKind.BOOLEAN:
        return private.gold_answer if correct else ("True" if private.gold_answer == "False" else "False")
    if task.output_kind is OutputKind.OPTION:
        return private.gold_answer if correct else next(c for c in task.allowed_choices if c != private.gold_answer)
    return private.gold_answer if correct else "5"


def write_synthetic_run(paths: RunPaths, cfg: dict[str, Any], *, selection_layout: str = "v2") -> dict[str, Any]:
    paths.ensure()
    grid = grids_from_config(cfg)["main_extension"]
    (paths.config_lock.parent).mkdir(parents=True, exist_ok=True)
    paths.config_lock.write_text(json.dumps({"config": cfg}, sort_keys=True))
    jobs, policies, selections, journal = [], [], [], []
    publics: dict[str, list[dict[str, Any]]] = {}
    privates: dict[str, list[dict[str, Any]]] = {}
    seq = 0
    node_index = 0
    summary: dict[str, Any] = {"items": {}}
    for task, private in tasks():
        publics.setdefault(task.benchmark, []).append(dump_record(task))
        privates.setdefault(task.benchmark, []).append(dump_record(private))
        graph = h.ref_graph(grid, task)
        wrong_heads = {graph.checkpoints[name].candidate_ids[0] for name, bit in zip(HEADS, TABLE[task.item_id]) if not bit}
        invalid = graph.named[f"root/{INVALID_ROOT_SLOT}"]
        for nid, spec in graph.nodes.items():
            node_index += 1
            jobs.append({"node_id": nid, "record": dump_record(spec)})
            correct = nid not in wrong_heads
            content = ("I think the answer is 4." if nid == invalid
                       else f"<rationale>synthetic</rationale>\n<final>{_answer(task, private, correct)}</final>")
            cost = Decimal("0.000100") + Decimal(node_index) * Decimal("0.000001")
            use = Usage(prompt_tokens=100 + node_index % 50, completion_tokens=300 + node_index % 70,
                        reasoning_tokens=200 + node_index % 30, cached_prompt_tokens=0, actual_cost_usd=cost,
                        raw_usage={"cost": str(cost)})
            attempt_id = f"{nid}:1"
            result = dataclasses.replace(h.ok_result(spec, content, use=use), attempt_id=attempt_id,
                                         completion_status=CompletionStatus.COMPLETED)
            payload = {"attempt_id": attempt_id, "node_id": nid, "request_body": {"model": spec.model, "messages": []},
                       "response_status": 200, "response_headers": {"x-openrouter-cache-status": "MISS"},
                       "response_body_text": json.dumps(completion_payload(content, generation_id=result.generation_id,
                                                                            cost=float(cost))),
                       "result": dump_record(result)}
            raw_path = paths.raw_response(attempt_id)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            data = gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0)
            raw_path.write_bytes(data)
            result = dataclasses.replace(result, raw_response_path=paths.relative(raw_path),
                                         raw_response_sha256=sha256(data).hexdigest())
            status = ParseStatus.MISSING if nid == invalid else ParseStatus.VALID
            final = None if nid == invalid else _answer(task, private, correct)
            cand = Candidate(nid, nid, task.item_id, content, final, None, status, "stop", task.output_kind)
            parsed = paths.parsed(nid)
            parsed.parent.mkdir(parents=True, exist_ok=True)
            parsed.write_text(json.dumps({"result": dump_record(result), "candidate": dump_record(cand)}, sort_keys=True))
            grade = (GradeResult(nid, GradeStatus.INVALID_OUTPUT, None, False, "synthetic-v1", private.private_hash,
                                 detail_code="missing_final") if nid == invalid else
                     GradeResult(nid, GradeStatus.GRADED, correct, True, "synthetic-v1", private.private_hash))
            grade_path = paths.grade(nid)
            grade_path.parent.mkdir(parents=True, exist_ok=True)
            grade_path.write_text(json.dumps(dump_record(grade), sort_keys=True))
            for kind, amount in ((LedgerEventKind.RESERVE, Decimal("0.001")), (LedgerEventKind.SENT, Decimal("0")),
                                 (LedgerEventKind.SETTLE, cost)):
                seq += 1
                journal.append(dump_record(LedgerEvent(seq, kind, amount, h.T0, attempt_id=attempt_id, node_id=nid,
                                                       partition_id="main")))
        anc_order = list(graph.nodes)
        for name, rc in graph.checkpoints.items():
            anc = graph.ancestors_of(name)
            required = tuple(n for n in anc_order if n in anc)
            for view in (CheckpointView.CURRENT, CheckpointView.ARCHIVE):
                ids = rc.candidate_ids if view is CheckpointView.CURRENT else required
                cp = PolicyCheckpoint(f"main/{task.item_id}/{name}/{view.value}", task.item_id, rc.family, rc.team_size,
                                      rc.total_cycles, rc.peer_rounds, tuple(ids), required,
                                      "public_tests_only" if task.benchmark == "mbpp_plus" else "plurality", "v1",
                                      "main", name, view, 1024)
                policies.append(dump_record(cp))
                if view is not CheckpointView.CURRENT:
                    continue
                single = rc.family in ("private_single", "deep", "pooled")
                selections.append(SelectionResult(cp.policy_id, ids[0], "synthetic_first", "v1",
                                                  canonical_hash(list(ids)), None, 0.0,
                                                  SelectionStatus.SINGLE_CANDIDATE if single else SelectionStatus.SELECTED))
        summary["items"][task.item_id] = {"nodes": len(graph.nodes), "heads": dict(zip(HEADS, TABLE[task.item_id]))}
    for bench, rows in publics.items():
        _write_jsonl(paths.public_manifest(bench), rows)
    for bench, rows in privates.items():
        _write_jsonl(paths.private_manifest(bench), rows)
    _write_jsonl(paths.jobs("main"), jobs)
    _write_jsonl(paths.policies("main"), policies)
    _write_jsonl(paths.ledger_journal, journal)
    body = b"".join((json.dumps(dump_record(s), sort_keys=True) + "\n").encode() for s in sorted(selections, key=lambda s: s.policy_id))
    if selection_layout == "v2":
        sel_path, seal_path = paths.selections("main", 0), paths.selection_seal("main", 0)
        seal = {"sha256": sha256(body).hexdigest(), "count": len(selections), "sealed_at": h.T0, "previous_seal_sha256": None}
    else:
        sel_path = paths.run_dir / "selections" / "main.jsonl"
        seal_path = paths.run_dir / "selections" / "main.seal.json"
        seal = {"sha256": sha256(body).hexdigest(), "count": len(selections), "sealed_at": h.T0}
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_bytes(body)
    seal_path.write_text(json.dumps(seal, sort_keys=True))
    summary["selection_file"] = sel_path
    summary["nodes_total"] = len(jobs)
    os.chmod(paths.grades_dir, 0o700)
    return summary
