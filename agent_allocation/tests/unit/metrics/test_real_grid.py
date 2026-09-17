"""Agent 03's real item graphs (tests/fixtures/protocols/two_item_run) through selection, sealing and every table.

The synthetic run is gold-free; this test adds deterministic public checks (public evidence) and simulated grades
(joined only after sealing) so the production DAG shape -- ``deep/1``, ARCHIVE views of every checkpoint, CP/POOLED
on the extension grid -- is exercised against every metrics invariant with strict populations.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from agent_allocation.contracts import (
    Candidate, CheckStatus, GenerationResult, GradeResult, GradeStatus, ItemGraph, OutputKind, ParseStatus,
    PublicCheckResult, PublicTask, SelectionStatus, load_record,
)
from agent_allocation.metrics.tables import ItemRecords, PartitionRecords, build_partition_tables
from agent_allocation.paths import RunPaths
from agent_allocation.selection.plurality import memoized_equivalence
from agent_allocation.selection.readout import concrete_selector_is_legal, select_checkpoint
from agent_allocation.selection.seal import load_sealed_selections, seal_selections

RUN = Path(__file__).resolve().parents[2] / "fixtures" / "protocols" / "two_item_run"
SEED = 20260917
GOOD_CODE = "def similar_elements(a, b):\n    return set(a) & set(b)"


def _records(name, cls):
    return [load_record(cls, json.loads(line)) for line in gzip.decompress((RUN / name).read_bytes()).splitlines()]


@pytest.fixture(scope="module")
def run():
    if not (RUN / "graphs.jsonl.gz").exists():
        pytest.skip("agent 03's two-item synthetic run is not present")
    tasks = {t.item_id: t for t in _records("tasks.jsonl.gz", PublicTask)}
    graphs = _records("graphs.jsonl.gz", ItemGraph)
    candidates = {c.candidate_id: c for c in _records("candidates.jsonl.gz", Candidate)}
    results = {r.node_id: r for r in _records("results.jsonl.gz", GenerationResult)}
    return tasks, graphs, candidates, results


def _public_check(candidate: Candidate, task: PublicTask) -> PublicCheckResult:
    passed = int(candidate.final_answer == GOOD_CODE)
    return PublicCheckResult(candidate_id=candidate.candidate_id, item_id=task.item_id, code_sha256="sha",
                             status=CheckStatus.OK, cases_total=1, cases_passed=passed,
                             case_outcomes=("pass",) if passed else ("fail",), checker_version="synthetic-public",
                             public_inputs_sha256="inputs", environment_digest="env", cpu_seconds=0.0)


def _grade(candidate: Candidate, task: PublicTask) -> GradeResult:
    base = dict(candidate_id=candidate.candidate_id, grader_version="synthetic", private_eval_hash="synthetic")
    if candidate.parse_status is not ParseStatus.VALID:
        return GradeResult(status=GradeStatus.INVALID_OUTPUT, semantic_correct=None, operational_success=False, **base)
    if task.output_kind is OutputKind.PYTHON:
        correct = candidate.final_answer == GOOD_CODE
    else:
        correct = candidate.final_answer in {"1", "\\frac{2}{2}"}
    return GradeResult(status=GradeStatus.GRADED, semantic_correct=correct, operational_success=True, **base)


def test_real_graphs_select_seal_and_join(run, tmp_path):
    tasks, graphs, candidates, results = run
    paths = RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")
    equivalent = memoized_equivalence(lambda a, b: {a, b} <= {"1", "\\frac{2}{2}"} or a == b)
    checks = {cid: _public_check(c, tasks[c.item_id]) for cid, c in candidates.items()
              if tasks[c.item_id].output_kind is OutputKind.PYTHON and c.parse_status is ParseStatus.VALID}
    selections = []
    for graph in graphs:
        task = tasks[graph.item_id]
        for cp in graph.checkpoints:
            result = select_checkpoint(cp, task, candidates, checks, seed=SEED,
                                       math_equivalent=equivalent if task.output_kind is OutputKind.MATH else None)
            assert concrete_selector_is_legal(cp.selector_name, result.selector_name)
            selections.append(result)
    assert len(selections) == sum(len(g.checkpoints) for g in graphs) == 390
    statuses = Counter(s.status for s in selections)
    assert statuses[SelectionStatus.SELECTED] and statuses[SelectionStatus.SINGLE_CANDIDATE]
    seal_selections(paths, "main", selections[:200])
    seal_selections(paths, "main", selections)                       # second batch: only the unsealed rest
    sealed = load_sealed_selections(paths, "main")
    assert len(sealed.seals) == 2 and len(sealed.by_policy) == 390

    grades = {cid: _grade(c, tasks[c.item_id]) for cid, c in candidates.items()}
    records = PartitionRecords("main", tuple(
        ItemRecords(task=tasks[g.item_id], checkpoints=g.checkpoints,
                    candidates={n.node_id: candidates[n.node_id] for n in g.nodes},
                    results={n.node_id: results[n.node_id] for n in g.nodes},
                    specs={n.node_id: n for n in g.nodes}) for g in graphs))
    tables = build_partition_tables(sealed, records, grades)
    assert len(tables.checkpoints) == 390
    assert {r.delivered_state for r in tables.checkpoints} >= {"correct", "wrong"}
    assert all(r.delivered_state != "missing_selection" for r in tables.checkpoints)
    assert Counter(r.item_id for r in tables.bank) == {"math500/synthetic/0001": 25, "mbpp_plus/synthetic/0002": 25}
    assert {r.n for r in tables.root_strata} == {2, 3, 4, 5}
    families = Counter((r.item_id, r.family) for r in tables.team_transitions)
    assert families[("math500/synthetic/0001", "cp")] == 3 and families[("math500/synthetic/0001", "peer")] == 16
    assert families[("math500/synthetic/0001", "private_single")] == 24
    pooled = [r for r in tables.checkpoints if r.family == "pooled" and r.view == "current"]
    assert len(pooled) == 1 and not pooled[0].candidate_only and pooled[0].input_bank_coverage_hi is not None
    kinds = {r.generation_kind for r in tables.failures}
    assert kinds == {"root", "private", "peer", "deep", "communicate_then_private", "pooled"}
    assert sum(r.n_nodes for r in tables.failures) == 138 + 125
