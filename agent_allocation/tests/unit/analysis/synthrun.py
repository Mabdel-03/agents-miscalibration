"""Synthetic run directories laid out per docs/00_contract.md §3, built only from contract records.

Agent 07's stand-in until agents 03 (item graphs/checkpoints) and 06 (selections/metric fixtures) land. The DAG
follows the coordinate table in ``GridSpec``'s docstring and the checkpoint names in ``tools/reference_dag.py``.

Correctness is controlled by ``correct_fn(item_index, benchmark, spec) -> True | False | None`` (None = invalid
output). Text candidates answer "A" when correct, "B"/"C" otherwise; the toy selector is plurality over valid
final answers with pool-order tie-break (candidate-only), last answer for single readouts, synthesis for POOLED.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Callable

from agent_allocation.contracts import (
    BBH_SUBTASKS, BENCHMARKS, STUDY_ID, Candidate, CheckpointView, CheckStatus, CompletionStatus, EvalKind,
    GenerationKind, GenerationResult, GenerationSpec, GradeResult, GradeStatus, LedgerEvent, LedgerEventKind,
    OutputKind, PanelDraw, ParseStatus, PolicyCheckpoint, PublicCheckResult, PublicTask, SelectionResult,
    SelectionStatus, TransportStatus, Usage, Wave, canonical_hash, dump_record, public_task_hash,
)
from agent_allocation.paths import RunPaths

IN_PRICE = Decimal("0.0000001")
OUT_PRICE = Decimal("0.0000004")
KIND_OF_BENCH = {"math500": OutputKind.MATH, "mmlu_pro": OutputKind.OPTION, "bbh_logic4": OutputKind.OPTION,
                 "mbpp_plus": OutputKind.PYTHON}


@dataclass
class Grid:
    bank_size: int = 5
    private_single_cycles: int = 5
    team_sizes: tuple[int, ...] = (2, 4)
    rounds: int = 4
    deep_multipliers: tuple[int, ...] = (2,)
    archive_views: bool = True


@dataclass
class SynthSpec:
    waves: int = 2
    grid: Grid = field(default_factory=Grid)
    extension_waves: int = 1
    high_waves: int = 0
    partition: str = "main"
    base_budget: int = 1024
    correct_fn: Callable[[int, str, GenerationSpec], bool | None] | None = None
    retry_every: int = 7            # every k-th node gets a billed failed first attempt
    seed: int = 20260917
    seal: bool = True
    grade: bool = True


def default_correct(index: int, benchmark: str, spec: GenerationSpec) -> bool | None:
    h = int(hashlib.sha256(f"{index}:{spec.node_id}".encode()).hexdigest()[:8], 16)
    if h % 37 == 0:
        return None
    base = {"math500": 55, "mmlu_pro": 65, "bbh_logic4": 50, "mbpp_plus": 60}[benchmark]
    bonus = 3 * spec.update_round if spec.kind in (GenerationKind.PEER, GenerationKind.CP) else 0
    return (h % 100) < base + bonus


def item_ids(waves: int) -> list[tuple[str, str, str | None]]:
    out = []
    for w in range(waves):
        for bench in BENCHMARKS:
            for j in range(4):
                subtask = BBH_SUBTASKS[j] if bench == "bbh_logic4" else None
                out.append((f"{bench}/w{w}/i{j}", bench, subtask))
    return out


def _spec(common: dict, kind: GenerationKind, *, slot=0, n=0, r=0, budget=None, parents=()) -> GenerationSpec:
    return GenerationSpec(kind=kind, root_slot=slot, team_size=n, update_round=r,
                          reasoning_budget=budget or common["base"], parent_ids=tuple(parents),
                          **{k: v for k, v in common.items() if k != "base"})


def build_graph(task: PublicTask, partition: str, grid: Grid, base_budget: int, extension: bool
                ) -> tuple[list[GenerationSpec], list[PolicyCheckpoint]]:
    common = dict(run_id=STUDY_ID, partition_id=partition, item_id=task.item_id, benchmark=task.benchmark,
                  model="google/gemini-2.5-flash-lite", endpoint_lock_hash="synthetic-lock",
                  input_manifest_hash=public_task_hash(task), decoding_config_hash="synthetic-decoding",
                  prompt_version="prompts-v1-synthetic", completion_headroom=2048 if task.benchmark == "mbpp_plus"
                  else 1024, draw_id=0, base=base_budget)
    nodes: list[GenerationSpec] = []
    cps: list[PolicyCheckpoint] = []

    def add(spec: GenerationSpec) -> GenerationSpec:
        nodes.append(spec)
        return spec

    by_id: dict[str, GenerationSpec] = {}

    def ancestors(ids) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        todo = list(ids)
        while todo:
            nid = todo.pop(0)
            if nid not in seen:
                seen[nid] = None
                todo.extend(by_id[nid].parent_ids)
        order = {s.node_id: i for i, s in enumerate(nodes)}
        return tuple(sorted(seen, key=order.__getitem__))

    def checkpoint(name, family, pool, *, n=1, cycles=1, peer=0, view=CheckpointView.CURRENT, budget=base_budget,
                   selector=None):
        pool_ids = tuple(s.node_id for s in pool)
        cps.append(PolicyCheckpoint(
            policy_id=f"{partition}/{task.item_id}/{name}/{view.value}", item_id=task.item_id, family=family,
            team_size=n, total_cycles=cycles, peer_rounds=peer, candidate_ids=pool_ids,
            required_node_ids=ancestors(pool_ids), selector_name=selector or _selector(task, family, len(pool)),
            selector_version="synthetic-1", partition_id=partition, name=name, view=view, reasoning_budget=budget))

    roots = [add(_spec(common, GenerationKind.ROOT, slot=j)) for j in range(grid.bank_size)]
    for s in roots:
        by_id[s.node_id] = s
    for k in range(1, grid.bank_size + 1):
        checkpoint(f"sample/{k}", "sample", roots[:k], n=k)
    chain = [roots[0]]
    for t in range(1, grid.private_single_cycles):
        s = add(_spec(common, GenerationKind.PRIVATE, slot=0, r=t, parents=(chain[-1].node_id,)))
        by_id[s.node_id] = s
        chain.append(s)
    for t in range(1, grid.private_single_cycles + 1):
        checkpoint(f"private_single/{t}", "private_single", [chain[t - 1]], cycles=t)
        if grid.archive_views:
            checkpoint(f"private_single/{t}", "private_single", chain[:t], cycles=t, view=CheckpointView.ARCHIVE)
    max_n = max(grid.team_sizes)
    private = {(0, r): chain[r] for r in range(min(grid.rounds, grid.private_single_cycles - 1) + 1)}
    for j in range(1, max_n):
        private[j, 0] = roots[j]
        for r in range(1, grid.rounds + 1):
            s = add(_spec(common, GenerationKind.PRIVATE, slot=j, r=r, parents=(private[j, r - 1].node_id,)))
            by_id[s.node_id] = s
            private[j, r] = s
    for n in grid.team_sizes:
        for r in range(grid.rounds + 1):
            pool = [private[j, r] for j in range(n)]
            checkpoint(f"private_team/{n}/{r}", "private_team", pool, n=n, cycles=1 + r)
            if grid.archive_views:
                arch = [private[j, q] for q in range(r + 1) for j in range(n)]
                checkpoint(f"private_team/{n}/{r}", "private_team", arch, n=n, cycles=1 + r,
                           view=CheckpointView.ARCHIVE)
    peer_rounds: dict[tuple[int, int], list[GenerationSpec]] = {}
    for n in grid.team_sizes:
        prev = roots[:n]
        peer_rounds[n, 0] = prev
        checkpoint(f"peer/{n}/0", "peer", prev, n=n, cycles=1)
        for r in range(1, grid.rounds + 1):
            current = []
            for j in range(n):
                s = add(_spec(common, GenerationKind.PEER, slot=j, n=n, r=r, parents=[p.node_id for p in prev]))
                by_id[s.node_id] = s
                current.append(s)
            peer_rounds[n, r] = current
            checkpoint(f"peer/{n}/{r}", "peer", current, n=n, cycles=1 + r, peer=r)
            if grid.archive_views:
                arch = [x for q in range(r + 1) for x in peer_rounds[n, q]]
                checkpoint(f"peer/{n}/{r}", "peer", arch, n=n, cycles=1 + r, peer=r, view=CheckpointView.ARCHIVE)
            prev = current
    for m in grid.deep_multipliers:
        s = add(_spec(common, GenerationKind.DEEP, budget=base_budget * m))
        by_id[s.node_id] = s
        checkpoint(f"deep/{m}", "deep", [s], budget=base_budget * m)
    if extension and 4 in grid.team_sizes and grid.rounds >= 1:
        prev = peer_rounds[4, 1]
        checkpoint("cp/0", "cp", prev, n=4, cycles=2, peer=1)
        cp_all = list(prev)
        for h in range(1, 4):
            current = []
            for j in range(4):
                s = add(_spec(common, GenerationKind.CP, slot=j, n=4, r=h, parents=(prev[j].node_id,)))
                by_id[s.node_id] = s
                current.append(s)
            cp_all.extend(current)
            checkpoint(f"cp/{h}", "cp", current, n=4, cycles=2 + h, peer=1)
            if grid.archive_views:
                checkpoint(f"cp/{h}", "cp", cp_all, n=4, cycles=2 + h, peer=1, view=CheckpointView.ARCHIVE)
            prev = current
        s = add(_spec(common, GenerationKind.POOLED, n=4, parents=[x.node_id for x in roots[:4]]))
        by_id[s.node_id] = s
        checkpoint("pooled/4", "pooled", [s], n=4, cycles=2)
    return nodes, cps


def _selector(task: PublicTask, family: str, pool_size: int) -> str:
    if family in ("private_single", "deep", "pooled") or pool_size == 1:
        return "last_answer"
    return "public_tests_only" if task.benchmark == "mbpp_plus" else "plurality"


def build_run(root: Path, spec: SynthSpec | None = None) -> RunPaths:
    spec = spec or SynthSpec()
    correct_fn = spec.correct_fn or default_correct
    paths = RunPaths.for_run(root / "run", locks_dir=root / "locks", local_state_dir=root / "local")
    paths.ensure()
    items = item_ids(spec.waves)
    tasks = {iid: PublicTask(iid, bench, "synthetic-rev", f"Question {iid}?", KIND_OF_BENCH[bench],
                             allowed_choices=("A", "B", "C") if bench in ("mmlu_pro", "bbh_logic4") else (),
                             subtask=sub)
             for iid, bench, sub in items}
    _write_jsonl(paths.public_manifest("x").parent, None, None)
    for bench in BENCHMARKS:
        _write_lines(paths.public_manifest(bench), [dump_record(t) for t in tasks.values() if t.benchmark == bench])
    panels = {}
    for bench in BENCHMARKS:
        ids = tuple(i for i, b, _ in items if b == bench)
        panels[bench] = dump_record(PanelDraw(bench, (), ids, (), spec.seed, "synthetic_v1"))
    paths.panels.write_text(json.dumps(panels, sort_keys=True))
    waves = []
    for w in range(spec.waves):
        labels = ["main"] + (["extension"] if w < spec.extension_waves else []) + (["high"] if w < spec.high_waves
                                                                                     else [])
        waves.append(dump_record(Wave(w, tuple(i for i, _, _ in items if f"/w{w}/" in i), tuple(labels))))
    paths.waves.write_text(json.dumps(waves, sort_keys=True))

    partitions = [(spec.partition, spec.base_budget, spec.waves)]
    if spec.high_waves:
        partitions.append(("high", spec.base_budget * 4, spec.high_waves))
    ledger: list[LedgerEvent] = []
    for partition, budget, n_waves in partitions:
        jobs, policies, selections = [], [], []
        for index, (iid, bench, _) in enumerate(items):
            wave = int(iid.split("/w")[1].split("/")[0])
            if wave >= n_waves:
                continue
            task = tasks[iid]
            nodes, cps = build_graph(task, partition, spec.grid, budget,
                                     extension=(partition == "main" and wave < spec.extension_waves))
            outcome: dict[str, bool | None] = {}
            for node in nodes:
                jobs.append({"node_id": node.node_id, "record": dump_record(node)})
                value = correct_fn(index, bench, node)
                outcome[node.node_id] = value
                _write_node(paths, task, node, value, ledger, spec, len(ledger))
            policies.extend(dump_record(cp) for cp in cps)
            for cp in cps:
                selections.append(dump_record(_select(cp, task, outcome, paths)))
        _write_lines(paths.jobs(partition), jobs)
        _write_lines(paths.policies(partition), policies)
        sel_path = paths.selections(partition)
        _write_lines(sel_path, selections)
        if spec.seal:
            payload = sel_path.read_bytes()
            paths.selection_seal(partition).write_text(json.dumps(
                {"sha256": hashlib.sha256(payload).hexdigest(), "count": len(selections),
                 "sealed_at": "2026-09-17T08:00:00Z"}))
    _write_lines(paths.ledger_journal, [dump_record(e) for e in ledger])
    return paths


def _answer(value: bool | None, node: GenerationSpec, task: PublicTask) -> str | None:
    if value is None:
        return None
    if task.benchmark == "mbpp_plus":
        return "def f():\n    return 1\n" if value else f"def f():\n    return {int(node.node_id[:4], 16) % 3 + 2}\n"
    return "A" if value else ("B" if int(node.node_id[:2], 16) % 2 else "C")


def _write_node(paths: RunPaths, task: PublicTask, node: GenerationSpec, value: bool | None,
                ledger: list[LedgerEvent], spec: SynthSpec, seq0: int) -> None:
    prompt = 300 + 250 * len(node.parent_ids)
    completion = min(node.max_completion_tokens, 500 + int(node.node_id[:3], 16) % 700)
    reasoning = max(0, completion - 200)
    cost = IN_PRICE * prompt + OUT_PRICE * completion
    reservation = (IN_PRICE * (prompt + 64) + OUT_PRICE * node.max_completion_tokens) * Decimal("1.1")
    seq = len(ledger)
    attempt_no = 1
    if spec.retry_every and int(node.node_id[:6], 16) % spec.retry_every == 0:
        failed = f"{node.node_id}:1"
        ledger.extend([
            LedgerEvent(seq + 1, LedgerEventKind.RESERVE, reservation, "2026-09-17T07:00:00Z", failed, node.node_id,
                        node.partition_id),
            LedgerEvent(seq + 2, LedgerEventKind.SENT, Decimal("0"), "2026-09-17T07:00:01Z", failed, node.node_id,
                        node.partition_id),
            LedgerEvent(seq + 3, LedgerEventKind.SETTLE, Decimal("0.0000123"), "2026-09-17T07:00:02Z", failed,
                        node.node_id, node.partition_id),
        ])
        seq += 3
        attempt_no = 2
    attempt = f"{node.node_id}:{attempt_no}"
    ledger.extend([
        LedgerEvent(seq + 1, LedgerEventKind.RESERVE, reservation, "2026-09-17T07:00:03Z", attempt, node.node_id,
                    node.partition_id),
        LedgerEvent(seq + 2, LedgerEventKind.SENT, Decimal("0"), "2026-09-17T07:00:04Z", attempt, node.node_id,
                    node.partition_id),
        LedgerEvent(seq + 3, LedgerEventKind.SETTLE, cost, "2026-09-17T07:00:05Z", attempt, node.node_id,
                    node.partition_id),
    ])
    answer = _answer(value, node, task)
    content = f"<final>{answer}</final>" if answer is not None else "no final element"
    usage = Usage(prompt, completion, reasoning, 0, cost, {"cost": str(cost)})
    raw_rel = f"raw/{node.node_id[:2]}/{attempt.replace(':', '_')}.json.gz"
    raw_path = paths.raw_response(attempt)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt") as handle:
        json.dump({"attempt_id": attempt, "node_id": node.node_id,
                   "request_body": {"messages": [{"role": "user", "content": "x" * (prompt * 3)}]}}, handle)
    result = GenerationResult(node.node_id, attempt, TransportStatus.OK, CompletionStatus.COMPLETED, "gen",
                              node.model, "Google AI Studio", "stop", "STOP", content, True, usage, "MISS",
                              raw_rel, "0" * 64, 1.5 + len(node.parent_ids), 200)
    candidate = Candidate(node.node_id, node.node_id, node.item_id, content, answer, None,
                          ParseStatus.VALID if answer is not None else ParseStatus.MISSING, "stop", task.output_kind)
    parsed = paths.parsed(node.node_id)
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text(json.dumps({"result": dump_record(result), "candidate": dump_record(candidate)}))
    if task.benchmark == "mbpp_plus":
        check = paths.public_check(node.node_id)
        check.parent.mkdir(parents=True, exist_ok=True)
        passed = 1 if value else 0
        check.write_text(json.dumps(dump_record(PublicCheckResult(
            node.node_id, node.item_id, "c" * 64 if answer else None,
            CheckStatus.OK if answer else CheckStatus.INVALID_CANDIDATE, 1, passed, ("pass",) if passed else ("fail",),
            "checker-1", "p" * 64, "env", 0.25))))
    if spec.grade:
        grade_path = paths.grade(node.node_id)
        grade_path.parent.mkdir(parents=True, exist_ok=True)
        if value is None:
            grade = GradeResult(node.node_id, GradeStatus.INVALID_OUTPUT, False, False, "grader-1", "h")
        else:
            grade = GradeResult(node.node_id, GradeStatus.GRADED, bool(value), True, "grader-1", "h",
                                cpu_seconds=0.5)
        grade_path.write_text(json.dumps(dump_record(grade)))


def _select(cp: PolicyCheckpoint, task: PublicTask, outcome: dict[str, bool | None], paths: RunPaths
            ) -> SelectionResult:
    pool = list(cp.candidate_ids)
    answers = {nid: _answer_from_disk(paths, nid) for nid in pool}
    pool_hash = canonical_hash(pool)
    if len(pool) == 1 or cp.family in ("private_single", "deep", "pooled"):
        return SelectionResult(cp.policy_id, pool[-1], "last_answer", "synthetic-1", pool_hash, None, 0.0,
                               SelectionStatus.SINGLE_CANDIDATE.value)
    valid = [nid for nid in pool if answers[nid] is not None]
    if not valid:
        return SelectionResult(cp.policy_id, None, cp.selector_name, "synthetic-1", pool_hash, None, 0.0,
                               SelectionStatus.ABSTAINED_ALL_INVALID.value)
    counts: dict[str, int] = {}
    for nid in valid:
        counts[answers[nid]] = counts.get(answers[nid], 0) + 1
    best = max(counts.values())
    chosen = next(nid for nid in valid if counts[answers[nid]] == best)
    return SelectionResult(cp.policy_id, chosen, cp.selector_name, "synthetic-1", pool_hash, None, 0.01,
                           SelectionStatus.SELECTED.value)


def _answer_from_disk(paths: RunPaths, node_id: str) -> str | None:
    return json.loads(paths.parsed(node_id).read_text())["candidate"]["final_answer"]


def _write_lines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def _write_jsonl(directory: Path, _a, _b) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def reseal(paths: RunPaths, partition: str) -> None:
    payload = paths.selections(partition).read_bytes()
    count = sum(1 for line in payload.decode().splitlines() if line.strip())
    paths.selection_seal(partition).write_text(json.dumps(
        {"sha256": hashlib.sha256(payload).hexdigest(), "count": count, "sealed_at": "2026-09-17T08:00:00Z"}))


__all__ = ["Grid", "SynthSpec", "build_run", "build_graph", "default_correct", "item_ids", "reseal", "replace"]
