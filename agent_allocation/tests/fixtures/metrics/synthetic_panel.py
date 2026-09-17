"""Typed synthetic panel for agent 06 tests and agent 07 fixtures (no network, no model, no real gold).

``build_panel`` lays out one item graph per ``ItemSpec`` with the reference-DAG naming of ``tools/reference_dag.py``
(``sample/k``, ``private_single/t``, ``private_team/n/r``, ``peer/n/r``, ``deep/m``, ``cp/h``, ``pooled/4``) on a
small ``MiniGrid``, CURRENT and ARCHIVE views, parsed candidates, generation results/specs, simulated grades and
(code) public checks. ``select_panel`` runs the real selectors on public inputs only; grades are simulated from each
item's fixture answer key and are never passed to a selector.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from hashlib import sha256
from typing import Callable, Mapping, Sequence

from agent_allocation.contracts import (
    STUDY_ID, Candidate, CheckpointView, CheckStatus, CompletionStatus, GenerationKind, GenerationResult,
    GenerationSpec, GradeResult, GradeStatus, OutputKind, ParseStatus, PolicyCheckpoint, PublicCheckResult,
    PublicTask, SelectionResult, TransportStatus, Usage, public_task_hash,
)
from agent_allocation.metrics.tables import ItemRecords, PartitionRecords
from agent_allocation.selection.keys import vote_key
from agent_allocation.selection.readout import DECLARED_SELECTOR_VERSION, select_checkpoint

SEED = 20260917
MISSING_FINAL = "<missing-final>"       # answer marker: the response has no <final> element
TRUNCATED = "<truncated>"               # answer marker: finish_reason length with empty content


@dataclass(frozen=True)
class MiniGrid:
    bank: int = 6
    private_cycles: int = 6
    team_sizes: tuple[int, ...] = (2, 4)
    rounds: int = 2
    deep: tuple[int, ...] = (2, 4)
    extension: bool = True
    base_budget: int = 1024


@dataclass(frozen=True)
class ItemSpec:
    task: PublicTask
    answers: Mapping[str, str]                      # node name -> final answer or a marker; default answer below
    default_answer: str
    gold: frozenset[str] = frozenset()              # text: exact vote keys graded correct; code: see code_correct
    unknown_grades: frozenset[str] = frozenset()    # node names whose grade is EVALUATOR_ERROR
    ungraded: frozenset[str] = frozenset()          # node names with no grade file yet
    missing: frozenset[str] = frozenset()           # node names that never produced a candidate
    code_correct: frozenset[str] = frozenset()      # code: node names that pass private tests
    public: Mapping[str, tuple[int, CheckStatus]] = field(default_factory=dict)  # code: name -> (passed of 2, status)
    default_public: tuple[int, CheckStatus] | None = (1, CheckStatus.OK)          # code: every other node


@dataclass(frozen=True)
class Panel:
    partition: str
    records: PartitionRecords
    grades: dict[str, GradeResult]
    public_checks: dict[str, PublicCheckResult]
    node_ids: dict[tuple[str, str], str]            # (item_id, node name) -> node id


def node_names(grid: MiniGrid) -> tuple[list[tuple[str, str, tuple[str, ...], int]], dict[str, tuple[str, ...]]]:
    """(name, kind, parents, budget) in topological order, and checkpoint name -> candidate node names."""
    nodes: list[tuple[str, str, tuple[str, ...], int]] = []
    cps: dict[str, tuple[str, ...]] = {}
    a = grid.base_budget

    def add(name: str, kind: str, parents: tuple[str, ...] = (), budget: int = a) -> str:
        nodes.append((name, kind, parents, budget))
        return name

    roots = tuple(add(f"root/{j}", "root") for j in range(grid.bank))
    for k in range(1, grid.bank + 1):
        cps[f"sample/{k}"] = roots[:k]
    width = max(grid.team_sizes + ((4,) if grid.extension else ()))
    private = {(j, 0): roots[j] for j in range(width)}
    cps["private_single/1"] = (roots[0],)
    for r in range(1, grid.private_cycles):
        private[0, r] = add(f"private/0/{r}", "private", (private[0, r - 1],))
        cps[f"private_single/{r + 1}"] = (private[0, r],)
    for j in range(1, width):
        for r in range(1, grid.rounds + 1):
            private[j, r] = add(f"private/{j}/{r}", "private", (private[j, r - 1],))
    for n in grid.team_sizes:
        for r in range(grid.rounds + 1):
            cps[f"private_team/{n}/{r}"] = tuple(private[j, r] for j in range(n))
    for n in grid.team_sizes:
        prev = roots[:n]
        cps[f"peer/{n}/0"] = prev
        for r in range(1, grid.rounds + 1):
            prev = tuple(add(f"peer/{n}/{r}/{j}", "peer", prev) for j in range(n))
            cps[f"peer/{n}/{r}"] = prev
    for m in grid.deep:
        cps[f"deep/{m}"] = (add(f"deep/{m}", "deep", (), a * m),)
    if grid.extension:
        prev = cps["peer/4/1"]
        cps["cp/0"] = prev
        for h in range(1, 4):
            prev = tuple(add(f"cp/{h}/{j}", "communicate_then_private", (prev[j],)) for j in range(4))
            cps[f"cp/{h}"] = prev
        cps["pooled/4"] = (add("pooled/4", "pooled", roots[:4]),)
    return nodes, cps


def _family(name: str) -> str:
    return name.split("/")[0]


def _coords(name: str) -> dict[str, int]:
    parts = name.split("/")
    family, numbers = parts[0], [int(p) for p in parts[1:]]
    if family in ("private_team", "peer"):
        return {"team_size": numbers[0], "peer_rounds": numbers[1] if family == "peer" else 0,
                "total_cycles": numbers[1] + 1}
    if family == "cp":
        return {"team_size": 4, "peer_rounds": 1, "total_cycles": 2 + numbers[0]}
    if family == "pooled":
        return {"team_size": 4, "peer_rounds": 0, "total_cycles": 1}
    if family == "private_single":
        return {"team_size": 1, "peer_rounds": 0, "total_cycles": numbers[0]}
    return {"team_size": 1 if family == "deep" else numbers[0], "peer_rounds": 0, "total_cycles": 1}


def _content(answer: str, kind: OutputKind) -> tuple[str | None, str, CompletionStatus]:
    if answer == TRUNCATED:
        return None, "length", CompletionStatus.LENGTH
    if answer == MISSING_FINAL:
        return "<rationale>ran out of room</rationale>", "stop", CompletionStatus.COMPLETED
    body = f"```python\n{answer}\n```" if kind is OutputKind.PYTHON else answer
    return f"<rationale>fixture</rationale>\n<final>\n{body}\n</final>", "stop", CompletionStatus.COMPLETED


def build_panel(items: Sequence[ItemSpec], grid: MiniGrid = MiniGrid(), *, partition: str = "main",
                hashed_ids: bool = False, archive_views: bool = True) -> Panel:
    from agent_allocation.extraction.final import extract_candidate

    nodes, cps = node_names(grid)
    item_records: list[ItemRecords] = []
    grades: dict[str, GradeResult] = {}
    checks: dict[str, PublicCheckResult] = {}
    ids: dict[tuple[str, str], str] = {}
    for spec_item in items:
        task = spec_item.task
        node_ids: dict[str, str] = {}
        specs: dict[str, GenerationSpec] = {}
        candidates: dict[str, Candidate] = {}
        results: dict[str, GenerationResult] = {}
        for name, kind, parents, budget in nodes:
            readable = f"{task.item_id}:{name}"
            nid = sha256(readable.encode()).hexdigest() if hashed_ids else readable
            node_ids[name] = ids[(task.item_id, name)] = nid
            gen = GenerationSpec(run_id=STUDY_ID, partition_id=partition, item_id=task.item_id,
                                 benchmark=task.benchmark, kind=GenerationKind(kind), model="fixture-model",
                                 endpoint_lock_hash="lock", input_manifest_hash=public_task_hash(task),
                                 decoding_config_hash="decoding", prompt_version="prompts-v1-fixture",
                                 reasoning_budget=budget, completion_headroom=1024, root_slot=0, team_size=0,
                                 update_round=0, draw_id=0, parent_ids=tuple(node_ids[p] for p in parents))
            specs[nid] = gen
            if name in spec_item.missing:
                continue
            content, finish, completion = _content(spec_item.answers.get(name, spec_item.default_answer),
                                                   task.output_kind)
            usage = Usage(100, 50, None, None, Decimal("0.00002"))
            result = GenerationResult(node_id=gen.node_id, attempt_id=f"{nid}:1", transport_status=TransportStatus.OK,
                                      completion_status=completion, generation_id="gen", observed_model="fixture-model",
                                      observed_provider="fixture", finish_reason=finish, native_finish_reason=None,
                                      visible_content=content, reasoning_returned=False, usage=usage,
                                      cache_status="MISS", raw_response_path=None, raw_response_sha256=None,
                                      latency_seconds=0.1, http_status=200)
            candidate = replace(extract_candidate(result, gen, task), candidate_id=nid, node_id=nid)
            candidates[nid], results[nid] = candidate, replace(result, node_id=nid)
            public = spec_item.public.get(name, spec_item.default_public)
            if task.output_kind is OutputKind.PYTHON and public is not None:
                passed, status = public
                outcomes = tuple(["pass"] * passed + ["fail"] * (2 - passed)) if status is CheckStatus.OK else ()
                checks[nid] = PublicCheckResult(candidate_id=nid, item_id=task.item_id,
                                                code_sha256=sha256((candidate.final_answer or "").encode()).hexdigest(),
                                                status=status, cases_total=2, cases_passed=passed,
                                                case_outcomes=outcomes, checker_version="public-fixture-1",
                                                public_inputs_sha256="public-inputs", environment_digest="env",
                                                cpu_seconds=0.0)
            if name in spec_item.ungraded:
                continue
            grades[nid] = _grade(spec_item, name, candidate)
        checkpoints = []
        views = (CheckpointView.CURRENT, CheckpointView.ARCHIVE) if archive_views else (CheckpointView.CURRENT,)
        for cp_name, members in cps.items():
            pool = tuple(node_ids[m] for m in members)
            required = _ancestors(members, nodes, node_ids)
            for view in views:
                if view is CheckpointView.ARCHIVE and _family(cp_name) == "sample":
                    continue
                candidate_ids = pool if view is CheckpointView.CURRENT else required
                checkpoints.append(PolicyCheckpoint(
                    policy_id=f"{partition}/{task.item_id}/{cp_name}/{view.value}", item_id=task.item_id,
                    family=_family(cp_name), candidate_ids=candidate_ids, required_node_ids=required,
                    selector_name="public_tests_only" if task.output_kind is OutputKind.PYTHON else "plurality",
                    selector_version=DECLARED_SELECTOR_VERSION, partition_id=partition, name=cp_name, view=view,
                    reasoning_budget=grid.base_budget * (int(cp_name.split("/")[1]) if cp_name.startswith("deep")
                                                         else 1),
                    **_coords(cp_name)))
        item_records.append(ItemRecords(task=task, checkpoints=tuple(checkpoints), candidates=candidates,
                                        results=results, specs=specs))
    return Panel(partition=partition, records=PartitionRecords(partition, tuple(item_records)), grades=grades,
                 public_checks=checks, node_ids=ids)


def _ancestors(members: Sequence[str], nodes, node_ids: Mapping[str, str]) -> tuple[str, ...]:
    parents = {name: p for name, _, p, _ in nodes}
    found: set[str] = set()
    todo = list(members)
    while todo:
        name = todo.pop()
        if name not in found:
            found.add(name)
            todo.extend(parents[name])
    return tuple(node_ids[name] for name, _, _, _ in nodes if name in found)


def _grade(item: ItemSpec, name: str, candidate: Candidate) -> GradeResult:
    task = item.task
    base = dict(candidate_id=candidate.candidate_id, grader_version="fixture-grader-1", private_eval_hash="fixture")
    if name in item.unknown_grades:
        return GradeResult(status=GradeStatus.EVALUATOR_ERROR, semantic_correct=None, operational_success=True,
                           detail_code="fixture_evaluator_outage", **base)
    if candidate.parse_status is not ParseStatus.VALID:
        return GradeResult(status=GradeStatus.INVALID_OUTPUT, semantic_correct=None, operational_success=False,
                           detail_code=candidate.parse_status.value, **base)
    if task.output_kind is OutputKind.PYTHON:
        if name in item.public and item.public[name][1] is CheckStatus.INVALID_CANDIDATE:
            return GradeResult(status=GradeStatus.INVALID_OUTPUT, semantic_correct=None, operational_success=False,
                               detail_code="unparseable_program", **base)
        correct = name in item.code_correct
    else:
        key = vote_key(candidate, task)
        if key is None:                             # like grading.text: a final naming no legal answer
            return GradeResult(status=GradeStatus.INVALID_OUTPUT, semantic_correct=None, operational_success=False,
                               detail_code="not_a_legal_answer", **base)
        correct = key.exact in item.gold
    return GradeResult(status=GradeStatus.GRADED, semantic_correct=correct, operational_success=True, **base)


def select_panel(panel: Panel, *, seed: int = SEED, math_equivalent: Callable[[str, str], bool | None] | None = None,
                 zero_cpu: bool = True) -> list[SelectionResult]:
    """Run the real selectors on the panel's public inputs (never its grades)."""
    out = []
    for item in panel.records.items:
        checks = {cid: c for cid, c in panel.public_checks.items() if c.item_id == item.task.item_id}
        for cp in item.checkpoints:
            if any(cid not in item.candidates for cid in cp.candidate_ids):
                continue                                   # not terminal yet: no selection row
            result = select_checkpoint(cp, item.task, item.candidates, checks, seed=seed,
                                       math_equivalent=math_equivalent if item.task.output_kind is OutputKind.MATH
                                       else None)
            out.append(replace(result, cpu_seconds=0.0) if zero_cpu else result)
    return out


# ------------------------------------------------------------------------------------------------ standard panel

HALF = frozenset({"\\frac{1}{2}", "0.5", "1/2"})


def fixture_math_equivalent(a: str, b: str) -> bool | None:
    """Gold-blind stand-in for agent 04's Math-Verify helper: equal within one fixed class, None for "?"."""
    if "?" in (a, b):
        return None
    classes = ({"\\frac{1}{2}", "\\dfrac{1}{2}", "0.5", "1/2"}, {"2", "2.0"})
    return a == b or any(a in c and b in c for c in classes)


def standard_items() -> list[ItemSpec]:
    from agent_allocation.contracts import PublicTask

    def task(item_id, benchmark, kind, choices=(), subtask=None):
        return PublicTask(item_id=item_id, benchmark=benchmark, source_revision="fixture-rev", prompt="fixture",
                          output_kind=kind, allowed_choices=choices, subtask=subtask)

    math = ItemSpec(
        task=task("math500/fixture-1", "math500", OutputKind.MATH), default_answer="\\frac{1}{2}", gold=HALF,
        answers={"root/0": "\\frac{1}{2}", "root/1": "0.5", "root/2": "\\boxed{\\dfrac{1}{2}}", "root/3": "2",
                 "root/4": "2.0", "root/5": MISSING_FINAL, "private/0/1": "2", "private/0/2": "?",
                 "peer/2/1/0": "2", "peer/2/1/1": "2", "peer/4/1/2": "2", "cp/2/1": "2", "deep/2": "2"},
        unknown_grades=frozenset({"private/0/3"}))
    option = ItemSpec(
        task=task("mmlu_pro/fixture-2", "mmlu_pro", OutputKind.OPTION, tuple("ABCDEFGHIJ")), default_answer="C",
        gold=frozenset({"C"}),
        answers={"root/0": "B", "root/1": "C", "root/2": "B", "root/3": "(C)", "root/4": "K", "root/5": "(C)",
                 "private/0/1": "C", "private/0/2": "B", "private/0/3": "C", "private/0/4": TRUNCATED,
                 "peer/2/1/0": "B", "peer/2/1/1": "B", "peer/4/1/0": "B", "peer/4/1/1": "B", "peer/4/1/2": "B",
                 "peer/4/1/3": "B",
                 "peer/4/2/2": "B", "peer/4/2/3": "B", "cp/2/0": MISSING_FINAL, "cp/3/0": "B", "cp/3/1": "B",
                 "cp/3/2": "B"})
    boolean = ItemSpec(
        task=task("bbh_logic4/fixture-3", "bbh_logic4", OutputKind.BOOLEAN, subtask="boolean_expressions"),
        default_answer="False", gold=frozenset({"True"}),
        answers={"root/4": "True", "peer/4/2/1": "True", "deep/4": "true", "pooled/4": "True", "private/1/2": "True"})
    temporal = ItemSpec(
        task=task("bbh_logic4/fixture-4", "bbh_logic4", OutputKind.OPTION, ("(A)", "(B)", "(C)", "(D)"),
                  subtask="temporal_sequences"),
        default_answer="(A)", gold=frozenset({"A"}),
        answers={"root/0": TRUNCATED, "root/1": MISSING_FINAL, "root/2": TRUNCATED, "root/3": "(A)", "root/4": "D",
                 "private/0/1": "D", "private/1/1": MISSING_FINAL, "peer/2/1/0": "D", "peer/2/1/1": "D"},
        ungraded=frozenset({"peer/4/2/3"}), missing=frozenset({"cp/3/2"}))
    code = ItemSpec(
        task=task("mbpp_plus/fixture-5", "mbpp_plus", OutputKind.PYTHON), default_answer="def f(x):\n    return x + 1",
        answers={f"root/{j}": f"def f(x):\n    return x + {j}" for j in range(6)} | {"root/3": "def f(x:"},
        public={"root/0": (2, CheckStatus.OK), "root/1": (1, CheckStatus.OK), "root/2": (2, CheckStatus.OK),
                "root/3": (0, CheckStatus.INVALID_CANDIDATE), "root/4": (0, CheckStatus.INFRASTRUCTURE_ERROR),
                "root/5": (2, CheckStatus.OK), "peer/2/1/0": (2, CheckStatus.OK)},
        code_correct=frozenset({"root/2", "root/5", "peer/2/1/0", "private/0/2", "cp/1/3", "pooled/4"}))
    return [math, option, boolean, temporal, code]
