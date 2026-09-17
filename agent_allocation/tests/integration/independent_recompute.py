"""Agent 08's independent recomputation from raw per-item files (R07-3).

Called by agent 07's ``allocation audit`` (loaded by file path) as ``independent_recompute(run_dir)`` and by agent 08's
tests. It reads only the on-disk contract formats (docs/00_contract.md sections 3 and 6): job and policy manifests,
``parsed/``, ``raw/`` (sha256 and response body re-verified), ``grades/``, sealed selection files and the ledger
journal. It never imports analysis, metrics, selection, grading or protocol modules, and it never recomputes a
selection: it verifies selections and joins private grades only after the seals verify.

Public API
  independent_recompute(run_dir) -> {"contrasts": {name: float | None}, "checks": [{"name", "passed", "detail"}],
                                     "details": {...}}
  recompute_primary_contrasts(run_dir) -> {name: {"estimate", "lower", "upper", "per_benchmark", "n_items", ...}}
  recompute_item(run_dir, item_id) -> per-item trace: nodes, raw verification, selections, delivered correctness,
                                      coverage and policy cost for every checkpoint

Estimation rules (MASTER_SPEC sections 11-12): delivered Y is 1/0 from the private grade of the selected candidate;
INVALID_OUTPUT and all-invalid abstention count 0 (operational failure); evaluator errors, missing grades, missing or
paused selections are unknown. A contrast is ``None`` while any required Y in its fixed population is unknown, with
fixed-denominator lower/upper bounds reported beside it. Items weigh equally within a benchmark, except BBH, where
subtasks weigh equally; the headline value is the equal-benchmark macro average.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent_allocation.contracts import (
    BENCHMARKS,
    Candidate,
    CheckpointView,
    GenerationResult,
    GenerationSpec,
    GradeResult,
    GradeStatus,
    LedgerEvent,
    LedgerEventKind,
    ParseStatus,
    PolicyCheckpoint,
    PublicTask,
    SelectionResult,
    SelectionStatus,
    load_record,
)
from agent_allocation.paths import RunPaths

PRIMARY_I = "primary_i_peer_vs_private_team_n4_r4"
PRIMARY_II = "primary_ii_cp_minus_private_team_gain_h3"
CANDIDATE_ONLY = {"sample", "private_single", "private_team", "peer", "deep", "cp"}
_KEY_RE = re.compile(rb"sk-or-(?:v\d+-)?[A-Za-z0-9_\-]{16,}")


def _reference_impl():
    name = "agent08_reference_impl"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("reference_impl.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Run:
    paths: RunPaths
    config: dict[str, Any] | None
    tasks: dict[str, PublicTask] = field(default_factory=dict)
    jobs: dict[str, dict[str, GenerationSpec]] = field(default_factory=dict)          # partition -> node_id -> spec
    policies: dict[str, list[PolicyCheckpoint]] = field(default_factory=dict)         # partition -> checkpoints
    parsed: dict[str, tuple[GenerationResult, Candidate]] = field(default_factory=dict)
    grades: dict[str, GradeResult] = field(default_factory=dict)
    selections: dict[str, dict[str, SelectionResult]] = field(default_factory=dict)   # partition -> policy -> sel
    seal_problems: list[str] = field(default_factory=list)
    load_problems: list[str] = field(default_factory=list)
    journal: list[LedgerEvent] = field(default_factory=list)

    def spec(self, node_id: str) -> GenerationSpec | None:
        for nodes in self.jobs.values():
            if node_id in nodes:
                return nodes[node_id]
        return None


def _paths(run_dir: str | Path) -> RunPaths:
    run_dir = Path(run_dir)
    return RunPaths.for_run(run_dir, local_state_dir=run_dir / "state" / "agent08_unused_local")


def _config(paths: RunPaths) -> dict[str, Any] | None:
    if not paths.config_lock.is_file():
        return None
    data = json.loads(paths.config_lock.read_text())
    return data.get("config", data) if isinstance(data, dict) else None


def _selection_files(paths: RunPaths) -> dict[str, list[tuple[Path, Path]]]:
    """partition -> [(jsonl, seal)] for v2 batches (``selections/<p>/batch_NNNN``) and v1 (``selections/<p>.jsonl``)."""
    root = paths.run_dir / "selections"
    out: dict[str, list[tuple[Path, Path]]] = {}
    if not root.is_dir():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for batch in sorted(sub.glob("batch_*.jsonl")):
            out.setdefault(sub.name, []).append((batch, batch.with_name(batch.name[:-len(".jsonl")] + ".seal.json")))
    for flat in sorted(root.glob("*.jsonl")):
        out.setdefault(flat.stem, []).append((flat, flat.with_name(flat.stem + ".seal.json")))
    return out


def load(run_dir: str | Path) -> Run:
    paths = _paths(run_dir)
    run = Run(paths, _config(paths))
    for bench_file in sorted((paths.manifests_dir / "public").glob("*.jsonl")):
        for row in _jsonl(bench_file):
            task = load_record(PublicTask, row)
            run.tasks[task.item_id] = task
    for job_file in sorted((paths.manifests_dir / "jobs").glob("*.jsonl")):
        nodes: dict[str, GenerationSpec] = {}
        for row in _jsonl(job_file):
            spec = load_record(GenerationSpec, row["record"])
            if spec.node_id != row["node_id"]:
                run.load_problems.append(f"jobs/{job_file.name}: node_id {row['node_id'][:12]} != spec hash {spec.node_id[:12]}")
            if spec.node_id in nodes:
                run.load_problems.append(f"jobs/{job_file.name}: duplicate node {spec.node_id[:12]}")
            nodes[spec.node_id] = spec
        run.jobs[job_file.stem] = nodes
    for pol_file in sorted((paths.manifests_dir / "policies").glob("*.jsonl")):
        run.policies[pol_file.stem] = [load_record(PolicyCheckpoint, row) for row in _jsonl(pol_file)]
    for nodes in run.jobs.values():
        for node_id in nodes:
            parsed = paths.parsed(node_id)
            if parsed.is_file():
                data = json.loads(parsed.read_text())
                run.parsed[node_id] = (load_record(GenerationResult, data["result"]), load_record(Candidate, data["candidate"]))
            grade = paths.grade(node_id)
            if grade.is_file():
                run.grades[node_id] = load_record(GradeResult, json.loads(grade.read_text()))
    for partition, files in _selection_files(paths).items():
        by_policy: dict[str, SelectionResult] = {}
        previous = None
        for body_path, seal_path in files:
            body = body_path.read_bytes()
            rel = paths.relative(body_path)
            if not seal_path.is_file():
                run.seal_problems.append(f"{rel}: no seal; selections not joined")
                continue
            seal = json.loads(seal_path.read_text())
            lines = [line for line in body.splitlines() if line.strip()]
            if seal.get("sha256") != sha256(body).hexdigest() or seal.get("count") != len(lines):
                run.seal_problems.append(f"{rel}: seal mismatch (sha256/count); selections not joined")
                continue
            if "previous_seal_sha256" in seal and seal["previous_seal_sha256"] != previous:
                run.seal_problems.append(f"{rel}: broken seal chain")
            previous = sha256(seal_path.read_bytes()).hexdigest()
            for line in lines:
                sel = load_record(SelectionResult, json.loads(line))
                if sel.policy_id in by_policy:
                    run.seal_problems.append(f"{rel}: policy {sel.policy_id} sealed twice")
                by_policy[sel.policy_id] = sel
        run.selections[partition] = by_policy
    if paths.ledger_journal.is_file():
        run.journal = [load_record(LedgerEvent, row) for row in _jsonl(paths.ledger_journal)]
    return run


# ---------------------------------------------------------------------------------------------------------
# Delivered correctness and aggregation
# ---------------------------------------------------------------------------------------------------------

def candidate_state(run: Run, candidate_id: str) -> str:
    grade = run.grades.get(candidate_id)
    return _reference_impl().tri_state(grade)


def delivered(run: Run, partition: str, policy_id: str) -> int | None:
    sel = run.selections.get(partition, {}).get(policy_id)
    if sel is None:
        return None
    status = SelectionStatus(sel.status)
    if status is SelectionStatus.ABSTAINED_ALL_INVALID:
        return 0
    if status is SelectionStatus.PAUSED_CHECK_ERROR or sel.selected_candidate_id is None:
        return None
    state = candidate_state(run, sel.selected_candidate_id)
    return {"correct": 1, "wrong": 0, "invalid": 0}.get(state)


def _checkpoints(run: Run, partition: str, view: CheckpointView = CheckpointView.CURRENT) -> dict[tuple[str, str], PolicyCheckpoint]:
    return {(cp.item_id, cp.name): cp for cp in run.policies.get(partition, []) if cp.view == view}


def _aggregate(run: Run, per_item: Mapping[str, tuple[Fraction | None, Fraction, Fraction]]) -> dict[str, Any]:
    """per_item: item -> (estimate or None, lower, upper). Equal items within benchmark, equal BBH subtasks, macro."""
    per_bench: dict[str, dict[str, Any]] = {}
    for bench in BENCHMARKS:
        items = [i for i in per_item if run.tasks.get(i) and run.tasks[i].benchmark == bench]
        if not items:
            continue
        groups: dict[str | None, list[str]] = {}
        for item in items:
            key = run.tasks[item].subtask if bench == "bbh_logic4" else None
            groups.setdefault(key, []).append(item)

        def mean_of(index: int) -> Fraction | None:
            means = []
            for members in groups.values():
                vals = [per_item[i][index] for i in members]
                if any(v is None for v in vals):
                    return None
                means.append(sum(vals, Fraction(0)) / len(vals))
            return sum(means, Fraction(0)) / len(means)

        per_bench[bench] = {"estimate": mean_of(0), "lower": mean_of(1), "upper": mean_of(2), "n_items": len(items),
                            "n_unknown_items": sum(per_item[i][0] is None for i in items),
                            "subtasks": sorted(k for k in groups if k) if bench == "bbh_logic4" else []}
    def macro(key: str) -> Fraction | None:
        vals = [b[key] for b in per_bench.values()]
        if not vals or any(v is None for v in vals):
            return None
        return sum(vals, Fraction(0)) / len(vals)
    return {"estimate": macro("estimate"), "lower": macro("lower"), "upper": macro("upper"),
            "per_benchmark": per_bench, "n_items": len(per_item),
            "n_unknown_items": sum(v[0] is None for v in per_item.values())}


def _linear(terms: Iterable[tuple[int, int | None]]) -> tuple[Fraction | None, Fraction, Fraction]:
    terms = list(terms)
    known = sum((Fraction(c * v) for c, v in terms if v is not None), Fraction(0))
    lower = known + sum((Fraction(min(c, 0)) for c, v in terms if v is None), Fraction(0))
    upper = known + sum((Fraction(max(c, 0)) for c, v in terms if v is None), Fraction(0))
    estimate = known if all(v is not None for _, v in terms) else None
    return estimate, lower, upper


def recompute_primary_contrasts(run_dir: str | Path, *, _run: Run | None = None) -> dict[str, dict[str, Any]]:
    run = _run or load(run_dir)
    cps = _checkpoints(run, "main")
    items_i = sorted({item for item, name in cps if name == "peer/4/4"} & {item for item, name in cps if name == "private_team/4/4"})
    per_i = {}
    for item in items_i:
        per_i[item] = _linear([(1, delivered(run, "main", cps[item, "peer/4/4"].policy_id)),
                               (-1, delivered(run, "main", cps[item, "private_team/4/4"].policy_id))])
    items_ii = sorted(item for item, name in cps if name == "cp/3")
    per_ii = {}
    for item in items_ii:
        needed = ("cp/3", "cp/0", "private_team/4/4", "private_team/4/1")
        if any((item, n) not in cps for n in needed):
            per_ii[item] = (None, Fraction(-2), Fraction(2))
            continue
        y = {n: delivered(run, "main", cps[item, n].policy_id) for n in needed}
        per_ii[item] = _linear([(1, y["cp/3"]), (-1, y["cp/0"]), (-1, y["private_team/4/4"]), (1, y["private_team/4/1"])])

    def finish(agg: dict[str, Any]) -> dict[str, Any]:
        def f(v):
            return None if v is None else float(v)
        out = {k: f(agg[k]) for k in ("estimate", "lower", "upper")}
        out["exact"] = None if agg["estimate"] is None else str(agg["estimate"])
        out["n_items"], out["n_unknown_items"] = agg["n_items"], agg["n_unknown_items"]
        out["per_benchmark"] = {b: {"estimate": f(v["estimate"]), "lower": f(v["lower"]), "upper": f(v["upper"]),
                                    "n_items": v["n_items"], "n_unknown_items": v["n_unknown_items"],
                                    "subtasks": v["subtasks"]} for b, v in agg["per_benchmark"].items()}
        return out
    return {PRIMARY_I: finish(_aggregate(run, per_i)), PRIMARY_II: finish(_aggregate(run, per_ii))}


# ---------------------------------------------------------------------------------------------------------
# Per-item trace
# ---------------------------------------------------------------------------------------------------------

def _verify_raw(run: Run, node_id: str) -> list[str]:
    result, candidate = run.parsed[node_id]
    problems = []
    if not result.raw_response_path:
        return [f"{node_id[:12]}: parsed result has no raw pointer"]
    raw_path = run.paths.run_dir / result.raw_response_path
    if not raw_path.is_file():
        return [f"{node_id[:12]}: raw file missing ({result.raw_response_path})"]
    data = raw_path.read_bytes()
    if result.raw_response_sha256 and sha256(data).hexdigest() != result.raw_response_sha256:
        problems.append(f"{node_id[:12]}: raw sha256 mismatch")
    try:
        payload = json.loads(gzip.decompress(data))
    except (OSError, ValueError) as exc:
        return problems + [f"{node_id[:12]}: raw not gzip JSON ({type(exc).__name__})"]
    if payload.get("node_id") != node_id or payload.get("attempt_id") != result.attempt_id:
        problems.append(f"{node_id[:12]}: raw identity differs from parsed")
    try:
        body = json.loads(payload.get("response_body_text") or "{}")
        content = body["choices"][0]["message"].get("content")
    except (ValueError, KeyError, IndexError, TypeError):
        content = None
    if content is not None and (content != result.visible_content or content != candidate.visible_content):
        problems.append(f"{node_id[:12]}: visible content differs between raw response and parsed record")
    if candidate.candidate_id != node_id or candidate.node_id != node_id or result.node_id != node_id:
        problems.append(f"{node_id[:12]}: candidate/result identity differs from node")
    return problems


def recompute_item(run_dir: str | Path, item_id: str, *, _run: Run | None = None) -> dict[str, Any]:
    run = _run or load(run_dir)
    ref = _reference_impl()
    task = run.tasks.get(item_id)
    out: dict[str, Any] = {"item_id": item_id, "benchmark": task.benchmark if task else None,
                           "subtask": task.subtask if task else None, "partitions": {}}
    for partition, nodes in run.jobs.items():
        mine = {nid: s for nid, s in nodes.items() if s.item_id == item_id}
        if not mine:
            continue
        raw_problems: list[str] = []
        verified = 0
        for nid in mine:
            if nid in run.parsed:
                problems = _verify_raw(run, nid)
                raw_problems += problems
                verified += not problems
        kinds: dict[str, int] = {}
        for spec in mine.values():
            kinds[spec.kind.value] = kinds.get(spec.kind.value, 0) + 1
        checkpoints = {}
        for cp in run.policies.get(partition, []):
            if cp.item_id != item_id:
                continue
            closure = ref.ancestors(nodes, cp.candidate_ids) if all(c in nodes for c in cp.candidate_ids) else None
            states = [candidate_state(run, c) for c in cp.candidate_ids]
            usage = [run.parsed[n][0].usage if n in run.parsed else None for n in cp.required_node_ids]
            costs = [u.actual_cost_usd if u is not None else None for u in usage]
            reasoning = [u.reasoning_tokens if u is not None else None for u in usage]
            sel = run.selections.get(partition, {}).get(cp.policy_id)
            checkpoints[f"{cp.name}/{cp.view.value}"] = {
                "policy_id": cp.policy_id, "family": cp.family, "pool": list(cp.candidate_ids),
                "required_matches_closure": None if closure is None else closure == set(cp.required_node_ids),
                "nominal_generations": len(set(cp.required_node_ids)),
                "cost_usd": None if any(c is None for c in costs) else str(sum(costs, Decimal(0))),
                "reasoning_tokens": None if any(r is None for r in reasoning) else sum(reasoning),
                "selection_status": None if sel is None else SelectionStatus(sel.status).value,
                "selected": None if sel is None else sel.selected_candidate_id,
                "delivered_correct": delivered(run, partition, cp.policy_id) if cp.view == CheckpointView.CURRENT else None,
                "coverage": 1 if "correct" in states else (None if "unknown" in states else 0),
                "correct_bounds": list(ref.correct_count_bounds(states)),
            }
        out["partitions"][partition] = {"nodes": len(mine), "kind_counts": kinds,
                                        "parsed": sum(1 for n in mine if n in run.parsed),
                                        "raw_verified": verified, "raw_problems": raw_problems,
                                        "graded": sum(1 for n in mine if n in run.grades), "checkpoints": checkpoints}
    return out


# ---------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------

def _check(name: str, problems: list[str] | None, detail_ok: str) -> dict[str, Any]:
    if problems is None:
        return {"name": name, "passed": None, "detail": detail_ok}
    return {"name": name, "passed": not problems,
            "detail": detail_ok if not problems else f"{len(problems)} problem(s): " + "; ".join(problems[:5])}


def _checks(run: Run) -> list[dict[str, Any]]:
    ref = _reference_impl()
    out = []
    out.append(_check("manifest_node_identity", list(run.load_problems),
                      f"{sum(len(n) for n in run.jobs.values())} job records hash to their node_id"))
    raw_problems = [p for nid in run.parsed for p in _verify_raw(run, nid)]
    out.append(_check("raw_matches_parsed", raw_problems, f"{len(run.parsed)} parsed records re-verified against raw"))

    closure_problems, pool_problems, missing_rows, coverage_problems, population = [], [], [], [], []
    for partition, cps in run.policies.items():
        nodes = run.jobs.get(partition, {})
        names: dict[str, set[str]] = {}
        for cp in cps:
            names.setdefault(f"{cp.name}/{cp.view.value}", set()).add(cp.item_id)
            if all(c in nodes for c in cp.candidate_ids):
                closure = ref.ancestors(nodes, cp.candidate_ids)
                if closure != set(cp.required_node_ids):
                    closure_problems.append(f"{cp.policy_id}: required {len(set(cp.required_node_ids))} != closure {len(closure)}")
            if cp.view != CheckpointView.CURRENT:
                continue
            sel = run.selections.get(partition, {}).get(cp.policy_id)
            terminal = all(n in run.parsed for n in cp.required_node_ids)
            if sel is None:
                if terminal and run.selections.get(partition):
                    missing_rows.append(cp.policy_id)
                continue
            status = SelectionStatus(sel.status)
            if status in (SelectionStatus.SELECTED, SelectionStatus.SINGLE_CANDIDATE) and sel.selected_candidate_id is None:
                pool_problems.append(f"{cp.policy_id}: {status.value} without a candidate")
            if cp.family in CANDIDATE_ONLY and sel.selected_candidate_id is not None \
                    and sel.selected_candidate_id not in cp.candidate_ids:
                pool_problems.append(f"{cp.policy_id}: selected candidate outside the declared pool")
            if cp.family in CANDIDATE_ONLY:
                y = delivered(run, partition, cp.policy_id)
                states = [candidate_state(run, c) for c in cp.candidate_ids]
                coverage_problems += ref.delivered_vs_coverage_violations([{
                    "policy_id": cp.policy_id, "family": cp.family, "delivered_correct": y == 1 if y is not None else None,
                    "pool_states": states}])
        for family_prefix in ("sample/", "private_single/", "private_team/", "peer/", "deep/", "cp/", "pooled/"):
            sets = [items for name, items in names.items() if name.startswith(family_prefix)]
            if sets and any(s != sets[0] for s in sets):
                population.append(f"{partition}: {family_prefix} checkpoints cover different item sets")
    out.append(_check("ancestor_closure", closure_problems if run.policies else None,
                      "required_node_ids equal the parent closure of every checkpoint pool"))
    out.append(_check("selection_seals", list(run.seal_problems) if run.selections else None,
                      f"{sum(len(v) for v in run.selections.values())} sealed selection rows verified"))
    out.append(_check("selections_within_pool", pool_problems if run.selections else None,
                      "every candidate-only selection names a member of its declared pool"))
    out.append(_check("selector_rows_present", missing_rows if run.selections else None,
                      "every fully generated current checkpoint has a selector row"))
    out.append(_check("delivered_le_coverage", coverage_problems if run.selections else None,
                      "candidate-only delivered accuracy never exceeds pool coverage (POOLED exempt)"))
    out.append(_check("fixed_population", population if run.policies else None,
                      "each checkpoint family covers one fixed item set per partition"))

    # pass@k identities on fully known 25-root banks; transitions; archive monotonicity
    passk, transitions, archive = [], [], []
    for partition, cps in run.policies.items():
        by_item: dict[str, dict[str, PolicyCheckpoint]] = {}
        for cp in cps:
            by_item.setdefault(cp.item_id, {})[f"{cp.name}/{cp.view.value}"] = cp
        for item, named in by_item.items():
            bank = named.get("sample/25/current")
            if bank is not None:
                states = [candidate_state(run, c) for c in bank.candidate_ids]
                if all(s in ("correct", "wrong", "invalid") for s in states):
                    c = states.count("correct")
                    if ref.pass_at_k(25, c, 1) != Fraction(c, 25) or ref.pass_at_k(25, c, 25) != (1 if c else 0):
                        passk.append(f"{item}: pass@k identity broken")
            for fam, sizes in (("peer", (2, 3, 4, 5)), ("private_team", (2, 3, 4, 5))):
                for n in sizes:
                    for r in range(1, 5):
                        prev, cur = named.get(f"{fam}/{n}/{r - 1}/current"), named.get(f"{fam}/{n}/{r}/current")
                        if prev is None or cur is None:
                            continue
                        a = [candidate_state(run, x) for x in prev.candidate_ids]
                        b = [candidate_state(run, x) for x in cur.candidate_ids]
                        t = ref.transition_counts(a, b)
                        if t.fully_known and t.invalid_involved == 0 and b.count("correct") - a.count("correct") != t.w_to_c - t.c_to_w:
                            transitions.append(f"{item} {fam}/{n}/{r}: C_next-C_prev != W_to_C-C_to_W")
            for prefix in ("sample/", "private_single/", "cp/"):
                chain = sorted(((int(k.split("/")[1]), v) for k, v in named.items()
                                if k.startswith(prefix) and k.endswith("/archive") and k.count("/") == 2), key=lambda x: x[0])
                last = None
                for _, cp in chain:
                    states = [candidate_state(run, x) for x in cp.candidate_ids]
                    if "unknown" in states:
                        last = None
                        continue
                    cov = int("correct" in states)
                    if last is not None and cov < last:
                        archive.append(f"{item} {cp.name}: archive coverage decreased")
                    last = cov
    have_grades = bool(run.grades)
    out.append(_check("pass_at_k_identities", passk if have_grades else None, "pass@1 = mean correctness; pass@25 = coverage"))
    out.append(_check("transition_identity", transitions if have_grades else None, "C_next - C_prev = W_to_C - C_to_W"))
    out.append(_check("archive_coverage_monotone", archive if have_grades else None, "archive coverage never decreases"))

    invalid = []
    for nid, (result, cand) in run.parsed.items():
        grade = run.grades.get(nid)
        if grade is not None and cand.parse_status is not ParseStatus.VALID and grade.status is GradeStatus.GRADED:
            invalid.append(f"{nid[:12]}: {cand.parse_status.value} candidate graded as a semantic answer")
    out.append(_check("invalid_not_graded", invalid if have_grades else None,
                      "unparseable outputs are INVALID_OUTPUT, never a graded wrong answer"))

    ceiling = Decimal(str(((run.config or {}).get("budget") or {}).get("credit_ceiling_usd", 135)))
    ledger_problems: list[str] | None = None
    if run.journal:
        attempts_with_connect_failure = {r.attempt_id for r, _ in run.parsed.values()
                                         if r.transport_status.value == "connect_failed"}
        state = ref.LedgerChecker(ceiling, connect_failed=attempts_with_connect_failure).replay(run.journal)
        ledger_problems = list(state.violations)
        settled_attempts = {e.attempt_id for e in run.journal if e.kind in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX)}
        for result, _ in run.parsed.values():
            if result.transport_status.value == "ok" and result.attempt_id not in settled_attempts:
                ledger_problems.append(f"{result.attempt_id[:14]}: successful result never settled in the ledger")
    out.append(_check("ledger_replay", ledger_problems, f"journal replays within ceiling {ceiling}"))

    leaked = []
    for path in sorted(p for p in run.paths.run_dir.rglob("*") if p.is_file()):
        data = path.read_bytes()
        if path.suffix == ".gz":
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        if _KEY_RE.search(data):
            leaked.append(run.paths.relative(path))
    out.append(_check("no_credentials_in_artifacts", leaked, "no OpenRouter key pattern in any run artifact"))
    return out


def independent_recompute(run_dir: str | Path) -> dict[str, Any]:
    run = load(run_dir)
    contrasts = recompute_primary_contrasts(run_dir, _run=run)
    return {"contrasts": {name: value["estimate"] for name, value in contrasts.items()},
            "checks": _checks(run), "details": {"contrasts": contrasts}}
