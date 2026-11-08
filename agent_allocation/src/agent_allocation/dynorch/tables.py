"""dynorch_v1, WP-C: run, call and sealed-schema tables (contract ``docs/dynorch/00_contract.md`` section 7).

Inputs are the ``RunRecord`` JSON files WP-A's ``RunStore`` writes under ``<run_root>/runs/`` (contract section 5,
``dump_record`` style: nested ``spec`` and ``calls_detail``) and the plan manifest WP-B writes to
``<run_root>/manifests/dynorch_plan_<stage>.jsonl`` (record shape defined in ``docs/dynorch/requests/wpC.md``).
Outputs, under ``<run_root>/tables/``:

* ``dynorch_runs.parquet`` -- one row per RunRecord file, every attempt included (plan section 7);
* ``dynorch_calls.parquet`` -- one row per CallRecord, joined to its run;
* ``policy_outcomes_dynorch.parquet`` -- EXACTLY the sealed 36 columns and dtypes of
  ``forced_budget_v2/artifacts/merged/tables/policy_outcomes.parquet``, one row per planned cell (a cell without a
  terminal record is ``delivered_state="pending"``, ``y=NaN``), so ``analysis/stats.build_population`` never drops
  an item and ``core.py``'s ``C = B*R*A`` and ``rung = C // 1024`` hold (B10: ``team_size=1``, ``total_cycles=1``,
  ``reasoning_budget=cap``);
* ``policy_costs_dynorch.parquet`` -- the sealed 39 columns (``completion_tokens = completion_total + grace``,
  ``realized_tokens = completion_tokens``, ``requested_reasoning_tokens = cap``, standardized cost via
  ``analysis/costs.standardized_cost`` at the config's ``planning_prices``, ``cost_source="dynorch_ledger"``).

Nothing here writes into the sealed forced_budget_v2 run root; ``merged_outcomes`` / ``merged_costs`` concatenate in
memory. Every unexpected shape raises ``TablesError``; there is no silent fallback. Two explicit conventions for
quantities the sealed schema requires but a harness run does not have: ``reasoning_tokens`` is the proxy's
``reasoning_total`` when recorded and 0 when the raw ``/completions`` usage carried no reasoning split (counted and
returned by ``build_tables`` as ``reasoning_unknown_rows``; the report never uses harness reasoning tokens), and
``answer_tokens`` is the grace (the ladder's ANSWER call is also outside C).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation.analysis.costs import planning_prices, standardized_cost

A_TOKENS = 1024
FAMILY = "harness"
SELECTOR_NAME = "last_message"
SELECTOR_VERSION = "dynorch-v1"
SELECTOR_CONCRETE = "LAST_ANSWER"
COST_SOURCE = "dynorch_ledger"
VIEW = "current"
PANELS = "main"

#: Sealed ``policy_outcomes`` columns in sealed order (contract section 7).
OUTCOME_COLUMNS: tuple[str, ...] = (
    "partition_id", "policy_id", "item_id", "benchmark", "subtask", "stratum", "wave", "panels", "name", "view",
    "family", "team_size", "total_cycles", "peer_rounds", "reasoning_budget", "selector_name", "selector_version",
    "n_pool", "n_required", "n_required_terminal", "n_required_no_output", "n_pool_valid", "n_pool_correct",
    "n_pool_unknown", "coverage", "input_coverage", "selection_status", "selection_batch", "selector_concrete",
    "selected_candidate_id", "selected_in_pool", "sealed", "delivered_state", "y", "shard_id", "endpoint_tag",
)
#: Sealed ``policy_costs`` columns in sealed order (contract section 7).
COST_COLUMNS: tuple[str, ...] = (
    "policy_id", "partition_id", "item_id", "name", "view", "family", "nominal_generations",
    "nominal_generations_expected", "n_required_terminal", "requested_reasoning_tokens",
    "requested_completion_cap_tokens", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "cached_prompt_tokens", "realized_tokens", "success_cost_usd", "retry_cost_usd", "total_charged_usd",
    "standardized_uncached_cost_usd", "public_check_cpu_seconds", "selection_cpu_seconds",
    "private_grading_cpu_seconds", "total_latency_seconds", "critical_path_latency_seconds", "answer_tokens",
    "upstream_inference_cost_usd", "critical_path_generations", "results_missing", "transport_ok_nodes",
    "pool_size", "benchmark", "subtask", "stratum", "y", "delivered_state", "cost_source", "shard_id", "endpoint_tag",
)
# Pandas dtypes of the sealed tables (pandas 3: ``str`` is the arrow-backed string dtype the sealed files read back
# as; ``null`` arrow columns read back as object with None). Verified against the sealed parquet in test_tables.
_STR, _I64, _F64, _BOOL, _NULL = "str", "int64", "float64", "bool", "object"
OUTCOME_DTYPES: dict[str, str] = {
    **{c: _STR for c in OUTCOME_COLUMNS},
    "subtask": _NULL, "wave": _F64, "team_size": _I64, "total_cycles": _I64, "peer_rounds": _I64,
    "reasoning_budget": _I64, "n_pool": _I64, "n_required": _I64, "n_required_terminal": _I64,
    "n_required_no_output": _I64, "n_pool_valid": _I64, "n_pool_correct": _I64, "n_pool_unknown": _I64,
    "coverage": _F64, "input_coverage": _F64, "selection_batch": _I64, "selected_in_pool": _BOOL, "sealed": _BOOL,
    "y": _F64,
}
COST_DTYPES: dict[str, str] = {
    **{c: _STR for c in COST_COLUMNS},
    "nominal_generations": _I64, "nominal_generations_expected": _I64, "n_required_terminal": _I64,
    "requested_reasoning_tokens": _I64, "requested_completion_cap_tokens": _I64, "prompt_tokens": _I64,
    "completion_tokens": _I64, "reasoning_tokens": _I64, "cached_prompt_tokens": _I64, "realized_tokens": _I64,
    "public_check_cpu_seconds": _NULL, "selection_cpu_seconds": _F64, "private_grading_cpu_seconds": _F64,
    "total_latency_seconds": _F64, "critical_path_latency_seconds": _F64, "answer_tokens": _I64,
    "critical_path_generations": _I64, "results_missing": _I64, "transport_ok_nodes": _I64, "pool_size": _I64,
    "subtask": _NULL, "y": _F64,
}

#: RunSpec / RunRecord / CallRecord fields, contract section 5 (verbatim order).
RUN_SPEC_FIELDS: tuple[str, ...] = (
    "run_key", "arm", "harness", "item_id", "benchmark", "stratum", "shard_id", "endpoint_tag", "rung", "cap_tokens",
    "repeat", "attempt", "generation", "partition", "allowed_tools", "spawn_tools", "root_call_ceiling",
    "sampling_hash", "template_sha256", "prompt_version", "proxy_version", "harness_version",
)
RUN_RECORD_FIELDS: tuple[str, ...] = (
    "status", "calls", "conversations", "spawns", "calls_by_class", "completion_total", "reasoning_total",
    "prompt_total", "grace_tokens", "overdraft_tokens", "forced_close_kind", "exhausted", "refused_calls",
    "guard_triggered", "tool_parse_failures", "reinjection_hits", "reinjection_misses", "duplicates_served",
    "final_message", "final_source", "cli_result", "cli_result_mismatch", "parse_status", "final_answer",
    "grade_status", "correct", "grade_detail", "wall_seconds", "harness_exit_code", "harness_reported", "cost_usd",
    "started_at", "finished_at", "error", "rerun_reason",
)
CALL_RECORD_FIELDS: tuple[str, ...] = (
    "call_no", "conversation_key", "conversation_class", "requested_max_tokens", "granted_max_tokens",
    "queue_wait_seconds", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_prompt_tokens",
    "reasoning_chars", "visible_chars", "forced_close_kind", "grace_tokens", "finish_reason_upstream",
    "finish_reason_returned", "tool_calls", "spawn_calls", "tool_parse", "reinjection",
    "reasoning_effort_requested", "cost_usd", "generation_ids", "latency_seconds", "transport_status",
    "returned_status", "stream", "wire", "cache_status", "duplicate_served", "ignored_body_keys",
)
RUN_STATUSES = frozenset({"planned", "running", "done", "failed", "timeout", "transport_failed",
                          "refused_before_start"})
TERMINAL_STATUSES = frozenset({"done", "failed", "timeout", "transport_failed", "refused_before_start"})
INFRA_STATUSES = frozenset({"transport_failed", "refused_before_start"})
CALL_CLASSES = ("root", "solver", "other")
#: ``grade.finalize_run`` writes ``ParseStatus.VALID.value`` ("valid"), the sealed ladder's spelling, never "VALID".
PARSE_VALID = "valid"
DELIVERED_STATES = ("delivered", "op_no_output", "unknown", "pending")
PLAN_REQUIRED_KEYS: tuple[str, ...] = ("run_key", "arm", "harness", "item_id", "benchmark", "stratum", "shard_id",
                                       "endpoint_tag", "rung", "cap_tokens", "repeat", "attempt", "partition")
TABLE_NAMES = ("dynorch_runs", "dynorch_calls", "policy_outcomes_dynorch", "policy_costs_dynorch")


class TablesError(ValueError):
    """An input file, record or plan line whose shape the contract does not allow."""


@dataclass(frozen=True)
class PlannedCell:
    """One planned (arm, item, rung, repeat) cell from the plan manifest (contract section 6 ``plan_runs``)."""
    run_key: str
    arm: str
    harness: str
    item_id: str
    benchmark: str
    stratum: str
    shard_id: str
    endpoint_tag: str
    rung: int
    cap_tokens: int
    repeat: int
    attempt: int
    partition: str

    @property
    def name(self) -> str:
        return cell_name(self.arm, self.rung, self.repeat)

    @property
    def cell_key(self) -> tuple[str, str, str, int, int]:
        """``(partition, arm, item_id, rung, repeat)``: a cell is unique within its stage (the plan manifests of
        several stages are read together by ``build_tables``)."""
        return (self.partition, self.arm, self.item_id, self.rung, self.repeat)

    @property
    def policy_id(self) -> str:
        return policy_id(self.partition, self.item_id, self.name)


def cell_name(arm: str, rung: int, repeat: int) -> str:
    """Checkpoint name of a cell: ``"H1/16"``; repeat 2 and later ``"H1/16/r2"`` (contract section 7)."""
    if int(repeat) < 1:
        raise TablesError(f"repeat must be >= 1, got {repeat!r}")
    base = f"{arm}/{int(rung)}"
    return base if int(repeat) == 1 else f"{base}/r{int(repeat)}"


def policy_id(partition: str, item_id: str, name: str) -> str:
    """``policy_id = f"{partition}/{item_id}/{name}/current"`` (contract section 7)."""
    return f"{partition}/{item_id}/{name}/{VIEW}"


def run_key_of(study_id: str, arm: str, item_id: str, rung: int, repeat: int, attempt: int, *, partition: str) -> str:
    """Contract section 6 as amended by §10.22 (re-review minor 12): the partition is part of the identity --
    ``plan.run_key_for`` is the one formula (this used to hash without the partition, so synthetic plans and any
    reader using it disagreed with every real run_key)."""
    from agent_allocation.dynorch.plan import run_key_for
    return run_key_for(study_id, partition, arm, item_id, int(rung), int(repeat), int(attempt))


# ---------------------------------------------------------------------------------------------- readers

def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TablesError(f"{what}: expected an integer, got {value!r}")
    return int(value)


def _plan_line(obj: Any, source: str) -> PlannedCell:
    if not isinstance(obj, dict):
        raise TablesError(f"{source}: plan line is not a JSON object")
    spec = obj
    if "spec" in obj and isinstance(obj["spec"], dict):          # {"spec": RunSpec} wrapper
        spec = obj["spec"]
    tag = spec.get("__record__")
    if tag not in (None, "RunSpec"):
        raise TablesError(f"{source}: plan line tagged {tag!r}, expected RunSpec")
    missing = [k for k in PLAN_REQUIRED_KEYS if k not in spec]
    if missing:
        raise TablesError(f"{source}: plan line lacks {missing}")
    return PlannedCell(
        run_key=str(spec["run_key"]), arm=str(spec["arm"]), harness=str(spec["harness"]),
        item_id=str(spec["item_id"]), benchmark=str(spec["benchmark"]), stratum=str(spec["stratum"]),
        shard_id=str(spec["shard_id"]), endpoint_tag=str(spec["endpoint_tag"]),
        rung=_as_int(spec["rung"], f"{source}: rung"), cap_tokens=_as_int(spec["cap_tokens"], f"{source}: cap_tokens"),
        repeat=_as_int(spec["repeat"], f"{source}: repeat"), attempt=_as_int(spec["attempt"], f"{source}: attempt"),
        partition=str(spec["partition"]),
    )


def read_plan(path: str | Path) -> list[PlannedCell]:
    """Planned cells from ``manifests/dynorch_plan_<stage>.jsonl`` (one RunSpec-shaped object per line; a
    ``{"spec": {...}}`` wrapper is accepted). Duplicate cells (same arm/item/rung/repeat) are an error."""
    path = Path(path)
    if not path.is_file():
        raise TablesError(f"plan manifest not found: {path}")
    cells: list[PlannedCell] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TablesError(f"{path}:{lineno}: not JSON ({exc})") from None
            cell = _plan_line(obj, f"{path}:{lineno}")
            if cell.cell_key in seen:
                raise TablesError(f"{path}:{lineno}: duplicate planned cell {cell.cell_key}")
            seen.add(cell.cell_key)
            cells.append(cell)
    if not cells:
        raise TablesError(f"{path}: empty plan manifest")
    return cells


def validate_record(obj: Any, source: str = "<record>") -> dict[str, Any]:
    """Check a RunRecord JSON object against contract section 5 and return it (``__record__`` tags dropped)."""
    if not isinstance(obj, dict):
        raise TablesError(f"{source}: RunRecord is not a JSON object")
    tag = obj.get("__record__")
    if tag not in (None, "RunRecord"):
        raise TablesError(f"{source}: tagged {tag!r}, expected RunRecord")
    rec = {k: v for k, v in obj.items() if k != "__record__"}
    missing = [k for k in ("spec", *RUN_RECORD_FIELDS, "calls_detail") if k not in rec]
    if missing:
        raise TablesError(f"{source}: RunRecord lacks {missing}")
    spec = rec["spec"]
    if not isinstance(spec, dict):
        raise TablesError(f"{source}: RunRecord.spec is not an object")
    if spec.get("__record__") not in (None, "RunSpec"):
        raise TablesError(f"{source}: spec tagged {spec.get('__record__')!r}, expected RunSpec")
    spec = {k: v for k, v in spec.items() if k != "__record__"}
    missing = [k for k in RUN_SPEC_FIELDS if k not in spec]
    if missing:
        raise TablesError(f"{source}: RunSpec lacks {missing}")
    if rec["status"] not in RUN_STATUSES:
        raise TablesError(f"{source}: status {rec['status']!r} not in {sorted(RUN_STATUSES)}")
    if not isinstance(rec["calls_detail"], list):
        raise TablesError(f"{source}: calls_detail is not a list")
    calls = []
    for i, call in enumerate(rec["calls_detail"]):
        if not isinstance(call, dict):
            raise TablesError(f"{source}: calls_detail[{i}] is not an object")
        if call.get("__record__") not in (None, "CallRecord"):
            raise TablesError(f"{source}: calls_detail[{i}] tagged {call.get('__record__')!r}")
        call = {k: v for k, v in call.items() if k != "__record__"}
        missing = [k for k in CALL_RECORD_FIELDS if k not in call]
        if missing:
            raise TablesError(f"{source}: calls_detail[{i}] lacks {missing}")
        if call["conversation_class"] not in CALL_CLASSES:
            raise TablesError(f"{source}: calls_detail[{i}].conversation_class {call['conversation_class']!r}")
        calls.append(call)
    if not isinstance(rec["calls_by_class"], dict):
        raise TablesError(f"{source}: calls_by_class is not an object")
    for key in ("rung", "cap_tokens", "repeat", "attempt"):
        _as_int(spec[key], f"{source}: spec.{key}")
    rec["spec"] = spec
    rec["calls_detail"] = calls
    return rec


def read_run_records(run_root: str | Path) -> list[dict[str, Any]]:
    """Every ``*.json`` under ``<run_root>/runs/`` AND ``<run_root>/runs_superseded/`` as validated RunRecord
    objects, sorted by (arm, rung, repeat, item, attempt). A missing directory is an empty run.

    The superseded directory is where ``RunStore.archive_previous`` puts the earlier attempts of a cell (contract
    §5); reading only ``runs/`` meant every paid rerun was invisible to the tables -- ``retry_cost_usd`` was always
    0 and ``total_charged_usd == success_cost_usd``, so the $-frontier and the Gate 2/4 cost figures omitted the
    reruns the frozen rule says are "all journaled and paid" (review major 12).
    """
    root = Path(run_root)
    out: list[dict[str, Any]] = []
    for directory in (root / "runs", root / "runs_superseded"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                obj = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise TablesError(f"{path}: not JSON ({exc})") from None
            rec = validate_record(obj, str(path))
            rec["_source"] = str(path)
            out.append(rec)
    keys = [(r["spec"]["run_key"], r["_source"]) for r in out]
    if len({k for k, _ in keys}) != len(keys):
        seen: dict[str, str] = {}
        for key, source in keys:
            if key in seen:
                raise TablesError(f"run_key {key} appears twice: {seen[key]} and {source}")
            seen[key] = source
    out.sort(key=lambda r: (r["spec"]["arm"], int(r["spec"]["rung"]), int(r["spec"]["repeat"]),
                            r["spec"]["item_id"], int(r["spec"]["attempt"])))
    return out


# ---------------------------------------------------------------------------------------------- run / call tables

def _json_or_none(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def runs_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per RunRecord (every attempt): the spec's scalar fields, every scalar RunRecord field, the tuple
    fields as JSON strings, ``calls_<class>`` from ``calls_by_class`` and derived ``realized_share``,
    ``terminal``, ``valid_final``."""
    rows = []
    for rec in records:
        spec = rec["spec"]
        row: dict[str, Any] = {k: spec[k] for k in RUN_SPEC_FIELDS}
        for key in ("allowed_tools", "spawn_tools"):
            row[key] = json.dumps(list(spec[key]))
        row["name"] = cell_name(spec["arm"], spec["rung"], spec["repeat"])
        for key in RUN_RECORD_FIELDS:
            if key in ("calls_by_class", "harness_reported"):
                continue
            row[key] = rec[key]
        for cls in CALL_CLASSES:
            row[f"calls_{cls}"] = int(rec["calls_by_class"].get(cls, 0))
        row["harness_reported"] = _json_or_none(rec["harness_reported"])
        row["completion_with_grace"] = int(rec["completion_total"]) + int(rec["grace_tokens"])
        row["realized_share"] = float(rec["completion_total"]) / float(spec["cap_tokens"])
        row["terminal"] = rec["status"] in TERMINAL_STATUSES
        row["infra_failure"] = rec["status"] in INFRA_STATUSES
        row["valid_final"] = rec["parse_status"] == PARSE_VALID
        row["y"], row["delivered_state"] = outcome_of(rec)
        row["n_calls_detail"] = len(rec["calls_detail"])
        row["prompt_tokens_first_root"] = _first_root_prompt_tokens(rec)
        row["max_call_completion"] = max((int(c["completion_tokens"] or 0) for c in rec["calls_detail"]), default=0)
        row["cap_bound_calls"] = sum(1 for c in rec["calls_detail"]
                                     if c["completion_tokens"] is not None
                                     and int(c["completion_tokens"]) >= int(c["granted_max_tokens"]))
        row["source_file"] = rec.get("_source")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["arm", "rung", "repeat", "item_id", "attempt"]).reset_index(drop=True)
    return frame


def _first_root_prompt_tokens(rec: Mapping[str, Any]) -> int | None:
    for call in sorted(rec["calls_detail"], key=lambda c: int(c["call_no"])):
        if call["conversation_class"] == "root":
            return None if call["prompt_tokens"] is None else int(call["prompt_tokens"])
    return None


def calls_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per CallRecord with the run's identifying columns prepended (tuple fields as JSON strings)."""
    rows = []
    for rec in records:
        spec = rec["spec"]
        head = {k: spec[k] for k in ("run_key", "arm", "harness", "item_id", "benchmark", "rung", "cap_tokens",
                                     "repeat", "attempt", "partition", "shard_id")}
        head["name"] = cell_name(spec["arm"], spec["rung"], spec["repeat"])
        for call in rec["calls_detail"]:
            row = dict(head)
            for key in CALL_RECORD_FIELDS:
                value = call[key]
                if key in ("generation_ids", "ignored_body_keys"):
                    value = json.dumps(list(value))
                row[key] = value
            row["cap_bound"] = (call["completion_tokens"] is not None
                                and int(call["completion_tokens"]) >= int(call["granted_max_tokens"]))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["arm", "rung", "repeat", "item_id", "attempt", "call_no"]).reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------------------------- outcomes / costs

def outcome_of(rec: Mapping[str, Any] | None) -> tuple[float, str]:
    """``(y, delivered_state)`` of a cell's current record (B7; contract section 7 vocabulary).

    No record, or a non-terminal one -> ``pending`` / NaN; an infrastructure stop (``transport_failed``,
    ``refused_before_start``, i.e. the all-infra case of the rerun rule) -> ``unknown`` / NaN; otherwise the
    ladder's invalid-candidate rule: a VALID candidate is graded (True 1.0, False 0.0, evaluator unknown -> NaN
    ``unknown``) and anything not VALID is ``op_no_output`` with y = 0.
    """
    if rec is None or rec["status"] not in TERMINAL_STATUSES:
        return math.nan, "pending"
    if rec["status"] in INFRA_STATUSES:
        return math.nan, "unknown"
    if rec["parse_status"] != PARSE_VALID:
        return 0.0, "op_no_output"
    correct = rec["correct"]
    if correct is True:
        return 1.0, "delivered"
    if correct is False:
        return 0.0, "delivered"
    if correct is None:
        return math.nan, "unknown"
    raise TablesError(f"RunRecord.correct must be bool | None, got {correct!r}")


CellKey = tuple[str, str, str, int, int]


def current_records(plan: Sequence[PlannedCell], records: Sequence[Mapping[str, Any]]
                    ) -> tuple[dict[CellKey, Mapping[str, Any]], dict[CellKey, list[Mapping[str, Any]]]]:
    """Per planned cell: the record with the highest attempt (``current``) and the superseded earlier attempts.
    A record for a cell that is not in the plan is an error (nothing runs outside the plan)."""
    planned = {c.cell_key for c in plan}
    by_cell: dict[CellKey, list[Mapping[str, Any]]] = {}
    for rec in records:
        spec = rec["spec"]
        key = (spec["partition"], spec["arm"], spec["item_id"], int(spec["rung"]), int(spec["repeat"]))
        if key not in planned:
            raise TablesError(f"RunRecord {spec['run_key']} ({key}) is not a planned cell")
        by_cell.setdefault(key, []).append(rec)
    current: dict[CellKey, Mapping[str, Any]] = {}
    superseded: dict[CellKey, list[Mapping[str, Any]]] = {}
    for key, recs in by_cell.items():
        recs = sorted(recs, key=lambda r: int(r["spec"]["attempt"]))
        attempts = [int(r["spec"]["attempt"]) for r in recs]
        if len(set(attempts)) != len(attempts):
            raise TablesError(f"cell {key}: duplicate attempt numbers {attempts}")
        current[key] = recs[-1]
        superseded[key] = recs[:-1]
    return current, superseded


def policy_outcomes_table(plan: Sequence[PlannedCell], records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The sealed 36-column ``policy_outcomes`` frame with one row per planned cell (contract section 7)."""
    current, _ = current_records(plan, records)
    rows = []
    for cell in plan:
        rec = current.get(cell.cell_key)
        y, state = outcome_of(rec)
        terminal = rec is not None and rec["status"] in TERMINAL_STATUSES
        valid = rec is not None and rec["parse_status"] == PARSE_VALID
        correct = None if rec is None else rec["correct"]
        rows.append({
            "partition_id": cell.partition, "policy_id": cell.policy_id, "item_id": cell.item_id,
            "benchmark": cell.benchmark, "subtask": None, "stratum": cell.stratum, "wave": math.nan, "panels": PANELS,
            "name": cell.name, "view": VIEW, "family": FAMILY, "team_size": 1, "total_cycles": 1, "peer_rounds": 0,
            "reasoning_budget": cell.cap_tokens, "selector_name": SELECTOR_NAME, "selector_version": SELECTOR_VERSION,
            "n_pool": 1, "n_required": 1,
            "n_required_terminal": int(terminal and rec["status"] not in INFRA_STATUSES),
            "n_required_no_output": int(terminal and rec["final_message"] is None),
            "n_pool_valid": int(valid), "n_pool_correct": int(correct is True),
            "n_pool_unknown": int(state in ("pending", "unknown")),
            "coverage": math.nan if state in ("pending", "unknown") else float(correct is True),
            "input_coverage": math.nan,
            "selection_status": {"delivered": "selected", "op_no_output": "abstained_all_invalid"}.get(state, state),
            "selection_batch": 0, "selector_concrete": SELECTOR_CONCRETE,
            "selected_candidate_id": cell.run_key if rec is None else rec["spec"]["run_key"],
            "selected_in_pool": True, "sealed": bool(terminal), "delivered_state": state, "y": y,
            "shard_id": cell.shard_id if rec is None else rec["spec"]["shard_id"],
            "endpoint_tag": cell.endpoint_tag if rec is None else rec["spec"]["endpoint_tag"],
        })
    return enforce_schema(pd.DataFrame(rows, columns=list(OUTCOME_COLUMNS)), "policy_outcomes")


def _money(value: Any, what: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        raise TablesError(f"{what}: not a decimal string: {value!r}") from None


def policy_costs_table(plan: Sequence[PlannedCell], records: Sequence[Mapping[str, Any]],
                       price: Mapping[str, Decimal], *, grace_tokens: int = A_TOKENS
                       ) -> tuple[pd.DataFrame, int]:
    """The sealed 39-column ``policy_costs`` frame, one row per planned cell, and the number of terminal rows whose
    proxy record carried no reasoning split (their ``reasoning_tokens`` is 0; see the module docstring).

    ``success_cost_usd`` is the current attempt's ledger cost, ``retry_cost_usd`` the summed cost of superseded
    attempts of the same cell (all attempts are paid), ``total_charged_usd`` their sum;
    ``standardized_uncached_cost_usd`` prices ``prompt_total`` and ``completion_total + grace`` at ``price``.
    """
    current, superseded = current_records(plan, records)
    rows = []
    reasoning_unknown = 0
    for cell in plan:
        rec = current.get(cell.cell_key)
        y, state = outcome_of(rec)
        base = {"policy_id": cell.policy_id, "partition_id": cell.partition, "item_id": cell.item_id,
                "name": cell.name, "view": VIEW, "family": FAMILY, "nominal_generations": 1,
                "nominal_generations_expected": 1, "requested_reasoning_tokens": cell.cap_tokens,
                "requested_completion_cap_tokens": cell.cap_tokens + int(grace_tokens),
                "public_check_cpu_seconds": None, "selection_cpu_seconds": 0.0, "private_grading_cpu_seconds": math.nan,
                "pool_size": 1, "benchmark": cell.benchmark, "subtask": None, "stratum": cell.stratum, "y": y,
                "delivered_state": state, "cost_source": COST_SOURCE}
        retry = sum((_money(r["cost_usd"], f"{r['spec']['run_key']}.cost_usd") for r in superseded.get(cell.cell_key, [])),
                    Decimal(0))
        if rec is None or rec["status"] not in TERMINAL_STATUSES:
            rows.append({**base, "n_required_terminal": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "reasoning_tokens": 0, "cached_prompt_tokens": 0, "realized_tokens": 0,
                         "success_cost_usd": "0", "retry_cost_usd": str(retry), "total_charged_usd": str(retry),
                         "standardized_uncached_cost_usd": "0", "total_latency_seconds": math.nan,
                         "critical_path_latency_seconds": math.nan, "answer_tokens": 0,
                         "upstream_inference_cost_usd": "0", "critical_path_generations": 0, "results_missing": 1,
                         "transport_ok_nodes": 0, "shard_id": cell.shard_id, "endpoint_tag": cell.endpoint_tag})
            continue
        spec = rec["spec"]
        completion = int(rec["completion_total"]) + int(rec["grace_tokens"])
        if rec["reasoning_total"] is None:
            reasoning = 0
            reasoning_unknown += 1
        else:
            reasoning = int(rec["reasoning_total"])
        cached = sum(int(c["cached_prompt_tokens"] or 0) for c in rec["calls_detail"])
        latencies = [float(c["latency_seconds"]) for c in rec["calls_detail"]]
        success = _money(rec["cost_usd"], f"{spec['run_key']}.cost_usd")
        std = standardized_cost(int(rec["prompt_total"]), completion, price)
        infra = rec["status"] in INFRA_STATUSES
        rows.append({**base, "n_required_terminal": int(not infra), "prompt_tokens": int(rec["prompt_total"]),
                     "completion_tokens": completion, "reasoning_tokens": reasoning, "cached_prompt_tokens": cached,
                     "realized_tokens": completion, "success_cost_usd": str(success), "retry_cost_usd": str(retry),
                     "total_charged_usd": str(success + retry), "standardized_uncached_cost_usd": str(std),
                     "total_latency_seconds": float(sum(latencies)) if latencies else math.nan,
                     "critical_path_latency_seconds": float(rec["wall_seconds"]),
                     "answer_tokens": int(rec["grace_tokens"]), "upstream_inference_cost_usd": str(success),
                     "critical_path_generations": int(rec["calls"]), "results_missing": int(infra),
                     "transport_ok_nodes": int(not infra), "shard_id": spec["shard_id"],
                     "endpoint_tag": spec["endpoint_tag"]})
    return enforce_schema(pd.DataFrame(rows, columns=list(COST_COLUMNS)), "policy_costs"), reasoning_unknown


def enforce_schema(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reorder to the sealed column list and cast to the sealed dtypes; a missing or extra column raises."""
    columns, dtypes = ((OUTCOME_COLUMNS, OUTCOME_DTYPES) if kind == "policy_outcomes"
                       else (COST_COLUMNS, COST_DTYPES) if kind == "policy_costs" else (None, None))
    if columns is None:
        raise TablesError(f"unknown sealed table kind {kind!r}")
    extra = set(frame.columns) - set(columns)
    missing = set(columns) - set(frame.columns)
    if extra or missing:
        raise TablesError(f"{kind}: columns differ from the sealed schema (extra {sorted(extra)}, "
                          f"missing {sorted(missing)})")
    out = frame[list(columns)].copy()
    for col in columns:
        want = dtypes[col]
        if want == _NULL:
            if out[col].notna().any():
                raise TablesError(f"{kind}.{col}: the sealed column is null-typed; got non-null values")
            out[col] = pd.Series([None] * len(out), index=out.index, dtype=object)
        elif want == _STR:
            if out[col].isna().any():
                raise TablesError(f"{kind}.{col}: string column with missing values")
            out[col] = out[col].astype(str)
        elif want == _I64:
            if out[col].isna().any():
                raise TablesError(f"{kind}.{col}: int64 column with missing values")
            out[col] = out[col].astype("int64")
        elif want == _BOOL:
            out[col] = out[col].astype(bool)
        else:
            out[col] = out[col].astype("float64")
    return out


# ---------------------------------------------------------------------------------------------- building / merging

def price_of(cfg: Mapping[str, Any]) -> Mapping[str, Decimal]:
    """The planning price of the primary model (``analysis/costs.planning_prices``)."""
    prices = planning_prices(cfg)
    model = str(cfg["primary_model"])
    if model not in prices:
        raise TablesError(f"no planning price for {model!r} in the config")
    return prices[model]


def plan_manifest_path(run_root: str | Path, stage: str) -> Path:
    return Path(run_root) / "manifests" / f"dynorch_plan_{stage}.jsonl"


def plan_manifests(run_root: str | Path) -> list[Path]:
    """Every ``manifests/dynorch_plan_<stage>.jsonl`` under the run root, in stage order (canary, pilot, main,
    ext_e0, ...), so the tables cover all stages that were pre-registered."""
    order = {name: i for i, name in enumerate(("canary", "pilot", "main", "ext_e0", "ext_e1", "ext_e2", "ext_e3"))}
    found = list((Path(run_root) / "manifests").glob("dynorch_plan_*.jsonl"))
    return sorted(found, key=lambda p: (order.get(p.stem[len("dynorch_plan_"):], len(order)), p.name))


def build_tables(run_root: str | Path, cfg: Mapping[str, Any], *, stage: str | None = None,
                 out_dir: str | Path | None = None, grace_tokens: int | None = None) -> dict[str, Any]:
    """Build and write the four tables (contract section 7) for one stage, or for every pre-registered stage when
    ``stage`` is None (each cell keyed by its partition; a cell planned twice raises). Returns the frames plus
    ``reasoning_unknown_rows`` / ``n_planned`` / ``n_records``. ``out_dir`` defaults to ``<run_root>/tables``."""
    run_root = Path(run_root)
    manifests = [plan_manifest_path(run_root, stage)] if stage is not None else plan_manifests(run_root)
    if not manifests:
        raise TablesError(f"no plan manifest under {run_root / 'manifests'} (run a stage first)")
    plan: list[PlannedCell] = []
    seen: set[CellKey] = set()
    for manifest in manifests:
        for cell in read_plan(manifest):
            if cell.cell_key in seen:
                raise TablesError(f"{manifest}: cell {cell.cell_key} is planned in another manifest too")
            seen.add(cell.cell_key)
            plan.append(cell)
    records = read_run_records(run_root)
    records = [r for r in records if r["spec"]["partition"] in {c.partition for c in plan}]
    grace = int(cfg["dynorch"]["grace_tokens"]) if grace_tokens is None else int(grace_tokens)
    runs = runs_table(records)
    calls = calls_table(records)
    outcomes = policy_outcomes_table(plan, records)
    costs, unknown = policy_costs_table(plan, records, price_of(cfg), grace_tokens=grace)
    out = Path(out_dir) if out_dir is not None else run_root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    frames = {"dynorch_runs": runs, "dynorch_calls": calls, "policy_outcomes_dynorch": outcomes,
              "policy_costs_dynorch": costs}
    for name, frame in frames.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    return {**frames, "reasoning_unknown_rows": unknown, "n_planned": len(plan), "n_records": len(records)}


def _read_sealed(base_merged_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(base_merged_dir) / "tables" / f"{name}.parquet"
    if not path.is_file():
        raise TablesError(f"sealed table not found: {path}")
    return pd.read_parquet(path)


def _read_dynorch(dynorch_run_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(dynorch_run_dir) / "tables" / f"{name}.parquet"
    if not path.is_file():
        raise TablesError(f"dynorch table not found: {path}")
    return pd.read_parquet(path)


def _concat_sealed(base: pd.DataFrame, extra: pd.DataFrame, columns: Sequence[str], kind: str) -> pd.DataFrame:
    if list(base.columns) != list(columns):
        raise TablesError(f"sealed {kind} columns changed: {list(base.columns)}")
    if list(extra.columns) != list(columns):
        raise TablesError(f"dynorch {kind} columns differ from the sealed schema: {list(extra.columns)}")
    if (base["family"] == FAMILY).any():
        raise TablesError(f"sealed {kind} already contains family {FAMILY!r}")
    return pd.concat([base, extra], ignore_index=True)


def merged_outcomes(base_merged_dir: str | Path, dynorch_run_dir: str | Path) -> pd.DataFrame:
    """Sealed ``policy_outcomes`` + ``policy_outcomes_dynorch`` concatenated in memory (contract section 7)."""
    return _concat_sealed(_read_sealed(base_merged_dir, "policy_outcomes"),
                          _read_dynorch(dynorch_run_dir, "policy_outcomes_dynorch"), OUTCOME_COLUMNS,
                          "policy_outcomes")


def merged_costs(base_merged_dir: str | Path, dynorch_run_dir: str | Path) -> pd.DataFrame:
    """Sealed ``policy_costs`` + ``policy_costs_dynorch`` concatenated in memory (contract section 7)."""
    return _concat_sealed(_read_sealed(base_merged_dir, "policy_costs"),
                          _read_dynorch(dynorch_run_dir, "policy_costs_dynorch"), COST_COLUMNS, "policy_costs")


# ---------------------------------------------------------------------------------------------- synthetic builders

def synthetic_plan(cfg: Mapping[str, Any], items: Mapping[str, Mapping[str, str]], *, arms: Sequence[str],
                   rungs: Sequence[int], repeat: int = 1, partition: str = "main") -> list[PlannedCell]:
    """Planned cells for tests: ``items`` maps item_id -> {benchmark, shard_id, endpoint_tag}; run keys follow
    contract section 6. Arms come from the config's ``dynorch.arms`` / ``extension_arms`` (harness name)."""
    dyn = cfg["dynorch"]
    known = {**dyn["arms"], **dyn.get("extension_arms", {})}
    a_tokens = int(dyn["a_tokens"])
    cells = []
    for arm in arms:
        if arm not in known:
            raise TablesError(f"arm {arm!r} is not in the config")
        for item_id, meta in items.items():
            for rung in rungs:
                cells.append(PlannedCell(
                    run_key=run_key_of(str(cfg["study_id"]), arm, item_id, int(rung), repeat, 1, partition=partition),
                    arm=arm,
                    harness=str(known[arm]["harness"]), item_id=item_id, benchmark=str(meta["benchmark"]),
                    stratum=str(meta.get("stratum", meta["benchmark"])), shard_id=str(meta["shard_id"]),
                    endpoint_tag=str(meta["endpoint_tag"]), rung=int(rung), cap_tokens=int(rung) * a_tokens,
                    repeat=repeat, attempt=1, partition=partition))
    return cells


def write_plan(path: str | Path, cells: Iterable[PlannedCell]) -> Path:
    """Write a plan manifest in the shape ``read_plan`` accepts (flat RunSpec-like objects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in cells:
            fh.write(json.dumps({**c.__dict__, "__record__": "RunSpec"}) + "\n")
    return path


def synthetic_call(call_no: int, *, conversation_class: str, granted: int, completion: int, prompt: int,
                   spawn_calls: int = 0, forced_close_kind: str = "none", grace: int = 0,
                   cost_usd: str = "0.001", latency: float = 30.0) -> dict[str, Any]:
    """One CallRecord-shaped dict (contract section 5)."""
    return {"__record__": "CallRecord", "call_no": call_no, "conversation_key": f"conv{conversation_class}",
            "conversation_class": conversation_class, "requested_max_tokens": 32000, "granted_max_tokens": granted,
            "queue_wait_seconds": 0.0, "prompt_tokens": prompt, "completion_tokens": completion,
            "reasoning_tokens": None, "cached_prompt_tokens": 0, "reasoning_chars": 4 * completion,
            "visible_chars": 400, "forced_close_kind": forced_close_kind, "grace_tokens": grace,
            "finish_reason_upstream": "length" if completion >= granted else "stop",
            "finish_reason_returned": "tool_calls" if spawn_calls else "stop", "tool_calls": spawn_calls,
            "spawn_calls": spawn_calls, "tool_parse": "ok" if spawn_calls else "none", "reinjection": "n/a",
            "reasoning_effort_requested": "high", "cost_usd": cost_usd, "generation_ids": [f"gen-{call_no}"],
            "latency_seconds": latency, "transport_status": "OK", "returned_status": 200, "stream": True,
            "wire": "chat", "cache_status": None, "duplicate_served": False, "ignored_body_keys": []}


def synthetic_run_record(cell: PlannedCell, *, correct: bool | None, status: str = "done", spawns: int = 0,
                         valid: bool = True, realized_share: float = 0.9, grace: int = 0,
                         forced_close_kind: str = "none", attempt: int | None = None, cost_usd: str | None = None,
                         run_key: str | None = None, base_prompt: int = 5500, wall_seconds: float = 120.0,
                         guard_triggered: bool = False, tool_parse_failures: int = 0, exhausted: bool | None = None,
                         harness_version: str = "fake-0.0") -> dict[str, Any]:
    """A RunRecord-shaped dict for one cell (contract section 5) with an internally consistent call list:
    ``completion_total = realized_share * cap`` split over ``1 + spawns`` calls (root + solver conversations, plus a
    closing root call when there were spawns), ``prompt_total`` summed over calls."""
    cap = cell.cap_tokens
    attempt = cell.attempt if attempt is None else attempt
    key = cell.run_key if run_key is None else run_key
    total = int(round(realized_share * cap))
    n_calls = 1 if spawns == 0 else 2 * spawns + 1
    calls: list[dict[str, Any]] = []
    spent = 0
    prompt_total = 0
    for i in range(n_calls):
        share = total - spent if i == n_calls - 1 else total // n_calls
        cls = "root" if (i == 0 or i == n_calls - 1 or i % 2 == 0) else "solver"
        prompt = base_prompt + spent
        calls.append(synthetic_call(i + 1, conversation_class=cls, granted=cap - spent, completion=share,
                                    prompt=prompt, spawn_calls=1 if (cls == "root" and i < n_calls - 1 and spawns) else 0,
                                    forced_close_kind=forced_close_kind if i == n_calls - 1 else "none",
                                    grace=grace if i == n_calls - 1 else 0))
        spent += share
        prompt_total += prompt
    by_class = {"root": sum(1 for c in calls if c["conversation_class"] == "root"),
                "solver": sum(1 for c in calls if c["conversation_class"] == "solver"), "other": 0}
    final = None if not valid else "<final>42</final>"
    if status in INFRA_STATUSES:
        final, valid, correct = None, False, None
    cost = cost_usd if cost_usd is not None else str(Decimal(prompt_total) * Decimal("0.24") / Decimal(10 ** 6)
                                                    + Decimal(total + grace) * Decimal("2.5") / Decimal(10 ** 6))
    spec = {"__record__": "RunSpec", "run_key": key, "arm": cell.arm, "harness": cell.harness,
            "item_id": cell.item_id, "benchmark": cell.benchmark, "stratum": cell.stratum, "shard_id": cell.shard_id,
            "endpoint_tag": cell.endpoint_tag, "rung": cell.rung, "cap_tokens": cap, "repeat": cell.repeat,
            "attempt": attempt, "generation": 1, "partition": cell.partition, "allowed_tools": ["Agent"],
            "spawn_tools": ["Agent"], "root_call_ceiling": None, "sampling_hash": "s" * 8, "template_sha256": "t" * 8,
            "prompt_version": "dynorch-v1", "proxy_version": "0.0-test", "harness_version": harness_version}
    return {"__record__": "RunRecord", "spec": spec, "status": status, "calls": len(calls),
            "conversations": 1 + spawns, "spawns": spawns, "calls_by_class": by_class, "completion_total": total,
            "reasoning_total": None, "prompt_total": prompt_total, "grace_tokens": grace,
            "overdraft_tokens": max(0, total + grace - cap), "forced_close_kind": forced_close_kind,
            "exhausted": (forced_close_kind != "none") if exhausted is None else exhausted, "refused_calls": 0,
            "guard_triggered": guard_triggered, "tool_parse_failures": tool_parse_failures,
            "reinjection_hits": max(0, len(calls) - 1), "reinjection_misses": 0, "duplicates_served": 0,
            "final_message": final, "final_source": "proxy_root" if final else "none", "cli_result": final,
            "cli_result_mismatch": False if final else None,
            "parse_status": (PARSE_VALID if valid else "missing_final") if status not in INFRA_STATUSES else None,
            "final_answer": "42" if valid else None,
            "grade_status": ("graded" if correct is not None else "evaluator_error") if valid else None,
            "correct": correct if valid else None, "grade_detail": None, "wall_seconds": wall_seconds,
            "harness_exit_code": 0 if status == "done" else 1, "harness_reported": {"subtype": "success"},
            "cost_usd": cost, "started_at": "2026-09-23T00:00:00Z", "finished_at": "2026-09-23T00:02:00Z",
            "error": None if status == "done" else status, "rerun_reason": None, "calls_detail": calls}


def item_slug(item_id: str) -> str:
    return item_id.replace("/", "__")


def write_run_record(run_root: str | Path, record: Mapping[str, Any], *, superseded: bool = False) -> Path:
    """Write a RunRecord exactly where ``RunStore`` puts it (contract §5 / §10.22): the current attempt at
    ``runs/<partition>/<arm>/<rung>/<repeat>/<item_slug>.json`` and an earlier one at
    ``runs_superseded/<partition>/<arm>/<rung>/<repeat>/<item_slug>.a<attempt>.json``."""
    spec = record["spec"]
    partition = str(spec.get("partition") or "main")
    if superseded:
        path = (Path(run_root) / "runs_superseded" / partition / spec["arm"] / str(spec["rung"])
                / str(spec["repeat"]) / f"{item_slug(spec['item_id'])}.a{int(spec['attempt'])}.json")
    else:
        path = (Path(run_root) / "runs" / partition / spec["arm"] / str(spec["rung"]) / str(spec["repeat"])
                / f"{item_slug(spec['item_id'])}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=0))
    return path


__all__ = [
    "A_TOKENS", "COST_COLUMNS", "COST_DTYPES", "DELIVERED_STATES", "OUTCOME_COLUMNS", "OUTCOME_DTYPES", "PlannedCell",
    "TablesError", "build_tables", "calls_table", "cell_name", "current_records", "enforce_schema",
    "merged_costs", "merged_outcomes", "outcome_of", "plan_manifest_path", "policy_costs_table",
    "policy_id", "policy_outcomes_table", "price_of", "read_plan", "read_run_records", "run_key_of", "runs_table",
    "synthetic_call", "synthetic_plan", "synthetic_run_record", "validate_record", "write_plan", "write_run_record",
]
