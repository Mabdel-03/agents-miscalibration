"""WP-D: the frozen auto-advance gates of dynorch_v1 (plan §9; contract §8).

Every criterion is computed from written artifacts: WP-C's ``tables/dynorch_runs.parquet`` /
``tables/dynorch_calls.parquet`` (columns = the contract §5 ``RunRecord`` / ``CallRecord`` fields plus the spec
columns), WP-D's ``state/reconciliation_<stage>.json`` and the lock files. Results are ``contracts.GateResult``
records written by ``agent_allocation.gates.write_gate`` under ``<run_root>/gates/``. Decisions: ``advance`` (exit 0),
``stop_notify`` (exit 3: an owner must look), ``stop`` (exit 1: an invariant broke). Nothing here raises ceilings,
switches providers or edits prompts; failing criteria are reported, never amended.

Gate 0 = doctor + the offline unit suite (+ WP-E's smoke report when present). Gate 1 = the canary criteria.
Gate 2 = the pilot criteria per harness and ``reports/PILOT_REPORT.md`` (accuracies pooled per benchmark only).
Gate 4 = seal: a row per planned cell, completeness per (arm, rung), the /key delta, the grader re-run, the
sealed ladder point estimates, the table seal.
"""
from __future__ import annotations

import json
from agent_allocation.storage._atomic import replace_atomic
import math
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.contracts import GradeStatus, ParseStatus, GateCriterion, GateResult
from agent_allocation.gates import write_gate

ADVANCE, STOP_NOTIFY, STOP = "advance", "stop_notify", "stop"
EXIT_ADVANCE, EXIT_STOP, EXIT_STOP_NOTIFY = 0, 1, 3
EXIT_CODES = {ADVANCE: EXIT_ADVANCE, STOP_NOTIFY: EXIT_STOP_NOTIFY, STOP: EXIT_STOP}

# ---- Gate 1 thresholds (plan §9) ----------------------------------------------------------------------------------
G1_KEY_DELTA_TOLERANCE_USD = Decimal("0.02")
G1_GRACE_TRUNCATED_MAX_SHARE = 0.02
FORCED_CLOSE_KINDS = ("mid_call", "visible_continuation", "terminal")
# ---- Gate 2 thresholds (plan §9) ----------------------------------------------------------------------------------
#: The runner writes the codec values (``grade.finalize_run``): "valid" / "evaluator_error", never the enum names.
PARSE_VALID = ParseStatus.VALID.value
GRADE_EVALUATOR_ERROR = GradeStatus.EVALUATOR_ERROR.value
G2_VALID_FINAL_ADVANCE = 0.90
G2_VALID_FINAL_NOTIFY = 0.80
G2_TOOL_PARSE_FAILURE_MAX = 0.05
G2_SPAWN_SHARE_K16_MIN = 0.10
G2_GUARD_TRIGGERED_MAX = 0.05
G2_EVALUATOR_ERROR_MAX = 0.01
G2_REINJECTION_HIT_MIN = 0.95
G2_TIMEOUT_MAX = 0.02
G2_HARNESS_ARMS = ("H1", "H3")
G2_SPAWN_RUNG = 16
# ---- Gate 4 thresholds (plan §9) ----------------------------------------------------------------------------------
G4_COMPLETENESS_MIN = 0.95
G4_KEY_DELTA_USD = Decimal("0.50")
G4_KEY_DELTA_SHARE = Decimal("0.02")
SEALED_LADDER = {"deep/16": 0.7917, "sample/16": 0.4167}
LADDER_TOLERANCE = 0.0001
PRIMARY_CELLS = ("H1/16", "H3/16")

RUNS_COLUMNS: tuple[str, ...] = (
    "run_key", "arm", "harness", "item_id", "benchmark", "rung", "cap_tokens", "repeat", "attempt", "partition",
    "status", "calls", "spawns", "completion_total", "grace_tokens", "forced_close_kind", "exhausted",
    "guard_triggered", "tool_parse_failures", "reinjection_hits", "reinjection_misses", "final_message",
    "parse_status", "grade_status", "correct", "wall_seconds", "cost_usd", "shard_id", "endpoint_tag",
)
CALLS_COLUMNS: tuple[str, ...] = (
    "run_key", "call_no", "conversation_class", "granted_max_tokens", "completion_tokens", "prompt_tokens",
    "grace_tokens", "forced_close_kind", "tool_calls", "spawn_calls", "tool_parse", "reinjection", "wire",
    "duplicate_served",
)


class GateInputError(RuntimeError):
    """A table or document without the shape the gate needs (never a silent default)."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_columns(table: Any, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise GateInputError(f"{name} lacks columns {missing}")


@dataclass(frozen=True)
class Crit:
    """A criterion plus the decision its failure forces (``stop`` for invariants, ``stop_notify`` otherwise)."""
    criterion: GateCriterion
    severity: str = STOP_NOTIFY


def crit(name: str, observed: Any, threshold: str, passed: bool, evidence: str, *, severity: str = STOP_NOTIFY) -> Crit:
    return Crit(GateCriterion(name=name, observed=json.dumps(observed, default=str, sort_keys=True), threshold=threshold,
                              passed=bool(passed), evidence=evidence), severity)


def finish(gate: str, stage: str, crits: Sequence[Crit]) -> GateResult:
    """``advance`` when every criterion passes; ``stop`` when any failing criterion is an invariant; else
    ``stop_notify``."""
    failing = [c for c in crits if not c.criterion.passed]
    if not failing:
        decision = ADVANCE
    elif any(c.severity == STOP for c in failing):
        decision = STOP
    else:
        decision = STOP_NOTIFY
    return GateResult(gate=gate, stage=stage, passed=not failing, decision=decision,
                      criteria=tuple(c.criterion for c in crits), evaluated_at=utcnow())


def exit_code(result: GateResult) -> int:
    return EXIT_CODES.get(result.decision, EXIT_STOP)


def _share(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _isnan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


# ------------------------------------------------------------------------------------------------------------------
# Gate 0
# ------------------------------------------------------------------------------------------------------------------

def evaluate_gate0(doctor_checks: Sequence[Mapping[str, Any]], *, pytest_returncode: int | None, pytest_summary: str,
                   smoke_report: Mapping[str, Any] | None) -> GateResult:
    """Offline, $0: doctor preconditions, ``tests/unit/dynorch`` green, WP-E's real-CLI smoke report
    (``gates/gate0_smoke.json`` with ``passed`` and ``failures``) present and passing."""
    blocking = [c["check"] for c in doctor_checks if c.get("blocking", True) and not c.get("ok")]
    crits = [
        crit("doctor_preconditions", blocking, "no blocking doctor failure", not blocking, "dynorch-doctor", severity=STOP),
        crit("unit_suite_green", {"returncode": pytest_returncode, "summary": pytest_summary[-300:]},
             "pytest tests/unit/dynorch exit 0", pytest_returncode == 0, "tests/unit/dynorch", severity=STOP),
    ]
    if smoke_report is None:
        crits.append(crit("real_cli_smoke", None, "gates/gate0_smoke.json present with passed == true", False,
                          "gates/gate0_smoke.json"))
    else:
        crits.append(crit("real_cli_smoke", {"passed": smoke_report.get("passed"), "failures": smoke_report.get("failures")},
                          "every OpenClaude and Codex smoke check passed", smoke_report.get("passed") is True,
                          "gates/gate0_smoke.json"))
    return finish("gate0", "offline", crits)


# ------------------------------------------------------------------------------------------------------------------
# Shared table views
# ------------------------------------------------------------------------------------------------------------------

def grant_violations(runs: Any, calls: Any) -> dict[str, list[dict[str, Any]]]:
    """Per call: ``completion_tokens <= granted_max_tokens`` and, for non-terminal, non-duplicate calls of core arms,
    ``granted == cap − completion counted before it`` (B2). Terminal calls are granted the grace G, duplicates are
    served from the cache, so both are excluded from the equality."""
    require_columns(runs, ("run_key", "cap_tokens"), "dynorch_runs")
    require_columns(calls, ("run_key", "call_no", "granted_max_tokens", "completion_tokens", "forced_close_kind",
                            "duplicate_served"), "dynorch_calls")
    cap_of = dict(zip(runs["run_key"], runs["cap_tokens"]))
    over: list[dict[str, Any]] = []
    unequal: list[dict[str, Any]] = []
    ordered = calls.sort_values(["run_key", "call_no"])
    spent: dict[str, int] = {}
    for rec in ordered.to_dict("records"):
        key = rec["run_key"]
        if key not in cap_of:
            raise GateInputError(f"call row for unknown run {key!r}")
        granted = rec["granted_max_tokens"]
        completion = rec["completion_tokens"]
        if _isnan(completion) or _isnan(granted):
            continue
        granted, completion = int(granted), int(completion)
        if completion > granted:
            over.append({"run_key": key, "call_no": int(rec["call_no"]), "completion": completion, "granted": granted})
        if bool(rec["duplicate_served"]):
            continue
        remaining = int(cap_of[key]) - spent.get(key, 0)
        if rec["forced_close_kind"] != "terminal" and granted != remaining:
            unequal.append({"run_key": key, "call_no": int(rec["call_no"]), "granted": granted, "remaining": remaining})
        if rec["forced_close_kind"] != "terminal":
            spent[key] = spent.get(key, 0) + completion
    return {"over_grant": over, "grant_not_remaining": unequal}


def harness_prompt_overheads(runs: Any, calls: Any) -> dict[str, dict[str, Any]]:
    """``S_h = prompt_tokens(first root call of harness h) − prompt_tokens(H0 call, same item, same rung)`` per run,
    summarised per harness arm (plan §3)."""
    require_columns(runs, ("run_key", "arm", "item_id", "rung"), "dynorch_runs")
    require_columns(calls, ("run_key", "call_no", "conversation_class", "prompt_tokens"), "dynorch_calls")
    first_root: dict[str, int] = {}
    for rec in calls.sort_values(["run_key", "call_no"]).to_dict("records"):
        if rec["conversation_class"] == "root" and rec["run_key"] not in first_root and not _isnan(rec["prompt_tokens"]):
            first_root[rec["run_key"]] = int(rec["prompt_tokens"])
    h0: dict[tuple[str, int], int] = {}
    for rec in runs.to_dict("records"):
        if rec["arm"] == "H0" and rec["run_key"] in first_root:
            h0[(rec["item_id"], int(rec["rung"]))] = first_root[rec["run_key"]]
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for rec in runs.to_dict("records"):
        if rec["arm"] == "H0" or rec["run_key"] not in first_root:
            continue
        anchor = h0.get((rec["item_id"], int(rec["rung"])))
        if anchor is None:
            continue
        per_arm.setdefault(str(rec["arm"]), []).append({"run_key": rec["run_key"], "item_id": rec["item_id"],
                                                        "rung": int(rec["rung"]), "s_h": first_root[rec["run_key"]] - anchor})
    out: dict[str, dict[str, Any]] = {}
    for arm, rows in sorted(per_arm.items()):
        values = [r["s_h"] for r in rows]
        out[arm] = {"n": len(values), "mean": statistics.fmean(values),
                    "sd": statistics.pstdev(values) if len(values) > 1 else 0.0, "per_run": rows}
    return out


def reconciliation_crits(reconciliation: Mapping[str, Any] | None, *, tolerance_usd: Decimal, evidence: str) -> list[Crit]:
    """The reconciliation criteria shared by Gates 1, 2 and 4: key delta vs ledger delta, the generation sample,
    no upstream 402."""
    if reconciliation is None:
        return [crit("reconciliation_present", None, "state/reconciliation_<stage>.json written", False, evidence)]
    diff = Decimal(str(reconciliation["key_minus_ledger_usd"]))
    rows = reconciliation["generation_sample"]["rows"]
    bad = [r for r in rows if not r["fetched"] or r["tokens_match"] is False or r["cost_within_tolerance"] is False]
    provider_bad = [r for r in rows if r["provider_ok"] is False]
    return [
        crit("key_delta_within_tolerance", {"key_delta": reconciliation["key_delta_usd"],
                                            "ledger_delta": reconciliation["ledger_delta_usd"], "diff": str(diff)},
             f"|key − ledger| <= {tolerance_usd}", abs(diff) <= tolerance_usd, evidence),
        crit("generation_sample_matches", {"sampled": len(rows), "bad": [{k: r[k] for k in ("run_key", "call_no")} for r in bad]},
             "every sampled call: /generation fetched, tokens counted once, cost within $0.01 or 5 %",
             bool(rows) and not bad, evidence),
        crit("observed_provider_pinned", [{k: r[k] for k in ("run_key", "call_no", "provider_names")} for r in provider_bad],
             "every sampled generation served by the pinned provider", not provider_bad, evidence, severity=STOP),
        crit("no_upstream_402", bool(reconciliation.get("upstream_402")), "no PAYMENT_REQUIRED from upstream",
             not reconciliation.get("upstream_402"), evidence),
        crit("open_reservations_zero", reconciliation["ledger_after"]["open_reservations_usd"], "0",
             Decimal(str(reconciliation["ledger_after"]["open_reservations_usd"])) == 0, evidence),
    ]


# ------------------------------------------------------------------------------------------------------------------
# Gate 1 — canary
# ------------------------------------------------------------------------------------------------------------------

def evaluate_gate1(cfg: Mapping[str, Any], runs: Any, calls: Any, reconciliation: Mapping[str, Any] | None, *,
                   lock_decision: Mapping[str, Any] | None, key_hits: Sequence[str], home_new_files: Sequence[str] | None,
                   home_detail: Mapping[str, Any] | None = None,
                   evidence: str = "tables/dynorch_runs.parquet + tables/dynorch_calls.parquet + state/reconciliation_canary.json"
                   ) -> tuple[GateResult, dict[str, dict[str, Any]]]:
    """Plan §9 Gate 1 over the canary rows. Returns the result and the ``S_h`` document for
    ``locks/harness_prompts.json``."""
    require_columns(runs, RUNS_COLUMNS, "dynorch_runs")
    require_columns(calls, CALLS_COLUMNS, "dynorch_calls")
    grace_cap = int(cfg["dynorch"]["grace_tokens"])
    probe_rung = int(cfg["dynorch"]["canary"]["ceiling_probe_rung"])
    crits: list[Crit] = []
    action = (lock_decision or {}).get("action")
    crits.append(crit("endpoint_rule_locked", {"action": action, "shard_to_tag": (lock_decision or {}).get("shard_to_tag")},
                      "locks/openrouter_endpoints.json frozen with keep or switch", action in ("keep", "switch"),
                      "locks/openrouter_endpoints.json"))
    crits.append(crit("canary_runs_present", {"runs": int(len(runs)), "calls": int(len(calls))}, ">= 1 run with >= 1 call",
                      len(runs) > 0 and len(calls) > 0, evidence, severity=STOP))
    violations = grant_violations(runs, calls) if len(runs) else {"over_grant": [], "grant_not_remaining": []}
    crits.append(crit("completion_within_grant", violations["over_grant"], "every call completion_tokens <= granted",
                      not violations["over_grant"], evidence, severity=STOP))
    crits.append(crit("grant_equals_remaining", violations["grant_not_remaining"], "every non-terminal call granted == C − spent",
                      not violations["grant_not_remaining"], evidence, severity=STOP))
    no_usage = [{"run_key": r["run_key"], "call_no": int(r["call_no"])} for r in calls.to_dict("records")
                if _isnan(r["prompt_tokens"]) or _isnan(r["completion_tokens"])]
    crits.append(crit("usage_present", no_usage, "prompt and completion tokens on every call", not no_usage, evidence))
    crits.extend(reconciliation_crits(reconciliation, tolerance_usd=G1_KEY_DELTA_TOLERANCE_USD, evidence="state/reconciliation_canary.json"))

    closed = runs[runs["forced_close_kind"].isin(FORCED_CLOSE_KINDS)]
    truncated = closed[closed["grace_tokens"].fillna(0).astype(int) >= grace_cap]
    share = _share(len(truncated), len(closed))
    crits.append(crit("grace_truncated_share", {"forced_closes": int(len(closed)), "truncated": int(len(truncated)), "share": round(share, 4)},
                      f"<= {G1_GRACE_TRUNCATED_MAX_SHARE}", share <= G1_GRACE_TRUNCATED_MAX_SHARE, evidence))
    for arm in G2_HARNESS_ARMS:
        rows = closed[(closed["arm"] == arm) & closed["final_message"].map(_nonempty)]
        crits.append(crit(f"{arm}_forced_close_nonempty", int(len(rows)), ">= 1 forced close returning non-empty text",
                          len(rows) >= 1, evidence))
    h3_keys = set(runs[runs["arm"] == "H3"]["run_key"])
    h3_calls = calls[calls["run_key"].isin(h3_keys)]
    loop_runs = []
    for key, group in h3_calls.groupby("run_key"):
        if len(group) >= 2 and (group["tool_calls"].fillna(0).astype(int) >= 1).any() and (group["wire"] == "responses").all():
            loop_runs.append(key)
    crits.append(crit("codex_responses_tool_loop", loop_runs, ">= 1 H3 run: Responses wire, a tool call, >= 2 calls",
                      bool(loop_runs), evidence))
    probe = runs[(runs["rung"] == probe_rung) & runs["arm"].isin(G2_HARNESS_ARMS)]
    first_calls = calls[calls["call_no"] == calls.groupby("run_key")["call_no"].transform("min")]
    first_of = {r["run_key"]: r for r in first_calls.to_dict("records")}
    probe_rows = [{"arm": r["arm"], "item_id": r["item_id"], "spawns": int(r["spawns"]),
                   "forced_close_kind": r["forced_close_kind"],
                   "first_call_completion": None if r["run_key"] not in first_of else first_of[r["run_key"]]["completion_tokens"],
                   "first_call_forced_close": None if r["run_key"] not in first_of else first_of[r["run_key"]]["forced_close_kind"]}
                  for r in probe.to_dict("records")]
    expected = int(cfg["dynorch"]["canary"]["ceiling_probe_items"]) * len(G2_HARNESS_ARMS)
    crits.append(crit("ceiling_probe_recorded", probe_rows, f"{expected} rung-{probe_rung} runs (H1, H3) with a first call",
                      len(probe_rows) >= expected and all(r["first_call_completion"] is not None for r in probe_rows), evidence))
    spawned = sum(1 for r in probe_rows if r["spawns"] > 0)
    crits.append(crit("ceiling_probe_spawns", {"spawning_runs": spawned, "of": len(probe_rows)},
                      ">= 1 spawn across the k=16 probe (0 of 4 -> stop and ask the owner)", spawned > 0, evidence))
    overheads = harness_prompt_overheads(runs, calls)
    crits.append(crit("harness_prompts_measured", {arm: {k: v for k, v in d.items() if k != "per_run"} for arm, d in overheads.items()},
                      "S_H1 and S_H3 computable (H0 anchor at the same item and rung)",
                      all(arm in overheads for arm in G2_HARNESS_ARMS), "locks/harness_prompts.json"))
    crits.append(crit("no_key_string_in_run_tree", list(key_hits), "no key-shaped string under the run root", not key_hits,
                      "run root scan", severity=STOP))
    # review minor 21: "not scanned" is a NOTIFY with an explicit observed value, not a failed criterion -- an
    # lstat walk of ~995k NFS inodes routinely exceeded its budget and would have ended every canary in
    # stop_notify for a non-finding. The cheap path is the quota inode delta the stage marker records.
    scanned = home_new_files is not None
    crits.append(crit("nothing_written_under_home",
                      {"detail": dict(home_detail or {}),
                       "new": None if not scanned else list(home_new_files)[:20]},
                      "no file created under $HOME during the stage (quota inode delta, else a $HOME walk)",
                      not scanned or not home_new_files, "home scan"))
    if not scanned:
        crits.append(crit("home_scan_completed", dict(home_detail or {}),
                          "the $HOME check actually ran (quota delta or a completed walk)", False, "home scan"))
    return finish("gate1", "canary", crits), overheads


# ------------------------------------------------------------------------------------------------------------------
# Gate 2 — pilot
# ------------------------------------------------------------------------------------------------------------------

def harness_metrics(cfg: Mapping[str, Any], runs: Any, calls: Any, arm: str) -> dict[str, Any]:
    """The per-harness Gate 2 shares over one arm's pilot rows (the numbers, not the verdicts)."""
    grace_cap = int(cfg["dynorch"]["grace_tokens"])
    mine = runs[runs["arm"] == arm]
    keys = set(mine["run_key"])
    my_calls = calls[calls["run_key"].isin(keys)]
    n = int(len(mine))
    valid = int((mine["parse_status"] == PARSE_VALID).sum())
    parsed_calls = my_calls[my_calls["tool_parse"].isin(["ok", "failed"])]
    parse_failed = int((parsed_calls["tool_parse"] == "failed").sum())
    cap_over = mine[mine["completion_total"].fillna(0).astype(int) > mine["cap_tokens"].astype(int)]
    grace_over = mine[mine["grace_tokens"].fillna(0).astype(int) > grace_cap]
    at_k = mine[mine["rung"] == G2_SPAWN_RUNG]
    spawning = int((at_k["spawns"].fillna(0).astype(int) >= 1).sum())
    guard = int(mine["guard_triggered"].fillna(False).astype(bool).sum())
    graded = mine[mine["parse_status"] == PARSE_VALID]
    evaluator_errors = int((graded["grade_status"] == GRADE_EVALUATOR_ERROR).sum())
    hits = int(mine["reinjection_hits"].fillna(0).sum())
    misses = int(mine["reinjection_misses"].fillna(0).sum())
    timeouts = int((mine["status"] == "timeout").sum())
    return {
        "runs": n, "valid_final_share": _share(valid, n), "valid": valid,
        "tool_call_parse_failure_share": _share(parse_failed, len(parsed_calls)), "tool_parse_calls": int(len(parsed_calls)),
        "cap_overrun_runs": [r for r in cap_over["run_key"]], "grace_overrun_runs": [r for r in grace_over["run_key"]],
        "spawn_share_k16": _share(spawning, len(at_k)), "runs_k16": int(len(at_k)), "spawning_k16": spawning,
        "guard_triggered_share": _share(guard, n),
        "evaluator_error_share": _share(evaluator_errors, len(graded)), "graded": int(len(graded)),
        "reinjection_hit_rate": None if hits + misses == 0 else hits / (hits + misses), "reinjection_hits": hits,
        "reinjection_misses": misses,
        "timeout_share": _share(timeouts, n), "timeouts": timeouts,
    }


def evaluate_gate2(cfg: Mapping[str, Any], runs: Any, calls: Any, reconciliation: Mapping[str, Any] | None, *,
                   forecast: Mapping[str, Any] | None,
                   evidence: str = "tables/dynorch_runs.parquet + tables/dynorch_calls.parquet + reports/PILOT_REPORT.md") -> GateResult:
    """Plan §9 Gate 2 per harness (H1, H3) over the pilot rows, plus reconciliation and the re-fitted core forecast
    (``forecast`` = ``{"core_forecast_usd", "cuts_applied", "spent_usd"}`` from WP-C's ``forecast.py``, already
    carrying the 25 % contingency and at most cuts C1–C3)."""
    require_columns(runs, RUNS_COLUMNS, "dynorch_runs")
    require_columns(calls, CALLS_COLUMNS, "dynorch_calls")
    crits: list[Crit] = []
    for arm in G2_HARNESS_ARMS:
        m = harness_metrics(cfg, runs, calls, arm)
        share = m["valid_final_share"]
        severity = STOP if share < G2_VALID_FINAL_NOTIFY else STOP_NOTIFY
        crits.append(crit(f"{arm}_valid_final_share", {"n": m["runs"], "valid": m["valid"], "share": round(share, 4)},
                          f">= {G2_VALID_FINAL_ADVANCE} (>= {G2_VALID_FINAL_NOTIFY}: stop_notify; below: stop)",
                          m["runs"] > 0 and share >= G2_VALID_FINAL_ADVANCE, evidence, severity=severity))
        crits.append(crit(f"{arm}_tool_call_parse_failure_share", {"calls": m["tool_parse_calls"], "share": round(m["tool_call_parse_failure_share"], 4)},
                          f"<= {G2_TOOL_PARSE_FAILURE_MAX}", m["tool_call_parse_failure_share"] <= G2_TOOL_PARSE_FAILURE_MAX, evidence))
        crits.append(crit(f"{arm}_cap_overrun_runs", m["cap_overrun_runs"], "0", not m["cap_overrun_runs"], evidence, severity=STOP))
        crits.append(crit(f"{arm}_grace_overrun_runs", m["grace_overrun_runs"], "0", not m["grace_overrun_runs"], evidence, severity=STOP))
        crits.append(crit(f"{arm}_spawn_share_k16", {"runs_k16": m["runs_k16"], "spawning": m["spawning_k16"], "share": round(m["spawn_share_k16"], 4)},
                          f">= {G2_SPAWN_SHARE_K16_MIN}", m["runs_k16"] > 0 and m["spawn_share_k16"] >= G2_SPAWN_SHARE_K16_MIN, evidence))
        crits.append(crit(f"{arm}_guard_triggered_share", round(m["guard_triggered_share"], 4), f"<= {G2_GUARD_TRIGGERED_MAX}",
                          m["guard_triggered_share"] <= G2_GUARD_TRIGGERED_MAX, evidence))
        crits.append(crit(f"{arm}_evaluator_error_share", {"graded": m["graded"], "share": round(m["evaluator_error_share"], 4)},
                          f"<= {G2_EVALUATOR_ERROR_MAX}", m["evaluator_error_share"] <= G2_EVALUATOR_ERROR_MAX, evidence))
        rate = m["reinjection_hit_rate"]
        crits.append(crit(f"{arm}_reinjection_hit_rate", {"hits": m["reinjection_hits"], "misses": m["reinjection_misses"],
                                                           "rate": None if rate is None else round(rate, 4)},
                          f">= {G2_REINJECTION_HIT_MIN} (vacuous with no assistant tool turns)",
                          rate is None or rate >= G2_REINJECTION_HIT_MIN, evidence))
        crits.append(crit(f"{arm}_timeout_share", {"timeouts": m["timeouts"], "share": round(m["timeout_share"], 4)},
                          f"<= {G2_TIMEOUT_MAX}", m["timeout_share"] <= G2_TIMEOUT_MAX, evidence))
    crits.extend(reconciliation_crits(reconciliation, tolerance_usd=G1_KEY_DELTA_TOLERANCE_USD, evidence="state/reconciliation_pilot.json"))
    soft = Decimal(str(cfg["budget"]["soft_admission_ceiling_usd"]))
    if forecast is None:
        crits.append(crit("core_forecast_within_soft_ceiling", None, "re-fitted forecast available (forecast.py)", False, "forecast.py"))
    else:
        core = Decimal(str(forecast["core_forecast_usd"]))
        spent = Decimal(str(forecast["spent_usd"]))
        cuts = list(forecast.get("cuts_applied", []))
        allowed = all(str(c).startswith(("C1", "C2", "C3")) for c in cuts)
        crits.append(crit("core_forecast_within_soft_ceiling",
                          {"core_forecast_usd": str(core), "spent_usd": str(spent), "soft_ceiling_usd": str(soft), "cuts_applied": cuts},
                          "forecast (+25 %) <= soft ceiling − spent after at most C1–C3",
                          allowed and core <= soft - spent, "forecast.py"))
    return finish("gate2", "pilot", crits)


def pooled_accuracies(runs: Any) -> list[dict[str, Any]]:
    """Accuracy per (arm, rung, benchmark) pooled over items (never per item), plus a macro row per (arm, rung)."""
    rows: list[dict[str, Any]] = []
    for (arm, rung), group in runs.groupby(["arm", "rung"]):
        per_bench: dict[str, float] = {}
        for bench, sub in group.groupby("benchmark"):
            correct = sub["correct"].map(lambda v: 1.0 if v is True or v == 1 else 0.0)
            per_bench[str(bench)] = float(correct.mean())
            rows.append({"arm": arm, "rung": int(rung), "benchmark": str(bench), "n": int(len(sub)), "accuracy": per_bench[str(bench)]})
        rows.append({"arm": arm, "rung": int(rung), "benchmark": "macro", "n": int(len(group)),
                     "accuracy": statistics.fmean(per_bench.values()) if per_bench else float("nan")})
    return rows


def render_pilot_report(cfg: Mapping[str, Any], runs: Any, calls: Any, result: GateResult,
                        reconciliation: Mapping[str, Any] | None, forecast: Mapping[str, Any] | None) -> str:
    """``reports/PILOT_REPORT.md``: criteria, per-harness diagnostics, accuracies pooled per benchmark only,
    reconciliation and forecast, then the owner go/no-go instruction."""
    lines = [f"# PILOT_REPORT — {cfg['study_id']}", "",
             f"Evaluated {result.evaluated_at}; decision **{result.decision}** (passed={result.passed}).", "",
             "## Gate 2 criteria", "", "| criterion | observed | threshold | passed |", "|---|---|---|---|"]
    for c in result.criteria:
        lines.append(f"| {c.name} | `{c.observed}` | {c.threshold} | {'yes' if c.passed else 'NO'} |")
    lines += ["", "## Harness diagnostics", "", "| arm | runs | valid final | parse-failure share | spawn share k=16 | guard | reinjection | timeouts |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in sorted(set(runs["arm"])):
        m = harness_metrics(cfg, runs, calls, arm)
        rate = m["reinjection_hit_rate"]
        lines.append(f"| {arm} | {m['runs']} | {m['valid_final_share']:.3f} | {m['tool_call_parse_failure_share']:.3f} | "
                     f"{m['spawn_share_k16']:.3f} ({m['spawning_k16']}/{m['runs_k16']}) | {m['guard_triggered_share']:.3f} | "
                     f"{'n/a' if rate is None else f'{rate:.3f}'} | {m['timeouts']} |")
    kinds = runs.groupby(["arm", "forced_close_kind"]).size()
    lines += ["", "Forced-close kinds per arm: " + ", ".join(f"{a}/{k}={int(v)}" for (a, k), v in kinds.items()), ""]
    lines += ["## Accuracy pooled per benchmark (no per-item outcomes)", "", "| arm | rung | benchmark | n | accuracy |", "|---|---|---|---|---|"]
    for row in pooled_accuracies(runs):
        lines.append(f"| {row['arm']} | {row['rung']} | {row['benchmark']} | {row['n']} | {row['accuracy']:.3f} |")
    lines += ["", "## Reconciliation", ""]
    if reconciliation is None:
        lines.append("_no reconciliation document_")
    else:
        lines += [f"- key delta ${reconciliation['key_delta_usd']} vs ledger delta ${reconciliation['ledger_delta_usd']} "
                  f"(diff ${reconciliation['key_minus_ledger_usd']}); decision {reconciliation['decision']}",
                  f"- generation sample: {reconciliation['generation_sample']['sampled']} of {reconciliation['generation_sample']['population']} calls",
                  f"- upstream 402: {reconciliation['upstream_402']}"]
    lines += ["", "## Re-fitted core forecast", ""]
    if forecast is None:
        lines.append("_forecast unavailable (WP-C forecast.py)_")
    else:
        lines.append(f"- core forecast (+25 %): ${forecast['core_forecast_usd']}; spent so far ${forecast['spent_usd']}; "
                     f"soft ceiling ${cfg['budget']['soft_admission_ceiling_usd']}; cuts applied: {forecast.get('cuts_applied', [])}")
    lines += ["", "## Owner go/no-go", "",
              "The pilot never auto-advances. Record the decision with `live.record_decision` (see RUNBOOK: "
              "`allocation dynorch-gates --gate 2 ... --owner-go \"<text>\"`) before `freeze` and the main run.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------
# Gate 4 — seal
# ------------------------------------------------------------------------------------------------------------------

def cell_name(arm: str, rung: int, repeat: int) -> str:
    return f"{arm}/{rung}" if int(repeat) <= 1 else f"{arm}/{rung}/r{int(repeat)}"


def evaluate_gate4(cfg: Mapping[str, Any], planned: Sequence[Mapping[str, Any]], outcomes: Any,
                   reconciliation: Mapping[str, Any] | None, *, grader_rerun_mismatches: int | None,
                   ladder_estimates: Mapping[str, float] | None, sealed_tables: Mapping[str, str] | None,
                   stage: str = "main") -> GateResult:
    """Plan §9 Gate 4: every planned cell (``{arm, rung, item_id, repeat}``) has a finalised
    ``policy_outcomes_dynorch`` row; completeness ≥ 0.95 per (arm, rung) (below: partial, excluded from primary
    p-values — a partial primary cell is stop_notify); /key delta within $0.50 or 2 %; grader re-run identical;
    sealed ladder point estimates reproduced; tables sealed (path → sha256)."""
    require_columns(outcomes, ("name", "item_id", "y", "delivered_state"), "policy_outcomes_dynorch")
    crits: list[Crit] = []
    have = {(str(r["name"]), str(r["item_id"])): r for r in outcomes.to_dict("records")}
    missing, pending = [], []
    for cell in planned:
        key = (cell_name(cell["arm"], int(cell["rung"]), int(cell.get("repeat", 1))), str(cell["item_id"]))
        row = have.get(key)
        if row is None:
            missing.append("/".join(key))
        elif row["delivered_state"] == "pending":
            pending.append("/".join(key))
    crits.append(crit("every_planned_cell_has_row", {"planned": len(planned), "missing": missing[:20], "pending": pending[:20]},
                      "one finalised row per planned cell", bool(planned) and not missing and not pending,
                      "tables/policy_outcomes_dynorch.parquet", severity=STOP))
    completeness: dict[str, float] = {}
    for name, group in outcomes.groupby("name"):
        completeness[str(name)] = float(group["y"].notna().mean())
    partial = sorted(n for n, c in completeness.items() if c < G4_COMPLETENESS_MIN)
    primary_partial = [n for n in partial if n in PRIMARY_CELLS]
    crits.append(crit("completeness_per_cell", {"partial": partial, "completeness": {k: round(v, 4) for k, v in sorted(completeness.items())}},
                      f">= {G4_COMPLETENESS_MIN} per (arm, rung); partial cells are excluded from primary p-values; "
                      "a partial primary cell (H1/16, H3/16) stops for the owner", not primary_partial,
                      "tables/policy_outcomes_dynorch.parquet"))
    if reconciliation is None:
        crits.append(crit("reconciliation_present", None, f"state/reconciliation_{stage}.json written", False, "state/"))
    else:
        ledger_delta = abs(Decimal(str(reconciliation["ledger_delta_usd"])))
        tolerance = max(G4_KEY_DELTA_USD, ledger_delta * G4_KEY_DELTA_SHARE)
        crits.extend(reconciliation_crits(reconciliation, tolerance_usd=tolerance, evidence=f"state/reconciliation_{stage}.json"))
    crits.append(crit("grader_rerun_identical", grader_rerun_mismatches, "0 mismatches on re-grading every VALID final answer",
                      grader_rerun_mismatches == 0, "grading/text/grade.py re-run"))
    if ladder_estimates is None:
        crits.append(crit("sealed_ladder_reproduced", None, "deep/16 = 0.7917 and sample/16 = 0.4167 from the merged frame", False,
                          "forced_budget_v2/merged/tables/policy_outcomes.parquet"))
    else:
        bad = {k: ladder_estimates.get(k) for k, v in SEALED_LADDER.items()
               if ladder_estimates.get(k) is None or abs(float(ladder_estimates[k]) - v) > LADDER_TOLERANCE}
        crits.append(crit("sealed_ladder_reproduced", {k: ladder_estimates.get(k) for k in SEALED_LADDER}, "deep/16 = 0.7917, sample/16 = 0.4167",
                          not bad, "forced_budget_v2/merged/tables/policy_outcomes.parquet", severity=STOP))
    crits.append(crit("tables_sealed", sorted(sealed_tables) if sealed_tables else None,
                      "dynorch_runs, dynorch_calls, policy_outcomes_dynorch, policy_costs_dynorch hashed in tables/SEAL.json",
                      bool(sealed_tables) and all(f"{t}.parquet" in sealed_tables for t in
                                                  ("dynorch_runs", "dynorch_calls", "policy_outcomes_dynorch", "policy_costs_dynorch")),
                      "tables/SEAL.json"))
    return finish("gate4", stage, crits)


def ladder_estimates_from_base(base_run_root: Path) -> dict[str, float]:
    """Macro accuracy (mean over benchmarks of the mean y over items) of the sealed ladder cells, computed from the
    read-only ``merged/tables/policy_outcomes.parquet`` (``view == current``, ``partition_id == main``)."""
    import pandas as pd
    table = pd.read_parquet(Path(base_run_root) / "merged" / "tables" / "policy_outcomes.parquet")
    current = table[(table["view"] == "current") & (table["partition_id"] == "main")]
    out: dict[str, float] = {}
    for name in SEALED_LADDER:
        rows = current[current["name"] == name]
        if len(rows) != 96:
            raise GateInputError(f"sealed ladder cell {name} has {len(rows)} rows, expected 96")
        out[name] = float(rows.groupby("benchmark")["y"].mean().mean())
    return out


def seal_tables(tables_dir: Path, names: Sequence[str] = ("dynorch_runs", "dynorch_calls", "policy_outcomes_dynorch",
                                                          "policy_costs_dynorch")) -> dict[str, str]:
    """``tables/SEAL.json``: sha256 per parquet (write-once; an existing seal that differs raises)."""
    from hashlib import sha256
    from agent_allocation.storage._atomic import write_once
    digests: dict[str, str] = {}
    for name in names:
        path = Path(tables_dir) / f"{name}.parquet"
        if not path.exists():
            raise GateInputError(f"cannot seal: {path} is missing")
        digests[path.name] = sha256(path.read_bytes()).hexdigest()
    seal = Path(tables_dir) / "SEAL.json"
    doc = {"sealed_at": utcnow(), "sha256": digests}
    if seal.exists():
        existing = json.loads(seal.read_text())["sha256"]
        if existing != digests:
            raise GateInputError(f"{seal} exists with different digests; the tables changed after sealing")
        return existing
    write_once(seal, json.dumps(doc, indent=1, sort_keys=True).encode())
    return digests


#: Criteria an owner may waive (contract §10.37): ONLY those that measure whether a harness delegates. Money,
#: accounting, grants, provider pinning, answers, hygiene and every invariant stay binding; a ``stop`` is never waived.
WAIVABLE: dict[str, frozenset[str]] = {
    "gate1": frozenset({"ceiling_probe_spawns", "codex_responses_tool_loop"}),
    "gate2": frozenset({"H1_spawn_share_k16", "H3_spawn_share_k16"}),
}
WAIVER_FILE = "owner_waivers.json"


class WaiverError(ValueError):
    """A waiver that names a criterion outside ``WAIVABLE`` or carries no owner text."""


def waiver_path(paths: Any) -> Path:
    return Path(paths.state_dir) / WAIVER_FILE


def load_waivers(paths: Any) -> dict[str, dict[str, dict[str, str]]]:
    """``{gate: {criterion: {"text", "recorded_at"}}}`` -- the owner's standing waivers (empty when none)."""
    target = waiver_path(paths)
    return json.loads(target.read_text()) if target.exists() else {}


def record_waivers(paths: Any, gate: str, names: Sequence[str], text: str) -> dict[str, dict[str, dict[str, str]]]:
    """Store standing owner waivers for ``gate`` (contract §10.37). Refuses an empty text or any name outside
    ``WAIVABLE[gate]``; nothing is written on refusal."""
    if not text or not text.strip():
        raise WaiverError("a waiver needs the owner's text")
    allowed = WAIVABLE.get(gate, frozenset())
    bad = sorted(set(names) - allowed)
    if bad:
        raise WaiverError(f"not waivable at {gate}: {bad} (waivable: {sorted(allowed)})")
    waivers = load_waivers(paths)
    stamp = utcnow()
    waivers.setdefault(gate, {}).update({name: {"text": text.strip(), "recorded_at": stamp} for name in names})
    target = waiver_path(paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_atomic(target, (json.dumps(waivers, indent=1, sort_keys=True) + "\n").encode())
    return waivers


def apply_waivers(result: GateResult, waivers: Mapping[str, Mapping[str, Mapping[str, str]]]) -> GateResult:
    """The gate result under the owner's standing waivers (contract §10.37). A waived criterion stays FAILED
    (``passed`` is not rewritten; its threshold is prefixed ``WAIVED by owner``); the decision becomes ``advance``
    only when every failing criterion is waived, and a ``stop`` decision is returned unchanged."""
    gate_waivers = waivers.get(result.gate, {})
    if result.decision == STOP or not gate_waivers:
        return result
    criteria = tuple(
        replace(c, threshold=f"WAIVED by owner ({gate_waivers[c.name]['recorded_at']}): {c.threshold}")
        if (not c.passed and c.name in gate_waivers) else c
        for c in result.criteria)
    unwaived = [c for c in result.criteria if not c.passed and c.name not in gate_waivers]
    return replace(result, criteria=criteria, decision=ADVANCE if not unwaived else result.decision)


def record(paths: Any, result: GateResult) -> Path:
    """Write the result under ``<run_root>/gates/<gate>_<stage>.json`` via the base study's ``write_gate``."""
    write_gate(paths, result)
    return paths.gate_result(result.gate, result.stage)
