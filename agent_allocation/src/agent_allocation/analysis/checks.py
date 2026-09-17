"""Audit invariants (MASTER_SPEC §12 and §15) computed from stored outputs.

Each check returns :class:`CheckResult` with ``status`` in ``pass | fail | skip``. A skipped check names why it
could not run (e.g. no fully known panel yet); a skip is never reported as a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent_allocation.analysis.load import RunData

PARTITION_PANEL = {"main": "main", "high": "high", "replication": "replication", "pilot": "pilot"}
EXTENSION_FAMILIES = frozenset({"cp", "pooled"})


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str                 # pass | fail | skip
    detail: str
    evidence: str = ""
    violations: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, Any]:
        return {"check": self.name, "status": self.status, "detail": self.detail, "evidence": self.evidence,
                "n_violations": len(self.violations), "violations": "; ".join(self.violations[:20])}


def _result(name: str, violations: list[str], detail_pass: str, evidence: str) -> CheckResult:
    if violations:
        return CheckResult(name, "fail", f"{len(violations)} violation(s)", evidence, tuple(violations))
    return CheckResult(name, "pass", detail_pass, evidence)


# ----------------------------------------------------------------------------------------------------
# Fixed population
# ----------------------------------------------------------------------------------------------------

def check_fixed_population(run: RunData, outcomes: pd.DataFrame) -> CheckResult:
    """Every checkpoint of a (partition, benchmark, view) covers the same planned items.

    Extension-only families (CP, POOLED) must share one item set that is a subset of the base population;
    when waves exist, each partition's population must be whole waves forming a prefix of that panel's
    wave order (no outcome-dependent removal at the manifest level).
    """
    violations: list[str] = []
    if outcomes.empty:
        return CheckResult("fixed_population", "skip", "no planned checkpoints in the policy manifests",
                           "manifests/policies/")
    for (partition, benchmark, view), group in outcomes.groupby(["partition_id", "benchmark", "view"],
                                                               sort=True, dropna=False):
        sets = {name: frozenset(g["item_id"]) for name, g in group.groupby("name", sort=True)}
        dup = group.groupby("name")["item_id"].apply(lambda s: s.duplicated().any())
        for name in dup[dup].index:
            violations.append(f"{partition}/{benchmark}/{view}: {name} lists an item twice")
        base_names = [n for n in sets if group.loc[group["name"] == n, "family"].iloc[0] not in EXTENSION_FAMILIES]
        ext_names = [n for n in sets if n not in base_names]
        if base_names:
            base = sets[base_names[0]]
            for name in base_names[1:]:
                if sets[name] != base:
                    violations.append(f"{partition}/{benchmark}/{view}: {name} has {len(sets[name])} items, "
                                      f"{base_names[0]} has {len(base)}")
        else:
            base = frozenset()
        if ext_names:
            ext = sets[ext_names[0]]
            for name in ext_names[1:]:
                if sets[name] != ext:
                    violations.append(f"{partition}/{benchmark}/{view}: extension checkpoint {name} population "
                                      f"differs from {ext_names[0]}")
            if base_names and not ext <= base:
                violations.append(f"{partition}/{benchmark}/{view}: extension population is not nested in the "
                                  f"base population")
    if run.waves:
        violations.extend(_wave_prefix_violations(run, outcomes))
    return _result("fixed_population", violations,
                   "every checkpoint in each partition x benchmark x view covers the same planned items"
                   + ("" if run.waves else " (waves.json absent: wave-prefix check not run)"),
                   "manifests/policies/*.jsonl, manifests/waves.json")


def _wave_prefix_violations(run: RunData, outcomes: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    for partition, group in outcomes.groupby("partition_id", sort=True):
        panel = PARTITION_PANEL.get(str(partition))
        if panel is None:
            continue
        checks = [(panel, group[~group["family"].isin(EXTENSION_FAMILIES)])]
        if partition == "main":
            checks.append(("extension", group[group["family"].isin(EXTENSION_FAMILIES)]))
        for panel_name, subset in checks:
            if subset.empty:
                continue
            items = set(subset["item_id"])
            panel_waves = [w for w in sorted(run.waves, key=lambda w: w.index) if panel_name in w.panels]
            if not panel_waves:
                violations.append(f"{partition}: panel {panel_name!r} has checkpoints but no waves")
                continue
            included = [w for w in panel_waves if set(w.item_ids) & items]
            for wave in included:
                missing = set(wave.item_ids) - items
                if missing:
                    violations.append(f"{partition}/{panel_name}: wave {wave.index} partially planned "
                                      f"({len(missing)} items absent)")
            if included:
                last = max(panel_waves.index(w) for w in included)
                skipped = [w.index for w in panel_waves[: last + 1] if w not in included]
                if skipped:
                    violations.append(f"{partition}/{panel_name}: waves {skipped} skipped inside the planned prefix")
            stray = items - {i for w in panel_waves for i in w.item_ids}
            if stray:
                violations.append(f"{partition}/{panel_name}: {len(stray)} planned items belong to no "
                                  f"{panel_name!r} wave")
    return violations


# ----------------------------------------------------------------------------------------------------
# Sealed before join
# ----------------------------------------------------------------------------------------------------

def check_sealed_before_join(run: RunData, outcomes: pd.DataFrame) -> CheckResult:
    """Every selections file is sealed and unmodified, and no delivered outcome is joined from an unsealed one."""
    violations: list[str] = []
    for partition, seal in sorted(run.seals.items()):
        if seal.selections_present and not seal.valid:
            violations.append(f"{partition}: selection batches present but not all verify "
                              f"({'; '.join(seal.all_problems)})")
    if not outcomes.empty:
        joined = outcomes[outcomes["y"].notna() & ~outcomes["sealed"]]
        for partition, count in joined.groupby("partition_id").size().items():
            violations.append(f"{partition}: {count} delivered outcomes joined without a valid seal")
    sealed = sorted(f"{p} ({len(s.batches)} batches)" for p, s in run.seals.items() if s.valid)
    return _result("sealed_before_join", violations,
                   f"batch seals verify for {sealed or 'no partitions (no selections yet)'}; grades joined only to "
                   "verified batches", "selections/{partition}/batch_NNNN.jsonl + .seal.json")


# ----------------------------------------------------------------------------------------------------
# MASTER_SPEC §12 algebraic checks (fully known panels) and accounting invariants
# ----------------------------------------------------------------------------------------------------

TOL = 1e-9


def check_pass_at_1(bank: pd.DataFrame) -> CheckResult:
    """Finite-bank estimated pass@1 equals mean draw correctness over the whole bank."""
    if bank.empty:
        return CheckResult("pass_at_1_equals_mean_draw_correctness", "skip", "no sample checkpoints", "tables/sample_bank")
    rows = bank[(bank["k"] == 1) & bank["oracle_pass_at_k"].notna()]
    if rows.empty:
        return CheckResult("pass_at_1_equals_mean_draw_correctness", "skip", "no item has a fully known bank",
                           "tables/sample_bank")
    bad = rows[(rows["oracle_pass_at_k"].astype(float) - rows["mean_draw_correct"].astype(float)).abs() > TOL]
    return _result("pass_at_1_equals_mean_draw_correctness",
                   [f"{r.partition_id}/{r.item_id}: pass@1={r.oracle_pass_at_k} mean={r.mean_draw_correct}"
                    for r in bad.itertuples()], f"{len(rows)} fully known banks", "tables/sample_bank")


def check_pass_at_bank(bank: pd.DataFrame) -> CheckResult:
    """pass@K on a K-draw bank equals full-bank coverage."""
    name = "pass_at_bank_size_equals_full_bank_coverage"
    if bank.empty:
        return CheckResult(name, "skip", "no sample checkpoints", "tables/sample_bank")
    full = bank[(bank["k"] == bank["bank_size"]) & bank["oracle_pass_at_k"].notna() & bank["prefix_coverage"].notna()]
    if full.empty:
        return CheckResult(name, "skip", "no item has a fully known bank", "tables/sample_bank")
    bad = full[(full["oracle_pass_at_k"].astype(float) - full["prefix_coverage"].astype(float)).abs() > TOL]
    return _result(name, [f"{r.partition_id}/{r.item_id}: pass@{r.k}={r.oracle_pass_at_k} coverage={r.prefix_coverage}"
                          for r in bad.itertuples()], f"{len(full)} fully known banks", "tables/sample_bank")


def check_archive_monotone(outcomes: pd.DataFrame) -> CheckResult:
    """Archive coverage cannot decrease as candidates are added; archive coverage >= current coverage."""
    from agent_allocation.analysis.names import CheckpointNameError, parse_name

    name = "archive_coverage_monotone"
    archive = outcomes[outcomes["view"] == "archive"]
    if archive.empty:
        return CheckResult(name, "skip", "no archive-view checkpoints in the manifests", "manifests/policies/")
    violations: list[str] = []
    compared = 0
    series: dict[tuple, list[tuple[int, float]]] = {}
    for rec in archive[["partition_id", "item_id", "name", "coverage"]].itertuples(index=False):
        try:
            traj = parse_name(rec.name).trajectory
        except CheckpointNameError:
            continue
        if traj is None:
            continue
        series.setdefault((rec.partition_id, rec.item_id, traj[0]), []).append((traj[1], rec.coverage))
    for key, points in series.items():
        points.sort()
        for (s0, c0), (s1, c1) in zip(points, points[1:]):
            if pd.isna(c0) or pd.isna(c1):
                continue
            compared += 1
            if c1 < c0:
                violations.append(f"{key[0]}/{key[1]}/{'/'.join(map(str, key[2]))}: archive coverage {c0} at step "
                                  f"{s0} -> {c1} at step {s1}")
    cur = outcomes[outcomes["view"] == "current"].set_index(["partition_id", "item_id", "name"])["coverage"]
    arc = archive.set_index(["partition_id", "item_id", "name"])["coverage"]
    joined = pd.concat([cur.rename("current"), arc.rename("archive")], axis=1, join="inner").dropna()
    for idx, row in joined[joined["archive"] < joined["current"]].iterrows():
        violations.append(f"{'/'.join(map(str, idx))}: archive coverage {row['archive']} < current {row['current']}")
    if not compared and joined.empty:
        return CheckResult(name, "skip", "no fully known archive sequences", "tables/policy_outcomes")
    return _result(name, violations, f"{compared} consecutive archive pairs, {len(joined)} archive/current pairs",
                   "tables/policy_outcomes")


def check_transition_identity(transitions: pd.DataFrame) -> CheckResult:
    """C_next - C_prev = (W->C + I->C) - (C->W + C->I) on slot-aligned known transitions."""
    name = "transition_identity"
    if transitions.empty:
        return CheckResult(name, "skip", "no trajectory transitions", "tables/transitions")
    violations = [f"{r.partition_id}/{r.item_id}/{r.trajectory} {r.from_step}->{r.to_step}: pool sizes "
                  f"{r.slots_prev} vs {r.slots}"
                  for r in transitions[transitions["slots_prev"] != transitions["slots"]].itertuples()
                  if not str(r.trajectory).startswith("sample")]
    known = transitions[transitions["known"].astype(bool)]
    if known.empty:
        return CheckResult(name, "skip" if not violations else "fail", "no fully known transitions",
                           "tables/transitions", tuple(violations))
    lhs = known["correct_next"].astype(int) - known["correct_prev"].astype(int)
    rhs = (known["W_to_C"].astype(int) + known["I_to_C"].astype(int)
           - known["C_to_W"].astype(int) - known["C_to_I"].astype(int))
    for r in known[lhs != rhs].itertuples():
        violations.append(f"{r.partition_id}/{r.item_id}/{r.trajectory} {r.from_step}->{r.to_step}")
    return _result(name, violations, f"{len(known)} known transitions", "tables/transitions")


def check_delivered_within_coverage(outcomes: pd.DataFrame) -> CheckResult:
    """Candidate-only selectors return a pool member, so delivered accuracy <= coverage of that same pool.
    POOLED synthesis is exempt (it may exceed its input bank's coverage)."""
    from agent_allocation.analysis.outcomes import SYNTHESIS_FAMILIES

    name = "delivered_within_coverage_candidate_only"
    scoped = outcomes[~outcomes["family"].isin(SYNTHESIS_FAMILIES) & outcomes["y"].notna()]
    violations = [f"{r.policy_id}: delivered 1 with pool coverage 0"
                  for r in scoped[(scoped["y"] == 1.0) & (scoped["coverage"] == 0.0)].itertuples()]
    violations += [f"{r.policy_id}: selected candidate outside the declared pool"
                   for r in outcomes[outcomes["selected_in_pool"] == False].itertuples()  # noqa: E712
                   if r.family not in SYNTHESIS_FAMILIES]
    synth = outcomes[outcomes["family"].isin(SYNTHESIS_FAMILIES) & outcomes["y"].notna()
                     & outcomes["input_coverage"].notna()]
    exceed = int(((synth["y"] == 1.0) & (synth["input_coverage"] == 0.0)).sum())
    if scoped.empty:
        return CheckResult(name, "skip", "no known delivered outcomes", "tables/policy_outcomes", tuple(violations))
    return _result(name, violations, f"{len(scoped)} candidate-only rows checked; synthesis rows exempt "
                                     f"({exceed} POOLED rows exceed input-bank coverage, which is allowed)",
                   "tables/policy_outcomes")


def check_unknown_not_coerced(outcomes: pd.DataFrame) -> CheckResult:
    from agent_allocation.analysis.outcomes import KNOWN_STATES, UNKNOWN_STATES

    name = "unknown_outcomes_not_coerced"
    violations = [f"{r.policy_id}: state {r.delivered_state} carries y={r.y}"
                  for r in outcomes[outcomes["delivered_state"].isin(UNKNOWN_STATES) & outcomes["y"].notna()].itertuples()]
    known = outcomes[outcomes["delivered_state"].isin(list(KNOWN_STATES))]
    violations += [f"{r.policy_id}: state {r.delivered_state} has y={r.y}"
                   for r in known.itertuples() if r.y != KNOWN_STATES[r.delivered_state]]
    return _result(name, violations, f"{len(outcomes)} rows: unknown and unrun states stay NaN",
                   "tables/policy_outcomes")


def check_nominal_generations(costs: pd.DataFrame) -> CheckResult:
    name = "policy_ancestor_counts"
    if costs is None or costs.empty:
        return CheckResult(name, "skip", "no policy cost rows", "tables/policy_costs")
    current = costs[(costs["view"] == "current") & costs["nominal_generations_expected"].notna()]
    bad = current[current["nominal_generations"] != current["nominal_generations_expected"].astype(int)]
    return _result(name, [f"{r.policy_id}: {r.nominal_generations} ancestors, expected {r.nominal_generations_expected}"
                          for r in bad.itertuples()],
                   f"{len(current)} checkpoints charge n(1+R)-style ancestor counts", "tables/policy_costs")


MONOTONE_MEASURES = ("nominal_generations", "requested_reasoning_tokens", "success_cost_usd",
                     "standardized_uncached_cost_usd", "realized_tokens", "total_charged_usd")


def check_cumulative_costs(costs: pd.DataFrame) -> CheckResult:
    """Along nested prefixes (sample/k, private_single/t, private_team/n/r, peer/n/r, cp/h) per-item policy costs
    never decrease. Accuracy is deliberately not required to be monotone."""
    from agent_allocation.analysis.names import CheckpointNameError, parse_name

    name = "cumulative_costs_monotone"
    if costs is None or costs.empty:
        return CheckResult(name, "skip", "no policy cost rows", "tables/policy_costs")
    current = costs[costs["view"] == "current"]
    violations: list[str] = []
    series: dict[tuple, list[tuple[int, dict]]] = {}
    for rec in current.to_dict("records"):
        try:
            traj = parse_name(rec["name"]).trajectory
        except CheckpointNameError:
            continue
        if traj is None:
            continue
        series.setdefault((rec["partition_id"], rec["item_id"], traj[0]), []).append((traj[1], rec))
    pairs = 0
    for key, points in series.items():
        points.sort(key=lambda p: p[0])
        for (s0, a), (s1, b) in zip(points, points[1:]):
            for measure in MONOTONE_MEASURES:
                va, vb = a.get(measure), b.get(measure)
                if va is None or vb is None or pd.isna(va) or pd.isna(vb):
                    continue
                pairs += 1
                if float(vb) + TOL < float(va):
                    violations.append(f"{key[0]}/{key[1]}/{'/'.join(map(str, key[2]))}: {measure} {va} at step "
                                      f"{s0} > {vb} at step {s1}")
    return _result(name, violations, f"{pairs} consecutive-prefix comparisons", "tables/policy_costs")


def check_population_not_outcome_dependent(analysis: Any) -> CheckResult:
    """Primary-contrast populations contain every item planned for all their terms (no outcome-based removal)."""
    name = "primary_populations_fixed"
    primary = analysis.extras.get("primary") if analysis is not None else None
    if primary is None or primary.empty:
        return CheckResult(name, "skip", "no primary contrast rows", "tables/primary_contrasts")
    violations = []
    macro = primary[primary["level"] == "macro"]
    planned = analysis.extras.get("primary_planned", {})
    for contrast, expected in planned.items():
        rows = macro[macro["contrast"] == contrast]
        if not rows.empty and int(rows.iloc[0]["n_items"]) != expected:
            violations.append(f"{contrast}: population {int(rows.iloc[0]['n_items'])} items, planned {expected}")
    return _result(name, violations, "primary populations equal the planned item sets", "tables/primary_contrasts")


def check_price_units(analysis: Any) -> CheckResult:
    rows = analysis.extras.get("price_checks", []) if analysis is not None else []
    if not rows:
        return CheckResult("price_units", "skip", "no price information", "configs, locks/openrouter_endpoints.json")
    failed = [f"{r['check']}: {r['detail']}" for r in rows if not r["passed"]]
    return _result("price_units", failed, "; ".join(r["detail"] for r in rows), "planning_prices, endpoint lock")


def check_ledger(analysis: Any) -> CheckResult:
    rec = analysis.extras.get("reconciliation") if analysis is not None else None
    if rec is None:
        return CheckResult("ledger_invariants", "skip", "no reconciliation computed", "state/ledger_journal.jsonl")
    violations = list(rec["ledger"]["invariant_violations"]) + list(rec["usage"]["attempt_mismatches"])
    return _result("ledger_invariants", violations,
                   f"{rec['ledger']['events']} ledger events; peak exposure {rec['ledger']['peak_exposure_usd']} "
                   f"<= ceiling {rec['ledger']['ceiling_usd']}", "state/ledger_journal.jsonl")


def check_selector_legality(run: RunData) -> CheckResult:
    """Declared ``PolicyCheckpoint.selector_name`` equals the config family for its benchmark (``plurality`` |
    ``public_tests_only``), and every sealed ``SelectionResult.selector_name`` is a concrete selector that
    ``selection.readout.selector_for`` allows for that checkpoint shape (contract v2 R03-5 / R06-1)."""
    name = "selector_legality"
    try:
        from agent_allocation.selection.readout import selector_for
    except ImportError:
        return CheckResult(name, "skip", "selection.readout not importable (agent 06)", "selections/")
    declared_by_bench = {b.get("id"): b.get("selector") for b in run.config.get("benchmarks", [])}
    violations: list[str] = []
    checked = 0
    for partition in run.partitions:
        selections = run.selections.get(partition, {})
        for cp in run.policies.get(partition, []):
            task = run.tasks.get(cp.item_id)
            if task is None:
                violations.append(f"{cp.policy_id}: no public task for the item")
                continue
            expected = declared_by_bench.get(task.benchmark)
            if expected is not None and cp.selector_name != expected:
                violations.append(f"{cp.policy_id}: declared selector {cp.selector_name!r}, config says {expected!r}")
            sel = selections.get(cp.policy_id)
            if sel is None:
                continue
            try:
                legal = {selector_for(cp.family, cp.view, task.output_kind, len(cp.candidate_ids),
                                      math_equivalence=flag) for flag in (True, False)}
            except ValueError as exc:
                violations.append(f"{cp.policy_id}: checkpoint shape has no legal selector ({exc})")
                continue
            checked += 1
            if sel.selector_name not in legal:
                violations.append(f"{cp.policy_id}: concrete selector {sel.selector_name!r} not in {sorted(legal)}")
    return _result(name, violations, f"{checked} sealed selections use a legal concrete selector for their declared "
                                     "family and checkpoint shape", "manifests/policies/, selections/")


def run_checks(run: RunData, outcomes: pd.DataFrame, *, analysis: Any = None) -> list[CheckResult]:
    results = [check_fixed_population(run, outcomes), check_sealed_before_join(run, outcomes),
               check_selector_legality(run), check_unknown_not_coerced(outcomes),
               check_delivered_within_coverage(outcomes), check_archive_monotone(outcomes)]
    if analysis is not None:
        metrics = analysis.extras.get("item_metrics", {})
        bank = metrics.get("sample_bank", pd.DataFrame())
        results += [check_pass_at_1(bank), check_pass_at_bank(bank),
                    check_transition_identity(metrics.get("transitions", pd.DataFrame())),
                    check_nominal_generations(analysis.extras.get("policy_costs")),
                    check_cumulative_costs(analysis.extras.get("policy_costs")),
                    check_population_not_outcome_dependent(analysis), check_price_units(analysis),
                    check_ledger(analysis)]
    return results
