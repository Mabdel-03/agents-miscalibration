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
            violations.append(f"{partition}: selections present but seal invalid ({'; '.join(seal.problems)})")
    if not outcomes.empty:
        joined = outcomes[outcomes["y"].notna() & ~outcomes["sealed"]]
        for partition, count in joined.groupby("partition_id").size().items():
            violations.append(f"{partition}: {count} delivered outcomes joined without a valid seal")
    sealed = sorted(p for p, s in run.seals.items() if s.valid)
    return _result("sealed_before_join", violations,
                   f"seals verify for {sealed or 'no partitions (no selections yet)'}; grades joined only there",
                   "selections/*.jsonl, selections/*.seal.json")


def run_checks(run: RunData, outcomes: pd.DataFrame, *, analysis: Any = None) -> list[CheckResult]:
    return [check_fixed_population(run, outcomes), check_sealed_before_join(run, outcomes)]
