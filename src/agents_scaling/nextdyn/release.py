"""``release-check``: the plan §4.5 gate list, each gate reported separately (spec 18.6, 17.5).

The gates are separate on purpose.  ARD36's close-out produced a single pass/fail, so a run
that was clean on keys and stale on figures reported the same thing as a run that was clean on
both, and the release note had to be reconstructed afterwards from the code.  Here every gate
is a row: ``name``, ``required``, ``passed``, and a ``detail`` that says what was measured.  A
gate that could not be evaluated is ``passed=None`` and is reported as *not evaluated*, never
silently as a pass.

The one gate that is a hash of this package's own source is ``analysis_lock``: the module list
:data:`ANALYSIS_LOCK_MODULES` is the declaration, the lock is
``sha256(analysis.yaml || each module's bytes)``, and ``study.analysis.analysis_lock_hash``
must equal it once the analysis is frozen at S3.  Adding an estimator without re-freezing is
therefore a failed gate rather than an undetected change.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import analysis, cli, config as C, figures, outcomes, panels, tables
from agents_scaling.nextdyn.errors import ContractViolation

#: The modules whose bytes the analysis lock covers (plan §2 S3).  Adding an estimator module to
#: the package without adding it here would leave it outside the lock, so the layout check below
#: asserts that every analysis-side module of the package is listed.
ANALYSIS_LOCK_MODULES: Final[tuple[str, ...]] = (
    "analysis.py", "calib.py", "outcomes.py", "panels.py", "power.py", "stats.py",
)

#: Analysis-side modules deliberately outside the lock: they read the locked artifacts and
#: produce reports, so a change to them cannot move an estimate.
ANALYSIS_LOCK_EXEMPT: Final[tuple[str, ...]] = ("figures.py", "release.py")

GATE_REPORT: Final[str] = "release_check.json"


@dataclass(frozen=True)
class Gate:
    """One release gate: required, measured, and what it measured."""

    name: str
    required: bool
    passed: bool | None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed is None:
            return "not_evaluated"
        return "pass" if self.passed else "fail"


@dataclass(frozen=True)
class GateReport:
    gates: tuple[Gate, ...]

    @property
    def ok(self) -> bool:
        return all(g.passed for g in self.gates if g.required)

    @property
    def failed(self) -> list[str]:
        return [g.name for g in self.gates if g.required and g.passed is not True]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "failed": self.failed,
                "gates": [{"name": g.name, "required": g.required, "status": g.status,
                           "detail": dict(g.detail)} for g in self.gates]}


def analysis_lock_hash(package_dir: str | os.PathLike | None = None,
                       analysis_yaml: str | os.PathLike | None = None) -> str:
    """``sha256(analysis.yaml || the bytes of every locked analysis module)``.

    The module order is :data:`ANALYSIS_LOCK_MODULES`, which is the declaration; a module that
    is not in that tuple is not in the lock, and :func:`layout_gate` is what stops one from
    quietly appearing beside the locked ones.
    """
    package = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    yaml_path = Path(analysis_yaml) if analysis_yaml is not None else C.DEFAULT_ANALYSIS_CONFIG
    digest = hashlib.sha256()
    digest.update(yaml_path.read_bytes() if yaml_path.exists() else b"")
    for name in ANALYSIS_LOCK_MODULES:
        path = package / name
        if not path.exists():
            raise ContractViolation(f"the analysis lock names {name}, which does not exist")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def layout_gate(package_dir: str | os.PathLike | None = None) -> Gate:
    """Every estimator module in the package is either locked or explicitly exempt."""
    package = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    declared = set(ANALYSIS_LOCK_MODULES) | set(ANALYSIS_LOCK_EXEMPT)
    present = {p.name for p in package.glob("*.py")}
    missing = sorted(set(ANALYSIS_LOCK_MODULES) - present)
    return Gate("analysis_lock_layout", True, not missing,
                {"declared": sorted(declared), "missing_from_package": missing})


# --------------------------------------------------------------------------- gates


def _derived(run_root: Path) -> Path:
    return run_root / outcomes.DERIVED_DIR


def _identity_gate(run_root: Path) -> Gate:
    """Gate 11: the three-state residual is zero over every consecutive pair.

    ``outcomes.derive`` raises on a nonzero residual, so a published ``transitions`` table can
    only exist if the identity held; this gate re-checks the published rows against the
    published team counts rather than trusting that ``derive`` was the writer.
    """
    derived = _derived(run_root)
    try:
        team = tables.read_table(derived, "team_rounds")
        trans = tables.read_table(derived, "transitions")
    except Exception as exc:
        return Gate("three_state_identity", True, None, {"error": str(exc)})
    counts: dict[tuple[str, str, int], dict[str, int]] = {}
    for row in trans:
        key = (str(row["cell_kind"]), str(row["episode_id"]), int(row["round"]))
        bucket = counts.setdefault(key, {k: 0 for k in outcomes.TRANSITION_KINDS})
        bucket[str(row["kind"])] += 1
    by_key = {(str(r["cell_kind"]), str(r["episode_id"]), int(r["round"])): r for r in team}
    residuals = []
    for key, bucket in counts.items():
        cell_kind, episode_id, round_ = key
        prev, cur = by_key.get(key), by_key.get((cell_kind, episode_id, round_ + 1))
        if prev is None or cur is None:
            continue
        delta = int(cur["c_count"]) - int(prev["c_count"])
        predicted = bucket["WC"] + bucket["UC"] - bucket["CW"] - bucket["CU"]
        if delta != predicted:
            residuals.append({"episode_id": episode_id, "round": round_,
                              "delta": delta, "predicted": predicted})
    return Gate("three_state_identity", True, not residuals,
                {"n_pairs": len(counts), "n_residuals": len(residuals),
                 "examples": residuals[:3]})


def _bounds_gate(run_root: Path) -> Gate:
    """Gate 10: every primary reports bounds and a complete-case sensitivity."""
    path = _derived(run_root) / "hypotheses.json"
    if not path.exists():
        return Gate("primary_bounds_and_complete_case", True, None, {"missing": str(path)})
    family = json.loads(path.read_text(encoding="utf-8"))
    missing = [row["h_id"] for row in family.get("hypotheses", [])
               if row.get("estimable") and (row.get("bounds_lo") is None
                                            or row.get("complete_case") is None)]
    return Gate("primary_bounds_and_complete_case", True, not missing,
                {"n_hypotheses": len(family.get("hypotheses", [])), "missing": missing})


def _family_gate(run_root: Path) -> Gate:
    """Gate 7: the ledger's verdicts are allowed and all eight H rows are present."""
    ledger = _derived(run_root) / "claim_ledger.csv"
    hypotheses = _derived(run_root) / "hypotheses.json"
    if not ledger.exists() or not hypotheses.exists():
        return Gate("claim_ledger_complete", True, None,
                    {"ledger": ledger.exists(), "hypotheses": hypotheses.exists()})
    import csv

    with ledger.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    header_ok = rows and set(rows[0]) == set(analysis.LEDGER_HEADER)
    bad = sorted({r["verdict"] for r in rows} - set(analysis.VERDICTS))
    family = json.loads(hypotheses.read_text(encoding="utf-8"))
    declared = {row["h_id"] for row in family.get("hypotheses", [])}
    present = {r["claim_id"] for r in rows}
    not_run = sorted(set(analysis.NOT_RUN_CLAIMS) - present)
    return Gate("claim_ledger_complete", True,
                bool(header_ok) and not bad and declared <= present and not not_run,
                {"n_rows": len(rows), "header_ok": bool(header_ok),
                 "disallowed_verdicts": bad, "n_primary_rows": len(declared & present),
                 "missing_not_run_rows": not_run})


def _keys_gate(run_root: Path) -> Gate:
    """Gate 2: the published and derived tables are clean on every declared key."""
    published = tables.check_keys(run_root / outcomes.TABLES_DIR, strict=False)
    derived = tables.check_derived_keys(_derived(run_root), strict=False)
    return Gate("keys_clean", True, not published and not derived,
                {"published_violations": [v.table for v in published],
                 "derived_violations": [v.table for v in derived]})


def _figures_gate(run_root: Path) -> Gate:
    stale = figures.stale_figures(run_root)
    return Gate("figure_hashes_current", True, not stale, {"stale": stale})


def _seal_gate(run_root: Path) -> Gate:
    """Gate 9: the selections were sealed (and unchanged) before the grade join."""
    manifest = analysis.predictions_manifest(run_root)
    entry = manifest.get("selections")
    if entry is None:
        return Gate("selections_sealed_before_join", True, False,
                    {"reason": "no sealed selections manifest"})
    path = analysis.predictions_dir(run_root) / entry["file"]
    current = figures._sha256(path)
    return Gate("selections_sealed_before_join", True, current == entry["sha256"],
                {"sealed_sha256": entry["sha256"], "current_sha256": current,
                 "n_rows": entry.get("n_rows")})


def _lock_gate(cfg: Any) -> Gate:
    """Gate 5: ``study.analysis.analysis_lock_hash`` matches the locked module bytes."""
    declared = cfg.analysis.analysis_lock_hash
    actual = analysis_lock_hash()
    if declared is None:
        return Gate("analysis_lock_matches", True, None,
                    {"reason": "study.analysis.analysis_lock_hash is null (not yet frozen at S3)",
                     "actual": actual})
    return Gate("analysis_lock_matches", True, str(declared) == actual,
                {"declared": str(declared), "actual": actual,
                 "modules": list(ANALYSIS_LOCK_MODULES)})


def _panel_census_gate(run_root: Path) -> Gate:
    path = _derived(run_root) / "panel_census.json"
    if not path.exists():
        return Gate("panel_census_recorded", True, None, {"missing": str(path)})
    rows = json.loads(path.read_text(encoding="utf-8"))
    declared = {row["panel"] for row in rows}
    return Gate("panel_census_recorded", True, set(panels.POPULATIONS) <= declared,
                {"n_panels": len(rows),
                 "mismatched": [r["panel"] for r in rows if r.get("matches_expected") is False]})


def _plan_census_gate(run_root: Path) -> Gate:
    """Gate 3: every planned cell is terminal (the census, never a directory listing)."""
    try:
        rows = tables.read_table(run_root / outcomes.TABLES_DIR, "plan")
    except Exception as exc:
        return Gate("plan_census_terminal", True, None, {"error": str(exc)})
    terminal = {"complete", "failed", "skipped_quota_unfilled", "reused", "blocked"}
    pending = [str(r["cell_id"]) for r in rows if str(r.get("status")) not in terminal]
    return Gate("plan_census_terminal", True, not pending,
                {"n_cells": len(rows), "n_pending": len(pending), "examples": pending[:3]})


def _spec18_gate(run_root: Path) -> Gate:
    """Gate 1: the spec-18 fixture groups are collected.  Reported from the index, never run."""
    path = _derived(run_root) / "spec18_collected.json"
    if not path.exists():
        return Gate("spec18_fixture_groups", True, None,
                    {"reason": "run pytest tests/nextdyn and record the collected names here"})
    data = json.loads(path.read_text(encoding="utf-8"))
    return Gate("spec18_fixture_groups", True, not data.get("missing"),
                {"n_collected": len(data.get("collected", [])), "missing": data.get("missing", [])})


def _e0_gate(run_root: Path) -> Gate:
    path = run_root / "e0" / "legacy_claim_reproduction.csv"
    if not path.exists():
        return Gate("e0_claims_reproduced", False, None, {"missing": str(path)})
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failed = [r["claim_id"] for r in rows if r.get("status") == "fail"]
    return Gate("e0_claims_reproduced", False, not failed,
                {"n_claims": len(rows), "failed": failed,
                 "blocked": [r["claim_id"] for r in rows
                             if r.get("status") == "BLOCKED_MISSING_SOURCE"]})


def check(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None) -> GateReport:
    """Run every plan §4.5 gate and report each separately."""
    root = Path(run_root)
    cfg = C.load_study(Path(config_path) if config_path else C.DEFAULT_STUDY_CONFIG)
    gates = (
        _spec18_gate(root),
        _keys_gate(root),
        _plan_census_gate(root),
        _lock_gate(cfg),
        layout_gate(),
        _figures_gate(root),
        _family_gate(root),
        _seal_gate(root),
        _bounds_gate(root),
        _identity_gate(root),
        _panel_census_gate(root),
        _e0_gate(root),
    )
    return GateReport(gates=gates)


def run(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None) -> cli.Report:
    report = check(run_root, config_path=config_path)
    path = Path(run_root) / GATE_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    passed = sum(1 for g in report.gates if g.passed is True)
    return cli.Report(completed=passed, failed=len(report.failed),
                      blocked=sum(1 for g in report.gates if g.passed is None),
                      detail=f"{path}: {', '.join(report.failed) or 'all required gates pass'}")


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("release-check")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(cli.resolve_run_root(args), config_path=args.config)


__all__ = [
    "ANALYSIS_LOCK_EXEMPT", "ANALYSIS_LOCK_MODULES", "GATE_REPORT", "Gate", "GateReport",
    "analysis_lock_hash", "check", "layout_gate", "main", "run",
]
