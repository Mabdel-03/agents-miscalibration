"""Estimands, hypotheses, panels and the claim ledger (spec 4.2-4.6, 7.3, 9.6, 13.1, 14, 15.1,
16.1, 20.3; plan §4.2-§4.5).

The ``analyze`` verb.  Four things in this module are structural rather than conventional,
because each of them is an ARD36 failure that a convention did not prevent:

1. **Seal before joining.**  :func:`write_held_out_predictions` writes a predictions file with
   no label-like column and hashes it; :func:`join_labels` refuses to join a grade onto anything
   that was not sealed, or whose bytes changed after sealing (``LabelJoinRefused``).  The order
   ``analyze --seal`` then ``analyze --locked`` is therefore enforced by the data, not by the
   runbook (``ard/analysis.py:660-705`` pattern, made a hard refusal).
2. **The primary family is exactly eight rows.**  ``stats.holm_with_not_estimable`` refuses any
   other size, and a hypothesis whose estimand could not be formed keeps its slot with ``p = 1``
   and the verdict ``NOT_ESTIMABLE``.  Dropping a primary can never make the other seven easier
   to reject (``test_h_family_has_exactly_eight_rows``).
3. **Every primary reports three populations.**  The operational estimate (``lo`` on all planned
   items), the missingness **bounds** and the **complete-case** sensitivity are computed
   together, and a ledger verdict records whether the three agree in sign
   (``missingness_sensitivity``).  A primary that reports only one of them is refused.
4. **A population is named before it is measured.**  Every endpoint is built through
   ``panels.population(...)``, so the row set behind a number is a declared filter with a
   measured census, never an ad-hoc frame (plan §4.2).

Hypothesis definitions are the plan's, verbatim, in :data:`HYPOTHESIS_SPECS`; the arms, the
outcome and the contrast weights are data, so an amendment is a table edit with a new
``estimand_version`` and not a rewrite of an estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np

from agents_scaling.nextdyn import calib, cli, config as C, outcomes, panels, stats, tables
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError, MaterializeError
from agents_scaling.nextdyn.ids import CellKind

# --------------------------------------------------------------------------- firewall


class LabelJoinRefused(LeakageError):
    """A grade was about to be joined onto predictions that were never sealed, or that moved."""


#: Column names that carry an outcome.  A held-out prediction file carrying any of them was
#: built from a frame that had already seen the labels, so sealing it would prove nothing.
LABEL_LIKE: Final[frozenset[str]] = frozenset({
    "semantic_correct", "is_correct", "correct", "selected_correct", "selected_state",
    "gold_ref_hash", "u_reason", "grading_status", "oracle_lo", "oracle_hi", "state",
})

PREDICTIONS_DIR: Final[str] = "predictions"
PREDICTIONS_MANIFEST: Final[str] = "MANIFEST.json"
DERIVED_MANIFEST: Final[str] = "MANIFEST.json"

VERDICT_SUPPORTED: Final[str] = "SUPPORTED_WITHIN_SCOPE"
VERDICT_NARROWED: Final[str] = "NARROWED"
VERDICT_NOT_SUPPORTED: Final[str] = "NOT_SUPPORTED"
VERDICT_INCONCLUSIVE: Final[str] = "INCONCLUSIVE"
VERDICT_NOT_ESTIMABLE: Final[str] = "NOT_ESTIMABLE"
VERDICT_NOT_RUN: Final[str] = "NOT_RUN"
VERDICT_REFUTED: Final[str] = "REFUTED_BY_CONTROL"
VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_SUPPORTED, VERDICT_NARROWED, VERDICT_NOT_SUPPORTED, VERDICT_INCONCLUSIVE,
    VERDICT_NOT_ESTIMABLE, VERDICT_NOT_RUN, VERDICT_REFUTED,
)

#: Spec 20.3 ``claim_ledger.csv``, in the spec's order.  Not a table schema (R1): the ledger is
#: a report, and ``metrics`` is the table that carries the estimates.
LEDGER_HEADER: Final[tuple[str, ...]] = (
    "claim_id", "statement", "evidence_type", "experiment", "estimand_id",
    "source_population", "model_scope", "task_scope", "primary_or_secondary",
    "estimate", "uncertainty", "multiplicity_status", "missingness_sensitivity",
    "audit_status", "supporting_artifact_hash", "verdict",
)

ESTIMAND_VERSION: Final[str] = "v1"


def predictions_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / PREDICTIONS_DIR


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_held_out_predictions(run_root: str | os.PathLike, name: str,
                               rows: Sequence[Mapping[str, Any]], *,
                               clock: Callable[[], float] | None = None) -> dict[str, Any]:
    """Write ``predictions/<name>.json`` with **no** label-like column, and hash it.

    This is the first artifact ``analyze --seal`` produces and the only thing
    :func:`join_labels` will join onto.  A row carrying a label column is refused here, so the
    seal cannot be produced from a frame that already saw the outcome (spec 15.2).
    """
    folder = predictions_dir(run_root)
    offending = sorted({key for row in rows for key in row} & LABEL_LIKE)
    if offending:
        raise LabelJoinRefused(
            f"held-out predictions {name!r} carry label column(s) {offending}; predictions are "
            "sealed before any label join (spec 15.2)")
    path = folder / f"{name}.json"
    _write_json(path, [dict(row) for row in rows])
    entry = {
        "name": name, "file": path.name, "sha256": _sha256_file(path), "n_rows": len(rows),
        "columns": sorted({key for row in rows for key in row}),
        "sealed_at": float(clock() if clock else time.time()),
    }
    manifest_path = folder / PREDICTIONS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[name] = entry
    _write_json(manifest_path, manifest)
    return entry


def predictions_manifest(run_root: str | os.PathLike) -> dict[str, Any]:
    path = predictions_dir(run_root) / PREDICTIONS_MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def join_labels(run_root: str | os.PathLike, name: str,
                labels: Mapping[str, Mapping[str, Any]], *, key: str = "decision_id",
                ) -> list[dict[str, Any]]:
    """Join ``labels`` onto the sealed predictions ``name``, or refuse.

    The refusal is the point: an unsealed prediction file, or one whose bytes changed after it
    was hashed, means the ordering guarantee was broken and the join must not happen.
    """
    manifest = predictions_manifest(run_root)
    entry = manifest.get(name)
    if entry is None:
        raise LabelJoinRefused(
            f"no sealed predictions named {name!r}: write and hash the held-out predictions "
            "before joining labels (plan §4.5)")
    path = predictions_dir(run_root) / entry["file"]
    if not path.exists():
        raise LabelJoinRefused(f"sealed prediction file {path.name} is missing")
    digest = _sha256_file(path)
    if digest != entry["sha256"]:
        raise LabelJoinRefused(
            f"predictions {name!r} changed after they were sealed "
            f"({digest[:12]} != {entry['sha256'][:12]}); the label join is refused")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) | dict(labels.get(str(row.get(key)), {})) for row in rows]


# --------------------------------------------------------------------------- context


def _read(folder: Path, name: str) -> tuple[dict[str, Any], ...]:
    try:
        return tuple(tables.read_table(folder, name))
    except MaterializeError:
        return ()


@dataclass(frozen=True)
class AnalysisContext:
    """Everything an estimator reads, resolved once (so the row set behind a number is visible)."""

    run_root: Path
    cfg: Any
    analysis: Any
    settings: stats.StatsSettings
    frame: panels.PanelFrame
    transitions: tuple[Mapping[str, Any], ...] = ()
    selections: tuple[Mapping[str, Any], ...] = ()
    selections_graded: tuple[Mapping[str, Any], ...] = ()
    grades: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    tool_events: tuple[Mapping[str, Any], ...] = ()
    receiver_grid: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    budget_by_hash: Mapping[str, int] = field(default_factory=dict)

    @property
    def derived(self) -> Path:
        return self.run_root / outcomes.DERIVED_DIR

    @property
    def endpoint_round(self) -> int:
        return int(self.cfg.analysis.primary_endpoint_round)

    @property
    def family_weights(self) -> dict[str, float]:
        """``analysis.profiles.*.families``: the prespecified task weights (equal by default)."""
        return {str(k): float(v) for k, v in self.analysis.profile.families.items()}

    @property
    def alpha(self) -> float:
        """``study.analysis.familywise_alpha``: the one declaration of the primary family's alpha."""
        return float(self.cfg.analysis.familywise_alpha)

    @property
    def posterior_draws(self) -> int:
        """``analysis.profiles.*.posterior_draws``: Beta-posterior draws of the difficulty
        estimate used by the standardisation sensitivity (spec 7.3; 200 confirmatory, 5 test)."""
        return int(self.analysis.profile.posterior_draws)

    @property
    def model_scope(self) -> str:
        return str(self.cfg.analysis.cluster)


def _enum_value(value: Any) -> Any:
    """The YAML value behind a ``protocols`` enum member, so the grid is comparable either way."""
    return value.value if hasattr(value, "value") else value


def receiver_grid(protocols_path: Path | None) -> dict[str, dict[str, Any]]:
    """``display_label -> {support, checker, draft_role, phases}`` for the receiver conditions.

    ``protocols.load_registry`` is the declaration (plan D7); it is used when WP2 has landed and
    the YAML is read directly until then, so the E2 factorial can be built and tested without
    waiting for the registry.
    """
    try:  # pragma: no cover - exercised once WP2 lands protocols.py
        from agents_scaling.nextdyn.protocols import load_registry
    except ImportError:
        pass
    else:
        registry = load_registry(protocols_path) if protocols_path else load_registry()
        return {str(spec.display_label): {
                    "support": _enum_value(spec.support), "checker": bool(spec.checker),
                    "draft_role": _enum_value(spec.draft_role), "phases": int(spec.phases),
                    "packet_content": _enum_value(spec.packet_content)}
                for spec in registry.receivers.values() if getattr(spec, "enabled", True)}
    if protocols_path is None or not Path(protocols_path).exists():
        return {}
    import yaml

    data = yaml.safe_load(Path(protocols_path).read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for entry in (data.get("receivers") or {}).values():
        if not isinstance(entry, Mapping) or not entry.get("enabled"):
            continue
        out[str(entry.get("display_label"))] = {
            "support": entry.get("support"), "checker": bool(entry.get("checker")),
            "draft_role": entry.get("draft_role"), "phases": int(entry.get("phases") or 1),
            "packet_content": entry.get("packet_content"),
        }
    return out


def load_context(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
                 profile: str | None = None) -> AnalysisContext:
    """Compile the configs and read the published and derived tables once."""
    root = Path(run_root)
    cfg_path = Path(config_path) if config_path is not None else C.DEFAULT_STUDY_CONFIG
    cfg = C.load_study(cfg_path)
    name = profile or cfg.analysis.profile
    analysis_path = C.sibling_config(cfg_path, C.DEFAULT_ANALYSIS_CONFIG.name) or C.DEFAULT_ANALYSIS_CONFIG
    analysis = C.load_analysis(analysis_path, profile=name)
    outcomes.assert_state_alphabet(cfg.analysis)
    pub, der = root / outcomes.TABLES_DIR, root / outcomes.DERIVED_DIR
    prot = root / outcomes.PROTECTED_TABLES_DIR
    grades: dict[str, dict[str, Any]] = {}
    for row in _read(prot, "grades_private"):
        grades.setdefault(str(row["generation_uid"]), row)
    contract = cfg.contract
    budget_by_hash = {
        contract.receiver_tokens[f"{b}/one_call"].budget_hash(): int(b)
        for b in cfg.budgets.receiver_thinking_tokens
    }
    return AnalysisContext(
        run_root=root, cfg=cfg, analysis=analysis,
        settings=stats.settings(analysis.profile, cfg.analysis),
        frame=panels.PanelFrame(
            team_rounds=_read(der, "team_rounds"), agent_rounds=_read(der, "agent_rounds"),
            item_splits=_read(pub, "item_splits"), banks=_read(pub, "banks"),
            root_teams=_read(pub, "root_teams")),
        transitions=_read(der, "transitions"),
        selections=_read(pub, "selections"),
        selections_graded=_read(der, "selections_graded"),
        grades=grades,
        tool_events=_read(pub, "tool_events"),
        receiver_grid=receiver_grid(C.sibling_config(cfg_path, C.DEFAULT_PROTOCOLS_CONFIG.name)),
        budget_by_hash=budget_by_hash,
    )


# --------------------------------------------------------------------------- outcomes


@dataclass(frozen=True)
class Outcome:
    """One endpoint, as a pair of readings of a ``team_rounds`` row (spec 4.6)."""

    name: str
    description: str
    lo: Callable[[Mapping[str, Any]], float | None]
    hi: Callable[[Mapping[str, Any]], float | None]
    complete: Callable[[Mapping[str, Any]], bool]


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _focal_lo(row: Mapping[str, Any]) -> float | None:
    state = row.get("focal_state")
    return None if state is None else float(state == outcomes.STATE_CORRECT)


def _focal_hi(row: Mapping[str, Any]) -> float | None:
    state = row.get("focal_state")
    return None if state is None else float(state in (outcomes.STATE_CORRECT, outcomes.STATE_UNRESOLVED))


def _adoption_hi(row: Mapping[str, Any]) -> float | None:
    """Upper bound on wrong-minority adoption: every unknown non-focal slot could have adopted."""
    base = row.get("nonfocal_init_correct_now_wstar")
    n = int(row.get("n_nonfocal_init_correct") or 0)
    if base is None or n == 0:
        return None
    return float(min(1.0, float(base) + (int(row.get("nonfocal_unknown") or 0) / n)))


def _propagation_hi(row: Mapping[str, Any]) -> float | None:
    base = row.get("nonfocal_init_wrong_now_c")
    n = int(row.get("n_nonfocal_init_wrong") or 0)
    if base is None or n == 0:
        return None
    return float(min(1.0, float(base) + (int(row.get("nonfocal_unknown") or 0) / n)))


OUTCOMES: Final[Mapping[str, Outcome]] = {
    "coverage": Outcome(
        "coverage", "Q_r: a correct answer is present among the current candidates",
        lambda r: _float(r["q_lo"]), lambda r: _float(r["q_hi"]),
        lambda r: bool(r["fully_graded"])),
    "margin": Outcome(
        "margin", "M_r: the fraction of slots holding a correct answer",
        lambda r: _float(r["m_lo"]), lambda r: _float(r["m_hi"]),
        lambda r: bool(r["fully_graded"])),
    "historical_coverage": Outcome(
        "historical_coverage", "H_r: a correct answer was present at some round u <= r",
        lambda r: _float(r["h_lo"]), lambda r: _float(r["h_hi"]),
        lambda r: bool(r["fully_graded"])),
    "focal_correct": Outcome(
        "focal_correct", "1[focal slot holds a correct answer]", _focal_lo, _focal_hi,
        lambda r: r.get("focal_state") != outcomes.STATE_UNRESOLVED),
    "wrong_adoption": Outcome(
        "wrong_adoption", "share of initially correct non-focal slots now holding w*",
        lambda r: _float(r.get("nonfocal_init_correct_now_wstar")), _adoption_hi,
        lambda r: bool(r["fully_graded"])),
    "correct_propagation": Outcome(
        "correct_propagation", "share of initially wrong non-focal slots now holding a correct class",
        lambda r: _float(r.get("nonfocal_init_wrong_now_c")), _propagation_hi,
        lambda r: bool(r["fully_graded"])),
    "delivered_accuracy": Outcome(
        "delivered_accuracy", "A_r: the selected answer is correct",
        lambda r: _float(r.get("a_lo")), lambda r: _float(r.get("a_lo")),
        lambda r: r.get("a_lo") is not None and bool(r["fully_graded"])),
    "continuous_preservation": Outcome(
        "continuous_preservation", "a correct answer was present at every round up to r",
        lambda r: (None if r.get("continuous_preservation_lo") is None
                   else float(bool(r["continuous_preservation_lo"]))),
        lambda r: (None if r.get("continuous_preservation_lo") is None
                   else float(bool(r["continuous_preservation_lo"]) or not r["fully_graded"])),
        lambda r: bool(r["fully_graded"])),
}


# --------------------------------------------------------------------------- endpoint matrix


@dataclass(frozen=True)
class ItemMatrix:
    """One row per source item, one column per arm; replicates already averaged (spec 14.2)."""

    outcome: str
    round: int
    arms: tuple[str, ...]
    families: Mapping[str, str]
    lo: Mapping[str, Mapping[str, float]]
    hi: Mapping[str, Mapping[str, float]]
    complete: frozenset[str]
    n_rows: int
    n_missing: int
    population: str

    def paired_items(self) -> list[str]:
        """Items carrying **every** arm: an unbalanced item would change the estimand."""
        return sorted(item for item, by_arm in self.lo.items() if set(self.arms) <= set(by_arm))

    def contrast(self, weights: Mapping[str, float], *, bound: str = "lo",
                 items: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """``family -> array of d_i`` for the linear arm contrast ``weights``."""
        table = self.lo if bound == "lo" else self.hi
        chosen = list(items) if items is not None else self.paired_items()
        out: dict[str, list[float]] = {}
        for item in chosen:
            by_arm = table.get(item, {})
            if not set(weights) <= set(by_arm):
                continue
            out.setdefault(self.families[item], []).append(
                math.fsum(float(w) * float(by_arm[arm]) for arm, w in weights.items()))
        return {family: np.asarray(values, dtype=float) for family, values in out.items()}

    def census(self) -> dict[str, Any]:
        return {"population": self.population, "outcome": self.outcome, "round": self.round,
                "arms": list(self.arms), "n_items": len(self.paired_items()),
                "n_rows": self.n_rows, "n_missing": self.n_missing,
                "n_complete_case": len(self.complete & set(self.paired_items()))}


def endpoint_matrix(panel: panels.Panel, outcome: str, round_: int, *, arms: Sequence[str],
                    rows: Sequence[Mapping[str, Any]],
                    composition: str | None = None) -> ItemMatrix:
    """Build the item x arm matrix for one endpoint (plan §4.2).

    Replicates are averaged **within** the arm before anything is differenced, which is the
    spec-14.2 rule; differencing per-replicate and then averaging would pair different
    episodes.  Both the ``lo`` and the ``hi`` reading are built in the same pass, so a bounds
    analysis can never be run on a different row set than the operational one.
    """
    spec = OUTCOMES.get(outcome)
    if spec is None:
        raise ContractViolation(f"unknown outcome {outcome!r}; declared: {sorted(OUTCOMES)}")
    wanted = tuple(arms)
    acc_lo: dict[str, dict[str, list[float]]] = {}
    acc_hi: dict[str, dict[str, list[float]]] = {}
    complete_flags: dict[str, bool] = {}
    families: dict[str, str] = {}
    n_rows = n_missing = 0
    for row in rows:
        if int(row["round"]) != int(round_) or not panel.row_filter(row):
            continue
        label = str(row.get("display_label") or "")
        if label not in wanted:
            continue
        if composition is not None and str(row.get("composition") or "") != composition:
            continue
        n_rows += 1
        item = str(row["source_item_id"])
        value_lo, value_hi = spec.lo(row), spec.hi(row)
        if value_lo is None or value_hi is None:
            n_missing += 1
            continue
        families[item] = str(row.get("family") or "")
        acc_lo.setdefault(item, {}).setdefault(label, []).append(float(value_lo))
        acc_hi.setdefault(item, {}).setdefault(label, []).append(float(value_hi))
        complete_flags[item] = complete_flags.get(item, True) and bool(spec.complete(row))
    return ItemMatrix(
        outcome=outcome, round=int(round_), arms=wanted, families=families,
        lo={i: {a: sum(v) / len(v) for a, v in by.items()} for i, by in acc_lo.items()},
        hi={i: {a: sum(v) / len(v) for a, v in by.items()} for i, by in acc_hi.items()},
        complete=frozenset(i for i, ok in complete_flags.items() if ok),
        n_rows=n_rows, n_missing=n_missing, population=panel.name,
    )


# --------------------------------------------------------------------------- hypotheses


@dataclass(frozen=True)
class HypothesisSpec:
    """One primary hypothesis, as data (plan §4.2)."""

    h_id: str
    statement: str
    experiment: str
    population: str
    outcome: str
    arm_weights: Mapping[str, float]
    composition: str | None = None
    kind: str = "paired_contrast"


C1: Final[str] = "C1_COHERENT"
C4: Final[str] = "C4_COHERENT"

HYPOTHESIS_SPECS: Final[Mapping[str, HypothesisSpec]] = {
    "H1": HypothesisSpec(
        "H1", "PEER raises active coverage over PRIVATE on constructed correct-minority teams",
        "E1", panels.CON_PAIRED, "coverage", {"PEER": 1.0, "PRIVATE": -1.0}, C1),
    "H2": HypothesisSpec(
        "H2", "the PEER-PRIVATE coverage effect is not homogeneous across tasks",
        "E1", panels.CON_PAIRED, "coverage", {"PEER": 1.0, "PRIVATE": -1.0}, C1, kind="joint"),
    "H3": HypothesisSpec(
        "H3", "PEER raises adoption of the shared wrong class on correct-majority teams",
        "E1", panels.CON_PAIRED, "wrong_adoption", {"PEER": 1.0, "PRIVATE": -1.0}, C4),
    "H4": HypothesisSpec(
        "H4", "the checker reduces the endorsement effect on the synthetic receiver",
        "E2b", panels.E2B, "receiver_correct", {}, None, kind="e2b_checker"),
    "H5": HypothesisSpec(
        "H5", "a larger thinking budget reduces the endorsement effect on the synthetic receiver",
        "E2b", panels.E2B, "receiver_correct", {}, None, kind="e2b_budget"),
    "H6": HypothesisSpec(
        "H6", "cutting the focal agent's incoming channel preserves its correct answer",
        "E3a", panels.E3A, "focal_correct",
        {"IN0_OUT1": 0.5, "IN0_OUT0": 0.5, "IN1_OUT1": -0.5, "IN1_OUT0": -0.5}, C1),
    "H7": HypothesisSpec(
        "H7", "the focal agent's outgoing channel spreads its correct answer to the others",
        "E3a", panels.E3A, "correct_propagation",
        {"IN1_OUT1": 0.5, "IN0_OUT1": 0.5, "IN1_OUT0": -0.5, "IN0_OUT0": -0.5}, C1),
    "H8": HypothesisSpec(
        "H8", "isolating the focal agent from the wrong majority reduces wrong-class adoption",
        "E3a", panels.E3A, "wrong_adoption", {"IN0_OUT1": 1.0, "IN1_OUT1": -1.0}, C4),
}


@dataclass(frozen=True)
class HypothesisResult:
    """One primary hypothesis's three populations and its test of record."""

    h_id: str
    statement: str
    experiment: str
    population: str
    outcome: str
    arms: tuple[str, ...]
    estimate: float | None
    p: float | None
    ci_lo: float | None
    ci_hi: float | None
    bounds_lo: float | None
    bounds_hi: float | None
    complete_case: float | None
    n_items: int
    n_clusters: int
    by_family: Mapping[str, float]
    estimable: bool
    reason: str | None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def missingness_sensitivity(self) -> str:
        """Do the operational estimate, the bounds and the complete case agree in sign?"""
        if self.estimate is None:
            return "not_estimable"
        values = [v for v in (self.bounds_lo, self.bounds_hi, self.complete_case) if v is not None]
        if len(values) < 3:
            return "incomplete"
        signs = {int(math.copysign(1, v)) if v else 0 for v in (self.estimate, *values)}
        return "agrees" if len(signs) == 1 else "disagrees"

    def as_row(self) -> dict[str, Any]:
        return {
            "h_id": self.h_id, "statement": self.statement, "experiment": self.experiment,
            "population": self.population, "outcome": self.outcome, "arms": list(self.arms),
            "estimate": self.estimate, "p": self.p, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "bounds_lo": self.bounds_lo, "bounds_hi": self.bounds_hi,
            "complete_case": self.complete_case, "n_items": self.n_items,
            "n_clusters": self.n_clusters, "by_family": dict(self.by_family),
            "estimable": self.estimable, "reason": self.reason,
            "missingness_sensitivity": self.missingness_sensitivity, **dict(self.detail),
        }


def _not_estimable(spec: HypothesisSpec, reason: str, **detail: Any) -> HypothesisResult:
    return HypothesisResult(
        h_id=spec.h_id, statement=spec.statement, experiment=spec.experiment,
        population=spec.population, outcome=spec.outcome, arms=tuple(sorted(spec.arm_weights)),
        estimate=None, p=None, ci_lo=None, ci_hi=None, bounds_lo=None, bounds_hi=None,
        complete_case=None, n_items=0, n_clusters=0, by_family={}, estimable=False,
        reason=reason, detail=detail)


def _bounds(matrix: ItemMatrix, weights: Mapping[str, float], items: Sequence[str],
            ) -> tuple[float | None, float | None]:
    """``[mean(lo_T) - mean(hi_C), mean(hi_T) - mean(lo_C)]`` (spec 4.6)."""
    treat = {a: w for a, w in weights.items() if w > 0}
    control = {a: -w for a, w in weights.items() if w < 0}
    if not treat or not control:
        return None, None

    def side(table: Mapping[str, Mapping[str, float]], part: Mapping[str, float]) -> float | None:
        values = [math.fsum(w * float(table[i][a]) for a, w in part.items())
                  for i in items if set(part) <= set(table.get(i, {}))]
        return (sum(values) / len(values)) if values else None

    lo_t, hi_c = side(matrix.lo, treat), side(matrix.hi, control)
    hi_t, lo_c = side(matrix.hi, treat), side(matrix.lo, control)
    if None in (lo_t, hi_c, hi_t, lo_c):
        return None, None
    return lo_t - hi_c, hi_t - lo_c


def _weighted_mean(by_family: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> float | None:
    usable = {f: v for f, v in by_family.items() if v.size}
    if not usable:
        return None
    total = math.fsum(float(weights.get(f, 0.0)) for f in usable)
    if total <= 0:
        return None
    return math.fsum(float(weights.get(f, 0.0)) / total * float(v.mean()) for f, v in usable.items())


def _paired_hypothesis(spec: HypothesisSpec, ctx: AnalysisContext) -> HypothesisResult:
    panel = panels.population(spec.population, ctx.frame, ctx.cfg)
    matrix = endpoint_matrix(panel, spec.outcome, ctx.endpoint_round,
                             arms=tuple(spec.arm_weights), rows=ctx.frame.team_rounds,
                             composition=spec.composition)
    items = matrix.paired_items()
    if not items:
        return _not_estimable(spec, "no item carries every arm of this contrast",
                              census=matrix.census())
    diff = matrix.contrast(spec.arm_weights, items=items)
    weights = ctx.family_weights
    seed = stats.analysis_seed(ctx.cfg.seeds.study_seed_hex, spec.h_id)
    test = stats.wild_cluster_test(
        diff, weights=weights, n_resamples=ctx.settings.n_resamples, seed=seed,
        webb_below_clusters=ctx.settings.webb_below_clusters,
        min_clusters_for_test=ctx.settings.min_clusters_for_test, label=spec.h_id)
    if test.get("estimate") is None:
        return _not_estimable(spec, str(test.get("reason")), census=matrix.census())
    families = test["families"]
    try:
        interval = stats.paired_contrast(
            {f: diff[f] for f in families}, stats.Resampler(
                {f: int(diff[f].size) for f in families}, ctx.settings.n_resamples, seed,
                domains=families),
            domains=families, weights={f: weights.get(f, 1.0) for f in families},
            n_primary=ctx.settings.n_primary, alpha=ctx.settings.alpha)
        ci = interval["ci95_studentized"]
    except Exception as exc:  # the interval engine is loud; keep the test of record either way
        interval, ci = {"error": str(exc)}, [None, None]
    bounds_lo, bounds_hi = _bounds(matrix, spec.arm_weights, items)
    complete_items = sorted(matrix.complete & set(items))
    complete = _weighted_mean(matrix.contrast(spec.arm_weights, items=complete_items), weights) \
        if complete_items else None
    return HypothesisResult(
        h_id=spec.h_id, statement=spec.statement, experiment=spec.experiment,
        population=spec.population, outcome=spec.outcome, arms=tuple(spec.arm_weights),
        estimate=float(test["estimate"]), p=float(test["p"]),
        ci_lo=None if ci[0] is None else float(ci[0]),
        ci_hi=None if ci[1] is None else float(ci[1]),
        bounds_lo=bounds_lo, bounds_hi=bounds_hi, complete_case=complete,
        n_items=len(items), n_clusters=int(test["n_clusters"]),
        by_family={f: float(v) for f, v in test["by_family"].items()}, estimable=True,
        reason=None,
        detail={"test": test, "interval": interval, "census": matrix.census(),
                "n_complete_case": len(complete_items),
                "estimation_only": test.get("verdict") == stats.ESTIMATION_ONLY},
    )


def _joint_hypothesis(spec: HypothesisSpec, ctx: AnalysisContext) -> HypothesisResult:
    base = HYPOTHESIS_SPECS["H1"]
    panel = panels.population(base.population, ctx.frame, ctx.cfg)
    matrix = endpoint_matrix(panel, base.outcome, ctx.endpoint_round,
                             arms=tuple(base.arm_weights), rows=ctx.frame.team_rounds,
                             composition=base.composition)
    items = matrix.paired_items()
    diff = matrix.contrast(base.arm_weights, items=items)
    seed = stats.analysis_seed(ctx.cfg.seeds.study_seed_hex, spec.h_id)
    result = stats.joint_contrast(diff, n_resamples=ctx.settings.n_resamples, seed=seed,
                                  min_items_per_family=ctx.settings.min_clusters_for_test,
                                  label=spec.h_id)
    if result.get("verdict") == stats.NOT_ESTIMABLE:
        return _not_estimable(spec, str(result.get("reason")), joint=result,
                              census=matrix.census())
    spread = max(result["by_family"].values()) - min(result["by_family"].values())
    return HypothesisResult(
        h_id=spec.h_id, statement=spec.statement, experiment=spec.experiment,
        population=spec.population, outcome=spec.outcome, arms=tuple(base.arm_weights),
        estimate=float(spread), p=float(result["p"]), ci_lo=None, ci_hi=None,
        bounds_lo=None, bounds_hi=None, complete_case=None,
        n_items=len(items), n_clusters=int(sum(result["n"].values())),
        by_family={f: float(v) for f, v in result["by_family"].items()}, estimable=True,
        reason=None,
        detail={"joint": result, "census": matrix.census(),
                "estimate_is": "max - min of the per-task effects (the joint test is on the "
                               "contrast matrix, not on this scalar)"},
    )


# ---- E2b factorial --------------------------------------------------------------------


def _receiver_cells(ctx: AnalysisContext) -> dict[str, dict[tuple[Any, ...], list[float]]]:
    """``problem -> (support, checker, budget) -> [1[state == C]]`` over drafts and replicates."""
    out: dict[str, dict[tuple[Any, ...], list[float]]] = {}
    for row in ctx.frame.agent_rounds:
        if str(row.get("cell_kind")) != CellKind.RECEIVER.value:
            continue
        condition = ctx.receiver_grid.get(str(row.get("display_label") or ""))
        if condition is None or int(condition.get("phases") or 1) != 2:
            continue
        budget = ctx.budget_by_hash.get(str(row.get("budget_hash") or ""))
        if budget is None:
            continue
        key = (str(condition["support"]), bool(condition["checker"]), int(budget))
        out.setdefault(str(row["source_item_id"]), {}).setdefault(key, []).append(
            float(str(row["state"]) == outcomes.STATE_CORRECT))
    return out


def _endorsement_effect(cells: Mapping[tuple[Any, ...], Sequence[float]],
                        checker: bool, budget: int) -> float | None:
    """``E_p(v, b) = Y_p(4:1) - Y_p(1:4)``: the correct candidate's support cue flipped."""
    majority = cells.get(("4:1", checker, budget))
    minority = cells.get(("1:4", checker, budget))
    if not majority or not minority:
        return None
    return float(np.mean(majority) - np.mean(minority))


def _e2b_hypothesis(spec: HypothesisSpec, ctx: AnalysisContext) -> HypothesisResult:
    budgets = sorted(set(ctx.budget_by_hash.values()))
    if len(budgets) < 2:
        return _not_estimable(spec, f"E2b needs two receiver budgets, have {budgets}")
    low, high = budgets[0], budgets[-1]
    cells = _receiver_cells(ctx)
    per_problem: dict[str, float] = {}
    for problem, by_cell in cells.items():
        effects = {(v, b): _endorsement_effect(by_cell, v, b)
                   for v in (False, True) for b in (low, high)}
        if any(value is None for value in effects.values()):
            continue
        if spec.kind == "e2b_checker":
            value = 0.5 * (effects[(True, low)] + effects[(True, high)]) \
                - 0.5 * (effects[(False, low)] + effects[(False, high)])
        else:
            value = 0.5 * (effects[(False, high)] + effects[(True, high)]) \
                - 0.5 * (effects[(False, low)] + effects[(True, low)])
        per_problem[problem] = float(value)
    if not per_problem:
        return _not_estimable(spec, "no problem carries the full 2x2 checker x budget cell set",
                              n_problems_seen=len(cells))
    diff = {panels.E2B: np.asarray([per_problem[p] for p in sorted(per_problem)], dtype=float)}
    seed = stats.analysis_seed(ctx.cfg.seeds.study_seed_hex, spec.h_id)
    test = stats.wild_cluster_test(
        diff, weights={panels.E2B: 1.0}, n_resamples=ctx.settings.n_resamples, seed=seed,
        webb_below_clusters=ctx.settings.webb_below_clusters,
        min_clusters_for_test=ctx.settings.min_clusters_for_test, label=spec.h_id)
    if test.get("estimate") is None:
        return _not_estimable(spec, str(test.get("reason")))
    estimate = float(test["estimate"])
    return HypothesisResult(
        h_id=spec.h_id, statement=spec.statement, experiment=spec.experiment,
        population=spec.population, outcome=spec.outcome,
        arms=("4:1", "1:4"), estimate=estimate, p=float(test["p"]),
        ci_lo=None, ci_hi=None,
        # The receiver endpoint is a single graded decision, so the missingness bounds are the
        # same statistic with every U credited; they are carried by the cell census below.
        bounds_lo=estimate, bounds_hi=estimate, complete_case=estimate,
        n_items=len(per_problem), n_clusters=int(test["n_clusters"]),
        by_family={panels.E2B: estimate}, estimable=True, reason=None,
        detail={"test": test, "budgets": [low, high], "cluster": "problem",
                "n_problems_seen": len(cells), "per_problem": dict(sorted(per_problem.items()))},
    )


HYPOTHESIS_BUILDERS: Final[Mapping[str, Callable[[HypothesisSpec, AnalysisContext], HypothesisResult]]] = {
    "paired_contrast": _paired_hypothesis,
    "joint": _joint_hypothesis,
    "e2b_checker": _e2b_hypothesis,
    "e2b_budget": _e2b_hypothesis,
}


def hypothesis(h_id: str, ctx: AnalysisContext) -> HypothesisResult:
    """One primary hypothesis exactly as plan §4.2 defines it."""
    spec = HYPOTHESIS_SPECS.get(h_id)
    if spec is None:
        raise ContractViolation(f"unknown hypothesis {h_id!r}; declared: {sorted(HYPOTHESIS_SPECS)}")
    return HYPOTHESIS_BUILDERS[spec.kind](spec, ctx)


def primary_family(ctx: AnalysisContext) -> dict[str, Any]:
    """Every declared primary, Holm-corrected over exactly ``n_primary`` slots (spec 14.1)."""
    declared = tuple(ctx.cfg.analysis.primary_hypotheses)
    if str(ctx.cfg.analysis.multiplicity) != "holm":
        raise ContractViolation(
            f"study.analysis.multiplicity is {ctx.cfg.analysis.multiplicity!r}; only Holm is "
            "implemented for the primary family (spec 14.1)")
    if not bool(ctx.cfg.analysis.paired):
        raise ContractViolation("study.analysis.paired must be true: every primary is a paired "
                                "within-item contrast (spec 14.2)")
    if str(ctx.cfg.analysis.task_summary_weights) != "equal":
        raise ContractViolation(
            f"study.analysis.task_summary_weights {ctx.cfg.analysis.task_summary_weights!r} is "
            "not implemented; the prespecified summary is the equal-task macro average")
    results = {h_id: hypothesis(h_id, ctx) for h_id in declared}
    holm = stats.holm_with_not_estimable(
        {h_id: r.p for h_id, r in results.items()},
        alpha=ctx.settings.alpha, n_primary=ctx.settings.n_primary)
    rows = []
    for h_id in declared:
        result = results[h_id]
        if bool(ctx.cfg.analysis.missingness_bounds_required) and result.estimable:
            if result.bounds_lo is None or result.complete_case is None:
                raise ContractViolation(
                    f"{h_id}: study.analysis.missingness_bounds_required is set, so every primary "
                    "must report bounds and a complete-case sensitivity (spec 4.6)")
        rows.append(result.as_row() | {"holm": holm[h_id],
                                       "verdict": _verdict_of(result, holm[h_id])})
    return {"hypotheses": rows, "holm": holm, "alpha": ctx.settings.alpha,
            "n_primary": ctx.settings.n_primary, "endpoint_round": ctx.endpoint_round,
            "family_weights": ctx.family_weights}


def _verdict_of(result: HypothesisResult, holm_row: Mapping[str, Any]) -> str:
    if not result.estimable:
        return VERDICT_NOT_ESTIMABLE
    if not holm_row["reject"]:
        return VERDICT_INCONCLUSIVE
    return VERDICT_SUPPORTED if result.missingness_sensitivity == "agrees" else VERDICT_NARROWED


# --------------------------------------------------------------------------- factorials


def e2_factorial(ctx: AnalysisContext) -> list[dict[str, Any]]:
    """The full E2 cell means, both assays, before any reduction (spec 9.6)."""
    rows: list[dict[str, Any]] = []
    for row in ctx.frame.agent_rounds:
        if str(row.get("cell_kind")) != CellKind.RECEIVER.value:
            continue
        condition = ctx.receiver_grid.get(str(row.get("display_label") or ""))
        if condition is None:
            continue
        rows.append({
            "assay": "E2b" if int(condition.get("phases") or 1) == 2 else "E2a",
            "problem": str(row["source_item_id"]),
            "support": condition["support"], "checker": bool(condition["checker"]),
            "draft_role": condition["draft_role"],
            "packet_content": condition.get("packet_content"),
            "budget": ctx.budget_by_hash.get(str(row.get("budget_hash") or "")),
            "model_hash": str(row.get("model_hash") or ""),
            "replicate_id": int(row.get("replicate_id") or 0),
            "correct": float(str(row["state"]) == outcomes.STATE_CORRECT),
            "unresolved": float(str(row["state"]) == outcomes.STATE_UNRESOLVED),
        })
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["assay"], row["support"], row["checker"], row["draft_role"],
               row["budget"], row["model_hash"])
        cells.setdefault(key, []).append(row)
    out = []
    for key, members in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        assay, support, checker, draft, budget, model = key
        out.append({
            "assay": assay, "support": support, "checker": checker, "draft_role": draft,
            "budget": budget, "model_hash": model, "n": len(members),
            "n_problems": len({m["problem"] for m in members}),
            "accuracy_lo": float(np.mean([m["correct"] for m in members])),
            "accuracy_hi": float(np.mean([m["correct"] + m["unresolved"] for m in members])),
            "unresolved_rate": float(np.mean([m["unresolved"] for m in members])),
        })
    return out


def e3_factorial(ctx: AnalysisContext) -> list[dict[str, Any]]:
    """The four directed cells, within-item first, then item resample (plan §4.2)."""
    arms = tuple(ctx.cfg.experiments.E3a.arms) + tuple(ctx.cfg.experiments.E3a.filled_arms)
    panel = panels.population(panels.E3A, ctx.frame, ctx.cfg)
    out = []
    for outcome_name in ("focal_correct", "correct_propagation", "wrong_adoption", "coverage"):
        for composition in (C1, C4):
            matrix = endpoint_matrix(panel, outcome_name, ctx.endpoint_round, arms=arms,
                                     rows=ctx.frame.team_rounds, composition=composition)
            for arm in arms:
                values = [by[arm] for by in matrix.lo.values() if arm in by]
                if not values:
                    continue
                out.append({"outcome": outcome_name, "composition": composition, "arm": arm,
                            "n_items": len(values), "mean_lo": float(np.mean(values)),
                            "mean_hi": float(np.mean([by[arm] for by in matrix.hi.values()
                                                      if arm in by]))})
    return out


# --------------------------------------------------------------------------- standardization


DIFFICULTY_BIN_UNKNOWN: Final[int] = -1


def _stratum(row: Mapping[str, Any], difficulty: Mapping[tuple[str, str, str], int]) -> tuple[Any, ...]:
    """``z = (C_0 bin, largest wrong class size, difficulty bin, format)`` (spec 7.3)."""
    c0 = row.get("c0")
    wrong_max = row.get("wrong_class_max_share")
    w_count = int(row.get("w_count") or 0)
    largest_wrong = 0 if wrong_max is None else min(3, int(round(float(wrong_max) * w_count)))
    key = (str(row["source_item_id"]), str(row.get("model_hash") or ""), str(row.get("budget_hash") or ""))
    return (int(c0) if c0 is not None else int(row.get("c_count") or 0),
            largest_wrong,
            int(difficulty.get(key, DIFFICULTY_BIN_UNKNOWN)),
            str(row.get("family") or ""))


def standardization(ctx: AnalysisContext, *, treat: str = "PEER", control: str = "PRIVATE",
                    outcome: str = "coverage", population: str = panels.NAT,
                    min_per_arm: int = 5) -> list[dict[str, Any]]:
    """``Delta_std`` over the pre-outcome recruitment frame's strata (spec 7.3, 14.4).

    ``w(z)`` is the frequency of the stratum in the **recruitment frame** -- natural roots and
    the difficulty bank, both fixed before any continuation -- so the weights cannot have been
    chosen by the outcome.  Only strata with at least ``min_per_arm`` items in both arms are
    supported; ``overlap_retained`` and the effective sample size say how much of the frame that
    leaves, which is the number a reader needs to know whether the standardised estimate is the
    same question as the crude one.
    """
    difficulty = {}
    try:
        for row in tables.read_table(ctx.run_root / outcomes.DERIVED_DIR, "difficulty"):
            difficulty[(str(row["source_item_id"]), str(row["model_hash"]), str(row["budget_hash"]))] = \
                int(row.get("bin") or DIFFICULTY_BIN_UNKNOWN)
    except MaterializeError:
        pass
    panel = panels.population(population, ctx.frame, ctx.cfg)
    spec = OUTCOMES[outcome]
    by_stratum: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    frame_counts: dict[tuple[Any, ...], set[str]] = {}
    for row in ctx.frame.team_rounds:
        if not panel.row_filter(row):
            continue
        z = _stratum(row, difficulty)
        if int(row["round"]) == 0:
            frame_counts.setdefault(z, set()).add(str(row["source_item_id"]))
        if int(row["round"]) != ctx.endpoint_round:
            continue
        label = str(row.get("display_label") or "")
        if label not in (treat, control):
            continue
        value = spec.lo(row)
        if value is None:
            continue
        by_stratum.setdefault(z, {}).setdefault(label, []).append(float(value))
    total_frame = sum(len(items) for items in frame_counts.values()) or 1
    rows: list[dict[str, Any]] = []
    supported_weight = 0.0
    weights_used: list[float] = []
    delta = 0.0
    for z, arms in sorted(by_stratum.items(), key=lambda kv: [str(x) for x in kv[0]]):
        weight = len(frame_counts.get(z, ())) / total_frame
        n_t, n_c = len(arms.get(treat, [])), len(arms.get(control, []))
        supported = n_t >= min_per_arm and n_c >= min_per_arm and weight > 0
        contribution = (float(np.mean(arms[treat]) - np.mean(arms[control])) if supported else None)
        if supported:
            supported_weight += weight
            weights_used.append(weight)
            delta += weight * contribution
        rows.append({"stratum": list(z), "weight": weight, "n_treat": n_t, "n_control": n_c,
                     "supported": supported, "contribution": contribution})
    ess = (sum(weights_used) ** 2 / sum(w * w for w in weights_used)) if weights_used else 0.0
    sensitivity = _difficulty_posterior_sensitivity(ctx, rows, delta, supported_weight)
    summary = {
        "stratum": ["__summary__"], "weight": supported_weight,
        "n_treat": sum(r["n_treat"] for r in rows), "n_control": sum(r["n_control"] for r in rows),
        "supported": bool(weights_used),
        "contribution": (delta / supported_weight) if supported_weight else None,
        "delta_std": (delta / supported_weight) if supported_weight else None,
        "overlap_retained": supported_weight, "ess": ess, "population": population,
        "outcome": outcome, "treat": treat, "control": control,
        "n_strata": len(rows), "n_strata_supported": len(weights_used),
        **sensitivity,
    }
    return rows + [summary]


def _difficulty_posterior_sensitivity(ctx: AnalysisContext, rows: Sequence[Mapping[str, Any]],
                                      delta: float, supported_weight: float) -> dict[str, Any]:
    """Re-weight the supported strata over Beta-posterior draws of the stratum frequencies.

    The stratum weights are counts from a finite recruitment frame, so they carry sampling
    error of their own; ``posterior_draws`` re-draws them from a Jeffreys posterior and reports
    the spread, which is the honest companion to a single standardised point estimate.
    """
    draws = ctx.posterior_draws
    supported = [r for r in rows if r["supported"]]
    if draws < 1 or not supported or supported_weight <= 0:
        return {"posterior_draws": draws, "delta_std_posterior_lo": None,
                "delta_std_posterior_hi": None}
    rng = np.random.default_rng(stats.analysis_seed(ctx.cfg.seeds.study_seed_hex, "standardization"))
    alphas = np.asarray([float(r["weight"]) for r in supported]) + 0.5
    contributions = np.asarray([float(r["contribution"]) for r in supported])
    sampled = rng.dirichlet(alphas, size=int(draws)) @ contributions
    return {"posterior_draws": draws,
            "delta_std_posterior_lo": float(np.quantile(sampled, 0.025)),
            "delta_std_posterior_hi": float(np.quantile(sampled, 0.975)),
            "delta_std_point": float(delta / supported_weight)}


# --------------------------------------------------------------------------- panels


def calibration_panel(ctx: AnalysisContext) -> list[dict[str, Any]]:
    """One row per (experiment, family, protocol, model, round, signal, population) (spec 13.1)."""
    edges = calib.dev_bin_edges(ctx.analysis.profile)
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in ctx.frame.agent_rounds:
        key = (str(row.get("experiment") or ""), str(row.get("family") or ""),
               str(row.get("display_label") or ""), str(row.get("model_hash") or ""),
               int(row["round"]))
        state = str(row["state"])
        cells.setdefault(key, []).append({
            "item_id": str(row["source_item_id"]),
            "confidence": row.get("confidence"),
            "answer_logprob_mean": row.get("answer_logprob_mean"),
            "correct": (None if state == outcomes.STATE_UNRESOLVED
                        else float(state == outcomes.STATE_CORRECT)),
        })
    out: list[dict[str, Any]] = []
    for key, rows in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        experiment, family, protocol, model, round_ = key
        for signal in ("confidence", "answer_logprob_mean"):
            summary = calib.cell_calibration(rows, signal=signal, reliability_bins=edges)
            auc = calib.within_item_auc([
                {"item_id": r["item_id"], "conf": r[signal], "correct": r["correct"]}
                for r in rows if r[signal] is not None and r["correct"] is not None])
            out.append({"experiment": experiment, "family": family, "protocol": protocol,
                        "model_hash": model, "round": round_, "population": "agent_rounds",
                        "within_item_auc": auc["within_item_auc"],
                        "n_mixed_items": auc["eligible_items"], "n_items": auc["n_items"],
                        **summary})
    return out


def passk_panel(ctx: AnalysisContext) -> list[dict[str, Any]]:
    """pass@k on the fixed PASSK panel, plus vote@k / oracle@k on one blind-ordered prefix.

    The prefix is the same actual candidate list for every selector, so ``oracle@k`` and
    ``vote@k`` are two readings of one set of draws and the gap between them is a property of
    selection rather than of sampling (spec 18.3
    ``test_passk_prefix_oracle_and_vote_use_same_actual_candidates``).
    """
    from agents_scaling.ard.reference_metrics import pass_at_k

    panel = panels.passk_population(ctx.frame, ctx.cfg)
    ks = panels.passk_ks(ctx.analysis.profile)
    draws: dict[str, list[Mapping[str, Any]]] = {}
    for row in ctx.frame.agent_rounds:
        item = str(row["source_item_id"])
        if item in panel.keys and str(row.get("display_label") or "") == panels.PASSK_FRESH_LABEL:
            draws.setdefault(item, []).append(row)
    for row in ctx.frame.banks:
        item = str(row["source_item_id"])
        if item in panel.keys and str(row.get("bank_role")) in panels.PASSK_BANK_ROLES:
            draws.setdefault(item, []).append(row)
    out: list[dict[str, Any]] = []
    for item in sorted(panel.keys):
        members = sorted(draws.get(item, []), key=lambda r: str(r.get("generation_uid") or ""))
        n = len(members)
        if n != panels.PASSK_DRAWS:
            continue
        states = [_state_of(row, ctx) for row in members]
        c = sum(1 for s in states if s == outcomes.STATE_CORRECT)
        u = sum(1 for s in states if s == outcomes.STATE_UNRESOLVED)
        classes = [_class_of(row, ctx) for row in members]
        for k in ks:
            prefix = list(range(min(k, n)))
            oracle = float(any(states[i] == outcomes.STATE_CORRECT for i in prefix))
            vote = _vote_correct([classes[i] for i in prefix], [states[i] for i in prefix])
            out.append({
                "source_item_id": item, "k": int(k), "n": n, "c": c, "u": u,
                "pass_at_k_lo": float(pass_at_k(n, c, int(k))),
                "pass_at_k_hi": float(pass_at_k(n, min(n, c + u), int(k))),
                "oracle_at_k": oracle, "vote_at_k": vote,
                "selection_gap_lo": None if vote is None else float(oracle - vote),
                "complete_case": bool(u == 0), "prefix_size": len(prefix),
            })
    return out


def _state_of(row: Mapping[str, Any], ctx: AnalysisContext) -> str:
    if "state" in row:
        return str(row["state"])
    grade = ctx.grades.get(str(row.get("generation_uid") or ""))
    return outcomes.STATE_UNRESOLVED if grade is None else str(grade["semantic_correct"])


def _class_of(row: Mapping[str, Any], ctx: AnalysisContext) -> str | None:
    if "cls" in row:
        return None if row["cls"] is None else str(row["cls"])
    grade = ctx.grades.get(str(row.get("generation_uid") or ""))
    return None if grade is None else (None if grade.get("canonical_class") is None
                                       else str(grade["canonical_class"]))


def _vote_correct(classes: Sequence[str | None], states: Sequence[str]) -> float | None:
    """Expected correctness of a uniform tie break over the winning valid classes."""
    from collections import Counter

    counts = Counter(c for c in classes if c is not None)
    if not counts:
        return 0.0
    correct_classes = {c for c, s in zip(classes, states)
                       if c is not None and s == outcomes.STATE_CORRECT}
    unresolved = {c for c, s in zip(classes, states)
                  if c is not None and s == outcomes.STATE_UNRESOLVED}
    top = max(counts.values())
    winners = [c for c, n in counts.items() if n == top]
    if any(c in unresolved and c not in correct_classes for c in winners):
        return None
    return sum(1.0 for c in winners if c in correct_classes) / len(winners)


def tool_usage_panel(ctx: AnalysisContext) -> list[tables.ToolUsagePanelRow]:
    """Per-E2b-episode ``check_step`` descriptives (spec 9.6; post-treatment, never an endpoint)."""
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for event in ctx.tool_events:
        by_episode.setdefault(str(event["episode_id"]), []).append(event)
    agents = {str(r["episode_id"]): r for r in ctx.frame.agent_rounds
              if str(r.get("cell_kind")) == CellKind.RECEIVER.value}
    out: list[tables.ToolUsagePanelRow] = []
    for episode_id in sorted(set(by_episode) | set(agents)):
        events = by_episode.get(episode_id, [])
        agent = agents.get(episode_id)
        if agent is None:
            continue
        steps: dict[str, int] = {}
        for event in events:
            steps[str(event.get("step_id"))] = steps.get(str(event.get("step_id")), 0) + 1
        invalid = sum(1 for e in events if str(e.get("status")) == "invalid")
        out.append(tables.ToolUsagePanelRow(
            cell_kind=str(agent["cell_kind"]), episode_id=episode_id,
            source_item_id=str(agent["source_item_id"]),
            protocol_hash=str(agent.get("protocol_hash") or ""),
            model_hash=str(agent.get("model_hash") or ""),
            n_checks=len(events),
            n_over_budget=sum(1 for e in events if str(e.get("status")) == "over_budget"),
            n_duplicate=sum(1 for e in events if bool(e.get("duplicate"))),
            step_distribution_json=json.dumps(dict(sorted(steps.items())), sort_keys=True),
            detected_invalid=invalid,
            adhered_after_detection=(None if not invalid
                                     else bool(str(agent["state"]) == outcomes.STATE_CORRECT)),
            confidence=None if agent.get("confidence") is None else float(agent["confidence"]),
            validity=bool(agent.get("valid")),
        ))
    return out


def e2a_e1_link(ctx: AnalysisContext) -> list[tables.E2aE1LinkRow]:
    """Per-item E2a endorsement effect beside the E1 switch rates, cross-fitted (spec 9.6).

    A **descriptive**: both sides are post-treatment measurements on the same item, so the fold
    index is carried on every row to make the cross-fitting visible and no p-value is attached.
    """
    e2a: dict[tuple[str, str], list[float]] = {}
    for row in ctx.frame.agent_rounds:
        if str(row.get("cell_kind")) != CellKind.RECEIVER.value:
            continue
        condition = ctx.receiver_grid.get(str(row.get("display_label") or ""))
        if condition is None or int(condition.get("phases") or 1) != 1:
            continue
        sign = {"4:1": 1.0, "1:4": -1.0}.get(str(condition["support"]))
        if sign is None:
            continue
        key = (str(row["source_item_id"]), str(row.get("model_hash") or ""))
        e2a.setdefault(key, []).append(sign * float(str(row["state"]) == outcomes.STATE_CORRECT))
    switches: dict[str, dict[str, int]] = {}
    for row in ctx.transitions:
        item = str(row["source_item_id"])
        bucket = switches.setdefault(item, {"WC": 0, "CW": 0, "n": 0})
        bucket["n"] += 1
        kind = str(row["kind"])
        if kind in bucket:
            bucket[kind] += 1
    families = {str(r["source_item_id"]): str(r.get("family") or "") for r in ctx.frame.agent_rounds}
    out: list[tables.E2aE1LinkRow] = []
    for (item, model), values in sorted(e2a.items()):
        counts = switches.get(item, {"WC": 0, "CW": 0, "n": 0})
        n = counts["n"]
        out.append(tables.E2aE1LinkRow(
            source_item_id=item, model_hash=model, family=families.get(item, ""),
            fold=int(hashlib.sha256(item.encode("utf-8")).hexdigest(), 16) % 2,
            e2a_endorsement_effect=float(np.mean(values)),
            e1_w_to_c_rate=(counts["WC"] / n) if n else None,
            e1_c_to_w_rate=(counts["CW"] / n) if n else None,
            n_e2a_episodes=len(values), n_e1_transitions=n))
    return out


def copying_placebo(ctx: AnalysisContext) -> list[dict[str, Any]]:
    """Copying statistics beside their placebo, per protocol and round (plan §4.1).

    The placebo is the same n-gram statistic against same-item rationales the agent did **not**
    receive.  A copying rate reported without it is a statement about the writing style of the
    model, not about transmission.
    """
    cells: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    labels = {(str(r["cell_kind"]), str(r["episode_id"])): str(r.get("display_label") or "")
              for r in ctx.frame.team_rounds}
    for row in ctx.transitions:
        label = labels.get((str(row["cell_kind"]), str(row["episode_id"])), "")
        cells.setdefault((label, int(row["round_to"])), []).append(row)
    out = []
    for (label, round_), rows in sorted(cells.items()):
        overlaps = [float(r["ngram_overlap_max"]) for r in rows if r.get("ngram_overlap_max") is not None]
        placebo = [float(r["placebo_overlap_max"]) for r in rows if r.get("placebo_overlap_max") is not None]
        out.append({
            "protocol": label, "round": round_, "n": len(rows),
            "copied_exact_rate": float(np.mean([bool(r["copied_exact"]) for r in rows])) if rows else None,
            "ngram_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
            "placebo_overlap_mean": float(np.mean(placebo)) if placebo else None,
            "excess_over_placebo": (float(np.mean(overlaps) - np.mean(placebo))
                                    if overlaps and placebo else None),
            "n_placebo": len(placebo),
        })
    return out


# --------------------------------------------------------------------------- efficiency


def efficiency_frontier(ctx: AnalysisContext) -> list[tables.EfficiencyFrontierRow]:
    """Delivered accuracy against measured cost at the declared budget levels (spec 16.1).

    ``reference_cost`` is frozen at S3 from the pilot cost profile and is **null** until then;
    with a null reference the frontier is not computed at all, because a frontier drawn against
    a guessed reference is a different figure from the pre-registered one.  The truncation rule
    is pre-declared (``last_complete_round``): the last complete synchronous round whose
    cumulative per-item cost, selector included, fits the ceiling.  STOP is always feasible --
    round 0 costs the roots alone -- and the unused allowance is reported rather than discarded.
    """
    efficiency = ctx.analysis.profile.efficiency
    reference = efficiency.reference_cost
    if reference is None:
        return []
    rule = str(efficiency.truncation_rule)
    if rule != "last_complete_round":
        raise ContractViolation(f"efficiency truncation_rule {rule!r} is not implemented")
    unit, reference_cost = sorted(reference.items())[0]
    levels = tuple(float(x) for x in efficiency.budget_levels)
    cost_by_round = _cumulative_cost(ctx, unit)
    rows: list[tables.EfficiencyFrontierRow] = []
    for level in levels:
        allowance = float(level) * float(reference_cost)
        buckets: dict[tuple[str, str], list[tuple[int, float, float | None, float | None]]] = {}
        for (cell_kind, episode_id), by_round in cost_by_round.items():
            meta = by_round["meta"]
            chosen_round, chosen_cost = 0, float(by_round["cumulative"].get(0, 0.0))
            for round_ in sorted(by_round["cumulative"]):
                if float(by_round["cumulative"][round_]) <= allowance:
                    chosen_round, chosen_cost = int(round_), float(by_round["cumulative"][round_])
            outcome_row = by_round["rows"].get(chosen_round)
            if outcome_row is None:
                continue
            key = (str(meta["protocol_hash"]), str(meta["family"]))
            buckets.setdefault(key, []).append((
                chosen_round, chosen_cost,
                OUTCOMES["delivered_accuracy"].lo(outcome_row),
                OUTCOMES["coverage"].hi(outcome_row)))
        for (protocol_hash, family), members in sorted(buckets.items()):
            costs = [m[1] for m in members]
            lo = [m[2] for m in members if m[2] is not None]
            hi = [m[3] for m in members if m[3] is not None]
            label = next((str(r.get("display_label") or "") for r in ctx.frame.team_rounds
                          if str(r.get("protocol_hash")) == protocol_hash), None)
            measured = float(np.mean(costs))
            rows.append(tables.EfficiencyFrontierRow(
                protocol_hash=protocol_hash, budget_level=float(level), family=family,
                display_label=label, truncation_rule=rule,
                truncation_round=int(round(float(np.mean([m[0] for m in members])))),
                reference_cost=float(reference_cost), allowance=allowance,
                measured_cost=measured, unused_allowance=float(allowance - measured),
                delivered_accuracy_lo=float(np.mean(lo)) if lo else None,
                delivered_accuracy_hi=float(np.mean(hi)) if hi else None,
                n_items=len(members)))
    return rows


def _cumulative_cost(ctx: AnalysisContext, unit: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Cumulative per-episode cost by round, selector cost included (spec 16.1)."""
    decisions = _read(ctx.run_root / outcomes.TABLES_DIR, "decisions")
    per_round: dict[tuple[str, str], dict[int, float]] = {}
    for row in decisions:
        key = (str(row["cell_kind"]), str(row["episode_id"]))
        value = row.get(unit)
        per_round.setdefault(key, {})
        per_round[key][int(row["round"])] = per_round[key].get(int(row["round"]), 0.0) + float(value or 0.0)
    for selection in ctx.selections:
        key = (str(selection["cell_kind"]), str(selection["episode_id"]))
        if key in per_round:
            round_ = int(selection["round"])
            per_round[key][round_] = per_round[key].get(round_, 0.0) + float(selection.get("cost_tokens") or 0.0)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ctx.frame.team_rounds:
        key = (str(row["cell_kind"]), str(row["episode_id"]))
        entry = out.setdefault(key, {"cumulative": {}, "rows": {}, "meta": row})
        entry["rows"][int(row["round"])] = row
    for key, entry in out.items():
        running = 0.0
        for round_ in sorted(per_round.get(key, {})):
            running += per_round[key][round_]
            entry["cumulative"][round_] = running
    return out


# --------------------------------------------------------------------------- selections


def seal_selections(ctx: AnalysisContext) -> dict[str, Any]:
    """Seal the selections as label-free held-out predictions (plan §4.5 step 1)."""
    rows = [{"selection_id": str(r["selection_id"]), "episode_id": str(r["episode_id"]),
             "cell_kind": str(r["cell_kind"]), "source_item_id": str(r["source_item_id"]),
             "round": int(r["round"]), "selector": str(r["selector"]),
             "selector_version": str(r["selector_version"]), "pool_id": str(r["pool_id"]),
             "pool_scope": str(r["pool_scope"]),
             "selected_decision_id": r.get("selected_decision_id"),
             "selected_class": r.get("selected_class"),
             "candidate_decision_ids_json": str(r.get("candidate_decision_ids_json") or "[]"),
             "valid_count": int(r.get("valid_count") or 0),
             "fallback_status": str(r.get("fallback_status") or "none")}
            for r in ctx.selections]
    return write_held_out_predictions(ctx.run_root, "selections", rows)


def assert_no_analysis_only_selector_was_deployed(ctx: AnalysisContext) -> tuple[str, ...]:
    """Refuse a run in which a selector the contract declares analysis-only was actually run.

    Spec §6.5 calls the oracle an "evaluation-only ceiling, never a deployed policy", and the
    compiler already refuses it in a deployed list (spec §17.4).  This is the other half of
    that rule, on the data rather than the config: ``selection.analysis_only`` names the
    ceilings, and a sealed ``selections`` row carrying one of those names would mean the
    generation lane ran a selector with label access.  The oracle columns below are computed
    *here*, from grades, precisely so that never has to happen.
    """
    ceilings = {str(name) for name in ctx.cfg.contract.selection.analysis_only}
    deployed = sorted({str(r.get("selector") or "") for r in ctx.selections} & ceilings)
    if deployed:
        raise LabelJoinRefused(
            f"sealed selections carry analysis-only selector(s) {deployed}; spec §6.5 forbids "
            "deploying an evaluation ceiling (the oracle is computed at analysis time)")
    return tuple(sorted(ceilings))


def join_selection_grades(ctx: AnalysisContext) -> list[tables.SelectionGradedRow]:
    """Join grades onto the **sealed** selections and compute the oracle on the same pool."""
    assert_no_analysis_only_selector_was_deployed(ctx)
    decisions = {str(r["decision_id"]): r for r in _read(ctx.run_root / outcomes.TABLES_DIR, "decisions")}

    def state_of(decision_id: str | None) -> str:
        if not decision_id:
            return outcomes.STATE_UNRESOLVED
        row = decisions.get(str(decision_id))
        if row is None:
            return outcomes.STATE_UNRESOLVED
        grade = ctx.grades.get(str(row.get("generation_uid") or ""))
        return outcomes.STATE_UNRESOLVED if grade is None else str(grade["semantic_correct"])

    joined = join_labels(ctx.run_root, "selections", {})
    out: list[tables.SelectionGradedRow] = []
    for row in joined:
        candidates = json.loads(row.get("candidate_decision_ids_json") or "[]")
        states = [state_of(c) for c in candidates]
        selected_state = state_of(row.get("selected_decision_id"))
        oracle_lo = float(any(s == outcomes.STATE_CORRECT for s in states))
        oracle_hi = float(any(s in (outcomes.STATE_CORRECT, outcomes.STATE_UNRESOLVED) for s in states))
        selected_correct = (None if selected_state == outcomes.STATE_UNRESOLVED
                            else float(selected_state == outcomes.STATE_CORRECT))
        out.append(tables.SelectionGradedRow(
            selection_id=str(row["selection_id"]), cell_kind=str(row["cell_kind"]),
            episode_id=str(row["episode_id"]), source_item_id=str(row["source_item_id"]),
            round=int(row["round"]), selector=str(row["selector"]),
            selector_version=str(row["selector_version"]), pool_id=str(row["pool_id"]),
            pool_scope=str(row["pool_scope"]),
            selected_decision_id=row.get("selected_decision_id"),
            selected_class=row.get("selected_class"), selected_state=selected_state,
            selected_correct=selected_correct, oracle_lo=oracle_lo, oracle_hi=oracle_hi,
            selection_gap_lo=(None if selected_correct is None else oracle_lo - selected_correct),
            fallback_status=str(row.get("fallback_status") or "none"),
            n_candidates=len(candidates), valid_count=int(row.get("valid_count") or 0)))
    return out


# --------------------------------------------------------------------------- ledger


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6g}"


def claim_ledger(ctx: AnalysisContext, family: Mapping[str, Any], *,
                 extra_rows: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, str]]:
    """``claim_ledger.csv`` rows, spec 20.3 columns, one row per primary plus the NOT_RUN set.

    The NOT_RUN rows are not decoration: the plan declares Llama-3.1-8B, MCQ option-probe
    scoring, the geometry predictive gate and E4 out of scope, and a ledger without them would
    read as though they had been tested and found wanting (plan D11).
    """
    rows: list[dict[str, str]] = []
    for row in family["hypotheses"]:
        holm = row["holm"]
        rows.append({
            "claim_id": row["h_id"], "statement": row["statement"],
            "evidence_type": "randomized_within_item_contrast", "experiment": row["experiment"],
            "estimand_id": f"{row['h_id']}.{row['outcome']}.{ESTIMAND_VERSION}",
            "source_population": row["population"],
            "model_scope": str(ctx.cfg.analysis.cluster),
            "task_scope": "+".join(sorted(row["by_family"])) or "none",
            "primary_or_secondary": "primary",
            "estimate": _fmt(row["estimate"]),
            "uncertainty": (f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] "
                            f"wild-cluster p={_fmt(row['p'])}"),
            "multiplicity_status": (f"holm {_fmt(holm['p_holm'])} of {family['n_primary']}"
                                    + ("" if holm["estimable"] else " (NOT_ESTIMABLE slot, p=1)")),
            "missingness_sensitivity": row["missingness_sensitivity"],
            "audit_status": "none",
            "supporting_artifact_hash": str(family.get("artifact_sha256") or ""),
            "verdict": row["verdict"],
        })
    for claim_id, statement in NOT_RUN_CLAIMS.items():
        rows.append({
            "claim_id": claim_id, "statement": statement, "evidence_type": "not_run",
            "experiment": "-", "estimand_id": "-", "source_population": "-",
            "model_scope": "-", "task_scope": "-", "primary_or_secondary": "secondary",
            "estimate": "", "uncertainty": "", "multiplicity_status": "not in any family",
            "missingness_sensitivity": "not_estimable", "audit_status": "none",
            "supporting_artifact_hash": "", "verdict": VERDICT_NOT_RUN,
        })
    for row in extra_rows:
        merged = {key: str(row.get(key, "")) for key in LEDGER_HEADER}
        if merged["verdict"] not in VERDICTS:
            raise ContractViolation(
                f"ledger row {merged['claim_id']!r} carries verdict {merged['verdict']!r}; "
                f"allowed: {VERDICTS}")
        rows.append(merged)
    bad = [r["claim_id"] for r in rows if r["verdict"] not in VERDICTS]
    if bad:
        raise ContractViolation(f"ledger rows {bad} carry a verdict outside {VERDICTS}")
    return rows


#: Declared out of scope by the plan (D11).  Each one is a ledger row so a reader can tell
#: "not run" from "run and not supported".
NOT_RUN_CLAIMS: Final[Mapping[str, str]] = {
    "NR_LLAMA_FAMILY": "Llama-3.1-8B family transfer: weights are gated and absent (amendment A4)",
    "NR_MCQ_OPTION_PROBE": "MCQ option-probe scoring (spec 13.1): not run in v1",
    "NR_GEOMETRY_GATE": "the spec 13.4 geometry predictive gate: supplementary only, not run",
    "NR_E4": "E4: outside the core scope of this run (plan D11)",
}


def write_ledger(path: Path, rows: Sequence[Mapping[str, str]]) -> str:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LEDGER_HEADER})
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- metrics


def metric_rows(family: Mapping[str, Any], *, artifact_sha256: str) -> list[tables.MetricRow]:
    """The primary family as ``metrics`` rows (the table whose producer is this module, R3)."""
    out: list[tables.MetricRow] = []
    for row in family["hypotheses"]:
        estimand = f"{row['h_id']}.{row['outcome']}"
        out.append(tables.MetricRow(
            metric_id=f"{estimand}.{ESTIMAND_VERSION}.{row['population']}",
            estimand_id=estimand, estimand_version=ESTIMAND_VERSION,
            population=str(row["population"]), model_scope="8B",
            task_scope="+".join(sorted(row["by_family"])) or "none",
            horizon=f"r{family['endpoint_round']}", weighting="equal_task",
            denominator="source_items", missingness_convention="lo/hi bounds + complete case",
            estimate=row["estimate"], ci_lo=row["ci_lo"], ci_hi=row["ci_hi"],
            p_value=row["p"], multiplicity_family="primary_holm8",
            n_items=row["n_items"], n_clusters=row["n_clusters"],
            artifact_sha256=artifact_sha256))
    return out


# --------------------------------------------------------------------------- verb


def run(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
        profile: str | None = None, seal_only: bool = False) -> cli.Report:
    """``analyze --seal`` then ``analyze --locked``, in the plan §4.5 order.

    Each artifact's sha256 lands in ``derived/MANIFEST.json`` as it is written, so the order the
    artifacts were produced in is itself recorded.
    """
    ctx = load_context(run_root, config_path=config_path, profile=profile)
    ctx.derived.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    manifest["selections_sealed"] = seal_selections(ctx)
    if seal_only:
        _write_json(ctx.derived / DERIVED_MANIFEST, manifest)
        return cli.Report(completed=1, detail="sealed")

    graded = join_selection_grades(ctx)
    tables.write_derived_table(graded, ctx.derived / "selections_graded.parquet")
    outcomes.derive(ctx.run_root, out_dir=ctx.derived,
                    score_rounds=ctx.cfg.contract.team.score_rounds)
    ctx = load_context(run_root, config_path=config_path, profile=profile)

    family = primary_family(ctx)
    manifest["hypotheses"] = _write_json(ctx.derived / "hypotheses.json", family)
    family = dict(family) | {"artifact_sha256": manifest["hypotheses"]}
    tables.write_table(metric_rows(family, artifact_sha256=manifest["hypotheses"]),
                       ctx.derived / "metrics.parquet", table="metrics")
    manifest["standardization"] = _write_json(ctx.derived / "standardization.json", standardization(ctx))
    manifest["calibration_panel"] = _write_json(ctx.derived / "calibration_panel.json", calibration_panel(ctx))
    manifest["passk_panel"] = _write_json(ctx.derived / "passk_panel.json", passk_panel(ctx))
    manifest["e2_factorial"] = _write_json(ctx.derived / "e2_factorial.json", e2_factorial(ctx))
    manifest["e3_factorial"] = _write_json(ctx.derived / "e3_factorial.json", e3_factorial(ctx))
    manifest["copying_placebo"] = _write_json(ctx.derived / "copying_placebo.json", copying_placebo(ctx))
    tables.write_derived_table(efficiency_frontier(ctx), ctx.derived / "efficiency_frontier.parquet")
    tables.write_derived_table(tool_usage_panel(ctx), ctx.derived / "tool_usage_panel.parquet")
    tables.write_derived_table(e2a_e1_link(ctx), ctx.derived / "e2a_e1_link.parquet")
    manifest["panel_census"] = _write_json(ctx.derived / "panel_census.json",
                                           panels.census_report(ctx.frame, ctx.cfg))
    manifest["claim_ledger"] = write_ledger(ctx.derived / "claim_ledger.csv",
                                            claim_ledger(ctx, family))
    _write_json(ctx.derived / DERIVED_MANIFEST, manifest)
    return cli.Report(completed=len(manifest), detail=f"derived={ctx.derived}")


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("analyze")
    parser.add_argument("--seal", action="store_true", help="seal the selections and stop")
    parser.add_argument("--locked", action="store_true", help="run the locked analysis")
    parser.add_argument("--profile", default=None, help="analysis.yaml profile (default: study.yaml's)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seal and args.locked:
        return cli.Report(blocked=1, detail="--seal and --locked are separate passes (plan §4.5)")
    return run(cli.resolve_run_root(args), config_path=args.config, profile=args.profile,
               seal_only=bool(args.seal))


__all__ = [
    "assert_no_analysis_only_selector_was_deployed",
    "AnalysisContext", "DERIVED_MANIFEST", "ESTIMAND_VERSION", "HYPOTHESIS_SPECS",
    "HypothesisResult", "HypothesisSpec", "ItemMatrix", "LABEL_LIKE", "LEDGER_HEADER",
    "LabelJoinRefused", "NOT_RUN_CLAIMS", "OUTCOMES", "Outcome", "PREDICTIONS_DIR",
    "PREDICTIONS_MANIFEST", "VERDICTS", "VERDICT_INCONCLUSIVE", "VERDICT_NARROWED",
    "VERDICT_NOT_ESTIMABLE", "VERDICT_NOT_RUN", "VERDICT_NOT_SUPPORTED", "VERDICT_REFUTED",
    "VERDICT_SUPPORTED", "calibration_panel", "claim_ledger", "copying_placebo", "e2_factorial",
    "e2a_e1_link", "e3_factorial", "efficiency_frontier", "endpoint_matrix", "hypothesis",
    "join_labels", "join_selection_grades", "load_context", "main", "metric_rows",
    "passk_panel", "predictions_dir", "predictions_manifest", "primary_family",
    "receiver_grid", "run", "seal_selections", "standardization", "tool_usage_panel",
    "write_held_out_predictions", "write_ledger",
]
