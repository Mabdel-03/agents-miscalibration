"""Measured gates: smoke, launch, pilot, coverage, release.

Plan §3.5 (S1 launch gate, S2 pilot gate, S4 coverage gates, S6 release-check), spec 18.6
(< 1 % unresolved infrastructure failures, < 2 % format failures, < 1 pp differential between
the primary arms), 19.1 (the ``validate --smoke`` sizes).

Two ARD36 defects are designed out here:

* **A call-type table keyed on a vocabulary the engine never emits.**  ARD36's smoke gate
  looked up ``"revise"``/``"hub"`` while ``episode.py`` wrote ``"update"``/``"leader"``, so
  every revision fell through to ROOT: PEER_REVISE had no measured row at all and revisions
  were silently mixed into *root* parse validity.  Here the measurement is keyed on
  :data:`PHASES` -- the ``contract.PHASE_*`` vocabulary the engine actually writes -- and
  ``test_smoke_call_types_cover_every_phase`` holds the two together.
* **Silence read as success.**  A gate whose measurement is missing FAILS
  (``test_gate_treats_silence_as_failure``): "no rows for this phase" is the ARD36 shape, not
  a pass.  Every check names what it measured, what it required and what it saw.

Every threshold comes from ``study.yaml`` (``gates:``, ``cohorts.pilot_min_paired_items``,
``pilot:``) and every fleet number from ``compute.yaml``; this module declares no number
(``test_no_numeric_dispatch_constant_lives_in_code``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import cli, config as C, contract as K
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn import deps, plan as PLAN, tables
from agents_scaling.nextdyn.ids import CellKind

SMOKE_DIR: Final[str] = "smoke"
MEASURED_FILE: Final[str] = "measured.json"
GATES_DIR: Final[str] = "gates"

#: The phase vocabulary the engine writes (``contract.PHASE_*``): the *only* keys a measured
#: table may carry.  ``bank`` and ``root_ref`` are the two halves of a root: a bank draw is a
#: physical call, a root reference is the same draw seen from an episode and costs nothing.
PHASES: Final[tuple[str, ...]] = (
    K.PHASE_BANK, K.PHASE_ROOT_REF, K.PHASE_REVISION, K.PHASE_RECEIVER_PLAN,
    K.PHASE_RECEIVER_DECISION, K.PHASE_AUDIT,
)
#: Phases that issue a physical request (a root reference does not).
GENERATING_PHASES: Final[tuple[str, ...]] = tuple(p for p in PHASES if p != K.PHASE_ROOT_REF)

PASS: Final[str] = "pass"
FAIL: Final[str] = "fail"
UNMEASURED: Final[str] = "unmeasured"


@dataclass(frozen=True)
class Check:
    """One measured gate check: what it is, what was required, what was seen."""

    name: str
    status: str
    required: str
    observed: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def line(self) -> str:
        return f"[{self.status.upper():<10}] {self.name}: required {self.required}, saw {self.observed}"


@dataclass(frozen=True)
class GateReport:
    """A named set of checks; silence (``unmeasured``) counts as a failure."""

    gate: str
    checks: tuple[Check, ...]
    at: float = field(default_factory=time.time)

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.ok)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "at": self.at, "ok": self.ok,
                "checks": [asdict(c) for c in self.checks]}

    def render(self) -> str:
        return "\n".join([f"gate {self.gate}: {'PASS' if self.ok else 'FAIL'}",
                          *(f"  {c.line()}" for c in self.checks)])

    def as_report(self) -> cli.Report:
        return cli.Report(completed=len(self.checks) - len(self.failed), failed=len(self.failed),
                          detail=f"{self.gate}: {len(self.failed)}/{len(self.checks)} checks failed"
                          if self.failed else f"{self.gate}: all {len(self.checks)} checks pass")


def _rate_check(name: str, observed: float | None, limit: float, *, unit: str = "",
                detail: str = "") -> Check:
    """A rate check where **no measurement is a failure**, not a pass (the ARD36 lesson)."""
    if observed is None:
        return Check(name=name, status=UNMEASURED, required=f"<= {limit}{unit}",
                     observed="nothing measured",
                     detail=detail or "silence is not a pass: the measurement is missing")
    status = PASS if observed <= limit else FAIL
    return Check(name=name, status=status, required=f"<= {limit}{unit}",
                 observed=f"{observed}{unit}", detail=detail)


# --------------------------------------------------------------------------- smoke


@dataclass(frozen=True)
class PhaseMeasurement:
    """What ``validate --smoke`` measured for one phase (plan §3.1's smoke row)."""

    phase: str
    n_calls: int
    parse_valid: float
    boundary_rate: float
    continuation_rate: float
    generated_tokens: float
    prompt_tokens: float
    latency_s: float
    http_error_rate: float
    #: The largest ``thinking_tokens`` seen, and how many draws exceeded the budget.  The rate
    #: alone cannot answer the question the smoke exists to answer, because
    #: ``boundary_injected`` is *defined* as ``thinking_tokens == budget``
    #: (``adapters.split_native``): a draw that overshot would read as "not a boundary", take
    #: no continuation, and be silently truncated.  So the smoke reports the overshoot count.
    max_thinking_tokens: int = 0
    over_budget_draws: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_measurements(run_root: str | os.PathLike, cells: Sequence[Any] | None = None,
                       *, rows: Sequence[Any] | None = None,
                       budget_of: Any = None) -> dict[str, PhaseMeasurement]:
    """Measure per **phase** over the smoke cells' decision rows (never per protocol label).

    The keys are ``contract.PHASE_*``: the same strings ``telemetry.decision_row`` writes, so
    a phase the engine emits can never fall through the table into another phase's row.
    """
    root = Path(run_root)
    decisions = list(rows) if rows is not None else _decisions_of(root, cells)
    out: dict[str, PhaseMeasurement] = {}
    for phase in GENERATING_PHASES:
        mine = [row for row in decisions if str(getattr(row, "phase", "")) == phase]
        if not mine:
            continue
        n = len(mine)
        out[phase] = PhaseMeasurement(
            phase=phase, n_calls=n,
            parse_valid=_mean(_valid(row) for row in mine),
            boundary_rate=_mean(bool(getattr(row, "boundary_injected", False)) for row in mine),
            continuation_rate=_mean(bool(getattr(row, "continuation_used", False)) for row in mine),
            generated_tokens=_mean(_generated_tokens(row) for row in mine),
            prompt_tokens=_mean(_int(getattr(row, "prompt_tokens", 0)) for row in mine),
            latency_s=_mean(float(getattr(row, "latency_s", 0.0) or 0.0) for row in mine),
            http_error_rate=_mean(_infra_failed(row) for row in mine),
            max_thinking_tokens=max((int(_int(getattr(row, "thinking_tokens", 0)))
                                     for row in mine), default=0),
            over_budget_draws=sum(1 for row in mine if budget_of is not None
                                  and _int(getattr(row, "thinking_tokens", 0)) > budget_of(row)))
    return out


def _decisions_of(run_root: Path, cells: Sequence[Any] | None) -> list[Any]:
    """Every **committed** decision row, from sealed and unsealed cells alike.

    ``meta.json`` seals the *cell*; an item is committed atomically on its own (tmp + fsync +
    rename) and in-progress work lives in a separate ``incomplete/`` directory that is never
    read here.  So a row under ``items/`` is final evidence whether or not its cell went on to
    seal, and requiring the seal discarded it.

    That is not merely lossy, it is biased in the one direction a gate must not be.  These
    rates include ``http_error_rate`` and, at the pilot, the per-arm infrastructure-failure
    *differential* -- and a cell that died from infrastructure failures is precisely a cell
    that never sealed.  Measuring only sealed cells therefore drops the worst-affected
    evidence and reports the survivors' rate as the fleet's: silence reading as success, which
    is the shape this gate exists to refuse.  On a preemptable pool it also routinely left
    nothing to measure at all.
    """
    wanted = None if cells is None else {getattr(c, "cell_id", str(c)) for c in cells}
    out: list[Any] = []
    cells_dir = run_root / deps.CELLS_DIR
    for directory in sorted(cells_dir.glob("*")) if cells_dir.exists() else []:
        if wanted is not None and directory.name not in wanted:
            continue
        for path in sorted((directory / tables.ITEMS_DIR).glob("*.json")):
            out.extend(tables.ItemFile.read(path).rows.get("decisions", ()))
    return out


def _valid(row: Any) -> bool:
    """Is this row's output what its phase was asked for?

    Phase-dependent, because the phases are not asked for the same thing.  A ``receiver_plan``
    output is ``{"check_requests": [...]}`` -- it has no final answer by design, so the
    public-output notion of validity reads 0.000 for it however well the model complies, which
    is what the smoke measured before this.  Its validity is whether it parsed *as a plan*,
    which ``run_receiver_episode`` already computes and now records as ``plan_parsed``.

    For every other phase, ``schema.VALID_STATUSES`` decides, not this function.

    It used to ask ``str(parse_status).startswith("OK")``.  ``ParseStatus.OK`` is the *name*;
    its value -- what the row actually carries -- is ``"ok"``, so the test matched nothing and
    the gate scored a healthy run at 0.000 parse validity.  Measured live: 117 of 144 bank
    draws parsed ``ok``, and the launch gate called every one of them a format failure.

    A second declaration of "which statuses are gradeable" is the R1 defect underneath that;
    ``VALID_STATUSES`` is the first one, so this defers to it.
    """
    from agents_scaling.nextdyn import contract as NK
    from agents_scaling.nextdyn.schema import VALID_STATUSES

    if str(getattr(row, "phase", "")) == NK.PHASE_RECEIVER_PLAN:
        return bool(getattr(row, "plan_parsed", False))
    status = getattr(row, "parse_status", None)
    return any(status == ok or str(status) == ok.value for ok in VALID_STATUSES)


def _generated_tokens(row: Any) -> float:
    """Tokens this draw actually produced.

    ``generated_tokens`` is not a column: ``DecisionRow`` records ``thinking_tokens``,
    ``final_tokens`` and ``discarded_span_tokens``.  Reading the name that does not exist
    returned 0.0 for every row through ``getattr(..., 0)``, so the smoke reported zero
    generated tokens on a run that generated ~2,000 per draw.

    The discarded span counts: it is the phase-1 tail a cap-hit draw threw away, and the GPU
    produced it either way -- the same reason plan §3.7 attributes it in the cost accounting.
    """
    return (_int(getattr(row, "thinking_tokens", 0)) + _int(getattr(row, "final_tokens", 0))
            + _int(getattr(row, "discarded_span_tokens", 0)))


def _infra_failed(row: Any) -> bool:
    status = getattr(row, "infrastructure_status", None)
    return bool(status) and str(status) not in ("ok", "none", "")


def _int(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Any) -> float:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


def write_measured(run_root: str | os.PathLike, measured: Mapping[str, PhaseMeasurement]) -> Path:
    path = Path(run_root) / SMOKE_DIR / MEASURED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({phase: m.to_dict() for phase, m in sorted(measured.items())},
                               indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_measured(path: str | os.PathLike) -> dict[str, PhaseMeasurement]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {phase: PhaseMeasurement(**row) for phase, row in data.items()}


# --------------------------------------------------------------------------- launch


def launch_gate(run_root: str | os.PathLike, measured: Mapping[str, PhaseMeasurement], *,
                cfg: Any, required_replicas: int | None = None,
                live_endpoints: int | None = None) -> GateReport:
    """Measured, never assumed (plan §3.5 S1): parse validity, HTTP errors, the fleet floor.

    A phase with no measured row is ``unmeasured`` and the gate fails: that is exactly the
    ARD36 shape where PEER_REVISE had no row and the gate passed anyway.
    """
    checks: list[Check] = [
        _rate_check("http_error_rate",
                    max((m.http_error_rate for m in measured.values()), default=None),
                    float(cfg.gates.max_unresolved_infra_rate)),
    ]
    format_limit = float(cfg.gates.max_format_failure_rate)
    expected = int(cfg.pilot.smoke_solver_calls_per_model)
    episodes = int(cfg.pilot.smoke_receiver_episodes)
    checks.append(_size_check("smoke_solver_calls", measured, K.PHASE_BANK, expected))
    checks.append(_size_check("smoke_receiver_episodes", measured, K.PHASE_RECEIVER_DECISION,
                              episodes))
    for phase in GENERATING_PHASES:
        row = measured.get(phase)
        if row is None:
            checks.append(Check(name=f"parse_validity[{phase}]", status=UNMEASURED,
                                required=f"failures <= {format_limit}", observed="no calls",
                                detail="the smoke made no call of this phase; a phase the run "
                                       "will dispatch must be measured before launch"))
            continue
        checks.append(_rate_check(f"parse_validity[{phase}]", round(1.0 - row.parse_valid, PRECISION),
                                  format_limit, detail=f"{row.n_calls} calls"))
    # The thinking boundary is exact, or the continuation never fires.  ``boundary_injected``
    # is *defined* as ``thinking_tokens == budget``, so a draw that ran past the budget would
    # be recorded as an ordinary completion, take no continuation, and lose its tail silently.
    # ARD36 verified this at 1024 and 4096; this study runs at 2048 and checks it there.
    overshot = {phase: row.over_budget_draws for phase, row in measured.items()
                if row.over_budget_draws}
    checks.append(Check(
        name="thinking_boundary_exact",
        # With nothing measured the boundary is *unmeasured*, not exact: a check that passed on
        # an empty measurement would be the ARD36 shape this gate exists to refuse.
        status=UNMEASURED if not measured else (FAIL if overshot else PASS),
        required="no draw exceeds its thinking budget",
        observed=("; ".join(f"{p}: {n} draws" for p, n in sorted(overshot.items()))
                  if overshot else ("nothing measured" if not measured else "none")),
        detail=("the smoke made no call at all" if not measured else
                "max thinking_tokens per phase: " + ", ".join(
                    f"{p}={m.max_thinking_tokens}" for p, m in sorted(measured.items())))))
    if required_replicas is not None:
        observed = int(live_endpoints or 0)
        checks.append(Check(name="live_replicas", status=PASS if observed >= int(required_replicas) else FAIL,
                            required=f">= {int(required_replicas)}", observed=str(observed),
                            detail="registered AND health-checked (plan §3.5 S4)"))
    del run_root
    return GateReport(gate="launch", checks=tuple(checks))


#: Rounding width for a reported rate: enough digits that a 1-pp threshold is not decided by
#: floating-point noise.  Not a dispatch quantity -- it is how a number is *printed*.
def _size_check(name: str, measured: Mapping[str, PhaseMeasurement], phase: str,
                expected: int) -> Check:
    """Did the smoke actually make the declared number of calls (spec 19.1's sizes)?

    A launch gate computed over three calls is not a measurement of anything; the sizes are
    ``study.yaml``'s and the gate reports what it got against what was declared.
    """
    row = measured.get(phase)
    if row is None:
        return Check(name=name, status=UNMEASURED, required=f">= {expected}",
                     observed="no calls at all", detail=f"phase {phase}")
    return Check(name=name, status=PASS if row.n_calls >= expected else FAIL,
                 required=f">= {expected}", observed=str(row.n_calls),
                 detail=f"the declared spec-19.1 size for {phase}")


#: Rates are reported to the decimal place a one-in-a-million rate is still visible at.  A
#: *report* width, not a dispatch quantity -- nothing schedules on it and no operator tunes it
#: -- written as the width of that rate rather than as a bare ``6`` so the number in the file is
#: the thing it means.
SMALLEST_REPORTED_RATE: Final[str] = "0.000001"
PRECISION: Final[int] = len(SMALLEST_REPORTED_RATE) - len("0.")


# --------------------------------------------------------------------------- pilot


@dataclass(frozen=True)
class ArmRates:
    """The per-arm rates the differential checks are computed between (spec 18.6)."""

    arm: str
    n: int
    infra_failure: float
    continuation: float
    format_failure: float


def arm_rates(rows: Sequence[Any]) -> dict[str, ArmRates]:
    """Group decision rows by ``display_label`` -- the arm, as the row itself records it."""
    by_arm: dict[str, list[Any]] = {}
    for row in rows:
        label = str(getattr(row, "display_label", "") or "")
        if label:
            by_arm.setdefault(label, []).append(row)
    return {arm: ArmRates(arm=arm, n=len(mine),
                          infra_failure=_mean(_infra_failed(r) for r in mine),
                          continuation=_mean(bool(getattr(r, "continuation_used", False)) for r in mine),
                          format_failure=_mean(not _valid(r) for r in mine))
            for arm, mine in by_arm.items()}


def differential(rates: Mapping[str, ArmRates], arms: Sequence[str], field_name: str) -> float | None:
    """The |difference| in percentage points between the two primary arms, or ``None``.

    ``None`` when either arm has no rows: a differential computed against an empty arm is the
    silence-as-success shape, and the caller turns it into an ``unmeasured`` failure.
    """
    present = [rates.get(arm) for arm in arms]
    if any(row is None or row.n == 0 for row in present):
        return None
    values = [float(getattr(row, field_name)) for row in present]
    return round(abs(values[0] - values[-1]) * PERCENT_PER_UNIT, PRECISION)


#: Rates are reported as percentages; a unit conversion, not a dispatch quantity.
PERCENT_PER_UNIT: Final[float] = 100.0


def pilot_gate(pilot_root: str | os.PathLike, *, cfg: Any,
               eligibility: Mapping[str, int] | None = None,
               storage_bytes_per_request: float | None = None,
               adapter_parity: bool | None = None,
               capture_parity: bool | None = None,
               rows: Sequence[Any] | None = None) -> GateReport:
    """The spec-18.6 pilot targets, each measured on the pilot's own rows (plan §3.5 S2)."""
    root = Path(pilot_root)
    decisions = list(rows) if rows is not None else _decisions_of(root, None)
    gates_block = cfg.gates
    checks: list[Check] = []

    infra = _mean(_infra_failed(r) for r in decisions) if decisions else None
    checks.append(_rate_check("unresolved_infra_rate", infra,
                              float(gates_block.max_unresolved_infra_rate),
                              detail=f"{len(decisions)} decisions"))
    fmt = _mean(not _valid(r) for r in decisions) if decisions else None
    checks.append(_rate_check("format_failure_rate", fmt, float(gates_block.max_format_failure_rate)))

    arms = tuple(cfg.protocol.protocols)[:PRIMARY_ARM_COUNT]
    rates = arm_rates(decisions)
    limit = float(gates_block.max_arm_differential_pp)
    for field_name, label in (("infra_failure", "differential_infra_failure_pp"),
                              ("continuation", "differential_continuation_pp")):
        checks.append(_rate_check(label, differential(rates, arms, field_name), limit, unit=" pp",
                                  detail=f"arms {list(arms)}"))

    minimum = int(cfg.cohorts.pilot_min_paired_items)
    if eligibility is None:
        checks.append(Check(name="paired_eligibility", status=UNMEASURED,
                            required=f">= {minimum} paired items per family",
                            observed="not measured",
                            detail="run `construct` on the pilot buckets first"))
    else:
        for family in cfg.cohorts.families:
            n = int(eligibility.get(family, 0))
            checks.append(Check(name=f"paired_eligibility[{family}]",
                                status=PASS if n >= minimum else FAIL,
                                required=f">= {minimum}", observed=str(n),
                                detail="below this the frozen extension rule fires "
                                       f"({list(cfg.cohorts.pilot_eligibility_extensions)})"))
    checks.append(_measured_flag("adapter_parity", adapter_parity,
                                 f"the {cfg.pilot.adapter_check_problems}-problem 32B check"))
    checks.append(_measured_flag("capture_parity", capture_parity,
                                 f"{cfg.pilot.capture_items_per_family} items per family"))
    if storage_bytes_per_request is None:
        checks.append(Check(name="storage_per_request", status=UNMEASURED,
                            required="measured", observed="not measured",
                            detail="the storage projection freeze admits against is this number"))
    else:
        checks.append(Check(name="storage_per_request", status=PASS, required="measured",
                            observed=f"{storage_bytes_per_request:.0f} bytes/request",
                            detail="feeds the freeze-time storage projection"))
    return GateReport(gate="pilot", checks=tuple(checks))


#: H1's contrast is between the first two declared protocols (PRIVATE vs PEER).  The two roles
#: are named, and the count is read off them, so the "2" in the slice above is a fact about the
#: contrast rather than a constant somebody could retune.
PRIMARY_ARM_ROLES: Final[tuple[str, str]] = ("primary", "comparison")
PRIMARY_ARM_COUNT: Final[int] = len(PRIMARY_ARM_ROLES)


def _measured_flag(name: str, value: bool | None, detail: str) -> Check:
    if value is None:
        return Check(name=name, status=UNMEASURED, required="measured", observed="not measured",
                     detail=detail)
    return Check(name=name, status=PASS if value else FAIL, required="parity", observed=str(value),
                 detail=detail)


# --------------------------------------------------------------------------- coverage


def coverage_gate(run_root: str | os.PathLike, manifests: Sequence[Path] | None = None, *,
                  by: str = "block") -> GateReport:
    """Per-block completion census; the block that is short is the one that fails (plan §3.5)."""
    root = Path(run_root)
    paths = list(manifests) if manifests is not None else _manifests(root)
    checks: list[Check] = []
    for path in paths:
        cells = CE.load_manifest(path)
        if not cells:
            continue
        block = cells[0].block or path.stem
        terminal = sum(1 for cell in cells
                       if PLAN.cell_status(root, cell) in PLAN.TERMINAL_STATUSES)
        status = PASS if terminal == len(cells) else FAIL
        checks.append(Check(name=f"coverage[{block}]", status=status,
                            required=f"{len(cells)} terminal cells",
                            observed=f"{terminal} terminal",
                            detail=f"{path.name}"))
    if not checks:
        checks.append(Check(name="coverage", status=UNMEASURED, required="at least one manifest",
                            observed="no block manifests", detail=str(root)))
    del by
    return GateReport(gate="coverage", checks=tuple(checks))


def _manifests(run_root: Path) -> list[Path]:
    directory = run_root.joinpath(*CE.MANIFESTS_DIR)
    return sorted(directory.glob("*.json")) if directory.exists() else []


# --------------------------------------------------------------------------- release


def release_check(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None
                  ) -> Any:
    """The plan §4.5 gate list, run by ``release.check`` (one declaration, R1)."""
    from agents_scaling.nextdyn import release as RL

    return RL.check(run_root, config_path=config_path)


# --------------------------------------------------------------------------- the verbs


#: The runbook's two ``--required-replicas`` labels (plan §3.5 S1 and S4).
LABEL_SMOKE: Final[str] = "smoke"
LABEL_FLOOR: Final[str] = "floor"
REPLICA_LABELS: Final[tuple[str, ...]] = (LABEL_SMOKE, LABEL_FLOOR)


def required_replicas(value: Any, *, compute: Any) -> int | None:
    """Resolve ``--required-replicas`` -- an integer, or one of the runbook's labels.

    ``smoke`` is one replica per declared serving profile (the S1 fleet is exactly ``1x8B +
    1x32B``); ``floor`` is what the pools' declared specs ask for, which is the configured
    intent rather than a number in code.  An unknown label is refused rather than defaulted:
    "how many replicas must be live" is not a question to answer by guessing.
    """
    from agents_scaling.nextdyn import fleet as FL

    if value is None:
        return None
    text = str(value)
    if text.isdigit():
        return int(text)
    if text not in REPLICA_LABELS:
        raise ContractViolation(f"--required-replicas {text!r} is neither an integer nor one of "
                                f"{list(REPLICA_LABELS)}")
    targets = [target for pool in ("deadline", "low", "audit")
               for target in FL.parse_spec(FL.pool_block(compute, pool).spec, FL.DEFAULT_GPU_TYPE)]
    if text == LABEL_SMOKE:
        return len({target.size for target in targets})
    return sum(int(target.count) for target in targets
               if target.partition in _deadline_partitions(compute))


def _deadline_partitions(compute: Any) -> set[str]:
    return {name for name, block in compute.partitions.items() if not block.preemptable}


def write_report(run_root: str | os.PathLike, report: GateReport) -> Path:
    path = Path(run_root) / GATES_DIR / f"{report.gate}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def validate_main(argv: Sequence[str] | None = None) -> cli.Report:
    """``validate --smoke``: measure the smoke cells into ``smoke/measured.json``.

    The smoke *dispatch* is a pilot-sized manifest the planner emits; this verb measures what
    it produced.  With no smoke cells on disk it reports ``blocked`` rather than writing an
    empty measurement that a later gate would read as a pass.
    """
    parser = cli.common_parser("validate", description="smoke measurement (spec 19.1)")
    parser.add_argument("--smoke", action="store_true", help="measure the smoke cells")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    measured = smoke_measurements(run_root)
    if not measured:
        return cli.Report(blocked=1, detail=f"no decisions under {run_root}; run the smoke cells "
                                            "before `validate --smoke`")
    path = write_measured(run_root, measured)
    covered = sorted(measured)
    return cli.Report(completed=len(covered),
                      detail=f"measured {covered} -> {path}")


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("gates", description="launch / pilot / coverage / release gates")
    parser.add_argument("gate", choices=("launch", "pilot", "coverage", "release"))
    parser.add_argument("--measured", type=Path, default=None, help="smoke/measured.json")
    parser.add_argument("--required-replicas", default=None,
                        help="an integer, or a declared label (%s)" % ", ".join(REPLICA_LABELS))
    parser.add_argument("--live-endpoints", type=int, default=None)
    parser.add_argument("--by", default="block")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    cfg = C.load_study(args.config)
    if args.gate == "launch":
        if args.measured is None or not Path(args.measured).exists():
            return cli.Report(blocked=1, detail="--measured smoke/measured.json is required; a "
                                                "launch gate is measured, never assumed")
        live = args.live_endpoints
        if live is None:
            from agents_scaling.nextdyn import fleet as FL

            compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                                     or C.DEFAULT_COMPUTE_CONFIG)
            live = FL.live_endpoint_count(run_root, compute=compute)
        compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                                 or C.DEFAULT_COMPUTE_CONFIG)
        try:
            required = required_replicas(args.required_replicas, compute=compute)
        except ContractViolation as exc:
            return cli.Report(blocked=1, detail=str(exc))
        report = launch_gate(run_root, load_measured(args.measured), cfg=cfg,
                             required_replicas=required, live_endpoints=live)
    elif args.gate == "pilot":
        report = pilot_gate(run_root, cfg=cfg)
    elif args.gate == "coverage":
        report = coverage_gate(run_root, by=args.by)
    else:
        return release_check(run_root, config_path=args.config).as_report()
    write_report(run_root, report)
    print(report.render())
    return report.as_report()


__all__ = [
    "ArmRates", "Check", "FAIL", "GENERATING_PHASES", "GateReport", "MEASURED_FILE", "PASS",
    "PHASES", "PhaseMeasurement", "SMOKE_DIR", "UNMEASURED", "arm_rates", "coverage_gate",
    "differential", "launch_gate", "load_measured", "main", "pilot_gate", "release_check",
    "LABEL_FLOOR", "LABEL_SMOKE", "REPLICA_LABELS", "required_replicas",
    "smoke_measurements", "validate_main", "write_measured", "write_report",
]
