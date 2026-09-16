"""Write-once ``FROZEN.yaml`` / ``frozen_manifest.json``: the same class writes and verifies.

Plan §3.5 S3, spec 11.2 / 17.4 (a confirmation run declares its locks, its authorized budget
and its sample size before any confirmation outcome is opened), contract §2.5.

ARD36's defect D-05 was a frozen key written where the gate did not read it: the writer put
``b0_flops`` in one place, the gate looked in another, and every cell passed a gate that
checked nothing.  Here :class:`FrozenManifest` is both the writer and the reader --
:meth:`FrozenManifest.write` refuses to write a manifest that :meth:`verify` would reject, and
:func:`gate` (what every cell calls before its first request) loads through the same class.
``test_freeze_round_trip_gate_passes_on_the_file_it_just_wrote`` is therefore not a happy-path
test but the structural claim.

Two files with identical content: the spec calls the artifact ``frozen_manifest.json`` and the
runbook (and ``study/``'s runner) look for ``FROZEN.yaml``.  Rather than choose, both names
carry the same canonical JSON bytes and :func:`load_frozen` reads whichever exists, refusing a
run root where the two disagree.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.experiment import io
from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import cli, config as C, contract as K, deps, plan as PLAN
from agents_scaling.nextdyn import workload as WL
from agents_scaling.nextdyn.errors import ContractViolation, LockError, ManifestError
from agents_scaling.study import identity

FROZEN_YAML: Final[str] = "FROZEN.yaml"
FROZEN_JSON: Final[str] = "frozen_manifest.json"
AMENDMENTS_FILE: Final[str] = "amendments.jsonl"
AMENDMENTS_SNAPSHOT: Final[str] = "amendments.frozen.jsonl"
SCHEMA_VERSION: Final[int] = 1

#: The lock fields a confirmation run may not leave null (spec 17.4; plan §3.5 S3).  They are
#: launch *blockers*, not defaults: a null here means a human has not yet authorized the run.
REQUIRED_LOCKS: Final[tuple[str, ...]] = (
    "models_lock_hash", "datasets_lock_hash", "prompts_lock_hash", "source_inventory_hash",
    "prior_use_inventory_hash",
)
REQUIRED_BUDGETS: Final[tuple[str, ...]] = ("campaign_gpu_hours", "campaign_cpu_hours",
                                            "storage_limit_bytes")


@dataclass(frozen=True)
class FrozenManifest:
    """What the run froze, in one object that both writes and verifies it (R1)."""

    schema_version: int
    study_id: str
    stage: str
    contract_sha256: str
    config_sha256: str
    locks: Mapping[str, str]
    budgets: Mapping[str, float]
    manifests: Mapping[str, str]
    sample_size_lock: Mapping[str, int] | None
    analysis_lock_hash: str | None
    amendments_sha256: str
    cost_profile: Mapping[str, Any]
    projection: Mapping[str, Any]
    code_version: str | None
    host: str
    frozen_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "locks", dict(self.locks))
        object.__setattr__(self, "budgets", dict(self.budgets))
        object.__setattr__(self, "manifests", dict(self.manifests))

    # -- io -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenManifest":
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ManifestError(f"frozen manifest carries unknown key(s) {unknown}")
        missing = sorted(set(cls.__dataclass_fields__) - set(data))
        if missing:
            raise ManifestError(f"frozen manifest lacks key(s) {missing}")
        return cls(**dict(data))

    def canonical_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n").encode("utf-8")

    @property
    def sha256(self) -> str:
        return identity.sha256_hex(self.canonical_bytes())

    def write(self, run_root: str | os.PathLike, *, cfg: Any) -> Path:
        """Write both file names, once.  Refuses what :meth:`verify` would reject (D-05)."""
        root = Path(run_root)
        self.verify(cfg)                       # never write a manifest the gate would refuse
        primary = root / FROZEN_JSON
        payload = self.canonical_bytes()
        for name in (FROZEN_JSON, FROZEN_YAML):
            path = root / name
            if path.exists():
                if path.read_bytes() == payload:
                    continue
                raise LockError(
                    f"{path} exists with different content; FROZEN is write-once (a changed "
                    "design is a new run root, never an edit -- spec 11.2)")
            deps.atomic_write_bytes(path, payload)
        return primary

    def verify(self, cfg: Any) -> None:
        """Every check ``gate`` makes, as a method, so the writer runs exactly the reader."""
        if int(self.schema_version) != SCHEMA_VERSION:
            raise LockError(f"frozen schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if self.study_id != cfg.study.id:
            raise LockError(f"frozen manifest is for study {self.study_id!r}, config is "
                            f"{cfg.study.id!r}")
        for name in REQUIRED_LOCKS:
            if not self.locks.get(name):
                raise LockError(f"frozen manifest has a null {name}; lock the inputs before "
                                "freezing (spec 17.4 launch blocker)")
        for name in REQUIRED_BUDGETS:
            if self.budgets.get(name) is None:
                raise LockError(f"frozen manifest has a null budgets.{name}; the campaign "
                                "ceiling is authorized by a human, never defaulted (spec 16.4)")
        if not self.manifests:
            raise LockError("frozen manifest names no block manifest; freeze emits every block")
        if self.sample_size_lock is None:
            raise LockError("frozen manifest has a null confirmatory_sample_size_lock; run "
                            "`power` and copy its lock into study.yaml first")
        if not self.analysis_lock_hash:
            raise LockError("frozen manifest has a null analysis_lock_hash; the analysis is "
                            "locked before any confirmation outcome is opened (spec 14.1)")
        if self.config_sha256 != cfg.raw_sha256:
            raise LockError(f"study.yaml has changed since freeze "
                            f"({cfg.raw_sha256[:12]} != frozen {self.config_sha256[:12]})")


def frozen_path(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / FROZEN_JSON


def is_frozen(run_root: str | os.PathLike) -> bool:
    root = Path(run_root)
    return (root / FROZEN_JSON).exists() or (root / FROZEN_YAML).exists()


def load_frozen(run_root: str | os.PathLike) -> FrozenManifest:
    """Read the frozen manifest; refuse a run root whose two file names disagree."""
    root = Path(run_root)
    primary, secondary = root / FROZEN_JSON, root / FROZEN_YAML
    if not primary.exists() and not secondary.exists():
        raise LockError(f"{primary} does not exist; this run root is not frozen")
    if primary.exists() and secondary.exists() and primary.read_bytes() != secondary.read_bytes():
        raise LockError(f"{primary} and {secondary} disagree; one of them was edited")
    path = primary if primary.exists() else secondary
    return FrozenManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def gate(run_root: str | os.PathLike, cfg: Any) -> FrozenManifest | None:
    """What every cell calls before its first request (contract §2.5).

    ``None`` when the run root is not frozen *and* the configuration does not require it (a
    pilot or test run); a :class:`LockError` when it is frozen and something drifted, or when
    a confirmation run is not frozen at all.
    """
    confirmation = cfg.study.mode == K.MODE_CONFIRMATION
    if not is_frozen(run_root):
        if confirmation:
            raise LockError(
                f"{frozen_path(run_root)} does not exist and study.mode is "
                f"{cfg.study.mode!r}; a confirmation cell never runs unfrozen (spec 11.2)")
        return None
    frozen = load_frozen(run_root)
    frozen.verify(cfg)
    return frozen


# --------------------------------------------------------------------------- amendments


def snapshot_amendments(run_root: str | os.PathLike, *, source: Path | None = None) -> tuple[Path, str]:
    """Copy the amendment ledger into the run root and hash it (plan §3.5 S3).

    An amendment is a change to the frozen design (A1-A4); freezing without recording which
    amendments were in force at the time makes the ledger unfalsifiable later.
    """
    root = Path(run_root)
    src = Path(source) if source is not None else root / AMENDMENTS_FILE
    payload = src.read_bytes() if src.exists() else b""
    target = root / AMENDMENTS_SNAPSHOT
    if target.exists() and target.read_bytes() != payload:
        raise LockError(f"{target} exists with different content; the amendment snapshot is "
                        "taken once, at freeze")
    if not target.exists():
        deps.atomic_write_bytes(target, payload)
    return target, identity.sha256_hex(payload)


# --------------------------------------------------------------------------- the verb


def declared_locks(cfg: Any) -> dict[str, Any]:
    """Every lock hash the study declares, read by name (R2: no ``getattr`` indirection).

    ``container_digests`` is in the set because a run whose graders ran inside a container has
    to say which image; it is null while no family needs one, and null is a *recorded* answer.
    """
    locks = cfg.locks
    return {
        "models_lock_hash": locks.models_lock_hash,
        "datasets_lock_hash": locks.datasets_lock_hash,
        "prompts_lock_hash": locks.prompts_lock_hash,
        "source_inventory_hash": locks.source_inventory_hash,
        "prior_use_inventory_hash": locks.prior_use_inventory_hash,
        "git_commit": locks.git_commit,
        "container_digests": None if locks.container_digests is None
        else json.dumps(dict(locks.container_digests), sort_keys=True),
    }


def build_manifest(run_root: str | os.PathLike, cfg: Any, *, stage: str,
                   manifests: Mapping[str, str], projection: Mapping[str, Any],
                   cost_profile: Mapping[str, Any], amendments_sha256: str,
                   clock: Any = time.time) -> FrozenManifest:
    locks = {name: value for name, value in declared_locks(cfg).items() if value is not None}
    budgets = {name: getattr(cfg.budgets, name) for name in REQUIRED_BUDGETS}
    budgets = {name: value for name, value in budgets.items() if value is not None}
    contract = cfg.contract
    return FrozenManifest(
        schema_version=SCHEMA_VERSION, study_id=cfg.study.id, stage=stage,
        contract_sha256=contract.sha256(), config_sha256=cfg.raw_sha256,
        locks=locks, budgets=budgets, manifests=dict(manifests),
        sample_size_lock=(None if cfg.analysis.confirmatory_sample_size_lock is None
                          else dict(cfg.analysis.confirmatory_sample_size_lock)),
        analysis_lock_hash=cfg.analysis.analysis_lock_hash,
        amendments_sha256=amendments_sha256, cost_profile=dict(cost_profile),
        projection={**dict(projection), "spec_version": cfg.study.spec_version,
                    "contract_version": contract.contract_version},
        code_version=io.git_commit(),
        host=socket.gethostname(), frozen_at=float(clock()))


def freeze(run_root: str | os.PathLike, cfg: Any, *, stage: str = CE.STAGE_CONFIRMATION,
           inventory: PLAN.Inventory | None = None) -> tuple[FrozenManifest, PLAN.Plan]:
    """Emit every block manifest once, then write the frozen manifest (plan §3.4, §3.5 S3)."""
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    inventory = inventory if inventory is not None else PLAN.Inventory.from_run_root(root, cfg)
    dag = PLAN.build_dag(cfg, inventory, stage=stage)
    profile = WL.load_cost_profile(root / CE.STAGE_PILOT)
    load = WL.calls(dag.cells, contract=dag.state.contract, profile=profile,
                    reuse_fraction=dag.reuse)
    admission = WL.admit(load, budgets=cfg.budgets)
    if not admission.admitted:
        raise ContractViolation(f"refusing to freeze: {admission.detail}")
    if not admission.storage_admitted:
        raise ContractViolation(
            f"refusing to freeze: projected storage {admission.storage_bytes / 1e9:.1f} GB "
            f"exceeds storage_limit_bytes")
    _, amendments_sha = snapshot_amendments(root)
    manifests = dag.write_manifests(root, meta={"stage": stage, "study_id": cfg.study.id})
    frozen = build_manifest(root, cfg, stage=stage, manifests=manifests,
                            projection=admission.to_dict(), cost_profile=profile.to_dict(),
                            amendments_sha256=amendments_sha)
    frozen.write(root, cfg=cfg)
    PLAN.write_plan_table(root, PLAN.plan_rows(root, dag, manifest_hashes=manifests))
    return frozen, dag


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("freeze", description="write-once FROZEN.yaml + frozen_manifest.json")
    parser.add_argument("--stage", default=CE.STAGE_CONFIRMATION, choices=list(CE.STAGES))
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = C.load_study(args.config)
    run_root = cli.resolve_run_root(args)
    try:
        frozen, dag = freeze(run_root, cfg, stage=args.stage)
    except (ContractViolation, LockError, ManifestError) as exc:
        return cli.Report(blocked=1, detail=str(exc))
    return cli.Report(completed=len(frozen.manifests),
                      detail=f"frozen {frozen.sha256[:12]}: {len(dag.cells)} cells in "
                             f"{len(frozen.manifests)} block manifests")


__all__ = [
    "AMENDMENTS_FILE", "AMENDMENTS_SNAPSHOT", "FROZEN_JSON", "FROZEN_YAML", "FrozenManifest",
    "REQUIRED_BUDGETS", "REQUIRED_LOCKS", "SCHEMA_VERSION", "build_manifest", "freeze",
    "declared_locks", "frozen_path", "gate", "is_frozen", "load_frozen", "main",
    "snapshot_amendments",
]
