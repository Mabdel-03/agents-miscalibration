"""Bounded named paths into the legacy runs (spec 7.1; plan §2.6, D10).

Every E0 read goes through this module, and this module reads nothing that
``configs/nextdyn/legacy_sources.yaml`` does not name.  There is **no directory walk** in the
E0 package: a walk finds whatever happens to be on disk, which is how an analysis silently
changes when a stale ``tables/`` directory is left beside a repaired ``tables_v2/``.  ARD36's
own repair had exactly that shape, so the registry names ``tables_v2`` and ``tables_v1``
separately and a claim states which one it read.

A registered path that is not on disk is :data:`BLOCKED_MISSING_SOURCE` -- the source is
missing, the claim is not checkable, and no observation is invented.  That is a different
status from ``unavailable``, which means the input exists but no code that could have produced
the published number does.

Path forms admitted in the YAML:

``/absolute/path``        used as-is
``repo:relative/path``    resolved against the repository root
``<section>:relative``    resolved against another section's ``root`` (e.g. ``study_v4:data/x``)
``relative/path``         resolved against this section's own ``root``
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

import yaml

from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation

#: The status a claim carries when a named source is not on disk (spec 7.1).
BLOCKED_MISSING_SOURCE: Final[str] = "BLOCKED_MISSING_SOURCE"

DEFAULT_REGISTRY: Final[Path] = C.CONFIG_DIR / "legacy_sources.yaml"
REPO_PREFIX: Final[str] = "repo:"


@dataclass(frozen=True)
class Ard36Sources:
    """The ARD36 run root and the artifacts E0 reads from it (plan §2.6 A1-A5)."""

    root: str
    tables: str
    tables_v1: str
    cells: str
    items: str
    items_reserve: str
    items_summary: str
    provenance: str
    stats: str
    calibration_panel: str
    bank_summary: str


@dataclass(frozen=True)
class StudyV4Sources:
    """The study_v4 atlas run root (plan §2.6 B1-B6)."""

    root: str
    tables: str
    seals: str
    tasks: str
    data_sha256: str
    raw_bcb: str
    eval_hle: str
    eval_bcb: str
    geometry: str
    cells: str


@dataclass(frozen=True)
class FullSweepCache:
    """The Run0 analysis cache parquet files (plan §2.6 C1-C3)."""

    cells_dedup: str
    analysis_view: str
    items: str
    agents: str
    ingest_manifest: str


@dataclass(frozen=True)
class FullSweepSources:
    root: str
    cells: str
    cells_json: str
    contracts: str
    contracts_sha256: str
    cache: FullSweepCache
    findings: str


@dataclass(frozen=True)
class BcbRepairSources:
    """The BCB behavioural-repair inputs (spec 7.4)."""

    sandbox_sif: str
    input_generator_policy: str
    max_inputs_per_task: int
    audit_suite_inputs: int


@dataclass(frozen=True)
class LegacySources:
    """The whole registry, compiled strictly (an unknown key is an error, like every config)."""

    ard36: Ard36Sources
    study_v4: StudyV4Sources
    full_sweep_v1: FullSweepSources
    prior_roots: tuple[str, ...]
    bcb_repair: BcbRepairSources
    # not a YAML key: where the registry was read from (config.build skips provenance fields)
    path: Path = field(default=Path(), metadata={"provenance": True})

    @property
    def sections(self) -> dict[str, Any]:
        return {"ard36": self.ard36, "study_v4": self.study_v4, "full_sweep_v1": self.full_sweep_v1}


@dataclass(frozen=True)
class Resolved:
    """One named path, resolved and probed once."""

    name: str
    declared: str
    path: Path
    exists: bool
    is_dir: bool
    size: int | None
    sha256: str | None

    @property
    def status(self) -> str:
        return "ok" if self.exists else BLOCKED_MISSING_SOURCE


def load(path: str | os.PathLike | None = None) -> LegacySources:
    """Compile ``legacy_sources.yaml``; unknown and missing keys are both errors."""
    target = Path(path) if path is not None else DEFAULT_REGISTRY
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ConfigError(f"{target}: top level must be a mapping")
    sources = C.build(LegacySources, data, "legacy_sources")
    offending = C.scratch_paths(target)
    if offending:
        raise ContractViolation(
            f"{target}: paths under {C.SCRATCH_PREFIX} are refused (plan D3): {offending}")
    return LegacySources(ard36=sources.ard36, study_v4=sources.study_v4,
                         full_sweep_v1=sources.full_sweep_v1, prior_roots=sources.prior_roots,
                         bcb_repair=sources.bcb_repair, path=target)


def repo_root() -> Path:
    return C.REPO_ROOT


def resolve_declared(sources: LegacySources, section: str, declared: str) -> Path:
    """Turn one declared string into an absolute path under the four admitted forms."""
    if declared.startswith("/"):
        return Path(declared)
    if declared.startswith(REPO_PREFIX):
        return repo_root() / declared[len(REPO_PREFIX):]
    if ":" in declared:
        other, _, rest = declared.partition(":")
        target = sources.sections.get(other)
        if target is None:
            raise ContractViolation(
                f"legacy_sources.{section}: {declared!r} names section {other!r}, which is not "
                f"one of {sorted(sources.sections)}")
        return Path(target.root) / rest
    entry = sources.sections.get(section)
    if entry is None:
        raise ContractViolation(f"legacy_sources: unknown section {section!r}")
    return Path(entry.root) / declared


def _sha256(path: Path, *, limit: int = 1 << 28) -> str | None:
    """sha256 of a file up to ``limit`` bytes; a larger or unreadable file reports ``None``."""
    try:
        if path.stat().st_size > limit:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def resolve(sources: LegacySources, name: str, *, hash_files: bool = True) -> Resolved:
    """Resolve ``"<section>.<field>"`` (or ``"<section>.cache.<field>"``) and probe it.

    A name the registry does not declare raises: E0 may not read a path a human did not write
    down, and "the file was not where I guessed" must never look like "the claim failed".
    """
    parts = name.split(".")
    section = parts[0]
    entry: Any = sources.sections.get(section)
    if entry is None and section == "bcb_repair":
        entry = sources.bcb_repair
    if entry is None:
        raise ContractViolation(f"legacy_sources: unknown section {section!r} in {name!r}")
    for attr in parts[1:]:
        if not hasattr(entry, attr):
            raise ContractViolation(
                f"legacy_sources.{name}: {section} declares no {attr!r}; E0 never guesses a path")
        entry = getattr(entry, attr)
    if not isinstance(entry, str):
        raise ContractViolation(f"legacy_sources.{name} is not a path")
    path = resolve_declared(sources, section, entry)
    exists = path.exists()
    is_dir = exists and path.is_dir()
    size = (path.stat().st_size if exists and not is_dir else None)
    return Resolved(name=name, declared=entry, path=path, exists=exists, is_dir=is_dir,
                    size=size, sha256=(_sha256(path) if exists and not is_dir and hash_files else None))


def resolve_all(sources: LegacySources, *, hash_files: bool = False) -> list[Resolved]:
    """Every declared path of every section, in a stable order."""
    import dataclasses

    out: list[Resolved] = []
    for section, entry in sources.sections.items():
        for field in dataclasses.fields(entry):
            value = getattr(entry, field.name)
            if isinstance(value, str):
                out.append(resolve(sources, f"{section}.{field.name}", hash_files=hash_files))
            elif dataclasses.is_dataclass(value):
                for sub in dataclasses.fields(value):
                    out.append(resolve(sources, f"{section}.{field.name}.{sub.name}",
                                       hash_files=hash_files))
    return out


def cache_paths(sources: LegacySources) -> list[Resolved]:
    """The Run0 analysis-cache parquet files (``legacy_sources.full_sweep_v1.cache``).

    They live in the repository rather than in the run root (the ``repo:`` prefix), because the
    Run0 refresh pipeline is not runnable and the cache is the only readable form of that run's
    item-level data (plan D10).
    """
    import dataclasses

    cache = sources.full_sweep_v1.cache
    return [resolve(sources, f"full_sweep_v1.cache.{f.name}") for f in dataclasses.fields(cache)]


def missing(sources: LegacySources) -> list[str]:
    """Named paths that are not on disk; each one blocks the claims that read it."""
    return [r.name for r in resolve_all(sources) if not r.exists]


def prior_root_paths(sources: LegacySources) -> list[Path]:
    """The twelve prior run roots without contract sidecars (spec 5.2: ``unknown``, not untouched)."""
    return [Path(p) for p in sources.prior_roots]


__all__ = [
    "Ard36Sources", "BLOCKED_MISSING_SOURCE", "BcbRepairSources", "DEFAULT_REGISTRY",
    "FullSweepCache", "FullSweepSources", "LegacySources", "REPO_PREFIX", "Resolved",
    "StudyV4Sources", "cache_paths", "load", "missing", "prior_root_paths", "repo_root", "resolve",
    "resolve_all", "resolve_declared",
]
