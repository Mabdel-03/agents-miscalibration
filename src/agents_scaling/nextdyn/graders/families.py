"""Family dispatch, ``grader_hash`` and the protected gold store (spec 4.6, 5.1; plan §2.5).

Three facts live here and nowhere else:

* **dispatch** -- ``canonical_class(family, answer, target)`` and
  ``grade(family, answer, gold, target)`` resolve ``FamilySpec.canonicalizer`` /
  ``FamilySpec.grader`` (dotted names, so ``families.py`` itself stays runtime-safe) and call
  them with the uniform signature;
* **``grader_hash(family)``** -- sha256 over the grader module bytes, the canonicalizer module
  bytes, the pinned tool versions and the schema version, so a sympy or lark bump invalidates
  every stored label instead of silently re-basing accuracy (``ard/graders/__init__.py:115``
  pattern, re-keyed to nextdyn modules);
* **the gold store** -- ``<run_root>/protected/gold/<family>.jsonl``, mode 0600 inside a 0700
  directory, read *only* from the writer side.  A group- or world-readable gold file is a
  :class:`~agents_scaling.nextdyn.errors.LeakageError`, never a warning
  (``ard/graders/__init__.py:148-227`` generalized from two families to the registry).

``gold_ref_hash`` (the ``grades_private`` column) is sha256 of the canonical JSON of the gold
record, so a grade can be re-checked against the exact reference it used without the
reference leaving the protected namespace.

Writer side (R7): this module is the one that reads gold from disk.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn.errors import LeakageError, NextdynError
from agents_scaling.nextdyn.graders import GRADER_SCHEMA_VERSION, GradeTarget, Verdict
from agents_scaling.study import identity

PROTECTED_DIR: Final[str] = "protected"
GOLD_DIR: Final[str] = "gold"
GOLD_SUFFIX: Final[str] = ".jsonl"
PROTECTED_FILE_MODE: Final[int] = 0o600
PROTECTED_DIR_MODE: Final[int] = 0o700

#: The key of a gold record that names the item it belongs to.
GOLD_ITEM_KEY: Final[str] = "source_item_id"


class GoldUnavailable(NextdynError):
    """A reference answer was requested and the protected store does not hold it."""


# --------------------------------------------------------------------------- grader hash


def _module_digest(dotted: str) -> str:
    import importlib

    module = importlib.import_module(dotted.partition(":")[0])
    return identity.sha256_hex(Path(module.__file__).read_bytes())


def _tool_versions(family: str) -> dict[str, str]:
    """The pinned versions the family's rules depend on (empty for a pure-string grader)."""
    spec = F.family(family)
    if spec.grader.startswith("agents_scaling.nextdyn.graders.math:"):
        import sympy

        from agents_scaling.ard.graders import math_equiv

        return {"sympy": sympy.__version__, "lark": math_equiv.PINNED_LARK_VERSION,
                "latex_backend": math_equiv.LATEX_BACKEND,
                "math_equiv": identity.sha256_hex(Path(math_equiv.__file__).read_bytes())}
    if spec.grader.startswith("agents_scaling.nextdyn.graders.trace:"):
        from agents_scaling.ard.graders import trace

        return {"trace": identity.sha256_hex(Path(trace.__file__).read_bytes())}
    return {}


def grader_hash(family: str) -> str:
    """``sha256`` over the family's grading rules and their pinned tools."""
    spec = F.family(family)
    payload = {
        "schema": GRADER_SCHEMA_VERSION,
        "family": spec.name,
        "grader": spec.grader,
        "canonicalizer": spec.canonicalizer,
        "grader_module": _module_digest(spec.grader),
        "canonicalizer_module": _module_digest(spec.canonicalizer),
        "tools": _tool_versions(spec.name),
    }
    return identity.sha256_hex(identity.jcs(payload))


def grader_hashes() -> dict[str, str]:
    return {name: grader_hash(name) for name in F.FAMILIES}


# --------------------------------------------------------------------------- dispatch


def canonical_class(family: str, answer: str | None, target: GradeTarget) -> str | None:
    """The gold-independent voting class of ``answer`` for ``family``."""
    return F.resolve(F.family(family).canonicalizer)(answer, target)


def grade(family: str, answer: str | None, gold: Mapping[str, Any], target: GradeTarget) -> Verdict:
    """Grade ``answer`` against the protected ``gold`` record of ``target``'s item."""
    return F.resolve(F.family(family).grader)(answer, gold, target)


# --------------------------------------------------------------------------- BigCodeBench (E0)


def code_canonical_class(answer: str | None, target: GradeTarget) -> str | None:
    """AST-canonical class of a code answer (``study.selection.normalize.code_key``)."""
    del target
    if answer is None or not str(answer).strip():
        return None
    from agents_scaling.study.selection.normalize import code_key

    key, mode = code_key(str(answer))
    return None if key is None else f"code:{mode}:{key}"


def code_grade(answer: str | None, gold: Mapping[str, Any], target: GradeTarget) -> Verdict:
    """BigCodeBench is E0 only: its behavioural grading runs inside the pinned evaluator
    image from ``e0.bcb_repair`` (plan §2.6), never in a GRADE cell of this study."""
    del answer, gold
    raise GoldUnavailable(
        f"{target.family} is an E0-only family: candidate code is graded by e0.bcb_repair "
        f"inside the apptainer sandbox, not by graders.families.grade")


# --------------------------------------------------------------------------- the gold store


def protected_root(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / PROTECTED_DIR


def gold_dir(run_root: str | os.PathLike) -> Path:
    return protected_root(run_root) / GOLD_DIR


def gold_path(run_root: str | os.PathLike, family: str) -> Path:
    return gold_dir(run_root) / f"{F.family(family).name}{GOLD_SUFFIX}"


def _assert_protected(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise LeakageError(f"{path} is group/world accessible (mode {oct(mode)}); gold files must "
                           f"be {oct(PROTECTED_FILE_MODE)}")


def write_gold(run_root: str | os.PathLike, family: str, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write one family's gold as 0600 JSONL inside a 0700 directory, atomically."""
    path = gold_path(run_root, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, PROTECTED_DIR_MODE)
    lines = []
    for record in records:
        if not record.get(GOLD_ITEM_KEY):
            raise GoldUnavailable(f"a {family} gold record has no {GOLD_ITEM_KEY}")
        lines.append(json.dumps(dict(record), sort_keys=True, ensure_ascii=False, allow_nan=False))
    payload = ("\n".join(lines) + "\n" if lines else "").encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp")
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PROTECTED_FILE_MODE)
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(tmp, PROTECTED_FILE_MODE)
    os.replace(tmp, path)
    return path


@dataclass(frozen=True)
class GoldStore:
    """Evaluator-only reference answers, keyed ``(family, source_item_id)``."""

    run_root: str
    records: Mapping[str, Mapping[str, Mapping[str, Any]]]

    @classmethod
    def load(cls, run_root: str | os.PathLike, families: Sequence[str] | None = None) -> "GoldStore":
        names = list(families) if families is not None else sorted(F.FAMILIES)
        out: dict[str, dict[str, Mapping[str, Any]]] = {}
        for name in names:
            path = gold_path(run_root, name)
            if not path.exists():
                continue
            _assert_protected(path)
            table: dict[str, Mapping[str, Any]] = {}
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, Mapping) or not record.get(GOLD_ITEM_KEY):
                    raise GoldUnavailable(f"{path}:{lineno}: gold row has no {GOLD_ITEM_KEY}")
                table[str(record[GOLD_ITEM_KEY])] = record
            out[name] = table
        if not out:
            raise GoldUnavailable(f"{gold_dir(run_root)} holds no gold; run items_build first")
        return cls(run_root=str(run_root), records=out)

    def reference(self, family: str, source_item_id: str) -> Mapping[str, Any]:
        try:
            return self.records[family][source_item_id]
        except KeyError as exc:
            raise GoldUnavailable(f"no protected gold for {family} item {source_item_id!r}") from exc

    def ref_hash(self, family: str, source_item_id: str) -> str:
        """``grades_private.gold_ref_hash``: the identity of the reference that was used."""
        return identity.sha256_hex(identity.jcs(dict(self.reference(family, source_item_id))))

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))


__all__ = [
    "GOLD_DIR", "GOLD_ITEM_KEY", "GOLD_SUFFIX", "GoldStore", "GoldUnavailable",
    "PROTECTED_DIR_MODE", "PROTECTED_FILE_MODE", "canonical_class", "code_canonical_class",
    "code_grade", "gold_dir", "gold_path", "grade", "grader_hash", "grader_hashes",
    "protected_root", "write_gold",
]
