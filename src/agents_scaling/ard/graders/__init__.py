"""ARD36 graders: the evaluator-namespace surface and the gold-independent classes (WP3).

Spec §5.1 (per-family grading rules), §18 (``labels.parquet`` carries
``evaluator_status``/``grader_version``; ``items.parquet`` carries ``grader_hash``), plan
§5.1 ("gold ... lives only in the 0700/0600 protected namespace read by ``ard/graders``").

Two surfaces, and the split is the firewall:

* :func:`canonical_class` is **gold-independent** and safe for runtime use (voting,
  BALANCED packet construction).  It loads nothing.
* :func:`grade` needs the reference and therefore a :class:`GoldStore`, which reads the
  0600 files under ``<run_root>/data/protected/``.  Only evaluator-lane code may call it.
  A caller that has not bound a store gets :class:`GoldUnavailable`, never a silent
  default.

``grader_hash(family)`` is the content hash of the family's grader module **plus its
pinned tool versions** (WP3 requirement), so a lark or sympy bump, or any edit to the
grading rules, invalidates every stored label rather than silently re-basing accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

import sympy

from agents_scaling.ard.contract import Family
from agents_scaling.ard.graders import math_equiv, mmlu, trace
from agents_scaling.ard.items import (
    MATH_GOLD_FILE,
    MMLU_GOLD_FILE,
    PROTECTED_FILE_MODE,
    protected_dir,
)

STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_TIMEOUT = "timeout"
#: The four statuses of the WP0 ``GradeResult`` contract.
STATUSES: tuple[str, ...] = (STATUS_CORRECT, STATUS_INCORRECT, STATUS_AMBIGUOUS, STATUS_TIMEOUT)

#: Bumped whenever the grading *rules* change independently of the module bytes.
GRADER_SCHEMA_VERSION: int = 1
#: Family key used for the E3 receiver trace grader (not a natural-cohort family).
TRACE_FAMILY: str = "receiver_trace"


class GoldUnavailable(RuntimeError):
    """A reference answer was requested without a bound protected gold store."""


@dataclass(frozen=True)
class GradeResult:
    """WP0 interface ``grade(family, answer, item_id) -> GradeResult``."""

    item_id: str
    family: str
    status: str
    detail: str
    canonical_class: str | None
    grader_hash: str
    duration_s: float

    def to_row(self) -> dict:
        """``labels.parquet``-shaped view (spec §18)."""
        return {
            "item_id": self.item_id,
            "family": self.family,
            "is_correct": self.status == STATUS_CORRECT,
            "evaluator_status": self.status,
            "evaluator_detail": self.detail,
            "answer_class": self.canonical_class,
            "grader_version": self.grader_hash,
            "grading_duration": self.duration_s,
        }


# --------------------------------------------------------------------------- grader hash


def _module_digest(module) -> str:
    path = Path(module.__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_versions(family_key: str) -> dict[str, str]:
    if family_key == Family.MATH500.value:
        return {"lark": math_equiv.PINNED_LARK_VERSION, "sympy": sympy.__version__,
                "latex_backend": math_equiv.LATEX_BACKEND}
    if family_key == Family.MMLUPRO.value:
        return {"normalize": _module_digest(
            __import__("agents_scaling.study.selection.normalize", fromlist=["normalize"]))}
    return {}


def grader_module(family: Family | str):
    """The module implementing ``family``'s grading rules."""
    key = family.value if isinstance(family, Family) else str(family)
    if key == Family.MATH500.value:
        return math_equiv
    if key == Family.MMLUPRO.value:
        return mmlu
    if key == TRACE_FAMILY:
        return trace
    raise ValueError(f"unknown grader family {family!r}")


def grader_hash(family: Family | str) -> str:
    """``sha256`` over the grader module bytes, its pinned tool versions and the schema.

    This is the ``grader_hash`` column of ``items.parquet`` (spec §18) and the
    ``grader_version`` of ``labels.parquet``.
    """
    key = family.value if isinstance(family, Family) else str(family)
    payload = {
        "schema": GRADER_SCHEMA_VERSION,
        "family": key,
        "module": _module_digest(grader_module(key)),
        "facade": _module_digest(_this_module()),
        "tools": _tool_versions(key),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _this_module():
    import sys

    return sys.modules[__name__]


def grader_hashes() -> dict[str, str]:
    """``{family: grader_hash}`` for both natural families and the receiver traces."""
    return {key: grader_hash(key)
            for key in (Family.MATH500.value, Family.MMLUPRO.value, TRACE_FAMILY)}


# --------------------------------------------------------------------------- gold store


def _read_protected_jsonl(path: Path) -> list[dict]:
    """Read a 0600 protected file, refusing a group/world-readable one.

    Mirrors the fail-closed discipline of ``study/data/protected.py`` (which restricts its
    own importers to ``study.evaluation``) without importing it.
    """
    if not path.exists():
        raise FileNotFoundError(f"protected gold file missing: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"{path} is group/world accessible (mode {oct(mode)}); protected files must be "
            f"{oct(PROTECTED_FILE_MODE)}"
        )
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: protected row is not an object")
            rows.append(row)
    return rows


@dataclass(frozen=True)
class GoldStore:
    """Evaluator-only reference answers, keyed by ``item_id``."""

    run_root: str
    math: dict[str, dict]
    mmlu: dict[str, dict]

    @classmethod
    def load(cls, run_root: str | os.PathLike) -> "GoldStore":
        directory = protected_dir(run_root)
        math_rows = _read_protected_jsonl(directory / MATH_GOLD_FILE)
        mmlu_rows = _read_protected_jsonl(directory / MMLU_GOLD_FILE)
        return cls(
            run_root=str(run_root),
            math={row["item_id"]: row for row in math_rows},
            mmlu={row["item_id"]: row for row in mmlu_rows},
        )

    def reference(self, family: Family | str, item_id: str) -> dict:
        key = family.value if isinstance(family, Family) else str(family)
        table = self.math if key == Family.MATH500.value else self.mmlu
        try:
            return table[item_id]
        except KeyError as exc:
            raise GoldUnavailable(f"no protected gold for {key} item {item_id!r}") from exc


_BOUND_STORE: GoldStore | None = None


def bind_gold(run_root: str | os.PathLike) -> GoldStore:
    """Bind the process-wide gold store.  Evaluator lane only."""
    global _BOUND_STORE
    _BOUND_STORE = GoldStore.load(run_root)
    return _BOUND_STORE


def unbind_gold() -> None:
    global _BOUND_STORE
    _BOUND_STORE = None


def bound_gold() -> GoldStore:
    if _BOUND_STORE is None:
        raise GoldUnavailable(
            "no protected gold store is bound; call graders.bind_gold(run_root) from the "
            "evaluator lane (runtime code must never grade)"
        )
    return _BOUND_STORE


# --------------------------------------------------------------------------- public API


def canonical_class(family: Family | str, answer: str | None) -> str | None:
    """Gold-independent canonical class of ``answer`` (WP0 interface).

    MATH: the sympy canonical form of the parsed answer (string class as fallback).
    MMLU-Pro: the chosen option letter.  Receiver traces: the stated residue.
    Returns ``None`` for an empty or unkeyable answer — such a candidate is structurally
    valid but casts no vote.
    """
    key = family.value if isinstance(family, Family) else str(family)
    if key == Family.MATH500.value:
        return math_equiv.canonical_class(answer)
    if key == Family.MMLUPRO.value:
        return mmlu.canonical_class(answer)
    if key == TRACE_FAMILY:
        return trace.canonical_class(answer)
    raise ValueError(f"unknown grader family {family!r}")


def grade(
    family: Family | str,
    answer: str | None,
    item_id: str,
    *,
    store: GoldStore | None = None,
    timeout_s: float = math_equiv.DEFAULT_TIMEOUT_S,
) -> GradeResult:
    """Grade ``answer`` for ``item_id`` against the protected reference (WP0 interface)."""
    key = family.value if isinstance(family, Family) else str(family)
    gold = (store or bound_gold()).reference(key, item_id)
    started = time.monotonic()
    if key == Family.MATH500.value:
        verdict = math_equiv.compare(answer, gold["answer"], timeout_s=timeout_s)
        status, detail = verdict.status, verdict.detail
    elif key == Family.MMLUPRO.value:
        verdict = mmlu.grade(answer, gold["permuted_key"], options=gold.get("options"))
        status, detail = verdict.status, verdict.detail
    else:
        raise ValueError(f"unknown grader family {family!r}")
    return GradeResult(
        item_id=item_id,
        family=key,
        status=status,
        detail=detail,
        canonical_class=canonical_class(key, answer),
        grader_hash=grader_hash(key),
        duration_s=time.monotonic() - started,
    )


__all__ = [
    "GRADER_SCHEMA_VERSION",
    "GoldStore",
    "GoldUnavailable",
    "GradeResult",
    "STATUSES",
    "STATUS_AMBIGUOUS",
    "STATUS_CORRECT",
    "STATUS_INCORRECT",
    "STATUS_TIMEOUT",
    "TRACE_FAMILY",
    "bind_gold",
    "bound_gold",
    "canonical_class",
    "grade",
    "grader_hash",
    "grader_hashes",
    "grader_module",
    "math_equiv",
    "mmlu",
    "trace",
    "unbind_gold",
]
