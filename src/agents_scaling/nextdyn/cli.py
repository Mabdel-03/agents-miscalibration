"""One dispatcher, one flag vocabulary, one summary line (spec 17.5; plan §1.2, notes §5.13).

ARD36 grew ``--run-id`` / ``--run-root`` / ``--tables`` drift across its entry points.  Here
every verb takes the same three flags from :func:`common_parser` (``--config``, ``--run-root``,
``--manifest``), every ``main(argv)`` returns a :class:`Report`, and the dispatcher prints
exactly one line ``completed=.. failed=.. skipped=.. blocked=.. reused=..`` and exits non-zero
on ``failed>0`` (1) or ``blocked>0`` (2), except ``status``, which is a monitor and always
exits 0.

Verbs resolve **lazily** to ``"<module>:main"``: a verb whose module has not landed returns
``Report(blocked=1, detail=...)`` and exit 2 instead of a traceback, so the runbook can be
exercised end to end while work packages are pending (``test_every_runbook_verb_parses``).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.nextdyn.config import DEFAULT_STUDY_CONFIG, load_study
from agents_scaling.nextdyn.errors import NextdynError

RESULTS_ROOT_ENV: Final[str] = "ASYS_RESULTS_ROOT"

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_BLOCKED: Final[int] = 2

_M = "agents_scaling.nextdyn."

#: verb -> "module:callable"; the 13 spec-17.5 verbs plus the operational three and the
#: internal lane-driver entry.
VERBS: Final[Mapping[str, str]] = {
    "inventory": _M + "inventory:main",
    "audit-legacy": _M + "e0.main:main",
    "lock-inputs": _M + "locks:lock_inputs_main",
    "validate": _M + "gates:validate_main",
    "pilot": _M + "plan:pilot_main",
    "power": _M + "power:main",
    "plan": _M + "plan:main",
    "freeze": _M + "freeze:main",
    "run": _M + "dispatch:run_main",
    "materialize": _M + "materialize:main",
    "capture": _M + "capture:main",
    "analyze": _M + "analysis:main",
    "release-check": _M + "release:main",
    "fleet": _M + "fleet:main",
    "gates": _M + "gates:main",
    "status": _M + "status:main",
    "drive": _M + "dispatch:drive_main",
}
SPEC_VERBS: Final[tuple[str, ...]] = (
    "inventory", "audit-legacy", "lock-inputs", "validate", "pilot", "power", "plan", "freeze",
    "run", "materialize", "capture", "analyze", "release-check",
)
OPERATIONAL_VERBS: Final[tuple[str, ...]] = ("fleet", "gates", "status")
INTERNAL_VERBS: Final[tuple[str, ...]] = ("drive",)
#: Verbs that never fail the process: monitors report, they do not gate.
MONITOR_VERBS: Final[frozenset[str]] = frozenset({"status"})


@dataclass(frozen=True)
class Report:
    """The five spec-17.5 counters every verb returns."""

    completed: int = 0
    failed: int = 0
    skipped: int = 0
    blocked: int = 0
    reused: int = 0
    detail: str = ""

    def line(self) -> str:
        return (f"completed={self.completed} failed={self.failed} skipped={self.skipped} "
                f"blocked={self.blocked} reused={self.reused}")

    def exit_code(self, verb: str) -> int:
        if verb in MONITOR_VERBS:
            return EXIT_OK
        if self.failed > 0:
            return EXIT_FAILED
        if self.blocked > 0:
            return EXIT_BLOCKED
        return EXIT_OK

    def __add__(self, other: "Report") -> "Report":
        detail = "; ".join(d for d in (self.detail, other.detail) if d)
        return Report(self.completed + other.completed, self.failed + other.failed,
                      self.skipped + other.skipped, self.blocked + other.blocked,
                      self.reused + other.reused, detail)


def common_parser(prog: str, *, description: str | None = None) -> argparse.ArgumentParser:
    """The shared flags.  Verbs add their own on top (``--dry-run``, ``--smoke``, ...)."""
    ap = argparse.ArgumentParser(prog=f"python -m agents_scaling.nextdyn {prog}", description=description)
    ap.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG,
                    help="study config (default configs/nextdyn/study.yaml)")
    ap.add_argument("--run-root", type=Path, default=None,
                    help="run root (default $ASYS_RESULTS_ROOT/<study id>)")
    ap.add_argument("--manifest", type=Path, default=None, help="a frozen or block manifest")
    return ap


def results_root() -> Path:
    return Path(os.environ.get(RESULTS_ROOT_ENV) or DEFAULT_RESULTS_ROOT)


def resolve_run_root(args: argparse.Namespace) -> Path:
    """``--run-root`` if given, else ``$ASYS_RESULTS_ROOT/<study.id>`` from the config."""
    if args.run_root is not None:
        return Path(args.run_root)
    cfg = load_study(args.config)
    return results_root() / cfg.study.id


def resolve_verb(verb: str) -> Callable[[Sequence[str]], Report]:
    """Import ``VERBS[verb]``; :class:`NextdynError` with a clear message when the module or
    callable is absent (the caller turns it into a blocked report)."""
    try:
        target = VERBS[verb]
    except KeyError as exc:
        raise NextdynError(f"unknown verb {verb!r}; verbs: {', '.join(VERBS)}") from exc
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise NextdynError(f"verb {verb!r} needs {module_name} ({exc}); its work package has not landed") from exc
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise NextdynError(f"verb {verb!r}: {module_name} has no callable {attr!r}")
    return fn


def run_verb(verb: str, argv: Sequence[str]) -> Report:
    """Run one verb and return its report; a missing module is ``blocked``, never a traceback."""
    try:
        fn = resolve_verb(verb)
    except NextdynError as exc:
        return Report(blocked=1, detail=str(exc))
    report = fn(list(argv))
    if not isinstance(report, Report):
        return Report(failed=1, detail=f"verb {verb!r} returned {type(report).__name__}, not a Report")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: python -m agents_scaling.nextdyn <verb> [--config C] [--run-root R] [--manifest M] ...")
        print("verbs: " + ", ".join(SPEC_VERBS + OPERATIONAL_VERBS))
        return EXIT_OK if args else EXIT_BLOCKED
    verb, rest = args[0], args[1:]
    if verb not in VERBS:
        print(f"unknown verb {verb!r}; verbs: {', '.join(SPEC_VERBS + OPERATIONAL_VERBS)}", file=sys.stderr)
        print(Report(blocked=1).line())
        return EXIT_BLOCKED
    report = run_verb(verb, rest)
    if report.detail:
        print(f"[{verb}] {report.detail}", file=sys.stderr)
    print(report.line())
    return report.exit_code(verb)


__all__ = [
    "EXIT_BLOCKED", "EXIT_FAILED", "EXIT_OK", "INTERNAL_VERBS", "MONITOR_VERBS", "OPERATIONAL_VERBS",
    "RESULTS_ROOT_ENV", "Report", "SPEC_VERBS", "VERBS", "common_parser", "main", "resolve_run_root",
    "resolve_verb", "results_root", "run_verb",
]
