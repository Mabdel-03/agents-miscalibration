"""``allocation audit --run-dir DIR``: MASTER_SPEC §12 algebraic checks, the fixed-population and
sealed-before-join invariants, and agent 08's independent recompute hook.

Writes ``reports/AUDIT.md`` (+ HTML) and ``tables/audit_checks.{parquet,csv}``. Exit status: 0 when no check
fails (skips are listed, never counted as passes), 1 when any check fails.

Independent recompute hook: when ``tests/integration/independent_recompute.py`` exists under the project root
(or ``--recompute-hook PATH``), it is loaded by file path and its ``independent_recompute(run_dir)`` is called.
It must return ``{"contrasts": {name: estimate}, "checks": [{"name", "passed", "detail"}]}``; every contrast
it names is compared with this module's estimate (absolute tolerance 1e-9).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from agent_allocation.analysis import checks as C
from agent_allocation.analysis.load import load_run, resolve_config
from agent_allocation.paths import PROJECT_ROOT
from agent_allocation.reporting.render import bullet_list, markdown_to_html, md_table, write_table, write_text

HOOK_RELATIVE = Path("tests") / "integration" / "independent_recompute.py"
TOLERANCE = 1e-9


def load_hook(path: Path | None = None) -> Callable[[Path], Mapping[str, Any]] | None:
    target = path or (PROJECT_ROOT / HOOK_RELATIVE)
    if not target.is_file():
        return None
    spec = importlib.util.spec_from_file_location("agent_allocation_independent_recompute", target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "independent_recompute", None)


def hook_checks(hook: Callable[[Path], Mapping[str, Any]] | None, run_dir: Path,
                own_contrasts: Mapping[str, float | None]) -> list[C.CheckResult]:
    if hook is None:
        return [C.CheckResult("independent_recompute", "skip",
                              f"hook {HOOK_RELATIVE} not present (agent 08)", str(HOOK_RELATIVE))]
    try:
        payload = hook(run_dir)
    except Exception as exc:  # the hook is independent code: report, never crash the audit
        return [C.CheckResult("independent_recompute", "fail", f"hook raised {type(exc).__name__}: {exc}",
                              str(HOOK_RELATIVE), (repr(exc),))]
    out: list[C.CheckResult] = []
    violations: list[str] = []
    compared = 0
    for name, value in sorted(dict(payload.get("contrasts", {})).items()):
        mine = own_contrasts.get(name)
        if value is None and mine is None:
            continue
        compared += 1
        if value is None or mine is None or abs(float(value) - float(mine)) > TOLERANCE:
            violations.append(f"{name}: independent={value} analysis={mine}")
    out.append(C.CheckResult("independent_recompute_contrasts", "fail" if violations else
                             ("pass" if compared else "skip"),
                             f"{compared} contrast(s) compared" if compared else "hook returned no comparable "
                             "contrasts", str(HOOK_RELATIVE), tuple(violations)))
    for item in payload.get("checks", []):
        passed = item.get("passed")
        status = "pass" if passed is True else ("fail" if passed is False else "skip")
        out.append(C.CheckResult(f"independent:{item.get('name')}", status, str(item.get("detail", "")),
                                 str(HOOK_RELATIVE)))
    return out


def audit_run(run_dir: str | Path, *, config_path: str | None = None, hook_path: str | None = None,
              replicates: int | None = None) -> tuple[list[C.CheckResult], Path]:
    from agent_allocation.reporting.cli import build_analysis, run_paths

    paths = run_paths(run_dir)
    cfg, source = resolve_config(paths, config_path)
    run = load_run(paths, cfg, config_source=source)
    analysis = build_analysis(run, replicates=replicates)
    results = C.run_checks(run, analysis.outcomes, analysis=analysis)
    hook = load_hook(Path(hook_path) if hook_path else None)
    results.extend(hook_checks(hook, paths.run_dir, analysis.extras.get("primary_estimates", {})))
    frame = pd.DataFrame([r.as_row() for r in results])
    write_table(frame, paths.tables_dir, "audit_checks")
    text = audit_markdown(analysis, results)
    report = paths.reports_dir / "AUDIT.md"
    write_text(report, text)
    write_text(report.with_suffix(".html"), markdown_to_html(text, "Audit"))
    return results, report


def audit_markdown(analysis, results: list[C.CheckResult]) -> str:
    counts = {s: sum(1 for r in results if r.status == s) for s in ("pass", "fail", "skip")}
    lines = [f"# Audit: `{analysis.run.config.get('study_id')}`", ""]
    if not analysis.status["complete"]:
        lines += ["**INTERIM AUDIT: PARTIAL COLLECTION.** Checks that need fully known panels are skipped, not "
                  "passed.", ""]
    lines += [f"- Input digest: `{analysis.run.input_digest()}`",
              f"- Checks: {counts['pass']} pass, {counts['fail']} fail, {counts['skip']} skip", "",
              "## Checks", "", md_table(pd.DataFrame([r.as_row() for r in results]),
                                       ["check", "status", "detail", "evidence", "n_violations"])]
    failing = [r for r in results if r.status == "fail"]
    if failing:
        lines += ["## Violations", ""]
        for r in failing:
            lines += [f"### {r.name}", "", bullet_list(r.violations[:50])]
    return "\n".join(lines) + "\n"


def cmd_audit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation audit")
    parser.add_argument("--run-dir", default=os.environ.get("ALLOCATION_RUN_DIR"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--recompute-hook", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.run_dir:
        parser.error("--run-dir is required")
    results, report = audit_run(args.run_dir, config_path=args.config, hook_path=args.recompute_hook,
                                replicates=args.replicates)
    failed = [r.name for r in results if r.status == "fail"]
    print(f"audit: {len(failed)} failing check(s){': ' + ', '.join(failed) if failed else ''}; {report}",
          file=sys.stderr)
    return 1 if failed else 0
