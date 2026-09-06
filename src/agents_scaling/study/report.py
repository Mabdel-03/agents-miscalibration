"""Final study report assembler — ``python -m agents_scaling.study.report --run-id study_v4``.

Writes one Markdown report from the run artifacts (never from model text):

* ``FROZEN.yaml`` (study id, flagship, freeze time, code version, seeds),
* ``reconcile.json`` (``reconcile.py``: planned vs completed vs incomplete per manifest/module,
  aliased vs generated requests, FLOPs, judge-ambiguity and singleton-vote rates),
* ``tables/stats.json`` + ``tables/stats.md`` (``stats.py``: six-family Holm, A/O details,
  secondary tables — embedded verbatim),
* ``neural/*.json`` summaries when present (geometry, G-dev/G-confirm, C-dev/C-confirm),
* ``freeze/amendments.json`` (the amendment register, verbatim ids/deviations),
* fixed scientific caveats (plan §"Scientific caveats"; amendment-derived).

Options ``--run-stats`` / ``--run-reconcile`` regenerate those inputs first.  Missing inputs are
reported as missing, never silently skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"

CAVEATS = [
    "Qwen3-32B (the amended flagship, F1) is near floor on HLE-Verified; framing and orchestration effects there are imprecise and BigCodeBench carries most of the signal. Domain-specific estimates are reported beside every pooled effect.",
    "N_main = 400 (200 HLE + 200 BCB, salted-hash rank prefixes) makes families A and O estimation-oriented (planned power ≈ 0.5–0.65 at Δ = 0.06); intervals matter more than the Holm decisions.",
    "The HLE judge is the solver checkpoint itself (amendment E1): the multiple-choice audit substitute and the exported 100-candidate human-audit sample bound the judge error; a differential judge error across architectures is not excluded.",
    "Families R (recursive scaling), G-as-planned (activation edits) and M (content use) were not executed (amendment register); they enter the six-family Holm list with p = 1, which cannot enlarge the other families' error budget.",
    "Family C uses a development recalibrator fitted on the pilot's dev-split items only (≈18–20 items instead of the planned 60) after the C3 output-contract clarification; the C3 example number (0.25) is a possible mild anchor.",
    "Family G is estimated on the frozen report-anchor readouts, whose block, rank and penalty were selected on the pilot development split alone (20 items, 58 rows; amendment E5) rather than the planned 60 items, so it is estimation-oriented; cross-checkpoint coordinate comparisons were not attempted.",
    "Repeated episodes (E) are estimation-only (30 items × 6 episodes) and S_HISTORY is excluded from E (64 sequential calls).",
    "Episodes without a native final answer (invalid/truncated/no valid candidate) are scored incorrect under the frozen common failure rule and counted in the missingness table; infrastructure-incomplete items are listed by reconcile and excluded from the rank-prefix panels only through the prefix rule.",
]


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _frozen_scalars(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line and not line.startswith((" ", "\t", "#")) and ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().strip("'\"")
            if v:
                out[k.strip()] = v
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    return str(x)


def run_totals(rec: Mapping[str, Any], run_root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    """Run-level cost totals counted once per distinct cell.

    ``reconcile.json`` reports per (manifest, module); a cell that belongs to several manifests would
    be summed once per manifest.  The per-cell figures come from ``cells/<cell_id>/meta.json``."""
    rows = 0
    cell_ids: set[str] = set()
    for man, mods in (rec.get("manifests") or {}).items():
        rows += 1
        path = run_root / man
        if not path.exists():
            continue
        try:
            for c in json.loads(path.read_text()).get("cells", []):
                cell_ids.add(str(c["cell_id"]))
        except (ValueError, KeyError, TypeError):
            continue
    tot = {"cells": 0, "generated": 0, "aliased": 0, "flops": 0.0}
    for cid in cell_ids:
        meta_path = run_root / "cells" / cid / "meta.json"
        if not meta_path.exists():
            continue
        try:
            m = json.loads(meta_path.read_text())
        except ValueError:
            continue
        tot["cells"] += 1
        tot["generated"] += int(m.get("n_generated_requests") or 0)   # meta.json key (runner.py)
        tot["aliased"] += int(m.get("n_aliased_requests") or 0)
        tot["flops"] += float(m.get("flops_total") or 0)
    manifest_rows = sum(1 for mods in (rec.get("manifests") or {}).values() for _ in mods)
    return tot, {"rows": manifest_rows, "distinct_cells": len(cell_ids)}


def freshness(run_root: Path, stats: Mapping[str, Any] | None, rec: Mapping[str, Any] | None) -> list[str]:
    """Warnings when the report would publish inputs that no longer match the run.

    Guards the failure the review found: a stats.json or reconcile.json written earlier in the run
    silently becomes the final scientific result."""
    warn: list[str] = []
    if stats is None:
        warn.append("tables/stats.json is MISSING — the headline table is absent. Run `--run-stats`.")
    else:
        man = stats.get("manifest", {})
        try:
            import pyarrow.parquet as pq

            now = {name: pq.read_metadata(run_root / "tables" / f"{name}.parquet").num_rows
                   for name in ("selections", "banks", "episodes") if (run_root / "tables" / f"{name}.parquet").exists()}
        except Exception:
            now = {}
        was = {k: v for k, v in (man.get("tables") or {}).items() if k in now}
        stale = {k: (was.get(k), now.get(k)) for k in now if was.get(k) is not None and was[k] != now[k]}
        if stale:
            warn.append("tables/stats.json is STALE: it was computed on " + ", ".join(f"{k} {a} rows (now {b})" for k, (a, b) in sorted(stale.items())) + ". Re-run `--run-stats`.")
        n_res = man.get("n_resamples_primary")
        if n_res is not None and int(n_res) < 20_000:
            warn.append(f"tables/stats.json used {int(n_res):,} resamples; the spec (§9.3) requires 20,000. Re-run `--run-stats`.")
    if rec is None:
        warn.append("reconcile.json is MISSING — the scope table is absent. Run `--run-reconcile`.")
    else:
        on_disk = {p.name for p in run_root.glob("cells_*.json")}
        listed = set((rec.get("manifests") or {}).keys())
        missing = sorted(on_disk - listed)
        if missing:
            warn.append(f"reconcile.json is STALE: {len(missing)} manifest(s) present in the run root are absent from it ({', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}). Re-run `--run-reconcile`.")
    return warn


def scope_section(rec: dict[str, Any] | None, run_root: Path) -> list[str]:
    lines = ["## 2. Scope achieved (reconcile)", ""]
    if not rec:
        return lines + ["reconcile.json missing — run `python -m agents_scaling.study.reconcile --run-id <run>`.", ""]
    lines += ["| manifest | module/lane | planned cells | cells done | planned items | items done | incomplete | suspended | generated req | aliased req | PFLOP |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for man, mods in sorted(rec.get("manifests", {}).items()):
        for mod, v in sorted(mods.items()):
            lines.append(f"| {man} | {mod.replace('|', '/')} | {v.get('planned_cells')} | {v.get('cells_with_meta')} | {v.get('planned_items')} | {v.get('completed_items')} | {v.get('incomplete_items')} | {v.get('suspended_cells')} | {v.get('generated_requests')} | {v.get('aliased_requests')} | {float(v.get('flops_total', 0) or 0)/1e15:.0f} |")
    # Run-level totals must count each CELL once: a cell can appear in several manifests (a generate
    # manifest and the wave manifests built from its seal), so summing the rows above double-counts.
    tot, dedup = run_totals(rec, run_root)
    lines += ["", f"Run totals over {tot['cells']:,} distinct cells: generated requests {tot['generated']:,}; aliased (content-addressed reuse) {tot['aliased']:,}; logical work {tot['flops']/1e18:.1f} EFLOP."
                  + (f" (Summing the manifest rows would double-count {dedup['rows'] - tot['cells']:,} cell appearances.)" if dedup['rows'] > tot['cells'] else "")]
    j, v = rec.get("judge", {}), rec.get("vote", {})
    lines += ["", f"HLE judge: {j.get('judged_answers')} judged answers, ambiguous rate {_fmt(j.get('ambiguous_rate'), 4)}. VOTE selections: {v.get('vote_selections')}; no-valid-candidate rate {_fmt(v.get('no_valid_rate'), 3)}; all-singleton (tie-broken by HMAC) rate {_fmt(v.get('singleton_rate'), 3)}."]
    inc = sorted({(str(i.get("cell_id")), str(i.get("source_id")))
                  for mods in rec.get("manifests", {}).values() for v in mods.values() for i in (v.get("incomplete") or [])})
    if inc:
        lines += ["", f"Infrastructure-incomplete items (distinct, listed and never hidden; §10.4): {len(inc)}", ""] + [f"- {cid} / {sid}" for cid, sid in inc[:60]]
        if len(inc) > 60:
            lines.append(f"- … {len(inc) - 60} more in reconcile.json")
    return lines + [""]


def neural_section(run_root: Path) -> list[str]:
    nd = run_root / "neural"
    lines = ["## 4. Neural readouts (N1 capture, N3 analysis)", ""]
    for name, label in (("geometry_summary.json", "R3 geometry / functional diversity"), ("G_dev_summary.json", "G-dev (predictor selection, dev only)"), ("G_confirm_summary.json", "G-confirm (frozen predictor on confirmation)"), ("C_dev_summary.json", "C-dev (recalibrators, dev only)"), ("C_summary.json", "C (confirmation)")):
        d = _read_json(nd / name)
        if d is None:
            lines.append(f"- **{label}**: not run (`neural/{name}` missing).")
            continue
        prim = (d.get("primary") or {}).get("bootstrap") if isinstance(d.get("primary"), dict) else None
        if prim:
            lines.append(f"- **{label}**: contrast {_fmt(prim.get('estimate'), 4)} (95% CI {_fmt([prim.get('ci_low'), prim.get('ci_high')], 4)}), one-sided p = {_fmt(prim.get('p_one_sided'), 4)}; n = {d.get('n') or (d.get('primary') or {}).get('n')}; note: {d.get('note') or (d.get('primary') or {}).get('note') or ''}")
        else:
            keys = [k for k in d if k not in ("rows", "items")][:10]
            lines.append(f"- **{label}**: present; summary keys {keys}.")
    caps = sorted(p.name for p in (nd / "native").glob("*.jsonl")) if (nd / "native").exists() else []
    reps = sorted(p.name for p in (nd / "report").glob("*.jsonl")) if (nd / "report").exists() else []
    lines += ["", f"Capture chunks: native {len(caps)} jsonl/npz pairs, report {len(reps)}. Fidelity files: {sorted(p.name for p in (nd / 'native').glob('fidelity*')) if (nd / 'native').exists() else []}.", ""]
    return lines


def amendments_section(am: list[dict[str, Any]] | None) -> list[str]:
    lines = ["## 5. Amendment register (freeze/amendments.json, verbatim ids)", ""]
    if not am:
        return lines + ["missing", ""]
    lines += ["| id | deviation | spec clause | when |", "|---|---|---|---|"]
    for e in am:
        dev = str(e.get("deviation", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {e.get('id')} | {dev[:220]} | {str(e.get('spec_clause', '')).replace('|', '/')} | {str(e.get('when', '')).replace('|', '/')[:60]} |")
    return lines + [""]


def build_report(run_root: Path, *, generated_at: str) -> str:
    frozen = _frozen_scalars(run_root / "FROZEN.yaml")
    rec = _read_json(run_root / "reconcile.json")
    stats = _read_json(run_root / "tables" / "stats.json")
    stats_md = (run_root / "tables" / "stats.md").read_text() if (run_root / "tables" / "stats.md").exists() else None
    am = _read_json(run_root / "freeze" / "amendments.json")
    summary = _read_json(run_root / "tables" / "summary.json") or {}
    L: list[str] = []
    warnings = freshness(run_root, stats, rec)
    L += [f"# {frozen.get('study_id', run_root.name)} — final report", "", f"Generated {generated_at}. Run root `{run_root}`. Frozen at {frozen.get('frozen_at', '—')} (code `{frozen.get('code_version', '—')}`, flagship `{frozen.get('flagship', '—')}`, judge checkpoint `{frozen.get('judge_checkpoint', '—')}`). Specification: agents_scaling_local_future_state_reversal_experiment_spec_v4_0 with the pre-freeze amendment register below.", ""]
    if warnings:
        L += ["", "> **INPUT FRESHNESS WARNING — the numbers below may not describe the current run.**", ""] + [f"> - {w}" for w in warnings] + [""]
    L += ["## 1. Headline", ""]
    if stats:
        h = stats["holm"]
        fam = stats["families"]
        L += ["| family | executed | estimate | 95% CI | p | Holm-adjusted p | decision |", "|---|---|---|---|---|---|---|"]
        for f in ("A", "O", "R", "C", "G", "M"):
            d = fam[f]
            L.append(f"| {f} | {_fmt(d.get('executed'))} | {_fmt(d.get('estimate'))} | {_fmt(d.get('ci95'))} | {_fmt(h[f]['p'], 4)} | {_fmt(h[f]['p_holm'], 4)} | {'reject H0' if h[f]['reject'] else 'no rejection'} |")
        A, O = fam["A"], fam["O"]
        # The panel is chosen per family (stats.choose_panel / amendment P5): report the one actually
        # used, never the literal "prefix" sub-dict, which is a degenerate stub when the fallback fired.
        pk_a, pk_o = A.get("panel_used", "prefix"), O.get("panel_used", "prefix")
        means = ", ".join(f"{m['method']} {_fmt(m['mean'])}" for m in (O.get(pk_o, {}).get("methods") or [])) or "not estimable on this panel"
        L += ["", f"Family A (VOTE_AWARE effect on VOTE@5, TEAM_FRAME averaged): **{pk_a}** panel, n = {A.get('n')} (the common completed rank prefix held {A.get('n_prefix')} items per domain), estimate {_fmt(A.get('estimate'))}, by domain {_fmt(A.get(pk_a, {}).get('by_domain'))}. {A.get('panel_note', '')}",
              f"Family O (native final accuracy at B4, methods {O.get('methods')}): **{pk_o}** panel, global max-T p = {_fmt(O.get('p'), 4)}; per-method means {means}. {O.get('panel_note', '')}"]
        L += ["", f"Analysis manifest: seed {stats['manifest']['analysis_seed']}, {stats['manifest']['n_resamples_primary']:,} primary resamples, code `{stats['manifest']['code_version']}`, common prefix by module (aggregate) {summary.get('common_prefix_by_module')}."]
    else:
        L += ["tables/stats.json missing — run `python -m agents_scaling.study.stats --run-id <run>`."]
    L += [""] + scope_section(rec, run_root)
    L += ["## 3. Statistics (tables/stats.md, verbatim)", ""]
    L += [stats_md.replace("# study_v4 confirmatory statistics", "### study_v4 confirmatory statistics") if stats_md else "missing", ""]
    L += neural_section(run_root)
    L += amendments_section(am)
    L += ["## 6. Scientific caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    L += ["## 7. Artifacts", "", f"- Tables: `{run_root}/tables/` (selections/banks/episodes parquet, summary.json, stats.json, stats.md)", f"- Seals: `{run_root}/seals/<manifest sha>/{{POOLS,SELECTIONS}}.json`; freeze: `{run_root}/FROZEN.yaml`, `freeze/amendments.json`, `freeze/prior_exposure_manifest.json`; prompt hashes: `src/agents_scaling/study/prompts/PROMPT_HASHES.json`", f"- Requests (content-addressed): `{run_root}/requests/`; cells: `{run_root}/cells/<cell_id>/`; evaluator outputs: `{run_root}/eval/{{hle,bcb}}/`; HLE audit sample: `{run_root}/eval/hle_audit_sample.jsonl` (0600) when exported", f"- Neural: `{run_root}/neural/` (native/report captures, worklists, tables, summaries); forecasts: `{run_root}/forecast/`", f"- Reconciliation: `{run_root}/reconcile.json`", ""]
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--out", default=None, help="Markdown path (default docs/study_v4/10_final_report.md under the repo)")
    p.add_argument("--run-stats", action="store_true", help="regenerate tables/stats.json first (20k resamples)")
    p.add_argument("--run-reconcile", action="store_true", help="regenerate reconcile.json first")
    p.add_argument("--resamples", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    if args.run_reconcile:
        from agents_scaling.study import reconcile

        reconcile.main(["--run-id", args.run_id, "--results-root", args.results_root])
    if args.run_stats:
        from agents_scaling.study import stats

        stats.main(["--run-id", args.run_id, "--results-root", args.results_root] + (["--resamples", str(args.resamples)] if args.resamples else []))
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[3] / "docs" / "study_v4" / "10_final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_report(run_root, generated_at=dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"))
    out.write_text(text)
    print(f"[report] wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
