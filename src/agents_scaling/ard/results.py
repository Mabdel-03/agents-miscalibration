"""``results_summary.md`` and ``hardware_report.md`` — the run's two written deliverables.

``analysis.py`` renders the statistics (primaries under Holm, within-item discrimination, the
registered nonclaims). This module wraps that with the things a reader needs in order to know
*what the numbers are of*: which modules actually ran against the frozen ledger, which did not,
the amendments and defects that shaped the run, and the hardware it was produced on.

The ordering rule is the spec's: **confirmed, then exploratory, then unexecuted.** A module that
did not run is named in the report rather than omitted from it, because a reader who cannot see
what is missing cannot calibrate what is present.

    PYTHONPATH=src python -m agents_scaling.ard.results --run-root <root>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K

#: Modules of the frozen full-core tier, and how a reader should treat each one.
MODULES: tuple[tuple[str, str, str], ...] = (
    ("E1", "root banks (16 draws per item, both checkpoints)", "confirmatory input"),
    ("E2a", "natural protocols on 8B (FRESH, PRIVATE, PEER, BALANCED, HUB5)", "confirmatory"),
    ("E2b", "natural protocols on the frozen 96-item 32B panel", "secondary"),
    ("E2c", "challenge cohort (constructed C0=1 / C0=4 starts)", "secondary"),
    ("E2d", "4,096-token replication on the low-budget roots", "secondary"),
    ("E3", "receiver assay (synthetic programs, 5 presentations)", "secondary"),
    ("E4", "forks, the frozen controller, GATE and RANDOM_PROTECT", "confirmatory"),
    ("E5", "common private suffixes from sealed round-1 states", "secondary"),
    ("E6", "neural capture and geometry", "secondary"),
    ("E0", "legacy reanalysis of the study_v4 run", "context"),
)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def cell_census(run_root: Path) -> dict[str, dict[str, int]]:
    """``{kind/protocol: {cells, complete, items}}`` straight off the run directory."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"cells": 0, "complete": 0, "items": 0})
    cells = run_root / "cells"
    if not cells.exists():
        return {}
    for d in sorted(cells.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        parts = d.name.split(".")
        kind = parts[0]
        key = f"{kind}/{parts[2]}" if kind == "EPISODE" and len(parts) > 2 else kind
        row = out[key]
        row["cells"] += 1
        row["complete"] += int((d / "meta.json").exists())
        items = d / "items"
        row["items"] += len(list(items.glob("*.json"))) if items.exists() else 0
    return dict(out)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                       # noqa: BLE001 - provenance is best-effort
        return ""


def hardware_report(run_root: Path) -> str:
    """Partitions, replicas, precision and realized compute (plan §5.8 export)."""
    frozen = {}
    try:
        import yaml

        path = run_root / "FROZEN.yaml"
        if path.exists():
            frozen = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:                       # noqa: BLE001
        frozen = {}
    resources = _read_parquet(run_root / "tables" / "resources.parquet")
    decisions = _read_parquet(run_root / "tables" / "decisions.parquet")

    by_endpoint = Counter(str(r.get("endpoint_id") or "") for r in decisions)
    flops = defaultdict(float)
    tokens = defaultdict(int)
    for row in decisions:
        ck = str(row.get("checkpoint") or "?")
        block = row.get("flops") or {}
        if isinstance(block, str):
            try:
                block = json.loads(block)
            except ValueError:
                block = {}
        if isinstance(block, Mapping):
            flops[ck] += float(block.get("total") or 0.0)
        tokens[ck] += int(row.get("thinking_tokens") or 0) + int(row.get("final_tokens") or 0)

    lines = ["# ARD36 hardware report", "",
             "Everything below is measured from the run's own records, not from the plan.", ""]
    contract = frozen.get("contract") or {}
    lines += ["## Precision and checkpoints", "",
              f"- dtype: `{contract.get('dtype', K.DTYPE)}` (one precision throughout; spec §6.1)",
              f"- 8B revision: `{K.REVISIONS.get(K.CHECKPOINT_SMALL, '')}`",
              f"- 32B revision: `{K.REVISIONS.get(K.CHECKPOINT_LARGE, '')}`", ""]
    lines += ["## Realized compute", "",
              "| checkpoint | logical decisions | generated tokens | total FLOPs |",
              "|---|---:|---:|---:|"]
    per_ck = Counter(str(r.get("checkpoint") or "?") for r in decisions)
    for ck in sorted(per_ck):
        lines.append(f"| {ck} | {per_ck[ck]:,} | {tokens[ck]:,} | {flops[ck]:.3e} |")
    lines += ["", f"Distinct serving endpoints used: {len([e for e in by_endpoint if e])}.",
              f"Resource samples recorded: {len(resources):,}.", ""]
    if not resources:
        lines += ["> `resources.parquet` carries the per-request rows only: the per-replica "
                  "`nvidia-smi` sampler and the `sacct` GPU-second join were not executed, so "
                  "allocated GPU-seconds are not reported. Logical FLOPs above are complete.",
                  ""]
    return "\n".join(lines) + "\n"


def results_summary(run_root: Path) -> str:
    stats = _json(run_root / "tables" / "ard_stats.json")
    stats_md = (run_root / "tables" / "ard_stats.md")
    census = cell_census(run_root)
    panel = _read_parquet(run_root / "tables" / "calibration_panel.parquet")
    delta = _read_parquet(run_root / "tables" / "calibration_system_minus_agent.parquet")
    frozen_tier = ""
    try:
        import yaml

        frozen = yaml.safe_load((run_root / "FROZEN.yaml").read_text(encoding="utf-8")) or {}
        frozen_tier = ((frozen.get("tier") or {}).get("name") or "")
    except Exception:                       # noqa: BLE001
        frozen = {}

    lines = ["# ARD36 — results summary", "",
             "**Who Should Listen? Predicting and Controlling Correctness Dynamics in "
             "Multi-Agent Reasoning**", "",
             f"Tier as frozen: **{frozen_tier or 'unknown'}**. "
             f"Code version `{_git_sha(Path(__file__).resolve().parents[3])[:12]}`.", "",
             "Read this in the order it is written: what was confirmed, then what is "
             "exploratory, then what did not run. A module that did not run is named here "
             "rather than omitted, because a reader who cannot see what is missing cannot "
             "calibrate what is present.", ""]

    # ---- 1. primaries ------------------------------------------------------
    lines += ["## 1. Primary contrasts (confirmatory)", ""]
    primary = stats.get("primary") or {}
    contrasts = primary.get("contrasts") or {}
    if not contrasts:
        lines += ["> Not computed: `ard_stats.json` carries no primary contrasts. "
                  "Nothing in this section may be reported as a result.", ""]
    else:
        holm = primary.get("holm") or {}
        lines += ["Holm over exactly three contrasts, "
                  f"{stats.get('n_resamples', '?'):,} bootstrap resamples, "
                  f"alpha {stats.get('alpha')}.", "",
                  "| contrast | estimate | 95% CI | p | p (Holm) | reject | items |",
                  "|---|---:|---|---:|---:|---|---:|"]
        for name, res in contrasts.items():
            ci = res.get("ci95_studentized") or [None, None]
            adj = holm.get(name, {})
            def f(v):
                return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append(f"| {name} | {f(res.get('estimate'))} | "
                         f"[{f(ci[0])}, {f(ci[1])}] | {f(res.get('p'))} | "
                         f"{f(adj.get('p_holm'))} | {adj.get('reject')} | "
                         f"{res.get('n_items', '-')} |")
        lines.append("")

    # ---- 2. calibration ----------------------------------------------------
    lines += ["## 2. Confidence and miscalibration (exploratory, beyond the specification)", ""]
    if not panel:
        lines += ["> Not computed.", ""]
    else:
        lines += ["The string \"ECE\" does not occur in the 880-line specification; this axis "
                  "is carried at the user's request (amendment M1) and **no row of it is a "
                  "primary endpoint**. Missing confidence is excluded and reported, never "
                  "imputed to 0.5; the missingness is not random (it lands on cap-hit draws, "
                  "amendment I7), so `conf_missing_rate` is printed beside every estimate.", "",
                  "| protocol | ckpt | cohort | round | signal | valid n | miss | acc | "
                  "mean conf | ECE | within-item AUC |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(panel, key=lambda r: (str(r["protocol"]), str(r["checkpoint"]),
                                                int(r["round"]), str(r["signal"]))):
            if row["signal"] not in ("self_reported", "system_vote_fraction"):
                continue
            def g(v, n=3):
                return "n/a" if v is None else f"{v:.{n}f}"
            lines.append(f"| {row['protocol']} | {row['checkpoint']} | {row['cohort']} | "
                         f"{row['round']} | "
                         f"{row['signal']} | {row['valid_n']} | {g(row['conf_missing_rate'])} | "
                         f"{g(row['accuracy'])} | {g(row['mean_confidence'])} | "
                         f"{g(row['ece_primary'])} | {g(row['within_item_auc'])} |")
        lines.append("")
        if delta:
            lines += ["System minus agent ECE (the headline of this axis): positive means the "
                      "team-level signal is *less* calibrated than its members.", "",
                      "| protocol | ckpt | cohort | round | agent ECE | system ECE | delta |",
                      "|---|---|---|---:|---:|---:|---:|"]
            for row in sorted(delta, key=lambda r: (str(r["protocol"]), int(r["round"]),
                                                    str(r["system_signal"]))):
                if row["system_signal"] != "system_vote_fraction":
                    continue
                lines.append(f"| {row['protocol']} | {row['checkpoint']} | {row['cohort']} | "
                             f"{row['round']} | "
                             f"{row['agent_ece']:.4f} | {row['system_ece']:.4f} | "
                             f"{row['system_minus_agent_ece']:+.4f} |")
            lines.append("")

    # ---- 3. coverage -------------------------------------------------------
    lines += ["## 3. What ran", "", "| cell kind | complete / dispatched | item files |",
              "|---|---:|---:|"]
    for key in sorted(census):
        row = census[key]
        lines.append(f"| {key} | {row['complete']} / {row['cells']} | {row['items']} |")
    lines.append("")

    # ---- 4. unexecuted -----------------------------------------------------
    lines += ["## 4. What did not run", ""]
    ran = {k.split("/")[0] for k in census}
    unexecuted = []
    if "RECEIVER" not in ran:
        unexecuted.append(("E3", "receiver assay"))
    if not any(k.startswith("CHALLENGE") for k in census):
        unexecuted.append(("E2c", "challenge cohort — the constructed C0=1 / C0=4 starts"))
    if not any(k.startswith("SUFFIX") for k in census):
        unexecuted.append(("E5", "common private suffixes"))
    if not any(k.startswith("CAPTURE") for k in census):
        unexecuted.append(("E6", "neural capture and geometry"))
    if not any(k.startswith("SELECTOR") for k in census):
        unexecuted.append(("E4-audit", "set-wise LLM selector audit"))
    if unexecuted:
        for module, what in unexecuted:
            lines.append(f"- **{module}** — {what}. Not executed; no claim rests on it.")
    else:
        lines.append("- Every module of the frozen tier produced cells.")
    lines += ["", "The cost-matched long single-agent baseline (spec §17 item 9) was not run in "
              "any tier, so **no claim of superiority over optimised single-agent inference is "
              "made**.", ""]

    # ---- 5. nonclaims ------------------------------------------------------
    lines += ["## 5. Registered nonclaims (carried verbatim, spec §15.4)", ""]
    for claim in stats.get("nonclaims") or []:
        lines.append(f"- {claim}")
    lines += ["", "## 6. Provenance", "",
              f"- Frozen manifest: `FROZEN.yaml` (tier {frozen_tier or '?'})",
              "- Amendment register: `docs/ard36/02_amendments.md`",
              "- Defect register: `docs/ard36/03_defects.md`",
              "- Full statistics: `tables/ard_stats.json` and `tables/ard_stats.md`",
              "- Hardware: `hardware_report.md`", ""]
    if stats_md.exists():
        lines += ["<details><summary>Statistics as rendered by <code>analysis.py</code>"
                  "</summary>", "", stats_md.read_text(encoding="utf-8"), "</details>", ""]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.results",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args(argv)
    run_root = Path(args.run_root)

    summary = results_summary(run_root)
    hardware = hardware_report(run_root)
    (run_root / "results_summary.md").write_text(summary, encoding="utf-8")
    (run_root / "hardware_report.md").write_text(hardware, encoding="utf-8")
    print(json.dumps({"results_summary": str(run_root / "results_summary.md"),
                      "hardware_report": str(run_root / "hardware_report.md"),
                      "summary_bytes": len(summary), "hardware_bytes": len(hardware)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
